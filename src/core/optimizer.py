"""
Optimizer Module for EggFusion SLAM System

This module implements various optimization routines used in the SLAM system for
camera pose estimation and frame alignment. It provides GPU-accelerated functions
for projective geometry, ICP alignment, RGB photometric optimization, and sparse
feature correspondence optimization.

Key Components:
- Projective transformation with automatic differentiation
- ICP (Iterative Closest Point) geometric alignment
- RGB photometric alignment using image gradients
- Sparse feature correspondence optimization
- SE(3) pose update using Lie algebra

Mathematical Background:
- Uses SE(3) Lie group representation for camera poses
- Employs Levenberg-Marquardt damping for robust optimization
- Combines geometric and photometric constraints for dense tracking
- Supports both dense and sparse correspondence optimization

Author: EggFusion Team
Date: 2025
"""

import os
import numpy as np
import torch
import math
import cv2
import torch.nn.functional as F
from src.utils.camera_utils import lie


def projective_transform_with_correspondences(transform, kps1, kps2, invd1, invd2, intr):
    """
    Compute projective transformation for sparse feature correspondences.
    
    This function projects keypoints from two frames using a given transformation
    and computes the Jacobians for optimization. Used in sparse feature-based
    tracking and bundle adjustment.
    
    Args:
        transform (torch.Tensor): 4x4 SE(3) transformation matrix
        kps1 (torch.Tensor): Keypoints from frame 1 [N, 2] (u, v coordinates)
        kps2 (torch.Tensor): Keypoints from frame 2 [N, 2] (u, v coordinates)
        invd1 (torch.Tensor): Inverse depths for keypoints in frame 1 [N]
        invd2 (torch.Tensor): Inverse depths for keypoints in frame 2 [N]
        intr (tuple): Camera intrinsics (fx, fy, cx, cy)
    
    Returns:
        tuple: ((uv1, uv2), (dxdxi1, dxdxi2))
            - uv1, uv2: Projected keypoint coordinates [N, 2]
            - dxdxi1, dxdxi2: Jacobians w.r.t. SE(3) parameters [N, 2, 6]
    """
    fx, fy, cx, cy = intr
    
    # Initialize constants for computation
    O = torch.zeros_like(invd1)  # Zero tensor
    I = torch.ones_like(invd1)   # Identity tensor
    
    # Convert keypoints to normalized camera coordinates
    us1 = (kps1[..., 0] - cx) / fx  # Normalized u coordinate for frame 1
    vs1 = (kps1[..., 1] - cy) / fy  # Normalized v coordinate for frame 1
    zs1 = I                         # Homogeneous coordinate
    ds1 = invd1                     # Inverse depth
    
    us2 = (kps2[..., 0] - cx) / fx  # Normalized u coordinate for frame 2
    vs2 = (kps2[..., 1] - cy) / fy  # Normalized v coordinate for frame 2
    zs2 = I                         # Homogeneous coordinate
    ds2 = invd2                     # Inverse depth
    
    # Create homogeneous point representations
    Ps1 = torch.stack([us1, vs1, zs1, ds1], dim=-1).to(transform.device)
    Ps2 = torch.stack([us2, vs2, zs2, ds2], dim=-1).to(transform.device)
    
    # Apply transformations: forward and inverse
    Pt1 = (Ps1.reshape(-1, 4) @ torch.inverse(transform).T).reshape(-1, 4)
    Pt2 = (Ps2.reshape(-1, 4) @ transform.T).reshape(-1, 4)
    
    # Extract transformed coordinates
    ut1, vt1, zt1, dt1 = torch.unbind(Pt1, dim=-1)
    ut2, vt2, zt2, dt2 = torch.unbind(Pt2, dim=-1)
    
    # Normalize by depth (perspective division)
    ut1 = ut1 / zt1
    vt1 = vt1 / zt1
    dt1 = dt1 / zt1
    
    ut2 = ut2 / zt2
    vt2 = vt2 / zt2
    dt2 = dt2 / zt2

    # Compute Jacobians for SE(3) optimization
    Rinv = transform[:3, :3].T.repeat(kps1.shape[0], 1, 1)
    
    # Jacobian of projection w.r.t. 3D point for frame 1
    dxdP1 = torch.stack([
        dt1 * fx,        O,  -ut1 * dt1 * fx,      
               O, dt1 * fy,  -vt1 * dt1 * fy,
    ], dim=-1).reshape(-1, 2, 3)
    dxdP1 = torch.einsum('ijk,ikl->ijl', dxdP1, -Rinv)
    
    # Extract Jacobian components
    J00 = dxdP1[:, 0, 0]
    J01 = dxdP1[:, 0, 1]
    J02 = dxdP1[:, 0, 2]
    J10 = dxdP1[:, 1, 0]
    J11 = dxdP1[:, 1, 1]
    J12 = dxdP1[:, 1, 2]

    # Jacobian w.r.t. SE(3) parameters for frame 1 (translation + rotation)
    dxdxi1 = torch.stack([
        J00,    J01,    J02,          ut1 * vt1 * fx, -(1 + ut1 * ut1) * fx,    vt1 * fx,
        J10,    J11,    J12,    (1 + vt1 * vt1) * fy,    -   ut1 * vt1 * fy,   -ut1 * fy
    ], dim=-1).reshape(-1, 2, 6)
    
    # Jacobian w.r.t. SE(3) parameters for frame 2
    dxdxi2 = torch.stack([
        dt2 * fx,          O,  -ut2 * dt2 * fx,      - ut2 * vt2 * fx, (1 + ut2 * ut2) * fx, -vt2 * fx,
                O,  dt2 * fy,  -vt2 * dt2 * fy, -(1 + vt2 * vt2) * fy,       ut2 * vt2 * fy,  ut2 * fy
    ], dim=-1).reshape(-1, 2, 6)
    
    # Convert back to pixel coordinates
    uv1 = torch.stack([fx * ut1 + cx, fy * vt1 + cy], dim=-1).to(kps1.device)
    uv2 = torch.stack([fx * ut2 + cx, fy * vt2 + cy], dim=-1).to(kps2.device)
    
    return (uv1, uv2), (dxdxi1, dxdxi2)

def projective_transform(transform, disps, intr):
    """
    Compute dense projective transformation for all pixels in an image.
    
    This function transforms all pixels from one frame to another using a given
    SE(3) transformation. It computes the warped grid coordinates and Jacobians
    for dense tracking optimization.
    
    Args:
        transform (torch.Tensor): 4x4 SE(3) transformation matrix
        disps (torch.Tensor): Inverse depth/disparity map [H, W]
        intr (tuple): Camera intrinsics (fx, fy, cx, cy)
    
    Returns:
        tuple: (warped_grid, dxdxi)
            - warped_grid: Normalized grid coordinates [-1, 1] for sampling [H, W, 2]
            - dxdxi: Jacobians w.r.t. SE(3) parameters [H, W, 2, 6]
    """
    grid = torch.stack(torch.meshgrid(
        torch.arange(disps.shape[0]),
        torch.arange(disps.shape[1]),
        indexing='ij'), dim=-1).to(transform.device)  

    ht, wd = grid.shape[:2]
    fx, fy, cx, cy = intr
    grid_y, grid_x = torch.unbind(grid, dim=-1)

    I = torch.ones_like(grid_x)
    O = torch.zeros_like(grid_x)

    us = (grid_x - cx) / fx
    vs = (grid_y - cy) / fy
    zs = I
    ds = disps.squeeze()
    
    Ps = torch.stack([us, vs, zs, ds], dim=-1).to(transform.device)
    Pt = (Ps.reshape(-1, 4) @ transform.T).reshape(ht, wd, 4)

    ut, vt, zt, dt = torch.unbind(Pt, dim=-1)
    ut = ut / zt
    vt = vt / zt
    dt = dt / zt

    dxdxi = torch.stack([
        dt * fx,        O,  -ut * dt * fx,      - ut * vt * fx, (1 + ut * ut) * fx, -vt * fx,
              O,  dt * fy,  -vt * dt * fy, -(1 + vt * vt) * fy,       ut * vt * fy,  ut * fy
    ], dim=-1).reshape(ht, wd, 2, 6)
    
    warped_grid = torch.stack([fx * ut + cx, fy * vt + cy], dim=-1).to(grid.device)
    warped_grid[..., 0] = 2 * warped_grid[..., 0] / (wd - 1) - 1
    warped_grid[..., 1] = 2 * warped_grid[..., 1] / (ht - 1) - 1
    
    return warped_grid, dxdxi


def projective_transform2(transform, disps, intr):
    """
    Args:
        transform: [4, 4] torch.Tensor, SE(3) pose matrix
        disps: [H, W], inverse depth
        intr: (fx, fy, cx, cy)

    Returns:
        warped_grid: [H, W, 2], normalized [-1, 1]
        (dxdxi, dxdd): Jacobians for SE(3) and inverse depth
    """
    device = transform.device
    H, W = disps.shape[:2]
    fx, fy, cx, cy = intr

    # Generate meshgrid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Normalized coordinates
    us = (grid_x - cx) / fx
    vs = (grid_y - cy) / fy
    ones = torch.ones_like(us)

    inv_depth = disps.squeeze()  # inverse depth
    depth = 1.0 / (inv_depth + 1e-8)  # avoid div-by-zero

    # 3D point in camera frame
    X = torch.stack([us * depth, vs * depth, depth], dim=-1)  # [H, W, 3]
    X_h = torch.cat([X, ones[..., None]], dim=-1)  # [H, W, 4]

    # Transform
    Xt_h = X_h @ transform.T  # [H, W, 4]
    Xt = Xt_h[..., :3]
    x, y, z = Xt[..., 0], Xt[..., 1], Xt[..., 2]

    # Project to image plane
    ut = fx * (x / z) + cx
    vt = fy * (y / z) + cy

    warped_grid = torch.stack([ut, vt], dim=-1)
    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (W - 1) - 1.0
    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (H - 1) - 1.0

    inv_z = 1.0 / z
    inv_z2 = inv_z ** 2

    # d(pi)/dX
    J_proj = torch.empty(H, W, 2, 3, device=device)
    J_proj[..., 0, 0] = fx * inv_z
    J_proj[..., 0, 1] = 0.0
    J_proj[..., 0, 2] = -fx * x * inv_z2
    J_proj[..., 1, 0] = 0.0
    J_proj[..., 1, 1] = fy * inv_z
    J_proj[..., 1, 2] = -fy * y * inv_z2

    # Skew-symmetric matrix
    zero = torch.zeros_like(x)
    skew = torch.stack([
        zero, -z,    y,
        z,    zero, -x,
       -y,     x,  zero
    ], dim=-1).reshape(H, W, 3, 3)

    I3 = torch.eye(3, device=device).expand(H, W, 3, 3)
    J_se3 = torch.cat([I3, -skew], dim=-1)  # [H, W, 3, 6]

    dxdxi = torch.matmul(J_proj, J_se3)

    return warped_grid, dxdxi


def solve_block(H, b, lm=1.0e-6):
    """
    Solve linear system Hx = b using LM damping for numerical stability.
    
    Args:
        H (torch.Tensor): [N, N] Hessian matrix
        b (torch.Tensor): [N] gradient vector  
        lm (float): Levenberg-Marquardt damping factor (default: 1e-6)
        
    Returns:
        torch.Tensor: [N] solution vector dx
    """
    trace = torch.trace(H)
    H = H + trace * lm
    Hinv = torch.inverse(H.cpu())
    dx = Hinv @ b.cpu()
    return dx.cuda()

def rgb_optimization(frame1, frame2, level, coords, Jc):
    """
    RGB photometric optimization for dense visual odometry.
    
    Args:
        frame1: Reference frame with image pyramid
        frame2: Current frame with image pyramid and gradients
        level (int): Pyramid level for processing
        coords: Warped coordinates from projective transform
        Jc: Jacobian matrix from coordinate transformation
        
    Returns:
        tuple: (Hessian matrix, gradient vector, valid pixel count) for pose optimization
    """
    bound = 0.90
    inmask = (coords[..., 0] > -bound) & (coords[..., 0] < bound) & (coords[..., 1] > -bound) & (coords[..., 1] < bound)
    inmask = inmask.reshape(-1, 1)
    
    grad_mask = (frame2.grad_pyramid[level][..., 2] > 1)
    grad_mask = grad_mask.reshape(-1, 1)
    
    mask_prev = frame1.mask_pyramid[level]
    mask_prev = mask_prev.reshape(-1, 1)
    
    model_I = frame1.intensity_pyramid[level].permute([2, 0, 1])[None]
    frame_I = frame2.intensity_pyramid[level].permute([2, 0, 1])[None]

    sample_I = F.grid_sample(frame_I, coords[None], mode='bilinear', padding_mode='zeros', align_corners=True)
    
    Ji = F.grid_sample(frame2.grad_pyramid[level][..., :2].permute([2, 0, 1])[None], coords[None], mode='bilinear', padding_mode='zeros', align_corners=True).permute([2, 3, 0, 1])
    mask_curr = F.grid_sample(frame2.mask_pyramid[level].permute([2, 0, 1])[None].float(), coords[None], mode='nearest', padding_mode='zeros', align_corners=True).permute([2, 3, 0, 1])
    mask_curr = mask_curr > 0.8
    mask_curr = mask_curr.reshape(-1, 1)
       
    weight = inmask & mask_prev & grad_mask & mask_curr

    J = torch.matmul(Ji, Jc).reshape(-1, 6)
    r = (model_I - sample_I).reshape(-1, 1)

    J = J[weight.squeeze()]
    r = r[weight.squeeze()]

    Hcc = torch.matmul(J.T, J)
    gc  = torch.matmul(J.T, r)
    
    valid_count = int(weight.sum().item())

    return Hcc, gc, valid_count

def icp_optimization(frame1, frame2, level, transfrom, coords, angleThres, distThres):
    """
    Iterative Closest Point (ICP) optimization for geometric alignment.
    
    Args:
        frame1: Reference frame with vertex/normal pyramids
        frame2: Current frame with vertex/normal pyramids  
        level (int): Pyramid level for processing
        transfrom: Current SE(3) pose estimate
        coords: Warped coordinates from projective transform
        angleThres (float): Maximum normal angle deviation (degrees)
        distThres (float): Maximum point distance threshold
        
    Returns:
        tuple: (Hessian matrix, gradient vector, valid pixel count) for pose optimization
    """
    vmap_prev = frame1.vertex_pyramid[level]
    nmap_prev = frame1.normal_pyramid[level]
    mask_prev = frame1.mask_pyramid[level]
    vmap_curr = frame2.vertex_pyramid[level]
    nmap_curr = frame2.normal_pyramid[level]
    mask_curr = frame2.mask_pyramid[level]

    vprev = vmap_prev.reshape(-1, 3) @ transfrom[:3, :3].T + transfrom[:3, 3]
    nprev = nmap_prev.reshape(-1, 3) @ transfrom[:3, :3].T
    
    vcurr = F.grid_sample(vmap_curr.permute(2, 0, 1)[None], coords[None], mode='nearest', padding_mode='border', align_corners=True)[0].permute([1, 2, 0]).reshape(-1, 3)
    ncurr = F.grid_sample(nmap_curr.permute(2, 0, 1)[None], coords[None], mode='nearest', padding_mode='border', align_corners=True)[0].permute([1, 2, 0]).reshape(-1, 3)
    
    delta_v = vcurr - vprev
    cross_n = torch.cross(ncurr, nprev, dim=1)
    
    dist = torch.norm(delta_v, dim=-1)
    sine = torch.norm(cross_n, dim=-1)
    
    bound = 0.98
    inmask = (coords[..., 0] > -bound) & (coords[..., 0] < bound) & (coords[..., 1] > -bound) & (coords[..., 1] < bound)
    inmask = inmask.reshape(-1, 1)
    
    nan_mask = ~torch.isnan(cross_n)
    nan_mask = (nan_mask[..., 0] & nan_mask[..., 1] & nan_mask[..., 2]).reshape(-1, 1)
    
    pos_mask = vprev[..., -1] > 0
    pos_mask = pos_mask.reshape(-1, 1)

    valid = ((sine < angleThres * math.pi / 180) & (dist < distThres)).reshape(-1, 1)
    weight = nan_mask & inmask & pos_mask & valid & mask_prev.reshape(-1, 1) & mask_curr.reshape(-1, 1)
    
    r = torch.sum(ncurr * delta_v, dim=1).reshape(-1, 1)
    J = torch.cat([ncurr, torch.cross(vprev, ncurr, dim=1)], dim=1)
    
    # TODO: (pxk) normal maybe nan, it will crash!
    J = J[weight.squeeze()]
    r = r[weight.squeeze()]
    
    Hcc = torch.matmul(J.T, J)
    gc = torch.matmul(J.T, r)
    
    valid_count = int(weight.sum().item())
    
    return Hcc, gc, valid_count

def sparse_correspondence_optimization(transform, corres, intr):
    """
    Sparse feature correspondence optimization for visual odometry.
    
    Args:
        transform: SE(3) pose transformation matrix
        corres: Tuple of (kps1, kps2) correspondence arrays with [u,v,invd] format
        intr: Camera intrinsic parameters [fx, fy, cx, cy]
        
    Returns:
        tuple: (Hessian matrix, gradient vector, valid pixel count) for pose optimization
    """
    kps1 = corres[0][:, :2].cuda()
    kps2 = corres[1][:, :2].cuda()
    invd1 = corres[0][:, 2].cuda()
    invd2 = corres[1][:, 2].cuda()
    
    (proj1, proj2), (Jc1, Jc2) = projective_transform_with_correspondences(transform, kps1, kps2, invd1, invd2, intr)

    if len(corres[0]) == 0:
        return torch.zeros(6, 6).cuda(), torch.zeros(6, 1).cuda()

    r1 = kps2 - proj1
    r2 = kps1 - proj2
    
    Hcc1 = torch.matmul(Jc1.transpose(1, 2), Jc1).sum(dim=0)
    gc1  = torch.matmul(Jc1.transpose(1, 2), r1[..., None]).sum(dim=0)
    Hcc2 = torch.matmul(Jc2.transpose(1, 2), Jc2).sum(dim=0)
    gc2  = torch.matmul(Jc2.transpose(1, 2), r2[..., None]).sum(dim=0)

    Hcc = Hcc1 + Hcc2
    gc = gc1 + gc2
    
    valid_count = len(corres[0])
    
    return Hcc, gc, valid_count

def update_transform(transform, dx):
    """
    Update SE(3) transformation using exponential map parameterization.
    
    Args:
        transform: [4, 4] current SE(3) transformation matrix  
        dx: [6] Lie algebra increment [translation, rotation]
        
    Returns:
        torch.Tensor: Updated SE(3) transformation matrix
    """
    dR = lie.so3_to_SO3(dx[3:])
    dt = dx[:3]
    transform[:3, :3] = dR @ transform[:3, :3]
    transform[:3,  3] = dt + transform[:3,  3]
    return transform
