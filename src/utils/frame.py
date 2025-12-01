import torch
from torch import nn
import cv2
import os
import time
import numpy as np
from easydict import EasyDict as edict
from src.utils.camera_utils import getProjectionMatrix, getWorld2Camera, getCamera2World, focal2fov
from dataclasses import dataclass
from typing import List
import torch.nn.functional as F
import torch.nn as nn
import sys

from src.utils.cuda import gaussian_downsample, compute_vertex_and_normal, compute_gradient
from src.utils.cuda import bilateral_filter as cuda_bilateral_filter
from src.utils.cuda import gaussian_filter as cuda_gaussian_filter

RGB_COEFF = [0.299, 0.587, 0.114]

@dataclass
class PyraImageCUDA:
    nlevel              : int = 1
    intensity_pyramid   : List[torch.Tensor] = None
    intrinsic_pyramid   : List[torch.Tensor] = None
    disp_pyramid        : List[torch.Tensor] = None
    grad_pyramid        : List[torch.Tensor] = None
    mask_pyramid        : List[torch.Tensor] = None
    vertex_pyramid      : List[torch.Tensor] = None
    normal_pyramid      : List[torch.Tensor] = None

    def __init__(self, color, depth, mask, intr, nlevel=3, device="cuda:0"):
        self.device = device
        self.nlevel = nlevel
        
        self.color = color
        self.depth = depth
        self.mask = mask
        
        self.gray = (self.color[..., 0] * RGB_COEFF[2] + self.color[..., 1] * RGB_COEFF[1] + self.color[..., 2] * RGB_COEFF[0])[..., None]
        
        self.vmap, self.nmap = compute_vertex_and_normal(self.depth, intr)
        
        self.fx, self.fy, self.cx, self.cy = intr
        self.ht, self.wd= self.color.shape[:2]
    
        self._build_pyramid()
        
        self.gray = (self.gray * 255).to(torch.uint8).squeeze().cpu()

    def _build_pyramid(self):

        self.intensity_pyramid  = []
        self.intrinsic_pyramid  = []
        self.disp_pyramid       = []
        self.grad_pyramid       = []
        self.mask_pyramid       = []
        self.vertex_pyramid     = []
        self.normal_pyramid     = []
        
        depth = self.depth
        mask = self.mask
        gray = self.gray
        
        self.intensity_pyramid.append(gray)
        self.intrinsic_pyramid.append(torch.tensor([self.fx, self.fy, self.cx, self.cy]))
        self.vertex_pyramid.append(self.vmap)
        self.normal_pyramid.append(self.nmap)
        self.disp_pyramid.append(1.0 / (depth + 1e-6))
        self.mask_pyramid.append((mask > 0.9) & (depth > 0.1))    
            
        gradx, grady = compute_gradient(gray)
        grad_mag = torch.sqrt(gradx ** 2 + grady ** 2 + 1e-6)
        self.grad_pyramid.append(torch.stack([gradx, grady, grad_mag], dim=-1).contiguous())
        
        for l in range(1, self.nlevel):
            gray = gaussian_downsample(gray)
            self.intensity_pyramid.append(gray)

            intr = self.intrinsic_pyramid[-1] / (2 ** l)
            self.intrinsic_pyramid.append(intr)

            depth = gaussian_downsample(depth)
            depth = cuda_bilateral_filter(depth, 13, 0.03, 4.5)
            self.disp_pyramid.append(1.0 / (depth + 1e-6))
    
            mask = gaussian_downsample(mask)
            self.mask_pyramid.append((mask > 0.9) & (depth > 0.1))
            
            vertex = gaussian_downsample(self.vertex_pyramid[-1])
            self.vertex_pyramid.append(vertex)
            
            normal = gaussian_downsample(self.normal_pyramid[-1])
            normal = F.normalize(normal, dim=-1)
            self.normal_pyramid.append(normal)
            
            gradx, grady = compute_gradient(gray)
            grad_mag = torch.sqrt(gradx ** 2 + grady ** 2 + 1e-6)
            self.grad_pyramid.append(torch.stack([gradx, grady, grad_mag], dim=-1).contiguous())
    
    def release_pyramid(self):
        self.intensity_pyramid = None
        self.intrinsic_pyramid = None
        self.disp_pyramid = None
        self.grad_pyramid = None
        self.mask_pyramid = None
        self.vertex_pyramid = None
        self.normal_pyramid = None
        torch.cuda.empty_cache()
    
class Frame(nn.Module):
    def __init__(self, idx, ts, color, depth, mask, gt, param, nlevel=3, device="cuda:0"):
        super(Frame, self).__init__()
        
        self.param      = param
        self.gt         = gt
        self.device     = device
        self.uid        = idx
        self.ts         = ts
        self.cam_R      = torch.eye(3).to(device=device)
        self.cam_t      = torch.zeros(3).to(device=device)
        self.cam_R_gt   = torch.from_numpy(gt[:3, :3]).to(device=device)
        self.cam_t_gt   = torch.from_numpy(gt[:3,  3]).to(device=device)
        self.intr       = torch.tensor([param.fx, param.fy, param.cx, param.cy])

        scaled_depth = depth[..., None].astype(np.float32) / param.depth_scale

        self.color_array = color
        self.depth_array = depth
        
        self.color = torch.from_numpy(color).to(device=device).float() / 255.0
        self.depth = cuda_bilateral_filter(torch.from_numpy(scaled_depth).to(device=device).float(), 13, 0.03, 4.5)
        self.mask  = torch.from_numpy(mask).to(device=device).float()
        
        self.fx         = param.fx
        self.fy         = param.fy
        self.cx         = param.cx
        self.cy         = param.cy
        self.fovx       = param.fovx
        self.fovy       = param.fovy
        self.width      = param.width
        self.height     = param.height
        self.projmat    = param.projection_matrix.to(device=device)
        
        self.sparse_tracking = False
        self.pyramid    = PyraImageCUDA(self.color, self.depth, self.mask, self.intr, nlevel ,device)
        
    @staticmethod
    def init_from_dataset(dataset, idx, preload=True):
        ts, color, depth, mask, gt_pose = dataset.get_buffer_frame() if preload else dataset[idx]
        return Frame(idx, ts, color, depth, mask, gt_pose, dataset.params)
    
    def c2w_matrix(self, gt=False):
        return getCamera2World(self.cam_R, self.cam_t) if not gt else getCamera2World(self.cam_R_gt, self.cam_t_gt)
    
    def w2c_matrix(self, gt=False):
        return getWorld2Camera(self.cam_R, self.cam_t) if not gt else getWorld2Camera(self.cam_R_gt, self.cam_t_gt)

    @property
    def world_view_transform(self):
        return getWorld2Camera(self.cam_R, self.cam_t).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return self.world_view_transform @ self.projmat
    
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]
    
    def update_transform(self, R, t):
        
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).to(device=self.device)
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(device=self.device)
        
        self.cam_R = R.detach().to(device=self.cam_R.device).float()
        self.cam_t = t.detach().to(device=self.cam_t.device).float()
    
    def get_pointcloud(self, transform=None, sample_factor=1, array=True):
        fx, fy, cx, cy = self.intr
        y, x = torch.meshgrid(
            torch.arange(0, self.height, device=self.device), 
            torch.arange(0, self.width, device=self.device), 
            indexing="ij")
        
        XYZ = torch.cat([
            (x - cx) * self.depth / fx,
            (y - cy) * self.depth / fy,
            self.depth], dim=0)
        
        mask = (self.depth > 0.1)[0][::sample_factor, ::sample_factor]
        XYZ = XYZ[:, ::sample_factor, ::sample_factor].permute(1, 2, 0)
        RGB = self.color[:, ::sample_factor, ::sample_factor].permute(1, 2, 0)
        
        XYZ, RGB = XYZ[mask], RGB[mask]
        
        if transform is not None:
            if isinstance(transform, np.ndarray):
                transform = torch.from_numpy(transform).to(device=self.device)      
            R, t = transform[:3, :3], transform[:3, 3]
            XYZ = (XYZ @ R.T) + t[None, :]
        
        if array:
            XYZ = XYZ.cpu().numpy().astype(np.float32)
            RGB = RGB.cpu().numpy().astype(np.float32)
            
        return XYZ, RGB
    