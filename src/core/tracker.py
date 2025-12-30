"""
Tracker Module for EggFusion SLAM System

This module implements the camera tracking component of the SLAM system, which estimates
camera poses by aligning current frames with the dense 3D Gaussian surfel map.

The tracker supports both dense and sparse tracking modes:
- Dense tracking: Uses all pixels for photometric and geometric alignment
- Sparse tracking: Uses ORB-SLAM2 features for initial pose estimation
"""

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from src.core.optimizer import update_transform
from src.core.optimizer import rgb_optimization
from src.core.optimizer import icp_optimization
from src.core.optimizer import projective_transform, projective_transform2
from src.utils.cuda import projective_transform as projective_transform_cuda
from src.utils.cuda import solve_block
from src.utils.frame import PyraImageCUDA
import torch.nn.functional as F

def convert_poses(trajs):
    """
    Convert ORB-SLAM2 trajectory format to 4x4 transformation matrices.
    
    Args:
        trajs (list): List of trajectory entries from ORB-SLAM2
    
    Returns:
        tuple: (poses, timestamps)
    """
    poses = []
    stamps = []
    for traj in trajs:
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
        
        pose_ = np.eye(4)
        pose_[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
        pose_[:3, 3] = np.array([t0, t1, t2])
        
        poses.append(pose_)
        stamps.append(stamp)
    
    return poses, stamps

class Tracker(object):
    """
    Camera pose tracking system for EggFusion SLAM.
    
    Handles camera pose estimation by aligning incoming frames with the
    dense 3D Gaussian surfel map using multi-scale pyramid optimization.
    """
    
    def __init__(self, cfg):
        """
        Initialize the tracker with configuration parameters.
        
        Args:
            cfg: Configuration object containing tracking parameters
        """
        self.init = False
        
        self.pyramid_level = cfg.Tracking.pyramid_level
        self.pyramid_iters = cfg.Tracking.pyramid_iters
        
        self.angle_thres = cfg.Tracking.angle_threshold
        self.dist_thres = cfg.Tracking.distance_threshold
        self.residual_thres = cfg.Tracking.residual_thres
        self.dx_thres = cfg.Tracking.dx_threshold
        
        self.use_rgb = cfg.Tracking.use_rgb
        self.use_sparse = cfg.Tracking.use_sparse
        self.rgb_weight = cfg.Tracking.rgb_weight
        self.only_mapping = cfg.System.only_mapping
        
        self.prev_delta_transform = torch.eye(4).to(torch.float32).cuda()
                
        if self.use_sparse:
            self.orb_cfg = cfg.Tracking.orb_config
            self.orb_voc = cfg.Tracking.orb_voc
            self.init_sparse_tracker()
    
    def init_sparse_tracker(self):
        """
        Initialize the ORB-SLAM2 backend for sparse feature tracking.
        """
        import orbslam2
        
        self.orb_backend = orbslam2.System(
            self.orb_voc,
            self.orb_cfg,
            orbslam2.Sensor.RGBD
        )
        
        self.orb_backend.set_use_viewer(False)
        self.orb_backend.initialize(False)

    def sparse_tracking(self, frame):
        """
        Perform sparse feature-based tracking using ORB-SLAM2.
        
        Args:
            frame: Current frame containing RGB-D data and timestamp
            
        Returns:
            torch.Tensor: 4x4 transformation matrix representing camera pose
        """
        self.orb_backend.track_with_orb_feature(
            frame.color_array,
            frame.depth_array,  
            frame.ts
        )
        
        traj_history = self.orb_backend.get_trajectory_points()
        
        pose_es_t1, _ = convert_poses(traj_history[-2:])
        
        init_transform = torch.from_numpy(pose_es_t1[-1]).to(torch.float32).cuda()
        
        return init_transform
    
    def tracking_frame(self, curr_frame, model_map):
        """
        Perform dense tracking between current frame and model map.
        
        Args:
            curr_frame: Current input frame with RGB-D data and camera intrinsics
            model_map (dict): Rendered model data containing rendered images and masks
        
        Side Effects:
            Updates curr_frame.transform with the estimated camera pose
        """
        pyramid_curr = curr_frame.pyramid
        pyramid_prev = PyraImageCUDA(
            color=model_map["rendered_color"],
            depth=model_map["rendered_depth"],
            mask=model_map["opacity_mask"][..., None],
            intr=curr_frame.intr,
            nlevel=curr_frame.pyramid.nlevel,
            device=curr_frame.device
        )
        
        prev_transform = model_map["transform"]
        
        if self.use_sparse:
            init_transform = self.sparse_tracking(curr_frame)
            delta_transform = init_transform.inverse() @ prev_transform.inverse()
        else:
            delta_transform = torch.eye(4).to(torch.float32).cuda()
        
        # Use a temporary transform during dense optimization; only commit if converged
        dense_delta = delta_transform.clone()
        dense_converged = False
        for l in range(self.pyramid_level):
            for _ in range(self.pyramid_iters[l]):
                level = self.pyramid_level - 1 - l
                dx, converged = self.tracking_optimization(
                    pyramid_prev, pyramid_curr, level, dense_delta, curr_frame.uid
                )
                # apply update to the temporary transform
                dense_delta = update_transform(dense_delta, dx)
                if converged:
                    dense_converged = True

        if dense_converged:
            curr_transform = dense_delta @ prev_transform
        else:
            curr_transform = delta_transform @ prev_transform

        curr_frame.update_transform(curr_transform[:3, :3], curr_transform[:3, 3])
        
        
    def tracking(self, frame, model_map):
        """
        Main tracking interface for camera pose estimation.
        
        Args:
            frame: Input frame to track
            model_map: Rendered model data for alignment
        """
        if self.only_mapping:
            frame.update_transform(frame.cam_R_gt, frame.cam_t_gt)
            return
        
        if not self.init:
            self.init = True
            frame.update_transform(frame.cam_R_gt, frame.cam_t_gt)
            return
        
        self.tracking_frame(frame, model_map)
        
    def tracking_optimization(self, model, frame, level, transform, fid):
        """
        Single-level tracking optimization using ICP and RGB alignment.
        
        Args:
            model: Pyramid representation of rendered model
            frame: Pyramid representation of current frame
            level (int): Current pyramid level
            transform (torch.Tensor): Current pose estimate (4x4 matrix)
            fid (int): Frame ID for debugging purposes
            
        Returns:
            tuple: (dx, converged)
                - dx (torch.Tensor): Pose update vector (6-DOF)
                - converged (bool): Whether the solver is considered converged
        """
        coords, Jc = projective_transform(
            transform, 
            model.disp_pyramid[level],
            model.intrinsic_pyramid[level]
        )

        A_icp = torch.tensor(0.0).cuda()
        b_icp = torch.tensor(0.0).cuda()
        A_rgb = torch.tensor(0.0).cuda()
        b_rgb = torch.tensor(0.0).cuda()
        icp_cnt = 0
        rgb_cnt = 0

        A_icp, b_icp, icp_cnt = icp_optimization(
            model, frame, level, transform, coords, 
            self.angle_thres,
            self.dist_thres
        )

        if self.use_rgb:
            A_rgb, b_rgb, rgb_cnt = rgb_optimization(
                model, frame, level, coords, Jc
            )

        # Combine ICP and RGB alignment terms with weighting
        A = A_icp + self.rgb_weight * A_rgb
        b = b_icp + self.rgb_weight * b_rgb

        dx = solve_block(A, b, lm=1.0e-6)
        dx = dx.reshape(-1)

        valid_count = icp_cnt + rgb_cnt
        try:
            res_norm = float(torch.norm(b).item())
        except Exception:
            res_norm = float('inf')
        residual_est = res_norm / max(1.0, (valid_count) ** 0.5)

        dx_norm = float(torch.norm(dx).item())

        converged = (residual_est < float(self.residual_thres)) and (dx_norm < float(self.dx_thres))
        
        return dx, converged