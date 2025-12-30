import os
import torch
import numpy as np
from torch.functional import F
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
import sys
from easydict import EasyDict as edict
from src.utils.frame import Frame
from src.utils.eval_utils import eval_traj_func, matrix_to_tum_format, eval_render_func, compute_gradients
from src.core.utils import transform_map
from src.core.utils import compute_incident_angle
from src.core.utils import compute_confidence
from src.core.utils import depth2pcd
from src.core.tracker import Tracker
from src.core.mapper import Mapping
from src.core.gaussian_surfels import GaussianSurfels
from src.utils.cuda import compute_gradient

class EGGFusion:
    def __init__(self, cfg):
        self.cfg            = cfg
        self.tracker        = Tracker(cfg)
        self.mapper         = Mapping(cfg)
        self.frame_map      = None
        self.model_map      = None

        self.save_dir           = cfg.System.save_dir
        self.final_global_opt   = cfg.System.final_global_opt
        self.reco_normal_thres  = cfg.System.reco_normal_threshold
        self.reco_depth_thres   = cfg.System.reco_depth_threshold
        self.reco_opacity_thres = cfg.System.reco_opacity_threshold
        self.depth_range_min    = cfg.System.depth_range_min
        self.depth_range_max    = cfg.System.depth_range_max

        self.use_occ_grid       = True
        self.occ_dilation       = True

        self.traj               = edict({'ts': [], 'ref': [], 'est': []})

    def reconstruct(self, frame: Frame):
        self.tracker.tracking(frame, self.model_map)
        self.preprocess(frame)
        self.mapper.mapping(frame, self.frame_map)
        self.postprocess(frame)
        self.append_trajectory(frame)

    def postprocess(self, frame):
        
        rendered_map = self.mapper.get_render_output(frame)
        
        normal1 = self.frame_map["normal_map_c"]
        normal2 = rendered_map["render_normal"]
        angle = torch.acos(F.cosine_similarity(normal1, normal2, dim=-1)) * 180 / np.pi
        normal_mask = angle < self.reco_normal_thres
         
        depth1 = self.frame_map["depth_map"]
        depth2 = rendered_map["render_depth"]
        depth_range_mask = (depth2 > self.depth_range_min) & (depth2 < self.depth_range_max)
        depth_mask = (torch.abs(depth1 - depth2) < self.reco_depth_thres) & (self.frame_map["geo_mask"] > 0) & depth_range_mask
     
        opacity_mask = rendered_map["render_opacity"] > self.reco_opacity_thres
        valid_mask = normal_mask & depth_mask.squeeze() & opacity_mask.squeeze()

        color_render = rendered_map["render_color"]
        depth_render = rendered_map["render_depth"]

        # fill in with frame map
        color_render[~valid_mask] = self.frame_map["color_map"][~valid_mask]
        depth_render[~valid_mask] = self.frame_map["depth_map"][~valid_mask]
            
        self.model_map = {}
        self.model_map["rendered_color"]    = color_render
        self.model_map["rendered_depth"]    = depth_render
        self.model_map["mask"]              = valid_mask
        self.model_map["opacity_mask"]      = opacity_mask.squeeze()
        self.model_map["transform"]         = frame.w2c_matrix()
        
        
    def preprocess(self, frame: Frame):
            
        color_map       = frame.color
        depth_map       = frame.depth
        vertex_map      = frame.pyramid.vmap
        normal_map      = frame.pyramid.nmap
        mask            = frame.pyramid.mask.bool()
        time            = frame.uid
        
        gradx, grady = compute_gradient(depth_map)
        gradxy = torch.stack([gradx, grady], dim=-1)
        gradxy = torch.norm(gradxy, dim=-1)
        edge_mask = gradxy > 0.1
        
        simlarity = compute_incident_angle(normal_map, frame.intr)[..., 0]
        normal_mask = simlarity < np.sin(self.reco_normal_thres * np.pi / 180)
        
        # Mask out inf values in normal_map
        inf_mask = torch.isinf(normal_map).any(dim=-1)
        invalid_mask = normal_mask | (normal_map == 0).all(dim=-1) | edge_mask | inf_mask
        depth_map[invalid_mask] = 0
        normal_map[invalid_mask] = 0
        vertex_map[invalid_mask] = 0

        coords = torch.meshgrid(torch.arange(frame.height), torch.arange(frame.width), indexing="ij")
        coords = torch.stack([coords[1], coords[0]], dim=-1).cuda()
        confidence_map = compute_confidence(coords, torch.tensor([frame.cx, frame.cy]).cuda(), 400, 0.72)

        R = frame.c2w_matrix()[:3, :3]
        t = frame.c2w_matrix()[:3,  3]
        tO = torch.zeros(3).to(frame.device)
        
        self.frame_map = {}
        self.frame_map["color_map"] = color_map
        self.frame_map["depth_map"] = depth_map
        self.frame_map["vertex_map_c"] = vertex_map
        self.frame_map["normal_map_c"] = normal_map
        self.frame_map["confidence_map"] = confidence_map.float()
        self.frame_map["rgb_mask"] = mask
        self.frame_map["geo_mask"] = ~invalid_mask.unsqueeze(-1)
        self.frame_map["time"] = time
        self.frame_map["vertex_map_w"] = transform_map(frame.pyramid.vmap, R, t)
        self.frame_map["normal_map_w"] = transform_map(frame.pyramid.nmap, R, tO)
        
    def finish(self):

        print("Finishing...")
        
        keyframe_ids = self.mapper.keyframe_manager.get_keyframe_ids()
        print("Keyframes IDs: {}".format(keyframe_ids))

        self.mapper.keyframe_optimization()
        
        save_dir = os.path.join(self.save_dir, "final_surfels.ply")
        self.mapper.surfels0.save_ply(save_dir)
        print("Saved surfels to {}".format(save_dir))

    def reload(self, path):
        self.mapper.surfels0.load_ply(path)
        print("Reloaded surfels from {}".format(path))

    def append_trajectory(self, frame):
        self.traj.ts.append(frame.ts)
        self.traj.ref.append(frame.c2w_matrix(gt=True).cpu().numpy())
        self.traj.est.append(frame.c2w_matrix().cpu().numpy())
    
    def evaluate_trajectory(self):
        
        tum_pose_ref = []
        tum_pose_est = []
        
        for ts, ref_matrix, est_matrix in zip(self.traj.ts, self.traj.ref, self.traj.est):
            tum_pose_ref.append(matrix_to_tum_format(ts, ref_matrix))
            tum_pose_est.append(matrix_to_tum_format(ts, est_matrix))
        
        np.savetxt(os.path.join(self.save_dir, 'trajectory_ref_tum.txt'), tum_pose_ref)
        np.savetxt(os.path.join(self.save_dir, 'trajectory_est_tum.txt'), tum_pose_est)

        pose_ref = np.array(self.traj.ref)
        pose_est = np.array(self.traj.est)
        
        np.savetxt(os.path.join(self.save_dir, 'trajectory_ref.txt'), pose_ref.reshape(-1, 16))
        np.savetxt(os.path.join(self.save_dir, 'trajectory_est.txt'), pose_est.reshape(-1, 16))

        ates = []
        for fid in tqdm(range(1, len(pose_ref) + 1)):
            est = pose_est[:fid, :3, 3]
            ref = pose_ref[:fid, :3, 3]
            ate = eval_traj_func(ref, est)
            ates.append(ate)
        ates = np.array(ates)

        ate_rmse = ates[-1]

        plt.plot(range(len(ates)), ates)
        plt.ylim(0, max(ates) + 0.1)
        plt.title("ate:{}".format(ates[-1]))
        plt.savefig(os.path.join(self.save_dir, "ates.png"))

        plt.figure()
        plt.plot(pose_est[:, 0, 3], pose_est[:, 1, 3])
        plt.plot(pose_ref[:, 0, 3], pose_ref[:, 1, 3])
        plt.legend(["es", "gt"])
        plt.savefig(os.path.join(self.save_dir, "traj_xy.jpg"))
        plt.figure()
        plt.plot(pose_est[:, 1, 3], pose_est[:, 2, 3])
        plt.plot(pose_ref[:, 1, 3], pose_ref[:, 2, 3])
        plt.legend(["es", "gt"])
        plt.savefig(os.path.join(self.save_dir, "traj_yz.jpg"))
        plt.figure()
        plt.plot(pose_est[:, 0, 3], pose_est[:, 2, 3])
        plt.plot(pose_ref[:, 0, 3], pose_ref[:, 2, 3])
        plt.legend(["es", "gt"])
        plt.savefig(os.path.join(self.save_dir, "traj_xz.jpg"))

        print(f'ATE RMSE: {ate_rmse:.05f}cm')
