import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.functional import F
from easydict import EasyDict as edict
from collections import deque
from src.core.gaussian_surfels import GaussianSurfels
from src.core.render import Renderer
from src.core.utils import compute_confidence
from src.core.utils import transform_map
from src.utils.cuda import compute_vertex_and_normal
from src.utils.frame import Frame
from diff_gaussian_rasterization import preprocess_surfels
from diff_gaussian_rasterization import project_surfels_to_frame


cfg_debug = False


def check_nan(name, x):
    if torch.isnan(x).any():
        print(f"⚠ NaN detected in {name}")
    if torch.isinf(x).any():
        print(f"⚠ Inf detected in {name}")
    if (x.abs() > 1e6).any():
        print(f"⚠ Large value detected in {name}")

class KeyFrame(Frame):
    def __init__(self, frame, frame_map, time, fid):
        self.fid            = fid
        self.time           = time
        self.uid            = frame.uid
        self.fx             = frame.fx
        self.fy             = frame.fy
        self.cx             = frame.cx
        self.cy             = frame.cy
        self.fovx           = frame.fovx
        self.fovy           = frame.fovy
        self.cam_R          = frame.cam_R
        self.cam_t          = frame.cam_t
        self.width          = frame.width
        self.height         = frame.height
        self.projmat        = frame.projmat
        self.succ_tracking  = frame.sparse_tracking
        self.frame_map = {
            "color_map"         : frame_map["color_map"],
            "depth_map"         : frame_map["depth_map"],
            "normal_map_c"      : frame_map["normal_map_c"],
            "rgb_mask"          : frame_map["rgb_mask"],
            "geo_mask"          : frame_map["geo_mask"],
        }

    def cpu(self):        
        self.frame_map["color_map"] = self.frame_map["color_map"].cpu()
        self.frame_map["depth_map"] = self.frame_map["depth_map"].cpu()
        self.frame_map["normal_map_c"] = self.frame_map["normal_map_c"].cpu()
        self.frame_map["rgb_mask"] = self.frame_map["rgb_mask"].cpu()
        self.frame_map["geo_mask"] = self.frame_map["geo_mask"].cpu()

    def cuda(self):
        self.frame_map["color_map"] = self.frame_map["color_map"].cuda()
        self.frame_map["depth_map"] = self.frame_map["depth_map"].cuda()
        self.frame_map["normal_map_c"] = self.frame_map["normal_map_c"].cuda()
        self.frame_map["rgb_mask"] = self.frame_map["rgb_mask"].cuda()
        self.frame_map["geo_mask"] = self.frame_map["geo_mask"].cuda()

class KeyFrameManager(object):
    def __init__(self, cfg):
        
        self.keyframe_list      = {}
        self.check_keyframe_R   = cfg.Tracking.check_keyframe_R
        self.check_keyframe_t   = cfg.Tracking.check_keyframe_t
        self.window_size        = cfg.Tracking.sliding_window_size
        self.sliding_window     = deque(maxlen=self.window_size)
    
    def check_keyframe(self, frame_ele):
        
        keyframe_id = len(self.keyframe_list)
        keyframe = KeyFrame(frame_ele["frame"], frame_ele["frame_map"], frame_ele["time"], keyframe_id)
        keyframe.cpu()
                        
        if keyframe.time == 0:
            self.keyframe_list[keyframe.uid] = keyframe
            return True

        prev_id     = self.get_keyframe_ids()[-1]
        prev_cam_R  = self.get_keyframe_by_id(prev_id).c2w_matrix()[:3, :3]
        prev_cam_t  = self.get_keyframe_by_id(prev_id).c2w_matrix()[:3,  3]
        curr_cam_R  = keyframe.c2w_matrix()[:3, :3]
        curr_cam_t  = keyframe.c2w_matrix()[:3,  3]
        
        cos_theta = (torch.trace(prev_cam_R.T @ curr_cam_R) - 1) / 2
        delta_R = torch.rad2deg(torch.acos(cos_theta))
        delta_t = torch.norm(prev_cam_t - curr_cam_t, p=2)
        
        if delta_R > self.check_keyframe_R or delta_t > self.check_keyframe_t:
            self.keyframe_list[keyframe.uid] = keyframe
            return True
        
        return False
            
    def get_keyframe_ids(self):
        return sorted(self.keyframe_list.keys())
    
    def get_keyframe_by_id(self, uid):
        return self.keyframe_list[uid]

    def get_keyframe_num(self):
        return len(self.keyframe_list)
    
    def node_distance(self, node1, node2):
        keyframe1 = self.keyframe_list[node1]
        keyframe2 = self.keyframe_list[node2]
        
        return abs(keyframe1.fid - keyframe2.fid)

class Mapping(object):
    def __init__(self, cfg) -> None:

        self.surfels0                   = GaussianSurfels(cfg) # global map 
        self.surfels1                   = GaussianSurfels(cfg) # unstable surfles
        self.surfels2                   = GaussianSurfels(cfg) # stable surfles
        
        self.stable_mask                = torch.empty(0).cuda()
        self.inview_mask                = torch.empty(0).cuda()
        self.surface_mask               = torch.empty(0).cuda()
        
        self.renderer                   = Renderer(cfg)
        self.keyframe_manager           = KeyFrameManager(cfg)
        
        self.local_map_iter             = cfg.Mapping.local_map_iter
        self.add_opacity_thres          = cfg.Mapping.add_opacity_thres
        self.add_depth_thres            = cfg.Mapping.add_depth_thres
        self.add_color_thres            = cfg.Mapping.add_color_thres
        self.sample_ratio               = cfg.Mapping.sample_ratio
        self.time                       = 0

        self.sample_ratio_init          = cfg.Mapping.sample_ratio_init
        self.local_map_iter_init        = cfg.Mapping.local_map_iter_init
        self.final_global_opt_iter      = cfg.Mapping.final_global_opt_iter

        self.init_scale_ratio           = cfg.Mapping.init_scale_ratio
        self.fusion_dist_thres          = cfg.Mapping.fusion_dist_thres
        self.cull_dist_thres            = cfg.Mapping.cull_dist_thres

        self.sw_optimize_freq           = cfg.Mapping.sw_optimize_freq
        self.sw_add_freq                = cfg.Mapping.sw_add_freq

        self.depth_weight               = cfg.Mapping.depth_weight
        self.color_weight               = cfg.Mapping.color_weight
        self.normal_weight              = cfg.Mapping.normal_weight
        self.reg_weight                 = cfg.Mapping.reg_weight
        self.reg_weight_n               = cfg.Mapping.reg_weight_n

        self.status_threshold           = cfg.Mapping.state_threshold

        self.alpha_p                    = cfg.Surfel.alpha_p
        self.alpha_n                    = cfg.Surfel.alpha_n

        self.only_mapping               = cfg.System.only_mapping
        self.use_sparse                 = cfg.Tracking.use_sparse
                
        self.sw_lr_params = edict({
            "position_lr"   : cfg.Mapping.position_lr,
            "feature_lr"    : cfg.Mapping.feature_lr,
            "opacity_lr"    : cfg.Mapping.opacity_lr,
            "scaling_lr"    : cfg.Mapping.scaling_lr,
            "rotation_lr"   : cfg.Mapping.rotation_lr,
        })
        
        self.global_lr_params = edict({
            "position_lr"   : cfg.Mapping.final_position_lr,
            "feature_lr"    : cfg.Mapping.final_feature_lr,
            "opacity_lr"    : cfg.Mapping.final_opacity_lr,
            "scaling_lr"    : cfg.Mapping.final_scaling_lr,
            "rotation_lr"   : cfg.Mapping.final_rotation_lr,
        })
        
    def mapping(self, frame, frame_map):
        
        frame_ele = {"time": self.time, "frame": frame, "frame_map": frame_map}

        self.frame_map = frame_map
        self.surfels_preprocess(frame)

        if self.time % self.sw_add_freq == 0:
            self.keyframe_manager.sliding_window.append(frame_ele)

        if self.time % self.sw_optimize_freq == 0:
            self.keyframe_manager.check_keyframe(frame_ele)
            self.frame_batch_optimization(frame)

        self.surfels_postprocess(frame)
        self.time += 1

    def keyframe_optimization(self, keyframe_num=-1):
        
        is_global_opt = (keyframe_num == -1)

        if is_global_opt:
            keyframe_num = self.keyframe_manager.get_keyframe_num()

        keyframe_num = min(keyframe_num, self.keyframe_manager.get_keyframe_num())

        geo_surfels_params = {
            "position" : self.surfels0.get_xyz.detach(),
            "normal": self.surfels0.get_normal.detach()
        }
        
        optimizer = torch.optim.Adam(self.surfels0.parametrize(self.global_lr_params), lr=0.0)
        
        for keyframe in self.keyframe_manager.keyframe_list.values():
            keyframe.cuda()

        random_idx = torch.randperm(keyframe_num)
        desc = "keyframe optimization" if not is_global_opt else "global optimization"
        pbar = tqdm(range(self.final_global_opt_iter * keyframe_num), desc=desc)
        for idx in pbar:
            random_idx = torch.randperm(keyframe_num)[0]
            keyframe_id = self.keyframe_manager.get_keyframe_ids()[random_idx]
            keyframe = self.keyframe_manager.keyframe_list[keyframe_id]

            rgb_mask = keyframe.frame_map["rgb_mask"].squeeze()
            geo_mask = keyframe.frame_map["geo_mask"].squeeze()

            render_ouput = self.renderer.render(keyframe, self.total_params)
            
            loss = self.compute_loss(render_ouput, keyframe.frame_map, (rgb_mask, geo_mask), geo_surfels_params)
            
            loss.backward()
                        
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix({"loss": loss.item()})

        for keyframe in self.keyframe_manager.keyframe_list.values():
            keyframe.cpu()
        torch.cuda.synchronize()

    def surfels_preprocess(self, frame):

        color_map   = self.frame_map["color_map"]
        depth_map   = self.frame_map["depth_map"]
        vertex_map  = self.frame_map["vertex_map_w"]
        normal_map  = self.frame_map["normal_map_w"]
        depth_mask  = self.frame_map["depth_map"] > 0
        
        if self.time > 0:

            self.get_render_output(frame)

            ht = frame.height
            wd = frame.width
            R = frame.c2w_matrix()[:3, :3]
            t = frame.c2w_matrix()[:3, 3]
            t0 = torch.zeros(3).to(frame.device)

            render_vmap, render_nmap = compute_vertex_and_normal(self.model_map["render_depth"], frame.intr)

            model_vmap = transform_map(render_vmap, R, t)
            model_nmap = transform_map(render_nmap, R, t0)
            model_mask = (self.model_map["render_depth"].squeeze() > 0.1) | (render_nmap == 0).all(dim=-1)

            frame_imap, depth_buff = project_surfels_to_frame(
                self.surfels0._xyz,
                self.surfels0._rotation,
                self.surfels0._stable_mask,
                frame.intr.to(frame.device),
                frame.world_view_transform,
                frame.full_proj_transform,
                1.0,
                frame.height,
                frame.width
            )

            preprocess_surfels(
                self.surfels0._xyz,
                self.surfels0._rotation,
                self.surfels0._scaling,
                self.surfels0.get_color,
                self.surfels0._confidence,
                self.surfels0._tic,
                self.surfels0._eta,
                self.surfels0._sigma2,
                self.surfels0._observe_count,
                self.surfels0._error_count,
                self.surfels0._stable_mask,
                frame.intr.to(frame.device),
                frame.world_view_transform,
                frame.full_proj_transform,
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                self.frame_map["depth_map"],
                self.frame_map["geo_mask"],
                frame_imap,
                depth_buff,
                model_vmap,
                model_nmap,
                model_mask,
                self.surfels0._inview_mask,
                self.surfels0._surface_mask,
                self.fusion_dist_thres,
                self.alpha_p,
                self.alpha_n
            )
            
            opacity_mask = self.model_map["render_opacity"] < self.add_opacity_thres
            depth_err = (self.model_map["render_depth"] - depth_map) # fillin the foreground hole, so the depth error is positive awared
            depth_res_mask = depth_err > self.add_depth_thres
            sample_mask = (opacity_mask | depth_res_mask ) & depth_mask
            sample_ratio = self.sample_ratio

        else:
            sample_mask = depth_mask
            sample_ratio = self.sample_ratio_init

        xyz, normal, color, dist, confidence, eta, sigma2 = self.sample_for_init_surfels(frame.intr, depth_map, vertex_map, normal_map, color_map, sample_mask, sample_ratio)
        self.surfels0.create_surfels(xyz, normal, color, dist, confidence, eta, sigma2, self.time)

        if self.time == 0:
            self.get_render_output(frame)

    def surfels_postprocess(self, frame):
        
        num1 = self.stable_mask.sum().item()
        num2 = self.stable_mask.size(0) - num1
        mask = self.surfels0._observe_count > 3

        self.stable_mask = self.surfels0.get_confidence > 10

        self.surfels0._stable_mask = self.stable_mask
    
    def frame_batch_optimization(self, frame):
        
        optimizer = torch.optim.Adam(self.surfels0.parametrize(self.sw_lr_params), lr=0.0)
        
        window_size = len(self.keyframe_manager.sliding_window)
        
        geo_surfels_params = {
            "position" : self.surfels0.get_xyz.detach(),
            "normal": self.surfels0.get_normal.detach()
        }
        
        iters_num = self.local_map_iter * window_size if self.time > 0 else self.local_map_iter_init
        pbar = tqdm(range(iters_num), desc="frame batch optimization")
        color_est = None
        color_ref = None
        for iter in pbar:
            
            random_idx = torch.randperm(window_size)[0]
            frame_ele = self.keyframe_manager.sliding_window[random_idx]

            rgb_mask = frame_ele["frame_map"]["rgb_mask"].squeeze()
            geo_mask = frame_ele["frame_map"]["geo_mask"].squeeze()
            
            render_ouput = self.renderer.render(frame_ele["frame"], self.total_params)
            
            loss = self.compute_loss(render_ouput, frame_ele["frame_map"], (rgb_mask, geo_mask), geo_surfels_params)
            
            loss.backward()
                        
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix({"loss": loss.item()})
        
        if cfg_debug:
            color_est = render_ouput["color"].permute([1, 2, 0])
            color_ref = frame_ele["frame_map"]["color_map"]
            color_est_np = (color_est * 255).detach().cpu().numpy().astype(np.uint8)[..., ::-1]
            color_ref_np = (color_ref * 255).detach().cpu().numpy().astype(np.uint8)[..., ::-1]
            im = np.concatenate([color_est_np, color_ref_np], axis=1)
            cv2.imwrite(f"./debug/color_comp_{self.time:05d}_1_after.png", im)

        self.surfels2.detach()


    def compute_loss(self, render_output, frame_input, render_mask, geo_surfels_params):

        est_color   = render_output["color"].permute([1, 2, 0])
        est_depth   = render_output["depth"].permute([1, 2, 0])
        est_normal  = render_output["normal"].permute([1, 2, 0])
        ref_color   = frame_input["color_map"]
        ref_depth   = frame_input["depth_map"]
        ref_normal  = frame_input["normal_map_c"]

        normal_loss = torch.tensor(0.0, device=est_color.device)
        depth_loss  = torch.tensor(0.0, device=est_color.device)
        reg_loss    = torch.tensor(0.0, device=est_color.device)

        rgb_mask, geo_mask = render_mask
        render_mask = rgb_mask & geo_mask

        check_nan("est_color", render_output["color"])
        check_nan("est_depth", render_output["depth"])
        check_nan("est_normal", render_output["normal"])
        check_nan("ref_color", frame_input["color_map"])
        check_nan("ref_depth", frame_input["depth_map"])
        check_nan("ref_normal", frame_input["normal_map_c"])
        check_nan("geo_pos", geo_surfels_params['position'])
        check_nan("geo_normal", geo_surfels_params['normal'])
        check_nan("surfels0_xyz", self.surfels0.get_xyz)
        check_nan("surfels0_normal", self.surfels0.get_normal)

        # cv2.imwrite("rgb_mask.png", rgb_mask.cpu().numpy() * 255)
        # cv2.imwrite("geo_mask.png", geo_mask.cpu().numpy() * 255)

        # ---- Color loss
        color_loss = torch.abs(ref_color - est_color)[render_mask].mean()

        # ---- Depth loss
        if ref_depth is not None and self.depth_weight > 0:
            depth_error = ref_depth - est_depth
            if render_mask.any():
                depth_loss = torch.abs(depth_error[render_mask]).mean()

        # ---- Normal loss
        if ref_normal is not None and self.normal_weight > 0:
            # Clamp cosine_similarity for safety
            cos_dist = 1 - F.cosine_similarity(ref_normal, est_normal, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
            if render_mask.any():
                normal_loss = torch.abs(cos_dist[render_mask]).mean()

        # ---- Regularization
        if self.reg_weight > 0:
            reg_position = torch.norm(geo_surfels_params['position'] - self.surfels0.get_xyz)
            reg_normal = 1 - F.cosine_similarity(
                geo_surfels_params['normal'],
                self.surfels0.get_normal,
                dim=-1
            ).clamp(-1 + 1e-6, 1 - 1e-6)
            reg_loss = reg_position.mean() + self.reg_weight_n * reg_normal.abs().mean()

        # ---- Total loss
        total_loss = self.color_weight * color_loss + self.depth_weight * depth_loss + self.normal_weight * normal_loss + self.reg_weight * reg_loss

        if torch.isnan(total_loss):
            print("NaN in loss, stopping training")
            exit(0)

        return total_loss

    def sample_for_init_surfels(self, intr, depth_map, vertices_map, normals_map, colors_map, sample_mask, sample_ratio):
        
        fx, fy, cx, cy = intr
        sample_num = int(sample_mask.sum() * sample_ratio)
        
        if sample_num == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        
        invalid_normal_mask = (normals_map == 0).all(dim=-1)
        sample_mask = sample_mask & ~invalid_normal_mask[..., None]

        if sample_num > sample_mask.sum():
            sample_num = sample_mask.sum()
        
        pad = 7
        edge_mask = torch.zeros_like(sample_mask).to(torch.bool)
        edge_mask[pad:-pad, pad:-pad] = True
        sample_mask = sample_mask & edge_mask
                
        ht, wd = sample_mask.shape[:2]
        indeices = torch.arange(ht * wd).to(torch.long).cuda()
        indeices = indeices[sample_mask.reshape(-1)]
        indeices = indeices[torch.randperm(indeices.shape[0])[:sample_num]]
        
        points_sampled = vertices_map.reshape(-1, 3)[indeices]
        colors_sampled = colors_map.reshape(-1, 3)[indeices]
        normals_sampled = normals_map.reshape(-1, 3)[indeices]
        
        depths_sampled = depth_map.reshape(-1)[indeices]
        dist = torch.stack([
            self.init_scale_ratio * depths_sampled / fx, 
            self.init_scale_ratio * depths_sampled / fy, 
            torch.zeros_like(depths_sampled)], dim=-1)
                
        confidences = (1.0 / depths_sampled) ** 2

        sigma2_p = (depths_sampled * self.alpha_p) ** 2
        sigma2_n = (depths_sampled * self.alpha_n) ** 2
        sigma2 = torch.stack([sigma2_p, sigma2_n], dim=-1)

        x0 = torch.cat([points_sampled, normals_sampled], dim=-1)

        eta = torch.zeros_like(x0)
        eta[:, :3] = points_sampled / sigma2_p[:, None]
        eta[:, 3:] = normals_sampled / sigma2_n[:, None]

        return points_sampled, normals_sampled, colors_sampled, dist, confidences, eta, sigma2

    def get_render_output(self, frame):
        
        with torch.no_grad():    
            render_output = self.renderer.render(frame, self.total_params)
            torch.cuda.synchronize()

        self.model_map = {}
        self.model_map["render_color"]      = render_output["color"].permute([1, 2, 0])
        self.model_map["render_depth"]      = render_output["depth"].permute([1, 2, 0])
        self.model_map["render_normal"]     = render_output["normal"].permute([1, 2, 0])
        self.model_map["render_opacity"]    = render_output["opacity"].permute([1, 2, 0])

        return self.model_map

    def merge_surfels(self) -> None:

        _xyz1           = self.surfels1._xyz
        _features_dc1   = self.surfels1._features_dc
        _features_rest1 = self.surfels1._features_rest
        _scaling1       = self.surfels1._scaling
        _rotation1      = self.surfels1._rotation
        _opacity1       = self.surfels1._opacity
        _confidence1    = self.surfels1._confidence
        _eta            = self.surfels1._eta
        _sigma2         = self.surfels1._sigma2
        _observe_count1 = self.surfels1._observe_count
        _tic1           = self.surfels1._tic
        _error_count1   = self.surfels1._error_count
        _inview_mask1   = self.surfels1._inview_mask
        _surface_mask1  = self.surfels1._surface_mask

        _xyz2           = self.surfels2._xyz
        _features_dc2   = self.surfels2._features_dc
        _features_rest2 = self.surfels2._features_rest
        _scaling2       = self.surfels2._scaling
        _rotation2      = self.surfels2._rotation
        _opacity2       = self.surfels2._opacity
        _confidence2    = self.surfels2._confidence
        _eta            = self.surfels2._eta
        _sigma2         = self.surfels2._sigma2
        _observe_count2 = self.surfels2._observe_count
        _tic2           = self.surfels2._tic
        _error_count2   = self.surfels2._error_count
        _inview_mask2   = self.surfels2._inview_mask
        _surface_mask2  = self.surfels2._surface_mask

        _rotation1  = torch.nan_to_num(_rotation1, nan=1.0) # TODO: (pxk) cuda illegal access!
        _rotation2  = torch.nan_to_num(_rotation2, nan=1.0)

        self.surfels0._xyz = torch.cat([_xyz1, _xyz2], dim=0)
        self.surfels0._features_dc = torch.cat([_features_dc1, _features_dc2], dim=0)
        self.surfels0._features_rest = torch.cat([_features_rest1, _features_rest2], dim=0)
        self.surfels0._scaling = torch.cat([_scaling1, _scaling2], dim=0)
        self.surfels0._rotation = torch.cat([_rotation1, _rotation2], dim=0)
        self.surfels0._opacity = torch.cat([_opacity1, _opacity2], dim=0)

        self.surfels0._confidence = torch.cat([_confidence1, _confidence2], dim=0)
        self.surfels0._eta = torch.cat([_eta, _eta], dim=0)
        self.surfels0._sigma2 = torch.cat([_sigma2, _sigma2], dim=0)
        self.surfels0._observe_count = torch.cat([_observe_count1, _observe_count2], dim=0)
        self.surfels0._tic = torch.cat([_tic1, _tic2], dim=0)
        self.surfels0._error_count = torch.cat([_error_count1, _error_count2], dim=0)
        self.surfels0._inview_mask = torch.cat([_inview_mask1, _inview_mask2], dim=0)
        self.surfels0._surface_mask = torch.cat([_surface_mask1, _surface_mask2], dim=0)

        stable_mask = torch.zeros(self.surfels0._xyz.shape[0]).float().cuda()
        stable_mask[:self.surfels1.size] = 1 # unstable
        stable_mask[self.surfels1.size:] = 2 # stable

        return stable_mask

    @property
    def total_params(self):

        xyz         = self.surfels0.get_xyz
        opacity     = self.surfels0.get_opacity
        scaling     = self.surfels0.get_scaling
        rotations   = self.surfels0.get_rotation
        normal      = self.surfels0.get_normal
        shs         = self.surfels0.get_features
        radius      = self.surfels0.get_radius

        rotations    = torch.nan_to_num(rotations, nan=1.0)

        return {
            "xyz"           : xyz.contiguous(),
            "opacity"       : opacity.contiguous(),
            "scales"        : scaling.contiguous(),
            "rotations"     : rotations.contiguous(),
            "normal"        : normal.contiguous(),
            "shs"           : shs.contiguous(),
            "radius"        : radius.contiguous(),
        }