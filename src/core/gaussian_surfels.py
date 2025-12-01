import os
from torch import nn
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from src.utils.sh_utils import RGB2SH, SH2RGB
from src.core.utils import (
    build_rotation, 
    build_covariance_from_scaling_rotation, 
    inverse_sigmoid, 
    compute_rot
)

class GaussianSurfels(object):

    def __init__(self, cfg) -> None:
        self._xyz               = torch.empty(0).float().cuda()
        self._features_dc       = torch.empty(0).float().cuda()
        self._features_rest     = torch.empty(0).float().cuda()
        self._scaling           = torch.empty(0).float().cuda()
        self._rotation          = torch.empty(0).float().cuda()
        self._opacity           = torch.empty(0).float().cuda()

        self._eta               = torch.empty(0).float().cuda()
        self._sigma2            = torch.empty(0).float().cuda()
        self._confidence        = torch.empty(0).float().cuda()
        self._observe_count     = torch.empty(0).int().cuda()
        self._tic               = torch.empty(0).int().cuda()
        self._error_count       = torch.empty(0).int().cuda()
        self._inview_mask       = torch.empty(0).bool().cuda()
        self._surface_mask      = torch.empty(0).bool().cuda()
        self._stable_mask       = torch.empty(0).bool().cuda()

        self.init_opacity       = cfg.Surfel.init_opacity
        self.scale_factor       = cfg.Surfel.scale_factor
        self.min_radius         = cfg.Surfel.min_radius
        self.max_radius         = cfg.Surfel.max_radius
        self.max_sh_degree      = cfg.Surfel.max_sh_degree
        self.active_sh_degree   = cfg.Surfel.active_sh_degree
        
        assert self.active_sh_degree <= self.max_sh_degree

        self.stable_grad_coeff  = cfg.Surfel.stable_grad_coeff
        self.confidence_thres   = cfg.Surfel.confidence_thres

        self.setup_functions()

    def setup_functions(self):
        
        self.scaling_activation         = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation      = build_covariance_from_scaling_rotation
        self.opacity_activation         = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation        = torch.nn.functional.normalize


    def delete(self, delte_mask):
        if delte_mask.sum() == 0:
            return
        
        self._xyz                   = self._xyz[~delte_mask]
        self._features_dc           = self._features_dc[~delte_mask]
        self._features_rest         = self._features_rest[~delte_mask]
        self._scaling               = self._scaling[~delte_mask]
        self._rotation              = self._rotation[~delte_mask]
        self._opacity               = self._opacity[~delte_mask]
        self._eta                    = self._eta[~delte_mask]
        self._sigma2                = self._sigma2[~delte_mask]
        self._confidence            = self._confidence[~delte_mask]
        self._observe_count         = self._observe_count[~delte_mask]
        self._tic                   = self._tic[~delte_mask]
        self._error_count           = self._error_count[~delte_mask]
        self._inview_mask           = self._inview_mask[~delte_mask]
        self._surface_mask          = self._surface_mask[~delte_mask]
        self._stable_mask           = self._stable_mask[~delte_mask]
        
    def remove(self, remove_mask):
        xyz                         = self._xyz[remove_mask]
        features_dc                 = self._features_dc[remove_mask]
        features_rest               = self._features_rest[remove_mask]
        scaling                     = self._scaling[remove_mask]
        rotation                    = self._rotation[remove_mask]
        opacity                     = self._opacity[remove_mask]
        eta                          = self._eta[remove_mask]
        sigma2                      = self._sigma2[remove_mask]
        confidence                  = self._confidence[remove_mask]
        observe_count               = self._observe_count[remove_mask]
        tic                         = self._tic[remove_mask]
        error_count                 = self._error_count[remove_mask]
        inview_mask                 = self._inview_mask[remove_mask]
        surface_mask                = self._surface_mask[remove_mask]
        stable_mask                 = self._stable_mask[remove_mask]

        surfel_params = {
            "xyz"                   : xyz,
            "features_dc"           : features_dc,
            "features_rest"         : features_rest,
            "scaling"               : scaling,
            "rotation"              : rotation,
            "opacity"               : opacity,
            "observe_count"         : observe_count,
            "tic"                   : tic,
            "confidence"            : confidence,
            "eta"                    : eta,
            "sigma2"                : sigma2,
            "error_count"           : error_count,
            "inview_mask"           : inview_mask,
            "surface_mask"          : surface_mask,
            "stable_mask"           : stable_mask
        }
        
        self.delete(remove_mask)
        
        return surfel_params

    def detach(self):
        self._xyz                   = self._xyz.detach()
        self._features_dc           = self._features_dc.detach()
        self._features_rest         = self._features_rest.detach()
        self._scaling               = self._scaling.detach()
        self._rotation              = self._rotation.detach()
        self._opacity               = self._opacity.detach()
        self._observe_count         = self._observe_count.detach()
        self._tic                   = self._tic.detach()
        self._eta                    = self._eta.detach()
        self._sigma2                = self._sigma2.detach()
        self._confidence            = self._confidence.detach()
        self._error_count           = self._error_count.detach()
        self._inview_mask           = self._inview_mask.detach()
        self._surface_mask          = self._surface_mask.detach()
        self._stable_mask           = self._stable_mask.detach()

    def parametrize(self, cfg):
        self._xyz                   = nn.Parameter(self._xyz.requires_grad_(True))
        self._features_dc           = nn.Parameter(self._features_dc.requires_grad_(True))
        self._features_rest         = nn.Parameter(self._features_rest.requires_grad_(True))
        self._scaling               = nn.Parameter(self._scaling.requires_grad_(True))
        self._rotation              = nn.Parameter(self._rotation.requires_grad_(True))
        self._opacity               = nn.Parameter(self._opacity.requires_grad_(True))
        
        l = [
            {"params": [self._xyz],             "lr": cfg.position_lr,          "name": "xyz"},
            {"params": [self._features_dc],     "lr": cfg.feature_lr,           "name": "f_dc"},
            {"params": [self._features_rest],   "lr": cfg.feature_lr / 20.0,    "name": "f_rest"},
            {"params": [self._opacity],         "lr": cfg.opacity_lr,           "name": "opacity"},
            {"params": [self._scaling],         "lr": cfg.scaling_lr,           "name": "scaling"},
            {"params": [self._rotation],        "lr": cfg.rotation_lr,          "name": "rotation"},
        ]
        return l

    def cat(self, paramters):
        self._xyz                   = torch.cat([self._xyz, paramters["xyz"]])
        self._features_dc           = torch.cat([self._features_dc, paramters["features_dc"]])
        self._features_rest         = torch.cat([self._features_rest, paramters["features_rest"]])
        self._scaling               = torch.cat([self._scaling, paramters["scaling"]])
        self._rotation              = torch.cat([self._rotation, paramters["rotation"]], dim=0)
        self._opacity               = torch.cat([self._opacity, paramters["opacity"]])
        self._observe_count         = torch.cat([self._observe_count, paramters["observe_count"]])
        self._tic                   = torch.cat([self._tic, paramters["tic"]])
        self._eta                   = torch.cat([self._eta, paramters["eta"]])
        self._sigma2                = torch.cat([self._sigma2, paramters["sigma2"]])
        self._confidence            = torch.cat([self._confidence, paramters["confidence"]])
        self._error_count           = torch.cat([self._error_count, paramters["error_count"]])
        self._inview_mask           = torch.cat([self._inview_mask, paramters["inview_mask"]])
        self._surface_mask          = torch.cat([self._surface_mask, paramters["surface_mask"]])
        self._stable_mask           = torch.cat([self._stable_mask, paramters["surface_mask"]])

    def create_surfels(self, xyz, normal, color, dist, confidence, eta, sigma2, time):

        assert xyz.shape[0] == color.shape[0] and color.shape[0] == normal.shape[0]
        if xyz.shape[0] < 1:
            return
        
        valid_mask  = normal.sum(dim=-1) != 0
        xyz         = xyz[valid_mask]
        normal      = normal[valid_mask]
        color       = color[valid_mask]
        
        points_num  = xyz.shape[0]

        features = torch.zeros((points_num, 3, (self.max_sh_degree + 1) ** 2)).cuda().float()
        features[:, :3,  0] = RGB2SH(color)

        scales = torch.log(dist.cuda().float())
        scales[..., -1] = -1.0e10
        
        z_axis = torch.tensor([0, 0, 1]).repeat(points_num, 1).cuda().float()
        rots = compute_rot(z_axis, normal)
        
        opacities = inverse_sigmoid(self.init_opacity * torch.ones((points_num, 1)).cuda().float())
        
        observe_count = torch.zeros((points_num)).cuda().int()
        
        tic = torch.ones((points_num)).cuda().int() * time
        
        confidence = confidence.cuda().float()
        
        error_count = torch.zeros_like(confidence).int()

        inview_mask = torch.ones_like(confidence).bool()
        surface_mask = torch.ones_like(confidence).bool()
        stable_mask = torch.zeros_like(confidence).bool()

        params = {
            "xyz"           : xyz,
            "features_dc"   : features[..., 0:1].transpose(1, 2).contiguous(),
            "features_rest" : features[..., 1:].transpose(1, 2).contiguous(),
            "scaling"       : scales,
            "rotation"      : rots,
            "opacity"       : opacities,
            "observe_count" : observe_count,
            "tic"           : tic,
            "confidence"    : confidence,
            "eta"           : eta,
            "sigma2"        : sigma2,
            "error_count"   : error_count,
            "inview_mask"   : inview_mask,
            "surface_mask"  : surface_mask,
            "stable_mask"   : stable_mask,
        }
        self.cat(params)


    @property
    def get_params(self):
        non_empty       = self.size > 0
        xyz             = self.get_xyz              if non_empty else torch.empty(0)
        opacity         = self.get_opacity          if non_empty else torch.empty(0)
        scales          = self.get_scaling          if non_empty else torch.empty(0)
        rotations       = self.get_rotation         if non_empty else torch.empty(0)
        normal          = self.get_normal           if non_empty else torch.empty(0)
        shs             = self.get_features         if non_empty else torch.empty(0)
        radius          = self.get_radius           if non_empty else torch.empty(0)
        observe_count   = self.get_observe_count    if non_empty else torch.empty(0)
        tic             = self.get_tic              if non_empty else torch.empty(0)
        eta             = self._eta                  if non_empty else torch.empty(0)
        sigma2          = self._sigma2              if non_empty else torch.empty(0)
        confidence      = self._confidence          if non_empty else torch.empty(0)
        error_count     = self._error_count         if non_empty else torch.empty(0)
        inview_mask     = self._inview_mask         if non_empty else torch.empty(0)
        surface_mask    = self._surface_mask        if non_empty else torch.empty(0)
        stable_mask     = self._stable_mask         if non_empty else torch.empty(0)
        
        params = {
            "xyz"           : xyz.cuda().float(),
            "opacity"       : opacity.cuda().float(),
            "scales"        : scales.cuda().float(),
            "rotations"     : rotations.cuda().float(),
            "shs"           : shs.cuda().float(),
            "radius"        : radius.cuda().float(),
            "normal"        : normal.cuda().float(),
            "observe_count" : observe_count.cuda().int(),
            "tic"           : tic.cuda().int(),
            "eta"            : eta.cuda().float(),
            "sigma2"        : sigma2.cuda().float(),
            "confidence"    : confidence.cuda().float(),
            "error_count"   : error_count.cuda().int(),
            "inview_mask"   : inview_mask.cuda().int(),
            "surface_mask"  : surface_mask.cuda().int(),
            "stable_mask"   : stable_mask.cuda().int()
        }
        return params


    def construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        l.append("opacity")
        return l
    
    def save_ply(self, path):
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz         = self._xyz.detach().cpu().numpy()
        f_dc        = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest      = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scaling     = self._scaling.detach().cpu().numpy()
        rotation    = self._rotation.detach().cpu().numpy()
        opacities   = self._opacity.detach().cpu().numpy()

        N = len(xyz)
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(N, dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, scaling, rotation, opacities), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])),axis=1)
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scaling_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rotation")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        self._xyz           = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc   = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity       = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling       = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation      = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def size(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_radius(self):
        scales = self.get_scaling
        min_length, _ = torch.min(scales, dim=1)
        radius = (torch.sum(scales, dim=1) - min_length) / 2
        return radius

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_R(self):
        return build_rotation(self.rotation_activation(self._rotation))

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        scales = self.get_scaling
        R = self.get_R
        min_indices = torch.argmin(scales, dim=1)
        normal = torch.gather(
            R.transpose(1, 2),
            1,
            min_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )
        normal = normal[:, 0, :]
        mag = torch.norm(normal, dim=-1, keepdim=True)
        return normal / (mag + 1e-8)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_color(self):
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()
        color = SH2RGB(f_dc.reshape(-1, 3))
        return color

    @property
    def get_base_color(self):
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()
        color = SH2RGB(f_dc.reshape(-1, 3))
        return color

    @property
    def get_observe_count(self):
        return self._observe_count

    @property
    def get_tic(self):
        return self._tic
    
    @property
    def get_confidence(self):
        return torch.sum(1.0 / self._sigma2, dim=-1)