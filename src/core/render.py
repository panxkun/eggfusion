import numpy as np
import math
import torch

from src.utils.frame import Frame as Camera
from src.core.utils import transform_map

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings as GaussianRasterizationSettings_depth,
    GaussianRasterizer as GaussianRasterizer_depth,
)

from src.core.utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid
)

class Renderer:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, cfg):
        self.raster_settings    = None
        self.rasterizer         = None
        self.bg_color           = torch.tensor([0, 0, 0]).cuda().float()
        self.scaling_modifier   = 1.0
        self.max_sh_degree      = cfg.Surfel.max_sh_degree

        if cfg.Surfel.active_sh_degree < 0:
            self.active_sh_degree = self.max_sh_degree
        else:
            self.active_sh_degree = cfg.Surfel.active_sh_degree

        self.setup_functions()

    def get_scaling(self, scaling):
        return self.scaling_activation(scaling)

    def get_rotation(self, rotaion):
        return self.rotation_activation(rotaion)

    def get_covariance(self, scaling, rotaion, scaling_modifier=1):
        return self.covariance_activation(scaling, scaling_modifier, rotaion)

    def render(self, viewpoint_camera: Camera, gaussian_data):
        
        tanfovx = math.tan(viewpoint_camera.fovx * 0.5)
        tanfovy = math.tan(viewpoint_camera.fovy * 0.5)
        
        ht = viewpoint_camera.height
        wd = viewpoint_camera.width
        
        self.raster_settings    = GaussianRasterizationSettings_depth(
            image_height        = int(viewpoint_camera.height),
            image_width         = int(viewpoint_camera.width),
            projmatrix          = viewpoint_camera.full_proj_transform,
            cx                  = viewpoint_camera.cx,
            cy                  = viewpoint_camera.cy,
            tanfovx             = tanfovx,
            tanfovy             = tanfovy,
            bg                  = self.bg_color,
            scale_modifier      = self.scaling_modifier,
            viewmatrix          = viewpoint_camera.world_view_transform,
            sh_degree           = self.active_sh_degree,
            campos              = viewpoint_camera.camera_center,
            prefiltered         = False,
            debug               = False,
        )

        self.rasterizer = GaussianRasterizer_depth(raster_settings=self.raster_settings)

        tile_mask = torch.ones((ht + 15) // 16, (wd + 15) // 16, dtype=torch.int32).cuda()
        
        render_results = self.rasterizer(
            means3D         = gaussian_data["xyz"],
            opacities       = gaussian_data["opacity"],
            shs             = gaussian_data["shs"],
            colors_precomp  = None,
            scales          = gaussian_data["scales"],
            rotations       = gaussian_data["rotations"],
            cov3D_precomp   = None,
            tile_mask       = tile_mask,
        )

        rendered_image      = render_results[0]
        rendered_normal     = render_results[1]
        rendered_depth      = render_results[2]
        rendered_opacity    = render_results[3]
        
        results = {
            "color"     : rendered_image,
            "depth"     : rendered_depth,
            "normal"    : rendered_normal,
            "opacity"   : rendered_opacity
        }
        return results
