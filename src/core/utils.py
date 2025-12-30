import numpy as np
import torch
import torch.nn.functional as F


def l1_loss(network_output, gt, weight=None):
    if weight is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return torch.mean(torch.abs((network_output - gt)).sum(dim=-1) * weight)
    

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def quaternion_from_axis_angle(axis, angle):
    axis = axis / (torch.norm(axis, p=2, dim=-1, keepdim=True) + 1e-8)
    half_angle = angle / 2
    real_part = torch.cos(half_angle).type_as(axis)
    complex_part = axis * torch.sin(half_angle).type_as(axis)
    quaternion = torch.cat([real_part, complex_part], dim=1)
    return quaternion

def compute_rot(init_vec, target_vec):
    axis = torch.cross(init_vec, target_vec, dim=1)
    axis = axis / (torch.norm(axis, p=2, dim=-1, keepdim=True) + 1e-8)
    angle = torch.acos(torch.sum(init_vec * target_vec, dim=1)).unsqueeze(-1)
    rots = quaternion_from_axis_angle(axis, angle)
    return rots

def transform_map(points, R, t):
    h, w, c = points.shape
    points = points.reshape(-1, c)
    new_points = points @ R.T + t
    return new_points.reshape(h, w, c)

def compute_incident_angle(normal_map, intr):
    H, W, C = normal_map.shape
    fx, fy, cx, cy = intr
    h_grid, w_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    proj_map = torch.ones(H, W, 3).cuda()
    proj_map[..., 0] = (w_grid.cuda() - cx) / fx
    proj_map[..., 1] = (h_grid.cuda() - cy) / fy
    mag = torch.norm(proj_map, dim=-1, keepdim=True)
    proj_map = proj_map / (mag + 1e-8)
    view_normal_dist = torch.abs(F.cosine_similarity(normal_map, proj_map, dim=-1))
    return view_normal_dist[..., None]

def compute_confidence(coords, center, max_radius, two_sigma_2):
    radialDist = torch.norm(coords - center, dim=-1) / max_radius
    confidence = torch.exp(-radialDist ** 2 / two_sigma_2)
    return confidence


def depth2pcd(depth, intr):
    # Create a grid of (x, y) coordinates
    h, w = depth.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.squeeze()

    fx, fy, cx, cy = intr

    # Convert to camera coordinates
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy

    # Stack to create point cloud
    pcd = np.concatenate((x_cam[..., np.newaxis], y_cam[..., np.newaxis], z[..., np.newaxis]), axis=-1)
    return pcd