import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import collections
from easydict import EasyDict as edict
import math

class Lie():
    def skew_sym_mat(self, x):
        ssm = torch.tensor([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ], device=x.device, dtype=x.dtype)
        return ssm

    def so3_to_SO3(self, theta):
        W = self.skew_sym_mat(theta)
        angle = torch.norm(theta)
        I = torch.eye(3, device=theta.device, dtype=theta.dtype)
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        
        return torch.where(angle < 1e-5,
            I + W + 0.5 * W @ W,
            I + (sin_angle / angle) * W + ((1 - cos_angle) / (angle**2)) * W @ W
        )

    def SO3_to_so3(self, R):
        trace = torch.trace(R)
        theta = torch.acos((trace - 1) / 2)
        lnR = (R - R.transpose(-2, -1)) / (2 * torch.sin(theta))
        w0 = lnR[..., 2, 1]
        w1 = lnR[..., 0, 2]
        w2 = lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def V(self, theta):
        W = self.skew_sym_mat(theta)
        W2 = W @ W
        angle = torch.norm(theta)
        I = torch.eye(3, device=theta.device, dtype=theta.dtype)
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        
        return torch.where(angle < 1e-5,
            I + 0.5 * W + (1.0 / 6.0) * W2,
            I + W * ((1.0 - cos_angle) / (angle**2)) + W2 * ((angle - sin_angle) / (angle**3))
        )

    def se3_to_SE3(self, tau):
        theta, rho = tau.split([3, 3], dim=0)
        T = torch.eye(4, device=tau.device, dtype=tau.dtype)
        T[:3, :3] = self.so3_to_SO3(theta)
        T[:3,  3] = self.V(theta) @ rho
        return T

    def SE3_to_se3(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        theta = self.SO3_log(R)
        rho = self.V(theta).inverse() @ t
        return torch.cat([rho, theta])


lie = Lie()

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getProjectionMatrix(znear, zfar, cx, cy, fx, fy, W, H):
    left    = ((2 * cx - W) / W - 1.0) * W / 2.0
    right   = ((2 * cx - W) / W + 1.0) * W / 2.0
    top     = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom  = ((2 * cy - H) / H - 1.0) * H / 2.0
    
    left    = znear / fx * left
    right   = znear / fx * right
    top     = znear / fy * top
    bottom  = znear / fy * bottom

    z_sign = 1.0
    
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def getProjectionMatrix_v2(znear, zfar, fovX, fovY):
    
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top     = tanHalfFovY * znear
    bottom  = -top
    right   = tanHalfFovX * znear
    left    = -right

    z_sign = 1.0

    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2Camera(R, t):
    Rt = torch.eye(4, device=R.device)
    Rt[:3, :3] = R
    Rt[:3,  3] = t
    return Rt

def getCamera2World(R, t):
    w2c = getWorld2Camera(R, t)
    return torch.inverse(w2c)