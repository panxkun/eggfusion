import json
import os
import numpy as np
import numpy
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import ms_ssim
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics.functional.image.lpips")

lpips_estimator = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def compute_gradients(depth_map):
    # Handle both numpy arrays and PyTorch tensors
    if isinstance(depth_map, torch.Tensor):
        # PyTorch version for CUDA tensors
        if len(depth_map.shape) == 3:
            depth_map = depth_map.squeeze()
        
        # Compute gradients using torch.diff
        Gx = torch.diff(depth_map, dim=1, append=depth_map[:, -1:])
        Gy = torch.diff(depth_map, dim=0, append=depth_map[-1:, :])
        Gxy = torch.sqrt(Gx**2 + Gy**2)
        return Gx, Gy, Gxy
    else:
        # Original numpy version
        Gx = np.diff(depth_map, axis=1, append=depth_map[:, -1:])
        Gy = np.diff(depth_map, axis=0, append=depth_map[-1:, :])
        Gxy = np.sqrt(Gx**2 + Gy**2)
        return Gx, Gy, Gxy

def matrix_to_tum_format(ts, matrix):
    R = matrix[:3, :3]
    
    q = Rotation.from_matrix(R).as_quat()
    p = matrix[:3,  3]
    
    tum_line = [ts, p[0], p[1], p[2], q[0], q[1], q[2], q[3]]

    return tum_line

def eval_traj_func(poses_ref, poses_est):

    def align(model, data):
        """Align two trajectories using the method of Horn (closed-form).

        Args:
            model -- first trajectory (3xn)
            data -- second trajectory (3xn)

        Returns:
            rot -- rotation matrix (3x3)
            trans -- translation vector (3x1)
            trans_error -- translational error per point (1xn)

        """
        numpy.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1)
        data_zerocentered = data - data.mean(1)

        W = numpy.zeros((3, 3))
        for column in range(model.shape[1]):
            W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
        S = numpy.matrix(numpy.identity(3))
        if numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0:
            S[2, 2] = -1
        rot = U * S * Vh
        trans = data.mean(1) - rot * model.mean(1)

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error
    if isinstance(poses_est, torch.Tensor):
        poses_est = poses_est.cpu().numpy()
    if isinstance(poses_ref, torch.Tensor):
        poses_ref = poses_ref.cpu().numpy()

    poses_est = np.matrix(poses_est.transpose())
    poses_ref = np.matrix(poses_ref.transpose())
    _, _, trans_error = align(poses_est, poses_ref)
    ate_rmse = numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)) * 100
    return ate_rmse

    return ape_stat


def eval_render_func(ref_color, ref_depth, est_color, est_depth):

    depth_mask = (ref_depth > 0).squeeze()

    est_color[depth_mask == False] = 0.0
    ref_color[depth_mask == False] = 0.0
    
    mse_loss = torch.nn.functional.mse_loss(est_color[depth_mask], ref_color[depth_mask])
    pnsr = 10 * torch.log10(1 / mse_loss)

    ssim = ms_ssim(est_color.permute([2, 0, 1])[None], ref_color.permute([2, 0, 1])[None], data_range=1.0, size_average=True)
        
    lpips = lpips_estimator(torch.clamp(est_color.permute([2, 0, 1])[None], 0.0, 1.0), torch.clamp(ref_color.permute([2, 0, 1])[None], 0.0, 1.0))
    
    depth_l1 = torch.abs(est_depth - ref_depth)[depth_mask].mean()

    return pnsr.item(), ssim.item(), lpips.item(), depth_l1.item()
    
    
