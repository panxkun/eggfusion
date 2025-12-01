import torch
import cuda_tracking_ext as cuda_tracking

class ProjectiveTransformFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transform, disps, intr):
        
        ht, wd = disps.shape[:2]
        
        intr = torch.Tensor(intr).cuda()
        warped_grid = torch.zeros(ht, wd, 2, device=disps.device)
        jacob_dxdxi = torch.zeros(ht, wd, 2, 6, device=disps.device)
        
        cuda_tracking.projective_transform_cuda(transform, disps, intr, warped_grid, jacob_dxdxi)
        
        return warped_grid, jacob_dxdxi

def projective_transform(transform, disps, intr):# -> Any | Any | None:
    return ProjectiveTransformFunction.apply(transform, disps, intr)

class RGBOptimizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame1, frame2, level, coords, J):
        
        intensity_prev = frame1.intensity_pyramid[level]
        intensity_curr = frame2.intensity_pyramid[level]
        grad_prev = frame1.grad_pyramid[level]
        grad_curr = frame2.grad_pyramid[level]
        mask_prev = frame1.mask_pyramid[level]
        mask_curr = frame2.mask_pyramid[level]
 
        Hcc = torch.zeros(6, 6, device=coords.device)
        gc  = torch.zeros(6, 1, device=coords.device)
        
        bound = 0.95
        
        cuda_tracking.rgb_optimization_cuda(
            intensity_prev, intensity_curr, grad_prev, grad_curr, mask_prev, mask_curr, coords, J, Hcc, gc, bound
        )
        
        return Hcc, gc

def rgb_optimization(frame1, frame2, level, coords, J):
    return RGBOptimizeFunction.apply(frame1, frame2, level, coords, J)


class ICPOptimizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame1, frame2, level, transfrom, coords, J, angleThres, distThres):

        vmap_prev = frame1.vertex_pyramid[level]
        nmap_prev = frame1.normal_pyramid[level]    
        mask_prev = frame1.mask_pyramid[level]
        vmap_curr = frame2.vertex_pyramid[level]
        nmap_curr = frame2.normal_pyramid[level]
        mask_curr = frame2.mask_pyramid[level]

        Hcc = torch.zeros(6, 6, device=coords.device)
        gc  = torch.zeros(6, 1, device=coords.device)

        bound = 0.98
        cuda_tracking.icp_optimization_cuda(
            vmap_prev, nmap_prev, mask_prev, vmap_curr, nmap_curr, mask_curr, transfrom, coords, Hcc, gc, angleThres, distThres, bound
        )
        
        return Hcc, gc

def icp_optimization(frame1, frame2, level, transfrom, coords, J, angleThres=None, distThres=None):
    return ICPOptimizeFunction.apply(frame1, frame2, level, transfrom, coords, J, angleThres, distThres)

class ComputeVertexAndNormalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, intr):
        
        depth = depth.float()
        ht, wd = depth.shape[:2]
        fx, fy, cx, cy = intr
        vertex_map = torch.zeros(ht, wd, 3, device=depth.device)
        normal_map = torch.zeros(ht, wd, 3, device=depth.device)
        
        cuda_tracking.compute_vertex_and_normal_cuda(depth, fx, fy, cx, cy, vertex_map, normal_map)
        
        return vertex_map, normal_map

def compute_vertex_and_normal(depth, intr):
    return ComputeVertexAndNormalFunction.apply(depth, intr)


class GaussianFilterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame, window_size, sigma):
        
        frame = frame.float()
        ht, wd, ch = frame.shape[:3]
        
        out = torch.zeros_like(frame)
        
        cuda_tracking.gaussian_filter_cuda(frame, out, wd, ht, ch, window_size, sigma)
        
        return out
    
def gaussian_filter(frame, window_size, sigma):
    return GaussianFilterFunction.apply(frame, window_size, sigma)


class BilaFilterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame, windos_size, sigma_color, sigma_space):
        
        frame = frame.float()
        ht, wd = frame.shape[:2]
        
        out = torch.zeros_like(frame)
        
        cuda_tracking.bilateral_filter_cuda(frame, out, wd, ht, windos_size, sigma_color, sigma_space)
        
        return out
    

def bilateral_filter(frame, window_size, sigma_color, sigma_space):
    return BilaFilterFunction.apply(frame, window_size, sigma_color, sigma_space)


class GaussianDownsampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame):
        
        frame = frame.float()
        ht, wd, ch= frame.shape[:3]

        out = torch.zeros(ht // 2, wd // 2, ch, device=frame.device)
        
        cuda_tracking.gaussian_downsample_cuda(frame, out, wd, ht, ch)
        
        return out

def gaussian_downsample(frame):
    return GaussianDownsampleFunction.apply(frame)

class ComputeGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame):
        
        frame = frame.float()
        ht, wd = frame.shape[:2]
        
        grad_x = torch.zeros(ht, wd, device=frame.device)
        grad_y = torch.zeros(ht, wd, device=frame.device)
        
        cuda_tracking.compute_gradients_cuda(frame, grad_x, grad_y, wd, ht)
        
        return grad_x, grad_y
    

def compute_gradient(frame):
    return ComputeGradientFunction.apply(frame)


class SolveBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b, lm):
        
        A = A.float()
        b = b.float()
        
        x = torch.zeros_like(b).to(b.device).float()
        
        cuda_tracking.solve_block_cuda(A, b, lm, x)
        
        return x
    
def solve_block(A, b, lm=1.0e-6):
    return SolveBlockFunction.apply(A, b, lm)