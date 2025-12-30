#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>


__device__ __forceinline__ float3 cross(const float3& v1, const float3& v2) {
    return make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );
}

__device__ __forceinline__ float3 dot(const float3& v1, const float3& v2) {
    return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__device__ __forceinline__ float3 sum(const float3& v1, const float3& v2){
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __forceinline__ float sum(const float3& v){
    return v.x + v.y + v.z;
}

__device__ __forceinline__ float norm(const float3& v){
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float3 normalize(const float3& v){
    float inv_norm = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inv_norm, v.y * inv_norm, v.z * inv_norm);
}

__device__ __forceinline__ float3 nearest_interpolate(
    const float* __restrict__ map, 
    const int& wd, 
    const int& ht, 
    const float& x, 
    const float& y) {

    int ix = min(max(__float2int_rn(x), 0), wd - 1);
    int iy = min(max(__float2int_rn(y), 0), ht - 1);

    int idx = iy * wd + ix;
    
    return make_float3(map[idx * 3], map[idx * 3 + 1], map[idx * 3 + 2]);
}

__device__ __host__ __forceinline__ float3 operator+(const float3& v1, const float3& v2){
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __host__ __forceinline__ float3 operator-(const float3& v1, const float3& v2){
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}


__device__ __forceinline__ float bilinear_interpolate(
    const float* __restrict__ map, 
    const int& wd, const int& ht, 
    const float& x, const float& y) {

    int x0 = __float2int_rn(x);
    int y0 = __float2int_rn(y);

    x0 = (x0 < 0) ? 0 : (x0 >= wd - 1 ? wd - 2 : x0); 
    y0 = (y0 < 0) ? 0 : (y0 >= ht - 1 ? ht - 2 : y0);

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w10 = dx * (1.0f - dy);
    float w11 = dx * dy;

    float v00 = map[y0 * wd + x0];
    float v01 = map[y1 * wd + x0];
    float v10 = map[y0 * wd + x1];
    float v11 = map[y1 * wd + x1];

    return w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
}


__global__ void reduceSumKernel(
    float* __restrict__ input, 
    float* __restrict__ output, 
    const int w, 
    const int h,
    const int channels) {

    extern __shared__ float sharedData[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.y;

    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int index = y * w + x;

    sharedData[threadId] = 0;

    for (int c = 0; c < channels; ++c) {
        if (x < w && y < h) {
            sharedData[threadId] += input[index * channels + c];
        }
    }

    __syncthreads();


    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            sharedData[threadId] += sharedData[threadId + stride];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

__constant__ float c_transform[16];
__constant__ float c_intr[4];

__global__ void projective_transform_kernel(
    const float* __restrict__ disps, 
    float* __restrict__ warped_grid,
    float* __restrict__ jacob_dxdxi,
    const int wd, 
    const int ht) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * wd + x;

    if(x >= wd || y >= ht) return;

    float fx = c_intr[0];
    float fy = c_intr[1];
    float cx = c_intr[2];
    float cy = c_intr[3];
    float disp = disps[idx];

    float us = (x - cx) / fx;
    float vs = (y - cy) / fy;
    float zs = 1.0f;

    float Ps[4] = {us, vs, zs, disp};
    float Pt[4] = {0};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        Pt[i] = c_transform[i * 4 + 0] * Ps[0] +
                c_transform[i * 4 + 1] * Ps[1] +
                c_transform[i * 4 + 2] * Ps[2] +
                c_transform[i * 4 + 3] * Ps[3];
    }

    float ut = Pt[0] / Pt[2];
    float vt = Pt[1] / Pt[2];
    float zt = Pt[2];
    float dt = Pt[3] / Pt[2];

    float warped_x = fx * ut + cx;
    float warped_y = fy * vt + cy;
    warped_grid[2 * idx + 0] = 2.0f * warped_x / (wd - 1) - 1.0f;
    warped_grid[2 * idx + 1] = 2.0f * warped_y / (ht - 1) - 1.0f;

    float O = 0.0f; // Zero placeholder
    jacob_dxdxi[12 * idx + 0] = dt * fx;
    jacob_dxdxi[12 * idx + 1] = O;
    jacob_dxdxi[12 * idx + 2] = -ut * dt * fx;
    jacob_dxdxi[12 * idx + 3] = -ut * vt * fx;
    jacob_dxdxi[12 * idx + 4] = (1 + ut * ut) * fx;
    jacob_dxdxi[12 * idx + 5] = -vt * fx;
    jacob_dxdxi[12 * idx + 6] = O;
    jacob_dxdxi[12 * idx + 7] = dt * fy;
    jacob_dxdxi[12 * idx + 8] = -vt * dt * fy;
    jacob_dxdxi[12 * idx + 9] = -(1 + vt * vt) * fy;
    jacob_dxdxi[12 * idx + 10] = ut * vt * fy;
    jacob_dxdxi[12 * idx + 11] = ut * fy;

    float t0 = c_transform[3];
    float t1 = c_transform[7];
    float t2 = c_transform[11];
}

void launch_projective_transform_kernel(
    const torch::Tensor transform, 
    const torch::Tensor disps, 
    const torch::Tensor intr,
    torch::Tensor warped_grid, 
    torch::Tensor jacob_dxdxi
) {
    int ht = disps.size(0);
    int wd = disps.size(1);

    cudaMemcpyToSymbol(c_transform, transform.data_ptr<float>(), 16 * sizeof(float));
    cudaMemcpyToSymbol(c_intr, intr.data_ptr<float>(), 4 * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((wd + blockSize.x - 1) / blockSize.x, (ht + blockSize.y - 1) / blockSize.y);

    projective_transform_kernel<<<gridSize, blockSize>>>(
        disps.data_ptr<float>(), 
        warped_grid.data_ptr<float>(), 
        jacob_dxdxi.data_ptr<float>(), 
        wd, 
        ht
    );
}


__global__ void rgb_optimization_kernel(
    const float* __restrict__ intensity_prev, 
    const float* __restrict__ intensity_curr,
    const float* __restrict__ grad_prev,
    const float* __restrict__ grad_curr, 
    const bool* __restrict__ mask_prev, 
    const bool* __restrict__ mask_curr, 
    const float* __restrict__ coords, 
    const float* __restrict__ JacobianI, 
    float* __restrict__ JtJJtr,
    const int& wd, 
    const int& ht, 
    const float& bound
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x >= wd || y >= ht) return;

    float coord_x = coords[2 * idx + 0];
    float coord_y = coords[2 * idx + 1];

    if (coord_x < -bound || coord_x > bound || coord_y < -bound || coord_y > bound || 
        !mask_prev[idx] || !mask_curr[idx]) return;

    const float warped_x = (coord_x + 1.0f) * 0.5f * (wd  - 1);
    const float warped_y = (coord_y + 1.0f) * 0.5f * (ht - 1);

    // Bilinear interpolation for intensity and gradients
    const float intensity_est = bilinear_interpolate(intensity_curr, wd, ht, warped_x, warped_y);
    const float gx = bilinear_interpolate(grad_curr, wd, ht, warped_x, warped_y);
    const float gy = bilinear_interpolate(grad_curr + wd * ht, wd, ht, warped_x, warped_y);

    float J[6] = {
        JacobianI[idx * 12 + 0] * gx + JacobianI[idx * 12 + 0 + 6] * gy,
        JacobianI[idx * 12 + 1] * gx + JacobianI[idx * 12 + 1 + 6] * gy,
        JacobianI[idx * 12 + 2] * gx + JacobianI[idx * 12 + 2 + 6] * gy,
        JacobianI[idx * 12 + 3] * gx + JacobianI[idx * 12 + 3 + 6] * gy,
        JacobianI[idx * 12 + 4] * gx + JacobianI[idx * 12 + 4 + 6] * gy,
        JacobianI[idx * 12 + 5] * gx + JacobianI[idx * 12 + 5 + 6] * gy
    };

    float r = intensity_prev[idx] - intensity_est;

    int offset = idx * 27;
    JtJJtr[offset + 0] = J[0] * J[0]; 
    JtJJtr[offset + 1] = J[0] * J[1]; 
    JtJJtr[offset + 2] = J[0] * J[2];
    JtJJtr[offset + 3] = J[0] * J[3]; 
    JtJJtr[offset + 4] = J[0] * J[4]; 
    JtJJtr[offset + 5] = J[0] * J[5];
    JtJJtr[offset + 6] = J[1] * J[1]; 
    JtJJtr[offset + 7] = J[1] * J[2]; 
    JtJJtr[offset + 8] = J[1] * J[3];
    JtJJtr[offset + 9] = J[1] * J[4]; 
    JtJJtr[offset + 10] = J[1] * J[5]; 
    JtJJtr[offset + 11] = J[2] * J[2];
    JtJJtr[offset + 12] = J[2] * J[3]; 
    JtJJtr[offset + 13] = J[2] * J[4]; 
    JtJJtr[offset + 14] = J[2] * J[5];
    JtJJtr[offset + 15] = J[3] * J[3]; 
    JtJJtr[offset + 16] = J[3] * J[4]; 
    JtJJtr[offset + 17] = J[3] * J[5];
    JtJJtr[offset + 18] = J[4] * J[4]; 
    JtJJtr[offset + 19] = J[4] * J[5]; 
    JtJJtr[offset + 20] = J[5] * J[5];
    JtJJtr[offset + 21] = J[0] * r; 
    JtJJtr[offset + 22] = J[1] * r; 
    JtJJtr[offset + 23] = J[2] * r;
    JtJJtr[offset + 24] = J[3] * r; 
    JtJJtr[offset + 25] = J[4] * r; 
    JtJJtr[offset + 26] = J[5] * r;
}

void launch_rgb_optimization_kernel(
    const torch::Tensor intensity_prev, 
    const torch::Tensor intensity_curr, 
    const torch::Tensor grad_prev, 
    const torch::Tensor grad_curr, 
    const torch::Tensor mask_prev, 
    const torch::Tensor mask_curr, 
    const torch::Tensor coords, 
    const torch::Tensor J,
    torch::Tensor Hcc, 
    torch::Tensor gc, 
    float bound
) {
    int ht = intensity_prev.size(0);
    int wd = intensity_prev.size(1);

    dim3 blockSize(16, 16);
    dim3 gridSize((wd + blockSize.x - 1) / blockSize.x, (ht + blockSize.y - 1) / blockSize.y);

    torch::Tensor JtJJtr = torch::zeros({ht, wd, 27}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    rgb_optimization_kernel<<<gridSize, blockSize>>>(
        intensity_prev.data_ptr<float>(), 
        intensity_curr.data_ptr<float>(), 
        grad_prev.data_ptr<float>(), 
        grad_curr.data_ptr<float>(), 
        mask_prev.data_ptr<bool>(), 
        mask_curr.data_ptr<bool>(), 
        coords.data_ptr<float>(), 
        J.data_ptr<float>(), 
        JtJJtr.data_ptr<float>(), 
        wd, 
        ht, 
        bound
    );

    cudaDeviceSynchronize();

    torch::Tensor sum = torch::zeros({27}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // reduceSumKernel<<<gridSize, blockSize>>>(
    //     JtJJtr.data_ptr<float>(), 
    //     sum.data_ptr<float>(), 
    //     wd, 
    //     ht, 
    //     27
    // );

    cudaDeviceSynchronize();
}


__global__ void icp_optimization_kernel(
    const float* __restrict__ vmap_prev, 
    const float* __restrict__ nmap_prev,
    const bool* __restrict__ mask_prev,
    const float* __restrict__ vmap_curr, 
    const float* __restrict__ nmap_curr, 
    const bool* __restrict__ mask_curr,
    const float* __restrict__ transform, 
    const float* __restrict__ coords, 
    float* JtJJtr, 
    const int wd, 
    const int ht, 
    const float angleThres, 
    const float distThres, 
    const float bound
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * wd + x;

    if (x >= wd || y >= ht) return;

    // Extract normalized coordinates
    float coord_x = coords[2 * idx + 0];
    float coord_y = coords[2 * idx + 1];

    // Check bounds
    if (coord_x < -bound || coord_x > bound || coord_y < -bound || coord_y > bound) return;

    const float warped_x = (coord_x + 1.0f) * 0.5f * (wd  - 1);
    const float warped_y = (coord_y + 1.0f) * 0.5f * (ht - 1);

    float3 v_prev = make_float3(vmap_prev[idx * 3 + 0], vmap_prev[idx * 3 + 1], vmap_prev[idx * 3 + 2]);
    float3 n_prev = make_float3(nmap_prev[idx * 3 + 0], nmap_prev[idx * 3 + 1], nmap_prev[idx * 3 + 2]);

    float3 v_prev_transformed = make_float3(
        transform[0] * v_prev.x + transform[1] * v_prev.y + transform[2] * v_prev.z,
        transform[4] * v_prev.x + transform[5] * v_prev.y + transform[6] * v_prev.z,
        transform[8] * v_prev.x + transform[9] * v_prev.y + transform[10] * v_prev.z
    );
    v_prev_transformed.x += transform[3];
    v_prev_transformed.y += transform[7];
    v_prev_transformed.z += transform[11];

    float3 n_prev_transformed = make_float3(
        transform[0] * n_prev.x + transform[1] * n_prev.y + transform[2] * n_prev.z,
        transform[4] * n_prev.x + transform[5] * n_prev.y + transform[6] * n_prev.z,
        transform[8] * n_prev.x + transform[9] * n_prev.y + transform[10] * n_prev.z
    );

    if(v_prev_transformed.z <= 0) return;

    // Nearest neighbor sampling for 3D map
    float3 v_curr = nearest_interpolate(vmap_curr, wd, ht, warped_x, warped_y);
    float3 n_curr = nearest_interpolate(nmap_curr, wd, ht, warped_x, warped_y);

    float3 delta_v = v_curr - v_prev_transformed;
    float3 cross_n = cross(n_curr, n_prev_transformed);
 
    if(isnan(cross_n.x) || isnan(cross_n.y) || isnan(cross_n.z)) return;

    float dist = norm(delta_v);
    float sine = norm(cross_n);

    if (dist >= distThres || sine >= angleThres || mask_prev[idx] == 0 || mask_curr[idx] == 0) return;

    float3 Jt = n_curr;
    float3 JR = cross(v_prev_transformed, n_curr);

    float J[6] = {Jt.x, Jt.y, Jt.z, JR.x, JR.y, JR.z};
    float r = delta_v.x * n_curr.x + delta_v.y * n_curr.y + delta_v.z * n_curr.z;

    int offset = idx * 27;
    JtJJtr[offset + 0] = J[0] * J[0]; 
    JtJJtr[offset + 1] = J[0] * J[1]; 
    JtJJtr[offset + 2] = J[0] * J[2];
    JtJJtr[offset + 3] = J[0] * J[3]; 
    JtJJtr[offset + 4] = J[0] * J[4]; 
    JtJJtr[offset + 5] = J[0] * J[5];
    JtJJtr[offset + 6] = J[1] * J[1]; 
    JtJJtr[offset + 7] = J[1] * J[2]; 
    JtJJtr[offset + 8] = J[1] * J[3];
    JtJJtr[offset + 9] = J[1] * J[4]; 
    JtJJtr[offset + 10] = J[1] * J[5]; 
    JtJJtr[offset + 11] = J[2] * J[2];
    JtJJtr[offset + 12] = J[2] * J[3]; 
    JtJJtr[offset + 13] = J[2] * J[4]; 
    JtJJtr[offset + 14] = J[2] * J[5];
    JtJJtr[offset + 15] = J[3] * J[3]; 
    JtJJtr[offset + 16] = J[3] * J[4]; 
    JtJJtr[offset + 17] = J[3] * J[5];
    JtJJtr[offset + 18] = J[4] * J[4]; 
    JtJJtr[offset + 19] = J[4] * J[5]; 
    JtJJtr[offset + 20] = J[5] * J[5];
    JtJJtr[offset + 21] = J[0] * r; 
    JtJJtr[offset + 22] = J[1] * r; 
    JtJJtr[offset + 23] = J[2] * r;
    JtJJtr[offset + 24] = J[3] * r; 
    JtJJtr[offset + 25] = J[4] * r; 
    JtJJtr[offset + 26] = J[5] * r;
}


void launch_icp_optimization_kernel(
    const torch::Tensor vmap_prev, 
    const torch::Tensor nmap_prev,
    const torch::Tensor mask_prev, 
    const torch::Tensor vmap_curr, 
    const torch::Tensor nmap_curr, 
    const torch::Tensor mask_curr, 
    const torch::Tensor transform, 
    const torch::Tensor coords, 
    torch::Tensor Hcc, 
    torch::Tensor gc, 
    float angleThres, 
    float distThres, 
    float bound
) {
    int ht = vmap_prev.size(0);
    int wd = vmap_prev.size(1);

    dim3 blockSize(16, 16);
    dim3 gridSize((wd + blockSize.x - 1) / blockSize.x, (ht + blockSize.y - 1) / blockSize.y);

    cudaEvent_t startEvent, stopEvent;
    float kernelTime1 = 0.0f, kernelTime2 = 0.0f;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    torch::Tensor JtJJtr = torch::zeros({ht, wd, 27}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    icp_optimization_kernel<<<gridSize, blockSize>>>(
        vmap_prev.data_ptr<float>(), 
        nmap_prev.data_ptr<float>(), 
        mask_prev.data_ptr<bool>(), 
        vmap_curr.data_ptr<float>(), 
        nmap_curr.data_ptr<float>(), 
        mask_curr.data_ptr<bool>(), 
        transform.data_ptr<float>(), 
        coords.data_ptr<float>(), 
        JtJJtr.data_ptr<float>(), 
        wd, 
        ht, 
        angleThres, 
        distThres, 
        bound
    );
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&kernelTime1, startEvent, stopEvent);
    std::cout << "First kernel execution time: " << kernelTime1 << " ms" << std::endl;

    cudaEventRecord(startEvent);
    
    torch::Tensor sum = torch::zeros({27}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // reduceSumKernel<<<gridSize, blockSize>>>(
    //     JtJJtr.data_ptr<float>(), 
    //     sum.data_ptr<float>(), 
    //     wd, 
    //     ht, 
    //     27
    // );

    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&kernelTime2, startEvent, stopEvent);
    std::cout << "Second kernel execution time: " << kernelTime2 << " ms" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaDeviceSynchronize();
}

__constant__ float c_gauss_kernel[25];

__global__ void gaussian_downsample_kernel(
    const float* input_image,
    float* output_image,
    const int wd,
    const int ht,
    const int ch,
    const int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int dst_wd = wd / 2;
    int dst_ht = ht / 2;

    if (x >= dst_wd || y >= dst_ht) return;

    int radius = kernel_size / 2;
    float count = 0.0f;
    float sum[3] = {0.0f};

    for (int dy = -radius; dy <= radius; ++dy) 
    {
        for (int dx = -radius; dx <= radius; ++dx) 
        {
            int nx = 2 * x + dx;
            int ny = 2 * y + dy;

            if (nx >= 0 && nx < wd && ny >= 0 && ny < ht) {
                float weight = c_gauss_kernel[(dy + radius) * kernel_size + (dx + radius)];

                for (int c = 0; c < ch; ++c) {
                    float neighbor_value = input_image[(ny * wd + nx) * ch + c];
                    sum[c] += neighbor_value * weight;
                }
                count += weight;
            }
        }
    }

    for (int c = 0; c < ch; ++c) {
        output_image[(y * dst_wd + x) * ch + c] = sum[c] / count;
    }
}


void gaussian_downsample(
    const torch::Tensor input_image,
    torch::Tensor output_image,
    const int wd,
    const int ht,
    const int ch){

    float kernel[25] = {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 
                        24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1};

    cudaMemcpyToSymbol(c_gauss_kernel, kernel, 25 * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((wd / 2 + blockSize.x - 1) / blockSize.x, (ht / 2 + blockSize.y - 1) / blockSize.y);
    
    gaussian_downsample_kernel<<<gridSize, blockSize>>>(
        input_image.data_ptr<float>(), 
        output_image.data_ptr<float>(), 
        wd, ht, ch, 5);
    
    cudaDeviceSynchronize();
}


__global__ void compute_vertex_map_kernel(
    const float* depth_map, 
    float* vertex_map, 
    const int width, 
    const int height, 
    const float fx, 
    const float fy, 
    const float cx, 
    const float cy) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float Z = depth_map[idx];

        float X = (x - cx) * Z / fx;
        float Y = (y - cy) * Z / fy;

        vertex_map[idx * 3 + 0] = X;
        vertex_map[idx * 3 + 1] = Y;
        vertex_map[idx * 3 + 2] = Z;
    }
}


__global__ void compute_normal_map_kernel(
    const float* vertex_map, 
    float* normal_map,
    const int wd, 
    const int ht) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= wd || y >= ht) return;

    int idx = y * wd + x;

    float3 v00, v01, v10;
    v00.x = vertex_map[idx * 3 + 0];
    v00.y = vertex_map[idx * 3 + 1];
    v00.z = vertex_map[idx * 3 + 2];

    if (x + 1 < wd) {
        v10.x = vertex_map[(idx + 1) * 3 + 0];
        v10.y = vertex_map[(idx + 1) * 3 + 1];
        v10.z = vertex_map[(idx + 1) * 3 + 2];
    } else {
        v10 = v00;
    }

    if (y + 1 < ht) {
        v01.x = vertex_map[(idx + wd) * 3 + 0];
        v01.y = vertex_map[(idx + wd) * 3 + 1];
        v01.z = vertex_map[(idx + wd) * 3 + 2];
    } else {
        v01 = v00;
    }

    float3 n = normalize(cross(v01 - v00, v10 - v00));

    if (isnan(n.x) || isnan(n.y) || isnan(n.z)) {
        n = make_float3(0.0f, 0.0f, 0.0f);
    }

    normal_map[idx * 3 + 0] = n.x;
    normal_map[idx * 3 + 1] = n.y;
    normal_map[idx * 3 + 2] = n.z;
}

void compute_vertex_and_normal_map(
    const torch::Tensor depth_image, 
    const float fx, 
    const float fy, 
    const float cx, 
    const float cy, 
    torch::Tensor vertex_map, 
    torch::Tensor normal_map) {
    
    int ht = depth_image.size(0);
    int wd = depth_image.size(1);
    
    float* depth_map_ptr = depth_image.data_ptr<float>();
    float* vertex_map_ptr = vertex_map.data_ptr<float>();
    float* normal_map_ptr = normal_map.data_ptr<float>();

    dim3 blockSize(16, 16);
    dim3 gridSize((wd + blockSize.x - 1) / blockSize.x, (ht + blockSize.y - 1) / blockSize.y);

    compute_vertex_map_kernel<<<gridSize, blockSize>>>(
        depth_map_ptr, vertex_map_ptr, wd, ht, fx, fy, cx, cy);
    
    cudaDeviceSynchronize();

    compute_normal_map_kernel<<<gridSize, blockSize>>>(
        vertex_map_ptr, normal_map_ptr, wd, ht);

    cudaDeviceSynchronize();
}


__global__ void gaussian_filter_kernel(
    const float* input_image,
    float* output_image,
    const int wd,
    const int ht,
    const int channels,
    const int window_size,
    const float sigma_s2_inv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= wd || y >= ht) return;

    float sum1[3] = {0};
    float sum2 = 0.0f;
    int radius = window_size / 2;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < wd && ny >= 0 && ny < ht)
            {
                float space2 = dx * dx + dy * dy;
                float weight = expf(-space2 * sigma_s2_inv);

                for (int c = 0; c < channels; ++c)
                {
                    float neighbor_value = input_image[(ny * wd + nx) * channels + c];
                    sum1[c] += neighbor_value * weight;
                }

                sum2 += weight;
            }
        }
    }

    for (int c = 0; c < channels; ++c)
    {
        int output_idx = (y * wd + x) * channels + c;
        output_image[output_idx] = sum1[c] / sum2;
    }
}

void gaussian_filter(
    const torch::Tensor input_image,
    torch::Tensor output_image,
    const int wd,
    const int ht,
    const int channels,
    const int window_size,
    const float sigma_s){

    const float* input_image_ptr = input_image.data_ptr<float>();
    float* output_image_ptr = output_image.data_ptr<float>();

    dim3 block_size(16, 16);
    dim3 grid_size((wd + block_size.x - 1) / block_size.x,
                   (ht + block_size.y - 1) / block_size.y);

    float sigma_s2_inv = 1.0f / (2.0f * sigma_s * sigma_s);

    gaussian_filter_kernel<<<grid_size, block_size>>>(
        input_image_ptr, output_image_ptr, wd, ht, channels, window_size, sigma_s2_inv);

    cudaDeviceSynchronize();
}

__global__ void bilateral_filter_kernel(
    const float* input_image, 
    float* output_image,
    const int wd, 
    const int ht, 
    const int window_size, 
    const float sigma_c2_inv, 
    const float sigma_s2_inv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= wd || y >= ht) return;

    float center_value = input_image[y * wd + x];

    float sum1 = 0.0f;
    float sum2 = 0.0f;
    int radius = window_size / 2;
    
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < wd && ny >= 0 && ny < ht)
            {
                float neighbor_value = input_image[ny * wd + nx];
                float dc = center_value - neighbor_value;

                float space2 = dx * dx + dy * dy;
                float color2 = dc * dc;

                float weight = expf(-space2 * sigma_s2_inv - color2 * sigma_c2_inv);

                sum1 += neighbor_value * weight;
                sum2 += weight;
            }
        }
    }

    output_image[y * wd + x] = sum1 / sum2;
}


void bilateral_filter(
    const torch::Tensor input_image, 
    torch::Tensor output_image, 
    const int wd, 
    const int ht,
    const int window_size,
    const float sigma_c, 
    const float sigma_s) {

    const float* input_image_ptr = input_image.data_ptr<float>();
    float* output_image_ptr = output_image.data_ptr<float>();

    // Define grid and block size
    dim3 block_size(16, 16);
    dim3 grid_size((wd + block_size.x - 1) / block_size.x, 
                   (ht + block_size.y - 1) / block_size.y);

    float sigma_s2_inv = 1.0f / (2.0f * sigma_s * sigma_s);
    float sigma_c2_inv = 1.0f / (2.0f * sigma_c * sigma_c);

    bilateral_filter_kernel<<<grid_size, block_size>>>(
        input_image_ptr, output_image_ptr, wd, ht, window_size, sigma_c2_inv, sigma_s2_inv);
    
    cudaDeviceSynchronize();
}

__constant__ float c_gsx3x3[9];
__constant__ float c_gsy3x3[9];

__global__ void gradient_kernel(
    const float* input_image,
    float* grad_x,
    float* grad_y,
    const int wd,
    const int ht)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * wd + x;

    if (x >= wd || y >= ht) return;

    int radius = 1;

    float grad_x_value = 0.0f;
    float grad_y_value = 0.0f;

    int kernel_index = 8;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < wd && ny >= 0 && ny < ht)
            {
                float neighbor_value = input_image[ny * wd + nx];

                grad_x_value += neighbor_value * c_gsx3x3[kernel_index];
                grad_y_value += neighbor_value * c_gsy3x3[kernel_index];
            }

            -- kernel_index;
        }
    }

    grad_x[idx] = grad_x_value;
    grad_y[idx] = grad_y_value;
}


void compute_gradients(
    const torch::Tensor input_image,
    torch::Tensor grad_x,
    torch::Tensor grad_y,
    const int wd,
    const int ht)
{
    float gsx3x3[9] = {
        0.52201, 0.00000, -0.52201, 0.79451, -0.00000, -0.79451, 0.52201, 0.00000, -0.52201
    };

    float gsy3x3[9] = {
        0.52201, 0.79451, 0.52201, 0.00000, 0.00000, 0.00000, -0.52201, -0.79451, -0.52201
    };

    cudaMemcpyToSymbol(c_gsx3x3, gsx3x3, 9 * sizeof(float));
    cudaMemcpyToSymbol(c_gsy3x3, gsy3x3, 9 * sizeof(float));

    dim3 block_size(16, 16);
    dim3 grid_size((wd + block_size.x - 1) / block_size.x,
                   (ht + block_size.y - 1) / block_size.y);

    gradient_kernel<<<grid_size, block_size>>>(
        input_image.data_ptr<float>(), 
        grad_x.data_ptr<float>(), 
        grad_y.data_ptr<float>(), 
        wd, 
        ht);

    cudaDeviceSynchronize();
}


void solveBlock(
    const torch::Tensor A,
    const torch::Tensor b,
    const float lm,
    torch::Tensor x
){
    auto start1 = std::chrono::high_resolution_clock::now();

    torch::Tensor A_cpu = A.to(torch::kCPU);
    torch::Tensor b_cpu = b.to(torch::kCPU);

    Eigen::Map<Eigen::MatrixXf> A_eigen(A_cpu.data_ptr<float>(), A_cpu.size(0), A_cpu.size(1));
    Eigen::Map<Eigen::VectorXf> b_eigen(b_cpu.data_ptr<float>(), b_cpu.size(0));

    Eigen::MatrixXf A_regularized = A_eigen + lm * Eigen::MatrixXf::Identity(A_eigen.rows(), A_eigen.cols());
    Eigen::VectorXf x_eigen = A_regularized.colPivHouseholderQr().solve(b_eigen);

    torch::Tensor x_cpu = torch::from_blob(x_eigen.data(), {x_eigen.size()}, torch::dtype(torch::kFloat32));
    
    x = x.view({x_eigen.size()});
    x.copy_(x_cpu.to(x.device()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("projective_transform_cuda", &launch_projective_transform_kernel, "Projective transform kernel");
    m.def("rgb_optimization_cuda", &launch_rgb_optimization_kernel, "RGB optimization kernel");
    m.def("icp_optimization_cuda", &launch_icp_optimization_kernel, "ICP optimization kernel");
    m.def("compute_vertex_and_normal_cuda", &compute_vertex_and_normal_map, "Compute vertex and normal map using CUDA");
    m.def("gaussian_filter_cuda", &gaussian_filter, "Gaussian filter implementation");
    m.def("bilateral_filter_cuda", &bilateral_filter, "Bilateral filter implementation");
    m.def("gaussian_downsample_cuda", &gaussian_downsample, "Gaussian downsample implementation");
    m.def("compute_gradients_cuda", &compute_gradients, "Compute gradients using CUDA");
    m.def("solve_block_cuda", &solveBlock, "Solve block");
}
