#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

struct JtJJtrSE3 {
    float aa, ab, ac, ad, ae, af, ag, bb, bc, bd, be, bf, bg, cc, cd, ce, cf, cg, 
          dd, de, df, dg, ee, ef, eg, ff, fg;

    float residual, inliers;

    __device__ inline void add(const JtJJtrSE3& a) {
        aa += a.aa;
        ab += a.ab;
        ac += a.ac;
        ad += a.ad;
        ae += a.ae;
        af += a.af;
        ag += a.ag;

        bb += a.bb;
        bc += a.bc;
        bd += a.bd;
        be += a.be;
        bf += a.bf;
        bg += a.bg;

        cc += a.cc;
        cd += a.cd;
        ce += a.ce;
        cf += a.cf;
        cg += a.cg;

        dd += a.dd;
        de += a.de;
        df += a.df;
        dg += a.dg;

        ee += a.ee;
        ef += a.ef;
        eg += a.eg;

        ff += a.ff;
        fg += a.fg;

        residual += a.residual;
        inliers += a.inliers;
    }
};


__global__ void reduceSum(JtJJtrSE3* in, JtJJtrSE3* out, int N) {
    JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}


__inline__ __device__ JtJJtrSE3 blockReduceSum(JtJJtrSE3 val) 
{
    static __shared__ JtJJtrSE3 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const JtJJtrSE3 zero = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

__inline__ __device__ JtJJtrSE3 warpReduceSum(JtJJtrSE3 val) 
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.aa += __shfl_down_sync(0xFFFFFFFF, val.aa, offset);
        val.ab += __shfl_down_sync(0xFFFFFFFF, val.ab, offset);
        val.ac += __shfl_down_sync(0xFFFFFFFF, val.ac, offset);
        val.ad += __shfl_down_sync(0xFFFFFFFF, val.ad, offset);
        val.ae += __shfl_down_sync(0xFFFFFFFF, val.ae, offset);
        val.af += __shfl_down_sync(0xFFFFFFFF, val.af, offset);
        val.ag += __shfl_down_sync(0xFFFFFFFF, val.ag, offset);

        val.bb += __shfl_down_sync(0xFFFFFFFF, val.bb, offset);
        val.bc += __shfl_down_sync(0xFFFFFFFF, val.bc, offset);
        val.bd += __shfl_down_sync(0xFFFFFFFF, val.bd, offset);
        val.be += __shfl_down_sync(0xFFFFFFFF, val.be, offset);
        val.bf += __shfl_down_sync(0xFFFFFFFF, val.bf, offset);
        val.bg += __shfl_down_sync(0xFFFFFFFF, val.bg, offset);

        val.cc += __shfl_down_sync(0xFFFFFFFF, val.cc, offset);
        val.cd += __shfl_down_sync(0xFFFFFFFF, val.cd, offset);
        val.ce += __shfl_down_sync(0xFFFFFFFF, val.ce, offset);
        val.cf += __shfl_down_sync(0xFFFFFFFF, val.cf, offset);
        val.cg += __shfl_down_sync(0xFFFFFFFF, val.cg, offset);

        val.dd += __shfl_down_sync(0xFFFFFFFF, val.dd, offset);
        val.de += __shfl_down_sync(0xFFFFFFFF, val.de, offset);
        val.df += __shfl_down_sync(0xFFFFFFFF, val.df, offset);
        val.dg += __shfl_down_sync(0xFFFFFFFF, val.dg, offset);

        val.ee += __shfl_down_sync(0xFFFFFFFF, val.ee, offset);
        val.ef += __shfl_down_sync(0xFFFFFFFF, val.ef, offset);
        val.eg += __shfl_down_sync(0xFFFFFFFF, val.eg, offset);

        val.ff += __shfl_down_sync(0xFFFFFFFF, val.ff, offset);
        val.fg += __shfl_down_sync(0xFFFFFFFF, val.fg, offset);

        val.residual += __shfl_down_sync(0xFFFFFFFF, val.residual, offset);
        val.inliers += __shfl_down_sync(0xFFFFFFFF, val.inliers, offset);
    }

  return val;
}
