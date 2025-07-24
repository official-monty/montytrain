#ifndef THREADS
#define THREADS 512
#endif

__device__ void warpReduce(volatile float* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

extern "C" __global__ void kernel(
    const int in_size,
    const int batch_size,
    const float* weights,
    const float* biases,
    const float* input,
    const int* moves,
    float* output
) {
    extern __shared__ float sdata[]; 

    const int loc_in_batch = blockIdx.y;
    const int loc_in_moves = blockIdx.x;
    const int tid = threadIdx.x;
    const int locmb = loc_in_batch * 64 + loc_in_moves;
    const int move = moves[locmb];

    const float4* tW = reinterpret_cast<const float4*>(weights + in_size * move);
    const float4* tI = reinterpret_cast<const float4*>(input + in_size * loc_in_batch);

    if (move != -1)
    {
        float local = 0.0F;
        for (int idx = tid; idx < in_size / 4; idx += blockDim.x)
        {
            const float4 tw = tW[idx];
            const float4 ti = tI[idx];
            local += tw.x * ti.x + tw.y * ti.y + tw.z * ti.z + tw.w * ti.w;
        }

        sdata[tid] = local;
        __syncthreads();

        if (THREADS >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
        if (THREADS >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
        if (THREADS >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
        if (THREADS >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }

        for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
        {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid < 32)
        {
            warpReduce(sdata, tid);
        }

        if (tid == 0)
        {
            output[locmb] = sdata[0];
        }
    }
    else if (tid == 0)
    {
        output[locmb] = -10000.0F;
    }
}
