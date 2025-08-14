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

    const int loc_in_batch = blockIdx.x;
    const int loc_in_moves = threadIdx.x;
    const int locmb = loc_in_batch * 64 + loc_in_moves;
    const int move = moves[locmb];

    const float4* tW = reinterpret_cast<const float4*>(weights + in_size * move);
    const float4* tI = reinterpret_cast<const float4*>(input + in_size * loc_in_batch);

    if (move != -1)
    {
        float local = biases[move];
        for (int idx = 0; idx < in_size / 4; idx += 1)
        {
            const float4 tw = tW[idx];
            const float4 ti = tI[idx];
            local += tw.x * ti.x + tw.y * ti.y + tw.z * ti.z + tw.w * ti.w;
        }

        output[locmb] = local;
    }
    else
    {
        output[locmb] = -10000.0F;
    }
}
