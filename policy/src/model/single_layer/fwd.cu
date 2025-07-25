#ifndef THREADS
#define THREADS 512
#endif

__device__ __forceinline__ void warpReduce(volatile float* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__ __forceinline__ float op(const float x)
{
    const float clamped = max(min(x, 1.0F), 0.0F);
    return clamped * clamped;
}

extern "C" __global__ void kernel(
    const int in_size,
    const int nnz,
    const float* l0w,
    const float* l0b,
    const float* l1w,
    const float* l1b,
    const int* input,
    float* hl_output,
    float* output
) {
    extern __shared__ float sdata[];

    const int loc = blockIdx.x;
    const int tid = threadIdx.x;

    if (input[nnz * loc] == -1)
    {
        if (tid == 0)
        {
            output[loc] = -10000.0F;
        }

        return;
    }

    int* sI = reinterpret_cast<int*>(sdata + THREADS);

    if (threadIdx.x < nnz)
    {
        for (int i = threadIdx.x; i < nnz; i += THREADS)
        {
            sI[i] = input[nnz * loc + i];
        }
    }

    __syncthreads();

    float local = 0.0F;

    for (int row = tid; row < in_size / 4; row += THREADS)
    {
        float4 sum = reinterpret_cast<const float4*>(l0b)[row];

        for (int i = 0; i < nnz; i++)
        {
            const int j = sI[i];

            if (j == -1)
            {
                break;
            }

            const float4 a = reinterpret_cast<const float4*>(l0w + j * in_size)[row];

            sum.x += a.x;
            sum.y += a.y;
            sum.z += a.z;
            sum.w += a.w;
        }

        sum.x = op(sum.x);
        sum.y = op(sum.y);
        sum.z = op(sum.z);
        sum.w = op(sum.w);

        reinterpret_cast<float4*>(hl_output + loc * in_size)[row] = sum;

        const float4 tw = reinterpret_cast<const float4*>(l1w)[row];
        local += tw.x * sum.x + tw.y * sum.y + tw.z * sum.z + tw.w * sum.w;
    }

    sdata[tid] = local;
    __syncthreads();

    if (THREADS >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (THREADS >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (THREADS >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (THREADS >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }

    if (tid < 32)
    {
        warpReduce(sdata, tid);
    }

    if (tid == 0)
    {
        output[loc] = sdata[0] + l1b[0];
    }
}
