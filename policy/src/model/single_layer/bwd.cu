__device__ float op(const float in)
{
    return in > 0.0F && in < 1.0F ? 2.0F * sqrtf(in) : 0.0F;
}

extern "C" __global__ void kernel(
    const int in_size,
    const int nnz,
    const int* input,
    const int* moves,
    const int* stms,
    const float* l1w,
    const float* hl_output,
    const float* out_grad,
    float* l0wg,
    float* l0bg,
    float* l1wg,
    float* l1bg
) {
    extern __shared__ float sdata[];

    const int loc = blockIdx.x;
    const int tid = threadIdx.x;

    const int move_idx = moves[loc];

    if (move_idx == -1)
    {
        return;
    }

        const int move = 2 * move_idx + stms[loc];

    int* sI = reinterpret_cast<int*>(sdata + 4 * blockDim.x);

    if (threadIdx.x < nnz)
    {
        for (int i = threadIdx.x; i < nnz; i += blockDim.x)
        {
            sI[i] = input[nnz * loc + i];
        }
    }

    __syncthreads();

    const float grd = out_grad[loc];

    if (tid == 0) atomicAdd(l1bg + move, grd);

    for (int row = tid; row < in_size / 4; row += blockDim.x)
    {
        const float4 this_hl_out = reinterpret_cast<const float4*>(hl_output + loc * in_size)[row];

        sdata[4 * tid    ] = this_hl_out.x;
        sdata[4 * tid + 1] = this_hl_out.y;
        sdata[4 * tid + 2] = this_hl_out.z;
        sdata[4 * tid + 3] = this_hl_out.w;
        __syncthreads();

        float* tWg = l1wg + in_size * move + row;
        atomicAdd(tWg                 , grd * sdata[tid                 ]);
        atomicAdd(tWg + blockDim.x    , grd * sdata[tid + blockDim.x    ]);
        atomicAdd(tWg + blockDim.x * 2, grd * sdata[tid + blockDim.x * 2]);
        atomicAdd(tWg + blockDim.x * 3, grd * sdata[tid + blockDim.x * 3]);
        __syncthreads();

        float4 this_grd = reinterpret_cast<const float4*>(l1w + in_size * move)[row];
        this_grd.x *= grd * op(this_hl_out.x);
        this_grd.y *= grd * op(this_hl_out.y);
        this_grd.z *= grd * op(this_hl_out.z);
        this_grd.w *= grd * op(this_hl_out.w);

        sdata[4 * tid    ] = this_grd.x;
        sdata[4 * tid + 1] = this_grd.y;
        sdata[4 * tid + 2] = this_grd.z;
        sdata[4 * tid + 3] = this_grd.w;
        __syncthreads();

        for (int i = 0; i < nnz; i++)
        {
            const int j = sI[i];

            if (j == -1)
            {
                break;
            }

            float* tIg = l0wg + j * in_size + row;
            atomicAdd(tIg                 , sdata[tid                 ]);
            atomicAdd(tIg + blockDim.x    , sdata[tid + blockDim.x    ]);
            atomicAdd(tIg + blockDim.x * 2, sdata[tid + blockDim.x * 2]);
            atomicAdd(tIg + blockDim.x * 3, sdata[tid + blockDim.x * 3]);
        }

        float* tBg = l0bg + row;
        atomicAdd(tBg                 , sdata[tid                 ]);
        atomicAdd(tBg + blockDim.x    , sdata[tid + blockDim.x    ]);
        atomicAdd(tBg + blockDim.x * 2, sdata[tid + blockDim.x * 2]);
        atomicAdd(tBg + blockDim.x * 3, sdata[tid + blockDim.x * 3]);

        __syncthreads();
    }
}