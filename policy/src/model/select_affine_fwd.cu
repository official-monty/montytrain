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

        for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
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
