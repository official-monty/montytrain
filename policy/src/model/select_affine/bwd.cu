extern "C" __global__ void kernel(
    const int in_size,
    const int batch_size,
    const float* weights,
    const float* input,
    const int* moves,
    const float* output_grad,
    float* input_grad,
    float* weights_grad,
    float* biases_grad
) {
    extern __shared__ float sdata[];

    const int loc_in_batch = blockIdx.x;
    const int loc_in_moves = threadIdx.x;
    const int locmb = loc_in_batch * 64 + loc_in_moves;
    const int move = moves[locmb];
    
    if (move != -1)
    {
        const float grd = output_grad[locmb];

        const float4* tW = reinterpret_cast<const float4*>(weights + in_size * move);
        const float4* tI = reinterpret_cast<const float4*>(input + in_size * loc_in_batch);

        atomicAdd(biases_grad + move, grd);

        for (int idx = 0; idx < in_size / 4; idx += 1)
        {
            const float4 ti = tI[idx];
            const float4 tw = tW[idx];

            float* tWg = weights_grad + in_size * move + 4 * idx;
            atomicAdd(tWg    , grd * ti.x);
            atomicAdd(tWg + 1, grd * ti.y);
            atomicAdd(tWg + 2, grd * ti.z);
            atomicAdd(tWg + 3, grd * ti.w);

            float* tIg = input_grad + in_size * loc_in_batch + 4 * idx;
            atomicAdd(tIg    , grd * tw.x);
            atomicAdd(tIg + 1, grd * tw.y);
            atomicAdd(tIg + 2, grd * tw.z);
            atomicAdd(tIg + 3, grd * tw.w);
        }
    }
}