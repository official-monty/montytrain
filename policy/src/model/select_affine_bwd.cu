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
    const int loc_in_batch = blockIdx.y;
    const int loc_in_moves = blockIdx.x;
    const int tid = threadIdx.x;
    const int locmb = loc_in_batch * 64 + loc_in_moves;
    const int move = moves[locmb];
    
    if (move != -1)
    {
        const float grd = output_grad[locmb];

        const float* tW = weights + in_size * move;
        const float* tI = input + in_size * loc_in_batch;
        float* tWg = weights_grad + in_size * move;
        float* tIg = input_grad + in_size * loc_in_batch;

        if (tid == 0) atomicAdd(biases_grad + move, grd);

        for (int idx = tid; idx < in_size; idx += blockDim.x)
        {
            atomicAdd(tWg + idx, grd * tI[idx]);
            atomicAdd(tIg + idx, grd * tW[idx]);
        }
    }
}