extern "C" __global__ void kernel(
    const int single_size,
    const int batch_size,
    const float* output_grad,
    const int* moves,
    float* input_grad
) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= 64 * batch_size) return;

    const int move = moves[tid];
    if (move != -1) atomicAdd(input_grad + single_size * (tid / 64) + move, output_grad[tid]);
}
