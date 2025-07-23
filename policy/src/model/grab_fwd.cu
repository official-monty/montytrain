extern "C" __global__ void kernel(
    const int single_size,
    const int batch_size,
    const float* input,
    const int* moves,
    float* output
) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= 64 * batch_size) return;

    const int move = moves[tid];
    output[tid] = move != -1 ? input[single_size * (tid / 64) + move] : -10000.0F;
}
