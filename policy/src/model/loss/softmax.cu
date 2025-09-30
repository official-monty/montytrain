#ifndef STUFF
#define SIZE 64
#endif

constexpr int size = SIZE;

extern "C" __global__ void kernel(const int k, const float* input, float* output)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= k)
        return;

    const float* thisColumn = input + size * tid;
    float* thisOutput = output + size * tid;

    float maximum = thisColumn[0];

    for (int i = 1; i < size; i++) {
        maximum = max(maximum, thisColumn[i]);
    }

    float total = 0.0F;

    for (int i = 0; i < size; i++) {
        const float exp = expf(thisColumn[i] - maximum);
        thisOutput[i] = exp;
        total += exp;
    }

    for (int i = 0; i < size; i++) {
        thisOutput[i] /= total;
    }
}