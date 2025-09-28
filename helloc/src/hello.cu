#include <cuda_runtime.h>

__global__ void nop_kernel() {}

extern "C" void launch_nop()
{
    nop_kernel<<<1, 1, 0, 0>>>();
    cudaDeviceSynchronize(); // Helloworld 里直接同步，简单直观
}
