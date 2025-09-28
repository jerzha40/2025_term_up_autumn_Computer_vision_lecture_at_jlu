#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void hello_kernel()
{
    printf("Hello from GPU kernel!\n");
}

extern "C" void launch_hello_from_gpu()
{
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
