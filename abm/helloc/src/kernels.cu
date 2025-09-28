#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstddef>

__global__ void vecAddKernel(const float *A, const float *B, float *C, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x)
    {
        C[idx] = A[idx] + B[idx];
    }
}
void vector_add_gpu(const float *hA, const float *hB, float *hC, size_t n)
{
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMalloc(&dC, n * sizeof(float));
    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    if (grid > 65535)
        grid = 65535;
    vecAddKernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}