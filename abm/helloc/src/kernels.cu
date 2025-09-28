#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <stdio.h>

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
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaHostRegister((void *)hA, n * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister((void *)hB, n * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister((void *)hC, n * sizeof(float), cudaHostRegisterDefault);
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMalloc(&dC, n * sizeof(float));
    cudaEventRecord(e0);
    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(e1);
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    if (grid > 65535)
        grid = 65535;
    vecAddKernel<<<grid, block>>>(dA, dB, dC, n);
    cudaEventRecord(e2);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);
    float tH2D = 0, tK = 0, tD2H = 0;
    cudaEventElapsedTime(&tH2D, e0, e1);
    cudaEventElapsedTime(&tK, e1, e2);
    cudaEventElapsedTime(&tD2H, e2, e3);

    printf("breakdown  H2D: %.3f ms, kernel: %.3f ms, D2H: %.3f ms\n", tH2D, tK, tD2H);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaHostUnregister((void *)hA);
    cudaHostUnregister((void *)hB);
    cudaHostUnregister((void *)hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
void cuda_warmup()
{
    cudaFree(0);
}
