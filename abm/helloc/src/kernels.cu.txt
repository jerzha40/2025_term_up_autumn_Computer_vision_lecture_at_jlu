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
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    printf("VRAM free %.1f GB / total %.1f GB\n", freeB / 1e9, totalB / 1e9);

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
void vecadd_alloc(float *&dA, float *&dB, float *&dC, size_t n)
{
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMalloc(&dC, n * sizeof(float));
}
void vecadd_run(const float *dA, const float *dB, float *dC, size_t n)
{
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    int maxGridX = 0;
    cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, 0);
    grid = (grid > maxGridX ? maxGridX : grid);
    vecAddKernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();
}
void vecadd_free(float *dA, float *dB, float *dC)
{
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
void vector_add_gpu_chunked(const float *hA, const float *hB, float *hC, size_t n)
{
    // 1) 选块大小：用可用显存的 80% 估算（3 个数组：A,B,C）
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    size_t maxElems = (freeB * 8 / 10) / (3 * sizeof(float));
    if (maxElems == 0)
        maxElems = 1;

    // 2) 准备一次性缓冲 & 流
    // 顶部准备：两套缓冲 + 两个流
    float *dA[2] = {}, *dB[2] = {}, *dC[2] = {};
    cudaStream_t s[2];
    cudaMalloc(&dA[0], maxElems * sizeof(float));
    cudaMalloc(&dB[0], maxElems * sizeof(float));
    cudaMalloc(&dC[0], maxElems * sizeof(float));
    cudaMalloc(&dA[1], maxElems * sizeof(float));
    cudaMalloc(&dB[1], maxElems * sizeof(float));
    cudaMalloc(&dC[1], maxElems * sizeof(float));
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    size_t chunk = 0;
    for (size_t off = 0; off < n; off += maxElems, ++chunk)
    {
        int p = int(chunk & 1);
        if (chunk >= 2)
            cudaStreamSynchronize(s[p]); // 重用这套缓冲前先等它上一次的D2H完

        size_t m = std::min(maxElems, n - off);
        cudaMemcpyAsync(dA[p], hA + off, m * sizeof(float), cudaMemcpyHostToDevice, s[p]);
        cudaMemcpyAsync(dB[p], hB + off, m * sizeof(float), cudaMemcpyHostToDevice, s[p]);

        int block = 512;
        int grid = int((m + block - 1) / block);
        int maxGridX = 0;
        cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, 0);
        if (grid > maxGridX)
            grid = maxGridX;

        vecAddKernel<<<grid, block, 0, s[p]>>>(dA[p], dB[p], dC[p], m);
        cudaMemcpyAsync(hC + off, dC[p], m * sizeof(float), cudaMemcpyDeviceToHost, s[p]);
    }
    cudaStreamSynchronize(s[0]);
    cudaStreamSynchronize(s[1]);
    cudaStreamDestroy(s[0]);
    cudaStreamDestroy(s[1]);
    cudaFree(dA[0]);
    cudaFree(dB[0]);
    cudaFree(dC[0]);
    cudaFree(dA[1]);
    cudaFree(dB[1]);
    cudaFree(dC[1]);
}
