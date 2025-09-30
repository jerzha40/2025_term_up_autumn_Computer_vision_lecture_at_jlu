#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <kernels.cuh>

__global__ void vecAddKernel(const float *A, const float *B, float *C, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x)
    {
        C[idx] = A[idx] + B[idx];
    }
}

static inline void ck(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}

void vecadd_gpu(const float *hA, const float *hB, float *hC, size_t n)
{
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    ck(cudaMalloc(&dA, n * sizeof(float)), "cudaMalloc dA");
    ck(cudaMalloc(&dB, n * sizeof(float)), "cudaMalloc dB");
    ck(cudaMalloc(&dC, n * sizeof(float)), "cudaMalloc dC");
    ck(cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    ck(cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    int block = 256;
    int grid = int((n + block - 1) / block);
    int maxGridX = 0;
    cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, 0);
    grid = std::min(grid, maxGridX);

    vecAddKernel<<<grid, block>>>(dA, dB, dC, n);
    ck(cudaGetLastError(), "kernel");
    ck(cudaMemcpy(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void vecadd_gpu_chunked(const float *hA, const float *hB, float *hC, size_t n)
{
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    size_t maxElems = std::max<size_t>(1, (freeB * 8 / 10) / (3 * sizeof(float)));
    float *dA[2]{}, *dB[2]{}, *dC[2]{};
    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);
    cudaMalloc(&dA[0], maxElems * sizeof(float));
    cudaMalloc(&dB[0], maxElems * sizeof(float));
    cudaMalloc(&dC[0], maxElems * sizeof(float));
    cudaMalloc(&dA[1], maxElems * sizeof(float));
    cudaMalloc(&dB[1], maxElems * sizeof(float));
    cudaMalloc(&dC[1], maxElems * sizeof(float));

    size_t chunk = 0;
    for (size_t off = 0; off < n; off += maxElems, ++chunk)
    {
        int p = int(chunk & 1);
        if (chunk >= 2)
            cudaStreamSynchronize(s[p]);
        size_t m = std::min(maxElems, n - off);
        cudaMemcpyAsync(dA[p], hA + off, m * sizeof(float), cudaMemcpyHostToDevice, s[p]);
        cudaMemcpyAsync(dB[p], hB + off, m * sizeof(float), cudaMemcpyHostToDevice, s[p]);

        int block = 512;
        int grid = int((m + block - 1) / block);
        int maxGridX = 0;
        cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, 0);
        grid = std::min(grid, maxGridX);
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
