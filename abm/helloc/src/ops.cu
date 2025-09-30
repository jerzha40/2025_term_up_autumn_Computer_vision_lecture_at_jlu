#include <tensor.cuh>
#include <kernels.cuh>
#include <algorithm>

// CUDA kernel
__global__ void addKernel(const float *A, const float *B, float *C, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t idx = i; idx < n; idx += blockDim.x * gridDim.x)
    {
        C[idx] = A[idx] + B[idx];
    }
}

Tensor add(const Tensor &A, const Tensor &B)
{
    if (A.shape != B.shape)
        throw std::runtime_error("shape mismatch");
    Tensor C(A.shape, A.device);
    const size_t n = A.numel();

    if (A.device == Device::CPU)
    {
        for (size_t i = 0; i < n; ++i)
            C.data[i] = A.data[i] + B.data[i];
    }
    else
    {
        const int block = 256;
        int grid = static_cast<int>((n + block - 1) / block);
        int maxGridX = 0;
        cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, 0);
        grid = std::min(grid, maxGridX);
        addKernel<<<grid, block>>>(A.data, B.data, C.data, n);
        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
        cudaDeviceSynchronize();
    }
    return C; // move
}
