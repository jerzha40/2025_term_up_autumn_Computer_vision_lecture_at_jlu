// fastgpu.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

__global__ void sumsq_partials64(const double* __restrict__ x,
                                 size_t n,
                                 double* __restrict__ partials) {
    extern __shared__ double sm[];
    unsigned tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if (i < n) { double t = x[i]; v = t * t; }
    sm[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sm[0];
}

// 导出一个纯 C 符号，便于 ctypes 绑定
extern "C" __declspec(dllexport)
double sumsq_cuda64(const double* x, size_t n) {
    if (!x || n == 0) return 0.0;

    const int block = 256;
    const int grid  = int((n + block - 1) / block);

    double *d_x = nullptr, *d_partials = nullptr;
    cudaError_t st;

    st = cudaMalloc(&d_x, n * sizeof(double));
    if (st != cudaSuccess) return -1.0;

    st = cudaMalloc(&d_partials, grid * sizeof(double));
    if (st != cudaSuccess) { cudaFree(d_x); return -2.0; }

    st = cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    if (st != cudaSuccess) { cudaFree(d_partials); cudaFree(d_x); return -3.0; }

    sumsq_partials64<<<grid, block, block * sizeof(double)>>>(d_x, n, d_partials);
    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) { cudaFree(d_partials); cudaFree(d_x); return -4.0; }

    std::vector<double> h_partials(grid);
    st = cudaMemcpy(h_partials.data(), d_partials, grid * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_partials);
    cudaFree(d_x);

    if (st != cudaSuccess) return -5.0;

    double acc = 0.0;
    for (double v : h_partials) acc += v;
    return acc;
}
