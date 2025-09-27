#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

__global__ void sumsq_partials(const double* __restrict__ x, size_t n, double* __restrict__ partials) {
    extern __shared__ double smem[];
    const unsigned tid = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if (i < n) { double t = x[i]; v = t * t; }
    smem[tid] = v;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = smem[0];
}

double sumsq_cuda(py::array_t<double, py::array::c_style | py::array::forcecast> x) {
    auto info = x.request();
    size_t n = static_cast<size_t>(info.size);
    if (n == 0) return 0.0;

    const double* h = static_cast<const double*>(info.ptr);

    const int block = 256;
    const int grid  = static_cast<int>((n + block - 1) / block);

    double *d_x = nullptr, *d_partials = nullptr;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_partials, grid * sizeof(double));

    cudaMemcpy(d_x, h, n * sizeof(double), cudaMemcpyHostToDevice);
    sumsq_partials<<<grid, block, block * sizeof(double)>>>(d_x, n, d_partials);

    std::vector<double> partials(grid);
    cudaMemcpy(partials.data(), d_partials, grid * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_partials);
    cudaFree(d_x);

    double acc = 0.0;
    for (double v : partials) acc += v;
    return acc;
}

PYBIND11_MODULE(fastcuda, m) {
    m.doc() = "CUDA-accelerated sumsq";
    m.def("sumsq", &sumsq_cuda, "Sum of squares on GPU");
}
