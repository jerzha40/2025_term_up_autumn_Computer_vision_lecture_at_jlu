// fastgpu.cu  (UTF-8)
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

template<typename T>
__device__ inline T my_sqr(T x){ return x*x; }

template<typename T>
__global__ void sumsq_accum_kernel(const T* __restrict__ x, size_t n, T* __restrict__ out){
    extern __shared__ unsigned char smem_raw[];
    T* sm = reinterpret_cast<T*>(smem_raw);

    // 每线程做 grid-stride 累加，减少全局内存访问往返
    T v = T(0);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x){
        v += my_sqr(x[i]);
    }
    sm[threadIdx.x] = v;
    __syncthreads();

    // 经典块内归约
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1){
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }

    // 每块一次 atomicAdd 到全局 out
    if (threadIdx.x == 0){
        atomicAdd(out, sm[0]);
    }
}

// double 的 atomicAdd 在 CC >= 6.0 才原生支持；老卡这里给个回退（可选）。
// 仅当在设备端且 __CUDA_ARCH__ < 600 时，提供 double 的 atomicAdd 兜底实现
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val){
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
extern "C" {

__declspec(dllexport)
float sumsq_cuda32_chunked(const float* x, size_t n, size_t chunk_elems){
    if (!x || n==0) return 0.f;
    if (chunk_elems == 0) chunk_elems = 64*1024*1024 / sizeof(float); // 默认 ~64MB

    const int block = 256;
    float host_acc = 0.f;

    float *d_x = nullptr, *d_out = nullptr;
    // 预分配一次，循环复用
    cudaError_t st = cudaMalloc(&d_x, chunk_elems * sizeof(float));
    if (st != cudaSuccess) return -1.f;
    st = cudaMalloc(&d_out, sizeof(float));
    if (st != cudaSuccess){ cudaFree(d_x); return -2.f; }

    size_t off = 0;
    while (off < n){
        size_t m = (n - off > chunk_elems) ? chunk_elems : (n - off);
        // 清零设备端累加器
        cudaMemset(d_out, 0, sizeof(float));
        // H2D（可选：cudaHostRegister 再加速，这里保持简单稳妥）
        st = cudaMemcpy(d_x, x + off, m * sizeof(float), cudaMemcpyHostToDevice);
        if (st != cudaSuccess){ cudaFree(d_out); cudaFree(d_x); return -3.f; }

        int grid = int((m + block - 1) / block);
        grid = (grid < 1) ? 1 : grid;
        size_t shmem = size_t(block) * sizeof(float);
        sumsq_accum_kernel<float><<<grid, block, shmem>>>(d_x, m, d_out);
        st = cudaDeviceSynchronize();
        if (st != cudaSuccess){ cudaFree(d_out); cudaFree(d_x); return -4.f; }

        float chunk_sum = 0.f;
        cudaMemcpy(&chunk_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        host_acc += chunk_sum;
        off += m;
    }

    cudaFree(d_out);
    cudaFree(d_x);
    return host_acc;
}

__declspec(dllexport)
double sumsq_cuda64_chunked(const double* x, size_t n, size_t chunk_elems){
    if (!x || n==0) return 0.0;
    if (chunk_elems == 0) chunk_elems = 32*1024*1024 / sizeof(double); // 默认 ~32MB

    const int block = 256;
    double host_acc = 0.0;

    double *d_x = nullptr, *d_out = nullptr;
    cudaError_t st = cudaMalloc(&d_x, chunk_elems * sizeof(double));
    if (st != cudaSuccess) return -1.0;
    st = cudaMalloc(&d_out, sizeof(double));
    if (st != cudaSuccess){ cudaFree(d_x); return -2.0; }

    size_t off = 0;
    while (off < n){
        size_t m = (n - off > chunk_elems) ? chunk_elems : (n - off);
        cudaMemset(d_out, 0, sizeof(double));
        st = cudaMemcpy(d_x, x + off, m * sizeof(double), cudaMemcpyHostToDevice);
        if (st != cudaSuccess){ cudaFree(d_out); cudaFree(d_x); return -3.0; }

        int grid = int((m + block - 1) / block);
        grid = (grid < 1) ? 1 : grid;
        size_t shmem = size_t(block) * sizeof(double);
        sumsq_accum_kernel<double><<<grid, block, shmem>>>(d_x, m, d_out);
        st = cudaDeviceSynchronize();
        if (st != cudaSuccess){ cudaFree(d_out); cudaFree(d_x); return -4.0; }

        double chunk_sum = 0.0;
        cudaMemcpy(&chunk_sum, d_out, sizeof(double), cudaMemcpyDeviceToHost);
        host_acc += chunk_sum;
        off += m;
    }

    cudaFree(d_out);
    cudaFree(d_x);
    return host_acc;
}

} // extern "C"
