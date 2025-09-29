#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <kernels.cuh>
#include <cuda_runtime.h>
const size_t N = 1000'000'000;
bool nearly_equal(float a, float b, float eps = 1e-5);
bool array_close(const std::vector<float> &a, const std::vector<float> &b, float eps = 1e-5);
int main()
{
    std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, +1.0f);
    for (size_t i = 0; i < N; i++)
    {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    std::chrono::steady_clock::time_point t0, t1, g0, g1;
    long long ms, gms;
    std::cout << "START Cpu\n";
    t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        C_cpu[i] = A[i] + B[i];
    }
    t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "CPU Time:" << ms << " ms\n";
    t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        C_cpu[i] = A[i] + B[i];
    }
    t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "CPU Time:" << ms << " ms\n";

    cuda_warmup();
    std::cout << "START Gpu\n";
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    vecadd_alloc(dA, dB, dC, N);
    g0 = std::chrono::steady_clock::now();
    cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    vecadd_run(dA, dB, dC, N);
    cudaMemcpy(C_gpu.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost);
    g1 = std::chrono::steady_clock::now();
    vecadd_free(dA, dB, dC);
    gms = std::chrono::duration_cast<std::chrono::milliseconds>(g1 - g0).count();
    std::cout << "GPU total time (HtoD + kernel + DtoH): " << gms << " ms\n";
    vecadd_alloc(dA, dB, dC, N);
    g0 = std::chrono::steady_clock::now();
    cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    vecadd_run(dA, dB, dC, N);
    cudaMemcpy(C_gpu.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost);
    g1 = std::chrono::steady_clock::now();
    vecadd_free(dA, dB, dC);
    gms = std::chrono::duration_cast<std::chrono::milliseconds>(g1 - g0).count();
    std::cout << "GPU total time (HtoD + kernel + DtoH): " << gms << " ms\n";

    std::cout << (array_close(C_cpu, C_gpu) ? "OK\n" : "MISMATCH\n");
    return 0;
}
bool nearly_equal(float a, float b, float eps)
{
    return std::abs(a - b) <= eps * (1.0f + std::max(std::abs(a), std::abs(b)));
}
bool array_close(const std::vector<float> &a, const std::vector<float> &b, float eps)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++)
    {
        if (!nearly_equal(a[i], b[i], eps))
        {
            return false;
        }
    }
    return true;
}
