#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "kernels.cuh"
const size_t N = 10'000'000;
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

    std::cout << "START\n";
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        C_cpu[i] = A[i] + B[i];
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "CPU Time:" << ms << " ms\n";
    return 0;
}
bool nearly_equal(float a, float b, float eps = 1e-5)
{
    return std::abs(a - b) <= eps * (1.0f + std::max(std::abs(a), std::abs(b)));
}
bool array_close(const std::vector<float> &a, const std::vector<float> &b, float eps = 1e-5)
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
