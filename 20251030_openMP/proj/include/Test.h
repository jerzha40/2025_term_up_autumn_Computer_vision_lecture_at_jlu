#ifndef TEST_H
#define TEST_H
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;

namespace Test
{
    static void matmul_serial(const float *A, const float *B, float *C, int N)
    {
        std::fill(C, C + 1LL * N * N, 0.0f);
        // 选择 i-k-j 的循环顺序，A[i,k] 与 B[k,j] 都是连续访问，更友好
        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < N; ++k)
            {
                float aik = A[1LL * i * N + k];
                const float *__restrict b_row = &B[1LL * k * N];
                float *__restrict c_row = &C[1LL * i * N];
                for (int j = 0; j < N; ++j)
                {
                    c_row[j] += aik * b_row[j];
                }
            }
        }
    }

    static void matmul_omp(const float *A, const float *B, float *C, int N)
    {
        std::fill(C, C + 1LL * N * N, 0.0f);
// 外层并行 i；内部仍用 cache 友好的 i-k-j 次序
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < N; ++k)
            {
                float aik = A[1LL * i * N + k];
                const float *__restrict b_row = &B[1LL * k * N];
                float *__restrict c_row = &C[1LL * i * N];
                for (int j = 0; j < N; ++j)
                {
                    c_row[j] += aik * b_row[j];
                }
            }
        }
    }
    int Testmain()
    {
        const int N = 1024; // 矩阵阶数
        const long long NN = 1LL * N * N;
        printf("N = %d\n", N);

        // 生成确定性数据（便于复现）
        std::vector<float> A(NN), B(NN), C1(NN), C2(NN);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (long long i = 0; i < NN; ++i)
        {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }

        // 预热避免首次运行偏慢
        {
            std::vector<float> warm(NN);
            matmul_serial(A.data(), B.data(), warm.data(), N);
        }

        // 串行
        auto t0 = Clock::now();
        matmul_serial(A.data(), B.data(), C1.data(), N);
        auto t1 = Clock::now();
        double t_serial = std::chrono::duration<double>(t1 - t0).count();

        // 并行（可用 OMP_NUM_THREADS 控制线程数）
        auto t2 = Clock::now();
        matmul_omp(A.data(), B.data(), C2.data(), N);
        auto t3 = Clock::now();
        double t_omp = std::chrono::duration<double>(t3 - t2).count();

        // 结果校验（相对误差）
        double max_rel_err = 0.0;
        for (long long i = 0; i < NN; ++i)
        {
            double a = C1[i], b = C2[i];
            double denom = std::max(1e-6, std::abs(a));
            max_rel_err = std::max(max_rel_err, std::abs(a - b) / denom);
        }

        // 统计：乘加算作 2 次浮点运算
        const double flops = 2.0 * N * 1.0 * N * N;
        double gflops_serial = flops / t_serial / 1e9;
        double gflops_omp = flops / t_omp / 1e9;

        int threads = 1;
#pragma omp parallel
        {
#pragma omp master
            threads = omp_get_num_threads();
        }

        printf("Threads: %d\n", threads);
        printf("Serial   : %.3f s,  %.2f GFLOPS\n", t_serial, gflops_serial);
        printf("OpenMP   : %.3f s,  %.2f GFLOPS\n", t_omp, gflops_omp);
        printf("Speedup  : %.2fx\n", t_serial / t_omp);
        printf("Max rel err: %.3e\n", max_rel_err);
        return 0;
    }
} // namespace Test
#endif