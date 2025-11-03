#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h> // for size_t
#include <omp.h>
#include <chrono>
using Clock = std::chrono::high_resolution_clock;

/* 用固定种子生成确定性随机数 */
void generate_matrices(float *A, float *B, size_t NN)
{
    srand(42); // 固定种子，保证每次一致
    for (size_t i = 0; i < NN; ++i)
    {
        A[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        B[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

void matmul_serial_C_TrC(const float *A, const float *B, float *C, int N)
{
    // this is output jump on mat C
    // ci(j) = aik * bk(j)
    long long NN = (long long)N * N;

    // 清零结果矩阵
    for (long long i = 0; i < NN; i++)
        C[i] = 0.0f;

    // 三重循环 i-k-j：cache 友好型访问
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            float aik = A[(long long)i * N + k];
            const float *b_row = &B[(long long)k * N];
            float *c_row = &C[(long long)i * N];
            for (int j = 0; j < N; j++)
            {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}
void matmul_serial_C_TrN(const float *A, const float *B, float *C, int N)
{
    // this is a plain mat mul
    // cij = ai(k) * b(k)j
    long long NN = (long long)N * N;

    // 清零结果矩阵
    for (long long i = 0; i < NN; i++)
        C[i] = 0.0f;

    // 三重循环 i-k-j：cache 友好型访问
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            const float *ai = &A[(long long)i * N];
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += ai[k] * B[k * N + j];
            }
        }
    }
}
void matmul_serial_C_TrB(const float *A, const float *B, float *C, int N)
{
    // this is a trans on B
    // cij = ai(k) * bj(k)
    long long NN = (long long)N * N;

    // 清零结果矩阵
    for (long long i = 0; i < NN; i++)
        C[i] = 0.0f;

    // 三重循环 i-k-j：cache 友好型访问
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            const float *ai = &A[(long long)i * N];
            const float *bj = &A[(long long)j * N];
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += ai[k] * bj[k];
            }
        }
    }
}

void matmul_omp_C_TrC(const float *A, const float *B, float *C, int N)
{
    // this is output jump on mat C
    // ci(j) = aik * bk(j)
    long long NN = (long long)N * N;

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < NN; i++)
        C[i] = 0.0f;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            float aik = A[(long long)i * N + k];
            const float *b_row = &B[(long long)k * N];
            float *c_row = &C[(long long)i * N];
            for (int j = 0; j < N; j++)
            {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}
void matmul_omp_C_TrN(const float *A, const float *B, float *C, int N)
{
    // this is a plain mat mul
    // cij = ai(k) * b(k)j
    long long NN = (long long)N * N;

    // 清零结果矩阵
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < NN; i++)
        C[i] = 0.0f;

    // 三重循环 i-k-j：cache 友好型访问
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            const float *ai = &A[(long long)i * N];
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += ai[k] * B[k * N + j];
            }
        }
    }
}
void matmul_omp_C_TrB(const float *A, const float *B, float *C, int N)
{
    // this is a trans on B
    // cij = ai(k) * bj(k)
    long long NN = (long long)N * N;

    // 清零结果矩阵
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < NN; i++)
        C[i] = 0.0f;

    // 三重循环 i-k-j：cache 友好型访问
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            const float *ai = &A[(long long)i * N];
            const float *bj = &A[(long long)j * N];
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += ai[k] * bj[k];
            }
        }
    }
}
int main()
{
    const int N = 1024; // 矩阵阶数
    const long long NN = 1LL * N * N;
    printf("N = %d\n", N);

    // 生成确定性数据（便于复现）
    float *A = (float *)malloc(sizeof(float) * NN);
    float *B = (float *)malloc(sizeof(float) * NN);
    float *C1 = (float *)malloc(sizeof(float) * NN);
    float *C2 = (float *)malloc(sizeof(float) * NN);
    if (!A || !B || !C1 || !C2)
    {
        printf("内存分配失败\n");
        return 1;
    }

    generate_matrices(A, B, NN);
    freopen("out.txt", "w", stdout);
    {
        printf("\nikj循环顺序测试\n");
        // 串行
        auto t0 = Clock::now();
        matmul_serial_C_TrC(A, B, C1, N);
        auto t1 = Clock::now();
        double t_serial = std::chrono::duration<double>(t1 - t0).count();

        // 并行（可用 OMP_NUM_THREADS 控制线程数）
        auto t2 = Clock::now();
        matmul_omp_C_TrC(A, B, C2, N);
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
    }
    {
        printf("\nijk 循环顺序测试\n");
        // 串行
        auto t0 = Clock::now();
        matmul_serial_C_TrN(A, B, C1, N);
        auto t1 = Clock::now();
        double t_serial = std::chrono::duration<double>(t1 - t0).count();

        // 并行（可用 OMP_NUM_THREADS 控制线程数）
        auto t2 = Clock::now();
        matmul_omp_C_TrN(A, B, C2, N);
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
    }
    {
        printf("\nijk 转置B测试\n");
        // 串行
        auto t0 = Clock::now();
        matmul_serial_C_TrB(A, B, C1, N);
        auto t1 = Clock::now();
        double t_serial = std::chrono::duration<double>(t1 - t0).count();

        // 并行（可用 OMP_NUM_THREADS 控制线程数）
        auto t2 = Clock::now();
        matmul_omp_C_TrB(A, B, C2, N);
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
    }

    free(A);
    free(B);
    free(C1);
    free(C2);
    return 0;
}
