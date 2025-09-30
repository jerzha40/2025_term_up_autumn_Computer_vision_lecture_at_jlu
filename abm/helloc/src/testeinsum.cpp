#include <iostream>
#include <cmath>
#include <perco/core/tensor.h>
namespace perco
{
    Tensor einsum(const std::string &, const Tensor &, const Tensor &);
}

static void fill_seq(perco::Tensor &T)
{
    float *p = T.data();
    for (uint64_t i = 0; i < T.numel(); ++i)
        p[i] = static_cast<float>(i);
}

static perco::Tensor cpu_mm(const perco::Tensor &A, const perco::Tensor &B)
{
    const auto &a = A.shape();
    const auto &b = B.shape();
    if (a.size() != 2 || b.size() != 2 || a[1] != b[0])
        throw std::runtime_error("shape mismatch");
    perco::Tensor C({a[0], b[1]});
    for (uint64_t i = 0; i < a[0]; ++i)
        for (uint64_t k = 0; k < b[1]; ++k)
        {
            float acc = 0.f;
            for (uint64_t j = 0; j < a[1]; ++j)
                acc += A.at({i, j}) * B.at({j, k});
            C.at({i, k}) = acc;
        }
    return C;
}

int main()
{
    std::cout << "sldkfjsldkf" << "\n";
    using perco::Tensor;
    Tensor A({2, 3}), B({3, 4});
    fill_seq(A); // A = [0..5] reshape 2x3
    fill_seq(B); // B = [0..11] reshape 3x4

    auto C_gpu = perco::einsum("ij,jk->ik", A, B);
    auto C_cpu = cpu_mm(A, B);

    // 检查 & 打印
    double max_abs_err = 0.0;
    for (uint64_t i = 0; i < C_cpu.numel(); ++i)
        max_abs_err = std::max(float(max_abs_err), std::abs(C_cpu.data()[i] - C_gpu.data()[i]));

    std::cout << "max_abs_err = " << max_abs_err << "\n";
    std::cout << "C_cpu: ";
    for (uint64_t i = 0; i < C_cpu.numel(); ++i)
        std::cout << C_cpu.data()[i] << (i + 1 == C_cpu.numel() ? '\n' : ' ');
    std::cout << "C_gpu: ";
    for (uint64_t i = 0; i < C_gpu.numel(); ++i)
        std::cout << C_gpu.data()[i] << (i + 1 == C_gpu.numel() ? '\n' : ' ');

    if (max_abs_err <= 1e-6)
    {
        std::cout << "OK\n";
        return 0;
    }
    std::cout << "MISMATCH\n";
    return 1;
}
