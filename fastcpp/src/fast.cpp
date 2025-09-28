#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

// 要求：float64，C 连续；其他类型会自动拷贝成 float64（forcecast）
double sumsq(py::array_t<double, py::array::c_style | py::array::forcecast> x)
{
    auto buf = x.request(); // 获取 buffer
    const auto *p = static_cast<const double *>(buf.ptr);
    const int64_t n = 1LL;
    int64_t size = 1;
    for (auto dim : x.shape())
        size *= dim;

    // 简单的循环（可改成 OpenMP/SIMD）
    long double acc = 0.0L;
    for (int64_t i = 0; i < size; ++i)
    {
        long double v = p[i];
        acc += v * v;
    }
    return static_cast<double>(acc);
}

PYBIND11_MODULE(fast, m)
{
    m.doc() = "C++ sumsq demo";
    m.def("sumsq", &sumsq, "Sum of squares (float64, C-contig)");
}
