#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernels.cuh"

namespace py = pybind11;

py::array_t<float> vecadd(py::array_t<float, py::array::c_style | py::array::forcecast> A,
                          py::array_t<float, py::array::c_style | py::array::forcecast> B,
                          bool chunked = false)
{
    if (A.ndim() != 1 || B.ndim() != 1)
        throw std::runtime_error("A,B must be 1D float arrays");
    if (A.shape(0) != B.shape(0))
        throw std::runtime_error("length mismatch");
    size_t n = (size_t)A.shape(0);

    auto C = py::array_t<float>(n);
    auto rA = A.unchecked<1>();
    auto rB = B.unchecked<1>();
    auto rC = C.mutable_unchecked<1>();

    const float *pA = rA.data(0);
    const float *pB = rB.data(0);
    float *pC = rC.mutable_data(0);

    if (chunked)
        vecadd_gpu_chunked(pA, pB, pC, n);
    else
        vecadd_gpu(pA, pB, pC, n);
    return C;
}

PYBIND11_MODULE(_fastoper, m)
{
    m.doc() = "CUDA vector add demo";
    m.def("vecadd", &vecadd, py::arg("A"), py::arg("B"), py::arg("chunked") = false,
          "C = A + B on GPU (optionally chunked)");
}
