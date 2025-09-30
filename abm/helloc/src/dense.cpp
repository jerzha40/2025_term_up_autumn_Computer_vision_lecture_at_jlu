#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <dense.cuh>

namespace py = pybind11;

py::array_t<float> dense_forward_py(py::array_t<float> input,
                                    py::array_t<float> weights,
                                    py::array_t<float> bias)
{
    auto buf_in = input.request();
    auto buf_w = weights.request();
    auto buf_b = bias.request();

    size_t batch = buf_in.shape[0];
    size_t in_features = buf_in.shape[1];
    size_t out_features = buf_w.shape[1];

    auto output = py::array_t<float>({batch, out_features});
    auto buf_out = output.request();

    dense_forward((float *)buf_in.ptr, (float *)buf_w.ptr, (float *)buf_b.ptr,
                  (float *)buf_out.ptr, batch, in_features, out_features);

    return output;
}

PYBIND11_MODULE(_fastnn, m)
{
    m.def("dense_forward", &dense_forward_py, "CUDA Dense Forward");
    // 还可以继续加 backward
}
