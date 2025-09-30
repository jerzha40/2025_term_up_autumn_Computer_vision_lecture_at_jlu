#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tensor.cuh>
// #include <stddef.h>
// #include <stdint.h>
#include <vector>
// #include <common.h>
namespace py = pybind11;

// --- numpy 互转 ---
// from numpy -> Tensor
static Tensor tensor_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                                const std::string &device)
{
    std::vector<int64_t> shape(arr.ndim());
    for (size_t i = 0; i < arr.ndim(); ++i)
        shape[i] = static_cast<int64_t>(arr.shape(i));
    Device dev = (device == "cuda" || device == "CUDA") ? Device::CUDA : Device::CPU;

    Tensor t(shape, dev);
    size_t n = t.numel();
    const float *src = static_cast<const float *>(arr.data());
    if (dev == Device::CPU)
    {
        std::memcpy(t.data, src, n * sizeof(float));
    }
    else
    {
        copy_h2d(t.data, src, n);
    }
    return t; // move
}

// to numpy (always returns CPU array)
static py::array_t<float> tensor_to_numpy(const Tensor &t)
{
    std::vector<size_t> shp(t.shape.begin(), t.shape.end());
    py::array_t<float> out(shp);
    float *dst = static_cast<float *>(out.request().ptr);
    if (t.device == Device::CPU)
    {
        std::memcpy(dst, t.data, t.numel() * sizeof(float));
    }
    else
    {
        copy_d2h(dst, t.data, t.numel());
    }
    return out;
}

PYBIND11_MODULE(_fasttorch, m)
{
    m.doc() = "minimal C++/CUDA tensor backend";

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        // 允许两种构造：([shape], Device) 或 ([shape], "cpu"/"cuda")
        .def(py::init<const std::vector<int64_t> &, Device>(),
             py::arg("shape"), py::arg("device") = Device::CPU)
        .def(py::init([](const std::vector<int64_t> &shape, const std::string &device)
                      {
            Device dev = (device == "cuda" || device == "CUDA") ? Device::CUDA : Device::CPU;
            return Tensor(shape, dev); }),
             py::arg("shape"), py::arg("device") = "cpu")
        .def_property_readonly("shape", [](const Tensor &t)
                               { return t.shape; })
        .def_property_readonly("device", [](const Tensor &t)
                               { return t.device == Device::CPU ? "cpu" : "cuda"; })
        .def("numel", &Tensor::numel);

    // 算子
    m.def("add", &add, "C = A + B");

    // numpy 互转
    m.def("from_numpy", &tensor_from_numpy, py::arg("array"), py::arg("device") = "cpu");
    m.def("to_numpy", &tensor_to_numpy, py::arg("tensor"));
}
