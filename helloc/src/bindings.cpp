#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

// 在 hello.cu 里实现
extern "C" void launch_nop();

static std::string hello_cpu()
{
    return "hello from C++ (CPU)";
}

static std::string hello_cuda()
{
    launch_nop(); // 启个空 CUDA kernel，证明 CUDA 通了
    return "hello from CUDA";
}

PYBIND11_MODULE(hellocuda, m)
{
    m.doc() = "HelloWorld: Python ↔ C++ ↔ CUDA (MSVC)";
    m.def("hello_cpu", &hello_cpu, "Hello from C++");
    m.def("hello_cuda", &hello_cuda, "Hello from CUDA");
}
