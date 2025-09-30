#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

enum class Device
{
    CPU,
    CUDA
};

struct Tensor
{
    float *data = nullptr;
    std::vector<int64_t> shape;
    Device device;

    Tensor(const std::vector<int64_t> &shape, Device dev = Device::CPU)
        : shape(shape), device(dev)
    {
        size_t n = numel();
        if (device == Device::CPU)
        {
            data = static_cast<float *>(std::malloc(n * sizeof(float)));
            if (!data)
                throw std::bad_alloc{};
        }
        else
        {
            auto err = cudaMalloc(&data, n * sizeof(float));
            if (err != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    // 禁止拷贝，支持移动，避免双重释放
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept
        : data(other.data), shape(std::move(other.shape)), device(other.device)
    {
        other.data = nullptr;
    }
    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            this->~Tensor();
            data = other.data;
            shape = std::move(other.shape);
            device = other.device;
            other.data = nullptr;
        }
        return *this;
    }

    ~Tensor()
    {
        if (data)
        {
            if (device == Device::CPU)
                std::free(data);
            else
                cudaFree(data);
        }
    }

    size_t numel() const
    {
        size_t n = 1;
        for (auto d : shape)
            n *= static_cast<size_t>(d);
        return n;
    }
};

// ---- 声明算子接口（在 ops.cu 里实现） ----
Tensor add(const Tensor &A, const Tensor &B);

// host <-> device 拷贝
inline void copy_h2d(float *d_dst, const float *h_src, size_t n)
{
    auto err = cudaMemcpy(d_dst, h_src, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}
inline void copy_d2h(float *h_dst, const float *d_src, size_t n)
{
    auto err = cudaMemcpy(h_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}
