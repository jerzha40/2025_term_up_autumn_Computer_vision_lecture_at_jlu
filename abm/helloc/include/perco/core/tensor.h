#ifndef PERCO_TENSOR_H
#define PERCO_TENSOR_H
#include <vector>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
namespace perco
{
    enum class DType
    {
        Float32
    };
    class Tensor
    {
    public:
        Tensor() = default;
        ~Tensor() = default;
        explicit Tensor(std::initializer_list<uint64_t> shape)
            : shape_(shape),
              dtype_(DType::Float32),
              data_(numel()) {}
        const std::vector<uint64_t> &shape() const { return shape_; }
        uint64_t numel() const
        {
            uint64_t n = 1;
            for (auto d : shape_)
                n *= d;
            return n;
        }
        float *data() { return data_.data(); }
        const float *data() const { return data_.data(); }
        std::vector<uint64_t> strides() const
        {
            std::vector<uint64_t> s(shape_.size());
            if (shape_.empty())
                return s;
            // 最后一维的 stride = 1
            s.back() = 1;
            // 从倒数第二维开始往前推
            for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i)
            {
                s[i] = s[i + 1] * shape_[i + 1];
            }
            return s;
        }
        uint64_t offset(std::initializer_list<uint64_t> indices) const
        {
            if (indices.size() != shape_.size())
            {
                throw std::invalid_argument("perco::Tensor::offset: indices rank mismatch");
            }
            auto s = strides();
            uint64_t off = 0;
            size_t d = 0;
            for (uint64_t idx : indices)
            {
                if (idx >= shape_[d])
                {
                    throw std::out_of_range("perco::Tensor::offset: index out of bounds");
                }
                off += idx * s[d];
                ++d;
            }
            return off;
        }
        float &at(std::initializer_list<uint64_t> indices)
        {
            return data_[offset(indices)];
        }
        const float &at(std::initializer_list<uint64_t> indices) const
        {
            return data_[offset(indices)];
        }
        explicit Tensor(const std::vector<uint64_t> &shape)
            : shape_(shape), dtype_(DType::Float32), data_(numel()) {}

    private:
        std::vector<uint64_t> shape_;
        DType dtype_;
        std::vector<float> data_;
    };
    Tensor einsum(const std::string &spec, const Tensor &A, const Tensor &B);
} // namespace perco
#endif // PERCO_TENSOR_H
