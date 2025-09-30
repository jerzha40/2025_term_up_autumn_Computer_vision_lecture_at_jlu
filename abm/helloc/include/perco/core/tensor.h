#ifndef PERCO_TENSOR_H
#define PERCO_TENSOR_H
#include <vector>
#include <cstdint>
namespace perco
{
    class Tensor
    {
    public:
        Tensor() = default;
        ~Tensor() = default;
        explicit Tensor(std::initializer_list<uint64_t> shape)
            : shape_(shape) {}
        const std::vector<uint64_t> &shape() const { return shape_; }
        uint64_t numel() const
        {
            uint64_t n = 1;
            for (auto d : shape_)
                n *= d;
            return n;
        }
    private:
        std::vector<uint64_t> shape_;
    };
} // namespace perco
#endif // PERCO_TENSOR_H
