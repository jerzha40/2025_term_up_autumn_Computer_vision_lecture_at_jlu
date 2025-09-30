#include <iostream>
#include <perco/core/tensor.h>
int main()
{
    using perco::Tensor;
    Tensor a({2, 3});    // 2x3
    Tensor b({2, 3, 4}); // 2x3x4
    Tensor c({5});       // 长度为 5 的向量
    auto print = [](const Tensor &t, const char *name)
    {
        std::cout << name << ".shape = [";
        const auto &s = t.shape();
        for (size_t i = 0; i < s.size(); ++i)
        {
            std::cout << s[i] << (i + 1 < s.size() ? ", " : "");
        }
        std::cout << "], numel = " << t.numel() << "\n";
    };
    print(a, "a");
    print(b, "b");
    print(c, "c");
    return 0;
}
