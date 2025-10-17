#include <Integral Image.h>
#define PIX(x, y, width) ((y) * (width) + (x))
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
long long *get_sum(Image *M)
{
    SIZE_T dataSize = M->iWidth * M->iHeight;
    long long *data = (long long *)GlobalAlloc(GPTR, dataSize * sizeof(long long));
    for (size_t i = 0; i < dataSize; i++)
    {
        data[i] = -255;
    }
    data[PIX(0, 0, M->iWidth)] = M->ImageData[PIX(0, 0, M->iWidth)];
    for (size_t i = 1; i < M->iWidth; i++)
    {
        data[PIX(0, i, M->iWidth)] = M->ImageData[PIX(0, i, M->iWidth)] + data[PIX(0, i - 1, M->iWidth)];
    }
    for (size_t i = 1; i < M->iHeight; i++)
    {
        data[PIX(i, 0, M->iWidth)] = M->ImageData[PIX(i, 0, M->iWidth)] + data[PIX(i - 1, 0, M->iWidth)];
    }
    for (size_t i = 1; i < M->iWidth; i++)
    {
        for (size_t j = 1; j < M->iHeight; j++)
        {
            data[PIX(i, j, M->iWidth)] = M->ImageData[PIX(i, j, M->iWidth)] + data[PIX(i - 1, j, M->iWidth)] + data[PIX(i, j - 1, M->iWidth)] - data[PIX(i - 1, j - 1, M->iWidth)];
        }
    }
    return data;
}
BYTE get_mean(long long *SM, size_t si, size_t sj, size_t li, size_t lj, size_t width)
{
    double s = SM[PIX(si + li, sj + lj, width)];
    if (si > 0)
    {
        s -= SM[PIX(si - 1, sj + lj, width)];
    }
    if (sj > 0)
    {
        s -= SM[PIX(si + li, sj - 1, width)];
    }
    if (sj > 0 && si > 0)
    {
        s += SM[PIX(si - 1, sj - 1, width)];
    }
    return (BYTE)(s / ((li + 1) * (lj + 1)));
}
