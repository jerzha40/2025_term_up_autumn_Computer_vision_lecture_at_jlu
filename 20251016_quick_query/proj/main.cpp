#include <Image Processing.h>
#include <stdio.h>
#include <Integral Image.h>
#define PIX(x, y, width) ((y) * (width) + (x))
int main()
{
    Image *A;
    A = READ_BMP_FROM_FILE("Peppers512.bmp");
    bool st = SAVE_BMP(A->ImageData, A->iWidth, A->iHeight, "repeat.bmp", A->ImageType);
    printf("%d", st);
    Image B;

    // SIZE_T srcSize = A->iWidth * A->iHeight * 3;
    // LPBYTE src = (LPBYTE)GlobalAlloc(GPTR, srcSize);

    // 目标：灰度，大小 = A->iWidth * A->iHeight 字节（函数里把 SP[i] 写成 0..255）
    SIZE_T dstSize = A->iWidth * A->iHeight;
    LPBYTE dst = (LPBYTE)GlobalAlloc(GPTR, dstSize);
    if (!CHANGE_IMAGE_TO_GRAY(A->ImageData, dst, A->iWidth, A->iHeight, A->ImageType))
    {
        printf("CHANGE_IMAGE_TO_GRAY failed\n");
    }
    else
    {
        // 现在 dst 里是 512*512 的灰度图（每像素 1 字节）
        // 你可以在这里保存/显示/进一步处理
        printf("OK, gray size = %zu bytes\n", dstSize);
    }
    B.ImageData = dst;
    B.ImageType = RGB8;
    B.iHeight = A->iHeight;
    B.iWidth = A->iWidth;

    st = SAVE_BMP(B.ImageData, A->iWidth, A->iHeight, (char *)"gray.bmp", B.ImageType);

    long long *SB = get_sum(&B);

    for (size_t L = 0; L < 20; L++)
    {
        Image *AB = (Image *)GlobalAlloc(GPTR, sizeof(Image));
        AB->iWidth = B.iWidth - L;
        AB->iHeight = B.iHeight - L;
        AB->ImageType = B.ImageType;
        AB->ImageData = (LPBYTE)GlobalAlloc(GPTR, AB->iWidth * AB->iHeight);
        for (size_t i = 0; i < B.iWidth - L; i++)
        {
            for (size_t j = 0; j < B.iHeight - L; j++)
            {
                AB->ImageData[PIX(i, j, AB->iWidth)] = get_mean(SB, i, j, L, L, B.iWidth);
            }
        }
        char ss[20];
        sprintf(ss, "gray%llu.bmp", L);
        st = SAVE_BMP(AB->ImageData, AB->iWidth, AB->iHeight, ss, AB->ImageType);
    }

    GlobalFree(dst);
    // GlobalFree(src);

    return 0;
}