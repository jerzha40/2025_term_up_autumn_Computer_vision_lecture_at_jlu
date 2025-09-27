// fast.c
#include <math.h>
// export this function so Python can see it
__declspec(dllexport) double sumsq(const double *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i)
    {
        s += x[i] * x[i];
    }
    return s;
}
