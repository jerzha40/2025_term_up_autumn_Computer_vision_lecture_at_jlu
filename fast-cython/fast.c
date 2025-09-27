// fast.c  (unchanged)
#include <math.h>
double sumsq(const double *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += x[i] * x[i];
    return s;
}
