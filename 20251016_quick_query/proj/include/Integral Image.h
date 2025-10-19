/*
This is the main algroithm I have implement as the assigment
USE GBK to open the chinese characters
这是主要我作业算法写的函数
包括:
    get_sum
    get_mean
    get_var
20251017张津睿

函数解释：
get_sum:
        input a matrix M export a matrix data
        the data[a,b]=sum_{i=0}^{a}(sum_{j=0}^{b}(M[i,j]))
get_mean:
        input the sum matrix SM start at [si,sj]
        get a rectangle with Side length of [li+1,lj+1]
        the width is determined the row width of SM
*/
#ifndef _INTEGRAL_IMAGE_H_
#define _INTEGRAL_IMAGE_H_
#include "Base.h"
long long *get_sum(Image *M);
BYTE get_mean(long long *SM, size_t si, size_t sj, size_t li, size_t lj, size_t width);
#endif