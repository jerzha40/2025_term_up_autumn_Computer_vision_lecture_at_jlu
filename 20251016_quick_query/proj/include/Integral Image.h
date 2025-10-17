/*
This is the main algroithm I have implement as the assigment
USE GBK to open the chinese characters
这是主要我作业算法写的函数
包括:
    get_sum
    get_mean
    get_var
20251017张津睿
*/
#ifndef _INTEGRAL_IMAGE_H_
#define _INTEGRAL_IMAGE_H_
#include "Base.h"
long long *get_sum(Image *M);
BYTE get_mean(long long *SM, size_t si, size_t sj, size_t li, size_t lj, size_t width);
#endif