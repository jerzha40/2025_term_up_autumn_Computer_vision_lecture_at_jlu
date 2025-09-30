#pragma once
#include <cstddef>

void vecadd_gpu(const float *hA, const float *hB, float *hC, size_t n);
void vecadd_gpu_chunked(const float *hA, const float *hB, float *hC, size_t n);
void addKernel(const float *A, const float *B, float *C, size_t n);