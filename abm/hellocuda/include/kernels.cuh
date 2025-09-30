#pragma once

void vector_add_gpu(const float *hA, const float *hB, float *hC, size_t n);
void cuda_warmup();
void vecadd_alloc(float *&dA, float *&dB, float *&dC, size_t n);
void vecadd_run(const float *dA, const float *dB, float *dC, size_t n);
void vecadd_free(float *dA, float *dB, float *dC);
void vector_add_gpu_chunked(const float *hA, const float *hB, float *hC, size_t n);
