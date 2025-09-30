// dense.cuh
#pragma once
#include <cstddef>

void dense_forward(const float *input, const float *weights, const float *bias,
                   float *output, size_t batch, size_t in_features, size_t out_features);

void dense_backward(const float *input, const float *weights,
                    const float *d_output, float *d_input,
                    float *d_weights, float *d_bias,
                    size_t batch, size_t in_features, size_t out_features,
                    float learning_rate);
