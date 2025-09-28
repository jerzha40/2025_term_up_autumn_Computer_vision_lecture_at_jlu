cmake_minimum_required(VERSION 3.25)

project(helloc LANGUAGES CXX CUDA)

add_executable(main src/helloc.cpp src/kernels.cu)
