cmake_minimum_required(VERSION 3.25)

project(helloc LANGUAGES CXX CUDA)

include_directories(include)

add_executable(main src/helloc.cpp src/kernels.cu)
