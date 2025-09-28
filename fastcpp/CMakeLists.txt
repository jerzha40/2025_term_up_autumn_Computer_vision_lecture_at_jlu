cmake_minimum_required(VERSION 3.18)
project(fastcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
    add_compile_options(/O2 /EHsc /utf-8)
else()
    add_compile_options(-O3 -fPIC)
endif()

# ✅ 用 CMake 自带的 FindPython3，一次找齐 Python + 开发头 + NumPy
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

# 生成 Python 扩展模块 "fast"（import fast）
pybind11_add_module(fast src/fast.cpp)

# ✅ 使用 FindPython3 提供的 NumPy 头文件路径
target_include_directories(fast PRIVATE ${Python3_NumPy_INCLUDE_DIRS})

# 可选：更友好的错误信息
target_compile_definitions(fast PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)
