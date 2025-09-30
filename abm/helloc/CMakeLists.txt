cmake_minimum_required(VERSION 3.25)
project(hellocuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED) # 提供 CUDA::cudart 等

# 建模块，名字要叫 hellocuda（和 PYBIND11_MODULE 一致）
pybind11_add_module(_fastoper
    src/vadd.cpp
    src/kernels.cu
)
target_include_directories(_fastoper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(_fastoper PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "_fastoper" # 生成 hellocuda.pyd/.so
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
target_link_libraries(_fastoper PRIVATE CUDA::cudart)

install(TARGETS _fastoper
    LIBRARY DESTINATION hellocuda # Linux/Mac
    RUNTIME DESTINATION hellocuda # Windows 兼容
    ARCHIVE DESTINATION hellocuda
)
