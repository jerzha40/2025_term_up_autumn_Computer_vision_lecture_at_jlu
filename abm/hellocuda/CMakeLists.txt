cmake_minimum_required(VERSION 3.25)
project(hellocuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED) # 提供 CUDA::cudart 等

# 建模块，名字要叫 hellocuda（和 PYBIND11_MODULE 一致）
pybind11_add_module(hellocuda
    src/bindings.cpp
    src/kernels.cu
)

target_include_directories(hellocuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(hellocuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "hellocuda" # 生成 hellocuda.pyd/.so
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(hellocuda PRIVATE CUDA::cudart)

# 关键：把二进制扩展安装到包目录 "hellocuda"（即 python 包名）
# scikit-build-core 会把它和 python/hellocuda/__init__.py 一起打进 wheel
install(TARGETS hellocuda
    LIBRARY DESTINATION hellocuda # Linux/Mac
    RUNTIME DESTINATION hellocuda # Windows 兼容
    ARCHIVE DESTINATION hellocuda
)
