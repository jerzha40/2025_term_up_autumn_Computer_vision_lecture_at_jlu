cmake_minimum_required(VERSION 3.25)
project(hellocuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED) # 提供 CUDA::cudart 等

message(_______________${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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

# pybind11_add_module(_fasttorch
#     src/fasttorch.cpp
#     src/ops.cu
# )
# target_include_directories(_fasttorch PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# set_target_properties(_fasttorch PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     OUTPUT_NAME "_fasttorch" # 生成 hellocuda.pyd/.so
#     PREFIX "${PYTHON_MODULE_PREFIX}"
#     SUFFIX "${PYTHON_MODULE_EXTENSION}"
# )
# target_link_libraries(_fasttorch PRIVATE CUDA::cudart)
# install(TARGETS _fasttorch
#     LIBRARY DESTINATION hellocuda # Linux/Mac
#     RUNTIME DESTINATION hellocuda # Windows 兼容
#     ARCHIVE DESTINATION hellocuda
# )
# install(TARGETS _fasttorch
#     LIBRARY DESTINATION hellocuda
#     RUNTIME DESTINATION hellocuda
#     ARCHIVE DESTINATION hellocuda
# )

# --- fastnn (Dense 层示例) ---
pybind11_add_module(_fastnn
    src/dense.cpp
    src/dense.cu
)
target_include_directories(_fastnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(_fastnn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "_fastnn"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
target_link_libraries(_fastnn PRIVATE CUDA::cudart)
install(TARGETS _fastnn
    LIBRARY DESTINATION hellocuda
    RUNTIME DESTINATION hellocuda
    ARCHIVE DESTINATION hellocuda
)
