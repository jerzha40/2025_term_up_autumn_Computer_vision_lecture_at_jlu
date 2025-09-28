cmake_minimum_required(VERSION 3.18)

# === 把命令行里的两个参数写进 Cache（在 project() 之前） ===
# 如果系统里有 CUDA_PATH，就默认用它；也支持手动覆盖。
set(CUDA_PATH "D:/workspace-git/remote/jerzha40/2025_term_up_autumn_Computer_vision_lecture_at_jlu/venv/Lib/site-packages/nvidia/cu13")
set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc.exe")

project(cpp_hello LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86) # 按你的显卡改：61/75/86/89

add_executable(cpp_hello
    src/main.cpp
    src/add.cu
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cpp_hello PRIVATE CUDA::cudart)
