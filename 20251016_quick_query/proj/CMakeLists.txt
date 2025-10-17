cmake_minimum_required(VERSION 3.25)
# project(proj LANGUAGES CXX CUDA)
project(proj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# find_package(CUDAToolkit REQUIRED) # 提供 CUDA::cudart 等

message(_______________${CMAKE_CURRENT_SOURCE_DIR})
message(_______________${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE MYAPP_SOURCES "src/*.cpp")


add_library(IMPR ${MYAPP_SOURCES})

if(MSVC)
    set_target_properties(IMPR PROPERTIES MFC_FLAG 2)
    target_compile_definitions(IMPR PRIVATE _AFXDLL)
endif()

if(MSVC)
    target_precompile_headers(IMPR PRIVATE include/stdafx.h)
endif()

add_executable(MAIN main.cpp)
target_link_libraries(MAIN IMPR)
