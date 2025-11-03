cmake_minimum_required(VERSION 3.25)
project(proj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

# 启用 OpenMP
add_executable(${PROJECT_NAME}
    main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

