# Don't forget to set
# -DCMAKE_TOOLCHAIN_FILE=C:\tools\vcpkg\scripts\buildsystems\vcpkg.cmake
# when calling cmake

cmake_minimum_required(VERSION 3.28)
project(c_blas C)

set(CMAKE_C_STANDARD 11)

add_compile_options(-msse4.1 -mavx)
set(CMAKE_C_FLAGS_RELEASE "-Ofast")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

find_package(OpenBLAS CONFIG REQUIRED)

add_executable(c_blas main.c)

target_link_libraries(c_blas PRIVATE OpenBLAS::OpenBLAS)
