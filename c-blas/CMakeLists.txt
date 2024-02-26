# Don't forget to set
# -DCMAKE_TOOLCHAIN_FILE=C:\tools\vcpkg\scripts\buildsystems\vcpkg.cmake
# when calling cmake

cmake_minimum_required(VERSION 3.28)

project(c_blas CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)

if (MSVC)
    message("It is MSVC")
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-release" CACHE STRING "")
    add_compile_options(/arch:AVX)
    set(CMAKE_C_FLAGS_RELEASE "/O2")
    set(CMAKE_C_FLAGS_RELEASE "/O2")
else ()
    add_compile_options(-msse4.1 -mavx -march=native)
    set(CMAKE_C_FLAGS_RELEASE "-Ofast")
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
endif ()

find_package(OpenBLAS CONFIG REQUIRED)

add_executable(c_blas main.cpp)

target_link_libraries(c_blas PRIVATE OpenBLAS::OpenBLAS)
