cmake_minimum_required(VERSION 3.18)
project(gpu-gemm)

include(GNUInstallDirs)
include(CheckIncludeFileCXX)
include(CheckLibraryExists)

include_directories(BEFORE .)

set(CMAKE_CXX_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

add_executable(gemm src/gemm.cu)
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES 89)
target_link_libraries(gemm PRIVATE CUDA::cublas CUDA::curand)
target_compile_options(gemm PRIVATE --resource-usage --generate-line-info --extra-device-vectorization)
