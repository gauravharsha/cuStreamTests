cmake_minimum_required(VERSION 3.18)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(gpu)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

################ GREEN RELEASE VERSION #####################
set(GREEN_RELEASE origin/main)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)
enable_language("CUDA")
include(CheckLanguage)
check_language(CUDA)
include(CudaArchitectures)
define_cuda_architectures()
add_compile_definitions(WITH_CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS)
if (BLAS_FOUND)
    if (BLAS_LIBRARIES MATCHES ".*Accelerate")
        message(INFO " Using Apple BLAS")
    endif ()
    if (BLAS_LIBRARIES MATCHES ".*openblas")
        message(INFO " Using OpenBLAS")
        add_compile_definitions(USE_GEMM3M)
    endif ()
    add_compile_definitions(EIGEN_USE_BLAS)
endif ()

include(GreenDeps)
add_green_dependency(green-ndarray)

add_subdirectory(src)

add_executable(main.exe main.cpp)
target_link_libraries(main.exe PRIVATE accel-lib MPI::MPI_CXX)
