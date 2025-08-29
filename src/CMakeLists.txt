project(gpu-lib CXX)

add_library(accel-lib cuda_check.cpp cublas_routines_prec.cu cu_routines.cu)
set_property(TARGET accel-lib PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS}) #also check the same parameters in solvers
target_include_directories(accel-lib PUBLIC .)
target_link_libraries(accel-lib CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
target_link_libraries(accel-lib Eigen3::Eigen GREEN::NDARRAY)
