#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>


cublasStatus_t GEMM_STRIDED_BATCHED(cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        long long int          strideA,
        const cuDoubleComplex *B, int ldb,
        long long int          strideB,
        const cuDoubleComplex *beta,
        cuDoubleComplex       *C, int ldc,
        long long int          strideC,
        int batchCount);
cublasStatus_t GEMM_STRIDED_BATCHED(cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const cuComplex *alpha,
        const cuComplex *A, int lda,
        long long int          strideA,
        const cuComplex *B, int ldb,
        long long int          strideB,
        const cuComplex *beta,
        cuComplex       *C, int ldc,
        long long int          strideC,
        int batchCount);
