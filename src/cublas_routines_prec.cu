/*
* Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cusolverDn.h>
#include "cublas_routines_prec.h"


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
        int batchCount) {
  return cublasZgemmStridedBatched(handle, transa, transb, m, n, k,
          alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}
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
        int batchCount) {
  return cublasCgemmStridedBatched(handle, transa, transb, m, n, k,
          alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}
