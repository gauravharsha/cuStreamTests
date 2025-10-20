#include <curand.h>
#include <curand_kernel.h>
#include "common_defs.h"
#include "cuda_types_map.h"
#include <cstring>
#include "cublas_routines_prec.h"

using cxx_base_type = double;
using cxx_type = std::complex<double>;
using cuda_type = cuDoubleComplex;

using scalar_t     = cu_type_map<std::complex<double>>::cxx_base_type;
using cxx_complex  = cu_type_map<std::complex<double>>::cxx_type;
using cuda_complex = cu_type_map<std::complex<double>>::cuda_type;


__global__ void init_random_complex(cuda_complex* data, curandState* states, int n, unsigned long seed);
						
/**
 * @brief Run the stream/handle test with configurable sizes and stream count 
 * 
 * @param rank rank of the MPI process
 * @param ns number of spins (tasks dimension 1)
 * @param nao number of atomic orbitals (matrix dimension)
 * @param naux number of auxiliary functions
 * @param nts number of tau points (tasks dimension 2)
 * @param n_streams number of CUDA streams/handles to use (>=1)
 * @param nt_batch number of tau points to process in a batch
 */
void streams_and_handles(int rank, size_t ns, size_t nao, size_t naux, size_t nts, int n_streams, int nt_batch);
