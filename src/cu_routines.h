#include <curand.h>
#include <curand_kernel.h>
#include "common_defs.h"
#include "cuda_types_map.h"
#include <cstring>
#include "cublas_routines_prec.h"
#include <mpi.h>

using cxx_base_type = double;
using cxx_type = std::complex<double>;
using cuda_type = cuDoubleComplex;

using scalar_t     = cu_type_map<std::complex<double>>::cxx_base_type;
using cxx_complex  = cu_type_map<std::complex<double>>::cxx_type;
using cuda_complex = cu_type_map<std::complex<double>>::cuda_type;


__global__ void init_random_complex(cuda_complex* data, curandState* states, int n, unsigned long seed);
void streams_and_handles(MPI_Comm comm);
