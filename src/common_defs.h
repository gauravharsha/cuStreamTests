#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>

#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in)
template <size_t Dim>
using ztensor = green::ndarray::ndarray<std::complex<double>, Dim>;