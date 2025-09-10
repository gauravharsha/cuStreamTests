#include <stdio.h>
#include <exception>
#include "cu_routines.h"
#include "cuda_check.h"
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  // Check for GPU Devices
  int num_devices = 1;
  check_for_cuda(num_devices);

  // Run
  try {
    streams_and_handles(MPI_COMM_WORLD);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // Perform kernel
  MPI_Finalize();
  return 0;
}
