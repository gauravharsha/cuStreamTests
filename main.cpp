#include <stdio.h>
#include "cu_routines.h"
#include "cuda_check.h"
#include "cu_routines.h"


int main() {
  // Check for GPU Devices
  int num_devices = 1;
  check_for_cuda(num_devices);

  // Run
  streams_and_handles();

  // Perform kernel
  return 0;
}
