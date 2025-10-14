# cuStreamTests

This repository is a minimal code that recreates the essential steps from the [Green-Phys/GREEN-GPU](https://github.com/green-phys/green-gpu.git) for testing concurrency of CUBLAS calls from different CUDA Streams.

## Installation
```bash
mkdir build && cd build
cmake ..
make -j 4
make install
```

## Usage
The code allows you to tweak a whole bunch of parameters that are useful in a practical calculation:
* `ns`: Number of spins (default 2)\n"
* `nao`: Number of AOs (default 54)\n"
* `naux`: Number of aux functions (default 638)\n"
* `nts`: Number of time slices (default 10)\n"
* `nstreams`: Number of CUDA streams per rank (default 1)\n"

For most cases, we would like to vary `naux`, `nts` and `nao`. It is best to keep 1 stream per MPI rank.

1. To simply run the code and print GPU timings:
    ```bash
    # ensure we are in the build directory
    mpirun -n 4 ./main.exe --naux 300 --nts 100 --nstreams 1
    ```

2. To profile the code using Nsight Systems, we can use a bash script for nsys command:
    ```bash
    #!/bin/bash
    # Use $PMI_RANK for MPICH and $SLURM_PROCID with srun.
    if [ $SLURM_PROCID -eq 0 ]; then
      nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 -t nvtx,cuda "$@"
    else
      "$@"
    fi
    ```
    This profiles only the process with rank 0. 
    We can then run the job
    ```bash
    #!/bin/bash
    #SBATCH ... accounting options ...
    #SBATCH -N 1
    #SBATCH --ntasks-per-node=4
    #SBATCH --cpus-per-task=1
    #SBATCH --gpus-per-node=1
    #SBATCH -o slurm_output.o%j
    ##SBATCH --exclusive
    ##--Commented out--

    #Define environment variables
    export SLURM_CPU_BIND="cores"
    export Bin="path/to/main.exe"

    #Print Node info
    echo "My job ran on "
    echo $SLURM_NODELIST
    echo "Start Date/Time"
    date

    # Profile for Rank 0 only
    # srun nsys_profile.sh profile ./main.exe --naux 300 --nts 100 --nstreams 1

    # Profile for all Ranks
    srun nsys profile -o aux300_report_rank%q{SLURM_PROCID} \
      -t cuda,nvtx,cublas --cuda-memory-usage=true \
      $Bin/main.exe --naux 300 --nts 100 --nsreams 1

    #End of job info
    echo "End 
    ```