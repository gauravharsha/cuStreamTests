#include <nvtx3/nvToolsExt.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <ctime>
#include "cu_routines.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

// Global variables for NVTX coloring
static int nvtx_rank_color_offset = 0;

#define PUSH_RANGE(name,cid) { \
  int color_id = (cid) + nvtx_rank_color_offset; \
  color_id = color_id%num_colors;\
  nvtxEventAttributes_t eventAttrib = {0}; \
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = colors[color_id]; \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
  eventAttrib.message.ascii = name; \
  nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();


__global__ void init_random_complex(cuda_complex* data, curandState* states, int n, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    // Initialize the RNG state for this thread
    curand_init(seed, id, 0, &states[id]);

    // Generate random real and imaginary parts in (-1, 1)
    float real = 2.0f * curand_uniform(&states[id]) - 1.0f;
    float imag = 2.0f * curand_uniform(&states[id]) - 1.0f;

    // Store as cuComplex
    data[id] = make_cuDoubleComplex(real, imag);
}


void streams_and_handles(MPI_Comm comm) {
  // ---- MPI init/get rank/size (assumes MPI_Init/Finalize handled in main) ----
  int rank = 0, nprocs = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  // Get local rank on the node for correct GPU binding
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  int local_rank = 0;
  MPI_Comm_rank(node_comm, &local_rank);
  MPI_Comm_free(&node_comm);

  // Set NVTX color offset based on rank
  nvtx_rank_color_offset = rank;

  // Bind GPU by local_rank (respects CUDA_VISIBLE_DEVICES if set)
  int nDevices = 0;
  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0) {
    throw std::runtime_error("No CUDA devices visible on this node");
  }
  cudaSetDevice(local_rank % nDevices);

  const int n_streams = 2;
  PUSH_RANGE("Initialize", 0);
  const size_t ns = 2;
  const size_t nao = 54;
  const size_t naux = 638;
  const size_t nts = 10;

  const size_t ntnaux2 = nts * naux * naux;
  const size_t ntnao2 = nts * nao * nao;
  const size_t nauxnao2 = naux * nao * nao;
  const size_t nao2     = nao * nao;
  const size_t nauxnao  = naux * nao;

  // --- decide (s,t) tasks for this rank (round-robin for load balance) ---
  const int total_tasks = static_cast<int>(ns * nts); // each task is one (s,t)
  std::vector<int> my_tasks;
  my_tasks.reserve((total_tasks + nprocs - 1) / nprocs);
  for (int lin = rank; lin < total_tasks; lin += nprocs) my_tasks.push_back(lin);
  const size_t n_local_tasks = my_tasks.size();

  // allocate memory for stuff

  cuda_complex* Pqk0 = nullptr;
  if (cudaMalloc(&Pqk0, ntnaux2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("alloc Pq0");

  // Each rank only needs its local (s,t) slice of g_stij, shape: n_local_tasks * (nao×nao)
  cuda_complex* g_stij_local = nullptr;
  if (cudaMalloc(&g_stij_local, n_local_tasks * nao2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("alloc g_stij_local");

  // VQ can be shared conceptually, but we keep one copy per rank (simple, no comms)
  cuda_complex* VQ = nullptr;
  if (cudaMalloc(&VQ, nauxnao2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("alloc VQ");

  // Output Y for this rank (same shape as VQ since Y = g * VQ)
  cuda_complex* Y = nullptr;
  if (cudaMalloc(&Y, nauxnao2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("alloc Y");
  
  // Random initialize
  curandState* pq_state;
  curandState* g_state;
  curandState* vq_state;
  curandState* y_state;

  if (cudaMalloc(&pq_state, ntnaux2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("alloc pq_state");
  if (cudaMalloc(&g_state,  n_local_tasks * nao2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("alloc g_state");
  if (cudaMalloc(&vq_state, nauxnao2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("alloc vq_state");
  if (cudaMalloc(&y_state,  nauxnao2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("alloc y_state");

  int threads = 256;
  int blocks = (ntnaux2 + threads - 1) / threads;
   unsigned long seed = static_cast<unsigned long>(time(NULL)) + 1337ul * static_cast<unsigned long>(rank);

  // Pqk0_local
  blocks = (int)((ntnaux2 + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(Pqk0, pq_state, ntnaux2, seed);

  // g_stij_local
  const size_t n_local_g_elems = n_local_tasks * nao2;
  blocks = (int)((n_local_g_elems + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(g_stij_local, g_state, n_local_g_elems, seed+1);

  // VQ and Y
  blocks = (int)((nauxnao2 + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(VQ, vq_state, nauxnao2, seed+2);
  init_random_complex<<<blocks, threads>>>(Y,  y_state,  nauxnao2, seed+3);

  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("Initialization kernel failure");
  if (rank == 0) std::cout << "initialized completed\n";
  POP_RANGE;


  // Create streams and handles
  PUSH_RANGE("Create streams and handles", 1);
  std::vector<cudaStream_t> streams(n_streams);
  std::vector<cublasHandle_t> handles(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    if (cublasCreate(&handles[i]) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasCreate failed");
    if (cudaStreamCreate(&streams[i]) != cudaSuccess)
      throw std::runtime_error("cudaStreamCreate failed");
    cublasSetStream(handles[i], streams[i]);
  }
  POP_RANGE;

  for (int i = 0; i < n_streams; ++i) cudaStreamSynchronize(streams[i]);

  // Perform Batched DGEMM
  cuda_complex one  = cu_type_map<cxx_complex>::cast( 1., 0.);
  cuda_complex zero = cu_type_map<cxx_complex>::cast( 0., 0.);
  PUSH_RANGE("Per-rank GEMMs (single g/Y per MPI rank)", 3);
  // Dispatch GEMMs across MPI ranks: if at least 2 ranks, rank 0 runs GEMM1 and rank 1 runs GEMM2.
  // If only one rank, run both GEMMs sequentially.
  // Partition work across MPI ranks and round-robin streams/handles within each rank.
  // Each rank will execute every `mpi_size`-th `t` value starting from its rank.
  int local_task_counter = 0;
  for (size_t k = 0; k < n_local_tasks; ++k) {
    // map back to (s,t) if needed (not required for GEMM itself)
    // int lin = my_tasks[k];
    // int s = lin / (int)nts;
    // int t = lin % (int)nts;

    const size_t g_off = k * nao2;

    // Pick a stream/handle
    int which = local_task_counter % n_streams;
    cublasHandle_t h = handles[which];

    // Single GEMM per task:
    //   (nao x nao) * (nao x nauxnao)  -> (nao x nauxnao)
    // We keep your STRIDED_BATCHED wrapper for API uniformity; batchCount=1, strides as given.
    if (GEMM_STRIDED_BATCHED(h,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             nao, nauxnao, nao,
                             &one,
                             g_stij_local + g_off, nao, nao2,
                             VQ, nao, 0,
                             &zero,
                             Y, nao, nauxnao2,
                             1) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Rank " + std::to_string(rank) + " GEMM failed");
    }

    ++local_task_counter;
  }

  for (int i = 0; i < n_streams; ++i) cudaStreamSynchronize(streams[i]);
  POP_RANGE;

  MPI_Barrier(comm);

  // --- cleanup ---
  for (int i = 0; i < n_streams; ++i) {
    cublasDestroy(handles[i]);
    cudaStreamDestroy(streams[i]);
  }

  cudaFree(Pqk0);
  cudaFree(g_stij_local);
  cudaFree(VQ);
  cudaFree(Y);

  cudaFree(pq_state);
  cudaFree(g_state);
  cudaFree(vq_state);
  cudaFree(y_state);
}

