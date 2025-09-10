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

#define CUDA_CHECK(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ")+cudaGetErrorString(e)); \
} while(0)

#define CUBLAS_CHECK(cmd) do { \
  cublasStatus_t s = (cmd); \
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error"); \
} while(0)

void streams_and_handles(MPI_Comm comm) {
  // ---- MPI ranks (world + local) ----
  int rank=0, nprocs=1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  int local_rank=0;
  MPI_Comm_rank(node_comm, &local_rank);
  MPI_Comm_free(&node_comm);

  // ---- Bind GPU by local rank (respects CUDA_VISIBLE_DEVICES) ----
  int nDevices=0;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices == 0) throw std::runtime_error("No CUDA devices");
  CUDA_CHECK(cudaSetDevice(local_rank % nDevices));

  // ---- Problem sizes (same as your example) ----
  const int n_streams = 2;
  PUSH_RANGE("Initialize", 0);
  const size_t ns   = 2;
  const size_t nao  = 54;
  const size_t naux = 638;
  const size_t nts  = 10;

  const size_t ntnaux2  = nts * naux * naux;     // (#tau) * naux^2
  const size_t nao2     = nao * nao;
  const size_t nauxnao  = naux * nao;
  const size_t nauxnao2 = naux * nao * nao;

  // ---- Task partition: round-robin (good balance) ----
  const int total_tasks = (int)(ns * nts); // each (s,t) is a task
  std::vector<int> my_tasks;
  my_tasks.reserve((total_tasks + nprocs - 1) / nprocs);
  for (int lin = rank; lin < total_tasks; lin += nprocs) my_tasks.push_back(lin);
  const size_t n_local_tasks = my_tasks.size();

  // ---- Device allocations: per-rank single set ----
  cuda_complex *Pqk0=nullptr, *g_stij_local=nullptr, *VQ=nullptr;
  CUDA_CHECK(cudaMalloc(&Pqk0, ntnaux2 * sizeof(cuda_complex)));
  CUDA_CHECK(cudaMalloc(&g_stij_local, n_local_tasks * nao2 * sizeof(cuda_complex)));
  CUDA_CHECK(cudaMalloc(&VQ,   nauxnao2 * sizeof(cuda_complex)));

  // Double-buffer outputs: one per stream (prevents write hazards)
  cuda_complex* Y_buf[2] = {nullptr, nullptr};
  CUDA_CHECK(cudaMalloc(&Y_buf[0], nauxnao2 * sizeof(cuda_complex)));
  CUDA_CHECK(cudaMalloc(&Y_buf[1], nauxnao2 * sizeof(cuda_complex)));

  // ---- RNG states (only for the buffers we own) ----
  curandState *pq_state=nullptr, *g_state=nullptr, *vq_state=nullptr, *y_state0=nullptr, *y_state1=nullptr;
  CUDA_CHECK(cudaMalloc(&pq_state, ntnaux2 * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&g_state,  n_local_tasks * nao2 * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&vq_state, nauxnao2 * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&y_state0, nauxnao2 * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&y_state1, nauxnao2 * sizeof(curandState)));

  // ---- Random init (rank-unique seed) ----
  const int threads = 256;
  unsigned long seed = (unsigned long)time(NULL) + 1337ul * (unsigned long)rank;

  int blocks = (int)((ntnaux2 + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(Pqk0, pq_state, ntnaux2, seed);

  const size_t n_local_g_elems = n_local_tasks * nao2;
  blocks = (int)((n_local_g_elems + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(g_stij_local, g_state, n_local_g_elems, seed+1);

  blocks = (int)((nauxnao2 + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(VQ,      vq_state,  nauxnao2, seed+2);
  init_random_complex<<<blocks, threads>>>(Y_buf[0], y_state0, nauxnao2, seed+3);
  init_random_complex<<<blocks, threads>>>(Y_buf[1], y_state1, nauxnao2, seed+4);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (rank == 0) std::cout << "initialized completed\n";
  POP_RANGE;

  // ---- Create NON-BLOCKING streams & 1 handle per stream ----
  PUSH_RANGE("Create streams and handles", 1);
  std::vector<cudaStream_t> streams(n_streams);
  std::vector<cublasHandle_t> handles(n_streams);
  for (int i=0; i<n_streams; ++i) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handles[i]));
    CUBLAS_CHECK(cublasSetStream(handles[i], streams[i]));
  }
  POP_RANGE;

  // ---- Scalars ----
  cuda_complex one  = cu_type_map<cxx_complex>::cast( 1., 0.);
  cuda_complex zero = cu_type_map<cxx_complex>::cast( 0., 0.);

  // ---- Enqueue GEMMs alternating streams, NO sync inside loop ----
  PUSH_RANGE("Per-rank GEMMs (double-buffered, 2 streams)", 2);
  for (size_t k = 0; k < n_local_tasks; ++k) {
    const size_t g_off = k * nao2;

    int which = (int)k & 1;                 // 0,1 alternating
    cublasHandle_t h = handles[which];
    cuda_complex*   Y = Y_buf[which];

    // Single GEMM per task (STRIDED_BATCHED kept for interface compatibility)
    CUBLAS_CHECK(GEMM_STRIDED_BATCHED(
      h, CUBLAS_OP_N, CUBLAS_OP_N,
      (int)nao, (int)nauxnao, (int)nao,
      &one,
      g_stij_local + g_off, (int)nao, (long long)nao2,
      VQ,                    (int)nao, 0,
      &zero,
      Y,                     (int)nao, (long long)nauxnao2,
      1));
  }
  POP_RANGE;

  // ---- Join streams ONCE at the end (no serializing in the loop) ----
  PUSH_RANGE("Join", 3);
  for (int i=0; i<n_streams; ++i) CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  POP_RANGE;

  MPI_Barrier(comm);

  // ---- Cleanup ----
  for (int i=0; i<n_streams; ++i) {
    cublasDestroy(handles[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(Pqk0);
  cudaFree(g_stij_local);
  cudaFree(VQ);
  cudaFree(Y_buf[0]);
  cudaFree(Y_buf[1]);
  cudaFree(pq_state);
  cudaFree(g_state);
  cudaFree(vq_state);
  cudaFree(y_state0);
  cudaFree(y_state1);
}