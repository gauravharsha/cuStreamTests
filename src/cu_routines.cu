#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <ctime>
#include <cublas_v2.h>
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

void streams_and_handles(int rank, size_t ns, size_t nao, size_t naux, size_t nts, int n_streams, size_t nt_batch) {
  int nDevices=0;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices == 0) throw std::runtime_error("No CUDA devices");
  CUDA_CHECK(cudaSetDevice(0));

  if (n_streams < 1) throw std::runtime_error("n_streams must be >= 1");

  // Set NVTX color offset per-rank for better visualization
  nvtx_rank_color_offset = (rank % num_colors);

  // ---- Derived sizes ----
  const size_t ntnao2 = nts * nao * nao;
  const size_t ntnaux2  = nts * naux * naux;     // (#tau) * naux^2
  const size_t nao2     = nao * nao;
  const size_t nauxnao  = naux * nao;
  const size_t nauxnao2 = naux * nao * nao;

  // ---- Task partition: round-robin (good balance) ----
  PUSH_RANGE("Initialize", 0);
  cuda_complex *g_stij=nullptr, *VQ=nullptr, *Y=nullptr;
  CUDA_CHECK(cudaMalloc(&g_stij, ns * ntnao2 * sizeof(cuda_complex)));
  CUDA_CHECK(cudaMalloc(&VQ,     nauxnao2 * sizeof(cuda_complex)));
  CUDA_CHECK(cudaMalloc(&Y,      nt_batch * nauxnao2 * sizeof(cuda_complex)));

  // ---- RNG states (only for the buffers we own) ----
  curandState *g_state=nullptr, *vq_state=nullptr, *y_state=nullptr;
  CUDA_CHECK(cudaMalloc(&g_state,  ns * ntnao2 * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&vq_state, nauxnao2 * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&y_state,  nt_batch * nauxnao2 * sizeof(curandState)));

  // ---- Random init (rank-unique seed) ----
  const int threads = 256;
  unsigned long seed = (unsigned long)time(NULL) + 1337ul * (unsigned long)rank;

  blocks = (int)((ns * ntnao2 + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(g_stij, g_state, ntnao2, seed+1);

  blocks = (int)((nauxnao2 + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(VQ, vq_state, nauxnao2, seed+2);

  blocks = (int)((nauxnao2 + threads - 1) / threads);
  init_random_complex<<<blocks, threads>>>(Y, y_state, nt_batch * nauxnao2, seed+3);

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

  // ---- Synchronize the streams before starting ZGEMMs ----
  PUSH_RANGE("Synchronize", 3);
  for (int i=0; i<n_streams; ++i) CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  CUDA_CHECK(cudaDeviceSynchronize());
  POP_RANGE;

  auto total_flop_count = (double)ns * (double)nts * 8. * (double)nao * (double)nauxnao * (double)nao;

  // ---- Enqueue GEMMs alternating streams, NO sync inside loop ----
  auto start = std::chrono::high_resolution_clock::now();
  PUSH_RANGE("Per-rank GEMMs (streams)", 2);
  int n_repeat = 10;
  for (int repeat = 0; repeat < n_repeat; repeat++) {
    for (int s = 0; s < ns; ++s) {
      for (int t = 0; t < nts; t += nt_batch) {
        int st0      = s * nts + t;
        int nt_mult  = std::min(static_cast<int>(nt_batch), static_cast<int>(nts) - t);

        // Select stream/handle in round-robin across tasks
        size_t task_idx = (size_t)st0;
        int which = (int)(n_streams == 1 ? 0 : (task_idx % (size_t)n_streams));
        cublasHandle_t h = handles[which];

        // Single GEMM per task (STRIDED_BATCHED kept for interface compatibility)
        // Offsets: each (s,t) slice occupies contiguous blocks in g_stij and Y
        CUBLAS_CHECK(GEMM_STRIDED_BATCHED(
          h, CUBLAS_OP_N, CUBLAS_OP_N,
          (int)nao, (int)nauxnao, (int)nao,
          &one,
          g_stij + (size_t)st0 * nao2, (int)nao, (long long)nao2,
          VQ, (int)nao, 0,
          &zero,
          Y, (int)nao, (long long)nauxnao2,
          nt_mult
        ));
      }
    }
  }
  POP_RANGE;
  auto end = std::chrono::high_resolution_clock::now();

  // ---- Join streams ONCE at the end (no serializing in the loop) ----
  PUSH_RANGE("Synchronize", 3);
  for (int i=0; i<n_streams; ++i) CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  CUDA_CHECK(cudaDeviceSynchronize());
  POP_RANGE;

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "GEMMs on Rank " << rank << " completed in " << elapsed.count() << " seconds" << std::endl;
  std::cout << "GEMM rate on Rank " << rank << ": " << (double)ns*(double)nts/elapsed.count() << " GEMMs/second" << std::endl;
  std::cout << "FLOP rate on Rank " << rank << ": " << n_repeat * total_flop_count / elapsed.count() / 1e9 << " Giga FLOPs/second" << std::endl;

  // ---- Cleanup ----
  for (int i=0; i<n_streams; ++i) {
    cublasDestroy(handles[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(g_stij);
  cudaFree(VQ);
  cudaFree(Y);
  cudaFree(g_state);
  cudaFree(vq_state);
  cudaFree(y_state);
}