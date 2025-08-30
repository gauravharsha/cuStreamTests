#include <nvtx3/nvToolsExt.h>
#include "cu_routines.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
  int color_id = cid; \
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


void streams_and_handles() {
  const int n_streams = 2;
  PUSH_RANGE("Initialize", 0);
  const size_t ns = 2;
  const size_t nao = 54;
  const size_t naux = 638;
  const size_t nts = 10;
  size_t ntnaux2 = nts * naux * naux;
  size_t ntnao2 = nts * nao * nao;
  size_t nauxnao2 = naux * nao * nao;

  // allocate memory for stuff
  cuda_complex* Pqk0; // polarization
  cuda_complex* g_stij;
  cuda_complex* VQ;
  cuda_complex* Y1;
  cuda_complex* Y2;
  if (cudaMalloc(&Pqk0, ntnaux2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("failure allocating Pq0");
  if (cudaMalloc(&g_stij, ns * ntnao2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("failure allocating g_tij on device");
  if (cudaMalloc(&VQ, nauxnao2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("failure allocating VQ on device");
  if (cudaMalloc(&Y1, nauxnao2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("failure allocating Y1 on device");
  if (cudaMalloc(&Y2, nauxnao2 * sizeof(cuda_complex)) != cudaSuccess)
    throw std::runtime_error("failure allocating Y2 on device");

  // Random initialize
  curandState* pq_state;
  curandState* g_state;
  curandState* vq_state;
  curandState* y1_state;
  curandState* y2_state;

  if (cudaMalloc(&pq_state, ntnaux2 * sizeof(curandState)))
    throw std::runtime_error("failure allocating Pq0");
  if (cudaMalloc(&g_state, ns * ntnao2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("failure allocating g_tij on device");
  if (cudaMalloc(&vq_state, nauxnao2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("failure allocating VQ on device");
  if (cudaMalloc(&y1_state, nauxnao2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("failure allocating Y1 on device");
  if (cudaMalloc(&y2_state, nauxnao2 * sizeof(curandState)) != cudaSuccess)
    throw std::runtime_error("failure allocating Y2 on device");

  int threads = 256;
  int blocks = (ntnaux2 + threads - 1) / threads;
  init_random_complex<<<blocks, threads>>>(Pqk0, pq_state, ntnaux2, time(NULL));
  blocks = (ns * ntnao2 + threads - 1) / threads;
  init_random_complex<<<blocks, threads>>>(g_stij, g_state, ntnao2, time(NULL));
  blocks = (nauxnao2 + threads - 1) / threads;
  init_random_complex<<<blocks, threads>>>(VQ, vq_state, nauxnao2, time(NULL));
  blocks = (nauxnao2 + threads - 1) / threads;
  init_random_complex<<<blocks, threads>>>(Y1, y1_state, nauxnao2, time(NULL));
  init_random_complex<<<blocks, threads>>>(Y2, y2_state, nauxnao2, time(NULL));
  std::cout << "initialized completed" << std::endl;
  POP_RANGE;


  // Create streams and handles
  PUSH_RANGE("Create 2 streams and handles", 1);
  std::vector<cudaStream_t> _streams(n_streams);
  std::vector<cublasHandle_t> _handles(n_streams);
  for (int i=0; i<n_streams; i++) {
    if (cublasCreate(&_handles[i]) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(i) + ": error initializing cublas");
    if (cudaStreamCreate(&_streams[i]) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(i) + ": error initializing cuda Stream");
    cublasSetStream(_handles[i], _streams[i]);
  }
  POP_RANGE;


  // Perform Batched DGEMM
  PUSH_RANGE("Perform 2 GEMM calls", 3);
  cuda_complex  one     = cu_type_map<cxx_complex>::cast(1., 0.);
  cuda_complex  zero    = cu_type_map<cxx_complex>::cast(0., 0.);
  cuda_complex  m1      = cu_type_map<cxx_complex>::cast(-1., 0.);
  // cuda_complex* Y1t_Qin = X1t_tmQ_;  // name change, reuse memory
  // cuda_complex* Y2t_inP = X2t_Ptm_;  // name change, reuse memory
  size_t nauxnao = naux * nao;
  size_t nao2 = nao * nao;
  int st0 = 0;
  int st1 = 1;
  if (GEMM_STRIDED_BATCHED(_handles[0], CUBLAS_OP_N, CUBLAS_OP_N, nao, nauxnao, nao, &one, g_stij + st0 * nao2, nao,
                          nao2, VQ, nao, 0, &zero, Y1, nao, nauxnao2, 1) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
  }
  if (GEMM_STRIDED_BATCHED(_handles[1], CUBLAS_OP_N, CUBLAS_OP_N, nao, nauxnao, nao, &one, g_stij + st1 * nao2, nao,
                          nao2, VQ, nao, 0, &zero, Y2, nao, nauxnao2, 1) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
  }
  POP_RANGE;
}

