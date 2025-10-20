#include <stdio.h>
#include <exception>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <string>
#include <cstring>
#include <unordered_map>
#include "cu_routines.h"
#include "cuda_check.h"
#include <mpi.h>

namespace {

inline void print_help(const char* prog) {
  if (!prog) prog = "main.exe";
  std::cout <<
    "Usage: " << prog << " [--ns <u>] [--nao <u>] [--naux <u>] [--nts <u>] [--nstreams <int>]\n"
    "Options:\n"
    "  --ns <u>         Number of spins (default 2)\n"
    "  --nao <u>        Number of AOs (default 54)\n"
    "  --naux <u>       Number of aux functions (default 638)\n"
    "  --nts <u>        Number of time slices (default 10)\n"
    "  --nstreams <int> Number of CUDA streams per rank (default 1)\n"
    "  --nt_batch <int> Number of tau points to process in a batch (default 1)\n"
    "  -h, --help       Show this help and exit\n";
}

template <typename UInt>
bool parse_uint(const char* s, UInt& out) {
  try {
    if (!s || *s == '\0') return false;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<UInt>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_int(const char* s, int& out) {
  try {
    if (!s || *s == '\0') return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
  } catch (...) {
    return false;
  }
}

struct Config {
  size_t ns  = 2;
  size_t nao = 54;
  size_t naux = 638;
  size_t nts = 10;
  size_t nt_batch = 1;
  int n_streams = 1;
};

bool parse_args(int argc, char** argv, Config& cfg, bool& show_help) {
  show_help = false;
  for (int i = 1; i < argc; ++i) {
    std::string key(argv[i]);
    if (key == "-h" || key == "--help") {
      show_help = true;
      return true;
    }
    auto need_val = [&](int i)->bool{
      if (i + 1 >= argc) {
        std::cerr << "Missing value for option '" << key << "'\n";
        return false;
      }
      return true;
    };

    if (key == "--ns") {
      if (!need_val(i)) return false;
      size_t v; if (!parse_uint(argv[++i], v)) { std::cerr << "Invalid --ns\n"; return false; }
      cfg.ns = v;
    } else if (key == "--nao") {
      if (!need_val(i)) return false;
      size_t v; if (!parse_uint(argv[++i], v)) { std::cerr << "Invalid --nao\n"; return false; }
      cfg.nao = v;
    } else if (key == "--naux") {
      if (!need_val(i)) return false;
      size_t v; if (!parse_uint(argv[++i], v)) { std::cerr << "Invalid --naux\n"; return false; }
      cfg.naux = v;
    } else if (key == "--nts") {
      if (!need_val(i)) return false;
      size_t v; if (!parse_uint(argv[++i], v)) { std::cerr << "Invalid --nts\n"; return false; }
      cfg.nts = v;
    } else if (key == "--nstreams" || key == "--nstream" || key == "--n_streams") {
      if (!need_val(i)) return false;
      int v; if (!parse_int(argv[++i], v) || v <= 0) { std::cerr << "Invalid --nstreams\n"; return false; }
      cfg.n_streams = v;
    } else if (key == "--nt_batch" || key == "--ntbatch") {
      if (!need_val(i)) return false;
      size_t v; if (!parse_uint(argv[++i], v)) { std::cerr << "Invalid --nt_batch\n"; return false; }
      cfg.nt_batch = v;
    } else {
      std::cerr << "Unknown option: " << key << "\n";
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Defaults
  Config cfg;
  bool show_help = false;
  if (!parse_args(argc, argv, cfg, show_help)) {
    if (world_rank == 0) print_help(argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (show_help) {
    if (world_rank == 0) print_help(argv[0]);
    MPI_Finalize();
    return 0;
  }

  // Check for GPU devices (optionally accept num_devices in the future)
  int num_devices = 1;
  check_for_cuda(num_devices);

  MPI_Barrier(MPI_COMM_WORLD);
  auto start = std::chrono::high_resolution_clock::now();
  try {
    if (world_rank == 0) {
      std::cout << "Config: ns=" << cfg.ns
                << ", nao=" << cfg.nao
                << ", naux=" << cfg.naux
                << ", nts=" << cfg.nts
                << ", n_streams=" << cfg.n_streams
                << ", nt_batch=" << cfg.nt_batch << std::endl;
    }

    // Your existing entry point
    streams_and_handles(world_rank, cfg.ns, cfg.nao, cfg.naux, cfg.nts, cfg.n_streams, cfg.nt_batch);
  } catch (const std::exception& e) {
    if (world_rank == 0) fprintf(stderr, "Error: %s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  if (!world_rank) {
    std::cout << "Completed in " << duration_ms << " ms" << std::endl;
  }
  MPI_Finalize();
  return 0;
}
