// 分离式 baseline：plain CUTLASS GEMM 写完整输出，再调用 NCCL ReduceScatter。
// 这个 case 使用 MPI 一进程一 GPU，不接 fused GEMMRS kernel。
#include "sm80_gemmrs.cuh"

#include "mpi_cuda_ipc_utils.h"

#include <nccl.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Element = myflux::cutlass_utils::Element;

void
check_nccl(ncclResult_t status, const char *expr, const char *file, int line) {
  if (status != ncclSuccess) {
    std::cerr << "NCCL error at " << file << ":" << line << ": " << expr
              << " failed: " << ncclGetErrorString(status) << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

#define MYFLUX_BENCH_CHECK_NCCL(expr) check_nccl((expr), #expr, __FILE__, __LINE__)

struct Options {
  int m = 4096;
  int n = 12288;
  int k = 6144;
  int world_size = 8;
  int device_start = 0;
  int warmup = 10;
  int iters = 50;
  int avail_sms = -1;
};

bool
starts_with(const std::string &s, const char *prefix) {
  size_t n = std::strlen(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}

int
parse_int_arg(int argc, char **argv, int &i, const std::string &arg, const char *name) {
  std::string prefix = std::string(name) + "=";
  if (starts_with(arg, prefix.c_str())) {
    return std::stoi(arg.substr(prefix.size()));
  }
  if (i + 1 >= argc) {
    throw std::invalid_argument(std::string("missing value for ") + name);
  }
  return std::stoi(argv[++i]);
}

void
print_usage(const char *program) {
  std::cout << "Usage: mpirun -np <world_size> " << program << " [options]\n"
            << "  --m <int>             GEMM M, must be divisible by world size\n"
            << "  --n <int>             GEMM N\n"
            << "  --k <int>             local GEMM K per rank\n"
            << "  --world-size <int>    MPI world size, default 8\n"
            << "  --device-start <int>  first CUDA device ordinal, default 0\n"
            << "  --warmup <int>        warmup iterations, default 10\n"
            << "  --iters <int>         timed iterations, default 50\n"
            << "  --avail-sms <int>     CUTLASS avail_sms, default -1\n";
}

Options
parse_options(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--m" || starts_with(arg, "--m=")) {
      opt.m = parse_int_arg(argc, argv, i, arg, "--m");
    } else if (arg == "--n" || starts_with(arg, "--n=")) {
      opt.n = parse_int_arg(argc, argv, i, arg, "--n");
    } else if (arg == "--k" || starts_with(arg, "--k=")) {
      opt.k = parse_int_arg(argc, argv, i, arg, "--k");
    } else if (arg == "--world-size" || starts_with(arg, "--world-size=")) {
      opt.world_size = parse_int_arg(argc, argv, i, arg, "--world-size");
    } else if (arg == "--device-start" || starts_with(arg, "--device-start=")) {
      opt.device_start = parse_int_arg(argc, argv, i, arg, "--device-start");
    } else if (arg == "--warmup" || starts_with(arg, "--warmup=")) {
      opt.warmup = parse_int_arg(argc, argv, i, arg, "--warmup");
    } else if (arg == "--iters" || starts_with(arg, "--iters=")) {
      opt.iters = parse_int_arg(argc, argv, i, arg, "--iters");
    } else if (arg == "--avail-sms" || starts_with(arg, "--avail-sms=")) {
      opt.avail_sms = parse_int_arg(argc, argv, i, arg, "--avail-sms");
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opt.world_size <= 0 || opt.world_size > myflux::kMaxWorldSize) {
    throw std::invalid_argument("world size must be in (0, 8]");
  }
  if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.iters <= 0 || opt.warmup < 0) {
    throw std::invalid_argument("m/n/k/iters must be positive and warmup must be non-negative");
  }
  if (opt.m % opt.world_size != 0) {
    throw std::invalid_argument("m must be divisible by world size");
  }
  return opt;
}

size_t
bytes_for_half(int64_t elements) {
  return static_cast<size_t>(elements) * sizeof(Element);
}

struct RankBuffers {
  Element *input = nullptr;
  Element *weight = nullptr;
  Element *full_output = nullptr;
  Element *rs_output = nullptr;
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
};

void
malloc_bytes(void **ptr, size_t bytes) {
  MYFLUX_CHECK_CUDA(cudaMalloc(ptr, bytes));
}

RankBuffers
allocate_rank_buffers(const Options &opt, cudaStream_t stream) {
  RankBuffers buf;
  int m_per_rank = opt.m / opt.world_size;
  size_t input_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.k);
  size_t weight_bytes = bytes_for_half(static_cast<int64_t>(opt.n) * opt.k);
  size_t full_output_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.n);
  size_t rs_output_bytes = bytes_for_half(static_cast<int64_t>(m_per_rank) * opt.n);

  malloc_bytes(reinterpret_cast<void **>(&buf.input), input_bytes);
  malloc_bytes(reinterpret_cast<void **>(&buf.weight), weight_bytes);
  malloc_bytes(reinterpret_cast<void **>(&buf.full_output), full_output_bytes);
  malloc_bytes(reinterpret_cast<void **>(&buf.rs_output), rs_output_bytes);

  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.input, 1, input_bytes, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.weight, 2, weight_bytes, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.full_output, 0, full_output_bytes, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.rs_output, 0, rs_output_bytes, stream));

  auto args = myflux::sm80_gemmrs::make_plain_gemm128_args(
      opt.m, opt.n, opt.k, buf.input, buf.weight, buf.full_output, opt.avail_sms);
  buf.workspace_bytes = myflux::sm80_gemmrs::plain_gemm128_workspace_size(args);
  if (buf.workspace_bytes != 0) {
    malloc_bytes(&buf.workspace, buf.workspace_bytes);
    MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.workspace, 0, buf.workspace_bytes, stream));
  }
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  return buf;
}

void
free_rank_buffers(RankBuffers &buf) {
  cudaFree(buf.workspace);
  cudaFree(buf.rs_output);
  cudaFree(buf.full_output);
  cudaFree(buf.weight);
  cudaFree(buf.input);
}

void
run_gemm_then_rs_once(const Options &opt, RankBuffers &buf, ncclComm_t comm, cudaStream_t stream) {
  auto args = myflux::sm80_gemmrs::make_plain_gemm128_args(
      opt.m, opt.n, opt.k, buf.input, buf.weight, buf.full_output, opt.avail_sms);
  myflux::sm80_gemmrs::run_plain_gemm128(args, buf.workspace, stream);

  int recv_count = (opt.m / opt.world_size) * opt.n;
  MYFLUX_BENCH_CHECK_NCCL(ncclReduceScatter(
      buf.full_output, buf.rs_output, recv_count, ncclHalf, ncclSum, comm, stream));
}

float
time_loop(const Options &opt, RankBuffers &buf, ncclComm_t comm, cudaStream_t stream, MPI_Comm mpi_comm) {
  for (int i = 0; i < opt.warmup; ++i) {
    run_gemm_then_rs_once(opt, buf, comm, stream);
  }
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  MYFLUX_CASE_CHECK_MPI(MPI_Barrier(mpi_comm));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  MYFLUX_CHECK_CUDA(cudaEventCreate(&start));
  MYFLUX_CHECK_CUDA(cudaEventCreate(&stop));
  MYFLUX_CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < opt.iters; ++i) {
    run_gemm_then_rs_once(opt, buf, comm, stream);
  }
  MYFLUX_CHECK_CUDA(cudaEventRecord(stop, stream));
  MYFLUX_CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  MYFLUX_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  MYFLUX_CHECK_CUDA(cudaEventDestroy(start));
  MYFLUX_CHECK_CUDA(cudaEventDestroy(stop));
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  MYFLUX_CASE_CHECK_MPI(MPI_Barrier(mpi_comm));
  return elapsed_ms / static_cast<float>(opt.iters);
}

void
print_summary(const Options &opt, const std::vector<float> &rank_ms) {
  auto minmax = std::minmax_element(rank_ms.begin(), rank_ms.end());
  float avg = std::accumulate(rank_ms.begin(), rank_ms.end(), 0.0f) /
              static_cast<float>(rank_ms.size());
  float max_ms = *minmax.second;
  double gemm_flops = 2.0 * static_cast<double>(opt.m) * opt.n * opt.k;
  double tflops_per_rank = gemm_flops / (max_ms * 1.0e-3) / 1.0e12;

  std::cout << "gemm_then_nccl_rs per-rank ms:";
  for (size_t i = 0; i < rank_ms.size(); ++i) {
    std::cout << " [" << i << "]=" << std::fixed << std::setprecision(3) << rank_ms[i];
  }
  std::cout << "\n";
  std::cout << "summary: min=" << std::fixed << std::setprecision(3) << *minmax.first
            << " ms max=" << max_ms << " ms avg=" << avg
            << " ms per_rank_gemm_tflops_by_max_ms=" << tflops_per_rank << "\n";
}

}  // namespace

int
main(int argc, char **argv) {
  MYFLUX_CASE_CHECK_MPI(MPI_Init(&argc, &argv));
  int rank = 0;
  int nranks = 1;
  MYFLUX_CASE_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MYFLUX_CASE_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  try {
    Options opt = parse_options(argc, argv);
    if (opt.world_size != nranks) {
      throw std::invalid_argument("--world-size must match MPI_COMM_WORLD size");
    }

    int local_rank = myflux::case_utils::mpi_local_rank(rank);
    int local_world_size = myflux::case_utils::mpi_local_world_size(nranks);
    myflux::case_utils::set_flux_rank_env(local_rank, local_world_size);

    int device_count = 0;
    MYFLUX_CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (opt.device_start + local_rank >= device_count) {
      throw std::runtime_error("not enough visible CUDA devices for local rank");
    }
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + local_rank));

    if (rank == 0) {
      std::cout << "myflux sm80 plain GEMM + NCCL ReduceScatter MPI bench\n";
      std::cout << "M=" << opt.m << " N=" << opt.n << " localK=" << opt.k
                << " world_size=" << opt.world_size << " warmup=" << opt.warmup
                << " iters=" << opt.iters << "\n";
    }

    cudaStream_t stream = nullptr;
    MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    ncclUniqueId nccl_id;
    if (rank == 0) {
      MYFLUX_BENCH_CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }
    MYFLUX_CASE_CHECK_MPI(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm = nullptr;
    MYFLUX_BENCH_CHECK_NCCL(ncclCommInitRank(&comm, nranks, nccl_id, rank));

    RankBuffers buf = allocate_rank_buffers(opt, stream);
    MYFLUX_CASE_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    float rank_ms = time_loop(opt, buf, comm, stream, MPI_COMM_WORLD);

    auto all_ms = myflux::case_utils::gather_rank_ms(rank_ms, rank, nranks, MPI_COMM_WORLD);
    if (rank == 0) {
      print_summary(opt, all_ms);
    }

    MYFLUX_CASE_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    free_rank_buffers(buf);
    MYFLUX_BENCH_CHECK_NCCL(ncclCommDestroy(comm));
    MYFLUX_CHECK_CUDA(cudaStreamDestroy(stream));
  } catch (const std::exception &ex) {
    std::cerr << "rank " << rank << " failed: " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MYFLUX_CASE_CHECK_MPI(MPI_Finalize());
  return 0;
}
