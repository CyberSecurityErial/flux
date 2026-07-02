#include "performance_utils.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using myflux::perf::CudaEventTimer;

void
check_cublas(cublasStatus_t status, const char *expr, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error at " << file << ":" << line << ": " << expr
              << " failed with status " << static_cast<int>(status) << "\n";
    std::abort();
  }
}

void
check_nccl(ncclResult_t status, const char *expr, const char *file, int line) {
  if (status != ncclSuccess) {
    std::cerr << "NCCL error at " << file << ":" << line << ": " << expr
              << " failed: " << ncclGetErrorString(status) << "\n";
    std::abort();
  }
}

#define MYFLUX_CHECK_CUBLAS(expr) check_cublas((expr), #expr, __FILE__, __LINE__)
#define MYFLUX_CHECK_NCCL(expr) check_nccl((expr), #expr, __FILE__, __LINE__)

struct MyfluxSm80GemmRsLaunchParams {
  int m = 0;
  int n = 0;
  int k = 0;
  int rank = 0;
  int world_size = 1;
  const void *input = nullptr;
  const void *weight = nullptr;
  void **output_scatter_ptrs = nullptr;
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
};

extern "C" size_t myflux_sm80_gemmrs_get_workspace_size(
    int m,
    int n,
    int k,
    int world_size) __attribute__((weak));

extern "C" int myflux_sm80_gemmrs_launch(
    const MyfluxSm80GemmRsLaunchParams *params,
    cudaStream_t stream) __attribute__((weak));

class ThreadBarrier {
 public:
  explicit ThreadBarrier(int count) : count_(count) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int generation = generation_;
    if (++arrived_ == count_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return generation != generation_; });
  }

 private:
  const int count_;
  int arrived_ = 0;
  int generation_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
};

struct Options {
  int m = 4096;
  int n = 12288;
  int k = 6144;
  int world_size = 8;
  int warmup = 10;
  int iters = 50;
  int device_start = 0;
  bool run_baseline = true;
  bool run_fused = true;
  bool clear_fused_output = true;
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
  if (arg == name) {
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string("missing value for ") + name);
    }
    return std::stoi(argv[++i]);
  }
  throw std::invalid_argument(std::string("unhandled integer argument ") + arg);
}

void
print_usage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "  --m <int>                 GEMM M, must be divisible by world size\n"
            << "  --n <int>                 GEMM N\n"
            << "  --k <int>                 local GEMM K per rank\n"
            << "  --world-size <int>        number of local GPUs, default 8\n"
            << "  --device-start <int>      first CUDA device ordinal, default 0\n"
            << "  --warmup <int>            warmup iterations, default 10\n"
            << "  --iters <int>             timed iterations, default 50\n"
            << "  --mode baseline|fused|both\n"
            << "  --no-fused-clear          do not clear fused output before launch\n";
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
    } else if (arg == "--mode") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for --mode");
      }
      std::string mode(argv[++i]);
      opt.run_baseline = mode == "baseline" || mode == "both";
      opt.run_fused = mode == "fused" || mode == "both";
      if (!opt.run_baseline && !opt.run_fused) {
        throw std::invalid_argument("unsupported --mode: " + mode);
      }
    } else if (starts_with(arg, "--mode=")) {
      std::string mode = arg.substr(std::strlen("--mode="));
      opt.run_baseline = mode == "baseline" || mode == "both";
      opt.run_fused = mode == "fused" || mode == "both";
      if (!opt.run_baseline && !opt.run_fused) {
        throw std::invalid_argument("unsupported --mode: " + mode);
      }
    } else if (arg == "--no-fused-clear") {
      opt.clear_fused_output = false;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opt.world_size <= 0 || opt.world_size > 16) {
    throw std::invalid_argument("world size must be in (0, 16]");
  }
  if (opt.m % opt.world_size != 0) {
    throw std::invalid_argument("m must be divisible by world size");
  }
  if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.iters <= 0 || opt.warmup < 0) {
    throw std::invalid_argument("m/n/k/iters must be positive and warmup must be non-negative");
  }
  return opt;
}

struct RankBuffers {
  void *input = nullptr;             // 本 rank 的 GEMM 输入 A，形状 [M, K]
  void *weight = nullptr;            // 本 rank 的 GEMM 权重 W，形状 [N, K]
  void *full_output = nullptr;       // baseline GEMM 完整输出，形状 [M, N]
  void *baseline_output = nullptr;   // NCCL reduce-scatter 后属于本 rank 的输出 shard
  void *fused_output = nullptr;      // fused GEMM+RS 写入的本 rank 输出 shard
  void *fused_workspace = nullptr;   // fused kernel 需要的额外 workspace
  size_t fused_workspace_bytes = 0;  // fused workspace 字节数
};

size_t
bytes_for_half(int64_t elements) {
  return static_cast<size_t>(elements) * sizeof(__half);
}

void
enable_peer_access(const Options &opt) {
  for (int rank = 0; rank < opt.world_size; ++rank) {
    int device = opt.device_start + rank;
    MYFLUX_CHECK_CUDA(cudaSetDevice(device));
    for (int peer_rank = 0; peer_rank < opt.world_size; ++peer_rank) {
      if (peer_rank == rank) {
        continue;
      }
      int peer = opt.device_start + peer_rank;
      int can_access = 0;
      MYFLUX_CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, device, peer));
      if (!can_access) {
        continue;
      }
      cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else {
        MYFLUX_CHECK_CUDA(status);
      }
    }
  }
}

std::vector<RankBuffers>
allocate_buffers(const Options &opt) {
  std::vector<RankBuffers> buffers(opt.world_size);
  int m_per_rank = opt.m / opt.world_size;

  size_t input_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.k);
  size_t weight_bytes = bytes_for_half(static_cast<int64_t>(opt.n) * opt.k);
  size_t full_output_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.n);
  size_t shard_output_bytes = bytes_for_half(static_cast<int64_t>(m_per_rank) * opt.n);

  std::cout << "per-rank buffers: input=" << myflux::perf::format_bytes(input_bytes)
            << " weight=" << myflux::perf::format_bytes(weight_bytes)
            << " full_output=" << myflux::perf::format_bytes(full_output_bytes)
            << " shard_output=" << myflux::perf::format_bytes(shard_output_bytes) << "\n";

  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    auto &buf = buffers[rank];
    MYFLUX_CHECK_CUDA(cudaMalloc(&buf.input, input_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(&buf.weight, weight_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(&buf.full_output, full_output_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(&buf.baseline_output, shard_output_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(&buf.fused_output, shard_output_bytes));

    MYFLUX_CHECK_CUDA(cudaMemset(buf.input, 1, input_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.weight, 2, weight_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.full_output, 0, full_output_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.baseline_output, 0, shard_output_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.fused_output, 0, shard_output_bytes));

    if (myflux_sm80_gemmrs_get_workspace_size != nullptr) {
      buf.fused_workspace_bytes =
          myflux_sm80_gemmrs_get_workspace_size(opt.m, opt.n, opt.k, opt.world_size);
      if (buf.fused_workspace_bytes != 0) {
        MYFLUX_CHECK_CUDA(cudaMalloc(&buf.fused_workspace, buf.fused_workspace_bytes));
        MYFLUX_CHECK_CUDA(cudaMemset(buf.fused_workspace, 0, buf.fused_workspace_bytes));
      }
    }
  }
  return buffers;
}

void
free_buffers(const Options &opt, std::vector<RankBuffers> &buffers) {
  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    cudaFree(buffers[rank].input);
    cudaFree(buffers[rank].weight);
    cudaFree(buffers[rank].full_output);
    cudaFree(buffers[rank].baseline_output);
    cudaFree(buffers[rank].fused_output);
    cudaFree(buffers[rank].fused_workspace);
  }
}

void
run_baseline_once(
    const Options &opt,
    RankBuffers &buf,
    cublasHandle_t cublas,
    ncclComm_t comm,
    cudaStream_t stream) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // Row-major C[M, N] = A[M, K] * W[N, K]^T is expressed as column-major
  // C^T[N, M] = W^T[N, K] * A^T[K, M].
  MYFLUX_CHECK_CUBLAS(cublasGemmEx(
      cublas,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      opt.n,
      opt.m,
      opt.k,
      &alpha,
      buf.weight,
      CUDA_R_16F,
      opt.k,
      buf.input,
      CUDA_R_16F,
      opt.k,
      &beta,
      buf.full_output,
      CUDA_R_16F,
      opt.n,
      CUBLAS_COMPUTE_32F_FAST_16F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  int recv_count = (opt.m / opt.world_size) * opt.n;
  MYFLUX_CHECK_NCCL(ncclReduceScatter(
      buf.full_output,
      buf.baseline_output,
      recv_count,
      ncclHalf,
      ncclSum,
      comm,
      stream));
}

void
run_fused_once(
    const Options &opt,
    int rank,
    RankBuffers &buf,
    void **fused_output_ptrs,
    cudaStream_t stream) {
  if (myflux_sm80_gemmrs_launch == nullptr) {
    return;
  }

  if (opt.clear_fused_output) {
    size_t output_bytes = bytes_for_half(static_cast<int64_t>(opt.m / opt.world_size) * opt.n);
    MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.fused_output, 0, output_bytes, stream));
  }

  MyfluxSm80GemmRsLaunchParams params;
  params.m = opt.m;
  params.n = opt.n;
  params.k = opt.k;
  params.rank = rank;
  params.world_size = opt.world_size;
  params.input = buf.input;
  params.weight = buf.weight;
  params.output_scatter_ptrs = fused_output_ptrs;
  params.workspace = buf.fused_workspace;
  params.workspace_bytes = buf.fused_workspace_bytes;

  int status = myflux_sm80_gemmrs_launch(&params, stream);
  if (status != 0) {
    throw std::runtime_error("myflux_sm80_gemmrs_launch failed with status " +
                             std::to_string(status));
  }
}

float
time_loop(
    int warmup,
    int iters,
    cudaStream_t stream,
    ThreadBarrier &barrier,
    const std::function<void()> &fn) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  // 仅用于 benchmark：确保所有 rank 完成 warmup 后再一起开始计时。
  barrier.wait();

  CudaEventTimer timer;
  timer.start(stream);
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  float elapsed_ms = timer.stop(stream);
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  // 仅用于 benchmark：避免某些 rank 提前进入下一段测试或销毁资源。
  barrier.wait();
  return elapsed_ms / static_cast<float>(iters);
}

}  // namespace

int
main(int argc, char **argv) {
  Options opt = parse_options(argc, argv);

  int device_count = 0;
  MYFLUX_CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (opt.device_start + opt.world_size > device_count) {
    std::cerr << "Requested devices [" << opt.device_start << ", "
              << (opt.device_start + opt.world_size) << "), but only " << device_count
              << " CUDA devices are visible\n";
    return 1;
  }

  std::cout << "myflux sm80 gemm+rs case\n";
  std::cout << "shape: M=" << opt.m << " N=" << opt.n << " localK=" << opt.k
            << " world_size=" << opt.world_size << "\n";
  std::cout << "iters: warmup=" << opt.warmup << " timed=" << opt.iters << "\n";

  enable_peer_access(opt);
  auto buffers = allocate_buffers(opt);

  std::vector<int> devices(opt.world_size);
  for (int rank = 0; rank < opt.world_size; ++rank) {
    devices[rank] = opt.device_start + rank;
  }

  std::vector<ncclComm_t> comms(opt.world_size);
  MYFLUX_CHECK_NCCL(ncclCommInitAll(comms.data(), opt.world_size, devices.data()));

  std::vector<void *> fused_output_ptrs(opt.world_size, nullptr);
  for (int rank = 0; rank < opt.world_size; ++rank) {
    fused_output_ptrs[rank] = buffers[rank].fused_output;
  }

  std::vector<float> baseline_ms(opt.world_size, 0.0f);
  std::vector<float> fused_ms(opt.world_size, 0.0f);
  ThreadBarrier barrier(opt.world_size);
  std::vector<std::exception_ptr> errors(opt.world_size);

  auto worker = [&](int rank) {
    try {
      MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));

      cudaStream_t stream = nullptr;
      MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

      cublasHandle_t cublas = nullptr;
      MYFLUX_CHECK_CUBLAS(cublasCreate(&cublas));
      MYFLUX_CHECK_CUBLAS(cublasSetStream(cublas, stream));
      MYFLUX_CHECK_CUBLAS(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

      if (opt.run_baseline) {
        baseline_ms[rank] = time_loop(opt.warmup, opt.iters, stream, barrier, [&] {
          run_baseline_once(opt, buffers[rank], cublas, comms[rank], stream);
        });
      }

      if (opt.run_fused && myflux_sm80_gemmrs_launch != nullptr) {
        fused_ms[rank] = time_loop(opt.warmup, opt.iters, stream, barrier, [&] {
          run_fused_once(opt, rank, buffers[rank], fused_output_ptrs.data(), stream);
        });
      } else if (opt.run_fused && rank == 0) {
        std::cout << "fused launcher symbol is not linked; skipping fused timing\n";
      }

      MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
      MYFLUX_CHECK_CUBLAS(cublasDestroy(cublas));
      MYFLUX_CHECK_CUDA(cudaStreamDestroy(stream));
    } catch (...) {
      errors[rank] = std::current_exception();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(opt.world_size);
  for (int rank = 0; rank < opt.world_size; ++rank) {
    threads.emplace_back(worker, rank);
  }
  for (auto &thread : threads) {
    thread.join();
  }

  for (auto &error : errors) {
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
  }

  if (opt.run_baseline) {
    myflux::perf::print_rank_times("baseline_gemm_then_nccl_rs", baseline_ms);
  }
  if (opt.run_fused && myflux_sm80_gemmrs_launch != nullptr) {
    myflux::perf::print_rank_times("fused_sm80_gemmrs", fused_ms);
  }

  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_NCCL(ncclCommDestroy(comms[rank]));
  }
  free_buffers(opt, buffers);
  return 0;
}
