#include "sm80_gemmrs.cuh"

#ifdef MYFLUX_CHECK_CUDA
#undef MYFLUX_CHECK_CUDA
#endif

#include "performance_utils.h"

#include <cuda_runtime_api.h>

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

using Element = myflux::cutlass_utils::Element;
using FusedGemmRsConfig = myflux::sm80_gemmrs::
    GemmRsTileConfig<myflux::sm80_gemmrs::ThreadblockShape128x128x32, true>;
using myflux::perf::CudaEventTimer;

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
  int avail_sms = -1;
  bool clear_fused_output = true;
};

struct RankBuffers {
  Element *input = nullptr;         // 本 rank 的 GEMM 输入 A，形状 [M, K]
  Element *weight = nullptr;        // 本 rank 的 GEMM 权重 W，形状 [N, K]
  Element *fused_output = nullptr;  // 完整 [M, N] buffer，前 [M/world_size, N] 是本 rank 结果
  void *workspace = nullptr;        // CUTLASS fused GEMMRS workspace
  size_t workspace_bytes = 0;       // fused workspace 字节数
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
            << "  --m <int>             GEMM M, must be divisible by world size\n"
            << "  --n <int>             GEMM N\n"
            << "  --k <int>             local GEMM K per rank\n"
            << "  --world-size <int>    local GPU count, default 8\n"
            << "  --device-start <int>  first CUDA device ordinal, default 0\n"
            << "  --warmup <int>        warmup iterations, default 10\n"
            << "  --iters <int>         timed iterations, default 50\n"
            << "  --avail-sms <int>     CUTLASS avail_sms, default -1\n"
            << "  --no-fused-clear      skip per-iteration output clear for raw kernel timing\n";
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
    } else if (arg == "--no-fused-clear") {
      opt.clear_fused_output = false;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opt.world_size <= 0 || opt.world_size > myflux::kMaxWorldSize) {
    throw std::invalid_argument("world size must be in (0, kMaxWorldSize]");
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

myflux::sm80_gemmrs::GemmRsLaunchParams
make_fused_params(
    const Options &opt,
    int rank,
    const RankBuffers &buf,
    Element **fused_output_ptrs) {
  myflux::sm80_gemmrs::GemmRsLaunchParams params;
  params.problem = {opt.m, opt.n, opt.k};
  params.ptr_a = buf.input;
  params.ptr_b = buf.weight;
  params.output_scatter_ptrs = fused_output_ptrs;
  params.rank = rank;
  params.world_size = opt.world_size;
  params.alpha = 1.0f;
  params.avail_sms = opt.avail_sms;
  return params;
}

std::vector<RankBuffers>
allocate_buffers(const Options &opt) {
  std::vector<RankBuffers> buffers(opt.world_size);

  size_t input_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.k);
  size_t weight_bytes = bytes_for_half(static_cast<int64_t>(opt.n) * opt.k);
  size_t fused_output_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.n);

  std::cout << "per-rank buffers: input=" << myflux::perf::format_bytes(input_bytes)
            << " weight=" << myflux::perf::format_bytes(weight_bytes)
            << " fused_output=" << myflux::perf::format_bytes(fused_output_bytes) << "\n";

  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    auto &buf = buffers[rank];
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.input), input_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.weight), weight_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.fused_output), fused_output_bytes));

    MYFLUX_CHECK_CUDA(cudaMemset(buf.input, 1, input_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.weight, 2, weight_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.fused_output, 0, fused_output_bytes));
  }
  return buffers;
}

void
allocate_workspaces(
    const Options &opt,
    std::vector<RankBuffers> &buffers,
    Element **fused_output_ptrs) {
  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    auto args = myflux::sm80_gemmrs::make_gemmrs_args<FusedGemmRsConfig>(
        make_fused_params(opt, rank, buffers[rank], fused_output_ptrs));
    buffers[rank].workspace_bytes =
        myflux::sm80_gemmrs::gemmrs_workspace_size<FusedGemmRsConfig>(args);
    if (buffers[rank].workspace_bytes != 0) {
      MYFLUX_CHECK_CUDA(cudaMalloc(&buffers[rank].workspace, buffers[rank].workspace_bytes));
      MYFLUX_CHECK_CUDA(cudaMemset(buffers[rank].workspace, 0, buffers[rank].workspace_bytes));
    }
  }
}

void
free_buffers(const Options &opt, std::vector<RankBuffers> &buffers) {
  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    cudaFree(buffers[rank].workspace);
    cudaFree(buffers[rank].fused_output);
    cudaFree(buffers[rank].weight);
    cudaFree(buffers[rank].input);
  }
}

void
clear_fused_output_once(const Options &opt, RankBuffers &buf, cudaStream_t stream) {
  size_t output_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.n);
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.fused_output, 0, output_bytes, stream));
}

void
launch_fused_once(
    const Options &opt,
    int rank,
    RankBuffers &buf,
    Element **fused_output_ptrs,
    cudaStream_t stream) {
  auto args = myflux::sm80_gemmrs::make_gemmrs_args<FusedGemmRsConfig>(
      make_fused_params(opt, rank, buf, fused_output_ptrs));
  myflux::sm80_gemmrs::run_gemmrs<FusedGemmRsConfig>(args, buf.workspace, stream);
}

void
run_fused_iteration(
    const Options &opt,
    int rank,
    RankBuffers &buf,
    Element **fused_output_ptrs,
    cudaStream_t stream,
    ThreadBarrier &barrier) {
  if (opt.clear_fused_output) {
    clear_fused_output_once(opt, buf, stream);
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
    barrier.wait();
  }

  launch_fused_once(opt, rank, buf, fused_output_ptrs, stream);

  if (opt.clear_fused_output) {
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
    barrier.wait();
  }
}

float
time_fused_loop(
    const Options &opt,
    int rank,
    RankBuffers &buf,
    Element **fused_output_ptrs,
    cudaStream_t stream,
    ThreadBarrier &barrier) {
  for (int i = 0; i < opt.warmup; ++i) {
    run_fused_iteration(opt, rank, buf, fused_output_ptrs, stream, barrier);
  }
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  barrier.wait();

  CudaEventTimer timer;
  timer.start(stream);
  for (int i = 0; i < opt.iters; ++i) {
    run_fused_iteration(opt, rank, buf, fused_output_ptrs, stream, barrier);
  }
  float elapsed_ms = timer.stop(stream);
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  barrier.wait();
  return elapsed_ms / static_cast<float>(opt.iters);
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

  std::cout << "myflux sm80 fused gemm+rs case\n";
  std::cout << "shape: M=" << opt.m << " N=" << opt.n << " localK=" << opt.k
            << " world_size=" << opt.world_size << "\n";
  std::cout << "iters: warmup=" << opt.warmup << " timed=" << opt.iters
            << " avail_sms=" << opt.avail_sms
            << " clear_fused_output=" << (opt.clear_fused_output ? "true" : "false") << "\n";

  enable_peer_access(opt);
  auto buffers = allocate_buffers(opt);

  std::vector<Element *> fused_output_ptrs(opt.world_size, nullptr);
  for (int rank = 0; rank < opt.world_size; ++rank) {
    fused_output_ptrs[rank] = buffers[rank].fused_output;
  }
  allocate_workspaces(opt, buffers, fused_output_ptrs.data());

  std::vector<float> fused_ms(opt.world_size, 0.0f);
  std::vector<std::exception_ptr> errors(opt.world_size);
  ThreadBarrier barrier(opt.world_size);

  auto worker = [&](int rank) {
    try {
      MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));

      cudaStream_t stream = nullptr;
      MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      fused_ms[rank] = time_fused_loop(
          opt, rank, buffers[rank], fused_output_ptrs.data(), stream, barrier);
      MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
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

  myflux::perf::print_rank_times("fused_sm80_gemmrs", fused_ms);
  free_buffers(opt, buffers);
  return 0;
}
