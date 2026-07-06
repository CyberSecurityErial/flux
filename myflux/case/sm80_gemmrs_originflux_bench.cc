// Native Flux GEMM_RS benchmark case.
//
// This intentionally uses Flux's C++ OpRegistry/GemmReduceScatterArguments path
// directly and does not go through the Python/THS wrapper.
#include "performance_utils.h"

#include "cutlass/half.h"
#include "flux/args/gemm_rs.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Element = cutlass::half_t;

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
  int nnodes = 1;
  int device_start = 0;
  int warmup = 10;
  int iters = 50;
  int avail_sms = -1;
  bool force_sm80 = false;
};

struct RankBuffers {
  Element *input = nullptr;
  Element *weight = nullptr;
  Element *bias = nullptr;
  Element *output = nullptr;
  Element *reduce_buffer = nullptr;
  void *barrier = nullptr;
  void *workspace = nullptr;
  size_t barrier_bytes = 0;
  size_t workspace_bytes = 0;
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
            << "  --nnodes <int>        node count passed to Flux runtime config, default 1\n"
            << "  --device-start <int>  first CUDA device ordinal, default 0\n"
            << "  --warmup <int>        warmup iterations, default 10\n"
            << "  --iters <int>         timed iterations, default 50\n"
            << "  --avail-sms <int>     Flux/CUTLASS avail_sms, default -1\n"
            << "  --force-sm80          force native Flux meta to Sm80/GemmV2\n";
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
    } else if (arg == "--nnodes" || starts_with(arg, "--nnodes=")) {
      opt.nnodes = parse_int_arg(argc, argv, i, arg, "--nnodes");
    } else if (arg == "--device-start" || starts_with(arg, "--device-start=")) {
      opt.device_start = parse_int_arg(argc, argv, i, arg, "--device-start");
    } else if (arg == "--warmup" || starts_with(arg, "--warmup=")) {
      opt.warmup = parse_int_arg(argc, argv, i, arg, "--warmup");
    } else if (arg == "--iters" || starts_with(arg, "--iters=")) {
      opt.iters = parse_int_arg(argc, argv, i, arg, "--iters");
    } else if (arg == "--avail-sms" || starts_with(arg, "--avail-sms=")) {
      opt.avail_sms = parse_int_arg(argc, argv, i, arg, "--avail-sms");
    } else if (arg == "--force-sm80") {
      opt.force_sm80 = true;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opt.world_size <= 0 || opt.world_size > 8) {
    throw std::invalid_argument("world size must be in (0, 8]");
  }
  if (opt.nnodes <= 0 || opt.world_size % opt.nnodes != 0) {
    throw std::invalid_argument("nnodes must be positive and divide world size");
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

bytedance::flux::GemmReduceScatterArguments
make_origin_args(
    const Options &opt,
    int rank,
    const RankBuffers &buf,
    void **output_ptrs,
    void **reduce_buffer_ptrs,
    void **barrier_ptrs) {
  bytedance::flux::GemmReduceScatterArguments args{
      opt.m,
      opt.n,
      opt.k,
      rank,
      opt.world_size,
      opt.nnodes,
      1.0f,
      0.0f,
      buf.input,
      buf.weight,
      buf.bias,
      output_ptrs,
      reduce_buffer_ptrs,
      barrier_ptrs,
      opt.avail_sms};
  return args;
}

std::vector<RankBuffers>
allocate_buffers(const Options &opt) {
  std::vector<RankBuffers> buffers(opt.world_size);
  int m_per_rank = opt.m / opt.world_size;

  size_t input_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.k);
  size_t weight_bytes = bytes_for_half(static_cast<int64_t>(opt.k) * opt.n);
  size_t bias_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.n);
  size_t output_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.n);
  size_t reduce_bytes = bytes_for_half(
      static_cast<int64_t>(m_per_rank) * opt.nnodes * opt.nnodes * opt.n);

  std::cout << "per-rank buffers: input=" << myflux::perf::format_bytes(input_bytes)
            << " weight=" << myflux::perf::format_bytes(weight_bytes)
            << " bias=" << myflux::perf::format_bytes(bias_bytes)
            << " output=" << myflux::perf::format_bytes(output_bytes)
            << " reduce_buffer=" << myflux::perf::format_bytes(reduce_bytes) << "\n";

  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    auto &buf = buffers[rank];
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.input), input_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.weight), weight_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.bias), bias_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.output), output_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.reduce_buffer), reduce_bytes));

    MYFLUX_CHECK_CUDA(cudaMemset(buf.input, 1, input_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.weight, 2, weight_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.bias, 0, bias_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.output, 0, output_bytes));
    MYFLUX_CHECK_CUDA(cudaMemset(buf.reduce_buffer, 0, reduce_bytes));
  }
  return buffers;
}

void
free_buffers(const Options &opt, std::vector<RankBuffers> &buffers) {
  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    cudaFree(buffers[rank].workspace);
    cudaFree(buffers[rank].barrier);
    cudaFree(buffers[rank].reduce_buffer);
    cudaFree(buffers[rank].output);
    cudaFree(buffers[rank].bias);
    cudaFree(buffers[rank].weight);
    cudaFree(buffers[rank].input);
  }
}

template <class GemmOp>
void
allocate_runtime_workspaces(
    const Options &opt,
    GemmOp &gemm_op,
    std::vector<RankBuffers> &buffers,
    void **output_ptrs,
    void **reduce_buffer_ptrs,
    void **barrier_ptrs) {
  auto probe_args =
      make_origin_args(opt, 0, buffers[0], output_ptrs, reduce_buffer_ptrs, barrier_ptrs);
  size_t barrier_bytes = gemm_op->get_barrier_workspace_size(probe_args);
  size_t workspace_bytes = gemm_op->get_workspace_size(probe_args);

  std::cout << "per-rank runtime: barrier=" << myflux::perf::format_bytes(barrier_bytes)
            << " workspace=" << myflux::perf::format_bytes(workspace_bytes) << "\n";

  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    auto &buf = buffers[rank];
    buf.barrier_bytes = barrier_bytes;
    buf.workspace_bytes = workspace_bytes;
    if (barrier_bytes != 0) {
      MYFLUX_CHECK_CUDA(cudaMalloc(&buf.barrier, barrier_bytes));
      MYFLUX_CHECK_CUDA(cudaMemset(buf.barrier, 0, barrier_bytes));
    }
    if (workspace_bytes != 0) {
      MYFLUX_CHECK_CUDA(cudaMalloc(&buf.workspace, workspace_bytes));
      MYFLUX_CHECK_CUDA(cudaMemset(buf.workspace, 0, workspace_bytes));
    }
    barrier_ptrs[rank] = buf.barrier;
  }
}

template <class GemmOp>
float
time_origin_loop(
    const Options &opt,
    int rank,
    GemmOp &gemm_op,
    RankBuffers &buf,
    void **output_ptrs,
    void **reduce_buffer_ptrs,
    void **barrier_ptrs,
    cudaStream_t stream,
    ThreadBarrier &barrier) {
  auto args =
      make_origin_args(opt, rank, buf, output_ptrs, reduce_buffer_ptrs, barrier_ptrs);

  for (int i = 0; i < opt.warmup; ++i) {
    gemm_op->run(args, buf.workspace, stream);
  }
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  barrier.wait();

  myflux::perf::CudaEventTimer timer;
  timer.start(stream);
  for (int i = 0; i < opt.iters; ++i) {
    gemm_op->run(args, buf.workspace, stream);
  }
  float elapsed_ms = timer.stop(stream);
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  barrier.wait();
  return elapsed_ms / static_cast<float>(opt.iters);
}

void
print_summary(const Options &opt, const std::vector<float> &rank_ms) {
  myflux::perf::print_rank_times("origin_flux_gemmrs", rank_ms);
  auto summary = myflux::perf::summarize(rank_ms);
  double gemm_flops = 2.0 * static_cast<double>(opt.m) * opt.n * opt.k;
  double tflops_per_rank = gemm_flops / (summary.max_ms * 1.0e-3) / 1.0e12;
  std::cout << "origin_flux_gemmrs per_rank_gemm_tflops_by_max_ms=" << std::fixed
            << std::setprecision(3) << tflops_per_rank << "\n";
}

}  // namespace

int
main(int argc, char **argv) {
  Options opt = parse_options(argc, argv);

  int device_count = 0;
  MYFLUX_CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (opt.device_start + opt.world_size > device_count) {
    std::cerr << "requested " << opt.world_size << " devices from " << opt.device_start
              << ", visible device count is " << device_count << "\n";
    return 1;
  }

  MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start));
  bytedance::flux::ArchEnum arch =
      opt.force_sm80 ? bytedance::flux::_Sm80{}() : bytedance::flux::get_arch();
  bytedance::flux::SMCoreEnum sm_core =
      opt.force_sm80 ? bytedance::flux::_L20{}() : bytedance::flux::get_sm_core();
  auto gemm_impl = opt.force_sm80 || ((int)arch < (int)bytedance::flux::_Sm90{}())
                       ? bytedance::flux::_GemmV2{}()
                       : bytedance::flux::_GemmV3{}();
  auto meta = bytedance::flux::make_gemm_meta(
      bytedance::flux::_FP16{},
      arch,
      sm_core,
      bytedance::flux::_ReduceScatter{},
      bytedance::flux::_RCR{},
      gemm_impl,
      bytedance::flux::None{},
      bytedance::flux::make_reduce_scatter_meta(true, bytedance::flux::_IntraNode{}));
  auto rt_conf = bytedance::flux::make_runtime_config(
      opt.m,
      opt.n,
      opt.k,
      bytedance::flux::make_reduce_scatter_runtime_config(opt.world_size, opt.nnodes));

  std::cout << "origin Flux C++ GEMM_RS bench\n";
  std::cout << "shape: M=" << opt.m << " N=" << opt.n << " localK=" << opt.k
            << " world_size=" << opt.world_size << " nnodes=" << opt.nnodes << "\n";
  std::cout << "iters: warmup=" << opt.warmup << " timed=" << opt.iters
            << " avail_sms=" << opt.avail_sms << "\n";
  std::cout << "origin path: FP16 RCR ReduceScatter IntraNode fuse_reduction=true\n";
  std::cout << "origin dispatch: " << (opt.force_sm80 ? "forced Sm80/GemmV2" : "auto arch")
            << "\n";

  enable_peer_access(opt);
  auto buffers = allocate_buffers(opt);

  std::vector<void *> output_ptrs(opt.world_size, nullptr);
  std::vector<void *> reduce_buffer_ptrs(opt.world_size, nullptr);
  std::vector<void *> barrier_ptrs(opt.world_size, nullptr);
  for (int rank = 0; rank < opt.world_size; ++rank) {
    output_ptrs[rank] = buffers[rank].output;
    reduce_buffer_ptrs[rank] = buffers[rank].reduce_buffer;
  }

  auto probe_op = bytedance::flux::OpRegistry::instance().get_op(meta, rt_conf);
  allocate_runtime_workspaces(
      opt, probe_op, buffers, output_ptrs.data(), reduce_buffer_ptrs.data(), barrier_ptrs.data());

  std::vector<float> rank_ms(opt.world_size, 0.0f);
  std::vector<std::exception_ptr> errors(opt.world_size);
  ThreadBarrier barrier(opt.world_size);

  auto worker = [&](int rank) {
    try {
      MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
      auto gemm_op = bytedance::flux::OpRegistry::instance().get_op(meta, rt_conf);

      cudaStream_t stream = nullptr;
      MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      rank_ms[rank] = time_origin_loop(
          opt,
          rank,
          gemm_op,
          buffers[rank],
          output_ptrs.data(),
          reduce_buffer_ptrs.data(),
          barrier_ptrs.data(),
          stream,
          barrier);
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

  print_summary(opt, rank_ms);
  free_buffers(opt, buffers);
  return 0;
}
