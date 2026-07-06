#include "sm80_gemmrs.cuh"

#ifdef MYFLUX_CHECK_CUDA
#undef MYFLUX_CHECK_CUDA
#endif

#include "mpi_cuda_ipc_utils.h"
#include "performance_utils.h"

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <chrono>
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

enum class ExecMode {
  kAuto,
  kProcess,
  kThread,
};

enum class BarrierMode {
  kHost,
  kDevice,
};

const char *
exec_mode_name(ExecMode mode) {
  switch (mode) {
    case ExecMode::kAuto:
      return "auto";
    case ExecMode::kProcess:
      return "process";
    case ExecMode::kThread:
      return "thread";
  }
  return "unknown";
}

const char *
barrier_mode_name(BarrierMode mode) {
  switch (mode) {
    case BarrierMode::kHost:
      return "host";
    case BarrierMode::kDevice:
      return "device";
  }
  return "unknown";
}

ExecMode
parse_exec_mode(const std::string &name) {
  if (name == "auto") {
    return ExecMode::kAuto;
  }
  if (name == "process" || name == "mpi" || name == "multi-process") {
    return ExecMode::kProcess;
  }
  if (name == "thread" || name == "threads" || name == "multi-thread") {
    return ExecMode::kThread;
  }
  throw std::invalid_argument("unknown exec mode: " + name);
}

BarrierMode
parse_barrier_mode(const std::string &name) {
  if (name == "host" || name == "hostsync") {
    return BarrierMode::kHost;
  }
  if (name == "device" || name == "devicesync") {
    return BarrierMode::kDevice;
  }
  throw std::invalid_argument("unknown barrier mode: " + name);
}

struct Timings {
  float event_ms = 0.0f;
  float wall_ms = 0.0f;
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
  ExecMode exec_mode = ExecMode::kAuto;
  BarrierMode barrier_mode = BarrierMode::kHost;
};

struct RankBuffers {
  Element *input = nullptr;         // 本 rank 的 GEMM 输入 A，形状 [M, K]
  Element *weight = nullptr;        // 本 rank 的 GEMM 权重 W，形状 [N, K]
  Element *fused_output = nullptr;  // 完整 [M, N] buffer，前 [M/world_size, N] 是本 rank 结果
  int *sync_buffer = nullptr;       // device-sync barrier flags, 形状 [world_size]
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
  std::cout << "Usage:\n"
            << "  mpirun --bind-to none --map-by slot -np <world_size> " << program
            << " [options]\n"
            << "  " << program << " [options]    # single-process multi-thread path\n"
            << "Options:\n"
            << "  --m <int>             GEMM M, must be divisible by world size\n"
            << "  --n <int>             GEMM N\n"
            << "  --k <int>             local GEMM K per rank\n"
            << "  --world-size <int>    GPU/rank count, default 8\n"
            << "  --device-start <int>  first CUDA device ordinal, default 0\n"
            << "  --warmup <int>        warmup iterations, default 10\n"
            << "  --iters <int>         timed iterations, default 50\n"
            << "  --avail-sms <int>     CUTLASS avail_sms, default -1\n"
            << "  --exec-mode <name>    auto | process | thread, default auto\n"
            << "  --barrier <name>      host | device, default host\n";
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
    } else if (arg == "--exec-mode" || starts_with(arg, "--exec-mode=")) {
      std::string mode_value;
      if (starts_with(arg, "--exec-mode=")) {
        mode_value = arg.substr(std::strlen("--exec-mode="));
      } else {
        if (i + 1 >= argc) {
          throw std::invalid_argument("missing value for --exec-mode");
        }
        mode_value = argv[++i];
      }
      opt.exec_mode = parse_exec_mode(mode_value);
    } else if (arg == "--barrier" || starts_with(arg, "--barrier=")) {
      std::string barrier_value;
      if (starts_with(arg, "--barrier=")) {
        barrier_value = arg.substr(std::strlen("--barrier="));
      } else {
        if (i + 1 >= argc) {
          throw std::invalid_argument("missing value for --barrier");
        }
        barrier_value = argv[++i];
      }
      opt.barrier_mode = parse_barrier_mode(barrier_value);
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

RankBuffers
allocate_rank_buffers(const Options &opt, cudaStream_t stream) {
  RankBuffers buf;
  size_t input_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.k);
  size_t weight_bytes = bytes_for_half(static_cast<int64_t>(opt.n) * opt.k);
  size_t fused_output_bytes = bytes_for_half(static_cast<int64_t>(opt.m) * opt.n);
  size_t sync_bytes = myflux::peer_barrier::workspace_bytes(opt.world_size);

  MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.input), input_bytes));
  MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.weight), weight_bytes));
  MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.fused_output), fused_output_bytes));
  MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&buf.sync_buffer), sync_bytes));

  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.input, 1, input_bytes, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.weight, 2, weight_bytes, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.fused_output, 0, fused_output_bytes, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.sync_buffer, 0, sync_bytes, stream));
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  return buf;
}

void
allocate_workspace(
    const Options &opt,
    int rank,
    RankBuffers &buf,
    Element **fused_output_ptrs,
    cudaStream_t stream) {
  auto args = myflux::sm80_gemmrs::make_gemmrs_args<FusedGemmRsConfig>(
      make_fused_params(opt, rank, buf, fused_output_ptrs));
  buf.workspace_bytes = myflux::sm80_gemmrs::gemmrs_workspace_size<FusedGemmRsConfig>(args);
  if (buf.workspace_bytes != 0) {
    MYFLUX_CHECK_CUDA(cudaMalloc(&buf.workspace, buf.workspace_bytes));
    MYFLUX_CHECK_CUDA(cudaMemsetAsync(buf.workspace, 0, buf.workspace_bytes, stream));
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  }
}

void
free_rank_buffers(RankBuffers &buf) {
  cudaFree(buf.workspace);
  cudaFree(buf.sync_buffer);
  cudaFree(buf.fused_output);
  cudaFree(buf.weight);
  cudaFree(buf.input);
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

template <class HostBarrier>
void
run_fused_iteration(
    const Options &opt,
    int rank,
    RankBuffers &buf,
    Element **fused_output_ptrs,
    int **sync_ptrs,
    cudaStream_t stream,
    HostBarrier &&host_barrier) {
  if (opt.barrier_mode == BarrierMode::kHost) {
    clear_fused_output_once(opt, buf, stream);
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
    host_barrier();
    launch_fused_once(opt, rank, buf, fused_output_ptrs, stream);
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
    host_barrier();
    return;
  }

  clear_fused_output_once(opt, buf, stream);
  myflux::peer_barrier::launch_all_to_all_atomic(sync_ptrs, rank, opt.world_size, stream);
  launch_fused_once(opt, rank, buf, fused_output_ptrs, stream);
  // 复用同一 output buffer 连续迭代时，下一轮 clear 不能和上一轮 peer write 竞争。
  myflux::peer_barrier::launch_all_to_all_atomic(sync_ptrs, rank, opt.world_size, stream);
}

template <class HostBarrier>
Timings
time_fused_loop(
    const Options &opt,
    int rank,
    RankBuffers &buf,
    Element **fused_output_ptrs,
    int **sync_ptrs,
    cudaStream_t stream,
    HostBarrier &&host_barrier) {
  for (int i = 0; i < opt.warmup; ++i) {
    run_fused_iteration(opt, rank, buf, fused_output_ptrs, sync_ptrs, stream, host_barrier);
  }
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  host_barrier();

  CudaEventTimer timer;
  timer.start(stream);
  auto wall_start = std::chrono::steady_clock::now();
  for (int i = 0; i < opt.iters; ++i) {
    run_fused_iteration(opt, rank, buf, fused_output_ptrs, sync_ptrs, stream, host_barrier);
  }
  float elapsed_event_ms = timer.stop(stream);
  auto wall_stop = std::chrono::steady_clock::now();
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
  host_barrier();

  double elapsed_wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
  return {
      elapsed_event_ms / static_cast<float>(opt.iters),
      static_cast<float>(elapsed_wall_ms / static_cast<double>(opt.iters))};
}

void
run_mpi_case(const Options &opt, int rank, int nranks) {
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

  cudaStream_t stream = nullptr;
  MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  if (rank == 0) {
    std::cout << "myflux sm80 fused gemm+rs MPI case\n";
    std::cout << "shape: M=" << opt.m << " N=" << opt.n << " localK=" << opt.k
              << " world_size=" << opt.world_size << "\n";
    std::cout << "iters: warmup=" << opt.warmup << " timed=" << opt.iters
              << " avail_sms=" << opt.avail_sms << " exec_mode=" << exec_mode_name(opt.exec_mode)
              << " barrier=" << barrier_mode_name(opt.barrier_mode) << "\n";
  }

  RankBuffers buffers = allocate_rank_buffers(opt, stream);
  auto output_peers = myflux::case_utils::exchange_cuda_ipc_pointers(
      buffers.fused_output, rank, nranks, MPI_COMM_WORLD);
  myflux::case_utils::IpcPeerPointers<int> sync_peers;
  std::vector<int *> unused_sync_ptrs(nranks, nullptr);
  int **sync_ptrs = unused_sync_ptrs.data();
  if (opt.barrier_mode == BarrierMode::kDevice) {
    sync_peers = myflux::case_utils::exchange_cuda_ipc_pointers(
        buffers.sync_buffer, rank, nranks, MPI_COMM_WORLD);
    sync_ptrs = sync_peers.ptrs.data();
  }
  allocate_workspace(opt, rank, buffers, output_peers.ptrs.data(), stream);
  MYFLUX_CASE_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  auto mpi_barrier = [] { MYFLUX_CASE_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD)); };
  auto timings = time_fused_loop(
      opt,
      rank,
      buffers,
      output_peers.ptrs.data(),
      sync_ptrs,
      stream,
      mpi_barrier);

  auto event_ms = myflux::case_utils::gather_rank_ms(timings.event_ms, rank, nranks, MPI_COMM_WORLD);
  auto wall_ms = myflux::case_utils::gather_rank_ms(timings.wall_ms, rank, nranks, MPI_COMM_WORLD);
  if (rank == 0) {
    myflux::case_utils::print_rank_times("fused_sm80_gemmrs_event", event_ms);
    myflux::case_utils::print_rank_times("fused_sm80_gemmrs_wall", wall_ms);
  }

  MYFLUX_CASE_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  if (opt.barrier_mode == BarrierMode::kDevice) {
    sync_peers.close();
  }
  output_peers.close();
  free_rank_buffers(buffers);
  MYFLUX_CHECK_CUDA(cudaStreamDestroy(stream));
}

void
run_thread_case(const Options &opt) {
  int device_count = 0;
  MYFLUX_CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (opt.device_start + opt.world_size > device_count) {
    throw std::runtime_error("not enough visible CUDA devices for thread-mode world size");
  }

  std::cout << "myflux sm80 fused gemm+rs single-process thread case\n";
  std::cout << "shape: M=" << opt.m << " N=" << opt.n << " localK=" << opt.k
            << " world_size=" << opt.world_size << "\n";
  std::cout << "iters: warmup=" << opt.warmup << " timed=" << opt.iters
            << " avail_sms=" << opt.avail_sms << " exec_mode=" << exec_mode_name(opt.exec_mode)
              << " barrier=" << barrier_mode_name(opt.barrier_mode) << "\n";

  enable_peer_access(opt);

  std::vector<RankBuffers> buffers(opt.world_size);
  std::vector<cudaStream_t> setup_streams(opt.world_size, nullptr);
  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&setup_streams[rank], cudaStreamNonBlocking));
    buffers[rank] = allocate_rank_buffers(opt, setup_streams[rank]);
  }

  std::vector<Element *> output_ptrs(opt.world_size, nullptr);
  std::vector<int *> sync_ptrs(opt.world_size, nullptr);
  for (int rank = 0; rank < opt.world_size; ++rank) {
    output_ptrs[rank] = buffers[rank].fused_output;
    sync_ptrs[rank] = buffers[rank].sync_buffer;
  }
  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    allocate_workspace(opt, rank, buffers[rank], output_ptrs.data(), setup_streams[rank]);
    MYFLUX_CHECK_CUDA(cudaStreamDestroy(setup_streams[rank]));
  }

  std::vector<float> event_ms(opt.world_size, 0.0f);
  std::vector<float> wall_ms(opt.world_size, 0.0f);
  std::vector<std::exception_ptr> errors(opt.world_size);
  ThreadBarrier barrier(opt.world_size);

  auto worker = [&](int rank) {
    try {
      MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
      cudaStream_t stream = nullptr;
      MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      auto host_barrier = [&] { barrier.wait(); };
      auto timings = time_fused_loop(
          opt, rank, buffers[rank], output_ptrs.data(), sync_ptrs.data(), stream, host_barrier);
      event_ms[rank] = timings.event_ms;
      wall_ms[rank] = timings.wall_ms;
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

  myflux::case_utils::print_rank_times("fused_sm80_gemmrs_event", event_ms);
  myflux::case_utils::print_rank_times("fused_sm80_gemmrs_wall", wall_ms);

  for (int rank = 0; rank < opt.world_size; ++rank) {
    MYFLUX_CHECK_CUDA(cudaSetDevice(opt.device_start + rank));
    free_rank_buffers(buffers[rank]);
  }
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
    bool run_process = opt.exec_mode == ExecMode::kProcess ||
                       (opt.exec_mode == ExecMode::kAuto && nranks == opt.world_size);
    bool run_thread = opt.exec_mode == ExecMode::kThread ||
                      (opt.exec_mode == ExecMode::kAuto && nranks == 1 && opt.world_size > 1);
    if (run_process) {
      if (nranks != opt.world_size) {
        throw std::invalid_argument("process mode requires mpirun -np <world_size>");
      }
      run_mpi_case(opt, rank, nranks);
    } else if (run_thread) {
      if (nranks != 1) {
        throw std::invalid_argument("thread mode must be launched as a single process");
      }
      run_thread_case(opt);
    } else {
      throw std::invalid_argument(
          "run with either mpirun -np <world_size> --exec-mode process or one process --exec-mode thread");
    }
  } catch (const std::exception &ex) {
    std::cerr << "rank " << rank << " failed: " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MYFLUX_CASE_CHECK_MPI(MPI_Finalize());
  return 0;
}
