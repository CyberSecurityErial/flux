// cuBLASMp Matmul + ReduceScatter benchmark for the same row-major GEMMRS
// shape used by myflux:
//   output[M, N] = input[M, K_global] * weight[N, K_global]^T
// Each MPI rank owns K_local = K_global / nranks. cuBLASMp sees the same
// memory as column-major matrices and runs:
//   D[N, M] = A[K_global, N]^T * B[K_global, M]
// with D column-wise distributed, which is row-wise ReduceScatter in the
// row-major output view.
//
// This benchmark intentionally uses cuBLASMp's multi-process contract: launch
// one MPI process per GPU. It is separate from the single-process myflux cases.

#include <cublasmp.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void
check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << ": " << expr
              << " failed: " << cudaGetErrorString(status) << "\n";
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

void
check_mpi(int status, const char *expr, const char *file, int line) {
  if (status != MPI_SUCCESS) {
    char err[MPI_MAX_ERROR_STRING] = {};
    int len = 0;
    MPI_Error_string(status, err, &len);
    std::cerr << "MPI error at " << file << ":" << line << ": " << expr
              << " failed: " << std::string(err, len) << "\n";
    std::abort();
  }
}

void
check_cublasmp(cublasMpStatus_t status, const char *expr, const char *file, int line) {
  if (status != CUBLASMP_STATUS_SUCCESS) {
    std::cerr << "cuBLASMp error at " << file << ":" << line << ": " << expr
              << " failed with status " << static_cast<int>(status) << "\n";
    std::abort();
  }
}

#define MYFLUX_CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define MYFLUX_CHECK_NCCL(expr) check_nccl((expr), #expr, __FILE__, __LINE__)
#define MYFLUX_CHECK_MPI(expr) check_mpi((expr), #expr, __FILE__, __LINE__)
#define MYFLUX_CHECK_CUBLASMP(expr) check_cublasmp((expr), #expr, __FILE__, __LINE__)

struct Options {
  int64_t m = 4096;          // row-major output/input rows, also cuBLASMp D columns
  int64_t n = 12288;         // row-major output columns, also cuBLASMp D rows
  int64_t k = 6144;          // local K per rank
  int world_size = 8;        // must match MPI_COMM_WORLD size
  int device_start = 0;
  int warmup = 10;
  int iters = 50;
};

bool
starts_with(const std::string &s, const char *prefix) {
  size_t n = std::strlen(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}

int64_t
parse_i64_arg(int argc, char **argv, int &i, const std::string &arg, const char *name) {
  std::string prefix = std::string(name) + "=";
  if (starts_with(arg, prefix.c_str())) {
    return std::stoll(arg.substr(prefix.size()));
  }
  if (arg == name) {
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string("missing value for ") + name);
    }
    return std::stoll(argv[++i]);
  }
  throw std::invalid_argument(std::string("unhandled integer argument ") + arg);
}

int
parse_int_arg(int argc, char **argv, int &i, const std::string &arg, const char *name) {
  return static_cast<int>(parse_i64_arg(argc, argv, i, arg, name));
}

void
print_usage(const char *program) {
  std::cout << "Usage: mpirun -np <world_size> " << program << " [options]\n"
            << "  --m <int>             row-major GEMM M / tokens, default 4096\n"
            << "  --n <int>             row-major GEMM N / hidden, default 12288\n"
            << "  --k <int>             local K per rank, default 6144\n"
            << "  --world-size <int>    MPI world size, default 8\n"
            << "  --device-start <int>  first CUDA device ordinal, default 0\n"
            << "  --warmup <int>        warmup iterations, default 10\n"
            << "  --iters <int>         timed iterations, default 50\n";
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
      opt.m = parse_i64_arg(argc, argv, i, arg, "--m");
    } else if (arg == "--n" || starts_with(arg, "--n=")) {
      opt.n = parse_i64_arg(argc, argv, i, arg, "--n");
    } else if (arg == "--k" || starts_with(arg, "--k=")) {
      opt.k = parse_i64_arg(argc, argv, i, arg, "--k");
    } else if (arg == "--world-size" || starts_with(arg, "--world-size=")) {
      opt.world_size = parse_int_arg(argc, argv, i, arg, "--world-size");
    } else if (arg == "--device-start" || starts_with(arg, "--device-start=")) {
      opt.device_start = parse_int_arg(argc, argv, i, arg, "--device-start");
    } else if (arg == "--warmup" || starts_with(arg, "--warmup=")) {
      opt.warmup = parse_int_arg(argc, argv, i, arg, "--warmup");
    } else if (arg == "--iters" || starts_with(arg, "--iters=")) {
      opt.iters = parse_int_arg(argc, argv, i, arg, "--iters");
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.world_size <= 0) {
    throw std::invalid_argument("m/n/k/world-size must be positive");
  }
  if (opt.warmup < 0 || opt.iters <= 0) {
    throw std::invalid_argument("iters must be positive and warmup must be non-negative");
  }
  if (opt.m % opt.world_size != 0) {
    throw std::invalid_argument("m must be divisible by world-size for ReduceScatter output");
  }
  return opt;
}

int
get_env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  return std::atoi(value);
}

int
get_local_rank(int global_rank) {
  int rank = get_env_int("LOCAL_RANK", -1);
  if (rank >= 0) {
    return rank;
  }
  rank = get_env_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
  if (rank >= 0) {
    return rank;
  }
  rank = get_env_int("MV2_COMM_WORLD_LOCAL_RANK", -1);
  if (rank >= 0) {
    return rank;
  }
  rank = get_env_int("SLURM_LOCALID", -1);
  if (rank >= 0) {
    return rank;
  }
  return global_rank;
}

size_t
bytes_for_half(int64_t elements) {
  return static_cast<size_t>(elements) * sizeof(__half);
}

std::string
format_bytes(size_t bytes) {
  constexpr double kKiB = 1024.0;
  constexpr double kMiB = kKiB * 1024.0;
  constexpr double kGiB = kMiB * 1024.0;
  std::ostringstream os;
  os << std::fixed << std::setprecision(2);
  if (bytes >= static_cast<size_t>(kGiB)) {
    os << static_cast<double>(bytes) / kGiB << " GiB";
  } else if (bytes >= static_cast<size_t>(kMiB)) {
    os << static_cast<double>(bytes) / kMiB << " MiB";
  } else if (bytes >= static_cast<size_t>(kKiB)) {
    os << static_cast<double>(bytes) / kKiB << " KiB";
  } else {
    os << bytes << " B";
  }
  return os.str();
}

struct CudaEventTimer {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  CudaEventTimer() {
    MYFLUX_CHECK_CUDA(cudaEventCreate(&start));
    MYFLUX_CHECK_CUDA(cudaEventCreate(&stop));
  }

  ~CudaEventTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void begin(cudaStream_t stream) { MYFLUX_CHECK_CUDA(cudaEventRecord(start, stream)); }

  float end(cudaStream_t stream) {
    MYFLUX_CHECK_CUDA(cudaEventRecord(stop, stream));
    MYFLUX_CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    MYFLUX_CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
  }
};

void
run_matmul_rs_once(
    cublasMpHandle_t handle,
    cublasMpMatmulDescriptor_t matmul_desc,
    const Options &opt,
    int nranks,
    const void *d_a,
    cublasMpMatrixDescriptor_t desc_a,
    const void *d_b,
    cublasMpMatrixDescriptor_t desc_b,
    void *d_d,
    cublasMpMatrixDescriptor_t desc_d,
    void *d_work,
    size_t d_work_bytes,
    void *h_work,
    size_t h_work_bytes) {
  float alpha = 1.0f;
  float beta = 0.0f;
  const int64_t mp_m = opt.n;
  const int64_t mp_n = opt.m;
  const int64_t mp_k = opt.k * nranks;

  MYFLUX_CHECK_CUBLASMP(cublasMpMatmul(
      handle,
      matmul_desc,
      mp_m,
      mp_n,
      mp_k,
      &alpha,
      d_a,
      1,
      1,
      desc_a,
      d_b,
      1,
      1,
      desc_b,
      &beta,
      nullptr,
      1,
      1,
      nullptr,
      d_d,
      1,
      1,
      desc_d,
      d_work,
      d_work_bytes,
      h_work,
      h_work_bytes));
}

}  // namespace

int
main(int argc, char **argv) {
  MYFLUX_CHECK_MPI(MPI_Init(&argc, &argv));

  int rank = 0;
  int nranks = 1;
  MYFLUX_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MYFLUX_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  try {
    Options opt = parse_options(argc, argv);
    if (opt.world_size != nranks) {
      throw std::invalid_argument("--world-size must match MPI_COMM_WORLD size");
    }

    int device_count = 0;
    MYFLUX_CHECK_CUDA(cudaGetDeviceCount(&device_count));
    int local_rank = get_local_rank(rank);
    int device = opt.device_start + local_rank;
    if (device < 0 || device >= device_count) {
      throw std::runtime_error("selected CUDA device is out of range");
    }
    MYFLUX_CHECK_CUDA(cudaSetDevice(device));

    cudaStream_t stream = nullptr;
    MYFLUX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    ncclUniqueId nccl_id;
    if (rank == 0) {
      MYFLUX_CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }
    MYFLUX_CHECK_MPI(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t nccl_comm = nullptr;
    MYFLUX_CHECK_NCCL(ncclCommInitRank(&nccl_comm, nranks, nccl_id, rank));

    cublasMpHandle_t handle = nullptr;
    MYFLUX_CHECK_CUBLASMP(cublasMpCreate(&handle, stream));

    cublasMpGrid_t grid_col_major = nullptr;
    cublasMpGrid_t grid_row_major = nullptr;
    MYFLUX_CHECK_CUBLASMP(cublasMpGridCreate(
        nranks, 1, CUBLASMP_GRID_LAYOUT_COL_MAJOR, nccl_comm, &grid_col_major));
    MYFLUX_CHECK_CUBLASMP(cublasMpGridCreate(
        1, nranks, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, nccl_comm, &grid_row_major));

    const int64_t local_k = opt.k;
    const int64_t global_k = opt.k * nranks;
    const int64_t mp_m = opt.n;          // row-major N, column-major D rows
    const int64_t mp_n = opt.m;          // row-major M, column-major D cols
    const int64_t local_d_cols = mp_n / nranks;

    const int64_t mb_a = local_k;
    const int64_t nb_a = mp_m;
    const int64_t mb_b = local_k;
    const int64_t nb_b = local_d_cols;
    const int64_t mb_d = nb_a;
    const int64_t nb_d = nb_b;

    cublasMpMatrixDescriptor_t desc_a = nullptr;
    cublasMpMatrixDescriptor_t desc_b = nullptr;
    cublasMpMatrixDescriptor_t desc_d = nullptr;
    MYFLUX_CHECK_CUBLASMP(cublasMpMatrixDescriptorCreate(
        global_k, mp_m, mb_a, nb_a, 0, 0, mb_a, CUDA_R_16F, grid_col_major, &desc_a));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatrixDescriptorCreate(
        global_k, mp_n, mb_b, nb_b, 0, 0, mb_b, CUDA_R_16F, grid_col_major, &desc_b));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatrixDescriptorCreate(
        mp_m, mp_n, mb_d, nb_d, 0, 0, mb_d, CUDA_R_16F, grid_row_major, &desc_d));

    cublasMpMatmulDescriptor_t matmul_desc = nullptr;
    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasMpMatmulAlgoType_t algo_type = CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P;
    MYFLUX_CHECK_CUBLASMP(cublasMpMatmulDescriptorCreate(&matmul_desc, CUBLAS_COMPUTE_32F));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatmulDescriptorSetAttribute(
        matmul_desc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &trans_a, sizeof(trans_a)));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatmulDescriptorSetAttribute(
        matmul_desc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &trans_b, sizeof(trans_b)));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatmulDescriptorSetAttribute(
        matmul_desc,
        CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE,
        &algo_type,
        sizeof(algo_type)));

    __half *d_a = nullptr;  // weight row-major [N, K_local] viewed as col-major [K_local, N]
    __half *d_b = nullptr;  // input row-major [M, K_local] viewed as col-major [K_local, M]
    __half *d_d = nullptr;  // output shard row-major [M/nranks, N] viewed as col-major [N, M/nranks]
    size_t a_bytes = bytes_for_half(local_k * mp_m);
    size_t b_bytes = bytes_for_half(local_k * mp_n);
    size_t d_bytes = bytes_for_half(mp_m * local_d_cols);
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_a), a_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_b), b_bytes));
    MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_d), d_bytes));
    MYFLUX_CHECK_CUDA(cudaMemsetAsync(d_a, 2, a_bytes, stream));
    MYFLUX_CHECK_CUDA(cudaMemsetAsync(d_b, 1, b_bytes, stream));
    MYFLUX_CHECK_CUDA(cudaMemsetAsync(d_d, 0, d_bytes, stream));
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));

    float alpha = 1.0f;
    float beta = 0.0f;
    size_t d_work_bytes = 0;
    size_t h_work_bytes = 0;
    MYFLUX_CHECK_CUBLASMP(cublasMpMatmul_bufferSize(
        handle,
        matmul_desc,
        mp_m,
        mp_n,
        global_k,
        &alpha,
        d_a,
        1,
        1,
        desc_a,
        d_b,
        1,
        1,
        desc_b,
        &beta,
        nullptr,
        1,
        1,
        nullptr,
        d_d,
        1,
        1,
        desc_d,
        &d_work_bytes,
        &h_work_bytes));

    void *d_work = nullptr;
    if (d_work_bytes != 0) {
      MYFLUX_CHECK_CUBLASMP(cublasMpMalloc(grid_row_major, &d_work, d_work_bytes));
    }
    std::vector<int8_t> h_work(h_work_bytes);

    if (rank == 0) {
      std::cout << "myflux cuBLASMp Matmul+ReduceScatter bench\n";
      std::cout << "row-major shape: M=" << opt.m << " N=" << opt.n
                << " localK=" << opt.k << " world_size=" << nranks << "\n";
      std::cout << "cuBLASMp TN shape: m=" << mp_m << " n=" << mp_n
                << " k=" << global_k << " local_d_cols=" << local_d_cols << "\n";
      std::cout << "per-rank buffers: A=" << format_bytes(a_bytes)
                << " B=" << format_bytes(b_bytes) << " D=" << format_bytes(d_bytes)
                << " d_work=" << format_bytes(d_work_bytes)
                << " h_work=" << format_bytes(h_work_bytes) << "\n";
      std::cout << "iters: warmup=" << opt.warmup << " timed=" << opt.iters << "\n";
    }

    MYFLUX_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < opt.warmup; ++i) {
      run_matmul_rs_once(
          handle,
          matmul_desc,
          opt,
          nranks,
          d_a,
          desc_a,
          d_b,
          desc_b,
          d_d,
          desc_d,
          d_work,
          d_work_bytes,
          h_work.data(),
          h_work_bytes);
    }
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
    MYFLUX_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    CudaEventTimer timer;
    timer.begin(stream);
    for (int i = 0; i < opt.iters; ++i) {
      run_matmul_rs_once(
          handle,
          matmul_desc,
          opt,
          nranks,
          d_a,
          desc_a,
          d_b,
          desc_b,
          d_d,
          desc_d,
          d_work,
          d_work_bytes,
          h_work.data(),
          h_work_bytes);
    }
    float elapsed_ms = timer.end(stream);
    MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));
    float rank_ms = elapsed_ms / static_cast<float>(opt.iters);

    std::vector<float> all_ms(rank == 0 ? nranks : 0);
    MYFLUX_CHECK_MPI(MPI_Gather(&rank_ms, 1, MPI_FLOAT, all_ms.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
    if (rank == 0) {
      float min_ms = all_ms[0];
      float max_ms = all_ms[0];
      float sum_ms = 0.0f;
      std::cout << "cublasmp_matmul_rs per-rank ms:";
      for (int i = 0; i < nranks; ++i) {
        min_ms = std::min(min_ms, all_ms[i]);
        max_ms = std::max(max_ms, all_ms[i]);
        sum_ms += all_ms[i];
        std::cout << " [" << i << "]=" << std::fixed << std::setprecision(3) << all_ms[i];
      }
      std::cout << "\n";
      std::cout << "summary: min=" << std::fixed << std::setprecision(3) << min_ms
                << " ms max=" << max_ms << " ms avg=" << (sum_ms / nranks) << " ms\n";
    }

    MYFLUX_CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    if (d_work != nullptr) {
      MYFLUX_CHECK_CUBLASMP(cublasMpFree(grid_row_major, d_work));
    }
    MYFLUX_CHECK_CUDA(cudaFree(d_d));
    MYFLUX_CHECK_CUDA(cudaFree(d_b));
    MYFLUX_CHECK_CUDA(cudaFree(d_a));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatmulDescriptorDestroy(matmul_desc));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatrixDescriptorDestroy(desc_d));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatrixDescriptorDestroy(desc_b));
    MYFLUX_CHECK_CUBLASMP(cublasMpMatrixDescriptorDestroy(desc_a));
    MYFLUX_CHECK_CUBLASMP(cublasMpGridDestroy(grid_row_major));
    MYFLUX_CHECK_CUBLASMP(cublasMpGridDestroy(grid_col_major));
    MYFLUX_CHECK_CUBLASMP(cublasMpDestroy(handle));
    MYFLUX_CHECK_NCCL(ncclCommDestroy(nccl_comm));
    MYFLUX_CHECK_CUDA(cudaStreamDestroy(stream));
  } catch (const std::exception &ex) {
    std::cerr << "rank " << rank << " failed: " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MYFLUX_CHECK_MPI(MPI_Finalize());
  return 0;
}
