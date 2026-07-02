#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>

#include "sm80_gemmrs_kernel.cuh"

namespace {

struct Options {
  int device = 0;
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmup = 10;
  int iters = 50;
  int avail_sms = -1;
};

int
parse_int_arg(int argc, char **argv, int &i, const char *name) {
  if (i + 1 >= argc) {
    std::cerr << "missing value for " << name << "\n";
    std::exit(2);
  }
  return std::stoi(argv[++i]);
}

Options
parse_options(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device") == 0) {
      opts.device = parse_int_arg(argc, argv, i, "--device");
    } else if (std::strcmp(argv[i], "--m") == 0) {
      opts.m = parse_int_arg(argc, argv, i, "--m");
    } else if (std::strcmp(argv[i], "--n") == 0) {
      opts.n = parse_int_arg(argc, argv, i, "--n");
    } else if (std::strcmp(argv[i], "--k") == 0) {
      opts.k = parse_int_arg(argc, argv, i, "--k");
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      opts.warmup = parse_int_arg(argc, argv, i, "--warmup");
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opts.iters = parse_int_arg(argc, argv, i, "--iters");
    } else if (std::strcmp(argv[i], "--avail-sms") == 0) {
      opts.avail_sms = parse_int_arg(argc, argv, i, "--avail-sms");
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cout
          << "Usage: tmp_sm80_plain_gemm_case [--device id] [--m M] [--n N] [--k K]\n"
          << "                                  [--warmup W] [--iters I] [--avail-sms S]\n";
      std::exit(0);
    } else {
      std::cerr << "unknown argument: " << argv[i] << "\n";
      std::exit(2);
    }
  }
  if (opts.m <= 0 || opts.n <= 0 || opts.k <= 0 || opts.warmup < 0 || opts.iters <= 0) {
    std::cerr << "m/n/k/iters must be positive, warmup must be non-negative\n";
    std::exit(2);
  }
  return opts;
}

size_t
half_matrix_bytes(int rows, int cols) {
  return static_cast<size_t>(rows) * static_cast<size_t>(cols) *
         sizeof(myflux::cutlass_utils::Element);
}

}  // namespace

int
main(int argc, char **argv) {
  Options opts = parse_options(argc, argv);

  MYFLUX_CHECK_CUDA(cudaSetDevice(opts.device));
  cudaStream_t stream = nullptr;
  MYFLUX_CHECK_CUDA(cudaStreamCreate(&stream));

  using Element = myflux::cutlass_utils::Element;
  Element *a = nullptr;
  Element *b = nullptr;
  Element *d = nullptr;
  void *workspace = nullptr;

  // RCR: A 按 [M, K] row-major 存；B 以 ColumnMajor [K, N] 解释，
  // 物理上等价于一块 [N, K] row-major 权重。
  size_t bytes_a = half_matrix_bytes(opts.m, opts.k);
  size_t bytes_b = half_matrix_bytes(opts.n, opts.k);
  size_t bytes_d = half_matrix_bytes(opts.m, opts.n);

  MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&a), bytes_a));
  MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&b), bytes_b));
  MYFLUX_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d), bytes_d));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(a, 0x3c, bytes_a, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(b, 0x3c, bytes_b, stream));
  MYFLUX_CHECK_CUDA(cudaMemsetAsync(d, 0, bytes_d, stream));

  auto args = myflux::sm80_gemmrs::make_plain_gemm128_args(
      opts.m, opts.n, opts.k, a, b, d, opts.avail_sms);
  size_t workspace_size = myflux::sm80_gemmrs::plain_gemm128_workspace_size(args);
  if (workspace_size != 0) {
    MYFLUX_CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  }

  for (int i = 0; i < opts.warmup; ++i) {
    myflux::sm80_gemmrs::run_plain_gemm128(args, workspace, stream);
  }
  MYFLUX_CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  MYFLUX_CHECK_CUDA(cudaEventCreate(&start));
  MYFLUX_CHECK_CUDA(cudaEventCreate(&stop));
  MYFLUX_CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < opts.iters; ++i) {
    myflux::sm80_gemmrs::run_plain_gemm128(args, workspace, stream);
  }
  MYFLUX_CHECK_CUDA(cudaEventRecord(stop, stream));
  MYFLUX_CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  MYFLUX_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  double avg_ms = elapsed_ms / static_cast<double>(opts.iters);
  double flops = 2.0 * static_cast<double>(opts.m) * opts.n * opts.k;
  double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

  std::cout << "plain_sm80_rcr_gemm"
            << " m=" << opts.m
            << " n=" << opts.n
            << " k=" << opts.k
            << " avg_ms=" << avg_ms
            << " tflops=" << tflops
            << " workspace_bytes=" << workspace_size << "\n";

  MYFLUX_CHECK_CUDA(cudaEventDestroy(start));
  MYFLUX_CHECK_CUDA(cudaEventDestroy(stop));
  MYFLUX_CHECK_CUDA(cudaFree(workspace));
  MYFLUX_CHECK_CUDA(cudaFree(d));
  MYFLUX_CHECK_CUDA(cudaFree(b));
  MYFLUX_CHECK_CUDA(cudaFree(a));
  MYFLUX_CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
