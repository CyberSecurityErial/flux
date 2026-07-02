#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace myflux::perf {

inline void
check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(
        stderr,
        "CUDA error at %s:%d: %s failed: %s\n",
        file,
        line,
        expr,
        cudaGetErrorString(status));
    std::abort();
  }
}

class CudaEventTimer {
 public:
  CudaEventTimer();
  ~CudaEventTimer();

  CudaEventTimer(const CudaEventTimer &) = delete;
  CudaEventTimer &operator=(const CudaEventTimer &) = delete;

  void start(cudaStream_t stream);
  float stop(cudaStream_t stream);

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

struct Summary {
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  float avg_ms = 0.0f;
};

Summary summarize(const std::vector<float> &rank_ms);
void print_rank_times(const char *label, const std::vector<float> &rank_ms);
std::string format_bytes(size_t bytes);

}  // namespace myflux::perf

#define MYFLUX_CHECK_CUDA(expr) \
  ::myflux::perf::check_cuda((expr), #expr, __FILE__, __LINE__)
