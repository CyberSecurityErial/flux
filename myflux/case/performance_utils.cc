#include "performance_utils.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace myflux::perf {

CudaEventTimer::CudaEventTimer() {
  MYFLUX_CHECK_CUDA(cudaEventCreate(&start_));
  MYFLUX_CHECK_CUDA(cudaEventCreate(&stop_));
}

CudaEventTimer::~CudaEventTimer() {
  if (start_ != nullptr) {
    cudaEventDestroy(start_);
  }
  if (stop_ != nullptr) {
    cudaEventDestroy(stop_);
  }
}

void
CudaEventTimer::start(cudaStream_t stream) {
  MYFLUX_CHECK_CUDA(cudaEventRecord(start_, stream));
}

float
CudaEventTimer::stop(cudaStream_t stream) {
  MYFLUX_CHECK_CUDA(cudaEventRecord(stop_, stream));
  MYFLUX_CHECK_CUDA(cudaEventSynchronize(stop_));
  float elapsed_ms = 0.0f;
  MYFLUX_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
  return elapsed_ms;
}

Summary
summarize(const std::vector<float> &rank_ms) {
  Summary summary{};
  if (rank_ms.empty()) {
    return summary;
  }

  auto [min_it, max_it] = std::minmax_element(rank_ms.begin(), rank_ms.end());
  summary.min_ms = *min_it;
  summary.max_ms = *max_it;
  summary.avg_ms =
      std::accumulate(rank_ms.begin(), rank_ms.end(), 0.0f) / static_cast<float>(rank_ms.size());
  return summary;
}

void
print_rank_times(const char *label, const std::vector<float> &rank_ms) {
  auto summary = summarize(rank_ms);
  std::cout << label << " per-rank ms:";
  for (size_t i = 0; i < rank_ms.size(); ++i) {
    std::cout << " [" << i << "]=" << std::fixed << std::setprecision(3) << rank_ms[i];
  }
  std::cout << "\n";
  std::cout << label << " summary: min=" << std::fixed << std::setprecision(3) << summary.min_ms
            << " ms, max=" << summary.max_ms << " ms, avg=" << summary.avg_ms << " ms\n";
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

}  // namespace myflux::perf
