#pragma once

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace myflux::case_utils {

inline void
check_mpi(int status, const char *expr, const char *file, int line) {
  if (status != MPI_SUCCESS) {
    char err[MPI_MAX_ERROR_STRING] = {};
    int len = 0;
    MPI_Error_string(status, err, &len);
    std::cerr << "MPI error at " << file << ":" << line << ": " << expr
              << " failed: " << std::string(err, len) << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

#define MYFLUX_CASE_CHECK_MPI(expr) \
  ::myflux::case_utils::check_mpi((expr), #expr, __FILE__, __LINE__)

inline int
env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::atoi(value);
}

inline int
mpi_local_rank(int global_rank) {
  int rank = env_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
  if (rank >= 0) {
    return rank;
  }
  rank = env_int("MV2_COMM_WORLD_LOCAL_RANK", -1);
  if (rank >= 0) {
    return rank;
  }
  rank = env_int("SLURM_LOCALID", -1);
  if (rank >= 0) {
    return rank;
  }
  rank = env_int("LOCAL_RANK", -1);
  return rank >= 0 ? rank : global_rank;
}

inline int
mpi_local_world_size(int world_size) {
  int size = env_int("OMPI_COMM_WORLD_LOCAL_SIZE", -1);
  if (size > 0) {
    return size;
  }
  size = env_int("MV2_COMM_WORLD_LOCAL_SIZE", -1);
  if (size > 0) {
    return size;
  }
  size = env_int("LOCAL_WORLD_SIZE", -1);
  return size > 0 ? size : world_size;
}

inline void
set_flux_rank_env(int local_rank, int local_world_size) {
  std::string rank_s = std::to_string(local_rank);
  std::string size_s = std::to_string(local_world_size);
  setenv("LOCAL_RANK", rank_s.c_str(), 1);
  setenv("LOCAL_WORLD_SIZE", size_s.c_str(), 1);
}

template <class T>
struct IpcPeerPointers {
  std::vector<T *> ptrs;
  int self_rank = 0;

  void close() {
    for (int i = 0; i < static_cast<int>(ptrs.size()); ++i) {
      if (i != self_rank && ptrs[i] != nullptr) {
        cudaIpcCloseMemHandle(ptrs[i]);
        ptrs[i] = nullptr;
      }
    }
  }
};

template <class T>
IpcPeerPointers<T>
exchange_cuda_ipc_pointers(T *local_ptr, int rank, int world_size, MPI_Comm comm) {
  cudaIpcMemHandle_t local_handle{};
  cudaError_t handle_status = cudaIpcGetMemHandle(&local_handle, local_ptr);
  if (handle_status != cudaSuccess) {
    std::cerr << "rank " << rank << " cudaIpcGetMemHandle failed: "
              << cudaGetErrorString(handle_status) << "\n";
    MPI_Abort(comm, 1);
  }

  std::vector<cudaIpcMemHandle_t> handles(world_size);
  MYFLUX_CASE_CHECK_MPI(MPI_Allgather(
      &local_handle,
      sizeof(cudaIpcMemHandle_t),
      MPI_BYTE,
      handles.data(),
      sizeof(cudaIpcMemHandle_t),
      MPI_BYTE,
      comm));

  IpcPeerPointers<T> result;
  result.ptrs.resize(world_size, nullptr);
  result.self_rank = rank;
  for (int i = 0; i < world_size; ++i) {
    if (i == rank) {
      result.ptrs[i] = local_ptr;
      continue;
    }
    void *peer_ptr = nullptr;
    cudaError_t open_status =
        cudaIpcOpenMemHandle(&peer_ptr, handles[i], cudaIpcMemLazyEnablePeerAccess);
    if (open_status != cudaSuccess) {
      std::cerr << "rank " << rank << " cudaIpcOpenMemHandle peer " << i
                << " failed: " << cudaGetErrorString(open_status) << "\n";
      MPI_Abort(comm, 1);
    }
    result.ptrs[i] = reinterpret_cast<T *>(peer_ptr);
  }
  return result;
}

inline std::vector<float>
gather_rank_ms(float rank_ms, int rank, int world_size, MPI_Comm comm) {
  std::vector<float> all_ms(rank == 0 ? world_size : 0);
  MYFLUX_CASE_CHECK_MPI(MPI_Gather(
      &rank_ms, 1, MPI_FLOAT, all_ms.data(), 1, MPI_FLOAT, 0, comm));
  return all_ms;
}

inline void
print_rank_times(const char *label, const std::vector<float> &rank_ms) {
  if (rank_ms.empty()) {
    return;
  }
  auto [min_it, max_it] = std::minmax_element(rank_ms.begin(), rank_ms.end());
  float sum = 0.0f;
  std::cout << label << " per-rank ms:";
  for (size_t i = 0; i < rank_ms.size(); ++i) {
    sum += rank_ms[i];
    std::cout << " [" << i << "]=" << std::fixed << std::setprecision(3) << rank_ms[i];
  }
  std::cout << "\n";
  std::cout << label << " summary: min=" << std::fixed << std::setprecision(3) << *min_it
            << " ms, max=" << *max_it << " ms, avg=" << (sum / rank_ms.size()) << " ms\n";
}

}  // namespace myflux::case_utils
