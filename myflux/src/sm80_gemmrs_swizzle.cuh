#pragma once

#include "kernel_utils.cuh"

namespace myflux::sm80_gemmrs {

// 迁自 Flux: src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp。
// 保持 LOCAL_RANK/LOCAL_WORLD_SIZE 驱动的 rank-offset StreamK tile 顺序。
template <class BaseSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>
struct ThreadblockSwizzleStreamKRankOffset : public BaseSwizzle {
 public:
  int local_rank = 0;
  int local_world_size = 1;

  CUTLASS_HOST_DEVICE
  ThreadblockSwizzleStreamKRankOffset() {}

  ThreadblockSwizzleStreamKRankOffset(
      cutlass::gemm::GemmUniversalMode const mode_,
      cutlass::gemm::GemmCoord const problem_size_,
      cutlass::gemm::GemmCoord const tile_size_,
      int const batch_split_,
      int const sm_occupancy_,
      int const device_sms_,
      int const avail_sms_,
      size_t const element_A_bytes_,
      size_t const element_B_bytes_,
      size_t const element_C_bytes_,
      int const epilogue_acc_fragments_)
      : BaseSwizzle(
            mode_,
            problem_size_,
            tile_size_,
            batch_split_,
            sm_occupancy_,
            device_sms_,
            avail_sms_,
            element_A_bytes_,
            element_B_bytes_,
            element_C_bytes_,
            epilogue_acc_fragments_) {
    const char *local_rank_str = std::getenv("LOCAL_RANK");
    const char *local_world_size_str = std::getenv("LOCAL_WORLD_SIZE");
    if (local_rank_str && local_world_size_str) {
      local_rank = std::atoi(local_rank_str);
      local_world_size = std::atoi(local_world_size_str);
    }
  }

  CUTLASS_DEVICE
  cutlass::gemm::GemmCoord
  get_tile_offset(int tile_idx) const {
    auto coord = BaseSwizzle::get_tile_offset(tile_idx);
    int tiled_m = BaseSwizzle::tiled_shape().m();
    if (coord.m() >= tiled_m) {
      return coord;
    }
    int m = (coord.m() + tiled_m / local_world_size * local_rank) % tiled_m;
    coord.m() = m;
    return coord;
  }
};

using DefaultGemmRsThreadblockSwizzle =
    ThreadblockSwizzleStreamKRankOffset<cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>;

}  // namespace myflux::sm80_gemmrs
