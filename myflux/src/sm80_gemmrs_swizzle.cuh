#pragma once

#include "kernel_utils.cuh"

namespace myflux::sm80_gemmrs {

// Flux 对齐目标：src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp 里的
// ThreadblockSwizzleStreamKRankOffset。第一版先固定接口边界，后续迁移时保持
// LOCAL_RANK/LOCAL_WORLD_SIZE 语义，不在这里引入新的调度策略。
struct RankOffsetSwizzleParams {
  int local_rank = 0;
  int local_world_size = 1;
};

template <class BaseSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>
struct ThreadblockSwizzleStreamKRankOffset;

using DefaultGemmRsThreadblockSwizzle =
    ThreadblockSwizzleStreamKRankOffset<cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>;

}  // namespace myflux::sm80_gemmrs
