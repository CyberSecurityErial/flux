#pragma once

#include "kernel_utils.cuh"

namespace myflux::sm80_gemmrs {

// Flux 对齐目标：src/gemm_rs/epilogue_evt.hpp 里的 VisitorAuxStoreScatter。
// 这里只固定 NVLink/IntraNode 迁移需要的 public 参数形态，具体 callbacks/store
// 逻辑后续按 Flux 原实现搬进来。
template <class StrideMNL = cutlass_utils::StrideMNL>
struct GemmRsScatterStoreArguments {
  cutlass_utils::Element **scatter_ptr_aux = nullptr;
  StrideMNL dAux = {};
  int64_t rank = 0;
  int64_t world_size = 1;
};

template <
    class ThreadMap,
    class Element,
    cutlass::FloatRoundStyle RoundStyle,
    class StrideMNL,
    class ThreadblockShape,
    bool FuseReduction>
struct VisitorAuxStoreScatterNvlink;

template <class OutputTileThreadMap, class ThreadblockShape, bool FuseReduction>
using GemmRsScatterStoreEVT = VisitorAuxStoreScatterNvlink<
    OutputTileThreadMap,
    cutlass_utils::Element,
    cutlass::FloatRoundStyle::round_to_nearest,
    cutlass_utils::StrideMNL,
    ThreadblockShape,
    FuseReduction>;

}  // namespace myflux::sm80_gemmrs
