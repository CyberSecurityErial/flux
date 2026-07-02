#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/arch/memory.h"
#include "cutlass/numeric_conversion.h"

#include "kernel_utils.cuh"

namespace cutlass::arch {

// 迁自 Flux include/flux/cuda/memory_utils.hpp：SM80 fp16 reduce-scatter
// epilogue 只需要 4-byte f16x2 red.global.sys.add 路径。
// TODO: 如果后续同一个 TU 同时 include flux/cuda/memory_utils.hpp，需要把这个 helper
// 移到 myflux detail namespace，避免重复特化 cutlass::arch::global_red。
template <typename AccessType, int StoreBytes, typename ElementType>
struct global_red;

template <typename AccessType>
struct global_red<AccessType, 4, cutlass::half_t> {
  CUTLASS_DEVICE
  global_red(AccessType const &D, void *ptr, bool pred_guard) {
    uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p red.global.sys.add.noftz.f16x2 [%0], %1;\n"
        "}\n"
        :
        : "l"(ptr), "r"(data), "r"((int)pred_guard));
  }
};

}  // namespace cutlass::arch

namespace myflux::sm80_gemmrs {

// 固定 SM80/NVLink/IntraNode 的 scatter store 参数；对齐 Flux
// VisitorAuxStoreScatter 的 NVLink 分支，不包含 PCIe barrier queue 胶水。
// Buffer contract 对齐 Flux wrapper：
// - FuseReduction=false: scatter_ptr_aux[i] 指向完整 [M, N] scratch buffer。
// - FuseReduction=true: scatter_ptr_aux[i] 指向已清零的最终 [M/world_size, N] buffer。
// launcher 侧需要检查 rank/world_size/M/tile 合法性，visitor 不做 host 参数兜底。
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
    bool FuseReduction_>
struct VisitorAuxStoreScatterNvlink {
  static_assert(std::is_same_v<Element, cutlass::half_t>, "myflux SM80 GEMMRS only supports fp16 now");

  constexpr static bool FuseReduction = std::is_same_v<Element, cutlass::half_t> ? FuseReduction_ : false;
  using Arguments = GemmRsScatterStoreArguments<StrideMNL>;

  struct Params {
    Element *scatter_ptr_aux[kMaxWorldSize] = {};
    StrideMNL dAux = {};
    int64_t rank = 0;
    int64_t world_size = 1;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    (void)problem_shape;
    (void)workspace;
    Params params;
    params.dAux = args.dAux;
    params.rank = args.rank;
    params.world_size = args.world_size;
    for (int i = 0; i < args.world_size; ++i) {
      params.scatter_ptr_aux[i] = args.scatter_ptr_aux[i];
    }
    return params;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const &problem_shape, Arguments const &args) {
    (void)problem_shape;
    (void)args;
    return 0;
  }

  struct SharedStorage {
    cute::array_aligned<Element *, ThreadblockShape::kM> smem_rs_ptrs;
  };

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<Element>::value;
  using VecType = cutlass::uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreScatterNvlink() : params_ptr(nullptr), storage_ptr(nullptr) {}

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreScatterNvlink(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params), storage_ptr(&shared_storage) {}

  Params const *params_ptr;
  SharedStorage const *storage_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
        GTensor &&tC_gAux,
        RTensor &&tC_rAux,
        CTensor &&tC_cAux,
        ProblemShape problem_shape,
        Params const *params_ptr,
        int thread_idx,
        int tile_idx,
        int dst_rank,
        int m_end,
        Element **smem_rs_ptrs,
        uint32_t row_start,
        uint32_t col_start)
        : tC_gAux(cute::forward<GTensor>(tC_gAux)),
          tC_rAux(cute::forward<RTensor>(tC_rAux)),
          tC_cAux(cute::forward<CTensor>(tC_cAux)),
          params_ptr(params_ptr),
          problem_shape(problem_shape),
          thread_idx(thread_idx),
          tile_idx(tile_idx),
          dst_rank(dst_rank),
          m_end(m_end),
          row_start(row_start),
          col_start(col_start),
          rs_ptrs(smem_rs_ptrs) {}

    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    Params const *params_ptr;
    ProblemShape problem_shape;
    int thread_idx;
    int tile_idx;
    int dst_rank;
    int m_end;
    uint32_t row_start;
    uint32_t col_start;
    Element **rs_ptrs;
    static constexpr int kMaxRankSize = 4;
    int row_stride[kMaxRankSize];
    int col_stride[kMaxRankSize];

    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      (void)step_idx;
      clear(tC_rAux);
    }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frg_idx,
        cutlass::Array<ElementAccumulator, FragmentSize> const &frg_acc,
        cutlass::Array<ElementInput, FragmentSize> const &frg_input) {
      (void)iter_idx;
      (void)row_idx;
      (void)column_idx;
      (void)frg_acc;
      using ConvertInput =
          cutlass::NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      auto tC_rAux_frg = cute::recast<cutlass::Array<Element, FragmentSize>>(cute::coalesce(tC_rAux));
      tC_rAux_frg(frg_idx) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = cute::filter(tC_rAux);
      auto coord_v = cute::filter(tC_cAux(cute::_, cute::_, cute::_, step_idx));
      auto shape = cute::make_shape(m_end, cute::size<1>(problem_shape), cute::size<2>(problem_shape));
      const int cur_step_row = row_start + step_idx * row_stride[3];
      const int cur_step_col = col_start + step_idx * col_stride[3];

      using Shape_tC_gAux = decltype(cute::flatten(tC_gAux.shape()));
      constexpr int size2 = cute::size<2>(Shape_tC_gAux{});
      constexpr int size1 = cute::size<1>(Shape_tC_gAux{});
      constexpr int size0 = cute::size<0>(Shape_tC_gAux{});
      constexpr int size01 = size0 * size1;

      CUTLASS_PRAGMA_UNROLL
      for (int p2 = 0; p2 < size2; ++p2) {
        CUTLASS_PRAGMA_UNROLL
        for (int p1 = 0; p1 < size1; ++p1) {
          CUTLASS_PRAGMA_UNROLL
          for (int p0 = 0; p0 < size0; ++p0) {
            int row = cur_step_row + p0 * row_stride[0] + p1 * row_stride[1] + p2 * row_stride[2];
            int col = cur_step_col + p0 * col_stride[0] + p1 * col_stride[1] + p2 * col_stride[2];
            void *dst_ptr = rs_ptrs[row] + col;
            int pos = p0 + p1 * size0 + p2 * size01;
            bool guard = cute::elem_less(coord_v(pos), problem_shape);
            if constexpr (FuseReduction) {
              cutlass::arch::global_red<VecType, sizeof(VecType), cutlass::half_t>(
                  src_v(pos), dst_ptr, guard);
            } else {
              cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(pos), dst_ptr, guard);
            }
          }
        }
      }
    }

    CUTLASS_DEVICE void
    end_epilogue() {}
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      int thread_idx,
      ProblemShape problem_shape) {
    using namespace cute;

    Params const &params = *params_ptr;
    const int M = get<0>(problem_shape);
    const int N = get<1>(problem_shape);
    constexpr int kM = ThreadblockShape::kM;
    constexpr int kN = ThreadblockShape::kN;

    int tiled_m = (M + kM - 1) / kM;
    int tiled_n = (N + kN - 1) / kN;
    int tiled_m_per_rank = tiled_m / params.world_size;
    int dst_rank = threadblock_tile_offset.m() / tiled_m_per_rank;

    const int M_lines_per_rank = M / params.world_size;
    Element **smem_ptr = const_cast<Element **>(storage_ptr->smem_rs_ptrs.data());

    auto dst_offset = threadblock_tile_offset;
    int tile_idx = tiled_n * threadblock_tile_offset.m() + threadblock_tile_offset.n();
    int m_end = size<0>(problem_shape);
    const int m_offset_start = dst_offset.m() * kM;
    const int tid = threadIdx.x;

    if constexpr (FuseReduction) {
      if (threadIdx.x < kM) {
        int cur_row = m_offset_start + tid;
        int cur_dst_rank = cur_row / M_lines_per_rank;
        smem_ptr[tid] = params.scatter_ptr_aux[cur_dst_rank] + N * (cur_row % M_lines_per_rank);
      }
      __syncthreads();
    } else {
      if (threadIdx.x < kM) {
        int cur_row = m_offset_start + tid;
        int cur_dst_rank = cur_row / M_lines_per_rank;
        int target_line = params.rank * M_lines_per_rank + cur_row % M_lines_per_rank;
        smem_ptr[tid] = params.scatter_ptr_aux[cur_dst_rank] + N * target_line;
      }
      __syncthreads();
    }

    auto tC_gAux = [&]() {
      auto mAux = make_tensor(
          make_gmem_ptr(params.scatter_ptr_aux[params.rank]), problem_shape, params.dAux);
      return recast<VecType>(group_modes<3, 6>(ThreadMap::partition(mAux, thread_idx, dst_offset)));
    }();

    auto tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));

    auto cAux = make_identity_tensor(problem_shape);
    auto tC_cAux = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, dst_offset)),
        Shape<Int<VecLength>>{},
        (_0{}));

    uint32_t ptr_offset_start = reinterpret_cast<char *>(&tC_gAux(0)) -
                                reinterpret_cast<char *>(params.scatter_ptr_aux[params.rank]);
    uint32_t row_offset_start = ptr_offset_start / sizeof(Element) / N - m_offset_start;
    uint32_t col_offset_start = ptr_offset_start / sizeof(Element) % N;

    auto callback = Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_gAux),
        cute::move(tC_rAux),
        cute::move(tC_cAux),
        problem_shape,
        params_ptr,
        thread_idx,
        tile_idx,
        dst_rank,
        m_end,
        smem_ptr,
        row_offset_start,
        col_offset_start);

    auto flattened_stride = flatten(tC_gAux.stride());
    using Shape_tC_gAux = decltype(flatten(tC_gAux.shape()));
    constexpr int flatten_rank = rank(Shape_tC_gAux{});
    static_assert(flatten_rank == 6);
    const int kStrideUnit = N;
    callback.row_stride[0] = get<0>(flattened_stride) * VecLength / kStrideUnit;
    callback.col_stride[0] = get<0>(flattened_stride) * VecLength % kStrideUnit;
    callback.row_stride[1] = get<1>(flattened_stride) * VecLength / kStrideUnit;
    callback.col_stride[1] = get<1>(flattened_stride) * VecLength % kStrideUnit;
    callback.row_stride[2] = get<2>(flattened_stride) * VecLength / kStrideUnit;
    callback.col_stride[2] = get<2>(flattened_stride) * VecLength % kStrideUnit;
    callback.row_stride[3] = get<3>(flattened_stride) * VecLength / kStrideUnit;
    callback.col_stride[3] = get<3>(flattened_stride) * VecLength % kStrideUnit;
    static_assert(size<4>(Shape_tC_gAux{}) == 1);
    static_assert(size<5>(Shape_tC_gAux{}) == 1);
    return callback;
  }
};

template <class OutputTileThreadMap, class ThreadblockShape, bool FuseReduction>
using GemmRsScatterStoreEVT = VisitorAuxStoreScatterNvlink<
    OutputTileThreadMap,
    cutlass_utils::Element,
    cutlass::FloatRoundStyle::round_to_nearest,
    cutlass_utils::StrideMNL,
    ThreadblockShape,
    FuseReduction>;

}  // namespace myflux::sm80_gemmrs
