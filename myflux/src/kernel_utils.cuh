#pragma once

// 这里放 myflux kernel 侧的薄工具层：公共 include、固定场景的 CUTLASS 类型别名、
// 以及 host 侧检查工具。真正的 GEMMRS epilogue / swizzle 不放在这里。

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/fusion/visitor_store.hpp"

namespace myflux {

constexpr int kMaxWorldSize = 8;

#define MYFLUX_HOST_DEVICE CUTLASS_HOST_DEVICE
#define MYFLUX_DEVICE CUTLASS_DEVICE

template <typename T>
MYFLUX_HOST_DEVICE constexpr T
ceil_div(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
MYFLUX_HOST_DEVICE constexpr T
align_up(T x, T align) {
  return ceil_div(x, align) * align;
}

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

inline void
check_cutlass(cutlass::Status status, const char *expr, const char *file, int line) {
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(
        stderr,
        "CUTLASS error at %s:%d: %s failed with status %d\n",
        file,
        line,
        expr,
        static_cast<int>(status));
    std::abort();
  }
}

#define MYFLUX_CHECK_CUDA(expr) ::myflux::check_cuda((expr), #expr, __FILE__, __LINE__)
#define MYFLUX_CHECK_CUTLASS(expr) ::myflux::check_cutlass((expr), #expr, __FILE__, __LINE__)

namespace cutlass_utils {

using Element = cutlass::half_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ArchTag = cutlass::arch::Sm80;
using OpClass = cutlass::arch::OpClassTensorOp;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using Operator = cutlass::arch::OpMultiplyAdd;
using StrideMNL = cute::Stride<int64_t, cute::_1, int64_t>;

constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<Element>::value;
constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<Element>::value;
constexpr int kEpilogueStages = 1;

template <class ThreadblockShape, class WarpShape, int AlignmentC>
using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape,
    WarpShape,
    Element,
    AlignmentC,
    kEpilogueStages>;

// D = alpha * accumulator. 先不处理 bias/beta，这样临时 case 只验证 CUTLASS 组装。
template <class ElementD = Element>
using AlphaAccumEVT = cutlass::epilogue::threadblock::Sm80EVT<
    cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementD,
        ElementD,
        cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementAccumulator>,
    cutlass::epilogue::threadblock::VisitorAccFetch>;

template <class OutputTileThreadMap_, int AlignmentC>
using PlainStoreEVT = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap_,
    Element,
    cutlass::FloatRoundStyle::round_to_nearest,
    StrideMNL>;

template <class StoreD, class ComputeD>
using Sm80EpilogueEVT = cutlass::epilogue::threadblock::Sm80EVT<StoreD, ComputeD>;

// 固定 RCR half GEMM 的 CUTLASS kernel 拼装器。后面 GEMMRS 只需要把 StoreD 换成
// scatter-reduce store，把 ThreadblockSwizzle 换成 rank-offset swizzle。
template <
    class ThreadblockShape_,
    class WarpShape_,
    int AlignmentC_,
    int Stages_,
    class Epilogue_,
    class ThreadblockSwizzle_ = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>
struct Sm80RcrGemmWithVisitor {
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  static constexpr int AlignmentC = AlignmentC_;
  static constexpr int Stages = Stages_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kAlignmentA,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kAlignmentB,
      Element,
      LayoutC,
      AlignmentC,
      ElementAccumulator,
      Element,
      OpClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      Epilogue,
      ThreadblockSwizzle,
      Stages,
      Operator,
      kEpilogueStages>::GemmKernel;

  using Device = cutlass::gemm::device::GemmUniversalBase<Kernel>;
  using Arguments = typename Device::Arguments;
};

inline int
stride_b_rcr(int n, int k) {
  // B 是 ColumnMajor，逻辑形状 [K, N]，物理上可以复用 row-major [N, K] 权重。
  (void)n;
  return k;
}

inline int
stride_c_rowmajor(int m, int n) {
  (void)m;
  return n;
}

}  // namespace cutlass_utils
}  // namespace myflux
