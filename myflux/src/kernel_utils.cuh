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

// epilogue阶段每个thread结束作业后发射出去的每块tile定义
template <class ThreadblockShape, class WarpShape, int AlignmentC>
using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape,
    WarpShape,
    Element,
    AlignmentC,
    kEpilogueStages>;

// EVT是epi阶段的数据流定义方式。根据模版参数是第几个决定这个节点在树里面的位置，是根还是叶子，还是内部。                 
template <class ElementD = Element>
using AlphaAccumEVT = cutlass::epilogue::threadblock::Sm80EVT<
    cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementD,
        ElementD,
        cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementAccumulator>,
    cutlass::epilogue::threadblock::VisitorAccFetch>;

// 消费EVT产生的结果，写到每个thread的tile
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


namespace peer_barrier {

struct Args {
  int *sync_buffers[kMaxWorldSize];
  int rank = 0;
  int world_size = 1;
};

namespace detail {

static __device__ int
ld_acquire_sys(volatile int *ptr) {
  int state = 0;
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
  return state;
}

static __device__ void
store_release_sys(volatile int *ptr, int value) {
  asm volatile("st.release.sys.b32 [%0], %1;\n" : : "l"(ptr), "r"(value));
}

static __global__ void
all_to_all_atomic_kernel(Args args) {
  int world_size = args.world_size;
  int cur_rank = args.rank;
  if (threadIdx.x < world_size) {
    __threadfence_system();
    int *sync_buffer_dst = args.sync_buffers[threadIdx.x] + cur_rank;
#pragma unroll 1
    while (atomicCAS_system(sync_buffer_dst, 0, 1) != 0) {
    }
    __threadfence_system();
    int *wait_ptr = args.sync_buffers[cur_rank] + threadIdx.x;
#pragma unroll 1
    while (atomicCAS_system(wait_ptr, 1, 0) != 1) {
    }
    __threadfence_system();
  }
}

// Ring fallback mirrors Flux's CUDA IPC ring barrier. It avoids system atomic CAS,
// but uses a serialized token and is more fragile for continuous barriers.
static __global__ void
ring_kernel(Args args) {
  int world_size = args.world_size;
  int cur_rank = args.rank;
  int next_peer = (cur_rank + 1) % world_size;
  volatile int *ptr_next_peer = args.sync_buffers[next_peer];
  volatile int *ptr_cur_rank = args.sync_buffers[cur_rank];
  if (threadIdx.x != 0) {
    return;
  }
  if (cur_rank == 0) {
    store_release_sys(ptr_next_peer, 1);
  } else {
#pragma unroll 1
    while (ld_acquire_sys(ptr_cur_rank) != 1) {
    }
    store_release_sys(ptr_next_peer, 1);
  }
  __threadfence_system();
  if (cur_rank != world_size - 1) {
#pragma unroll 1
    while (ld_acquire_sys(ptr_next_peer) != 0) {
    }
  }
  ptr_cur_rank[0] = 0;
}

}  // namespace detail

inline void
launch_all_to_all_atomic(int **sync_buffers, int rank, int world_size, cudaStream_t stream) {
  if (world_size <= 1) {
    return;
  }
  if (world_size > kMaxWorldSize) {
    std::fprintf(stderr, "myflux peer barrier world_size exceeds kMaxWorldSize\n");
    std::abort();
  }
  Args args;
  args.rank = rank;
  args.world_size = world_size;
  for (int i = 0; i < world_size; ++i) {
    args.sync_buffers[i] = sync_buffers[i];
  }
  detail::all_to_all_atomic_kernel<<<1, kMaxWorldSize, 0, stream>>>(args);
  MYFLUX_CHECK_CUDA(cudaGetLastError());
}

inline void
launch_ring(int **sync_buffers, int rank, int world_size, cudaStream_t stream) {
  if (world_size <= 1) {
    return;
  }
  if (world_size > kMaxWorldSize) {
    std::fprintf(stderr, "myflux peer barrier world_size exceeds kMaxWorldSize\n");
    std::abort();
  }
  Args args;
  args.rank = rank;
  args.world_size = world_size;
  for (int i = 0; i < world_size; ++i) {
    args.sync_buffers[i] = sync_buffers[i];
  }
  detail::ring_kernel<<<1, kMaxWorldSize, 0, stream>>>(args);
  MYFLUX_CHECK_CUDA(cudaGetLastError());
}

inline size_t
workspace_bytes(int world_size) {
  return static_cast<size_t>(world_size) * sizeof(int);
}

}  // namespace peer_barrier

}  // namespace myflux
