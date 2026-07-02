#pragma once

#include "kernel_utils.cuh"
#include "sm80_gemmrs_epilogue.cuh"
#include "sm80_gemmrs_swizzle.cuh"

namespace myflux::sm80_gemmrs {

// Flux SM80 GEMMRS FP16/BF16 生成空间里的三个 tile 候选；临时测试时只在下面
// DefaultPlainGemmConfig 这一行切换 tile，不引入 registry/tuning config 胶水。
using ThreadblockShape128x128x32 = cutlass::gemm::GemmShape<128, 128, 32>;
using ThreadblockShape128x128x64 = cutlass::gemm::GemmShape<128, 128, 64>;
using ThreadblockShape128x256x32 = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape64x64x32 = cutlass::gemm::GemmShape<64, 64, 32>;

constexpr int kPlainGemmAlignmentC = 128 / cutlass::sizeof_bits<cutlass_utils::Element>::value;
constexpr int kPlainGemmStages = 3;
constexpr int kGemmRsStages = 3;

enum class GemmRsTileKind {
  k128x128x32,
  k128x128x64,
  k128x256x32,
};

struct GemmRsProblem {
  int m = 0;
  int n = 0;
  int k = 0;
};

struct GemmRsLaunchParams {
  GemmRsProblem problem = {};
  const cutlass_utils::Element *ptr_a = nullptr;
  const cutlass_utils::Element *ptr_b = nullptr;
  cutlass_utils::Element **output_scatter_ptrs = nullptr;
  int rank = 0;
  int world_size = 1;
  float alpha = 1.0f;
  int avail_sms = -1;
};

template <
    class ThreadblockShape_,
    bool FuseReduction_,
    int Stages_ = kGemmRsStages>
struct GemmRsTileConfig {
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape64x64x32;
  using ThreadblockSwizzle = DefaultGemmRsThreadblockSwizzle;
  static constexpr bool FuseReduction = FuseReduction_;
  static constexpr int Stages = Stages_;
  static constexpr int AlignmentC =
      (FuseReduction ? 32 : 128) / cutlass::sizeof_bits<cutlass_utils::Element>::value;
};

using GemmRs128x128x32 = GemmRsTileConfig<ThreadblockShape128x128x32, false>;
using GemmRs128x128x64 = GemmRsTileConfig<ThreadblockShape128x128x64, false>;
using GemmRs128x256x32 = GemmRsTileConfig<ThreadblockShape128x256x32, false>;
using DefaultGemmRsConfig = GemmRs128x128x32;

template <class GemmRsConfig>
struct GemmRsKernelTypes {
  using ThreadblockShape = typename GemmRsConfig::ThreadblockShape;
  using WarpShape = typename GemmRsConfig::WarpShape;
  using ThreadblockSwizzle = typename GemmRsConfig::ThreadblockSwizzle;
  static constexpr bool FuseReduction = GemmRsConfig::FuseReduction;
  static constexpr int AlignmentC = GemmRsConfig::AlignmentC;
  static constexpr int Stages = GemmRsConfig::Stages;

  using OutputTileThreadMap = cutlass_utils::OutputTileThreadMap<
      ThreadblockShape,
      WarpShape,
      AlignmentC>;

  using StoreD = GemmRsScatterStoreEVT<OutputTileThreadMap, ThreadblockShape, FuseReduction>;
  using ComputeD = cutlass_utils::AlphaAccumEVT<>;
  using Epilogue = cutlass_utils::Sm80EpilogueEVT<StoreD, ComputeD>;

  using Gemm = cutlass_utils::Sm80RcrGemmWithVisitor<
      ThreadblockShape,
      WarpShape,
      AlignmentC,
      Stages,
      Epilogue,
      ThreadblockSwizzle>;

  using Device = typename Gemm::Device;
  using Arguments = typename Gemm::Arguments;
};

// 后续实现时在这里做 host 侧薄检查：rank/world_size、scatter ptr、M/world_size、
// tiled_m/world_size 以及当前是否要求 M 对齐 ThreadblockShape::kM。
template <class GemmRsConfig>
typename GemmRsKernelTypes<GemmRsConfig>::Arguments
make_gemmrs_args(const GemmRsLaunchParams &params);

template <class GemmRsConfig>
size_t
gemmrs_workspace_size(const typename GemmRsKernelTypes<GemmRsConfig>::Arguments &args);

template <class GemmRsConfig>
void
run_gemmrs(
    const typename GemmRsKernelTypes<GemmRsConfig>::Arguments &args,
    void *workspace,
    cudaStream_t stream);

using PlainComputeD = cutlass_utils::AlphaAccumEVT<>;

template <class ThreadblockShape_, int Stages_ = kPlainGemmStages>
struct PlainGemmTileConfig {
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape64x64x32;
  static constexpr int AlignmentC = kPlainGemmAlignmentC;
  static constexpr int Stages = Stages_;

  using OutputTileThreadMap = cutlass_utils::OutputTileThreadMap<
      ThreadblockShape,
      WarpShape,
      AlignmentC>;

  using StoreD = cutlass_utils::PlainStoreEVT<OutputTileThreadMap, AlignmentC>;
  using ComputeD = PlainComputeD;
  using Epilogue = cutlass_utils::Sm80EpilogueEVT<StoreD, ComputeD>;

  using Gemm = cutlass_utils::Sm80RcrGemmWithVisitor<
      ThreadblockShape,
      WarpShape,
      AlignmentC,
      Stages,
      Epilogue>;

  using Device = typename Gemm::Device;
  using Arguments = typename Gemm::Arguments;
};

using PlainGemm128x128x32 = PlainGemmTileConfig<ThreadblockShape128x128x32>;
using PlainGemm128x128x64 = PlainGemmTileConfig<ThreadblockShape128x128x64>;
using PlainGemm128x256x32 = PlainGemmTileConfig<ThreadblockShape128x256x32>;

// 这里切换临时测试使用的 tile：PlainGemm128x128x32 / PlainGemm128x128x64 /
// PlainGemm128x256x32。当前默认对齐 Flux TP8 NVLink RCR 常见配置。
using DefaultPlainGemmConfig = PlainGemm128x128x32;

using PlainOutputTileThreadMap128 = typename DefaultPlainGemmConfig::OutputTileThreadMap;
using PlainStoreD128 = typename DefaultPlainGemmConfig::StoreD;
using PlainEpilogue128 = typename DefaultPlainGemmConfig::Epilogue;
using PlainGemm128 = typename DefaultPlainGemmConfig::Gemm;
using PlainGemm128Device = typename DefaultPlainGemmConfig::Device;
using PlainGemm128Arguments = typename DefaultPlainGemmConfig::Arguments;

template <class PlainConfig>
auto
make_plain_epilogue_args(
    typename cutlass_utils::Element *ptr_d,
    int m,
    int n,
    float alpha = 1.0f) {
  using Epilogue = typename PlainConfig::Epilogue;
  using EpilogueArguments = typename Epilogue::Arguments;
  using ComputeArguments = typename PlainConfig::ComputeD::Arguments;
  using StoreArguments = typename PlainConfig::StoreD::Arguments;

  auto stride_d = cute::make_stride(int64_t(n), cute::_1{}, int64_t(m) * n);
  ComputeArguments compute_args{{{alpha}}, {}, {}};
  StoreArguments store_args{ptr_d, stride_d};
  return EpilogueArguments{compute_args, store_args};
}

template <class PlainConfig>
inline typename PlainConfig::Arguments
make_plain_gemm_args(
    int m,
    int n,
    int k,
    const cutlass_utils::Element *ptr_a,
    const cutlass_utils::Element *ptr_b,
    cutlass_utils::Element *ptr_d,
    int avail_sms = -1) {
  auto epilogue_args = make_plain_epilogue_args<PlainConfig>(ptr_d, m, n, 1.0f);
  int stride_b = cutlass_utils::stride_b_rcr(n, k);
  int stride_d = cutlass_utils::stride_c_rowmajor(m, n);

  return typename PlainConfig::Arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k},
      1,
      epilogue_args,
      ptr_a,
      ptr_b,
      nullptr,
      nullptr,
      int64_t(m) * k,
      int64_t(n) * k,
      int64_t(m) * n,
      int64_t(m) * n,
      k,
      stride_b,
      n,
      stride_d,
      avail_sms);
}

template <class PlainConfig>
inline size_t
plain_gemm_workspace_size(const typename PlainConfig::Arguments &args) {
  return PlainConfig::Device::get_workspace_size(args);
}

template <class PlainConfig>
inline void
run_plain_gemm(
    const typename PlainConfig::Arguments &args,
    void *workspace,
    cudaStream_t stream) {
  typename PlainConfig::Device gemm;
  MYFLUX_CHECK_CUTLASS(PlainConfig::Device::can_implement(args));
  MYFLUX_CHECK_CUTLASS(gemm.initialize(args, workspace, stream));
  MYFLUX_CHECK_CUTLASS(gemm.run(stream));
}

inline PlainGemm128Arguments
make_plain_gemm128_args(
    int m,
    int n,
    int k,
    const cutlass_utils::Element *ptr_a,
    const cutlass_utils::Element *ptr_b,
    cutlass_utils::Element *ptr_d,
    int avail_sms = -1) {
  return make_plain_gemm_args<DefaultPlainGemmConfig>(m, n, k, ptr_a, ptr_b, ptr_d, avail_sms);
}

inline size_t
plain_gemm128_workspace_size(const PlainGemm128Arguments &args) {
  return plain_gemm_workspace_size<DefaultPlainGemmConfig>(args);
}

inline void
run_plain_gemm128(
    const PlainGemm128Arguments &args,
    void *workspace,
    cudaStream_t stream) {
  run_plain_gemm<DefaultPlainGemmConfig>(args, workspace, stream);
}

}  // namespace myflux::sm80_gemmrs
