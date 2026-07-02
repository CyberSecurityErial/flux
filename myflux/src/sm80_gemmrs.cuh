#pragma once

#include "kernel_utils.cuh"

namespace myflux::sm80_gemmrs {

// Flux SM80 GEMMRS FP16/BF16 生成空间里的三个 tile 候选；临时测试时只在下面
// DefaultPlainGemmConfig 这一行切换 tile，不引入 registry/tuning config 胶水。
using ThreadblockShape128x128x32 = cutlass::gemm::GemmShape<128, 128, 32>;
using ThreadblockShape128x128x64 = cutlass::gemm::GemmShape<128, 128, 64>;
using ThreadblockShape128x256x32 = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape64x64x32 = cutlass::gemm::GemmShape<64, 64, 32>;

constexpr int kPlainGemmAlignmentC = 128 / cutlass::sizeof_bits<cutlass_utils::Element>::value;
constexpr int kPlainGemmStages = 3;

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
