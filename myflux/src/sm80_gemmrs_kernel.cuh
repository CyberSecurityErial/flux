#pragma once

#include "kernel_utils.cuh"

namespace myflux::sm80_gemmrs {

// 第一版只固定一个 tile，方便先把 CUTLASS 组装和 launcher 跑通。
using ThreadblockShape128x128x32 = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape64x64x32 = cutlass::gemm::GemmShape<64, 64, 32>;

constexpr int kPlainGemmAlignmentC = 128 / cutlass::sizeof_bits<cutlass_utils::Element>::value;
constexpr int kPlainGemmStages = 3;

using PlainOutputTileThreadMap128 = cutlass_utils::OutputTileThreadMap<
    ThreadblockShape128x128x32,
    WarpShape64x64x32,
    kPlainGemmAlignmentC>;

using PlainStoreD128 = cutlass_utils::PlainStoreEVT<
    PlainOutputTileThreadMap128,
    kPlainGemmAlignmentC>;

using PlainComputeD = cutlass_utils::AlphaAccumEVT<>;
using PlainEpilogue128 = cutlass_utils::Sm80EpilogueEVT<PlainStoreD128, PlainComputeD>;

using PlainGemm128 = cutlass_utils::Sm80RcrGemmWithVisitor<
    ThreadblockShape128x128x32,
    WarpShape64x64x32,
    kPlainGemmAlignmentC,
    kPlainGemmStages,
    PlainEpilogue128>;

using PlainGemm128Device = typename PlainGemm128::Device;
using PlainGemm128Arguments = typename PlainGemm128::Arguments;

template <class GemmConfig>
auto
make_plain_epilogue_args(
    typename cutlass_utils::Element *ptr_d,
    int m,
    int n,
    float alpha = 1.0f) {
  using Epilogue = typename GemmConfig::Epilogue;
  using EpilogueArguments = typename Epilogue::Arguments;
  using ComputeArguments = typename PlainComputeD::Arguments;
  using StoreArguments = typename PlainStoreD128::Arguments;

  auto stride_d = cute::make_stride(int64_t(n), cute::_1{}, int64_t(m) * n);
  ComputeArguments compute_args{{{alpha}}, {}, {}};
  StoreArguments store_args{ptr_d, stride_d};
  return EpilogueArguments{compute_args, store_args};
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
  auto epilogue_args = make_plain_epilogue_args<PlainGemm128>(ptr_d, m, n, 1.0f);
  int stride_b = cutlass_utils::stride_b_rcr(n, k);
  int stride_d = cutlass_utils::stride_c_rowmajor(m, n);

  return PlainGemm128Arguments(
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

inline size_t
plain_gemm128_workspace_size(const PlainGemm128Arguments &args) {
  return PlainGemm128Device::get_workspace_size(args);
}

inline void
run_plain_gemm128(
    const PlainGemm128Arguments &args,
    void *workspace,
    cudaStream_t stream) {
  PlainGemm128Device gemm;
  MYFLUX_CHECK_CUTLASS(PlainGemm128Device::can_implement(args));
  MYFLUX_CHECK_CUTLASS(gemm.initialize(args, workspace, stream));
  MYFLUX_CHECK_CUTLASS(gemm.run(stream));
}

}  // namespace myflux::sm80_gemmrs
