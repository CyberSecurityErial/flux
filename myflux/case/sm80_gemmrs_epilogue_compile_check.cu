#include "sm80_gemmrs.cuh"

namespace {

using Shape = myflux::sm80_gemmrs::ThreadblockShape128x128x32;
using WarpShape = myflux::sm80_gemmrs::WarpShape64x64x32;

template <bool FuseReduction>
struct InstantiateGemmRsKernel {
  static constexpr int AlignmentC =
      (FuseReduction ? 32 : 128) / cutlass::sizeof_bits<myflux::cutlass_utils::Element>::value;

  using OutputTileThreadMap =
      myflux::cutlass_utils::OutputTileThreadMap<Shape, WarpShape, AlignmentC>;
  using StoreD = myflux::sm80_gemmrs::GemmRsScatterStoreEVT<
      OutputTileThreadMap,
      Shape,
      FuseReduction>;
  using ComputeD = myflux::cutlass_utils::AlphaAccumEVT<>;
  using Epilogue = myflux::cutlass_utils::Sm80EpilogueEVT<StoreD, ComputeD>;
  using Gemm = myflux::cutlass_utils::Sm80RcrGemmWithVisitor<
      Shape,
      WarpShape,
      AlignmentC,
      3,
      Epilogue,
      myflux::sm80_gemmrs::DefaultGemmRsThreadblockSwizzle>;
  using Device = typename Gemm::Device;
  using Arguments = typename Device::Arguments;

  static_assert(StoreD::VecLength > 0);
  static_assert(sizeof(typename StoreD::Params) > 0);
  static_assert(sizeof(typename StoreD::SharedStorage) > 0);
  static_assert(sizeof(Arguments) > 0);
};

using ScatterStoreKernel = InstantiateGemmRsKernel<false>;
using ScatterReduceKernel = InstantiateGemmRsKernel<true>;

static_assert(sizeof(ScatterStoreKernel) > 0);
static_assert(sizeof(ScatterReduceKernel) > 0);

}  // namespace

int
main() {
  return 0;
}
