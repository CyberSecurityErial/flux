#include "sm80_gemmrs.cuh"

namespace {

using Shape = myflux::sm80_gemmrs::ThreadblockShape128x128x32;
using WarpShape = myflux::sm80_gemmrs::WarpShape64x64x32;

template <bool FuseReduction>
struct InstantiateGemmRsKernel {
  using Config = myflux::sm80_gemmrs::GemmRsTileConfig<Shape, FuseReduction>;
  using KernelTypes = myflux::sm80_gemmrs::GemmRsKernelTypes<Config>;
  using StoreD = typename KernelTypes::StoreD;
  using Arguments = typename KernelTypes::Arguments;

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
