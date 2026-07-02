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
  using Element = myflux::cutlass_utils::Element;

  static_assert(StoreD::VecLength > 0);
  static_assert(sizeof(typename StoreD::Params) > 0);
  static_assert(sizeof(typename StoreD::SharedStorage) > 0);
  static_assert(sizeof(Arguments) > 0);

  static void make_args() {
    Element *scatter_ptrs[myflux::kMaxWorldSize] = {};
    auto ptr_a = reinterpret_cast<Element const *>(0x1000);
    auto ptr_b = reinterpret_cast<Element const *>(0x2000);
    auto ptr_d = reinterpret_cast<Element *>(0x3000);
    scatter_ptrs[0] = ptr_d;

    myflux::sm80_gemmrs::GemmRsLaunchParams params;
    params.problem = {Shape::kM, Shape::kN, Shape::kK};
    params.ptr_a = ptr_a;
    params.ptr_b = ptr_b;
    params.output_scatter_ptrs = scatter_ptrs;
    params.rank = 0;
    params.world_size = 1;
    params.alpha = 1.0f;
    params.avail_sms = -1;

    auto args = myflux::sm80_gemmrs::make_gemmrs_args<Config>(params);
    (void)args;
  }
};

using ScatterStoreKernel = InstantiateGemmRsKernel<false>;
using ScatterReduceKernel = InstantiateGemmRsKernel<true>;

static_assert(sizeof(ScatterStoreKernel) > 0);
static_assert(sizeof(ScatterReduceKernel) > 0);

}  // namespace

int
main() {
  ScatterStoreKernel::make_args();
  ScatterReduceKernel::make_args();
  return 0;
}
