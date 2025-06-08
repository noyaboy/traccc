#define TRACCC_DEFINE_FITTING_CONFIG
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

void load_fitting_config(const traccc::fitting_config& cfg) {
    cudaMemcpyToSymbol(g_fitting_cfg, &cfg, sizeof(traccc::fitting_config));
}

}  // namespace traccc::cuda::kernels
