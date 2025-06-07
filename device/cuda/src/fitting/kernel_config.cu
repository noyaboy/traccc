#define TRACCC_DEFINE_FITTING_CONFIG
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

void load_fitting_config(const fitting_config& cfg) {
    cudaMemcpyToSymbol(g_fit_cfg, &cfg, sizeof(fitting_config));
}

} // namespace traccc::cuda::kernels
