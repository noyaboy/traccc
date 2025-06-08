#define TRACCC_DEFINE_FINDING_CONFIG
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

void load_finding_config(const finding_config& cfg) {
    cudaMemcpyToSymbol(g_finding_cfg, &cfg, sizeof(finding_config));
}
}  // namespace traccc::cuda::kernels
