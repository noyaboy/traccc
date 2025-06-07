#define TRACCC_DEFINE_SEEDING_CONFIG
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

void load_seeding_config(const seedfinder_config& finder_cfg,
                         const seedfilter_config& filter_cfg) {
    cudaMemcpyToSymbol(g_seedfinder_cfg, &finder_cfg,
                       sizeof(seedfinder_config));
    cudaMemcpyToSymbol(g_seedfilter_cfg, &filter_cfg,
                       sizeof(seedfilter_config));
}

}  // namespace traccc::cuda::kernels
