#define TRACCC_DEFINE_CLUSTERING_CONFIG
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

void load_clustering_config(const traccc::clustering_config& cfg) {
    cudaMemcpyToSymbol(g_clustering_cfg, &cfg,
                       sizeof(traccc::clustering_config));
}

}  // namespace traccc::cuda::kernels
