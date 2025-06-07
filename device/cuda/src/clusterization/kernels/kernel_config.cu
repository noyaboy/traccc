#define TRACCC_DEFINE_CLUSTERING_CONFIG
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

void load_clustering_config(const clustering_config& cfg) {
    cudaMemcpyToSymbol(g_clust_cfg, &cfg, sizeof(clustering_config));
}

} // namespace traccc::cuda::kernels
