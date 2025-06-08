#pragma once
#include <cuda_runtime.h>

#include "traccc/clusterization/clustering_config.hpp"

namespace traccc::cuda::kernels {

#ifdef TRACCC_DEFINE_CLUSTERING_CONFIG
__constant__ traccc::clustering_config g_clustering_cfg;
#else
extern __constant__ traccc::clustering_config g_clustering_cfg;
#endif

void load_clustering_config(const traccc::clustering_config& cfg);

}  // namespace traccc::cuda::kernels
