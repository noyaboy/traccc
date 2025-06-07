#pragma once
#include <cuda_runtime.h>

#include "traccc/seeding/detail/seeding_config.hpp"

namespace traccc::cuda::kernels {

#ifdef TRACCC_DEFINE_SEEDING_CONFIG
__constant__ seedfinder_config g_seedfinder_cfg;
__constant__ seedfilter_config g_seedfilter_cfg;
#else
extern __constant__ seedfinder_config g_seedfinder_cfg;
extern __constant__ seedfilter_config g_seedfilter_cfg;
#endif

void load_seeding_config(const seedfinder_config& finder_cfg,
                         const seedfilter_config& filter_cfg);

}  // namespace traccc::cuda::kernels
