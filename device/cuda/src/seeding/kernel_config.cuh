#pragma once
#include <cuda_runtime.h>

#include "traccc/seeding/detail/seeding_config.hpp"

namespace traccc::cuda::kernels {

#ifdef TRACCC_DEFINE_SEEDING_CONFIG
__constant__ alignas(seedfinder_config) unsigned char g_seedfinder_cfg_bytes
    [sizeof(seedfinder_config)];
__constant__ seedfilter_config g_seedfilter_cfg;
#else
extern __constant__ alignas(
    seedfinder_config) unsigned char g_seedfinder_cfg_bytes
    [sizeof(seedfinder_config)];
extern __constant__ seedfilter_config g_seedfilter_cfg;
#endif

TRACCC_DEVICE inline const seedfinder_config& get_seedfinder_cfg() {
    return *reinterpret_cast<const seedfinder_config*>(g_seedfinder_cfg_bytes);
}

void load_seeding_config(const seedfinder_config& finder_cfg,
                         const seedfilter_config& filter_cfg);

}  // namespace traccc::cuda::kernels
