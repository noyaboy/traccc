#pragma once
#include <cuda_runtime.h>

#include "traccc/finding/finding_config.hpp"

namespace traccc::cuda::kernels {

#ifdef TRACCC_DEFINE_FINDING_CONFIG
__constant__ finding_config g_finding_cfg;
#else
extern __constant__ finding_config g_finding_cfg;
#endif

void load_finding_config(const finding_config& cfg);

}  // namespace traccc::cuda::kernels
