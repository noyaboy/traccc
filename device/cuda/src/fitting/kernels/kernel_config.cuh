#pragma once
#include <cuda_runtime.h>

#include "traccc/fitting/fitting_config.hpp"

namespace traccc::cuda::kernels {

#ifdef TRACCC_DEFINE_FITTING_CONFIG
__constant__ traccc::fitting_config g_fitting_cfg;
#else
extern __constant__ traccc::fitting_config g_fitting_cfg;
#endif

void load_fitting_config(const traccc::fitting_config& cfg);

}  // namespace traccc::cuda::kernels
