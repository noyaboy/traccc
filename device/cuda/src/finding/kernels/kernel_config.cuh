#pragma once
#include "traccc/finding/finding_config.hpp"
#include <cuda_runtime.h>

namespace traccc::cuda::kernels {
extern __device__ __constant__ finding_config g_finding_cfg;
void load_finding_config(const finding_config& cfg);
}
