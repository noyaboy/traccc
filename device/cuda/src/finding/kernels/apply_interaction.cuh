/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/device/apply_interaction.hpp"
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

template <typename detector_t>
__global__ void apply_interaction(
    device::apply_interaction_payload<detector_t> payload);

}  // namespace traccc::cuda::kernels
