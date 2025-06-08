/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "kernel_config.cuh"

namespace traccc::cuda::kernels {

template <typename propagator_t, typename bfield_t>
__global__ void propagate_to_next_surface(
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload);

}  // namespace traccc::cuda::kernels
