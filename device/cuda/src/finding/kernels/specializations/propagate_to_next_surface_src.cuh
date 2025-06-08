/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../../../utils/global_index.hpp"
#include "../propagate_to_next_surface.cuh"
#include "../kernel_config.cuh"

// Project include(s).
#include "traccc/finding/device/propagate_to_next_surface.hpp"

namespace traccc::cuda::kernels {

template <typename propagator_t, typename bfield_t>
__global__ void propagate_to_next_surface(
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    extern __shared__ unsigned int s_param_ids[];
    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);

    const unsigned int global_idx = details::global_index1();
    if (global_idx < param_ids.size()) {
        s_param_ids[threadIdx.x] = param_ids.at(global_idx);
    }
    __syncthreads();

    if (global_idx < param_ids.size()) {
        device::propagate_to_next_surface<propagator_t, bfield_t>(
            details::global_index1(), g_finding_cfg, payload,
            s_param_ids[threadIdx.x]);
    }
}

}  // namespace traccc::cuda::kernels
