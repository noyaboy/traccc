/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "../propagate_stage1.cuh"
#include "types.hpp"

namespace traccc::cuda::kernels {

template __global__ void
propagate_stage1<default_finding_algorithm::propagator_type,
                 default_finding_algorithm::bfield_type>(
    const finding_config,
    device::propagate_to_next_surface_payload<
        default_finding_algorithm::propagator_type,
        default_finding_algorithm::bfield_type>);

}  // namespace traccc::cuda::kernels