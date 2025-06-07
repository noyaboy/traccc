#pragma once

#include <cuda_runtime.h>

#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

/// Simple struct-of-arrays view used for the CUDA fitting kernels.
struct track_candidate_soa {
    float4* __restrict__ loc_var = nullptr;  ///< local0, local1, var0, var1
    unsigned int* __restrict__ offsets =
        nullptr;  ///< prefix sum offsets per track
};

struct track_state_soa {
    float4* __restrict__ loc_var = nullptr;  ///< local0, local1, var0, var1
    unsigned int* __restrict__ offsets =
        nullptr;  ///< prefix sum offsets per track
};

}  // namespace traccc::device
