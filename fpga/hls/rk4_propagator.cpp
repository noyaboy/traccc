/**
 * TRACCC FPGA RK4 Propagator - HLS Kernel Implementation
 *
 * This file contains the top-level HLS kernel for batch RK4 propagation.
 * Targeting AMD Alveo V80 with DSP58 native FP32 operations.
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 * Mozilla Public License Version 2.0
 */

#include "rk4_propagator.hpp"

namespace traccc_fpga {

//==============================================================================
// B-Field Grid Implementation
//==============================================================================

Vec3 BFieldGrid::at(float x, float y, float z) const {
#pragma HLS INLINE

    // Clamp to grid bounds
    float cx = std::fmax(BFIELD_MIN_X, std::fmin(x, BFIELD_MAX_X - 1.0f));
    float cy = std::fmax(BFIELD_MIN_Y, std::fmin(y, BFIELD_MAX_Y - 1.0f));
    float cz = std::fmax(BFIELD_MIN_Z, std::fmin(z, BFIELD_MAX_Z - 1.0f));

    // Calculate grid indices
    float fx = (cx - BFIELD_MIN_X) / dx;
    float fy = (cy - BFIELD_MIN_Y) / dy;
    float fz = (cz - BFIELD_MIN_Z) / dz;

    int ix = static_cast<int>(fx);
    int iy = static_cast<int>(fy);
    int iz = static_cast<int>(fz);

    // Clamp indices
    ix = std::max(0, std::min(ix, BFIELD_NX - 2));
    iy = std::max(0, std::min(iy, BFIELD_NY - 2));
    iz = std::max(0, std::min(iz, BFIELD_NZ - 2));

    // Interpolation weights
    float wx = fx - ix;
    float wy = fy - iy;
    float wz = fz - iz;

    // Trilinear interpolation
    // Grid layout: [iz][iy][ix][3] -> index = ((iz * NY + iy) * NX + ix) * 3
    auto idx = [](int x, int y, int z) -> int {
        return ((z * BFIELD_NY + y) * BFIELD_NX + x) * 3;
    };

    Vec3 result;

    if (data == nullptr) {
        // Default constant field if no grid data (2T in z, converted to native units)
        return Vec3(0.0f, 0.0f, 2.0f * UNIT_T);
    }

    // 8-corner interpolation for each component
    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL

        float v000 = data[idx(ix, iy, iz) + c];
        float v100 = data[idx(ix + 1, iy, iz) + c];
        float v010 = data[idx(ix, iy + 1, iz) + c];
        float v110 = data[idx(ix + 1, iy + 1, iz) + c];
        float v001 = data[idx(ix, iy, iz + 1) + c];
        float v101 = data[idx(ix + 1, iy, iz + 1) + c];
        float v011 = data[idx(ix, iy + 1, iz + 1) + c];
        float v111 = data[idx(ix + 1, iy + 1, iz + 1) + c];

        // Trilinear interpolation
        float v00 = v000 * (1.0f - wx) + v100 * wx;
        float v01 = v001 * (1.0f - wx) + v101 * wx;
        float v10 = v010 * (1.0f - wx) + v110 * wx;
        float v11 = v011 * (1.0f - wx) + v111 * wx;

        float v0 = v00 * (1.0f - wy) + v10 * wy;
        float v1 = v01 * (1.0f - wy) + v11 * wy;

        float v = v0 * (1.0f - wz) + v1 * wz;

        if (c == 0) result.x = v;
        else if (c == 1) result.y = v;
        else result.z = v;
    }

    return result;
}

//==============================================================================
// Coordinate Conversion Utilities
//==============================================================================

/**
 * Convert bound parameters to free state
 *
 * Note: This is simplified - actual implementation needs surface transform.
 * For HLS prototype, we assume a simple planar surface at z=0.
 */
HLS_INLINE FreeTrackState bound_to_free(const TrackParams& bound) {
#pragma HLS INLINE

    FreeTrackState free;

    // Simplified: assume surface at z=0, perpendicular to z-axis
    free.pos.x = bound.loc0;
    free.pos.y = bound.loc1;
    free.pos.z = 0.0f;

    // Direction from angles
    float sin_theta = std::sin(bound.theta);
    float cos_theta = std::cos(bound.theta);
    float sin_phi = std::sin(bound.phi);
    float cos_phi = std::cos(bound.phi);

    free.dir.x = sin_theta * cos_phi;
    free.dir.y = sin_theta * sin_phi;
    free.dir.z = cos_theta;

    free.qop = bound.qop;
    free.time = bound.time;
    free.path_length = 0.0f;

    return free;
}

/**
 * Convert free state back to bound parameters
 *
 * Note: Simplified - actual implementation needs surface intersection.
 */
HLS_INLINE TrackParams free_to_bound(const FreeTrackState& free) {
#pragma HLS INLINE

    TrackParams bound;

    // Simplified: project to z=target_z surface
    bound.loc0 = free.pos.x;
    bound.loc1 = free.pos.y;

    // Direction to angles
    float r = std::sqrt(free.dir.x * free.dir.x + free.dir.y * free.dir.y);
    bound.theta = std::atan2(r, free.dir.z);
    bound.phi = std::atan2(free.dir.y, free.dir.x);

    bound.qop = free.qop;
    bound.time = free.time;

    return bound;
}

}  // namespace traccc_fpga

//==============================================================================
// HLS Top-Level Kernel (outside namespace for extern "C" linkage)
//==============================================================================

using namespace traccc_fpga;

/**
 * Batch RK4 Propagation Kernel
 *
 * Processes n_tracks in parallel, each propagating to its next surface.
 *
 * Interface pragmas configure AXI memory-mapped interfaces for V80:
 * - in_params, out_params: AXI master connected to HBM2e
 * - bfield_data: AXI master connected to HBM2e (large grid)
 * - distances: AXI master (small array)
 */
extern "C" void rk4_propagate_kernel(
    int n_tracks,
    const TrackParams* in_params,
    TrackParams* out_params,
    const float* bfield_data,
    const float* distances
) {
#pragma HLS INTERFACE m_axi port=in_params offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out_params offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=bfield_data offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=distances offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=n_tracks
#pragma HLS INTERFACE s_axilite port=return

    // Set up B-field grid accessor
    BFieldGrid bfield;
    bfield.data = bfield_data;
    bfield.dx = (BFIELD_MAX_X - BFIELD_MIN_X) / (BFIELD_NX - 1);
    bfield.dy = (BFIELD_MAX_Y - BFIELD_MIN_Y) / (BFIELD_NY - 1);
    bfield.dz = (BFIELD_MAX_Z - BFIELD_MIN_Z) / (BFIELD_NZ - 1);

    // Process tracks
    TRACK_LOOP:
    for (int i = 0; i < n_tracks; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=42240 avg=6666
#pragma HLS PIPELINE II=1

        // Read input track
        TrackParams in_track = in_params[i];
        float dist = distances[i];

        // Convert to free state
        FreeTrackState free_state = bound_to_free(in_track);

        // Propagate
        PropagationResult result = propagate_to_surface(free_state, bfield, dist);

        // Convert back to bound parameters
        TrackParams out_track = free_to_bound(result.state);

        // Write output
        out_params[i] = out_track;
    }
}

//==============================================================================
// Alternative Kernel: Constant B-Field (for testing/validation)
//==============================================================================

/**
 * Simplified kernel with constant B-field
 * Useful for initial validation and synthesis estimates.
 */
extern "C" void rk4_propagate_const_bfield(
    int n_tracks,
    const TrackParams* in_params,
    TrackParams* out_params,
    float bfield_x,
    float bfield_y,
    float bfield_z,
    const float* distances
) {
#pragma HLS INTERFACE m_axi port=in_params offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out_params offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=distances offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=n_tracks
#pragma HLS INTERFACE s_axilite port=bfield_x
#pragma HLS INTERFACE s_axilite port=bfield_y
#pragma HLS INTERFACE s_axilite port=bfield_z
#pragma HLS INTERFACE s_axilite port=return

    // Constant B-field
    ConstBField bfield(bfield_x, bfield_y, bfield_z);

    // Process tracks
    TRACK_LOOP_CONST:
    for (int i = 0; i < n_tracks; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=42240 avg=6666
#pragma HLS PIPELINE II=1

        // Read input track
        TrackParams in_track = in_params[i];
        float dist = distances[i];

        // Convert to free state
        FreeTrackState free_state = bound_to_free(in_track);

        // Propagate with constant field
        PropagationResult result = propagate_to_surface(free_state, bfield, dist);

        // Convert back to bound parameters
        TrackParams out_track = free_to_bound(result.state);

        // Write output
        out_params[i] = out_track;
    }
}
