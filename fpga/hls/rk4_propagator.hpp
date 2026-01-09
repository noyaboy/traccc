/**
 * TRACCC FPGA RK4 Propagator - HLS Implementation
 *
 * Extracted from detray rk_stepper.ipp for Vitis HLS synthesis.
 * This is a standalone, template-free implementation targeting AMD Alveo V80.
 *
 * Reference: https://doi.org/10.1016/0029-554X(81)90063-X
 *            (Runge-Kutta-Nystrom integration for track propagation)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 * Mozilla Public License Version 2.0
 */

#ifndef RK4_PROPAGATOR_HPP
#define RK4_PROPAGATOR_HPP

#include <cmath>
#include <cstdint>

// For HLS synthesis
#ifdef __SYNTHESIS__
#include <ap_fixed.h>
#include <hls_math.h>
#define HLS_INLINE __attribute__((always_inline))
#else
#define HLS_INLINE inline
#include <algorithm>
#endif

namespace traccc_fpga {

//==============================================================================
// Configuration Constants
//==============================================================================

// RK4 stepping configuration
constexpr float RK_ERROR_TOL = 1e-4f;       // Error tolerance for adaptive stepping
constexpr float MIN_STEP_SIZE = 1e-4f;      // Minimum step size in mm
constexpr float MAX_STEP_SIZE = 1000.0f;    // Maximum step size in mm
constexpr int MAX_RK_TRIALS = 10;           // Max iterations for step size adjustment
constexpr int MAX_STEPS_PER_SURFACE = 1000; // Max RK4 steps per propagation

// Unit conversions (matching detray/ACTS conventions)
// Native units: length=mm, energy=GeV, charge=e, B-field=GeV/(e*mm)
// 1 Tesla = 0.000299792458 GeV/(e*mm) = c * 1e-9 in SI
constexpr float UNIT_T = 0.000299792458f;   // Tesla to native B-field units

// B-field grid dimensions (standard configuration)
constexpr int BFIELD_NX = 201;
constexpr int BFIELD_NY = 201;
constexpr int BFIELD_NZ = 301;
constexpr float BFIELD_MIN_X = -10000.0f;   // mm
constexpr float BFIELD_MAX_X = 10000.0f;    // mm
constexpr float BFIELD_MIN_Y = -10000.0f;   // mm
constexpr float BFIELD_MAX_Y = 10000.0f;    // mm
constexpr float BFIELD_MIN_Z = -15000.0f;   // mm
constexpr float BFIELD_MAX_Z = 15000.0f;    // mm

//==============================================================================
// Data Structures
//==============================================================================

/**
 * 3D Vector - used for positions, directions, and B-field values
 */
struct Vec3 {
    float x, y, z;

    HLS_INLINE Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    HLS_INLINE Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    HLS_INLINE Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    HLS_INLINE Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    HLS_INLINE Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }

    HLS_INLINE float norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    HLS_INLINE Vec3 normalized() const {
        float n = norm();
        return (n > 1e-10f) ? Vec3(x / n, y / n, z / n) : *this;
    }
};

HLS_INLINE Vec3 operator*(float s, const Vec3& v) {
    return Vec3(s * v.x, s * v.y, s * v.z);
}

/**
 * Cross product: a × b
 */
HLS_INLINE Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

/**
 * Track Parameters - 6 parameters in bound representation
 *
 * loc0, loc1: local position on surface
 * phi, theta: direction angles
 * qop: charge over momentum (q/p) in 1/GeV
 * time: time coordinate
 */
struct TrackParams {
    float loc0;     // Local coordinate 0 (mm)
    float loc1;     // Local coordinate 1 (mm)
    float phi;      // Azimuthal angle (rad)
    float theta;    // Polar angle (rad)
    float qop;      // q/p (1/GeV)
    float time;     // Time (ns)
};

/**
 * Free Track State - 8 parameters for propagation
 *
 * Used internally during RK4 stepping.
 */
struct FreeTrackState {
    Vec3 pos;       // Position (mm)
    Vec3 dir;       // Direction (unit vector)
    float qop;      // q/p (1/GeV)
    float time;     // Time (ns)
    float path_length;  // Accumulated path length (mm)
};

/**
 * RK4 Intermediate State
 *
 * Stores the k1, k2, k3, k4 values for RK4 integration.
 */
struct RK4State {
    Vec3 b_first;   // B-field at initial point
    Vec3 b_middle;  // B-field at midpoint
    Vec3 b_last;    // B-field at final point

    Vec3 t[4];      // Tangent direction at each RK stage
    Vec3 dtds[4];   // d(tangent)/ds = qop * (t × B) at each stage
    float qop[4];   // q/p at each stage (for material effects)
};

/**
 * Propagation Result
 */
struct PropagationResult {
    FreeTrackState state;   // Final track state
    float chi2;             // Chi-squared (if measurement matching done)
    uint8_t status;         // 0=success, 1=failed, 2=max_steps
    uint8_t n_steps;        // Number of RK4 steps taken
};

/**
 * B-Field Grid Access
 *
 * Simple trilinear interpolation on a regular 3D grid.
 * In actual HLS, this would read from HBM2e.
 */
struct BFieldGrid {
    // Grid data pointer (in HBM2e on V80)
    // Layout: [NZ][NY][NX][3] for Bx, By, Bz
    const float* data;

    // Grid spacing
    float dx, dy, dz;

    HLS_INLINE BFieldGrid() : data(nullptr), dx(100.0f), dy(100.0f), dz(100.0f) {}

    /**
     * Get B-field at position (x, y, z) using trilinear interpolation
     */
    Vec3 at(float x, float y, float z) const;
};

/**
 * Constant B-Field (for testing)
 *
 * Input values are in Tesla, internally converted to native units GeV/(e*mm).
 */
struct ConstBField {
    Vec3 field;  // Stored in native units GeV/(e*mm)

    HLS_INLINE ConstBField() : field(0.0f, 0.0f, 2.0f * UNIT_T) {}  // 2T in z-direction
    HLS_INLINE ConstBField(float bx_tesla, float by_tesla, float bz_tesla)
        : field(bx_tesla * UNIT_T, by_tesla * UNIT_T, bz_tesla * UNIT_T) {}

    HLS_INLINE Vec3 at(float /*x*/, float /*y*/, float /*z*/) const {
        return field;
    }
};

//==============================================================================
// RK4 Propagator Core Functions
//==============================================================================

/**
 * Evaluate dtds = d(tangent)/ds = qop * (t × B)
 *
 * This is the Lorentz force equation for a charged particle in a magnetic field.
 *
 * @param t     Current tangent direction (unit vector)
 * @param b     Magnetic field at current position
 * @param qop   Charge over momentum (q/p)
 * @return      Rate of change of tangent direction
 */
HLS_INLINE Vec3 evaluate_dtds(const Vec3& t, const Vec3& b, float qop) {
    return qop * cross(t, b);
}

/**
 * Single RK4 step with adaptive step size
 *
 * Advances the track state by one RK4 step, adjusting step size based on
 * local truncation error estimate.
 *
 * @param state         Current track state (modified in place)
 * @param bfield        B-field accessor (grid or constant)
 * @param dist_to_next  Distance to next surface (mm)
 * @param step_size     Suggested step size (mm), modified to next suggested size
 * @param actual_step   Output: actual step size taken (mm)
 * @return              True if step successful
 */
template<typename BField>
bool rk4_step(FreeTrackState& state, const BField& bfield,
              float dist_to_next, float& step_size, float& actual_step);

/**
 * Propagate track to next surface
 *
 * Main entry point for RK4 propagation. Propagates until:
 * - Distance to surface < threshold
 * - Maximum steps reached
 * - Track exits valid region
 *
 * @param initial       Initial track parameters
 * @param bfield        B-field accessor
 * @param dist_to_next  Distance to next surface (mm)
 * @return              Propagation result with final state
 */
template<typename BField>
PropagationResult propagate_to_surface(const FreeTrackState& initial,
                                        const BField& bfield,
                                        float dist_to_next);

//==============================================================================
// HLS Top-Level Kernel Interface (declared outside namespace for extern "C")
//==============================================================================

}  // namespace traccc_fpga

// Use types from namespace
using traccc_fpga::TrackParams;

/**
 * HLS Top-Level Kernel: Batch RK4 Propagation
 *
 * Processes multiple tracks in parallel using DSP58 pipelines.
 *
 * @param n_tracks      Number of tracks to process
 * @param in_params     Input track parameters (AXI stream or memory-mapped)
 * @param out_params    Output propagated parameters
 * @param bfield_data   B-field grid data in HBM2e
 * @param distances     Distance to next surface for each track
 */
extern "C" void rk4_propagate_kernel(
    int n_tracks,
    const TrackParams* in_params,
    TrackParams* out_params,
    const float* bfield_data,
    const float* distances
);

/**
 * HLS Kernel: Constant B-Field (for testing/validation)
 */
extern "C" void rk4_propagate_const_bfield(
    int n_tracks,
    const TrackParams* in_params,
    TrackParams* out_params,
    float bfield_x,
    float bfield_y,
    float bfield_z,
    const float* distances
);

namespace traccc_fpga {

//==============================================================================
// Implementation (included for header-only use in simulation)
//==============================================================================

template<typename BField>
bool rk4_step(FreeTrackState& state, const BField& bfield,
              float dist_to_next, float& step_size, float& actual_step) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1

    // Clamp step size to reasonable bounds
    float h = step_size;
    if (std::fabs(h) > std::fabs(dist_to_next)) {
        h = dist_to_next;
    }
    if (std::fabs(h) < MIN_STEP_SIZE) {
        h = (h >= 0) ? MIN_STEP_SIZE : -MIN_STEP_SIZE;
    }

    const float half_h = h * 0.5f;
    const float h_6 = h / 6.0f;
    const float h2 = h * h;

    RK4State rk;

    // Stage 1: Initial point
    rk.b_first = bfield.at(state.pos.x, state.pos.y, state.pos.z);
    rk.t[0] = state.dir;
    rk.qop[0] = state.qop;
    rk.dtds[0] = evaluate_dtds(rk.t[0], rk.b_first, rk.qop[0]);

    // Stage 2: Midpoint (first estimate)
    Vec3 pos1 = state.pos + half_h * rk.t[0] + (h2 * 0.125f) * rk.dtds[0];
    rk.b_middle = bfield.at(pos1.x, pos1.y, pos1.z);
    rk.t[1] = state.dir + half_h * rk.dtds[0];
    rk.qop[1] = state.qop;  // No material effects for now
    rk.dtds[1] = evaluate_dtds(rk.t[1], rk.b_middle, rk.qop[1]);

    // Stage 3: Midpoint (second estimate)
    rk.t[2] = state.dir + half_h * rk.dtds[1];
    rk.qop[2] = state.qop;
    rk.dtds[2] = evaluate_dtds(rk.t[2], rk.b_middle, rk.qop[2]);

    // Stage 4: Endpoint
    Vec3 pos2 = state.pos + h * rk.t[0] + (h2 * 0.5f) * rk.dtds[2];
    rk.b_last = bfield.at(pos2.x, pos2.y, pos2.z);
    rk.t[3] = state.dir + h * rk.dtds[2];
    rk.qop[3] = state.qop;
    rk.dtds[3] = evaluate_dtds(rk.t[3], rk.b_last, rk.qop[3]);

    // Error estimate (Eq. 82 from reference)
    Vec3 err_vec = (h2 / 6.0f) * (rk.dtds[0] - rk.dtds[1] - rk.dtds[2] + rk.dtds[3]);
    float error = err_vec.norm();

    // Adaptive step size scaling
    float scale = 1.0f;
    if (error > 1e-20f) {
        scale = std::sqrt(std::sqrt(RK_ERROR_TOL / error));
        scale = std::fmax(0.25f, std::fmin(scale, 4.0f));
    }

    // If error too large, reject step and reduce step size
    if (error > 4.0f * RK_ERROR_TOL) {
        step_size = h * scale;
        actual_step = 0.0f;  // No step taken
        return false;
    }

    // Accept step: Update track state
    // Position update (Eq. 82)
    state.pos = state.pos + h * (rk.t[0] + h_6 * (rk.dtds[0] + rk.dtds[1] + rk.dtds[2]));

    // Direction update (Eq. 82)
    Vec3 new_dir = state.dir + h_6 * (rk.dtds[0] + 2.0f * (rk.dtds[1] + rk.dtds[2]) + rk.dtds[3]);
    state.dir = new_dir.normalized();

    // Update path length
    state.path_length += std::fabs(h);
    actual_step = h;  // Return actual step taken

    // Suggest next step size
    step_size = h * scale;
    if (std::fabs(step_size) < MIN_STEP_SIZE) {
        step_size = (step_size >= 0) ? MIN_STEP_SIZE : -MIN_STEP_SIZE;
    }

    return true;
}

template<typename BField>
PropagationResult propagate_to_surface(const FreeTrackState& initial,
                                        const BField& bfield,
                                        float dist_to_next) {
#pragma HLS INLINE off

    PropagationResult result;
    result.state = initial;
    result.chi2 = 0.0f;
    result.status = 0;
    result.n_steps = 0;

    float remaining = dist_to_next;
    float step_size = dist_to_next;  // Start with full distance

    // Main propagation loop
    PROPAGATE_LOOP:
    for (int step = 0; step < MAX_STEPS_PER_SURFACE; ++step) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=34 avg=6

        // Check if we've arrived
        if (std::fabs(remaining) < MIN_STEP_SIZE) {
            result.n_steps = (step < 255) ? step : 255;
            return result;
        }

        // Try RK4 step with adaptive step size
        bool step_ok = false;
        float actual_step = 0.0f;

        RK4_TRIALS:
        for (int t = 0; t < MAX_RK_TRIALS; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10 avg=2

            step_ok = rk4_step(result.state, bfield, remaining, step_size, actual_step);
            if (step_ok) break;
        }

        if (!step_ok) {
            result.status = 1;  // Failed
            result.n_steps = (step < 255) ? step : 255;
            return result;
        }

        remaining -= actual_step;  // Use actual step taken, not suggested next step
    }

    result.status = 2;  // Max steps reached
    result.n_steps = 255;  // Cap to uint8_t max
    return result;
}

}  // namespace traccc_fpga

#endif  // RK4_PROPAGATOR_HPP
