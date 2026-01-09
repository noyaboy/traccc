/**
 * TRACCC FPGA RK4 Propagator - C-Simulation Testbench
 *
 * This testbench validates the HLS RK4 propagator against known results.
 * Run with: g++ -std=c++17 -O2 -o rk4_tb rk4_propagator_tb.cpp rk4_propagator.cpp -lm
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 * Mozilla Public License Version 2.0
 */

#include "rk4_propagator.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

using namespace traccc_fpga;

//==============================================================================
// Test Utilities
//==============================================================================

void print_track(const char* label, const TrackParams& p) {
    std::cout << label << ": "
              << "loc=(" << std::setw(8) << p.loc0 << ", " << std::setw(8) << p.loc1 << ") "
              << "phi=" << std::setw(8) << p.phi << " "
              << "theta=" << std::setw(8) << p.theta << " "
              << "qop=" << std::setw(10) << p.qop << std::endl;
}

void print_free_state(const char* label, const FreeTrackState& s) {
    std::cout << label << ": "
              << "pos=(" << s.pos.x << ", " << s.pos.y << ", " << s.pos.z << ") "
              << "dir=(" << s.dir.x << ", " << s.dir.y << ", " << s.dir.z << ") "
              << "qop=" << s.qop << " path=" << s.path_length << std::endl;
}

bool check_close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

//==============================================================================
// Test Cases
//==============================================================================

/**
 * Test 1: Straight line propagation (zero B-field)
 */
bool test_straight_line() {
    std::cout << "\n=== Test 1: Straight Line Propagation (B=0) ===" << std::endl;

    ConstBField zero_field(0.0f, 0.0f, 0.0f);

    // Track going in +z direction
    FreeTrackState initial;
    initial.pos = Vec3(0.0f, 0.0f, 0.0f);
    initial.dir = Vec3(0.0f, 0.0f, 1.0f);  // +z direction
    initial.qop = 1.0f;  // 1 GeV
    initial.time = 0.0f;
    initial.path_length = 0.0f;

    float dist = 100.0f;  // 100 mm

    print_free_state("Initial", initial);

    PropagationResult result = propagate_to_surface(initial, zero_field, dist);

    print_free_state("Final", result.state);
    std::cout << "Status: " << (int)result.status << " Steps: " << (int)result.n_steps << std::endl;

    // Expected: pos should be (0, 0, 100), dir unchanged
    bool pass = true;
    pass &= check_close(result.state.pos.x, 0.0f);
    pass &= check_close(result.state.pos.y, 0.0f);
    pass &= check_close(result.state.pos.z, 100.0f, 1.0f);
    pass &= check_close(result.state.dir.z, 1.0f);
    pass &= (result.status == 0);

    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

/**
 * Test 2: Circular motion in constant B-field
 *
 * A charged particle in a uniform B-field perpendicular to its velocity
 * should move in a circle with radius r = p / (|q|B).
 */
bool test_circular_motion() {
    std::cout << "\n=== Test 2: Circular Motion (B = 2T in z) ===" << std::endl;

    // 2 Tesla field in z-direction
    ConstBField bfield(0.0f, 0.0f, 2.0f);

    // 1 GeV particle moving in +x direction
    // Radius = p / (qB) = 1 GeV / (1 * 2T) = 0.5 GeV/T
    // In natural units: r = p / (0.3 * B) meters = 1 / (0.3 * 2) = 1.67 m = 1670 mm
    FreeTrackState initial;
    initial.pos = Vec3(0.0f, 0.0f, 0.0f);
    initial.dir = Vec3(1.0f, 0.0f, 0.0f);  // +x direction
    initial.qop = 1.0f;  // +1 GeV (positive charge)
    initial.time = 0.0f;
    initial.path_length = 0.0f;

    // Propagate 1/4 of the circumference
    // Circumference = 2 * pi * r ≈ 10,494 mm
    // 1/4 arc = ~2,623 mm
    float dist = 500.0f;  // Start with smaller distance

    print_free_state("Initial", initial);

    PropagationResult result = propagate_to_surface(initial, bfield, dist);

    print_free_state("Final", result.state);
    std::cout << "Status: " << (int)result.status << " Steps: " << (int)result.n_steps << std::endl;

    // For circular motion, the particle should curve
    // With positive charge and B in +z, Lorentz force F = qv×B points in -y initially
    // So particle curves toward -y
    bool pass = true;
    pass &= (result.status == 0);
    pass &= (result.state.pos.y < 0.0f);  // Should have curved in -y direction
    pass &= (result.n_steps > 0);

    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

/**
 * Test 3: Direction preservation (normalize check)
 */
bool test_direction_normalized() {
    std::cout << "\n=== Test 3: Direction Normalization ===" << std::endl;

    ConstBField bfield(0.0f, 0.0f, 2.0f);

    FreeTrackState initial;
    initial.pos = Vec3(0.0f, 0.0f, 0.0f);
    initial.dir = Vec3(0.6f, 0.8f, 0.0f);  // In xy-plane, normalized
    initial.qop = 0.5f;  // 2 GeV
    initial.time = 0.0f;
    initial.path_length = 0.0f;

    float dist = 200.0f;

    PropagationResult result = propagate_to_surface(initial, bfield, dist);

    float dir_norm = result.state.dir.norm();

    std::cout << "Final direction norm: " << dir_norm << std::endl;

    bool pass = check_close(dir_norm, 1.0f, 1e-5f);
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

/**
 * Test 4: Batch processing performance
 */
bool test_batch_performance() {
    std::cout << "\n=== Test 4: Batch Processing Performance ===" << std::endl;

    const int N_TRACKS = 6666;  // Average tracks per CKF step

    // Allocate buffers
    std::vector<TrackParams> in_params(N_TRACKS);
    std::vector<TrackParams> out_params(N_TRACKS);
    std::vector<float> distances(N_TRACKS);

    // Initialize with random tracks
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> loc_dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> phi_dist(-M_PI, M_PI);
    std::uniform_real_distribution<float> theta_dist(0.1f, M_PI - 0.1f);
    std::uniform_real_distribution<float> qop_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> dist_dist(10.0f, 500.0f);

    for (int i = 0; i < N_TRACKS; ++i) {
        in_params[i].loc0 = loc_dist(rng);
        in_params[i].loc1 = loc_dist(rng);
        in_params[i].phi = phi_dist(rng);
        in_params[i].theta = theta_dist(rng);
        in_params[i].qop = qop_dist(rng);
        in_params[i].time = 0.0f;
        distances[i] = dist_dist(rng);
    }

    std::cout << "Processing " << N_TRACKS << " tracks..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Call the kernel (C-simulation) - function is in global namespace
    ::rk4_propagate_const_bfield(
        N_TRACKS,
        in_params.data(),
        out_params.data(),
        0.0f, 0.0f, 2.0f,  // 2T in z
        distances.data()
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time: " << duration.count() << " us" << std::endl;
    std::cout << "Per track: " << (float)duration.count() / N_TRACKS << " us" << std::endl;

    // Verify outputs are valid
    int invalid_count = 0;
    for (int i = 0; i < N_TRACKS; ++i) {
        if (!std::isfinite(out_params[i].loc0) ||
            !std::isfinite(out_params[i].loc1) ||
            !std::isfinite(out_params[i].phi) ||
            !std::isfinite(out_params[i].theta) ||
            !std::isfinite(out_params[i].qop)) {
            invalid_count++;
        }
    }

    std::cout << "Invalid outputs: " << invalid_count << " / " << N_TRACKS << std::endl;

    bool pass = (invalid_count == 0);
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

/**
 * Test 5: Step count distribution
 */
bool test_step_count() {
    std::cout << "\n=== Test 5: Step Count Distribution ===" << std::endl;

    ConstBField bfield(0.0f, 0.0f, 2.0f);

    // Test various distances
    float test_distances[] = {10.0f, 50.0f, 100.0f, 200.0f, 500.0f};
    int n_tests = sizeof(test_distances) / sizeof(test_distances[0]);

    std::cout << std::setw(12) << "Distance" << std::setw(12) << "Steps" << std::endl;
    std::cout << std::string(24, '-') << std::endl;

    bool pass = true;
    for (int i = 0; i < n_tests; ++i) {
        FreeTrackState initial;
        initial.pos = Vec3(0.0f, 0.0f, 0.0f);
        initial.dir = Vec3(0.6f, 0.8f, 0.0f);
        initial.qop = 1.0f;
        initial.time = 0.0f;
        initial.path_length = 0.0f;

        PropagationResult result = propagate_to_surface(initial, bfield, test_distances[i]);

        std::cout << std::setw(12) << test_distances[i]
                  << std::setw(12) << (int)result.n_steps << std::endl;

        pass &= (result.status == 0);
        pass &= (result.n_steps > 0);
        pass &= (result.n_steps <= MAX_STEPS_PER_SURFACE);
    }

    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

/**
 * Test 6: Comparison with analytical helix
 */
bool test_helix_accuracy() {
    std::cout << "\n=== Test 6: Helix Accuracy (vs Analytical) ===" << std::endl;

    // 2 Tesla field in z
    // B = 2 T, p = 1 GeV, q = 1
    // Radius r = p / (0.3 * q * B) = 1 / 0.6 ≈ 1.667 m = 1667 mm
    const float B = 2.0f;  // Tesla
    const float p = 1.0f;  // GeV
    const float radius = p / (0.3f * B) * 1000.0f;  // mm

    std::cout << "Expected radius: " << radius << " mm" << std::endl;

    ConstBField bfield(0.0f, 0.0f, B);

    FreeTrackState initial;
    initial.pos = Vec3(0.0f, 0.0f, 0.0f);
    initial.dir = Vec3(1.0f, 0.0f, 0.0f);  // +x direction
    initial.qop = 1.0f / p;  // positive charge
    initial.time = 0.0f;
    initial.path_length = 0.0f;

    // Propagate 1/4 circle = pi/2 * radius
    float arc_length = M_PI / 2.0f * radius;
    std::cout << "Arc length (1/4 circle): " << arc_length << " mm" << std::endl;

    PropagationResult result = propagate_to_surface(initial, bfield, arc_length);

    // After 1/4 circle:
    // - Start: (0, 0, 0), dir = (+1, 0, 0)
    // - Expected: (r, -r, 0), dir = (0, -1, 0)
    // (Center of circle is at (0, -r, 0), particle goes CW when viewed from +z)

    float expected_x = radius;
    float expected_y = -radius;

    std::cout << "Expected position: (" << expected_x << ", " << expected_y << ", 0)" << std::endl;
    std::cout << "Actual position:   (" << result.state.pos.x << ", " << result.state.pos.y
              << ", " << result.state.pos.z << ")" << std::endl;

    float error_x = std::fabs(result.state.pos.x - expected_x);
    float error_y = std::fabs(result.state.pos.y - expected_y);
    float error = std::sqrt(error_x * error_x + error_y * error_y);

    std::cout << "Position error: " << error << " mm (" << (error / radius * 100) << "%)" << std::endl;

    // Allow 5% error for RK4 with adaptive stepping
    bool pass = (error / radius < 0.05f);
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "TRACCC FPGA RK4 Propagator Testbench" << std::endl;
    std::cout << "======================================" << std::endl;

    int pass_count = 0;
    int test_count = 0;

    test_count++; if (test_straight_line()) pass_count++;
    test_count++; if (test_circular_motion()) pass_count++;
    test_count++; if (test_direction_normalized()) pass_count++;
    test_count++; if (test_batch_performance()) pass_count++;
    test_count++; if (test_step_count()) pass_count++;
    test_count++; if (test_helix_accuracy()) pass_count++;

    std::cout << "\n======================================" << std::endl;
    std::cout << "Summary: " << pass_count << "/" << test_count << " tests passed" << std::endl;
    std::cout << "======================================" << std::endl;

    return (pass_count == test_count) ? 0 : 1;
}
