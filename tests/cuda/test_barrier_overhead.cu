/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

/**
 * Phase 1: Measure GPU Synchronization Barrier Overhead
 *
 * This test measures the overhead of cudaStreamSynchronize() and
 * cudaDeviceSynchronize() calls to establish a baseline for comparing
 * with potential FPGA synchronization overhead.
 *
 * Build (standalone):
 *   nvcc -O3 -o test_barrier_overhead test_barrier_overhead.cu
 *
 * Run:
 *   ./test_barrier_overhead
 *
 * Related: doc/survey-fpga.md Section 9.4.2.7 Phase 1
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std::chrono;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Dummy kernel to simulate propagate_to_next_surface workload
// Duration controllable via iterations parameter
__global__ void dummy_propagate_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
            val = sqrtf(val * val + 1.0f);
        }
        data[idx] = val;
    }
}

// Dummy kernel to simulate find_tracks/deduplication workload
__global__ void dummy_dedup_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 0.9999f + 0.0001f;
        }
        data[idx] = val;
    }
}

struct StepTiming {
    float propagate_ms;
    float barrier_ms;
    float dedup_ms;
    float total_ms;
};

void print_statistics(const char* name, const std::vector<float>& values) {
    if (values.empty()) return;

    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    float sum = std::accumulate(sorted.begin(), sorted.end(), 0.0f);
    float mean = sum / sorted.size();
    float median = sorted[sorted.size() / 2];
    float min_val = sorted.front();
    float max_val = sorted.back();
    float p99 = sorted[static_cast<size_t>(sorted.size() * 0.99)];

    printf("  %s: mean=%.1f, median=%.1f, min=%.1f, max=%.1f, p99=%.1f µs\n",
           name, mean, median, min_val, max_val, p99);
}

void run_barrier_test(int num_tracks, int num_steps, int propagate_iters,
                      int dedup_iters, int num_events) {
    printf("\n=== Configuration ===\n");
    printf("Tracks per step: %d\n", num_tracks);
    printf("Steps per event: %d\n", num_steps);
    printf("Events to process: %d\n", num_events);
    printf("Propagate kernel iterations: %d\n", propagate_iters);
    printf("Dedup kernel iterations: %d\n", dedup_iters);

    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_tracks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, num_tracks * sizeof(float)));

    // Create stream and events
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Events for timing
    cudaEvent_t step_start, propagate_end, barrier_end, step_end;
    CUDA_CHECK(cudaEventCreate(&step_start));
    CUDA_CHECK(cudaEventCreate(&propagate_end));
    CUDA_CHECK(cudaEventCreate(&barrier_end));
    CUDA_CHECK(cudaEventCreate(&step_end));

    // Warmup
    printf("\nWarming up...\n");
    for (int i = 0; i < 10; i++) {
        int blocks = (num_tracks + 255) / 256;
        dummy_propagate_kernel<<<blocks, 256, 0, stream>>>(d_data, num_tracks, propagate_iters);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        dummy_dedup_kernel<<<blocks, 256, 0, stream>>>(d_data, num_tracks, dedup_iters);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Storage for all timings
    std::vector<float> all_propagate_us, all_barrier_us, all_dedup_us, all_step_us;
    std::vector<float> event_times_us;

    printf("\nRunning %d events...\n", num_events);

    for (int event = 0; event < num_events; event++) {
        auto event_start = high_resolution_clock::now();

        for (int step = 0; step < num_steps; step++) {
            // Record step start
            CUDA_CHECK(cudaEventRecord(step_start, stream));

            // Kernel 1: Propagate (simulates propagate_to_next_surface)
            int blocks = (num_tracks + 255) / 256;
            dummy_propagate_kernel<<<blocks, 256, 0, stream>>>(
                d_data, num_tracks, propagate_iters);

            CUDA_CHECK(cudaEventRecord(propagate_end, stream));

            // BARRIER: This is what we're measuring
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(barrier_end, stream));

            // Kernel 2: Dedup (simulates find_tracks/deduplication)
            dummy_dedup_kernel<<<blocks, 256, 0, stream>>>(
                d_data, num_tracks, dedup_iters);

            CUDA_CHECK(cudaEventRecord(step_end, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Calculate timings
            float propagate_ms, barrier_ms, dedup_ms, total_ms;
            CUDA_CHECK(cudaEventElapsedTime(&propagate_ms, step_start, propagate_end));
            CUDA_CHECK(cudaEventElapsedTime(&barrier_ms, propagate_end, barrier_end));
            CUDA_CHECK(cudaEventElapsedTime(&dedup_ms, barrier_end, step_end));
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, step_start, step_end));

            // Convert to microseconds and store
            all_propagate_us.push_back(propagate_ms * 1000.0f);
            all_barrier_us.push_back(barrier_ms * 1000.0f);
            all_dedup_us.push_back(dedup_ms * 1000.0f);
            all_step_us.push_back(total_ms * 1000.0f);
        }

        auto event_end = high_resolution_clock::now();
        auto event_us = duration_cast<microseconds>(event_end - event_start).count();
        event_times_us.push_back(static_cast<float>(event_us));
    }

    // Print detailed statistics
    printf("\n=== Per-Step Timing Statistics (µs) ===\n");
    print_statistics("Propagate kernel", all_propagate_us);
    print_statistics("BARRIER (sync)  ", all_barrier_us);
    print_statistics("Dedup kernel    ", all_dedup_us);
    print_statistics("Total step      ", all_step_us);

    // Calculate totals
    float total_propagate = std::accumulate(all_propagate_us.begin(), all_propagate_us.end(), 0.0f);
    float total_barrier = std::accumulate(all_barrier_us.begin(), all_barrier_us.end(), 0.0f);
    float total_dedup = std::accumulate(all_dedup_us.begin(), all_dedup_us.end(), 0.0f);
    float total_step = std::accumulate(all_step_us.begin(), all_step_us.end(), 0.0f);

    int total_steps = num_events * num_steps;

    printf("\n=== Aggregate Statistics ===\n");
    printf("Total steps: %d\n", total_steps);
    printf("Average per step:\n");
    printf("  Propagate: %.1f µs\n", total_propagate / total_steps);
    printf("  BARRIER:   %.1f µs\n", total_barrier / total_steps);
    printf("  Dedup:     %.1f µs\n", total_dedup / total_steps);
    printf("  Total:     %.1f µs\n", total_step / total_steps);

    printf("\nPer event (%d steps):\n", num_steps);
    printf("  Propagate: %.1f µs\n", total_propagate / num_events);
    printf("  BARRIER:   %.1f µs\n", total_barrier / num_events);
    printf("  Dedup:     %.1f µs\n", total_dedup / num_events);
    printf("  Total:     %.1f µs\n", total_step / num_events);

    float barrier_per_event = total_barrier / num_events;
    float barrier_pct_of_23ms = (barrier_per_event / 23000.0f) * 100.0f;

    printf("\n=== BARRIER OVERHEAD ANALYSIS ===\n");
    printf("Barrier time per event: %.1f µs\n", barrier_per_event);
    printf("Barrier %% of 23ms budget: %.2f%%\n", barrier_pct_of_23ms);
    printf("Barrier per step: %.1f µs\n", total_barrier / total_steps);

    printf("\n=== INTERPRETATION ===\n");
    float barrier_per_step = total_barrier / total_steps;
    if (barrier_per_step < 20) {
        printf("Current GPU barrier overhead is MINIMAL (< 20µs/step)\n");
        printf("FPGA sync overhead will be the dominant factor.\n");
    } else if (barrier_per_step < 100) {
        printf("Current GPU barrier overhead is LOW (< 100µs/step)\n");
        printf("FPGA must keep sync overhead comparable.\n");
    } else {
        printf("Current GPU barrier overhead is SIGNIFICANT (> 100µs/step)\n");
        printf("FPGA sync overhead has more headroom.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(step_start));
    CUDA_CHECK(cudaEventDestroy(propagate_end));
    CUDA_CHECK(cudaEventDestroy(barrier_end));
    CUDA_CHECK(cudaEventDestroy(step_end));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
}

int main(int argc, char** argv) {
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== GPU Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Clock Rate: %.0f MHz\n", prop.clockRate / 1000.0f);
    printf("PCIe: Domain %04x Bus %02x Device %02x\n",
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    // Test configurations matching CKF workload
    // Propagate iterations tuned to give ~900µs kernel time (like real propagate)
    // Dedup iterations tuned to give ~100µs kernel time

    printf("\n");
    printf("########################################################\n");
    printf("# Test 1: Realistic CKF workload (6666 tracks, 15 steps)\n");
    printf("########################################################\n");
    run_barrier_test(
        6666,   // num_tracks (average from Nsys)
        15,     // num_steps
        1000,   // propagate_iters (tune for ~900µs)
        200,    // dedup_iters (tune for ~100µs)
        100     // num_events
    );

    printf("\n");
    printf("########################################################\n");
    printf("# Test 2: Light workload (minimal kernel time)\n");
    printf("########################################################\n");
    run_barrier_test(
        6666,   // num_tracks
        15,     // num_steps
        10,     // propagate_iters (minimal)
        10,     // dedup_iters (minimal)
        100     // num_events
    );

    printf("\n");
    printf("########################################################\n");
    printf("# Test 3: Heavy workload (longer kernels)\n");
    printf("########################################################\n");
    run_barrier_test(
        6666,   // num_tracks
        15,     // num_steps
        5000,   // propagate_iters (heavy)
        1000,   // dedup_iters (heavy)
        50      // num_events
    );

    printf("\n=== SUMMARY ===\n");
    printf("This test measures the inherent cudaStreamSynchronize() overhead.\n");
    printf("The BARRIER time represents the minimum synchronization cost that\n");
    printf("any GPU↔FPGA communication will also incur, plus additional\n");
    printf("PCIe transfer and FPGA signaling overhead.\n");
    printf("\n");
    printf("Next step: Run test_fpga_sync_prototype to measure full\n");
    printf("simulated GPU↔FPGA round-trip overhead.\n");

    return 0;
}
