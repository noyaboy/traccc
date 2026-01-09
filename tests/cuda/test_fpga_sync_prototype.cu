/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

/**
 * Phase 2: CPU-as-FPGA Synchronization Prototype
 *
 * This test simulates the GPU↔FPGA communication pattern using CPU as a
 * stand-in for the FPGA. It measures the full round-trip overhead including:
 *   - GPU → Host (D2H) transfer
 *   - "FPGA" computation (simulated with CPU sleep)
 *   - Host → GPU (H2D) transfer
 *   - GPU deduplication kernel
 *   - Full synchronization barrier
 *
 * Build (standalone):
 *   nvcc -O3 -o test_fpga_sync_prototype test_fpga_sync_prototype.cu -lpthread
 *
 * Run:
 *   ./test_fpga_sync_prototype
 *
 * Related: doc/survey-fpga.md Section 9.4.2.7 Phase 2
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace std::chrono;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Simulate GPU deduplication kernel
__global__ void dedup_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

struct TestConfig {
    const char* name;
    int num_tracks;       // Average tracks per step
    int param_bytes;      // Bytes per track (24 for params only, 176 for full)
    int num_steps;        // CKF steps (typically 15)
    int fpga_compute_us;  // Simulated FPGA compute time (µs)
    int dedup_iters;      // GPU dedup kernel iterations
    bool use_pinned;      // Use pinned (page-locked) memory
    bool use_async;       // Use async memcpy
};

struct StepTiming {
    long d2h_us;
    long fpga_compute_us;
    long h2d_us;
    long dedup_us;
    long sync_us;
    long total_us;
};

void print_timing_table(const std::vector<StepTiming>& timings) {
    printf("\n=== Per-Step Breakdown ===\n");
    printf("Step |  D2H(µs) | FPGA(µs) |  H2D(µs) | Dedup(µs) | Sync(µs) | Total(µs)\n");
    printf("-----|----------|----------|----------|-----------|----------|----------\n");

    for (size_t s = 0; s < timings.size(); s++) {
        const auto& t = timings[s];
        printf("%4zu | %8ld | %8ld | %8ld | %9ld | %8ld | %8ld\n",
               s, t.d2h_us, t.fpga_compute_us, t.h2d_us, t.dedup_us, t.sync_us, t.total_us);
    }
}

void run_sync_test(const TestConfig& cfg, int num_iterations) {
    printf("\n");
    printf("========================================\n");
    printf("Test: %s\n", cfg.name);
    printf("========================================\n");

    const size_t transfer_size = static_cast<size_t>(cfg.num_tracks) * cfg.param_bytes;

    printf("\n=== Configuration ===\n");
    printf("Tracks: %d\n", cfg.num_tracks);
    printf("Param bytes: %d\n", cfg.param_bytes);
    printf("Transfer size: %.1f KB\n", transfer_size / 1024.0);
    printf("Steps: %d\n", cfg.num_steps);
    printf("FPGA compute: %d µs\n", cfg.fpga_compute_us);
    printf("Pinned memory: %s\n", cfg.use_pinned ? "yes" : "no");
    printf("Async transfers: %s\n", cfg.use_async ? "yes" : "no");
    printf("Iterations: %d\n", num_iterations);

    // Allocate GPU memory
    void *d_params_in, *d_params_out;
    float* d_dedup_data;
    CUDA_CHECK(cudaMalloc(&d_params_in, transfer_size));
    CUDA_CHECK(cudaMalloc(&d_params_out, transfer_size));
    CUDA_CHECK(cudaMalloc(&d_dedup_data, cfg.num_tracks * sizeof(float)));

    // Allocate host memory (pinned or pageable)
    void* h_params;
    if (cfg.use_pinned) {
        CUDA_CHECK(cudaMallocHost(&h_params, transfer_size));
    } else {
        h_params = malloc(transfer_size);
    }
    memset(h_params, 0, transfer_size);

    // Create streams
    cudaStream_t stream_transfer, stream_compute;
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));

    // Warmup
    printf("\nWarming up...\n");
    for (int i = 0; i < 10; i++) {
        if (cfg.use_async) {
            CUDA_CHECK(cudaMemcpyAsync(h_params, d_params_in, transfer_size,
                                       cudaMemcpyDeviceToHost, stream_transfer));
            CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
            CUDA_CHECK(cudaMemcpyAsync(d_params_out, h_params, transfer_size,
                                       cudaMemcpyHostToDevice, stream_transfer));
            CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
        } else {
            CUDA_CHECK(cudaMemcpy(h_params, d_params_in, transfer_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_params_out, h_params, transfer_size, cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Storage for all iterations
    std::vector<long> event_times;
    std::vector<long> total_overhead_per_event;
    std::vector<std::vector<StepTiming>> all_step_timings(num_iterations);

    printf("\nRunning %d iterations...\n", num_iterations);

    for (int iter = 0; iter < num_iterations; iter++) {
        std::vector<StepTiming>& timings = all_step_timings[iter];
        timings.resize(cfg.num_steps);

        auto event_start = high_resolution_clock::now();

        for (int step = 0; step < cfg.num_steps; step++) {
            auto step_start = high_resolution_clock::now();

            // 1. GPU → "FPGA" (D2H transfer)
            auto d2h_start = high_resolution_clock::now();
            if (cfg.use_async) {
                CUDA_CHECK(cudaMemcpyAsync(h_params, d_params_in, transfer_size,
                                           cudaMemcpyDeviceToHost, stream_transfer));
                CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
            } else {
                CUDA_CHECK(cudaMemcpy(h_params, d_params_in, transfer_size,
                                      cudaMemcpyDeviceToHost));
            }
            auto d2h_end = high_resolution_clock::now();

            // 2. "FPGA" computation (simulate with CPU sleep)
            auto fpga_start = high_resolution_clock::now();
            if (cfg.fpga_compute_us > 0) {
                std::this_thread::sleep_for(microseconds(cfg.fpga_compute_us));
            }
            auto fpga_end = high_resolution_clock::now();

            // 3. "FPGA" → GPU (H2D transfer)
            auto h2d_start = high_resolution_clock::now();
            if (cfg.use_async) {
                CUDA_CHECK(cudaMemcpyAsync(d_params_out, h_params, transfer_size,
                                           cudaMemcpyHostToDevice, stream_transfer));
                CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
            } else {
                CUDA_CHECK(cudaMemcpy(d_params_out, h_params, transfer_size,
                                      cudaMemcpyHostToDevice));
            }
            auto h2d_end = high_resolution_clock::now();

            // 4. GPU deduplication kernel
            auto dedup_start = high_resolution_clock::now();
            int blocks = (cfg.num_tracks + 255) / 256;
            dedup_kernel<<<blocks, 256, 0, stream_compute>>>(
                d_dedup_data, cfg.num_tracks, cfg.dedup_iters);
            auto dedup_end = high_resolution_clock::now();

            // 5. Synchronize (MANDATORY BARRIER)
            auto sync_start = high_resolution_clock::now();
            CUDA_CHECK(cudaDeviceSynchronize());
            auto sync_end = high_resolution_clock::now();

            auto step_end = high_resolution_clock::now();

            // Record timings
            timings[step] = {
                .d2h_us = duration_cast<microseconds>(d2h_end - d2h_start).count(),
                .fpga_compute_us = duration_cast<microseconds>(fpga_end - fpga_start).count(),
                .h2d_us = duration_cast<microseconds>(h2d_end - h2d_start).count(),
                .dedup_us = duration_cast<microseconds>(dedup_end - dedup_start).count(),
                .sync_us = duration_cast<microseconds>(sync_end - sync_start).count(),
                .total_us = duration_cast<microseconds>(step_end - step_start).count()
            };

            // Swap buffers for next step
            std::swap(d_params_in, d_params_out);
        }

        auto event_end = high_resolution_clock::now();
        auto event_us = duration_cast<microseconds>(event_end - event_start).count();
        event_times.push_back(event_us);

        // Calculate overhead (D2H + H2D + Sync, excluding FPGA compute and dedup)
        long overhead = 0;
        for (const auto& t : timings) {
            overhead += t.d2h_us + t.h2d_us + t.sync_us;
        }
        total_overhead_per_event.push_back(overhead);
    }

    // Print detailed timing for first iteration
    print_timing_table(all_step_timings[0]);

    // Calculate statistics across all iterations
    printf("\n=== Aggregate Statistics (across %d iterations) ===\n", num_iterations);

    // Per-step averages
    long sum_d2h = 0, sum_fpga = 0, sum_h2d = 0, sum_dedup = 0, sum_sync = 0, sum_total = 0;
    for (const auto& iter_timings : all_step_timings) {
        for (const auto& t : iter_timings) {
            sum_d2h += t.d2h_us;
            sum_fpga += t.fpga_compute_us;
            sum_h2d += t.h2d_us;
            sum_dedup += t.dedup_us;
            sum_sync += t.sync_us;
            sum_total += t.total_us;
        }
    }
    int total_steps = num_iterations * cfg.num_steps;

    printf("\nAverage per step:\n");
    printf("  D2H transfer: %.1f µs\n", sum_d2h / (double)total_steps);
    printf("  FPGA compute: %.1f µs [simulated]\n", sum_fpga / (double)total_steps);
    printf("  H2D transfer: %.1f µs\n", sum_h2d / (double)total_steps);
    printf("  Dedup kernel: %.1f µs\n", sum_dedup / (double)total_steps);
    printf("  Sync barrier: %.1f µs\n", sum_sync / (double)total_steps);
    printf("  Total:        %.1f µs\n", sum_total / (double)total_steps);

    // Event-level statistics
    std::sort(event_times.begin(), event_times.end());
    long event_mean = std::accumulate(event_times.begin(), event_times.end(), 0L) / num_iterations;
    long event_median = event_times[num_iterations / 2];
    long event_min = event_times.front();
    long event_max = event_times.back();

    printf("\nPer event (%d steps):\n", cfg.num_steps);
    printf("  Mean:   %ld µs (%.1f%% of 23ms)\n", event_mean, event_mean / 230.0);
    printf("  Median: %ld µs (%.1f%% of 23ms)\n", event_median, event_median / 230.0);
    printf("  Min:    %ld µs\n", event_min);
    printf("  Max:    %ld µs\n", event_max);

    // Communication overhead analysis
    std::sort(total_overhead_per_event.begin(), total_overhead_per_event.end());
    long overhead_mean = std::accumulate(total_overhead_per_event.begin(),
                                          total_overhead_per_event.end(), 0L) / num_iterations;
    double overhead_pct = (overhead_mean / 23000.0) * 100.0;
    double per_step_overhead = overhead_mean / (double)cfg.num_steps;

    printf("\n=== COMMUNICATION OVERHEAD ANALYSIS ===\n");
    printf("Communication overhead (D2H + H2D + Sync):\n");
    printf("  Per event: %ld µs (%.1f%% of 23ms budget)\n", overhead_mean, overhead_pct);
    printf("  Per step:  %.1f µs\n", per_step_overhead);

    // Decision
    printf("\n=== DECISION ===\n");
    if (per_step_overhead < 100) {
        printf("✓ Per-step overhead < 100µs: FPGA LIKELY VIABLE\n");
        printf("  Proceed to Phase 4 (XRT benchmark on actual V80 hardware)\n");
    } else if (per_step_overhead < 200) {
        printf("⚠ Per-step overhead 100-200µs: MARGINAL\n");
        printf("  Proceed to Phase 3 (test async overlap mitigation)\n");
    } else if (per_step_overhead < 400) {
        printf("⚠ Per-step overhead 200-400µs: CONCERNING\n");
        printf("  Async overlap mitigation is REQUIRED for viability\n");
    } else {
        printf("✗ Per-step overhead > 400µs: FPGA PATH LIKELY NOT VIABLE\n");
        printf("  Consider alternative architectures or algorithmic changes\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_params_in));
    CUDA_CHECK(cudaFree(d_params_out));
    CUDA_CHECK(cudaFree(d_dedup_data));
    if (cfg.use_pinned) {
        CUDA_CHECK(cudaFreeHost(h_params));
    } else {
        free(h_params);
    }
    CUDA_CHECK(cudaStreamDestroy(stream_transfer));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
}

int main(int argc, char** argv) {
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== GPU Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("PCIe: Domain %04x Bus %02x Device %02x\n",
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    // Test configurations
    std::vector<TestConfig> configs = {
        // Baseline: params only, pinned memory, no FPGA compute
        {"Params only, pinned, no FPGA compute",
         6666, 24, 15, 0, 200, true, false},

        // With simulated FPGA compute times
        {"Params only, pinned, FPGA=100µs",
         6666, 24, 15, 100, 200, true, false},

        {"Params only, pinned, FPGA=200µs",
         6666, 24, 15, 200, 200, true, false},

        {"Params only, pinned, FPGA=500µs",
         6666, 24, 15, 500, 200, true, false},

        // Async transfers
        {"Params only, pinned, FPGA=200µs, async",
         6666, 24, 15, 200, 200, true, true},

        // Pageable memory (worst case)
        {"Params only, pageable, FPGA=200µs",
         6666, 24, 15, 200, 200, false, false},

        // Full track params (176 bytes)
        {"Full params (176B), pinned, FPGA=200µs",
         6666, 176, 15, 200, 200, true, false},

        // Min track count (best case)
        {"Params only, min tracks (128), FPGA=200µs",
         128, 24, 15, 200, 200, true, false},

        // Max track count (worst case)
        {"Params only, max tracks (42240), FPGA=200µs",
         42240, 24, 15, 200, 200, true, false},
    };

    const int num_iterations = 50;

    for (const auto& cfg : configs) {
        run_sync_test(cfg, num_iterations);
    }

    printf("\n");
    printf("########################################################\n");
    printf("# SUMMARY\n");
    printf("########################################################\n");
    printf("\n");
    printf("This test simulates GPU↔FPGA communication using CPU as FPGA.\n");
    printf("\n");
    printf("Key findings to look for:\n");
    printf("1. D2H + H2D transfer time for 156KB (params only): ~20-50µs\n");
    printf("2. Sync barrier overhead: ~10-30µs\n");
    printf("3. Total per-step communication overhead: ~50-100µs (optimistic)\n");
    printf("\n");
    printf("If per-step overhead > 200µs, proceed to Phase 3 to test\n");
    printf("whether async overlap can hide the latency.\n");
    printf("\n");
    printf("Note: Real FPGA will have additional kernel launch overhead\n");
    printf("(XRT dispatch) not captured in this simulation.\n");

    return 0;
}
