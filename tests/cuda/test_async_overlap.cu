/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

/**
 * Phase 3: Async Overlap Feasibility Test
 *
 * This test determines if GPU deduplication can overlap with FPGA
 * communication to hide synchronization latency. It compares:
 *   - Sequential: FPGA round-trip → Dedup → Sync → Next step
 *   - Overlapped: FPGA round-trip || Dedup(prev) → Sync both → Next step
 *
 * The overlap pattern uses double-buffering to allow the GPU to process
 * the previous step's data while waiting for the current step's FPGA
 * results.
 *
 * Build (standalone):
 *   nvcc -O3 -o test_async_overlap test_async_overlap.cu -lpthread
 *
 * Run:
 *   ./test_async_overlap
 *
 * Related: doc/survey-fpga.md Section 9.4.2.7 Phase 3
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

// Simulate deduplication kernel with configurable duration
// Higher iterations = longer kernel execution time
__global__ void dedup_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
            val = sqrtf(val * val + 0.001f);  // More expensive op
        }
        data[idx] = val;
    }
}

// Simulate chi² accumulation kernel
__global__ void chi2_kernel(float* chi2, float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(chi2, data[idx] * data[idx]);
    }
}

struct OverlapConfig {
    int num_tracks;
    int param_bytes;
    int num_steps;
    int fpga_compute_us;   // Simulated FPGA compute time
    int dedup_iterations;  // Control dedup kernel duration
};

struct TestResult {
    long total_us;
    double pct_of_budget;
    double per_step_us;
};

// Measure actual dedup kernel duration
float measure_dedup_duration(int num_tracks, int dedup_iterations) {
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_tracks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, num_tracks * sizeof(float)));

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    // Warmup
    int blocks = (num_tracks + 255) / 256;
    for (int i = 0; i < 5; i++) {
        dedup_kernel<<<blocks, 256>>>(d_data, num_tracks, dedup_iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    CUDA_CHECK(cudaEventRecord(start));
    dedup_kernel<<<blocks, 256>>>(d_data, num_tracks, dedup_iterations);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_data));

    return ms * 1000.0f;  // Return microseconds
}

TestResult test_sequential(const OverlapConfig& cfg, int num_iterations) {
    const size_t transfer_size = static_cast<size_t>(cfg.num_tracks) * cfg.param_bytes;

    void *d_buf, *h_buf;
    float *d_dedup, *d_chi2;
    CUDA_CHECK(cudaMalloc(&d_buf, transfer_size));
    CUDA_CHECK(cudaMalloc(&d_dedup, cfg.num_tracks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_chi2, sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_buf, transfer_size));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, transfer_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, transfer_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long> event_times;

    for (int iter = 0; iter < num_iterations; iter++) {
        auto start = high_resolution_clock::now();

        for (int step = 0; step < cfg.num_steps; step++) {
            // "FPGA" round-trip (sequential)
            CUDA_CHECK(cudaMemcpy(h_buf, d_buf, transfer_size, cudaMemcpyDeviceToHost));
            if (cfg.fpga_compute_us > 0) {
                std::this_thread::sleep_for(microseconds(cfg.fpga_compute_us));
            }
            CUDA_CHECK(cudaMemcpy(d_buf, h_buf, transfer_size, cudaMemcpyHostToDevice));

            // Dedup kernel (must wait for FPGA)
            int blocks = (cfg.num_tracks + 255) / 256;
            dedup_kernel<<<blocks, 256, 0, stream>>>(d_dedup, cfg.num_tracks, cfg.dedup_iterations);

            // Chi² accumulation
            CUDA_CHECK(cudaMemsetAsync(d_chi2, 0, sizeof(float), stream));
            chi2_kernel<<<blocks, 256, 0, stream>>>(d_chi2, d_dedup, cfg.num_tracks);

            // Sync before next step
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        auto end = high_resolution_clock::now();
        event_times.push_back(duration_cast<microseconds>(end - start).count());
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_dedup));
    CUDA_CHECK(cudaFree(d_chi2));
    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Calculate statistics
    std::sort(event_times.begin(), event_times.end());
    long median = event_times[num_iterations / 2];

    return {
        .total_us = median,
        .pct_of_budget = median / 230.0,
        .per_step_us = median / (double)cfg.num_steps
    };
}

TestResult test_overlapped(const OverlapConfig& cfg, int num_iterations) {
    const size_t transfer_size = static_cast<size_t>(cfg.num_tracks) * cfg.param_bytes;

    // Double buffering for overlap
    void *d_buf[2], *h_buf;
    float *d_dedup[2], *d_chi2;
    CUDA_CHECK(cudaMalloc(&d_buf[0], transfer_size));
    CUDA_CHECK(cudaMalloc(&d_buf[1], transfer_size));
    CUDA_CHECK(cudaMalloc(&d_dedup[0], cfg.num_tracks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dedup[1], cfg.num_tracks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_chi2, sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_buf, transfer_size));

    cudaStream_t stream_fpga, stream_dedup;
    CUDA_CHECK(cudaStreamCreate(&stream_fpga));
    CUDA_CHECK(cudaStreamCreate(&stream_dedup));

    cudaEvent_t fpga_done, dedup_done;
    CUDA_CHECK(cudaEventCreate(&fpga_done));
    CUDA_CHECK(cudaEventCreate(&dedup_done));

    // Warmup
    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf[0], transfer_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_buf[0], h_buf, transfer_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long> event_times;

    for (int iter = 0; iter < num_iterations; iter++) {
        auto start = high_resolution_clock::now();

        int curr = 0;
        int prev = 1;

        for (int step = 0; step < cfg.num_steps; step++) {
            // Stream 1: "FPGA" communication for CURRENT step
            CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf[curr], transfer_size,
                                       cudaMemcpyDeviceToHost, stream_fpga));
            CUDA_CHECK(cudaStreamSynchronize(stream_fpga));  // Wait for D2H

            // "FPGA compute" - this happens on CPU/FPGA, not GPU
            if (cfg.fpga_compute_us > 0) {
                std::this_thread::sleep_for(microseconds(cfg.fpga_compute_us));
            }

            CUDA_CHECK(cudaMemcpyAsync(d_buf[curr], h_buf, transfer_size,
                                       cudaMemcpyHostToDevice, stream_fpga));
            CUDA_CHECK(cudaEventRecord(fpga_done, stream_fpga));

            // Stream 2: Dedup + Chi² of PREVIOUS step's data (OVERLAPPED!)
            if (step > 0) {
                int blocks = (cfg.num_tracks + 255) / 256;
                dedup_kernel<<<blocks, 256, 0, stream_dedup>>>(
                    d_dedup[prev], cfg.num_tracks, cfg.dedup_iterations);
                CUDA_CHECK(cudaMemsetAsync(d_chi2, 0, sizeof(float), stream_dedup));
                chi2_kernel<<<blocks, 256, 0, stream_dedup>>>(d_chi2, d_dedup[prev], cfg.num_tracks);
                CUDA_CHECK(cudaEventRecord(dedup_done, stream_dedup));
            }

            // Wait for BOTH streams before proceeding
            // (CKF requires both current FPGA result and previous dedup to be done)
            CUDA_CHECK(cudaEventSynchronize(fpga_done));
            if (step > 0) {
                CUDA_CHECK(cudaEventSynchronize(dedup_done));
            }

            // Swap buffers
            std::swap(curr, prev);
        }

        // Final dedup for last step (no overlap possible)
        int blocks = (cfg.num_tracks + 255) / 256;
        dedup_kernel<<<blocks, 256, 0, stream_dedup>>>(
            d_dedup[prev], cfg.num_tracks, cfg.dedup_iterations);
        CUDA_CHECK(cudaMemsetAsync(d_chi2, 0, sizeof(float), stream_dedup));
        chi2_kernel<<<blocks, 256, 0, stream_dedup>>>(d_chi2, d_dedup[prev], cfg.num_tracks);
        CUDA_CHECK(cudaStreamSynchronize(stream_dedup));

        auto end = high_resolution_clock::now();
        event_times.push_back(duration_cast<microseconds>(end - start).count());
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_buf[0]));
    CUDA_CHECK(cudaFree(d_buf[1]));
    CUDA_CHECK(cudaFree(d_dedup[0]));
    CUDA_CHECK(cudaFree(d_dedup[1]));
    CUDA_CHECK(cudaFree(d_chi2));
    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaStreamDestroy(stream_fpga));
    CUDA_CHECK(cudaStreamDestroy(stream_dedup));
    CUDA_CHECK(cudaEventDestroy(fpga_done));
    CUDA_CHECK(cudaEventDestroy(dedup_done));

    // Calculate statistics
    std::sort(event_times.begin(), event_times.end());
    long median = event_times[num_iterations / 2];

    return {
        .total_us = median,
        .pct_of_budget = median / 230.0,
        .per_step_us = median / (double)cfg.num_steps
    };
}

void run_comparison(const OverlapConfig& cfg, int num_iterations) {
    const size_t transfer_size = static_cast<size_t>(cfg.num_tracks) * cfg.param_bytes;

    printf("\n");
    printf("========================================\n");
    printf("FPGA Compute: %d µs\n", cfg.fpga_compute_us);
    printf("========================================\n");
    printf("\n");
    printf("Configuration:\n");
    printf("  Tracks: %d\n", cfg.num_tracks);
    printf("  Transfer size: %.1f KB\n", transfer_size / 1024.0);
    printf("  Steps: %d\n", cfg.num_steps);
    printf("  Dedup iterations: %d\n", cfg.dedup_iterations);

    // Measure actual dedup kernel duration
    float dedup_us = measure_dedup_duration(cfg.num_tracks, cfg.dedup_iterations);
    printf("  Measured dedup duration: %.1f µs\n", dedup_us);

    printf("\nRunning tests (%d iterations each)...\n", num_iterations);

    TestResult seq = test_sequential(cfg, num_iterations);
    TestResult ovl = test_overlapped(cfg, num_iterations);

    printf("\n=== Results ===\n");
    printf("\n");
    printf("                    | Sequential | Overlapped | Improvement\n");
    printf("--------------------|------------|------------|------------\n");
    printf("Total (µs)          | %10ld | %10ld | %+.1f%%\n",
           seq.total_us, ovl.total_us,
           (1.0 - (double)ovl.total_us / seq.total_us) * 100.0);
    printf("Per step (µs)       | %10.1f | %10.1f | %+.1f%%\n",
           seq.per_step_us, ovl.per_step_us,
           (1.0 - ovl.per_step_us / seq.per_step_us) * 100.0);
    printf("%% of 23ms budget    | %9.1f%% | %9.1f%% | %+.1fpp\n",
           seq.pct_of_budget, ovl.pct_of_budget,
           seq.pct_of_budget - ovl.pct_of_budget);

    double improvement = (1.0 - (double)ovl.total_us / seq.total_us) * 100.0;
    double saved_us = seq.total_us - ovl.total_us;

    printf("\n=== Analysis ===\n");
    printf("Time saved by overlap: %.0f µs (%.1f%%)\n", (double)saved_us, improvement);
    printf("Theoretical max overlap: %.1f µs (dedup duration × %d steps)\n",
           dedup_us * (cfg.num_steps - 1), cfg.num_steps - 1);

    if (improvement > 20.0) {
        printf("\n✓ Overlap provides SIGNIFICANT benefit (>20%% improvement)\n");
        printf("  Async overlap mitigation is effective.\n");
    } else if (improvement > 10.0) {
        printf("\n⚠ Overlap provides MODERATE benefit (10-20%% improvement)\n");
        printf("  Async overlap helps but may not be sufficient alone.\n");
    } else if (improvement > 0.0) {
        printf("\n⚠ Overlap provides MINIMAL benefit (<10%% improvement)\n");
        printf("  Dedup kernel may be too fast to hide FPGA latency.\n");
    } else {
        printf("\n✗ Overlap provides NO benefit or is WORSE\n");
        printf("  Synchronization overhead dominates.\n");
    }

    // Check if we're within budget
    if (ovl.pct_of_budget < 15.0) {
        printf("\n✓ Overlapped execution is within acceptable budget (<15%%)\n");
    } else if (ovl.pct_of_budget < 25.0) {
        printf("\n⚠ Overlapped execution is marginally acceptable (15-25%%)\n");
    } else {
        printf("\n✗ Overlapped execution exceeds acceptable budget (>25%%)\n");
    }
}

int main(int argc, char** argv) {
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== GPU Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);

    const int num_iterations = 50;

    // Test with different FPGA compute times
    // Higher FPGA compute = more opportunity for overlap
    std::vector<int> fpga_times = {0, 50, 100, 200, 300, 500};

    // Base configuration: average tracks, params only
    OverlapConfig base_cfg = {
        .num_tracks = 6666,
        .param_bytes = 24,
        .num_steps = 15,
        .fpga_compute_us = 0,      // Set per test
        .dedup_iterations = 500    // ~100µs dedup kernel
    };

    printf("\n");
    printf("########################################################\n");
    printf("# ASYNC OVERLAP FEASIBILITY TEST\n");
    printf("########################################################\n");
    printf("\n");
    printf("This test compares sequential vs overlapped execution:\n");
    printf("\n");
    printf("Sequential:  [FPGA] → [Dedup] → [FPGA] → [Dedup] → ...\n");
    printf("Overlapped:  [FPGA] → [FPGA+Dedup] → [FPGA+Dedup] → ...\n");
    printf("                      ↑ overlapped ↑\n");
    printf("\n");
    printf("If overlap significantly reduces total time, async\n");
    printf("mitigation can hide FPGA synchronization latency.\n");

    for (int fpga_us : fpga_times) {
        OverlapConfig cfg = base_cfg;
        cfg.fpga_compute_us = fpga_us;
        run_comparison(cfg, num_iterations);
    }

    printf("\n");
    printf("########################################################\n");
    printf("# VARYING DEDUP KERNEL DURATION\n");
    printf("########################################################\n");
    printf("\n");
    printf("Testing with different dedup kernel durations to find\n");
    printf("the minimum GPU work needed to hide FPGA latency.\n");

    std::vector<int> dedup_iters = {100, 250, 500, 1000, 2000};

    for (int iters : dedup_iters) {
        OverlapConfig cfg = base_cfg;
        cfg.fpga_compute_us = 200;  // Fixed FPGA compute
        cfg.dedup_iterations = iters;

        float dedup_us = measure_dedup_duration(cfg.num_tracks, iters);
        printf("\n--- Dedup iterations: %d (~%.0f µs) ---\n", iters, dedup_us);
        run_comparison(cfg, num_iterations);
    }

    printf("\n");
    printf("########################################################\n");
    printf("# SUMMARY AND RECOMMENDATIONS\n");
    printf("########################################################\n");
    printf("\n");
    printf("Key findings:\n");
    printf("1. Overlap is most effective when dedup kernel time ≈ FPGA time\n");
    printf("2. If FPGA >> dedup, overlap provides diminishing returns\n");
    printf("3. Real CKF has additional GPU work (chi², sorting) that may help\n");
    printf("\n");
    printf("Recommendation:\n");
    printf("- If overlap saves >20%%: Async mitigation is viable\n");
    printf("- If overlap saves <10%%: Need to reduce FPGA compute time\n");
    printf("  or increase GPU work that can be overlapped\n");
    printf("\n");
    printf("Next step: If results are promising, proceed to Phase 4\n");
    printf("(XRT benchmark on actual Alveo V80 hardware)\n");

    return 0;
}
