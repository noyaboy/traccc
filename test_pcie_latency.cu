#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // Test different sizes
    const int sizes[] = {48 * 1024, 112 * 1024, 256 * 1024};  // 48KB, 112KB, 256KB
    const int steps = 15;
    const int warmup = 5;
    const int iterations = 10;

    printf("=== PCIe Round-Trip Latency Test ===\n");
    printf("Simulating GPU <-> FPGA communication\n");
    printf("Steps per event: %d\n\n", steps);

    // Get device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("PCI Bus ID: %04x:%02x:%02x.0\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    printf("Note: Run 'nvidia-smi -q | grep -i pcie' for actual PCIe Gen/width\n\n");

    for (int s = 0; s < 3; s++) {
        const int size = sizes[s];

        void *d_buf, *h_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, size));
        CUDA_CHECK(cudaMallocHost(&h_buf, size));  // Pinned memory for faster transfer

        // Warmup
        for (int i = 0; i < warmup; i++) {
            CUDA_CHECK(cudaMemcpy(h_buf, d_buf, size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Measure
        long total_us = 0;
        for (int iter = 0; iter < iterations; iter++) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < steps; i++) {
                CUDA_CHECK(cudaMemcpy(h_buf, d_buf, size, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            auto end = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }

        double avg_event_us = (double)total_us / iterations;
        double avg_step_us = avg_event_us / steps;
        double percent_of_23ms = (avg_event_us / 23000.0) * 100.0;

        printf("Transfer size: %d KB\n", size / 1024);
        printf("  Per event (%d steps): %.1f us\n", steps, avg_event_us);
        printf("  Per step (round-trip): %.1f us\n", avg_step_us);
        printf("  %% of 23ms event: %.2f%%\n\n", percent_of_23ms);

        CUDA_CHECK(cudaFree(d_buf));
        CUDA_CHECK(cudaFreeHost(h_buf));
    }

    printf("=== Interpretation ===\n");
    printf("< 1%%  : Communication overhead negligible\n");
    printf("1-5%% : Acceptable overhead\n");
    printf("5-10%%: Concerning, needs optimization\n");
    printf("> 10%%: May negate FPGA benefits\n\n");

    // ============================================================
    // ASYMMETRIC TRANSFER TEST (realistic GPU-FPGA scenarios)
    // Based on Nsys data: avg 6,666 tracks/step, 15 steps/event
    // ============================================================
    printf("=== Asymmetric Transfer Test (Realistic Scenarios) ===\n");
    printf("Based on Nsys: avg 6,666 tracks/step, 15 steps/event\n\n");

    // Scenario definitions: {name, gpu_to_fpga_bytes, fpga_to_gpu_bytes}
    struct Scenario {
        const char* name;
        int to_fpga;    // GPU -> FPGA (Host in test)
        int from_fpga;  // FPGA -> GPU (Host in test)
    };

    // Track counts from Nsys: min=128, max=42240, avg=6666
    // Data sizes per track:
    //   - Parameters only: 24 bytes (6 floats)
    //   - Full bound_track_params: 176 bytes (24 params + 144 cov + 8 barcode)
    //   - Jacobian: 144 bytes (6x6 matrix)
    //   - Chi²: 4 bytes
    const int avg_tracks = 6666;
    const int min_tracks = 128;
    const int max_tracks = 42240;

    Scenario scenarios[] = {
        // Scenario 1: Parameters only (minimal, as claimed in survey)
        {"Params only (avg tracks)", avg_tracks * 24, avg_tracks * 24},

        // Scenario 2: Full track parameters (params + covariance + barcode)
        {"Full params (avg tracks)", avg_tracks * 176, avg_tracks * 176},

        // Scenario 3: Full params + Jacobians returned
        {"Full + Jacobians (avg)", avg_tracks * 176, avg_tracks * (176 + 144)},

        // Scenario 4: Min track count (best case)
        {"Params only (min tracks)", min_tracks * 24, min_tracks * 24},

        // Scenario 5: Max track count (worst case)
        {"Params only (max tracks)", max_tracks * 24, max_tracks * 24},
    };

    const int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    for (int s = 0; s < n_scenarios; s++) {
        const Scenario& sc = scenarios[s];
        const int max_size = (sc.to_fpga > sc.from_fpga) ? sc.to_fpga : sc.from_fpga;

        void *d_buf_to, *d_buf_from, *h_buf_to, *h_buf_from;
        CUDA_CHECK(cudaMalloc(&d_buf_to, sc.to_fpga));
        CUDA_CHECK(cudaMalloc(&d_buf_from, sc.from_fpga));
        CUDA_CHECK(cudaMallocHost(&h_buf_to, sc.to_fpga));
        CUDA_CHECK(cudaMallocHost(&h_buf_from, sc.from_fpga));

        // Warmup
        for (int i = 0; i < warmup; i++) {
            CUDA_CHECK(cudaMemcpy(h_buf_to, d_buf_to, sc.to_fpga, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_buf_from, h_buf_from, sc.from_fpga, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Measure
        long total_us = 0;
        for (int iter = 0; iter < iterations; iter++) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < steps; i++) {
                // GPU -> FPGA (simulated as D2H)
                CUDA_CHECK(cudaMemcpy(h_buf_to, d_buf_to, sc.to_fpga, cudaMemcpyDeviceToHost));
                // FPGA -> GPU (simulated as H2D)
                CUDA_CHECK(cudaMemcpy(d_buf_from, h_buf_from, sc.from_fpga, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            auto end = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }

        double avg_event_us = (double)total_us / iterations;
        double avg_step_us = avg_event_us / steps;
        double percent_of_23ms = (avg_event_us / 23000.0) * 100.0;
        double total_per_step = (sc.to_fpga + sc.from_fpga) / 1024.0;

        printf("Scenario: %s\n", sc.name);
        printf("  GPU->FPGA: %d KB, FPGA->GPU: %d KB (total: %.1f KB/step)\n",
               sc.to_fpga / 1024, sc.from_fpga / 1024, total_per_step);
        printf("  Per event (%d steps): %.1f us\n", steps, avg_event_us);
        printf("  Per step: %.1f us\n", avg_step_us);
        printf("  %% of 23ms event: %.2f%%\n\n", percent_of_23ms);

        CUDA_CHECK(cudaFree(d_buf_to));
        CUDA_CHECK(cudaFree(d_buf_from));
        CUDA_CHECK(cudaFreeHost(h_buf_to));
        CUDA_CHECK(cudaFreeHost(h_buf_from));
    }

    return 0;
}
