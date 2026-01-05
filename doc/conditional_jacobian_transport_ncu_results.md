# Conditional Jacobian Aggregation: ncu Profiling Results

**Date:** 2026-01-05
**Machine:** NVIDIA GeForce RTX 2080 Ti, CUDA 12.4, Driver 560.x
**Profiler:** ncu 2024.1.1.0

---

## 1. Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 2080 Ti |
| Compute Capability | 7.5 |
| CUDA Version | 12.4 |
| ncu Version | 2024.1.1.0 |
| Baseline Commit | a48cc783 |
| Optimization Commit | 25894cca |

---

## 2. Key Finding: Register Reduction Achieved

**IMPORTANT:** Contrary to earlier cuobjdump analysis, ncu profiling reveals the optimization **successfully reduced register count from 128 to 96** for the `propagate_to_next_surface` kernel.

| Kernel | Baseline Registers | Optimization Registers | Difference |
|--------|-------------------|------------------------|------------|
| propagate_to_next_surface (mbf_on) | 128 | 96 | **-32 (-25%)** |

This register reduction enables higher occupancy and explains the throughput improvement.

---

## 3. Occupancy Analysis

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| Block Limit Registers | 4 | 5 | +1 block |
| Theoretical Active Warps/SM | 16 | 20 | +4 warps |
| Theoretical Occupancy (%) | 50.00 | 62.50 | **+12.50%** |
| Achieved Occupancy (%) | 39.28 | 48.62 | **+9.34%** |
| Achieved Active Warps/SM | 12.57 | 15.56 | +2.99 warps |

**Occupancy Limiter:** Registers (in both cases)

The register reduction from 128 to 96 allows one additional block per SM (from 4 to 5), increasing theoretical occupancy from 50% to 62.5%.

---

## 4. Memory Analysis

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| Memory Throughput (GB/s) | 218.34 | 244.69 | **+12.1%** |
| DRAM Throughput (%) | 33.17 | 37.74 | +4.57% |
| L1/TEX Hit Rate (%) | 54.12 | 46.31 | -7.81% |
| L2 Hit Rate (%) | 79.55 | 76.59 | -2.96% |
| Mem Busy (%) | 28.07 | 28.33 | +0.26% |

**Observation:** Memory throughput increased significantly (+12.1%), indicating better memory parallelism from higher occupancy. L1 hit rate decreased slightly, likely due to different memory access patterns with the conditional Jacobian transport.

---

## 5. Compute Analysis

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| Duration (ms) | 3.99 | 3.61 | **-9.5% (faster)** |
| Compute (SM) Throughput (%) | 8.86 | 9.19 | +0.33% |
| Executed IPC | 0.26 | 0.26 | 0 |
| SM Busy (%) | 6.57 | 6.50 | -0.07% |
| Executed Instructions | 82.3M | 78.5M | **-4.6%** |

**Observation:** Total executed instructions decreased by 4.6%, confirming that the conditional Jacobian transport skips unnecessary computation.

---

## 6. Instruction Statistics

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| Executed Instructions | 82,324,275 | 78,537,789 | **-3,786,486 (-4.6%)** |
| Issued Instructions | 82,528,994 | 78,727,622 | -3,801,372 (-4.6%) |
| Branch Instructions | 6,677,603 | 6,662,922 | -14,681 (-0.2%) |
| Branch Efficiency (%) | 93.26 | 93.24 | -0.02% |

---

## 7. Warp State Statistics

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| Warp Cycles Per Issued Instruction | 47.74 | 59.92 | +12.18 |
| Avg. Active Threads Per Warp | 8.73 | 8.69 | -0.04 |
| Scheduler: No Eligible (%) | 93.24 | 93.27 | +0.03% |
| Active Warps Per Scheduler | 3.23 | 4.03 | +0.80 |

---

## 8. Conclusions

### 8.1 Register Reduction Validated

The optimization achieves a **25% reduction in register usage** (128 → 96), directly contradicting the earlier cuobjdump analysis that reported no register change. This explains the performance improvement mechanism:

1. **Fewer registers** → More blocks per SM
2. **More blocks** → Higher occupancy (50% → 62.5% theoretical)
3. **Higher occupancy** → Better latency hiding and throughput

### 8.2 Performance Improvement Mechanism

The optimization improves performance through two complementary mechanisms:

1. **Register Reduction (-32 registers):** Enables 25% higher theoretical occupancy
2. **Instruction Reduction (-4.6%):** Skipping unnecessary Jacobian aggregation reduces total work

### 8.3 Measured Improvements

| Metric | Improvement |
|--------|-------------|
| Kernel Duration | -9.5% (3.99ms → 3.61ms) |
| Achieved Occupancy | +23.8% relative (39.3% → 48.6%) |
| Memory Throughput | +12.1% (218 → 245 GB/s) |
| Instructions Executed | -4.6% |

### 8.4 Current Bottleneck

The kernel remains **latency-bound** rather than compute or memory bound:
- SM Busy: ~6.5%
- Compute Throughput: ~9%
- Memory Throughput: ~37%
- Scheduler: 93% of cycles have no eligible warps

Further optimization opportunities exist in reducing warp stalls and improving instruction-level parallelism.

---

## 9. Correction to Previous Analysis

The earlier profiling report (`doc/conditional_jacobian_transport_profile_report.md`) incorrectly stated:
> "Register count: 128 in ALL variants (no reduction)"

This ncu analysis confirms the optimization **does achieve register reduction** from 128 to 96, which is the primary driver of the throughput improvement through increased occupancy.

---

## 10. Raw Data Files

| File | Description |
|------|-------------|
| `build_ncu_baseline/baseline_ncu_full.txt` | Full baseline ncu profile |
| `build_ncu_opt/optimization_ncu_full.txt` | Full optimization ncu profile |

---

## 11. Methodology

1. Profiled using `ncu --set full --print-summary per-kernel`
2. Single event, single CPU thread configuration
3. Focused on `propagate_to_next_surface` kernel (primary track finding kernel)
4. RTX 2080 Ti (sm_75) used for profiling
