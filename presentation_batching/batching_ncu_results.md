# NCU Profiling Results: Batching Optimization

**Date:** 2026-01-05
**Hardware:** NVIDIA GeForce RTX 2080 Ti (CC 7.5, 68 SMs)
**NCU Version:** Nsight Compute (full profile set)
**Dataset:** odd/geant4_ttbar_mu200

---

## Profile Commits

| Version | Commit | Description |
|---------|--------|-------------|
| Baseline | `5cd477ac` | No batching (single-event processing) |
| Optimized | `3ad492b5` | Batch-48 multi-event processing |

---

## Executive Summary

The NCU kernel-level profiling reveals the optimization achieves significant improvements:

| Metric | Improvement |
|--------|-------------|
| **fit_forward Occupancy** | +150% (15.4% → 38.4%) |
| **fit_backward Occupancy** | +100% (14.9% → 29.9%) |
| **fit_forward Memory Throughput** | +120% (61.5 → 135.7 GB/s) |
| **fit_backward Memory Throughput** | +154% (96.8 → 245.5 GB/s) |
| **propagate_to_next_surface Occupancy** | +22% (35.0% → 42.8%) |
| **propagate_to_next_surface Memory Throughput** | +18% (224 → 264 GB/s) |
| **find_tracks Memory Throughput** | +32% (43.8 → 57.7 GB/s) |

---

## Top Kernel Comparison

### 1. propagate_to_next_surface (CKF Propagation)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Grid Size | (236, 1, 1) | (1435, 1, 1) | **+508%** |
| Block Size | (128, 1, 1) | (128, 1, 1) | - |
| Duration | 2.62 ms | 13.55 ms | +417% |
| Achieved Occupancy | 35.0% | 42.8% | **+22%** |
| Memory Throughput | 224.5 GB/s | 264.0 GB/s | **+18%** |
| L1/TEX Hit Rate | 48.3% | 43.1% | -11% |
| L2 Hit Rate | 78.2% | 77.0% | -2% |

**Analysis:** The optimized version processes 6x more work per kernel invocation due to batching. Occupancy improvement (+22%) indicates better SM utilization. Memory throughput increases (+18%) due to improved memory access coalescing across batched events. The slight L1 hit rate decrease is expected with larger working sets but is offset by higher overall throughput.

---

### 2. find_tracks (CKF Track Finding)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Grid Size | (471, 1, 1) | (2870, 1, 1) | **+509%** |
| Block Size | (64, 1, 1) | (64, 1, 1) | - |
| Duration | 723 µs | 4.02 ms | +456% |
| Achieved Occupancy | 21.4% | 23.7% | **+11%** |
| Memory Throughput | 43.8 GB/s | 57.7 GB/s | **+32%** |

**Analysis:** Grid size increases 6x reflecting batch processing. Memory throughput improves significantly (+32%), indicating better utilization of the memory subsystem through batched data access patterns.

---

### 3. fit_forward (Kalman Forward Pass)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Grid Size | (112, 1, 1) | (482, 1, 1) | **+330%** |
| Block Size | (128, 1, 1) | (128, 1, 1) | - |
| Duration | 18.99 ms | 68.23 ms | +259% |
| Achieved Occupancy | 15.4% | 38.4% | **+150%** |
| Memory Throughput | 61.5 GB/s | 135.7 GB/s | **+121%** |

**Analysis:** Most significant improvement. Occupancy more than doubles from 15.4% to 38.4%, indicating the batched version keeps SMs much busier. Memory throughput more than doubles, showing the optimization effectively reduces memory stalls. The 4.3x grid size increase corresponds to batch processing of multiple events.

---

### 4. fit_backward (Kalman Backward Pass)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Grid Size | (112, 1, 1) | (482, 1, 1) | **+330%** |
| Block Size | (128, 1, 1) | (128, 1, 1) | - |
| Duration | 8.22 ms | 27.29 ms | +232% |
| Achieved Occupancy | 14.9% | 29.9% | **+100%** |
| Memory Throughput | 96.8 GB/s | 245.5 GB/s | **+154%** |

**Analysis:** Occupancy doubles from 14.9% to 29.9%. Memory throughput more than doubles (+154%), the highest improvement across all kernels. This indicates the backward pass particularly benefits from batched memory access patterns.

---

## Cache Performance

| Kernel | Metric | Baseline | Optimized | Change |
|--------|--------|----------|-----------|--------|
| propagate_to_next_surface | L1 Hit Rate | 48.3% | 43.1% | -11% |
| propagate_to_next_surface | L2 Hit Rate | 78.2% | 77.0% | -2% |
| find_tracks | L1 Hit Rate | ~60% | ~45% | -25% |
| find_tracks | L2 Hit Rate | ~82% | ~77% | -6% |

**Analysis:** Cache hit rates show slight decreases in the optimized version due to larger working sets from batched processing. However, this is a deliberate trade-off:
- The reduced cache hit rate is offset by significantly higher memory throughput
- Larger batches enable better coalescing and bandwidth utilization
- Overall performance improves despite lower cache hit rates

---

## Occupancy Analysis

| Kernel | Theoretical | Baseline Achieved | Optimized Achieved |
|--------|-------------|-------------------|-------------------|
| propagate_to_next_surface | 100% | 35.0% | 42.8% |
| find_tracks | 100% | 21.4% | 23.7% |
| fit_forward | 100% | 15.4% | 38.4% |
| fit_backward | 100% | 14.9% | 29.9% |

**Key Finding:** The fitting kernels show the most dramatic occupancy improvements. The baseline's low occupancy (14-15%) indicates severe under-utilization that batching directly addresses.

---

## Grid Size Scaling

| Kernel | Baseline Grid | Optimized Grid | Scale Factor |
|--------|---------------|----------------|--------------|
| propagate_to_next_surface | 236 | 1,435 | 6.1x |
| find_tracks | 471 | 2,870 | 6.1x |
| fit_forward | 112 | 482 | 4.3x |
| fit_backward | 112 | 482 | 4.3x |

**Analysis:** The ~6x increase in CKF kernel grid sizes corresponds to batch-48 processing (accounts for varying track counts per event). The ~4.3x increase in fitting kernels shows consistent batching behavior.

---

## Conclusions

### 1. Batching Successfully Improves GPU Utilization

The NCU profiles confirm that batching addresses the fundamental problem of low occupancy in single-event processing:

- **Fitting kernels** see 100-150% occupancy improvement (from ~15% to 30-38%)
- **CKF kernels** see 11-22% occupancy improvement (from 21-35% to 24-43%)

### 2. Memory Throughput Dramatically Improves

Batching enables better memory access patterns:

- **fit_backward**: +154% memory throughput (96.8 → 245.5 GB/s)
- **fit_forward**: +121% memory throughput (61.5 → 135.7 GB/s)
- **find_tracks**: +32% memory throughput (43.8 → 57.7 GB/s)
- **propagate_to_next_surface**: +18% memory throughput (224 → 264 GB/s)

### 3. Trade-off: Individual Kernel Duration vs Total Throughput

While individual kernel durations increase (due to processing more work), the key insight is:
- Fewer kernel launches needed (92% reduction in sync calls from NSYS data)
- Better GPU utilization per kernel
- Net result: 93% throughput improvement (15.61 → 30.12 ev/s)

### 4. Future Optimization Opportunities

NCU data suggests potential improvements:
- **Occupancy headroom**: fit_forward (38%) and fit_backward (30%) still have room for improvement
- **Cache optimization**: The L1 hit rate drop could be partially addressed with data layout changes
- **Register pressure**: All kernels use 48 registers/thread; optimization could reduce this

---

## References

- NSYS Results: `docs/batching_profile_results.md`
- Optimization Report: `docs/batching_report.md`
- NCU Guide: `docs/batching_ncu_guide.md`
- Profile Data: `profiles-batching/`
