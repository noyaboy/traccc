# Conditional Jacobian Transport Profiling Report

**Date:** 2026-01-04
**Test Environment:** Tesla V100-SXM2-32GB (compute capability 7.0)
**Dataset:** `odd/geant4_ttbar_mu200/` (10 events for profiling)
**Configuration:** 1 CPU thread (to isolate GPU behavior)
**Tools:** Nsight Systems (nsys) 2024.5.1

---

## 1. Overview

This report presents the nsys profiling results comparing baseline (`a48cc783`) and optimization (`25894cca`) commits for the conditional Jacobian transport implementation.

### Commits Under Test

| Commit | Description | Profile File |
|--------|-------------|--------------|
| `a48cc783` | First round of MBF cleanup (baseline) | `baseline_nsys.nsys-rep` |
| `25894cca` | Conditional Jacobian transport (optimization) | `optimization_nsys.nsys-rep` |

### Previous Benchmark Results (Non-Profiled)

| Commit | Throughput | Latency |
|--------|------------|---------|
| Baseline | 38.75 events/s | 25.80 ms/event |
| Optimization | 43.27 events/s | 23.11 ms/event |
| **Improvement** | **+11.67%** | **-10.4%** |

---

## 2. NVTX Range Summary

High-level timing breakdown from NVTX annotations.

### Baseline (`a48cc783`)

| Range | Time (%) | Total Time (ms) |
|-------|----------|-----------------|
| File reading | 62.3% | 7,342.28 |
| Event processing | 32.4% | 3,820.21 |
| Warm-up processing | 4.4% | 514.04 |

### Optimization (`25894cca`)

| Range | Time (%) | Total Time (ms) |
|-------|----------|-----------------|
| File reading | 59.2% | 6,893.96 |
| Event processing | 35.8% | 4,162.32 |
| Warm-up processing | 4.1% | 474.08 |

**Note:** Profiling overhead affects absolute times. Relative comparisons within each profile are more meaningful.

---

## 3. CUDA GPU Kernel Summary

### 3.1 Target Kernel: `propagate_to_next_surface`

The primary optimization target kernel.

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| GPU Time % | 63.4% | 65.6% | +2.2% |
| Total Time (ms) | 2,015.79 | 2,084.61 | +3.4% |
| Instances | 2,144 | 2,185 | +1.9% |
| Avg (µs) | 940.2 | 954.1 | +1.5% |
| Med (µs) | 864.5 | 881.3 | +2.0% |
| Min (µs) | 48.7 | 122.2 | +151% |
| Max (µs) | 2,793.5 | 3,031.4 | +8.5% |
| StdDev (µs) | 564.7 | 589.3 | +4.4% |

**Observation:** The per-instance average time remained essentially unchanged (~940-954 µs). The optimization did not significantly improve this kernel's execution time. More instances in the optimization run suggest more track candidates are being propagated.

### 3.2 Other CKF Kernels

| Kernel | Baseline Avg (µs) | Optimization Avg (µs) | Change |
|--------|-------------------|----------------------|--------|
| `find_tracks` | 143.8 | 125.8 | **-12.5%** |
| `build_tracks` | 512.1 | 70.0 | **-86.3%** |
| `apply_interaction` | 3.3 | 3.5 | +5.2% |
| `remove_duplicates` | 29.8 | 30.2 | +1.5% |

**Key Finding:** The `build_tracks` kernel shows a dramatic improvement from 512 µs to 70 µs per instance (-86.3%). This is likely the primary contributor to the overall throughput improvement.

### 3.3 Full Kernel Breakdown

#### Baseline Top 10 Kernels by GPU Time

| Rank | Kernel | Time (%) | Total (ms) | Instances | Avg (µs) |
|------|--------|----------|------------|-----------|----------|
| 1 | `propagate_to_next_surface` | 63.4% | 2,015.79 | 2,144 | 940.2 |
| 2 | `find_tracks` | 10.2% | 324.22 | 2,254 | 143.8 |
| 3 | `find_doublets` | 3.9% | 125.41 | 110 | 1,140.1 |
| 4 | `ccl_kernel` | 3.6% | 114.24 | 110 | 1,038.5 |
| 5 | `count_triplets` | 2.7% | 85.34 | 110 | 775.8 |
| 6 | `count_doublets` | 2.6% | 81.21 | 110 | 738.2 |
| 7 | `DeviceRadixSortOnesweep` | 2.2% | 71.13 | 2,556 | 27.8 |
| 8 | `build_tracks` | 1.8% | 56.33 | 110 | 512.1 |
| 9 | `DeviceMergeSortBlockSort` | 1.7% | 55.20 | 110 | 501.8 |
| 10 | `remove_duplicates` | 1.5% | 47.47 | 1,594 | 29.8 |

#### Optimization Top 10 Kernels by GPU Time

| Rank | Kernel | Time (%) | Total (ms) | Instances | Avg (µs) |
|------|--------|----------|------------|-----------|----------|
| 1 | `propagate_to_next_surface` | 65.6% | 2,084.61 | 2,185 | 954.1 |
| 2 | `find_tracks` | 9.1% | 288.65 | 2,295 | 125.8 |
| 3 | `find_doublets` | 4.0% | 126.64 | 110 | 1,151.3 |
| 4 | `ccl_kernel` | 3.3% | 105.57 | 110 | 959.7 |
| 5 | `count_triplets` | 2.7% | 85.74 | 110 | 779.5 |
| 6 | `DeviceRadixSortOnesweep` | 2.6% | 83.89 | 3,044 | 27.6 |
| 7 | `count_doublets` | 2.5% | 80.88 | 110 | 735.3 |
| 8 | `DeviceMergeSortBlockSort` | 1.8% | 55.96 | 110 | 508.7 |
| 9 | `remove_duplicates` | 1.6% | 49.41 | 1,635 | 30.2 |
| 10 | `find_triplets` | 1.2% | 37.63 | 110 | 342.1 |

**Notable Change:** `build_tracks` dropped from rank 8 (1.8%, 56.33 ms) to outside top 10 (0.2%, 7.70 ms).

---

## 4. CUDA API Summary

### 4.1 API Call Comparison

| API Call | Baseline Calls | Optimization Calls | Baseline Time (ms) | Optimization Time (ms) |
|----------|----------------|--------------------|--------------------|------------------------|
| `cudaStreamSynchronize` | 13,156 | 13,021 | 2,911.60 | 2,908.12 |
| `cudaMemcpyAsync` | 10,894 | 10,677 | 527.67 | 510.51 |
| `cudaLaunchKernel` | 23,047 | 24,430 | 175.54 | 167.00 |
| `cudaEventSynchronize` | 886 | 886 | 48.80 | 57.71 |
| `cudaMemsetAsync` | 8,360 | 9,667 | 38.21 | 49.83 |

**Observation:** More kernel launches in optimization (24,430 vs 23,047, +6%) due to more track candidates being processed.

---

## 5. CUDA GPU Memory Operations

### 5.1 Memory Transfer Time

| Operation | Baseline (ms) | Optimization (ms) | Change |
|-----------|---------------|-------------------|--------|
| Host-to-Device | 115.96 | 124.08 | +7.0% |
| Device-to-Host | 45.37 | 54.73 | +20.6% |
| Device-to-Device | 10.55 | 10.12 | -4.1% |
| Memset | 12.12 | 14.41 | +18.9% |
| **Total** | **184.00** | **203.34** | **+10.5%** |

### 5.2 Memory Transfer Size

| Operation | Baseline (MB) | Optimization (MB) | Change |
|-----------|---------------|-------------------|--------|
| Host-to-Device | 889.64 | 903.63 | +1.6% |
| Device-to-Host | 456.38 | 562.38 | +23.2% |
| Device-to-Device | 697.43 | 702.37 | +0.7% |
| Memset | 373.07 | 383.96 | +2.9% |

**Observation:** Optimization processes more data (more tracks found), resulting in higher memory traffic, particularly in Device-to-Host transfers (+23.2%).

---

## 6. Analysis and Conclusions

### 6.1 Primary Findings

1. **`propagate_to_next_surface` kernel unchanged**: The per-instance execution time remained essentially the same (~940-954 µs). The optimization did not improve this kernel's performance.

2. **`build_tracks` dramatically improved**: 512 µs → 70 µs per instance (-86.3%). This is the primary contributor to throughput gains.

3. **`find_tracks` improved**: 144 µs → 126 µs per instance (-12.5%).

4. **More work processed**: The optimization finds more track candidates:
   - `propagate_to_next_surface` instances: 2,144 → 2,185 (+1.9%)
   - `find_tracks` instances: 2,254 → 2,295 (+1.8%)
   - `remove_duplicates` instances: 1,594 → 1,635 (+2.6%)

### 6.2 Hypothesis Evaluation

| Claim | Expected | Observed | Status |
|-------|----------|----------|--------|
| `propagate_to_next_surface` speedup | -10-15% per instance | +1.5% per instance | **NOT VALIDATED** |
| Register pressure reduction | ~64 registers saved | Unknown (needs ncu) | **NEEDS VERIFICATION** |
| Occupancy improvement | +10-25% | Unknown (needs ncu) | **NEEDS VERIFICATION** |
| Overall throughput gain | +5-15% | +11.67% (benchmark) | **VALIDATED** |

### 6.3 Possible Explanations

1. **Throughput gain from different source**: The +11.67% throughput improvement appears to come from `build_tracks` (-86.3%) and `find_tracks` (-12.5%), not from `propagate_to_next_surface`.

2. **Kernel is memory-bound**: The `propagate_to_next_surface` kernel may be memory-bound rather than compute-bound, meaning register pressure reduction has limited impact.

3. **Register savings may be smaller than expected**: The actual register reduction may be less than the theoretical 64 registers.

4. **Profiling overhead**: nsys profiling adds overhead that may obscure small performance differences.

### 6.4 Recommended Next Steps

1. **Run ncu profiling** to verify:
   - Actual register count per thread (`launch__registers_per_thread`)
   - Achieved occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`)
   - Occupancy limiter (registers vs shared memory vs block size)
   - Memory throughput and compute utilization

2. **Investigate `build_tracks` improvement**: Understand why this kernel improved so dramatically.

3. **Profile with larger dataset**: Use more events to reduce variance and profiling overhead impact.

---

## 7. Raw Data

### 7.1 Profile Files

| File | Size | Location |
|------|------|----------|
| `baseline_nsys.nsys-rep` | 4.2 MB | `build/baseline_nsys.nsys-rep` |
| `optimization_nsys.nsys-rep` | 4.5 MB | `build/optimization_nsys.nsys-rep` |
| `baseline_nsys.sqlite` | - | `build/baseline_nsys.sqlite` |
| `optimization_nsys.sqlite` | - | `build/optimization_nsys.sqlite` |

### 7.2 Commands Used

```bash
# Baseline profiling
git checkout a48cc783
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . -j4
/usr/local/cuda-12.6/bin/nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=baseline_nsys \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=10 \
    --cpu-threads=1

# Optimization profiling
git checkout 25894cca
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . -j4
/usr/local/cuda-12.6/bin/nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=optimization_nsys \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=10 \
    --cpu-threads=1

# Generate stats
/usr/local/cuda-12.6/bin/nsys stats baseline_nsys.nsys-rep
/usr/local/cuda-12.6/bin/nsys stats optimization_nsys.nsys-rep
```

---

## 8. References

- `doc/conditional_jacobian_transport_report.md` - Benchmark results and implementation details
- `doc/conditional_jacobian_transport_plan.md` - Implementation plan
- `doc/conditional_jacobian_transport_profile_plan.md` - Profiling plan
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
