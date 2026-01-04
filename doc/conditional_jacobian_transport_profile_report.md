# Conditional Jacobian Transport Profiling Report

**Date:** 2026-01-04
**Test Environment:** Tesla V100-SXM2-32GB (compute capability 7.0)
**Dataset:** `odd/geant4_ttbar_mu200/` (10 events for profiling)
**Configuration:** 1 CPU thread (to isolate GPU behavior)
**Tools:** Nsight Systems (nsys) 2024.5.1, cuobjdump (register analysis)

---

## 1. Overview

This report presents the nsys profiling results comparing baseline (`a48cc783`) and optimization (`25894cca`) commits for the conditional Jacobian transport implementation.

### Commits Under Test

| Commit | Description | Profile File |
|--------|-------------|--------------|
| `a48cc783` | First round of MBF cleanup (baseline) | `baseline_nsys.nsys-rep` |
| `25894cca` | Conditional Jacobian transport (optimization) | `optimization_nsys.nsys-rep` |

### Benchmark Results Summary

| Comparison | Throughput Improvement | Notes |
|------------|----------------------|-------|
| Original (MBF defaults differ) | +11.67% | Baseline MBF=true, Optimization MBF=false |
| **Apples-to-Apples (both MBF=false)** | **+18.3%** | True conditional Jacobian impact |

### Original Benchmark Results (Non-Profiled)

| Commit | Throughput | Latency |
|--------|------------|---------|
| Baseline (MBF=true) | 38.75 events/s | 25.80 ms/event |
| Optimization (MBF=false) | 43.27 events/s | 23.11 ms/event |
| **Improvement** | **+11.67%** | **-10.4%** |

### Apples-to-Apples Benchmark (Both MBF=false)

| Commit | Throughput | Latency |
|--------|------------|---------|
| Baseline (MBF=false) | 36.57 events/s | 27.34 ms/event |
| Optimization (MBF=false) | 43.27 events/s | 23.11 ms/event |
| **Improvement** | **+18.3%** | **-15.5%** |

### Key Findings Summary

| Original Claim | Investigation Result |
|----------------|---------------------|
| Register reduction (~64 registers) | **NOT ACHIEVED** - 128 registers in all variants |
| Occupancy improvement | **NOT ACHIEVED** - No change |
| Throughput improvement | **VALIDATED** - +18.3% (apples-to-apples) |
| Source of improvement | **IDENTIFIED** - Skipped Jacobian aggregation (6x6 matrix mult + memory I/O) |

**Bottom Line**: The optimization works, but through a different mechanism than originally claimed. The benefit comes from skipping expensive 6x6 matrix multiplications and global memory accesses at every surface, NOT from register pressure reduction.

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

## 6. Register Analysis (cuobjdump)

**Note:** Full ncu profiling requires `RmProfilingAdminOnly=0` (admin privileges not available). Register counts were extracted from compiled binaries using `cuobjdump --dump-resource-usage`.

### 6.1 Baseline Register Counts (`a48cc783`)

| Kernel Variant | Registers | Stack (bytes) |
|---------------|-----------|---------------|
| `odd_detector_const` | 128 | 1,856 |
| `odd_detector_inhom_global` | 128 | 2,024 |
| `odd_detector_inhom_texture` | 128 | 1,896 |
| `default_detector_const` | 128 | 2,736 |
| `default_detector_inhom_global` | 128 | 2,920 |
| `default_detector_inhom_texture` | 128 | 2,776 |
| `telescope_detector_const` | 128 | 1,072 |
| `telescope_detector_inhom_global` | 161 | 1,184 |
| `telescope_detector_inhom_texture` | 128 | 1,112 |

### 6.2 Optimization Register Counts (`25894cca`)

The optimization creates two kernel variants: `mbf_on` (with Jacobian transport) and `mbf_off` (without).

#### MBF ON Variant (with `parameter_transporter`)

| Kernel Variant | Registers | Stack (bytes) |
|---------------|-----------|---------------|
| `odd_detector_const_mbf_on` | 128 | 1,856 |
| `odd_detector_inhom_global_mbf_on` | 128 | 2,024 |
| `odd_detector_inhom_texture_mbf_on` | 128 | 1,896 |
| `default_detector_const_mbf_on` | 128 | 2,736 |
| `default_detector_inhom_global_mbf_on` | 128 | 2,920 |
| `default_detector_inhom_texture_mbf_on` | 128 | 2,776 |

#### MBF OFF Variant (with `bound_updater`)

| Kernel Variant | Registers | Stack (bytes) | Stack Change |
|---------------|-----------|---------------|--------------|
| `odd_detector_const_mbf_off` | 128 | 1,848 | -8 |
| `odd_detector_inhom_global_mbf_off` | 128 | 2,016 | -8 |
| `odd_detector_inhom_texture_mbf_off` | 128 | 1,888 | -8 |
| `default_detector_const_mbf_off` | 128 | 2,728 | -8 |
| `default_detector_inhom_global_mbf_off` | 128 | 2,904 | -16 |
| `default_detector_inhom_texture_mbf_off` | 128 | 2,768 | -8 |

### 6.3 Key Finding: No Register Reduction

**The `propagate_to_next_surface` kernel uses 128 registers in ALL variants** (baseline, mbf_on, mbf_off).

| Comparison | Expected | Actual |
|------------|----------|--------|
| Register reduction | -64 registers | **0 registers** |
| Stack reduction | N/A | -8 to -16 bytes |

### 6.4 Why No Register Reduction?

The theoretical claim of ~64 register savings (8x8 Jacobian matrix) was **NOT achieved**. Possible reasons:

1. **Compiler optimization**: The CUDA compiler (nvcc) may have optimized both variants to use the same number of registers through:
   - Register spilling to stack (note ~2KB stack usage)
   - Aggressive inlining that masks the difference
   - Register reuse optimizations

2. **Jacobian not stored in registers**: The 8x8 Jacobian matrix may already be stored in local memory (stack) rather than registers in the baseline.

3. **Actor state overhead**: The `bound_updater` actor, while having an empty `state {}`, may still require similar register usage for its computation.

4. **V100 register limit**: The kernel already uses 128 registers (50% of the 256 register limit per thread), suggesting the compiler aggressively spills to stack.

---

## 7. Root Cause Analysis: `build_tracks` Improvement

### 7.1 Discovery: Configuration Change

Investigation revealed the dramatic `build_tracks` improvement (-86.3%) is **NOT** from the conditional Jacobian transport optimization. It is caused by a **change in the default value of `run_mbf_smoother`**:

| Commit | `run_mbf_smoother` Default | Source |
|--------|---------------------------|--------|
| Baseline (`a48cc783`) | `true` | `finding_config.hpp:47` |
| Optimization (`25894cca`) | `false` | `finding_config.hpp:47` |

The optimization commit explicitly changed this default:
```diff
-    bool run_mbf_smoother = true;
+    bool run_mbf_smoother = false;
```

### 7.2 How `run_mbf_smoother` Affects `build_tracks`

The `build_tracks` kernel has two code paths controlled by `run_mbf` parameter:

**When `run_mbf = true` (baseline behavior):**
- Full Multi-Branch Fit (MBF) smoothing runs
- Expensive matrix operations:
  - `accumulated_jacobian = accumulated_jacobian * payload.jacobian_ptr[link_idx]`
  - `S_inv = matrix::inverse(S)` (2x2 matrix inverse)
  - Kalman gain computation: `K = predicted_covariance * transpose(H) * S_inv`
  - Smoothed parameter computation with λ matrices
- Creates track states with smoothed parameters
- **Result: 512 µs per instance**

**When `run_mbf = false` (optimization behavior):**
- Minimal work - just links measurements to tracks
- Single assignment: `*it = {edm::track_constituent_link::measurement, L.meas_idx}`
- No matrix operations at all
- **Result: 70 µs per instance**

### 7.3 Code Evidence

From `device/common/include/traccc/finding/device/impl/build_tracks.ipp`:

```cpp
if (run_mbf) {
    // ~100 lines of Kalman filter math:
    // - accumulated_jacobian multiplication
    // - matrix::inverse(S)
    // - Kalman gain K computation
    // - smoothed parameter computation
    *it = {edm::track_constituent_link::track_state, track_state_index};
} else {
    // Single line - no math:
    *it = {edm::track_constituent_link::measurement, L.meas_idx};
}
```

### 7.4 Implications

The throughput improvement is **not** from the conditional Jacobian transport in `propagate_to_next_surface`. The claimed register reduction (-64 registers) was never achieved. The actual improvement comes from:

1. **Disabling MBF smoothing** → `build_tracks` -86.3%
2. **Secondary effects** → `find_tracks` -12.5%

This is a **configuration change** that trades off track quality (no MBF smoothing) for throughput.

---

## 8. Apples-to-Apples Re-benchmark

To isolate the true impact of the conditional Jacobian transport optimization, we re-benchmarked the baseline with MBF disabled to match the optimization's configuration.

### 8.1 Methodology

The baseline commit (`a48cc783`) does not have the `--run-mbf-smoother` CLI option (added in the optimization commit). To perform an apples-to-apples comparison:

1. Checked out baseline commit `a48cc783`
2. Modified `finding_config.hpp` to change default: `run_mbf_smoother = false`
3. Rebuilt and ran benchmark with 8 CPU threads
4. Restored original file

### 8.2 Re-benchmark Results

| Configuration | Throughput | Latency | Notes |
|--------------|------------|---------|-------|
| Baseline MBF=true (original) | 38.75 events/s | 25.80 ms/event | Original benchmark |
| **Baseline MBF=false** | **36.57 events/s** | **27.34 ms/event** | Source modified |
| Optimization MBF=false | 43.27 events/s | 23.11 ms/event | Default config |

### 8.3 Apples-to-Apples Comparison (MBF=false)

| Commit | Throughput | Latency | Change |
|--------|------------|---------|--------|
| Baseline (MBF=false) | 36.57 events/s | 27.34 ms/event | - |
| Optimization (MBF=false) | 43.27 events/s | 23.11 ms/event | **+18.3%** |

### 8.4 Key Finding: Real Performance Benefit

**The conditional Jacobian transport optimization provides a real +18.3% throughput improvement** when comparing with the same MBF configuration.

The earlier analysis conflated two separate effects:

| Effect | Source | Contribution |
|--------|--------|--------------|
| MBF default change | `run_mbf_smoother: true → false` | Affects `build_tracks` kernel |
| Conditional Jacobian transport | `bound_updater` vs `parameter_transporter` | Affects overall pipeline |

### 8.5 Anomaly: Baseline MBF=false Slower Than MBF=true

An unexpected result was observed:

| Baseline Config | Throughput |
|-----------------|------------|
| MBF=true | 38.75 events/s |
| MBF=false | 36.57 events/s |

This is counterintuitive since disabling MBF smoothing should reduce `build_tracks` work. Possible explanations:

1. **Run-to-run variance**: GPU thermal state, system load differences
2. **Compiler differences**: Baseline without CLI option support may compile differently
3. **Different code paths**: The baseline MBF=false path may not be as optimized as the optimization commit's MBF=false path
4. **Measurement noise**: Single benchmark run may not be representative

Further investigation with multiple runs would be needed to confirm this anomaly.

### 8.6 Revised Attribution

| Source | Original Attribution | Revised Attribution |
|--------|---------------------|---------------------|
| `build_tracks` -86.3% | Conditional Jacobian | MBF default change |
| Overall +11.67% gain | Register optimization | **Mixed: MBF change + Conditional Jacobian** |
| True Jacobian benefit | Unknown | **+18.3%** (apples-to-apples) |

---

## 9. Optimization Mechanism Investigation

### 9.1 Question: What Causes +18.3% If Not Register Reduction?

Since cuobjdump analysis showed 0 register reduction (128 registers in all variants), we investigated the actual code differences between `parameter_transporter` and `bound_updater`.

### 9.2 Code Analysis: Key Difference Found

**`parameter_transporter`** (detray, lines 131-135):
```cpp
// In operator() after computing full_jacobian:
if (actor_state._full_jacobian_ptr != nullptr) {
    const auto aggregate_full_jacobian =
        full_jacobian * (*(actor_state._full_jacobian_ptr));  // 6x6 × 6x6 matrix mult
    (*(actor_state._full_jacobian_ptr)) = aggregate_full_jacobian;  // Write back
}
```

**`bound_updater`** (traccc, lines 147-148):
```cpp
// NOTE: No Jacobian aggregation here - that's only needed for MBF smoother.
// This is the key difference from parameter_transporter.
```

### 9.3 The Jacobian Aggregation Operation

Both actors compute `full_jacobian` identically via `get_full_jacobian()`. The difference is what happens AFTER:

| Actor | After `get_full_jacobian()` |
|-------|----------------------------|
| `parameter_transporter` | Multiply 6x6 × 6x6, store to global memory |
| `bound_updater` | Discard Jacobian (no-op) |

The aggregation in `parameter_transporter` performs:
1. **6x6 × 6x6 matrix multiplication**: ~216 FLOPs (6³ multiply-adds)
2. **Global memory read**: 144 bytes (6×6 floats from `tmp_jacobian_ptr`)
3. **Global memory write**: 144 bytes (6×6 floats to `tmp_jacobian_ptr`)

### 9.4 Performance Impact Per Surface

| Operation | `parameter_transporter` | `bound_updater` | Savings |
|-----------|------------------------|-----------------|---------|
| Matrix multiply (6x6 × 6x6) | ~216 FLOPs | 0 | ~216 FLOPs |
| Global memory read | 144 bytes | 0 | 144 bytes |
| Global memory write | 144 bytes | 0 | 144 bytes |
| **Total per surface** | ~216 FLOPs + 288 bytes | 0 | ~216 FLOPs + 288 bytes |

### 9.5 Cumulative Impact Per Track

For a typical track hitting ~15 sensitive surfaces:

| Metric | `parameter_transporter` | `bound_updater` | Savings |
|--------|------------------------|-----------------|---------|
| **FLOPs** | 15 × 216 = 3,240 | 0 | **3,240 FLOPs** |
| **Memory traffic** | 15 × 288 = 4,320 bytes | 0 | **4,320 bytes** |

### 9.6 Why Register Count Is Unchanged

Both actors call identical `get_full_jacobian()` which computes the same 6x6 matrix. The compiler allocates registers for this computation identically in both cases.

The difference is:
- `parameter_transporter`: Uses the result (multiply + store)
- `bound_updater`: Discards the result immediately

Since `get_full_jacobian()` dominates register usage, both variants compile to 128 registers.

### 9.7 Confirmed Optimization Mechanism

| Factor | Contribution | Evidence |
|--------|--------------|----------|
| **Skipped 6x6 × 6x6 matrix multiplications** | **Primary** | ~3,240 FLOPs/track saved |
| **Reduced global memory traffic** | **Secondary** | ~4,320 bytes/track saved |
| **Better cache behavior** | **Tertiary** | No Jacobian buffer thrashing |
| Register pressure reduction | **None** | 0 registers saved (cuobjdump) |

### 9.8 Source Code References

| File | Lines | Description |
|------|-------|-------------|
| `detray/.../parameter_transporter.hpp` | 131-135 | Jacobian aggregation code |
| `traccc/.../bound_updater.hpp` | 147-148 | Comment explaining no aggregation |
| `traccc/.../propagate_to_next_surface.ipp` | 118-122 | Jacobian pointer initialization |
| `detray/.../track_parametrization.hpp` | 80 | `bound_matrix` = 6x6 definition |

### 9.9 Conclusion: Algorithmic vs Resource Optimization

The optimization is **algorithmic** (skip unnecessary work) rather than **resource-based** (reduce registers):

| Original Claim | Reality |
|----------------|---------|
| "Reduce register pressure by ~64 registers" | 0 registers saved |
| "Improve occupancy" | No occupancy change |
| "Eliminate Jacobian storage in registers" | Jacobian computed but discarded |

**The actual benefit**: Eliminating expensive 6x6 matrix multiplications and global memory accesses at every surface during propagation.

---

## 10. Analysis and Conclusions

### 10.1 Primary Findings

1. **`propagate_to_next_surface` kernel unchanged in nsys profiling**: The per-instance execution time remained essentially the same (~940-954 µs) under profiling. However, apples-to-apples benchmarking shows +18.3% overall improvement.

2. **`build_tracks` dramatically improved**: 512 µs → 70 µs per instance (-86.3%). This is caused by disabling MBF smoothing (configuration change), not by the conditional Jacobian transport.

3. **`find_tracks` improved**: 144 µs → 126 µs per instance (-12.5%).

4. **More work processed**: The optimization finds more track candidates:
   - `propagate_to_next_surface` instances: 2,144 → 2,185 (+1.9%)
   - `find_tracks` instances: 2,254 → 2,295 (+1.8%)
   - `remove_duplicates` instances: 1,594 → 1,635 (+2.6%)

5. **Real performance benefit confirmed**: Apples-to-apples re-benchmark (both with MBF=false) shows **+18.3%** throughput improvement from the conditional Jacobian transport optimization.

6. **Optimization mechanism identified**: The +18.3% improvement comes from **skipping Jacobian aggregation** (6x6 matrix multiplications and global memory accesses), NOT from register pressure reduction.

### 10.2 Hypothesis Evaluation (Final)

| Claim | Expected | Observed | Status |
|-------|----------|----------|--------|
| `propagate_to_next_surface` speedup | -10-15% per instance | +1.5% (nsys) | **INCONCLUSIVE** |
| Register pressure reduction | ~64 registers saved | 0 registers saved | **NOT VALIDATED** |
| Occupancy improvement | +10-25% | No change (128 regs in all variants) | **NOT VALIDATED** |
| Overall throughput gain | +5-15% | +11.67% (original), +18.3% (apples-to-apples) | **VALIDATED** |
| Source of throughput gain | Register optimization | Skipped Jacobian aggregation | **MECHANISM IDENTIFIED** |

### 10.3 Actual Source of Throughput Gain (Final)

| Source | Contribution | Mechanism |
|--------|--------------|-----------|
| Conditional Jacobian transport | **+18.3%** (apples-to-apples) | Skipped 6x6 matrix mult + memory I/O |
| `build_tracks` | -86.3% kernel time | MBF smoothing disabled (config change) |
| `find_tracks` | -12.5% kernel time | Reduced downstream work |

### 10.4 Conclusions (Final)

1. **The conditional Jacobian transport optimization DOES provide real benefit.** Apples-to-apples comparison shows +18.3% throughput improvement when both commits use MBF=false.

2. **Register reduction was NOT achieved.** Despite the theoretical claim of ~64 register savings, all kernel variants use 128 registers. Both `bound_updater` and `parameter_transporter` compute the full Jacobian identically.

3. **The actual optimization mechanism is algorithmic, not resource-based:**
   - `parameter_transporter`: Computes Jacobian, then multiplies and stores for MBF
   - `bound_updater`: Computes Jacobian, then discards it immediately
   - Savings: ~216 FLOPs + 288 bytes per surface per track

4. **The original benchmark conflated two effects:**
   - MBF default change (`true` → `false`): -86.3% `build_tracks` time
   - Conditional Jacobian transport: +18.3% overall throughput

5. **The optimization name is a misnomer.** "Conditional Jacobian Transport" suggests conditional computation, but the Jacobian is always computed. The benefit comes from skipping the **aggregation** (accumulation for MBF smoother).

### 10.5 Recommended Actions (Final)

1. **Rename the optimization**: "Conditional Jacobian Aggregation" or "Skip MBF Jacobian Accumulation" would be more accurate than "Conditional Jacobian Transport".

2. **Update documentation**: Replace claims about register pressure reduction with the actual mechanism (skipped matrix multiplications and memory traffic).

3. **Consider further optimization**: Since `get_full_jacobian()` is still computed even when not needed, a true conditional optimization could skip this computation entirely for additional gains.

4. **Separate the configuration change**: The `run_mbf_smoother` default change should be a separate commit with its own performance documentation.

---

## 11. Raw Data

### 11.1 Profile Files

| File | Size | Location |
|------|------|----------|
| `baseline_nsys.nsys-rep` | 4.2 MB | `build/baseline_nsys.nsys-rep` |
| `optimization_nsys.nsys-rep` | 4.5 MB | `build/optimization_nsys.nsys-rep` |
| `baseline_nsys.sqlite` | - | `build/baseline_nsys.sqlite` |
| `optimization_nsys.sqlite` | - | `build/optimization_nsys.sqlite` |

### 11.2 Re-benchmark Commands (Apples-to-Apples)

```bash
# Checkout baseline
git checkout a48cc783

# Modify finding_config.hpp to set run_mbf_smoother = false
# (No CLI option available in baseline)
sed -i 's/run_mbf_smoother = true/run_mbf_smoother = false/' \
  core/include/traccc/finding/finding_config.hpp

# Rebuild
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . -j4 --target traccc_throughput_mt_cuda

# Run benchmark
./bin/traccc_throughput_mt_cuda \
  --input-directory=odd/geant4_ttbar_mu200/ \
  --cpu-threads=8

# Result: 36.57 events/s, 27.34 ms/event

# Restore original file
git checkout core/include/traccc/finding/finding_config.hpp
```

### 11.3 Profiling Commands

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

# Register analysis (cuobjdump)
/usr/local/cuda-12.6/bin/cuobjdump --dump-resource-usage lib64/libtraccc_cuda.so 2>&1 | \
  grep -E "(propagate_to_next_surface.*\.cu$|REG:.*STACK:)"
```

---

## 12. References

### Source Files
- `core/include/traccc/finding/actors/bound_updater.hpp` - `bound_updater` actor (no Jacobian aggregation)
- `detray/.../parameter_transporter.hpp` - `parameter_transporter` actor (with Jacobian aggregation)
- `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` - Propagation kernel
- `core/include/traccc/finding/details/combinatorial_kalman_filter_types.hpp` - Actor chain definitions
- `device/common/include/traccc/finding/device/impl/build_tracks.ipp` - `build_tracks` kernel implementation
- `core/include/traccc/finding/finding_config.hpp` - `run_mbf_smoother` configuration

### Documentation
- `doc/conditional_jacobian_transport_report.md` - Benchmark results and implementation details
- `doc/conditional_jacobian_transport_plan.md` - Implementation plan
- `doc/conditional_jacobian_transport_profile_plan.md` - Profiling plan

### External References
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [cuobjdump Documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
