# Profiling Plan: Fitting-Enabled Pipeline Comparison

## Overview

Compare GPU performance between:
- **Baseline**: commit `e49d353a` (Merge pull request #1220 feat/mbf)
- **Current**: commit `f6486a32` (survey-fpga branch)

Both with Kalman fitting **enabled** in the full chain algorithm.

---

## Issue Identified

The current `full_chain_algorithm.cpp` **does not invoke fitting**:
- `m_fitting` is declared and initialized (lines 79-80, 142-144)
- But `operator()` never calls it (lines 163-212)
- Pipeline stops after CKF (track finding) and returns `track_candidates.tracks`

This explains why Section 5.2 GPU breakdown shows no fitting kernels.

---

## Code Changes Required

### File: `examples/run/cuda/full_chain_algorithm.cpp`

**Location**: After line 188 (track finding), before line 191 (copy results)

**Current code (lines 186-197)**:
```cpp
// Run the track finding (asynchronously).
const finding_algorithm::output_type track_candidates =
    m_finding(m_device_detector, m_field, measurements, track_params);

// Copy a limited amount of result data back to the host.
const auto host_tracks =
    m_copy.to(track_candidates.tracks, m_cached_pinned_host_mr, nullptr,
              vecmem::copy::type::device_to_host);
```

**Modified code** (based on alpaka reference `examples/run/alpaka/full_chain_algorithm.cpp:180-186`):
```cpp
// Run the track finding (asynchronously).
const finding_algorithm::output_type track_candidates =
    m_finding(m_device_detector, m_field, measurements, track_params);

// Run the track fitting (asynchronously).
const fitting_algorithm::output_type track_states =
    m_fitting(m_device_detector, m_field, track_candidates);

// Copy a limited amount of result data back to the host.
const auto host_tracks =
    m_copy.to(track_states.tracks, m_cached_pinned_host_mr, nullptr,
              vecmem::copy::type::device_to_host);
```

**Key differences from original attempt**:
- Pass `track_candidates` directly (NOT `.tracks`) to fitting
- Copy from `track_states.tracks` (the fitted output)

---

## Build Commands

### For Each Commit (use separate build directories)

```bash
# For baseline (e49d353a)
git checkout e49d353a
mkdir -p build_baseline && cd build_baseline

# Configure with CUDA and profiling support
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRACCC_BUILD_CUDA=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -GNinja

# Build throughput benchmark
ninja traccc_throughput_mt_cuda

# For current (f6486a32)
git checkout survey-fpga
mkdir -p build_current && cd build_current

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRACCC_BUILD_CUDA=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -GNinja

ninja traccc_throughput_mt_cuda
```

**Important**: Use separate build directories to avoid cmake cache issues when switching commits.

---

## Profiling Commands

**Note**: Run from repository root. Data paths are relative to working directory.

### 1. Profile Baseline (e49d353a)

```bash
# Ensure you're in repo root
cd /dicos_ui_home/noah/traccc

# Checkout baseline and apply patch
git checkout e49d353a
# Edit examples/run/cuda/full_chain_algorithm.cpp as described above

# Build (if not already done)
cd build_baseline && ninja traccc_throughput_mt_cuda && cd ..

# Profile with nsys
nsys profile \
    --output=baseline_fitting_e49d353a \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --stats=true \
    ./build_baseline/bin/traccc_throughput_mt_cuda \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --material-file=geometries/odd/odd-detray_material_detray.json \
    --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=100 \
    --cpu-threads=1 \
    --cold-run

# Export to SQLite for analysis
nsys export -t sqlite baseline_fitting_e49d353a.nsys-rep
```

### 2. Profile Current Commit (f6486a32)

```bash
# Ensure you're in repo root
cd /dicos_ui_home/noah/traccc

# Checkout current and apply patch
git checkout survey-fpga
# Edit examples/run/cuda/full_chain_algorithm.cpp as described above

# Build (if not already done)
cd build_current && ninja traccc_throughput_mt_cuda && cd ..

# Profile with nsys
nsys profile \
    --output=current_fitting_f6486a32 \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --stats=true \
    ./build_current/bin/traccc_throughput_mt_cuda \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --material-file=geometries/odd/odd-detray_material_detray.json \
    --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=100 \
    --cpu-threads=1 \
    --cold-run

# Export to SQLite for analysis
nsys export -t sqlite current_fitting_f6486a32.nsys-rep
```

---

## Metrics to Compare

### 1. Kernel-Level Metrics

| Metric | Description | SQL Query |
|--------|-------------|-----------|
| Kernel time % | Time per kernel as % of total | `SELECT name, SUM(duration)/1e6 as ms FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY name ORDER BY ms DESC` |
| Fitting kernel time | Time in kalman_fitting kernels | Filter for `*kalman*fit*` patterns |
| Finding kernel time | Time in CKF kernels | Filter for `*find*` or `*ckf*` patterns |
| Propagation time | Time in RK4/propagation | Filter for `*propagate*` patterns |

### 2. Pipeline Metrics

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| Total GPU time (ms) | | | |
| Fitting time (ms) | | | |
| Finding time (ms) | | | |
| Propagation time (ms) | | | |
| Memory transfers (MB) | | | |
| Peak device memory (MB) | | | |

### 3. Throughput Metrics

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| Events/second | | | |
| Tracks/second | | | |
| Throughput (MHz×events) | | | |

---

## Analysis Script

```python
#!/usr/bin/env python3
"""Compare profiling results between baseline and current."""

import sqlite3
import pandas as pd

def load_kernel_times(db_path):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        SUBSTR(name, 1, 80) as kernel_name,
        COUNT(*) as calls,
        SUM(duration)/1e6 as total_ms,
        AVG(duration)/1e6 as avg_ms
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    GROUP BY name
    ORDER BY total_ms DESC
    """
    return pd.read_sql(query, conn)

def compare_fitting_time(baseline_db, current_db):
    baseline = load_kernel_times(baseline_db)
    current = load_kernel_times(current_db)

    # Filter fitting-related kernels
    fit_patterns = ['fit', 'kalman']

    baseline_fit = baseline[baseline['kernel_name'].str.lower().str.contains('|'.join(fit_patterns))]
    current_fit = current[current['kernel_name'].str.lower().str.contains('|'.join(fit_patterns))]

    print("Baseline fitting kernels:")
    print(baseline_fit)
    print(f"\nBaseline fitting total: {baseline_fit['total_ms'].sum():.2f} ms")

    print("\nCurrent fitting kernels:")
    print(current_fit)
    print(f"\nCurrent fitting total: {current_fit['total_ms'].sum():.2f} ms")

if __name__ == "__main__":
    compare_fitting_time(
        "baseline_fitting_e49d353a.sqlite",
        "current_fitting_f6486a32.sqlite"
    )
```

---

## Expected Outcomes

### New Kernels to Appear

With fitting enabled, expect these additional kernel patterns:
- `kalman_fitting*` - main fitting kernel
- `fit_tracks*` - track fitting iteration
- Additional `propagate*` calls from fitting

### Performance Expectations

Based on Section 5.2 analysis (CKF-only pipeline):
- Current propagation: ~63% of GPU time
- Current finding: ~10% of GPU time

With fitting added:
- Additional propagation for re-fitting
- Additional covariance updates
- Estimated +5-15% total GPU time

---

## Notes

1. **Single-threaded profiling**: Use `--cpu-threads=1` for clean kernel timing
2. **Cold run**: Include `--cold-run` to exclude JIT/warmup from timing
3. **Same dataset**: Use identical input for fair comparison
4. **Same GPU**: Profile on same hardware
5. **No other load**: Ensure GPU is idle except for benchmark

---

## Checklist

- [ ] Checkout baseline (e49d353a)
- [ ] Apply fitting patch to `examples/run/cuda/full_chain_algorithm.cpp`
- [ ] Create and configure `build_baseline/` directory
- [ ] Build baseline with fitting enabled
- [ ] Profile baseline with nsys
- [ ] Export baseline to SQLite
- [ ] Checkout current (survey-fpga / f6486a32)
- [ ] Apply fitting patch to `examples/run/cuda/full_chain_algorithm.cpp`
- [ ] Create and configure `build_current/` directory
- [ ] Build current with fitting enabled
- [ ] Profile current with nsys
- [ ] Export current to SQLite
- [ ] Run comparison script
- [ ] Document findings in survey-fpga.md
