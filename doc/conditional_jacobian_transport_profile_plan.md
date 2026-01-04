# Conditional Jacobian Transport Profiling Plan

**Date:** 2026-01-04
**Purpose:** Validate optimization claims with nsys and ncu profiling
**Related:** `doc/conditional_jacobian_transport_report.md`

---

## 1. Overview

The conditional Jacobian transport optimization achieved **+11.67% throughput improvement** in benchmarks. This profiling plan aims to validate the theoretical claims about register pressure reduction and occupancy improvement.

### Current Benchmark Results

| Commit | Description | Throughput | Latency |
|--------|-------------|------------|---------|
| `a48cc783` | Baseline | 38.75 events/s | 25.80 ms/event |
| `25894cca` | Optimization | 43.27 events/s | 23.11 ms/event |

### Theoretical Claims to Validate

| Metric | Expected Baseline | Expected Optimization | Expected Change |
|--------|-------------------|----------------------|-----------------|
| Registers/thread | ~150-180 | ~86-116 | -64 registers |
| V100 Occupancy | 16-25% | 25-50% | +10-25% |
| Jacobian storage | 64 registers (8x8 floats) | 0 registers | -64 registers |

---

## 2. Test Environment

- **GPU:** Tesla V100-SXM2-32GB (compute capability 7.0)
- **Dataset:** `odd/geant4_ttbar_mu200/` (36 events)
- **Profiling threads:** 1 CPU thread (to isolate GPU behavior)
- **Tools:**
  - nsys (Nsight Systems) for timeline analysis
  - ncu (Nsight Compute) for kernel-level metrics

---

## 3. nsys (Nsight Systems) Profiling

### 3.1 Purpose

System-level analysis of GPU utilization, kernel timeline, and CPU-GPU interactions.

### 3.2 Key Metrics to Capture

| Category | Metrics | Purpose |
|----------|---------|---------|
| **Kernel Timeline** | Kernel start/end times | Identify kernel duration changes |
| **GPU Utilization** | Total GPU active time | Measure overall GPU efficiency |
| **Memory Transfers** | H2D, D2H copy times | Detect memory bottlenecks |
| **API Overhead** | CUDA API call times | Measure launch overhead |
| **Synchronization** | cudaStreamSynchronize times | Identify sync bottlenecks |

### 3.3 Target Kernels

The CKF pipeline includes multiple kernels. Focus on:

1. **`propagate_to_next_surface`** - Primary optimization target
2. `make_barcode_sequence` - Track building
3. `find_tracks` - Measurement finding
4. `build_tracks` - Track candidate construction
5. `apply_interaction` - Material interaction

### 3.4 Commands

**Baseline (a48cc783):**
```bash
git checkout a48cc783
# Rebuild with sm_70
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . -j4

# Profile with nsys
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=baseline_nsys \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=10 \
    --cpu-threads=1
```

**Optimization (25894cca):**
```bash
git checkout 25894cca
# Rebuild with sm_70
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . -j4

# Profile with nsys
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=optimization_nsys \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=10 \
    --cpu-threads=1
```

### 3.5 Analysis Commands

```bash
# Generate summary report
nsys stats baseline_nsys.nsys-rep
nsys stats optimization_nsys.nsys-rep

# Export to SQLite for detailed analysis
nsys export --type=sqlite baseline_nsys.nsys-rep
nsys export --type=sqlite optimization_nsys.nsys-rep
```

### 3.6 Expected Results

| Metric | Baseline | Optimization | Expected Change |
|--------|----------|--------------|-----------------|
| `propagate_to_next_surface` duration | X ms | Y ms | -10-15% |
| Total GPU time per event | ~25 ms | ~23 ms | -10% |
| Other kernel durations | Similar | Similar | No regression |

---

## 4. ncu (Nsight Compute) Profiling

### 4.1 Purpose

Detailed kernel-level metrics to validate register pressure and occupancy claims.

### 4.2 Key Metrics to Capture

#### Register Analysis
| Metric | Description | Expected Change |
|--------|-------------|-----------------|
| `launch__registers_per_thread` | Registers used per thread | -64 registers |
| `sm__sass_data_bytes_mem_local` | Local memory spill | Should remain 0 |

#### Occupancy Analysis
| Metric | Description | Expected Change |
|--------|-------------|-----------------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy | +10-25% |
| `sm__maximum_warps_per_active_cycle_pct` | Theoretical occupancy | +10-25% |
| Occupancy limiter | What limits occupancy | Registers → Other |

#### Memory Efficiency
| Metric | Description | Expected Change |
|--------|-------------|-----------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM utilization | Similar |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate.pct` | L1 hit rate | Similar |
| `lts__t_sectors_srcunit_tex_op_read_hit_rate.pct` | L2 hit rate | Similar |

#### Compute Efficiency
| Metric | Description | Expected Change |
|--------|-------------|-----------------|
| `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` | SM activity | Higher |
| `smsp__warps_issue_stalled_wait_any.avg.pct_of_peak_sustained_active` | Stall rate | Lower |
| `smsp__inst_executed.avg.per_cycle_active` | IPC | Higher |

### 4.3 Target Kernels

**Baseline kernel:**
```
propagate_to_next_surface<
  ckf_propagator_t<odd_detector::device, covfie::field<inhom_bfield>::view_t>,
  covfie::field<inhom_bfield>::view_t
>
```

**Optimization kernel:**
```
propagate_to_next_surface<
  ckf_propagator_no_mbf_t<odd_detector::device, covfie::field<inhom_bfield>::view_t>,
  covfie::field<inhom_bfield>::view_t
>
```

### 4.4 Commands

**Baseline (a48cc783):**
```bash
git checkout a48cc783
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . -j4

# Full metrics collection
ncu \
  --set full \
  --target-processes all \
  --output baseline_ncu \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=1 \
    --cpu-threads=1

# Or specific metrics only
ncu \
  --metrics \
    launch__registers_per_thread,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__maximum_warps_per_active_cycle_pct,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__cycles_active.avg.pct_of_peak_sustained_elapsed \
  --target-processes all \
  --output baseline_ncu_metrics \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=1 \
    --cpu-threads=1
```

**Optimization (25894cca):**
```bash
git checkout 25894cca
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . -j4

# Full metrics collection
ncu \
  --set full \
  --target-processes all \
  --output optimization_ncu \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=1 \
    --cpu-threads=1
```

### 4.5 Analysis Commands

```bash
# View summary in terminal
ncu --import baseline_ncu.ncu-rep --page raw
ncu --import optimization_ncu.ncu-rep --page raw

# Compare two profiles
ncu --import baseline_ncu.ncu-rep --import optimization_ncu.ncu-rep --page diff
```

### 4.6 Expected Results

| Metric | Baseline | Optimization | Validation |
|--------|----------|--------------|------------|
| Registers/thread | ~150-180 | ~86-116 | Confirm -64 |
| Achieved occupancy | 16-25% | 25-50% | Confirm improvement |
| Occupancy limiter | Registers | Block size or other | Confirm shift |
| Local memory spill | 0 | 0 | No regression |
| DRAM throughput | X GB/s | ~X GB/s | No regression |
| L1 hit rate | Y% | ~Y% | No regression |

---

## 5. Validation Checklist

### 5.1 Primary Claims

| # | Claim | Tool | Metric | Pass Criteria |
|---|-------|------|--------|---------------|
| 1 | ~64 register reduction | ncu | `launch__registers_per_thread` | Baseline - Opt ≈ 64 |
| 2 | Occupancy improvement | ncu | `sm__warps_active.avg.pct` | Opt > Baseline |
| 3 | `propagate_to_next_surface` speedup | nsys | Kernel duration | Opt < Baseline |
| 4 | No memory regression | ncu | DRAM throughput | Similar |
| 5 | No new bottlenecks | nsys | Other kernel times | Similar |

### 5.2 Secondary Checks

| # | Check | Tool | Purpose |
|---|-------|------|---------|
| 1 | No local memory spill | ncu | Ensure registers fit |
| 2 | IPC improvement | ncu | Validate compute efficiency |
| 3 | Warp stall reduction | ncu | Validate latency hiding |
| 4 | Memory transfer unchanged | nsys | No host-device overhead |

---

## 6. Reporting Template

After profiling, document results in `doc/conditional_jacobian_transport_profile_results.md`:

```markdown
# Profiling Results

## Register Analysis
| Kernel | Baseline Registers | Optimization Registers | Difference |
|--------|-------------------|------------------------|------------|
| propagate_to_next_surface | X | Y | Z |

## Occupancy Analysis
| Kernel | Baseline Occupancy | Optimization Occupancy | Improvement |
|--------|-------------------|------------------------|-------------|
| propagate_to_next_surface | X% | Y% | +Z% |

## Kernel Duration (nsys)
| Kernel | Baseline (ms) | Optimization (ms) | Speedup |
|--------|---------------|-------------------|---------|
| propagate_to_next_surface | X | Y | Z% |

## Memory Efficiency
| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| DRAM throughput | X GB/s | Y GB/s | Z% |
| L1 hit rate | X% | Y% | Z% |
| L2 hit rate | X% | Y% | Z% |

## Conclusion
[Summary of findings and whether theoretical claims are validated]
```

---

## 7. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| ncu permission denied | Run with `sudo` or add user to `nvpd` group |
| Kernel not found in profile | Use `--target-processes all` flag |
| Too many kernels profiled | Use `--kernel-name` filter |
| Profile file too large | Reduce `--input-events` |

### Kernel Name Filter

To profile only `propagate_to_next_surface`:
```bash
ncu \
  --kernel-name "propagate_to_next_surface" \
  --launch-skip 0 \
  --launch-count 10 \
  ...
```

---

## 8. References

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- `doc/conditional_jacobian_transport_report.md` - Benchmark results
- `doc/conditional_jacobian_transport_plan.md` - Implementation details
