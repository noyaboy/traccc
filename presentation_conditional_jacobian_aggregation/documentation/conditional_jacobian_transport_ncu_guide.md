# Conditional Jacobian Aggregation: ncu Profiling Guide

**Date:** 2026-01-04
**Purpose:** Step-by-step guide for ncu (Nsight Compute) profiling on a machine with root privileges
**Related:** `doc/conditional_jacobian_transport_profile_report.md`

---

## Overview

This guide enables ncu profiling to validate occupancy and memory metrics for the conditional Jacobian aggregation optimization. ncu requires admin privileges (blocked on original machine due to `RmProfilingAdminOnly=1`).

### What We Already Know (from nsys + cuobjdump)

| Metric | Result |
|--------|--------|
| Throughput improvement | +18.3% (apples-to-apples) |
| Register count | 128 in ALL variants (no reduction) |
| Mechanism | Skipped Jacobian aggregation (6x6 matrix mult) |

### What ncu Will Validate

| Metric | Purpose |
|--------|---------|
| Achieved occupancy | Confirm no occupancy change |
| L1/L2 hit rates | Quantify cache behavior |
| DRAM throughput | Measure memory savings |
| IPC / SM utilization | Validate compute savings |
| Warp stall analysis | Identify remaining bottlenecks |

---

## Prerequisites

1. **Root/sudo access** on the profiling machine
2. **CUDA toolkit** with ncu installed (CUDA 11.0+)
3. **Git access** to traccc repository
4. **GPU:** NVIDIA GPU with compute capability 7.0+ (V100, A100, etc.)
5. **Dataset:** `odd/geant4_ttbar_mu200/` available

---

## Step 1: Clone and Setup

```bash
# Clone repository (if not already cloned)
git clone <traccc-repo-url> traccc
cd traccc

# Fetch latest changes
git fetch origin

# Verify commits exist
git log --oneline a48cc783 -1  # Baseline
git log --oneline 25894cca -1  # Optimization
```

---

## Step 2: Profile Baseline Commit (a48cc783)

### 2.1 Checkout and Build Baseline

```bash
# Checkout baseline commit
git checkout a48cc783

# Create build directory
mkdir -p build && cd build

# Configure with your GPU architecture (adjust sm_XX as needed)
# V100 = sm_70, A100 = sm_80, H100 = sm_90
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DTRACCC_BUILD_CUDA=ON \
  -DTRACCC_BUILD_EXAMPLES=ON

# Build the throughput benchmark
cmake --build . -j$(nproc) --target traccc_throughput_mt_cuda

# Verify build succeeded
ls -la bin/traccc_throughput_mt_cuda
```

### 2.2 Run ncu Profiling on Baseline

```bash
# IMPORTANT: Run with sudo for ncu access
# Profile propagate_to_next_surface kernel (primary target)

sudo ncu \
  --set full \
  --kernel-name "propagate_to_next_surface" \
  --launch-skip 0 \
  --launch-count 10 \
  --target-processes all \
  -o baseline_ncu \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=../odd/geant4_ttbar_mu200/ \
    --input-events=1 \
    --cpu-threads=1

# Alternative: Specific metrics only (faster)
sudo ncu \
  --metrics \
    launch__registers_per_thread,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__maximum_warps_per_active_cycle_pct,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate.pct,\
    lts__t_sectors_srcunit_tex_op_read_hit_rate.pct,\
    smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
    smsp__inst_executed.avg.per_cycle_active \
  --kernel-name "propagate_to_next_surface" \
  --launch-skip 0 \
  --launch-count 10 \
  --target-processes all \
  -o baseline_ncu_metrics \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=../odd/geant4_ttbar_mu200/ \
    --input-events=1 \
    --cpu-threads=1
```

### 2.3 Export Baseline Results

```bash
# View summary in terminal
ncu --import baseline_ncu.ncu-rep --page raw > baseline_ncu_summary.txt

# Or view specific sections
ncu --import baseline_ncu.ncu-rep --page details > baseline_ncu_details.txt
```

---

## Step 3: Profile Optimization Commit (25894cca)

### 3.1 Checkout and Build Optimization

```bash
# Return to repo root
cd ..

# Checkout optimization commit
git checkout 25894cca

# Rebuild
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DTRACCC_BUILD_CUDA=ON \
  -DTRACCC_BUILD_EXAMPLES=ON

cmake --build . -j$(nproc) --target traccc_throughput_mt_cuda

# Verify build succeeded
ls -la bin/traccc_throughput_mt_cuda
```

### 3.2 Run ncu Profiling on Optimization

```bash
# Profile with full metrics
sudo ncu \
  --set full \
  --kernel-name "propagate_to_next_surface" \
  --launch-skip 0 \
  --launch-count 10 \
  --target-processes all \
  -o optimization_ncu \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=../odd/geant4_ttbar_mu200/ \
    --input-events=1 \
    --cpu-threads=1

# Alternative: Specific metrics only (faster)
sudo ncu \
  --metrics \
    launch__registers_per_thread,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__maximum_warps_per_active_cycle_pct,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate.pct,\
    lts__t_sectors_srcunit_tex_op_read_hit_rate.pct,\
    smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
    smsp__inst_executed.avg.per_cycle_active \
  --kernel-name "propagate_to_next_surface" \
  --launch-skip 0 \
  --launch-count 10 \
  --target-processes all \
  -o optimization_ncu_metrics \
  ./bin/traccc_throughput_mt_cuda \
    --input-directory=../odd/geant4_ttbar_mu200/ \
    --input-events=1 \
    --cpu-threads=1
```

### 3.3 Export Optimization Results

```bash
# View summary in terminal
ncu --import optimization_ncu.ncu-rep --page raw > optimization_ncu_summary.txt

# Or view specific sections
ncu --import optimization_ncu.ncu-rep --page details > optimization_ncu_details.txt
```

---

## Step 4: Compare Results

```bash
# Side-by-side comparison
ncu --import baseline_ncu.ncu-rep --import optimization_ncu.ncu-rep --page diff > ncu_comparison.txt

# Or view in GUI (if available)
ncu-ui baseline_ncu.ncu-rep optimization_ncu.ncu-rep
```

---

## Step 5: Record Results

Create the file `doc/conditional_jacobian_transport_ncu_results.md` with your findings.

### Template for Results Document

```markdown
# Conditional Jacobian Aggregation: ncu Profiling Results

**Date:** YYYY-MM-DD
**Machine:** [GPU model, CUDA version, driver version]
**Profiler:** ncu [version]

---

## 1. Test Environment

| Item | Value |
|------|-------|
| GPU | [e.g., Tesla V100-SXM2-32GB] |
| Compute Capability | [e.g., 7.0] |
| CUDA Version | [e.g., 12.6] |
| Driver Version | [e.g., 560.xx] |
| ncu Version | [e.g., 2024.3.0] |

---

## 2. Register Analysis

| Kernel | Baseline Registers | Optimization Registers | Difference |
|--------|-------------------|------------------------|------------|
| propagate_to_next_surface (mbf_on) | X | X | 0 |
| propagate_to_next_surface (mbf_off) | N/A | X | N/A |

---

## 3. Occupancy Analysis

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| Achieved Occupancy (%) | X | Y | +/-Z% |
| Theoretical Occupancy (%) | X | Y | +/-Z% |
| Occupancy Limiter | [registers/shared mem/blocks] | [registers/shared mem/blocks] | |

---

## 4. Memory Analysis

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| DRAM Throughput (%) | X | Y | +/-Z% |
| L1 Hit Rate (%) | X | Y | +/-Z% |
| L2 Hit Rate (%) | X | Y | +/-Z% |
| Global Memory Load Throughput | X GB/s | Y GB/s | +/-Z% |
| Global Memory Store Throughput | X GB/s | Y GB/s | +/-Z% |

---

## 5. Compute Analysis

| Metric | Baseline | Optimization | Change |
|--------|----------|--------------|--------|
| SM Utilization (%) | X | Y | +/-Z% |
| IPC (Instructions per Cycle) | X | Y | +/-Z% |
| Achieved FLOPs | X GFLOPS | Y GFLOPS | +/-Z% |

---

## 6. Warp Stall Analysis

| Stall Reason | Baseline (%) | Optimization (%) | Change |
|--------------|--------------|------------------|--------|
| Memory Dependency | X | Y | +/-Z% |
| Execution Dependency | X | Y | +/-Z% |
| Synchronization | X | Y | +/-Z% |
| Other | X | Y | +/-Z% |

---

## 7. Conclusions

### 7.1 Occupancy Validation

[Did occupancy change? If not, why? (e.g., still register-limited at 128)]

### 7.2 Memory Savings Validation

[Does DRAM throughput show reduced memory traffic as expected from skipped aggregation?]

### 7.3 Compute Savings Validation

[Does IPC/FLOPs show reduced computation as expected from skipped 6x6 matrix mult?]

### 7.4 Remaining Bottlenecks

[What is the current bottleneck? Memory-bound or compute-bound?]

---

## 8. Raw Data Files

| File | Description |
|------|-------------|
| `baseline_ncu.ncu-rep` | Full baseline profile |
| `optimization_ncu.ncu-rep` | Full optimization profile |
| `baseline_ncu_summary.txt` | Baseline text summary |
| `optimization_ncu_summary.txt` | Optimization text summary |
| `ncu_comparison.txt` | Side-by-side comparison |

---

## 9. Notes

[Any observations, issues encountered, or additional context]
```

---

## Step 6: Push Results

```bash
# Return to repo root
cd ..

# Add results document
git add doc/conditional_jacobian_transport_ncu_results.md

# Optionally add profile files (may be large)
# git add build/*.ncu-rep

# Commit
git commit -m "docs: Add ncu profiling results for conditional Jacobian aggregation"

# Push to remote
git push origin feature/register-presure
```

---

## Troubleshooting

### Permission Denied

```bash
# Verify running as root
sudo whoami  # Should print "root"

# Check driver setting
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# If 1, must use sudo
```

### Kernel Not Found

```bash
# List all kernels in the profile
ncu --import baseline_ncu.ncu-rep --list-kernels

# Use regex for kernel name
sudo ncu --kernel-name-base demangled --kernel-name ".*propagate_to_next_surface.*" ...
```

### Profile Too Large

```bash
# Reduce launch count
--launch-count 5

# Or use specific metrics instead of --set full
--metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active
```

### Build Errors

```bash
# Clean rebuild
rm -rf build/*
cmake .. [options]
cmake --build . -j$(nproc)
```

---

## Expected Results

Based on prior analysis, we expect:

| Metric | Expected |
|--------|----------|
| Register count | 128 (both variants) |
| Occupancy change | None (both limited by registers) |
| DRAM throughput | Lower in optimization (less memory traffic) |
| IPC | Similar or slightly higher in optimization |
| Memory stalls | Lower in optimization (less global memory access) |

These expectations are based on the mechanism being **skipped Jacobian aggregation** (no 6x6 matrix mult, no global memory read/write for accumulated Jacobian).

---

## Contact

If you encounter issues or have questions, please refer to:
- `doc/conditional_jacobian_transport_profile_report.md` - Existing profiling analysis
- `doc/conditional_jacobian_transport_profile_plan.md` - Original profiling plan
