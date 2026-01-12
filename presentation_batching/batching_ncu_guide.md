# NCU Profiling Guide for Root-Privileged Machine

**Purpose:** Run NVIDIA Nsight Compute (NCU) kernel profiling on baseline vs optimized commits.
**Prerequisite:** Root/sudo access required (`RmProfilingAdminOnly=1` on restricted systems)

---

## Overview

This guide provides step-by-step commands to profile the top GPU kernels:

| Kernel | % GPU Time | Priority |
|--------|------------|----------|
| `propagate_to_next_surface` | 29-33% | High |
| `fit_forward` | 26-32% | High |
| `fit_backward` | 12-16% | Medium |
| `find_tracks` | 5-10% | Medium |

**Commits to profile:**
- Baseline: `5cd477ac` (no batching)
- Optimized: `3ad492b5` (batch-48)

---

## Step 1: Clone and Setup

```bash
# Clone repository (if not already present)
git clone <repo-url> traccc-optimization
cd traccc-optimization

# Or pull latest if already cloned
git pull origin develop
```

---

## Step 2: Verify NCU Access

```bash
# Check NCU is available
which ncu || ls /usr/local/cuda*/bin/ncu

# Check if profiling requires root
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# If output is "RmProfilingAdminOnly: 1", use sudo for ncu commands

# Verify NCU version
ncu --version
```

---

## Step 3: Create Profiles Directory

```bash
mkdir -p profiles
```

---

## Step 4: Build and Profile Optimized Version (3ad492b5)

### 4.1 Checkout and Build

```bash
git checkout 3ad492b5

# Load cmake if using modules
module load cmake/3.28 2>/dev/null || true

# Configure (if build directory doesn't exist)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRACCC_BUILD_CUDA=ON
cd ..

# Build
cmake --build build --target traccc_throughput_st_cuda -j$(nproc)
```

### 4.2 Create Runner Script

```bash
cat > /tmp/run_optimized.sh << 'EOF'
#!/bin/bash
cd /path/to/traccc-optimization  # UPDATE THIS PATH
./build/bin/traccc_throughput_st_cuda \
    --batch-size 5 \
    --use-batched-api 1 \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --material-file=geometries/odd/odd-detray_material_detray.json \
    --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
    --input-directory=odd/geant4_ttbar_mu200 \
    --input-events=5 \
    --processed-events 5
EOF
chmod +x /tmp/run_optimized.sh

# UPDATE the path in the script
sed -i "s|/path/to/traccc-optimization|$(pwd)|g" /tmp/run_optimized.sh
```

### 4.3 Run NCU Profile (Optimized)

```bash
# Full profile (comprehensive but slow)
sudo ncu \
    --set full \
    --force-overwrite \
    -o profiles/ncu_optimized_3ad492b5 \
    /tmp/run_optimized.sh

# Alternative: Quick profile with key metrics only
sudo ncu \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
    --force-overwrite \
    -o profiles/ncu_optimized_3ad492b5 \
    /tmp/run_optimized.sh
```

### 4.4 Generate Optimized Report

```bash
# CSV export
ncu --import profiles/ncu_optimized_3ad492b5.ncu-rep --csv > profiles/ncu_optimized_report.csv

# Text summary
ncu --import profiles/ncu_optimized_3ad492b5.ncu-rep --print-summary per-kernel > profiles/ncu_optimized_summary.txt
```

---

## Step 5: Build and Profile Baseline Version (5cd477ac)

### 5.1 Checkout and Build

```bash
git checkout 5cd477ac

# Build
cmake --build build --target traccc_throughput_st_cuda -j$(nproc)
```

### 5.2 Create Runner Script

```bash
cat > /tmp/run_baseline.sh << 'EOF'
#!/bin/bash
cd /path/to/traccc-optimization  # UPDATE THIS PATH
./build/bin/traccc_throughput_st_cuda \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --material-file=geometries/odd/odd-detray_material_detray.json \
    --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
    --input-directory=odd/geant4_ttbar_mu200 \
    --input-events=5 \
    --processed-events 5
EOF
chmod +x /tmp/run_baseline.sh

# UPDATE the path in the script
sed -i "s|/path/to/traccc-optimization|$(pwd)|g" /tmp/run_baseline.sh
```

### 5.3 Run NCU Profile (Baseline)

```bash
# Full profile
sudo ncu \
    --set full \
    --force-overwrite \
    -o profiles/ncu_baseline_5cd477ac \
    /tmp/run_baseline.sh

# Alternative: Quick profile
sudo ncu \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
    --force-overwrite \
    -o profiles/ncu_baseline_5cd477ac \
    /tmp/run_baseline.sh
```

### 5.4 Generate Baseline Report

```bash
# CSV export
ncu --import profiles/ncu_baseline_5cd477ac.ncu-rep --csv > profiles/ncu_baseline_report.csv

# Text summary
ncu --import profiles/ncu_baseline_5cd477ac.ncu-rep --print-summary per-kernel > profiles/ncu_baseline_summary.txt
```

---

## Step 6: Return to Develop Branch

```bash
git checkout develop
```

---

## Step 7: Verify Output Files

```bash
ls -la profiles/ncu_*.ncu-rep profiles/ncu_*.csv profiles/ncu_*.txt
```

Expected files:
```
profiles/
├── ncu_baseline_5cd477ac.ncu-rep
├── ncu_baseline_report.csv
├── ncu_baseline_summary.txt
├── ncu_optimized_3ad492b5.ncu-rep
├── ncu_optimized_report.csv
└── ncu_optimized_summary.txt
```

---

## Step 8: Create Results Document

After profiling, create `docs/batching_ncu_results.md` with the following template:

```markdown
# NCU Profiling Results

**Date:** YYYY-MM-DD
**Hardware:** <GPU Model>
**NCU Version:** <version>

## Top Kernel Comparison

| Kernel | Metric | Baseline | Optimized | Change |
|--------|--------|----------|-----------|--------|
| propagate_to_next_surface | Achieved Occupancy | X% | Y% | +/-Z% |
| propagate_to_next_surface | Memory Throughput | X GB/s | Y GB/s | +/-Z% |
| fit_forward | Achieved Occupancy | X% | Y% | +/-Z% |
| fit_forward | Memory Throughput | X GB/s | Y GB/s | +/-Z% |
| ... | ... | ... | ... | ... |

## Roofline Analysis

[Include roofline chart observations]

## Cache Performance

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| L1 Hit Rate | X% | Y% | +/-Z% |
| L2 Hit Rate | X% | Y% | +/-Z% |

## Conclusions

[Summary of findings]
```

---

## Step 9: Commit and Push Results

```bash
git add profiles/ncu_*.csv profiles/ncu_*.txt docs/batching_ncu_results.md
git commit -m "docs: Add NCU profiling results for baseline vs optimized

Kernel-level profiling comparing 5cd477ac (baseline) vs 3ad492b5 (optimized):
- [Summary of key findings]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: <Your Name> <your@email.com>"

git push origin develop
```

---

## Troubleshooting

### NCU shows "No kernels were profiled"
- Ensure the application runs successfully without NCU first
- Try with `--target-processes all`
- Check if sudo is required: `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`

### Permission denied
- Use `sudo` for NCU commands
- Alternatively, set `RmProfilingAdminOnly=0` (requires reboot):
  ```bash
  sudo nvidia-smi -pm 1
  sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
  ```

### Build fails after checkout
- Clean build: `rm -rf build && mkdir build && cd build && cmake .. && cd ..`
- Ensure all dependencies are available

### Data files not found
- Verify `data/` directory exists with geometry and event files
- Check paths in runner scripts match your setup

---

## References

- NSYS Results: `docs/batching_profile_results.md`
- Optimization Report: `docs/batching_report.md`
- NCU Documentation: https://docs.nvidia.com/nsight-compute/
