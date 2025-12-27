---
name: optimization-workflow
description: Complete optimization workflow. Use BEFORE optimization to profile and find bottlenecks. Use AFTER optimization to verify correctness and measure improvement.
---

# Optimization Workflow

## BEFORE Optimization: Profile and Analyze

### 1. Profile with Nsight Systems

Run profiling to identify bottlenecks (see cuda-throughput-benchmark skill for full command):

```bash
/usr/local/cuda-12.6/bin/nsys profile --stats=true -o profile_output ./bin/traccc_throughput_mt_cuda ...
```

### 2. Get Kernel Stats

```bash
/usr/local/cuda-12.6/bin/nsys stats profile_output.nsys-rep --report cuda_gpu_kern_sum
/usr/local/cuda-12.6/bin/nsys stats profile_output.nsys-rep --report cuda_api_sum
```

### 3. Identify Bottlenecks

Look for:
- **GPU kernels** with highest time percentage
- **cudaStreamSynchronize** percentage (sync overhead)
- **cudaMemcpyAsync** frequency (transfer overhead)

### 4. Review Relevant Code

Key files for CKF optimization:
- `device/cuda/src/finding/combinatorial_kalman_filter.cuh`
- `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`
- `device/common/include/traccc/finding/device/impl/find_tracks.ipp`

### 5. Document Findings

Create `docs/benchmarks/profiling_analysis_<name>.md` with:
- Kernel time distribution table
- API time distribution table
- Identified bottlenecks
- Proposed optimizations ranked by effort/impact

---

## AFTER Optimization: Verify and Benchmark

### 1. Clean Build

```bash
cd /dicos_ui_home/noah/traccc
rm -rf build && mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DTRACCC_BUILD_CUDA=ON \
      -DTRACCC_BUILD_EXAMPLES=ON \
      ..
cmake --build . -j8
```

### 2. Run CUDA Tests

```bash
./bin/traccc_test_cuda
```

**Must pass:** All 710 tests

### 3. Run Benchmark

```bash
./bin/traccc_throughput_mt_cuda \
  --detector-file=../data/geometries/odd/odd-detray_geometry_detray.json \
  --material-file=../data/geometries/odd/odd-detray_material_detray.json \
  --grid-file=../data/geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=../data/geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=true \
  --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 \
  --processed-events=500 \
  --cpu-threads=1
```

**Baseline:** 25.32 events/s @ 1 thread

### 4. Verification Checklist

- [ ] Clean build completes without errors
- [ ] All 710 CUDA tests pass
- [ ] Benchmark runs without crashes
- [ ] Performance equal or better than baseline
- [ ] Document results in commit message

---

## Current Bottlenecks

| Component | Time % | Notes |
|-----------|--------|-------|
| propagate_to_next_surface | 64.3% | Runge-Kutta physics |
| cudaStreamSynchronize | 80.9% | Sync overhead |

## Reference Documents

- `docs/benchmarks/cuda_throughput_v1.0.0.md`
- `docs/benchmarks/profiling_analysis_v1.0.0.md`
- `docs/benchmarks/profiling_analysis_post_sync_removal.md`
