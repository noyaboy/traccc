---
name: optimization-verification
description: Verification workflow after completing any optimization. Use this after finishing any performance optimization to ensure functionality is preserved.
---

# Optimization Verification Workflow

## Required Steps After Any Optimization

After completing any optimization work, ALWAYS run the following verification steps:

### 1. Clean Build

```bash
cd /dicos_ui_home/noah/traccc
rm -rf build
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DTRACCC_BUILD_CUDA=ON \
      -DTRACCC_BUILD_EXAMPLES=ON \
      ..
cmake --build . -j8
```

### 2. Run CUDA Tests

```bash
cd /dicos_ui_home/noah/traccc/build
./bin/traccc_test_cuda
```

**Expected Result:** All 710 tests should pass.

### 3. Run Throughput Benchmark

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

**Baseline (v1.0.0):** 25.32 events/s @ 1 thread

## Verification Checklist

- [ ] Clean build completes without errors
- [ ] All 710 CUDA tests pass
- [ ] Throughput benchmark runs without crashes
- [ ] Performance is equal or better than baseline

## Notes

- Always use `-DCMAKE_CUDA_ARCHITECTURES=70` for Tesla V100 GPU
- If tests fail, revert the optimization and investigate
- Document any performance changes in commit message
