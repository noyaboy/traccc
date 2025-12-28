---
name: cuda-throughput-benchmark
description: CUDA throughput benchmark configuration and baseline results. Single source of truth for benchmark commands, build config, and performance baselines.
---

# CUDA Throughput Benchmark

## Dataset

Always use **geant4_ttbar_mu200** dataset:
- Input directory: `../data/odd/geant4_ttbar_mu200/`
- Input events: 36

## Build Configuration

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

## Benchmark Command

```bash
cd /dicos_ui_home/noah/traccc/build
./bin/traccc_throughput_mt_cuda \
  --detector-file=../data/geometries/odd/odd-detray_geometry_detray.json \
  --material-file=../data/geometries/odd/odd-detray_material_detray.json \
  --grid-file=../data/geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=../data/geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=true \
  --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 \
  --processed-events=500 \
  --cpu-threads=<N>
```

## Baseline Results (on Tesla V100-32GB)

| cpu-threads | ms/event | events/s | Scaling | Notes |
|-------------|----------|----------|---------|-------|
| 1           | 39.53    | 25.30    | 1.00x   | Most stable |
| 4           | 19.47    | 51.37    | 2.03x   | Good balance |
| 6           | 17.78    | 56.22    | 2.22x   | |
| 7           | 17.05    | 58.65    | 2.32x   | **Recommended** (stable) |
| 8           | 17.30    | 57.82    | 2.29x   | Max (may OOM occasionally) |
| 10+         | OOM      | -        | -       | Out of memory |

**Primary baseline:** 57.82 events/s @ 8 threads
**Recommended:** 58.65 events/s @ 7 threads (more stable, avoids OOM)

## CUDA Test Command

```bash
cd /dicos_ui_home/noah/traccc/build
./bin/traccc_test_cuda
```

**Expected:** 710 tests pass

## Profiling Command

```bash
cd /dicos_ui_home/noah/traccc/build
/usr/local/cuda-12.6/bin/nsys profile --stats=true -o profile_output \
  ./bin/traccc_throughput_mt_cuda \
  --detector-file=../data/geometries/odd/odd-detray_geometry_detray.json \
  --material-file=../data/geometries/odd/odd-detray_material_detray.json \
  --grid-file=../data/geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=../data/geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=true \
  --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 \
  --processed-events=100 \
  --cpu-threads=1
```

## Notes

- Max 8 threads before OOM on V100-32GB (9+ threads OOM)
- 7 threads recommended for stability (58.65 events/s vs 57.82 @ 8 threads)
- Use `-DCMAKE_CUDA_ARCHITECTURES=70` for Tesla V100
- Memory is the limiting factor for thread scaling
