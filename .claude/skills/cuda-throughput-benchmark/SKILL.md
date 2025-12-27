---
name: cuda-throughput-benchmark
description: Run CUDA throughput benchmarks on traccc. Use when user asks to run throughput benchmark, measure performance, or compare against baseline.
---

# CUDA Throughput Benchmark

## Baseline Command (59.21 events/s @ 8 threads)

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
  --cpu-threads=8
```

## Baseline Results (v1.0.0 on Tesla V100-32GB)

| cpu-threads | events/s | Scaling |
|-------------|----------|---------|
| 1           | 25.32    | 1.00x   |
| 4           | 51.30    | 2.03x   |
| 8           | 59.21    | 2.34x   |

## Usage

1. Run from `build/` directory
2. Adjust `--cpu-threads` (max 8 before OOM on V100-32GB)
3. Compare results against baseline 59.21 events/s

## Build Requirements

```bash
cmake -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DTRACCC_BUILD_CUDA=ON \
      -DTRACCC_BUILD_EXAMPLES=ON \
      ..
```
