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

## Baseline Results (v1.0.0 on Tesla V100-32GB)

| cpu-threads | ms/event | events/s | Scaling |
|-------------|----------|----------|---------|
| 1           | 39.49    | 25.32    | 1.00x   |
| 4           | 19.49    | 51.30    | 2.03x   |
| 8           | 16.89    | 59.21    | 2.34x   |

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

- Max 8 threads before OOM on V100-32GB
- Use `-DCMAKE_CUDA_ARCHITECTURES=70` for Tesla V100
