# CUDA Throughput Benchmark Results (v1.0.0)

## Test Environment

- **GPU**: Tesla V100-SXM2-32GB
- **CUDA Version**: 12.6
- **Host Compiler**: GCC 13.4.0
- **traccc Version**: v1.0.0 + PR #1224 (MBF cleanup fix)
- **Dataset**: odd/geant4_ttbar_mu200 (36 events)

## Benchmark Command

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
  --cpu-threads=<N>
```

## Results

| cpu-threads | ms/event | events/s | Scaling | Status |
|-------------|----------|----------|---------|--------|
| 1           | 39.49    | 25.32    | 1.00x   | OK     |
| 4           | 19.49    | 51.30    | 2.03x   | OK     |
| 8           | 16.89    | 59.21    | 2.34x   | OK     |
| 10          | -        | -        | -       | OOM    |
| 12          | -        | -        | -       | OOM    |
| 16          | -        | -        | -       | OOM    |

## Observations

1. **Peak Throughput**: 59.21 events/s achieved with 8 CPU threads
2. **GPU Memory Limitation**: Each concurrent CUDA instance requires ~3-4 GB of GPU memory. With 32GB total, only 8 instances can run before hitting out-of-memory errors.
3. **Scaling Efficiency**: 2.34x speedup with 8 threads (29% efficiency per thread)
4. **Bottleneck**: GPU memory is the limiting factor for scaling beyond 8 threads

## Build Configuration

```bash
cmake -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DTRACCC_BUILD_CUDA=ON \
      -DTRACCC_BUILD_EXAMPLES=ON \
      ..
```

## Notes

- The v1.0.0 release contains a bug in the MBF smoother buffer resize code (copy direction reversed). This was fixed by cherry-picking commit `09520e90` from PR #1224.
- All 710 CUDA tests pass after applying the fix.
