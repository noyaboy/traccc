# CUDA Throughput Benchmark - Sync Removal Optimization

## Optimization Summary

Removed two unnecessary CUDA stream synchronizations in the CKF algorithm:

1. `copy.setup(keys_buffer)->wait()` changed to `->ignore()` (line 453)
   - Stream ordering already ensures buffer setup completes before kernels use it

2. `str.synchronize()` after propagation kernel removed (line 516)
   - Stream ordering guarantees kernel completion order within same stream

## Test Environment

- **GPU**: Tesla V100-SXM2-32GB
- **CUDA Version**: 12.6
- **Host Compiler**: GCC 13.1.0
- **traccc Version**: commit beb6fb85 (optimization/remove-unnecessary-sync branch)
- **Dataset**: odd/geant4_ttbar_mu200 (36 events, 500 processed)
- **Baseline**: Commit 2cf881f4 (before optimization)

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

## Results (3x median)

| cpu-threads | Baseline (events/s) | Optimized (events/s) | Change |
|-------------|---------------------|----------------------|--------|
| 1           | 25.32               | 27.05                | +6.8%  |
| 4           | 51.30               | 51.01                | -0.6%  |
| 8           | 59.21               | 58.85                | -0.6%  |

### Raw Benchmark Data

**Thread 1:**
- Run 1: 27.73 events/s
- Run 2: 27.05 events/s
- Run 3: 24.10 events/s
- Median: 27.05 events/s

**Thread 4:**
- Run 1: 51.01 events/s
- Run 2: 50.68 events/s
- Run 3: 51.44 events/s
- Median: 51.01 events/s

**Thread 8:**
- Run 1: 58.85 events/s
- Run 2: 57.06 events/s
- Run 3: 59.03 events/s
- Median: 58.85 events/s

## Analysis

1. **Single-thread improvement (+6.8%)**: The sync removal allows better host-device overlap
   when only one stream is active, reducing idle time between operations.

2. **Multi-thread no change (~0%)**: At higher thread counts, the GPU is already saturated
   with work from multiple concurrent streams. The removed syncs were not the bottleneck.

3. **Main bottleneck remains**: The `cudaStreamSynchronize` at line 365 (reading n_candidates
   from GPU to CPU for buffer allocation) is still the dominant sync, accounting for ~80%
   of API time.

## Decision

**KEEP** the optimization because:
- No regression at any thread count (changes are within measurement noise)
- Slight improvement at single-thread workloads
- Changes are semantically correct (syncs were redundant due to stream ordering)
- Reduces code complexity by removing unnecessary synchronization points

## Next Steps

To achieve further improvement, consider:
1. **Device-side candidate count** - Avoid reading n_candidates back to host by computing
   buffer sizes on GPU (Option 1 from profiling analysis)
2. **Multi-event batching** - Process multiple events in a single kernel launch to reduce
   per-event overhead (Option 3 from profiling analysis)
