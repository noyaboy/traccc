---
name: optimization-analysis
description: Workflow for analyzing next optimization opportunities. Use this before starting any new optimization to identify bottlenecks through profiling.
---

# Optimization Analysis Workflow

## Required Steps Before Any Optimization

Before starting any optimization work, ALWAYS follow this analysis workflow:

### 1. Profile with Nsight Systems

```bash
cd /dicos_ui_home/noah/traccc/build
nsys profile --stats=true -o profile_output ./bin/traccc_throughput_mt_cuda \
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

### 2. Analyze GPU Kernel Time Distribution

Look for output like:
```
Time (%)  Total Time (ms)  Instances  Avg (ms)   Kernel Name
64.3%     2070.4           2124       0.975      propagate_to_next_surface
10.3%     332.3            2234       0.149      find_tracks
...
```

Focus on kernels with highest time percentage.

### 3. Analyze CUDA API Time Distribution

Look for synchronization overhead:
```
Time (%)  Total Time (ms)  Calls    API Call
81.2%     2989.4           12932    cudaStreamSynchronize
11.8%     433.9            10710    cudaMemcpyAsync
...
```

High `cudaStreamSynchronize` percentage indicates sync overhead opportunities.

### 4. Review Relevant Code

Based on profiling results, read and analyze:

- **Main CKF algorithm**: `device/cuda/src/finding/combinatorial_kalman_filter.cuh`
- **Propagation kernel**: `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`
- **Track finding kernel**: `device/common/include/traccc/finding/device/impl/find_tracks.ipp`

### 5. Identify Optimization Opportunities

Categories to look for:

| Category | Effort | Impact | Examples |
|----------|--------|--------|----------|
| Sync removal | Low | Medium | Remove unnecessary `str.synchronize()` |
| Kernel fusion | Medium | Medium | Combine sort + propagate |
| Memory optimization | Medium | High | Shared memory caching |
| Algorithm improvement | High | High | Multi-event batching |

### 6. Document Findings

Create analysis document at `docs/benchmarks/profiling_analysis_<version>.md` with:
- Profiling command used
- GPU kernel time distribution table
- CUDA API time distribution table
- Identified bottlenecks
- Proposed optimization opportunities ranked by effort/impact

## Current Bottlenecks (v1.0.0)

1. **propagate_to_next_surface** - 64.3% of GPU time
   - Runge-Kutta integration through magnetic field
   - Complex detector geometry navigation

2. **cudaStreamSynchronize** - 81.2% of API time
   - Multiple sync points per CKF step
   - Opportunity: Remove unnecessary syncs

## Reference Documents

- `docs/benchmarks/profiling_analysis_v1.0.0.md` - Baseline profiling analysis
- `docs/benchmarks/cuda_throughput_v1.0.0.md` - Baseline benchmark results
