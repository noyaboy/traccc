# CUDA Throughput Profiling Analysis (v1.0.0)

## Profiling Summary

**Command:**
```bash
nsys profile --stats=true -o throughput_profile ./bin/traccc_throughput_mt_cuda \
  --detector-file=../data/geometries/odd/odd-detray_geometry_detray.json \
  --material-file=../data/geometries/odd/odd-detray_material_detray.json \
  --grid-file=../data/geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=../data/geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=true --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 --processed-events=100 --cpu-threads=1
```

## GPU Kernel Time Distribution

| Kernel | Time (%) | Total Time (ms) | Instances | Avg (ms) |
|--------|----------|-----------------|-----------|----------|
| **propagate_to_next_surface** | **64.3%** | 2070.4 | 2124 | 0.975 |
| find_tracks | 10.3% | 332.3 | 2234 | 0.149 |
| find_doublets | 3.9% | 126.1 | 110 | 1.146 |
| count_triplets | 2.7% | 85.6 | 110 | 0.779 |
| ccl_kernel (clusterization) | 2.6% | 84.6 | 110 | 0.769 |
| count_doublets | 2.5% | 81.8 | 110 | 0.743 |
| DeviceRadixSortOnesweep | 2.3% | 73.9 | 2656 | 0.028 |
| build_tracks | 1.8% | 58.7 | 110 | 0.534 |
| Other kernels | ~9.6% | ~300 | - | - |

## CUDA API Time Distribution

| API Call | Time (%) | Total Time (ms) | Calls |
|----------|----------|-----------------|-------|
| **cudaStreamSynchronize** | **81.2%** | 2989.4 | 12932 |
| cudaMemcpyAsync | 11.8% | 433.9 | 10710 |
| cudaLaunchKernel | 3.3% | 122.8 | 23097 |
| cudaEventSynchronize | 1.2% | 45.9 | 886 |
| cudaMemsetAsync | 0.9% | 34.5 | 8592 |

## Key Bottleneck: propagate_to_next_surface (64.3%)

This kernel propagates track parameters from one detector surface to the next. It:
1. Creates a propagator state for each track parameter
2. Runs Runge-Kutta integration through the magnetic field
3. Handles navigation through the detector geometry
4. Updates track parameters on the next sensitive surface

**Why it's slow:**
- Complex physics calculations (Runge-Kutta integration)
- Heavy memory access pattern (detector geometry, magnetic field)
- Each thread processes one track independently (no data sharing)
- Variable work per thread (some tracks take more steps)

## CKF Algorithm Flow (Per Step)

1. **find_tracks** - Match measurements to tracks, Kalman update
2. **fill_finding_duplicate_removal_sort_keys** - Prepare for deduplication
3. **thrust::sort_by_key** - Sort tracks for deduplication
4. **remove_duplicates** - Remove duplicate track candidates
5. **fill_finding_propagation_sort_keys** - Sort by surface for propagation
6. **thrust::sort_by_key** - Sort tracks by surface
7. **propagate_to_next_surface** - Propagate to next layer

## Synchronization Overhead

The CKF has multiple synchronization points per step:
- Line 300: After MBF buffer resize
- Line 365: After find_tracks to get link count
- Line 474: After sort_by_key
- Line 515: After propagate_to_next_surface

Each sync blocks CPU-GPU parallelism and adds latency.

## Optimization Opportunities

### 1. Reduce Synchronization (Low Effort, Medium Impact)
- Remove sync at line 474 (after sort_by_key) - stream ordering handles this
- Batch D2H copies to reduce sync frequency
- Use CUDA events for finer-grained synchronization

### 2. Kernel Fusion (Medium Effort, Medium Impact)
- Fuse `fill_finding_propagation_sort_keys` + sort + `propagate_to_next_surface`
- Reduce kernel launch overhead and memory round-trips

### 3. Propagation Optimization (High Effort, High Impact)
- **Occupancy tuning**: Current uses `warp_size * 4 = 128` threads/block
- **Memory coalescing**: Sort tracks by surface before propagation (already done)
- **Shared memory**: Cache detector geometry/B-field per block
- **Warp-level optimization**: Use warp shuffle for reductions

### 4. Multi-Event Batching (High Effort, High Impact)
- Process multiple events in a single kernel launch
- Better GPU utilization for smaller events
- Requires significant restructuring of data layout

### 5. Asynchronous Pipeline (Medium Effort, High Impact)
- Overlap event N's D2H copy with event N+1's kernel execution
- Use multiple CUDA streams per event
- Requires careful memory management

## Recommended Next Steps

1. **Quick Win**: Remove unnecessary `str.synchronize()` at line 474
2. **Profile again** to measure impact
3. **Investigate propagation kernel** with Nsight Compute for detailed metrics
4. **Consider batching** events to improve GPU utilization

## Files Analyzed

- `device/cuda/src/finding/combinatorial_kalman_filter.cuh` - Main CKF algorithm
- `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` - Propagation kernel
- `device/common/include/traccc/finding/device/impl/find_tracks.ipp` - Track finding kernel
