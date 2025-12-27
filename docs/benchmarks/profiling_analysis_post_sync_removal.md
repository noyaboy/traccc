# CUDA Profiling Analysis (Post Sync Removal Optimization)

## Current State

After removing 2 unnecessary synchronizations (commit a9939663), the profile shows:

### GPU Kernel Time Distribution

| Kernel | Time (%) | Total Time (ms) | Instances | Avg (ms) |
|--------|----------|-----------------|-----------|----------|
| **propagate_to_next_surface** | **64.3%** | 2084.8 | 2134 | 0.977 |
| find_tracks | 10.3% | 332.6 | 2244 | 0.148 |
| find_doublets | 3.8% | 124.5 | 110 | 1.132 |
| ccl_kernel | 3.0% | 97.5 | 110 | 0.886 |
| count_triplets | 2.6% | 83.6 | 110 | 0.760 |
| count_doublets | 2.5% | 80.4 | 110 | 0.731 |
| DeviceRadixSortOnesweep | 2.3% | 75.6 | 2716 | 0.028 |
| build_tracks | 1.8% | 59.6 | 110 | 0.542 |

### CUDA API Time Distribution

| API Call | Time (%) | Total Time (ms) | Calls |
|----------|----------|-----------------|-------|
| **cudaStreamSynchronize** | **80.9%** | 2996.6 | 10816 |
| cudaMemcpyAsync | 12.2% | 451.7 | 10708 |
| cudaLaunchKernel | 3.3% | 121.0 | 23307 |
| cudaEventSynchronize | 1.3% | 46.4 | 886 |

## Key Observations

1. **propagate_to_next_surface still dominates** at 64.3% of GPU time
   - This kernel does Runge-Kutta integration through magnetic field
   - Complex physics calculations, inherently expensive
   - Variable work per thread (thread divergence)

2. **cudaStreamSynchronize remains the bottleneck** at 80.9% of API time
   - 10816 sync calls for 100 events = ~108 syncs per event
   - For ~21 CKF steps, that's ~5 syncs per step
   - Remaining syncs are in CKF (4), seeding (2), fitting (1), ambiguity resolution (8)

3. **Critical sync at line 365** happens every CKF step
   - Syncs to read link count (n_candidates) from GPU to CPU
   - n_candidates used for buffer allocation and kernel launch sizing
   - This is the highest-frequency sync in the CKF loop

## Recommended Next Optimizations

### Option 1: Device-Side Candidate Count (Medium Effort, High Impact)

**Problem**: Line 365 syncs to read n_candidates, which is used for:
- Buffer allocations (lines 379, 446, 451)
- Kernel launch grid sizing (lines 389, 416, 455, 506)
- Early exit conditions (lines 376, 441)

**Solution**: Keep n_candidates on device and restructure:
1. Pre-allocate buffers with n_max_candidates size
2. Pass candidate count via device pointer to kernels
3. Kernels check param_liveness instead of relying on exact launch size
4. Only sync at the end of CKF to get final results

**Expected Impact**: Could reduce syncs per event from ~108 to ~50 (-50%)

### ~~Option 2: Pinned Memory for Staging~~ (ALREADY IMPLEMENTED)

**Status**: ✅ Already implemented in `full_chain_algorithm`

**Analysis (2025-12-27)**: Investigation revealed that pinned memory is already being used:
- `full_chain_algorithm.hpp:120` declares `vecmem::cuda::host_memory_resource m_pinned_host_mr;`
- `full_chain_algorithm.cpp:77-78` passes `{m_cached_device_mr, &m_cached_pinned_host_mr}` to CKF
- The `mr.host` in CKF already points to pinned memory via `m_cached_pinned_host_mr`
- Profile confirms `cudaMallocHost` calls (pinned allocation)

No further optimization possible here.

### Option 3: Multi-Event Batching (High Effort, High Impact)

**Problem**: Each event processed independently, 2134 kernel launches for 100 events

**Solution**: Batch multiple events into single kernel launch
- Better GPU occupancy
- Fewer kernel launches and syncs
- Requires restructuring data layout

**Expected Impact**: Could improve throughput by 30-50% at high thread counts

## Files to Modify

- `device/cuda/src/finding/combinatorial_kalman_filter.cuh` - Main CKF algorithm
- `examples/run/cuda/throughput_mt.cpp` - For multi-event batching

## Recommendation

~~Start with **Option 2 (Pinned Memory)** as it's lowest effort, then proceed to **Option 1 (Device-Side Count)** for larger gains.~~

**Updated (2025-12-27)**: Option 2 is already implemented. The next optimization to pursue is:
1. **Option 1 (Device-Side Candidate Count)** - Medium effort, high impact (~50% sync reduction)
2. **Option 3 (Multi-Event Batching)** - High effort, high impact (30-50% throughput improvement)
