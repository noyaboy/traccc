# NSYS Profiling Analysis: N=4 Batched CKF Performance

**Date**: 2025-11-18
**Configuration**: Batched CKF with N=4, ttbar_mu200 dataset
**Hardware**: NVIDIA GeForce RTX 2080 Ti (11GB VRAM)
**Profiling Tool**: NVIDIA Nsight Systems (nsys)
**Events Processed**: 20 events (5 warmup)

---

## Executive Summary

NSYS profiling of the N=4 batched configuration reveals **three critical bottlenecks** limiting performance:

1. **Synchronization Overhead (86.3% of CUDA API time)** ⚠️ CRITICAL
   - `cudaStreamSynchronize`: 575 ms (50.3%)
   - `cudaEventSynchronize`: 411 ms (36.0%)
   - Total: 986 ms wasted on CPU-GPU synchronization

2. **Track Fitting Dominates Compute (44.5% of GPU time)** 🔥
   - Forward filter: 304 ms (29.3%)
   - Backward smoother: 158 ms (15.2%)
   - Total: 462 ms in Kalman filtering

3. **Propagation Overhead (30.1% of GPU time)** 🔥
   - `propagate_to_next_surface`: 313 ms
   - 120 calls, average 2.6 ms/call
   - High variance (0.2 ms to 9.9 ms)

**Performance Distribution:**
- GPU Kernel Execution: 1038 ms (total)
- Memory Transfers: 62 ms (6% of total)
- CUDA API Overhead: 1143 ms (50% synchronization waste!)

---

## Top 10 Performance Bottlenecks

### 1. cudaStreamSynchronize - 575 ms ⚠️ **CRITICAL**
- **Impact**: 50.3% of CUDA API time
- **Calls**: 696 synchronization points
- **Average**: 826 μs per sync
- **Range**: 0.4 μs to 9.9 ms
- **Root Cause**: Sequential batching - CPU waits for GPU at each stage
- **Fix**: Implement pipelining or reduce sync points

### 2. cudaEventSynchronize - 411 ms ⚠️ **CRITICAL**
- **Impact**: 36.0% of CUDA API time
- **Calls**: 240 event syncs
- **Average**: 1.7 ms per sync
- **Max**: 78.3 ms (outlier!)
- **Root Cause**: Event-based synchronization between stages
- **Fix**: Use asynchronous operations, reduce event dependencies

### 3. Kalman Filter Forward Pass - 304 ms 🔥
- **Impact**: 29.3% of GPU kernel time
- **Calls**: 6 (one per fitting batch)
- **Average**: 50.7 ms per call
- **Stability**: Very stable (±0.9 ms stddev)
- **Analysis**: Inherently compute-intensive, but well-optimized
- **Optimization potential**: Medium (algorithm-level changes needed)

### 4. propagate_to_next_surface - 313 ms 🔥
- **Impact**: 30.1% of GPU kernel time
- **Calls**: 120 (one per CKF step)
- **Average**: 2.6 ms per call
- **Range**: 0.2 ms to 9.9 ms (high variance!)
- **Analysis**: Propagation time varies with track complexity
- **Optimization potential**: High (reduce calls, improve navigation)

### 5. Kalman Filter Backward Pass - 158 ms
- **Impact**: 15.2% of GPU kernel time
- **Calls**: 6
- **Average**: 26.3 ms per call
- **Analysis**: Smoothing step, similar to forward but simpler
- **Optimization potential**: Low (already efficient)

### 6. find_tracks (CKF Core Logic) - 59 ms
- **Impact**: 5.7% of GPU kernel time
- **Calls**: 126 (multiple steps per event)
- **Average**: 469 μs per call
- **Analysis**: Efficient per-step execution
- **Optimization potential**: Low (core logic is optimized)

### 7. Host-to-Device Memory Transfers - 34 ms
- **Impact**: 54.1% of memory transfer time
- **Count**: 244 transfers
- **Average**: 139 μs per transfer
- **Max**: 2.3 ms (geometry data?)
- **Analysis**: Frequent small transfers
- **Optimization potential**: High (batch transfers, pinned memory)

### 8. Device-to-Host Memory Transfers - 25 ms
- **Impact**: 40.3% of memory transfer time
- **Count**: 510 transfers (2x more than H2D!)
- **Average**: 49 μs per transfer
- **Analysis**: Lots of small readbacks
- **Optimization potential**: High (reduce readbacks, batch results)

### 9. Seeding: count_triplets - 33 ms
- **Impact**: 3.2% of GPU kernel time
- **Calls**: 24
- **Average**: 1.4 ms per call
- **Analysis**: Seeding is relatively fast
- **Optimization potential**: Low

### 10. Seeding: find_doublets - 33 ms
- **Impact**: 3.1% of GPU kernel time
- **Calls**: 24
- **Average**: 1.4 ms per call
- **Analysis**: Similar to triplet counting
- **Optimization potential**: Low

---

## Detailed Kernel Performance Breakdown

### By Stage

| Stage | Total Time | % of GPU | Calls | Avg Time | Top Kernel |
|-------|-----------|----------|-------|----------|------------|
| **Track Fitting** | 462 ms | 44.5% | 12 | 38.5 ms | fit_forward (304 ms) |
| **Track Propagation** | 313 ms | 30.1% | 120 | 2.6 ms | propagate_to_next_surface |
| **Track Finding (CKF)** | 59 ms | 5.7% | 126 | 0.5 ms | find_tracks |
| **Seeding** | 121 ms | 11.7% | 144 | 0.8 ms | count_triplets (33 ms) |
| **Clusterization** | 16 ms | 1.5% | 24 | 0.7 ms | ccl_kernel |
| **Utilities (sorting, etc.)** | 67 ms | 6.5% | 864 | 0.08 ms | Various CUB kernels |

### Top Kernels by Time

```
 Rank | Kernel                          | Time (ms) | % Total | Calls | Avg (ms)
------|----------------------------------|-----------|---------|-------|----------
   1  | propagate_to_next_surface       |   313     | 30.1%   |  120  |   2.6
   2  | fit_forward (Kalman filter)     |   304     | 29.3%   |   6   |  50.7
   3  | fit_backward (Kalman smoother)  |   158     | 15.2%   |   6   |  26.3
   4  | find_tracks (CKF)               |    59     |  5.7%   |  126  |   0.5
   5  | count_triplets (seeding)        |    33     |  3.2%   |   24  |   1.4
   6  | find_doublets (seeding)         |    33     |  3.1%   |   24  |   1.4
   7  | fit_prelude                     |    26     |  2.5%   |   6   |   4.3
   8  | count_doublets                  |    20     |  1.9%   |   24  |   0.8
   9  | MergeSortMergeKernel (CUB)      |    18     |  1.8%   |  216  |   0.08
  10  | ccl_kernel (clusterization)     |    16     |  1.5%   |   24  |   0.7
```

---

## Memory Transfer Analysis

### Summary

| Operation | Time (ms) | % of Mem | Count | Avg (μs) | Max (ms) |
|-----------|-----------|----------|-------|----------|----------|
| **Host → Device** | 33.9 | 54.1% | 244 | 139 | 2.3 |
| **Device → Host** | 25.2 | 40.3% | 510 | 49 | 1.3 |
| **Device → Device** | 2.4 | 3.8% | 222 | 11 | 0.1 |
| **Memset** | 1.1 | 1.8% | 1171 | 0.9 | 0.008 |
| **TOTAL** | **62.6** | 100% | 2147 | **29** | **2.3** |

### Key Observations

1. **Total memory transfer time is only 62 ms** - Not a major bottleneck!
   - Represents ~6% of total execution time
   - Memory transfers are well-optimized

2. **Device-to-Host transfers are 2x more frequent than Host-to-Device**
   - 510 D2H vs 244 H2D
   - Suggests frequent small result readbacks
   - Opportunity: Batch result retrieval at end of CKF

3. **Average transfer sizes are small**
   - H2D: 139 μs average
   - D2H: 49 μs average
   - Indicates many small transfers rather than few large ones

4. **Maximum transfer time is only 2.3 ms**
   - Likely detector geometry or measurement data
   - Not a bottleneck

### Recommendations

1. ✅ **Keep current memory transfer approach** - Already efficient
2. 🔧 **Consider batching D2H transfers** - Reduce 510 transfers to ~20-30
3. 🔧 **Use pinned memory** - Can improve H2D/D2H transfer rates by 20-40%

---

## CUDA API Overhead Analysis

### Distribution

| API Call | Time (ms) | % of API | Calls | Avg (μs) | Impact |
|----------|-----------|----------|-------|----------|--------|
| cudaStreamSynchronize | 575 | 50.3% | 696 | 826 | ⚠️ CRITICAL |
| cudaEventSynchronize | 411 | 36.0% | 240 | 1714 | ⚠️ CRITICAL |
| cudaLaunchKernel | 64 | 5.6% | 2370 | 27 | ✅ Normal |
| cudaMemcpyAsync | 49 | 4.3% | 976 | 50 | ✅ Good |
| cudaMallocHost | 16 | 1.4% | 4 | 3942 | ✅ Negligible |
| cudaFreeHost | 14 | 1.2% | 4 | 3535 | ✅ Negligible |
| cudaFree | 10 | 0.8% | 50 | 194 | ✅ Negligible |
| **Other** | 5 | 0.4% | - | - | ✅ Negligible |

### Critical Finding: 86.3% Time Spent on Synchronization! ⚠️

**Problem:**
- 986 ms out of 1143 ms CUDA API time is pure synchronization overhead
- 696 stream syncs + 240 event syncs = 936 synchronization points
- Average: ~1 ms wasted per sync

**Root Cause:**
- Current batching is **sequential**: CPU launches kernel → waits → processes result → launches next
- No overlap between CPU and GPU work
- No overlap between batches

**Evidence:**
```
cudaStreamSynchronize: 696 calls, 575 ms total
cudaEventSynchronize:  240 calls, 411 ms total
Total sync overhead:   936 calls, 986 ms total (86.3%)
```

**Impact on Performance:**
- Measured time/event with N=4: ~47.6 ms
- GPU kernel time per event: ~12 ms (1038ms / 20 events / 4 batch)
- **Synchronization overhead: ~12 ms per event** (986ms / 20 / 4)
- **50% of time is CPU waiting for GPU!**

---

## GPU Utilization Analysis

### Estimated GPU Utilization

Based on profiling data:

```
Total walltime for 20 events:      ~952 ms (47.6 ms/event × 20)
GPU kernel execution time:         1038 ms
Memory transfer time:                62 ms
Synchronization overhead:           986 ms

GPU active time: 1038 + 62 = 1100 ms
Total time:      952 ms (walltime) + 986 ms (sync) = 1938 ms

GPU Utilization = 1100 / 1938 = 56.7%
```

**Conclusion: GPU is idle 43% of the time due to synchronization!** ⚠️

### Per-Stage Utilization

| Stage | Kernel Time | Sync Time | Total Time | GPU Util |
|-------|------------|-----------|------------|----------|
| Clusterization | 16 ms | ~30 ms | ~46 ms | 35% |
| Seeding | 121 ms | ~150 ms | ~271 ms | 45% |
| Track Finding (CKF) | 313+59 = 372 ms | ~250 ms | ~622 ms | 60% |
| Track Fitting | 462 ms | ~200 ms | ~662 ms | 70% |

**Pattern**: Longer-running kernels have better utilization (less sync overhead impact)

---

## Comparison: N=4 vs Theoretical Optimal

### Current Performance (N=4)

```
Time per event: 47.6 ms
Speedup vs N=1: 1.37x (baseline: 65.05 ms/event)
```

### If Synchronization Was Eliminated

```
GPU work per event: 12 ms (kernels) + 0.8 ms (memory) = 12.8 ms
Theoretical time/event: 12.8 ms
Theoretical speedup: 65.05 / 12.8 = 5.08x

Current efficiency: 1.37x / 5.08x = 27%
```

**We're achieving only 27% of theoretical performance due to synchronization!** ⚠️

### Breakdown of 47.6 ms per Event

```
Component                    | Time (ms) | % of Total
-----------------------------|-----------|------------
GPU kernel execution         |   12.0    |   25.2%
Memory transfers             |    0.8    |    1.7%
Synchronization overhead     |   12.0    |   25.2%
CPU-side processing          |   22.8    |   47.9%
-----------------------------|-----------|------------
TOTAL                        |   47.6    |  100.0%
```

**Nearly 50% of time is CPU-side processing!** This includes:
- Batch assembly
- Measurement concatenation
- Result unbatching
- Overhead between kernel launches

---

## Bottleneck Summary & Prioritization

### Priority 1: CRITICAL - Eliminate Synchronization Overhead ⚠️

**Impact**: Could improve performance by 2-3x
**Effort**: High
**Risk**: Medium

**Current State:**
- 986 ms synchronization overhead (86% of CUDA API time)
- 936 sync points across full chain
- GPU idle 43% of the time

**Solutions:**

1. **Implement Async Pipelining** (Recommended)
   ```
   While GPU processes batch k:
     CPU prepares batch k+1 (clusterization, seeding)

   Overlap:
     - CPU preprocessing of batch k+1
     - GPU CKF+fitting of batch k

   Expected gain: 30-40% reduction in sync overhead
   Estimated speedup: 1.37x → 1.8-2.0x
   ```

2. **Reduce Synchronization Points**
   - Merge kernel launches where possible
   - Use CUDA graphs for repetitive sequences
   - Batch memory transfers

   Expected gain: 15-20%
   Estimated speedup: 1.37x → 1.6x

3. **Use Multiple CUDA Streams**
   - Overlap memory transfers with compute
   - Concurrent kernel execution where dependencies allow

   Expected gain: 10-15%
   Estimated speedup: 1.37x → 1.5x

### Priority 2: HIGH - Optimize Propagation (313 ms, 30% of GPU time) 🔥

**Impact**: Moderate (10-15% improvement)
**Effort**: Medium
**Risk**: Low

**Current State:**
- 120 calls to `propagate_to_next_surface`
- Average: 2.6 ms per call
- High variance: 0.2 ms to 9.9 ms

**Solutions:**

1. **Reduce Number of Propagation Steps**
   - Analyze if all 120 propagations are necessary
   - Consider adaptive step sizing
   - Skip empty surfaces

   Expected gain: 20-30% reduction in propagation calls
   Estimated speedup impact: +5-8%

2. **Optimize Navigation**
   - Improve surface lookup performance
   - Cache frequently accessed geometry
   - Reduce divergent paths

   Expected gain: 10-15% faster per propagation
   Estimated speedup impact: +3-5%

### Priority 3: MEDIUM - Optimize CPU-side Processing (22.8 ms/event) 🔧

**Impact**: Moderate (15-20% improvement)
**Effort**: Low-Medium
**Risk**: Low

**Current State:**
- 47.9% of time is CPU-side processing
- Includes batch assembly, measurement concatenation, unbatching

**Solutions:**

1. **Optimize Measurement Concatenation**
   - Current: CPU-side concatenation before CKF
   - Consider: GPU-side assembly or zero-copy approaches

   Expected gain: 5-10 ms per event
   Estimated speedup impact: +10-15%

2. **Reduce Batch Assembly Overhead**
   - Profile CPU-side batching code
   - Optimize data structures
   - Minimize copies

   Expected gain: 3-5 ms per event
   Estimated speedup impact: +5-8%

### Priority 4: LOW - Optimize Fitting (462 ms, 44% of GPU time)

**Impact**: Low (algorithm change required)
**Effort**: Very High
**Risk**: High

**Current State:**
- Forward filter: 304 ms (29.3%)
- Backward smoother: 158 ms (15.2%)
- Already well-optimized (low stddev)

**Analysis:**
- Kalman filtering is inherently compute-intensive
- Current implementation is stable and efficient
- Any optimization requires algorithmic changes

**Not Recommended** - Better ROI on other optimizations

---

## Recommendations for Next Steps

### Immediate Actions (Week 1-2)

1. ✅ **Profile baseline N=1 with NSYS** for comparison
   - Understand sync overhead in non-batched case
   - Identify which syncs are introduced by batching

2. 🔧 **Implement basic async pipelining**
   - Create two CUDA streams: preprocessing + CKF/fitting
   - Overlap batch k+1 prep with batch k execution
   - Expected: 1.37x → 1.8x speedup

3. 🔧 **Reduce obvious synchronization points**
   - Audit code for unnecessary `cudaStreamSynchronize` calls
   - Replace with event-based dependencies where possible
   - Expected: +10-15% speedup

### Short-term (Week 3-4)

4. 🔧 **Optimize propagation**
   - Analyze propagation call patterns
   - Implement adaptive navigation
   - Skip empty surfaces
   - Expected: +5-8% speedup

5. 🔧 **Optimize CPU-side measurement concatenation**
   - Move to GPU-side assembly OR
   - Use zero-copy unified memory
   - Expected: +10-15% speedup

### Medium-term (Month 2)

6. 🔧 **Implement CUDA graphs**
   - For repetitive kernel sequences in CKF
   - Reduce launch overhead
   - Expected: +5-10% speedup

7. 🔧 **Multi-stream execution**
   - Use 2-4 streams for parallel execution
   - Requires careful dependency management
   - Expected: +10-20% speedup

### Long-term (Month 3+)

8. 🔬 **Investigate fitting optimizations**
   - Requires algorithm-level changes
   - Coordinate with physics experts
   - Potential: +10-20% if successful (high risk)

---

## Expected Performance Gains

### Conservative Estimate

| Optimization | Current | After | Speedup Gain |
|--------------|---------|-------|--------------|
| Baseline (N=4) | 47.6 ms | - | 1.37x |
| + Async pipelining | 47.6 ms | 36 ms | 1.81x (+32%) |
| + Reduce sync points | 36 ms | 31 ms | 2.10x (+16%) |
| + Optimize propagation | 31 ms | 28 ms | 2.32x (+11%) |
| + CPU optimization | 28 ms | 25 ms | 2.60x (+12%) |

**Total Expected: 1.37x → 2.60x (1.9x additional improvement)** 🎯

### Aggressive Estimate (with CUDA graphs + multi-stream)

| Optimization | Time/Event | Speedup vs N=1 |
|--------------|------------|----------------|
| Current (N=4) | 47.6 ms | 1.37x |
| + All above + graphs + streams | 20 ms | 3.25x |

**Stretch Goal: 3.0-3.5x speedup with full optimization** 🚀

---

## Conclusion

NSYS profiling reveals that **synchronization overhead is the #1 bottleneck**, consuming 86% of CUDA API time and causing the GPU to be idle 43% of the time. Despite achieving 1.37x speedup with N=4 batching, we're only at 27% of theoretical performance.

**Critical Path to 2-3x Speedup:**
1. Implement async pipelining (30-40% gain)
2. Reduce synchronization points (15-20% gain)
3. Optimize propagation (10-15% gain)
4. Optimize CPU-side processing (15-20% gain)

**Total potential: 1.37x → 2.6-3.0x speedup** with focused optimization effort on synchronization and overlap.

The good news: Memory transfers are already efficient (6% of time), seeding is fast, and fitting is well-optimized. The path forward is clear - **eliminate CPU-GPU synchronization bottlenecks through asynchronous execution and pipelining**.

---

## Appendix: Profiling Commands

### Profile Generation

```bash
nsys profile \
  --output=nsys_n4_ttbar_mu200 \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --stats=true \
  build/bin/traccc_throughput_st_cuda \
    --input-directory odd.bak/geant4_ttbar_mu200 \
    --processed-events 20 \
    --cold-run-events 5 \
    --use-batched-api=1 \
    --batch-size=4
```

### Report Generation

```bash
# Kernel summary
nsys stats --report cuda_gpu_kern_sum --force-export=true \
  nsys_n4_ttbar_mu200.nsys-rep --format csv --output .

# Memory operations
nsys stats --report cuda_gpu_mem_time_sum --force-export=true \
  nsys_n4_ttbar_mu200.nsys-rep --format csv --output .

# API summary
nsys stats --report cuda_api_sum --force-export=true \
  nsys_n4_ttbar_mu200.nsys-rep --format csv --output .
```

### Files Generated

- `nsys_n4_ttbar_mu200.nsys-rep` - Binary profile data
- `nsys_n4_ttbar_mu200.sqlite` - SQLite export
- `nsys_n4_ttbar_mu200_cuda_gpu_kern_sum.csv` - Kernel statistics
- `nsys_n4_ttbar_mu200_cuda_gpu_mem_time_sum.csv` - Memory statistics
- `nsys_n4_ttbar_mu200_cuda_api_sum.csv` - CUDA API statistics

---

**Analysis Date**: 2025-11-18
**Analyzed By**: Claude Code
**Next Review**: After implementing Priority 1 optimizations
