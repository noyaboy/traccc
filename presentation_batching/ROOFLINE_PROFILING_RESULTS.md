# Roofline & Performance Profiling Results

**Date:** 2025-11-18
**Configuration:** Baseline (with theta-based sorting)
**Objective:** Identify true bottleneck (compute, bandwidth, or latency)

---

## Executive Summary

### Critical Finding: Kernel is Severely Latency-Bound

The `propagate_to_next_surface` kernel is **severely latency-bound**, not bandwidth or compute bound:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **SM Throughput** | 9.29% | ❌ Very low compute utilization |
| **DRAM Throughput** | 42.04% | ⚠️ Moderate memory utilization |
| **Memory Throughput** | 42.04% | ⚠️ Not bandwidth-saturated |
| **L1/TEX Cache** | 16.07% | ❌ Very low cache utilization |
| **L2 Cache** | 29.24% | ⚠️ Moderate cache utilization |
| **FP32 Peak Performance** | 1% | ❌ Critically low |

**NCU Assessment:** *"This kernel exhibits low compute throughput and memory bandwidth utilization... below 60.0% of peak typically indicate **latency issues**."*

---

## Roofline Analysis Details

### GPU Speed Of Light Metrics (Step 0)

```
Section: GPU Speed Of Light Throughput
----------------------- ------------- -------------
Metric Name               Metric Unit  Metric Value
----------------------- ------------- -------------
DRAM Frequency          cycle/nsecond          6.78
SM Frequency            cycle/nsecond          1.30
Elapsed Cycles                  cycle    12,429,982
Memory Throughput                   %         42.04
DRAM Throughput                     %         42.04
Duration                      msecond          9.45
L1/TEX Cache Throughput             %         16.07
L2 Cache Throughput                 %         29.24
SM Active Cycles                cycle 11,812,919.78
Compute (SM) Throughput             %          9.29
----------------------- ------------- -------------
```

### Performance Characterization

**The kernel achieved only 1% of fp32 peak performance** - this is catastrophically low and indicates the GPU is mostly idle waiting for data/dependencies.

**Roofline Interpretation:**
- **NOT compute-bound**: SM throughput at 9.29% means GPU compute units are idle 90% of the time
- **NOT bandwidth-bound**: DRAM throughput at 42% means memory subsystem can deliver more data
- **IS latency-bound**: Low utilization of both compute and memory indicates stalls/waits

---

## What Causes Latency-Bound Performance?

### 1. Cache Miss Latency

**L1/TEX Cache: 16.07%** - extremely low utilization
- Cache hit rate is poor despite theta-based sorting
- Most memory accesses miss L1 and go to L2/DRAM
- Each L1 miss costs ~80-100 cycles of latency

**L2 Cache: 29.24%** - moderate utilization
- Better than L1 but still indicates cache pressure
- Theta sorting helps here (vs 30.13% in Phase 0 without sorting)

### 2. Memory Access Patterns

From baseline coalescing metrics:
- Load efficiency: **16.75%** → 83% of loaded data is wasted
- Store efficiency: **25.83%** → 74% of stored data is wasted

**Effect on latency:**
- Poor coalescing → More memory transactions
- More transactions → Longer memory queue latency
- Wasted bandwidth → Memory controller stalls

### 3. Warp Divergence

From earlier metrics:
- Warp active: **28.01%** of peak sustained active
- This means warps are idle 72% of the time

**Likely causes:**
- Different track lengths (barrel vs endcap)
- Varying material interactions (different detector regions)
- Branching in propagation logic (surface intersection tests)

**Effect:** When threads diverge, some threads idle while others execute → low warp occupancy → latency-bound

### 4. Data Dependencies

The track propagation is inherently sequential:
1. Read track parameters from step N
2. Propagate to next surface (compute-intensive)
3. Update track parameters
4. Write back for step N+1

**Chain of dependencies:**
```
Read params[i] → Compute propagation → Write params[i+1]
     ↓                                        ↓
  L1 miss (latency!)              Store (wait for L2/DRAM)
```

Each step must wait for previous step → pipeline bubbles → latency-bound

---

## Comparison: Baseline vs Phase 0

| Metric | Baseline (with sorting) | Phase 0 (no sorting) | Interpretation |
|--------|-------------------------|----------------------|----------------|
| **DRAM Throughput** | 38.63% avg | 30.13% avg | Sorting improves cache locality |
| **Load Efficiency** | 16.75% | 15.14% | Sorting doesn't hurt coalescing much |
| **SM Throughput** | 9.46% avg | N/A | Very low in both cases |
| **Performance** | 47.6 ms/event | 59.50 ms/event | Sorting 25% faster |

**Key insight:** Theta-based sorting helps with cache locality (38.63% vs 30.13% DRAM throughput) but doesn't solve the fundamental latency-bound problem.

---

## Root Causes of Latency Issues

Based on roofline + coalescing analysis:

### Primary Bottlenecks

1. **Poor Memory Coalescing (16.75% load efficiency)**
   - AoS layout (120-byte `bound_track_parameters` structs)
   - Random access via `links` array
   - Random measurement access patterns
   - **Impact:** 5-6x more memory transactions → queue latency

2. **Cache Thrashing (16.07% L1 utilization)**
   - Large working set (168k tracks × 120 bytes = 20MB)
   - Exceeds L2 cache (5.5MB on RTX 2080 Ti)
   - Frequent evictions → cache miss latency
   - **Impact:** ~80-100 cycle latency per L1 miss

3. **Warp Divergence (28% warp active)**
   - Mixed detector regions in same warp
   - Different branch execution paths
   - Variable-length track processing
   - **Impact:** Threads idle waiting for divergent paths

4. **Sequential Dependencies**
   - Each CKF step depends on previous step
   - Cannot overlap compute with memory
   - Pipeline stalls
   - **Impact:** Limits instruction-level parallelism

### Secondary Issues

5. **Low Arithmetic Intensity**
   - Propagation is memory-heavy, compute-light
   - Lots of loads/stores, minimal FLOPs
   - **Result:** 1% of fp32 peak performance

6. **Register Pressure**
   - Complex propagation logic requires many registers
   - May limit occupancy
   - Fewer warps in flight → less latency hiding

---

## Optimization Strategy

Given that the kernel is **latency-bound**, optimizations must focus on:

### Priority 1: Reduce Memory Latency

**A. Improve Memory Coalescing (Target: 80%+ load efficiency)**

Option 1: **Struct-of-Arrays (SoA) Transformation**
- Convert `bound_track_parameters` from AoS to SoA
- **Expected impact:** 16.75% → 80%+ load efficiency
- **Performance gain:** ~30-40% (reduces memory latency by 4-5x)
- **Effort:** High (major refactoring)

Option 2: **Physically Sort Data Instead of Indices**
- Sort `params` array by theta, not `param_ids`
- Eliminates indirection, improves coalescing
- **Expected impact:** 16.75% → 50-60% load efficiency
- **Performance gain:** ~15-20%
- **Effort:** Medium

**B. Improve Cache Hit Rate**

Option 3: **Reduce Working Set Size**
- Process smaller batches of tracks
- Keep active working set < L2 cache size (5.5MB)
- **Expected impact:** Better L1/L2 utilization
- **Performance gain:** ~10-15%
- **Effort:** Low (adjust batch size)

### Priority 2: Reduce Warp Divergence

**C. Improve Workload Balance**

Option 4: **Sort by Track Length + Theta**
- Group tracks with similar lengths together
- Reduces divergence within warps
- **Expected impact:** 28% → 40-50% warp active
- **Performance gain:** ~10-15%
- **Effort:** Medium

### Priority 3: Overlap Compute with Memory

**D. Kernel Fusion**

Option 5: **Fuse Propagation Steps**
- Process multiple CKF steps in single kernel launch
- Overlap memory transfers with compute
- **Expected impact:** Better pipeline utilization
- **Performance gain:** ~5-10%
- **Effort:** High

---

## Recommended Path Forward

### Phase 1: Quick Wins (1-2 weeks)

1. **Reduce batch size** to fit in L2 cache
   - Test N=1, N=2 batch sizes
   - Measure cache hit rate improvement
   - **Expected:** 10-15% speedup, low effort

2. **Profile warp divergence** in detail
   - Use `nvprof` or Nsight Compute warp state metrics
   - Identify hotspot divergent branches
   - **Goal:** Quantify divergence impact

### Phase 2: Major Optimization (4-6 weeks)

3. **Implement SoA transformation**
   - Start with `bound_track_parameters` only
   - Measure coalescing improvement
   - **Expected:** 30-40% speedup, high effort
   - **Risk:** Major API changes, testing burden

4. **Implement hybrid sorting** (if SoA not feasible)
   - Physically sort params array by theta
   - Track seed_idx through sorting
   - **Expected:** 15-20% speedup, medium effort

### Phase 3: Advanced Optimization (future)

5. **Kernel fusion** for pipelining
6. **Asynchronous memory transfers**
7. **Multi-stream execution**

---

## Conclusion

### Key Findings

1. **Kernel is severely latency-bound**
   - SM throughput: 9.29% (not compute-bound)
   - DRAM throughput: 42.04% (not bandwidth-bound)
   - Achieved 1% of fp32 peak (latency stalls dominate)

2. **Primary bottleneck: Poor memory coalescing**
   - 16.75% load efficiency → 5-6x too many transactions
   - Causes memory queue latency
   - AoS layout + random access patterns are root cause

3. **Secondary bottlenecks:**
   - Cache thrashing (16.07% L1 utilization)
   - Warp divergence (28% warp active)
   - Sequential dependencies (CKF step-by-step)

4. **Theta-based sorting helps but doesn't solve root problem**
   - Improves cache locality (38.63% vs 30.13% DRAM throughput)
   - Provides 25% speedup
   - But leaves kernel latency-bound

### Path Forward

**Immediate:** Keep theta-based sorting (validated by Phase 0)

**Short-term:** Reduce batch size to improve cache utilization

**Medium-term:** Implement SoA transformation to fix coalescing (30-40% expected gain)

**Long-term:** Consider kernel fusion to overlap compute with memory

---

## Data Sources

- **Roofline Analysis:** `/home/noah/project/traccc_v0_26_0/ncu_roofline_baseline.log`
- **Baseline Coalescing:** `/home/noah/project/traccc_v0_26_0/ncu_propagate_output.log`
- **Phase 0 Comparison:** `/home/noah/project/traccc/PHASE0_BASELINE_COMPARISON.md`
- **Phase 0 Results:** `/home/noah/project/traccc/PHASE0_MEASUREMENT_RESULTS.md`

---

**Status:** Roofline analysis complete. Latency-bound performance confirmed. SoA transformation is highest-priority optimization.
