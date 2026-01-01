# Summary: All Surveyed Approaches for Work Redistribution

**Date:** 2026-01-01
**Status:** Analysis Complete
**Issue:** GitHub #851 - Work Redistribution

---

## The Problem

**Warp divergence in CKF propagation:** ~62% of GPU cycles wasted because 32 threads in a warp wait for the slowest thread (1-31 RK4 steps variance, 97% of warps have at least one slow thread).

---

## Approaches Summary Table

| Approach | Addresses Warp Divergence? | Implementation Effort | Overhead | Status |
|----------|---------------------------|----------------------|----------|--------|
| **Phase 1-2: Pre-Sort/Two-Phase** | No (predictor invalid) | Low | None | INVALIDATED |
| **Phase 3: Block Work-Stealing** | No (barrier blocks) | Medium | Low | NO BENEFIT |
| **Chunked Propagation** | Partially (chunk boundaries) | High | 5-13% | SLOWER |
| **Lazy Checkpointing** | Partially | Medium | 2.3% worse | SLOWER |
| **Cooperative Stepping** | Partially (stage sync) | Very High | Unknown | UNCERTAIN |
| **Warp Specialization** | Yes (if predictor exists) | Low | Low | BLOCKED |
| **Cooperative Warp** | No (sequential RK4) | N/A | N/A | IMPOSSIBLE |
| **Dynamic Parallelism** | No (launch overhead) | Medium | 5-10x work | NOT VIABLE |
| **Global Work Queue** | No (wrong problem) | Low-Medium | ~10% | MINIMAL BENEFIT |

---

## Detailed Breakdown

### FAILED/INVALIDATED Approaches

| Approach | Why It Failed |
|----------|---------------|
| **Pre-Sort by \|qop\|** | Correlation = -0.03 (none). Low momentum tracks are actually FASTER, not slower. |
| **Two-Phase (fast/slow kernels)** | No predictor exists to classify tracks. Same failure as pre-sort. |
| **Block Work-Stealing** | `__syncthreads_or()` is block-wide barrier. All threads wait at barrier, defeating work-stealing. |
| **Chunked Propagation** | Works correctly but 5-13% SLOWER. Overhead from serialization (5%), shared memory (3%), iteration tracking (2%). |
| **Lazy Checkpointing** | 2.3% slower than eager. Branch misprediction overhead exceeds serialization savings. |
| **Cooperative Warp** | RK4 stages are strictly sequential (stage N needs stage N-1). No intra-track parallelism possible. |
| **Dynamic Parallelism** | Child kernel launch (~10 us) costs ~100 RK4 steps. Max slow track is 31 steps. Overhead exceeds work. |

---

### UNCERTAIN/BLOCKED Approaches

| Approach | Potential | Blocker |
|----------|-----------|---------|
| **Warp Specialization** | Could work if predictor found | No known parameter correlates with step count. Need to test theta, eta, volume. |
| **Cooperative Stepping** | Warp-level sync cheaper than block | Requires extensive detray modifications (~1150 lines). B-field and navigation still diverge. |
| **Global Work Queue** | Fixes inter-block imbalance | Problem is intra-warp, not inter-block. All blocks have similar fast/slow mix. ~5% benefit at best. |

---

## Root Cause Analysis

```
THE FUNDAMENTAL ISSUE:

1. Propagation is SEQUENTIAL
   - RK4 stages depend on previous stage
   - Navigation depends on current position
   - Cannot parallelize within a single track

2. Track lengths are UNPREDICTABLE
   - |qop| does NOT predict step count (r = -0.03)
   - No other known predictor
   - Cannot sort or classify tracks a priori

3. Detray propagator is MONOLITHIC
   - propagator.propagate() runs to completion
   - No pause/resume at iteration level
   - Chunking requires full state serialization (~240 bytes)

4. Overhead exceeds benefit
   - Checkpoint serialization: ~5% overhead
   - Shared memory pressure: ~3% overhead
   - Work redistribution benefit: <5% (most tracks complete in 1 chunk)
   - NET: Negative ROI
```

---

## What Each Approach Targets

```
                    WARP DIVERGENCE          BLOCK IMBALANCE
                    (intra-warp wait)        (inter-block wait)
                          |                        |
                          v                        v
+---------------------------------------------------------------------+
| Pre-Sort/Two-Phase     ---------------------------> NO PREDICTOR    |
| Block Work-Stealing    ---------------------------> BARRIER BLOCKS  |
| Chunked Propagation    ---> PARTIAL (at chunk boundaries)           |
| Cooperative Stepping   ---> PARTIAL (at stage boundaries)           |
| Warp Specialization    ---> YES (if predictor found)                |
| Cooperative Warp       ---------------------------> RK4 SEQUENTIAL  |
| Dynamic Parallelism    ---------------------------> LAUNCH > WORK   |
| Global Work Queue      ---------------------------------> MINIMAL   |
+---------------------------------------------------------------------+
```

---

## Effort vs Potential Matrix

```
                        LOW EFFORT              HIGH EFFORT
                             |                       |
    +------------------------+-----------------------+
    |                        |                       |
H   |  Warp Specialization   |  Cooperative Stepping |
I   |  (if predictor found)  |  (uncertain benefit)  |
G   |  [BLOCKED]             |  [UNCERTAIN]          |
H   |                        |                       |
    +------------------------+-----------------------+
P   |                        |                       |
O   |  Global Work Queue     |  Chunked Propagation  |
T   |  (wrong problem)       |  (negative ROI)       |
E   |  [MINIMAL]             |  [FAILED]             |
N   |                        |                       |
T   +------------------------+-----------------------+
I   |                        |                       |
A   |  Pre-Sort              |  Dynamic Parallelism  |
L   |  (no predictor)        |  (overhead > work)    |
    |  [INVALIDATED]         |  [NOT VIABLE]         |
L   |                        |                       |
O   |  Block Work-Stealing   |  Cooperative Warp     |
W   |  (barrier blocks)      |  (impossible)         |
    |  [NO BENEFIT]          |  [IMPOSSIBLE]         |
    |                        |                       |
    +------------------------+-----------------------+
```

---

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Step count range | 1-31 | Measured |
| Average steps | 5.45 | Measured |
| Fast tracks (1-9 steps) | 89% | Measured |
| Slow tracks (10+ steps) | 11% | Measured |
| Warps with slow thread | 97% | Calculated |
| Wasted cycles | ~62% | Theoretical |
| Chunked overhead | 5-13% | Benchmarked |
| Checkpoint size | ~240 bytes | Design |
| Child kernel launch | ~10 us | CUDA spec |
| Single RK4 step | ~0.1 us | Estimated |

---

## Remaining Options

### 1. Find a Predictor (Low Effort, High Potential)

```
Action: Instrument codebase to collect (theta, eta, volume, step_count)
Goal: Find any parameter with |correlation| > 0.3
If found: Warp specialization becomes viable
Status: NOT YET ATTEMPTED
```

### 2. Accept the Divergence (Zero Effort)

```
Action: None
Rationale: 62% wasted cycles is the cost of random track lengths
Trade-off: Simplicity vs performance
Status: CURRENT STATE
```

### 3. Upstream Detray Changes (Very High Effort, Uncertain)

```
Action: Propose incremental propagation API to detray maintainers
Changes: step_iteration(), checkpoint(), restore() methods
Benefit: Enable true iteration-level work redistribution
Risk: May still have overhead > benefit
Status: NOT ATTEMPTED
```

---

## Conclusion

**No viable solution found.** The problem is fundamentally constrained by:

1. **Sequential algorithm** - RK4 cannot be parallelized
2. **Unpredictable workload** - No predictor for step count
3. **Overhead exceeds benefit** - Any redistribution mechanism costs more than it saves

**Best remaining path:** Search for a step count predictor (theta, eta, starting volume). If found, warp specialization offers the best effort-to-benefit ratio.

---

## Related Documents

- `doc/problem_definition.md` - Warp divergence analysis and measurements
- `doc/work_redistribution_plan.md` - Phase 1-3 implementation plans
- `doc/work_redistribution_plan_phase3.md` - Persistent threads design
- `doc/issue_of_redesign_propagator.md` - Work-stealing pattern limitations
- `doc/chunked_propagator_redesign_plan.md` - Detray architecture survey
- `doc/chunked_propagator_redesign_implementation.md` - Chunked propagation implementation
- `doc/chunk_size_finding.md` - Chunk size vs precision analysis
- `doc/issue_of_chunked_propagator.md` - Performance overhead analysis (5-13%)

---

*This document summarizes all approaches investigated for GitHub Issue #851.*
