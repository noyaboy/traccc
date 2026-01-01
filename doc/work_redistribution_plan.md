# Work Redistribution Implementation Plan

**GitHub Issue:** [#851](https://github.com/acts-project/traccc/issues/851)
**Branch:** `feature/work-redistribution`
**Author:** Generated with Claude Code
**Date:** 2025-12-30

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [**CRITICAL: Correlation Analysis Results**](#2-critical-correlation-analysis-results)
3. [Problem Recap](#3-problem-recap)
4. [Implementation Strategy](#4-implementation-strategy)
5. [Phase 1: Pre-Sort Enhancement](#5-phase-1-pre-sort-enhancement) ⚠️ INVALIDATED
6. [Phase 2: Two-Phase Kernel Separation](#6-phase-2-two-phase-kernel-separation) ⚠️ INVALIDATED
7. [Phase 3: Persistent Threads (Recommended)](#7-phase-3-persistent-threads-recommended)
8. [Phase 4: Cooperative Stepping (Future)](#8-phase-4-cooperative-stepping-future)
9. [Testing Strategy](#9-testing-strategy)
10. [Benchmarking Plan](#10-benchmarking-plan)
11. [Risk Assessment](#11-risk-assessment)
12. [Timeline](#12-timeline)
13. [Appendix: Code References](#13-appendix-code-references)

---

## 1. Executive Summary

### Goal
Reduce warp divergence in the CKF propagation kernel from ~62% wasted cycles to <30% wasted cycles by implementing work redistribution strategies.

### Approach
Phased implementation starting with low-complexity, high-impact changes:

| Phase | Strategy | Expected Gain | Complexity | Status |
|-------|----------|---------------|------------|--------|
| 1 | Pre-Sort Enhancement | ~~10-20%~~ | LOW | ❌ **INVALIDATED** |
| 2 | Two-Phase Separation | ~~20-40%~~ | MEDIUM | ❌ **INVALIDATED** |
| 3 | Persistent Threads | 30-50% | HIGH | ✅ **RECOMMENDED** |
| 4 | Cooperative Stepping | +10-20% | VERY HIGH | Future work |

> **⚠️ UPDATE (v2.0):** Empirical correlation analysis shows |qop| does NOT predict
> step count. Phases 1-2 are invalidated. **Proceed directly to Phase 3.**

### Success Criteria
- Throughput improvement of >25% on ODD detector, ttbar_mu200 dataset
- No regression in track finding efficiency
- Validated correctness via comparison with baseline

---

## 2. CRITICAL: Correlation Analysis Results

> **⚠️ WARNING: The original assumptions of this plan have been invalidated by empirical data.**
>
> Instrumentation added on 2025-12-30 reveals that |qop| does NOT correlate with RK step count.
> **Phase 1 and Phase 2 strategies based on qop sorting/separation will NOT be effective.**

### 2.1 Measured Correlation

Instrumentation was added to record `(|qop|, step_count)` pairs during propagation.
Results from ODD detector with ttbar_mu200 dataset (5 events, ~94,000 propagations):

```
Pearson correlation (|qop| vs steps): -0.028 to -0.031
Interpretation: NO significant correlation
```

### 2.2 Average Steps by |qop| Bucket

| |qop| Range | Track Count | Avg Steps | Observation |
|-------------|-------------|-----------|-------------|
| [0, 0.1) | 10,136 | 5.63 | High momentum |
| [0.1, 0.2) | 20,221 | 5.84 | |
| [0.2, 0.3) | 15,584 | 5.68 | |
| [0.3, 0.4) | 9,451 | 5.76 | |
| [0.4, 0.5) | 6,707 | 5.91 | |
| [0.5, 0.6) | 5,347 | 5.82 | |
| [0.6, 0.7) | 4,449 | 5.71 | |
| [0.7, 0.8) | 3,465 | 5.63 | |
| [0.8, 0.9) | 2,914 | 5.44 | |
| [0.9, 1.0) | 2,470 | 5.37 | |
| **[1.0+)** | **14,088** | **3.60** | **Low momentum = FEWEST steps!** |

### 2.3 Key Findings

1. **Correlation is essentially ZERO** (-0.03) - |qop| does NOT predict step count
2. **The trend is OPPOSITE to the original assumption**:
   - Original assumption: Higher |qop| (lower momentum) → more RK steps
   - **Actual data: Higher |qop| → FEWER steps**
3. Low |qop| (high momentum) tracks average ~5.6-5.9 steps
4. High |qop| (low momentum, |qop|≥1) tracks average only ~3.5 steps

### 2.4 Why the Assumption Was Wrong

The original assumption was based on the idea that lower momentum tracks curve more in
the magnetic field, requiring more RK steps to maintain accuracy. However, the data shows:

- **Low momentum tracks may exit the detector volume sooner** (shorter paths)
- **High momentum tracks traverse more material** (longer paths through detector)
- **The dominant factor is path length, not curvature**

### 2.5 Impact on Implementation Strategy

| Phase | Original Strategy | Status | Recommendation |
|-------|-------------------|--------|----------------|
| 1 | Pre-sort by |qop| | ❌ **INVALIDATED** | Skip - no benefit |
| 2 | Two-phase by |qop| threshold | ❌ **INVALIDATED** | Skip - no benefit |
| 3 | Persistent threads | ✅ **STILL VALID** | **Recommended** - physics-agnostic |
| 4 | Cooperative stepping | ✅ **STILL VALID** | Future work |

### 2.6 Alternative Predictors to Explore

If pre-sorting is still desired, alternative predictors should be investigated:

1. **Theta angle** - tracks at small angles may traverse more layers
2. **Initial surface index** - geometry-based prediction
3. **Historical step counts** - use previous CKF iteration's step count
4. **Surface-to-surface distance** - geometric distance to next expected surface

However, **Phase 3 (Persistent Threads)** is now the recommended approach as it provides
load balancing without requiring prediction of step counts.

---

## 3. Problem Recap

### Measured Warp Divergence (from `doc/problem_definition.md`)

```
Step Count Distribution (94,208 propagations):
  [1-9]:   84,251 (89.43%)  ← Fast
  [10-19]:  9,744 (10.34%)  ← Medium
  [20-29]:    213 (0.23%)   ← Slow

Statistics: min=1, max=31, avg=5.45 steps
```

### Why Divergence is Severe

With 32 threads per warp:
- P(all 32 threads ≤9 steps) = 0.89³² = **2.8%**
- P(at least one thread ≥10 steps) = **97.2%**
- Estimated wasted cycles: **~62%**

### Root Cause

Adaptive RK4 stepping in detray (`rk_stepper.ipp:772-801`) adjusts step size based on:
- Track momentum (qop)
- Magnetic field strength
- Material density
- Numerical error tolerance

---

## 4. Implementation Strategy

### Architecture Overview

```
Current Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│ apply_interaction → find_tracks → sort_keys → propagate → ...  │
│                                       ↑                         │
│                              Simple 1-to-1 mapping              │
│                              128 threads/block                  │
│                              HIGH DIVERGENCE                    │
└─────────────────────────────────────────────────────────────────┘

Target Pipeline (Phase 3 - Persistent Threads):
┌─────────────────────────────────────────────────────────────────┐
│ apply_interaction → find_tracks → sort_keys → WORK_QUEUE →     │
│                                                    ↓            │
│                                         Persistent threads      │
│                                         Dynamic load balancing  │
│                                         Work-stealing pattern   │
│                              REDUCED DIVERGENCE                 │
└─────────────────────────────────────────────────────────────────┘

Note: Original PHASE_1 (sort) and PHASE_2 (two-phase) plans invalidated.
See Section 2 for correlation analysis results.
```

### File Modification Summary

> **Note:** Phases 1-2 invalidated. Only Phase 3 modifications are planned.

| File | Phase | Changes |
|------|-------|---------|
| `device/cuda/src/finding/combinatorial_kalman_filter.cuh` | 3 | Persistent thread launch |
| `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | 3 | Work queue logic |
| `device/common/include/traccc/finding/device/propagate_to_next_surface.hpp` | 3 | Payload extensions for work queue |
| `core/include/traccc/finding/finding_config.hpp` | 3 | Config parameters for persistent threads |

---

## 5. Phase 1: Pre-Sort Enhancement

> **⚠️ INVALIDATED: See [Section 2](#2-critical-correlation-analysis-results)**
>
> Empirical data shows |qop| does NOT correlate with step count (r = -0.03).
> This phase will NOT provide the expected benefits. **Skip to Phase 3.**

### 5.1 Objective (Original - Now Invalid)

Sort tracks by predicted step count to improve warp coherence. Tracks with similar expected step counts will be grouped together, reducing intra-warp divergence.

### 5.2 Current Implementation

**File:** `device/common/include/traccc/edm/device/sort_key.hpp`
**Lines:** 16-23

```cpp
// Current: Sort by theta deviation from horizontal
template <detray::concepts::algebra algebra_t>
TRACCC_HOST_DEVICE inline sort_key get_sort_key(
    const bound_track_parameters<algebra_t>& params) {
    return math::fabs(params.theta() - constant<traccc::scalar>::pi_2);
}
```

### 5.3 Proposed Changes (DO NOT IMPLEMENT)

> **🚫 DO NOT IMPLEMENT: The code below is based on invalidated assumptions.**
> Empirical data shows |qop| does NOT predict step count (r = -0.03).

#### 5.3.1 New Sort Key Function (INVALID)

~~**File:** `device/common/include/traccc/edm/device/sort_key.hpp`~~

```cpp
// ❌ DO NOT IMPLEMENT - ASSUMPTION PROVEN WRONG
// Original assumption: "Lower |qop| = fewer RK steps"
// Actual finding: NO correlation (r = -0.03), trend is OPPOSITE if anything
//
// Data shows:
//   |qop| < 0.5 (high momentum): avg 5.8 steps
//   |qop| >= 1.0 (low momentum): avg 3.6 steps  <-- FEWER, not more!
```

#### 5.3.2 Configuration Parameters (NOT NEEDED)

Since qop-based sorting provides no benefit, these configuration parameters are not needed.

### 5.4 Why This Approach Failed

The original hypothesis was:
- Lower momentum tracks curve more in B-field → more RK steps needed

The actual data shows:
- Low momentum tracks exit detector sooner → shorter paths → fewer steps
- High momentum tracks traverse more material → longer paths → more steps
- **Path length dominates over curvature effects**

---

## 6. Phase 2: Two-Phase Kernel Separation

> **⚠️ INVALIDATED: See [Section 2](#2-critical-correlation-analysis-results)**
>
> The qop-based classification assumes high |qop| = more steps, but data shows the OPPOSITE.
> Low momentum tracks (|qop| ≥ 1) average only 3.6 steps vs 5.8 for high momentum.
> **This phase will NOT provide the expected benefits. Skip to Phase 3.**

### 6.1 Objective (Original - Now Invalid)

Separate tracks into "fast" (high momentum, 1-9 steps) and "slow" (low momentum, 10+ steps) categories, launching separate kernels for each to maximize warp utilization.

> **❌ INCORRECT ASSUMPTION:** The objective assumed high momentum = fewer steps.
> **Actual data:** Low momentum (|qop| ≥ 1) averages 3.6 steps, high momentum averages 5.8 steps.

### 6.2 Design (DO NOT IMPLEMENT)

> **🚫 DO NOT IMPLEMENT: The design below is based on invalidated assumptions.**

The original design assumed:
- High momentum (low |qop|) → fast (1-9 steps) → 89% of tracks
- Low momentum (high |qop|) → slow (10+ steps) → 11% of tracks

**Actual data shows the OPPOSITE trend:**
- High momentum (low |qop|): avg 5.8 steps
- Low momentum (high |qop| ≥ 1): avg 3.6 steps ← actually FASTER!

Any qop-based partitioning would group tracks incorrectly.

### 6.3 Implementation Details (ARCHIVED - DO NOT USE)

> **🚫 The code examples below are archived for reference only. DO NOT IMPLEMENT.**
>
> The classification logic `(qop_abs < threshold) ? fast : slow` is INVERTED
> from what the data shows. Implementing this would make performance WORSE.

```cpp
// ❌ DO NOT IMPLEMENT
// This classification is WRONG based on empirical data:
//
// Original logic: low |qop| = fast, high |qop| = slow
// Actual data:    low |qop| = MORE steps, high |qop| = FEWER steps
//
// The assumption that momentum predicts step count is INVALID.
```

### 6.4 Why This Approach Failed

The two-phase approach requires a reliable predictor to classify tracks into "fast" and "slow" groups. The original plan used |qop| (momentum), but:

1. **No correlation exists** between |qop| and step count (r = -0.03)
2. **The trend is opposite** to what was assumed
3. **Any qop-based partitioning would be random** with respect to actual step counts

Without a reliable predictor, two-phase separation cannot improve warp efficiency.

### 6.5 Alternative Approaches (Future Investigation)

If a two-phase approach is still desired, alternative predictors could be explored:

1. **Historical step counts** - use step count from previous CKF iteration
2. **Surface geometry** - distance to next expected surface
3. **Detector region** - different regions may have different step count distributions

However, **Phase 3 (Persistent Threads)** is recommended as it provides load balancing
without requiring prediction of step counts.

---

## 7. Phase 3: Persistent Threads (RECOMMENDED)

> **✅ NOW THE PRIMARY RECOMMENDED APPROACH**
>
> Since |qop| does not predict step count, physics-based sorting is ineffective.
> Persistent threads provide load balancing WITHOUT requiring prediction of work.
> **This is now the recommended first implementation.**

### 7.1 Objective

Implement a work-stealing queue pattern (similar to `find_tracks.ipp`) for propagation to achieve near-perfect load balancing.

### 7.2 Why This Is Now Recommended

- **Physics-agnostic**: Does not require predicting step counts from track parameters
- **Proven pattern**: Already successfully used in `find_tracks.ipp`
- **Dynamic load balancing**: Adapts to actual work distribution at runtime
- **Expected gain**: 30-50% improvement in warp efficiency

### 7.3 Reference Implementation

**File:** `device/common/include/traccc/finding/device/impl/find_tracks.ipp`
**Lines:** 141-498

Key patterns to adapt:
1. Shared buffer for work items (lines 152-168)
2. Atomic push/pop via `vecmem::device_atomic_ref` (line 159)
3. Barrier synchronization via `barrier.blockBarrier()` (lines 170, 478)
4. Convergence detection via `barrier.blockOr()` (lines 141, 262)

### 7.4 Proposed Work Queue Structure

```cpp
/// Work item for propagation queue
struct propagation_work_item {
    unsigned int param_id;       // Track parameter index
    unsigned int link_idx;       // Link index for this track
    unsigned int step;           // Current CKF step
    unsigned int status;         // 0=pending, 1=in_progress, 2=complete
};

/// Shared memory layout for persistent thread block
struct propagation_shared_state {
    // Work queue (2x block size for overflow)
    propagation_work_item queue[256];

    // Queue management
    unsigned int queue_head;      // Atomic: next item to dequeue
    unsigned int queue_tail;      // Atomic: next slot for enqueue
    unsigned int active_count;    // Atomic: threads currently working

    // Completion tracking
    unsigned int completed_count; // Atomic: finished propagations
};
```

### 7.5 Implementation Outline

> **Note:** This code uses traccc's abstraction layer for portability:
> - `barrier.blockBarrier()` instead of raw `__syncthreads()`
> - `vecmem::device_atomic_ref` instead of raw `atomicAdd/atomicSub`
> - See `device/cuda/src/utils/barrier.hpp` for barrier implementation

```cpp
template <typename barrier_t>
TRACCC_DEVICE void propagate_persistent(
    const global_index_t globalIndex,
    const propagate_to_next_surface_payload<...>& payload,
    propagation_shared_state& shared,
    barrier_t& barrier) {

    // Initialize queue with work items (once per block)
    if (threadIdx.x == 0) {
        for (unsigned int i = blockIdx.x * blockDim.x;
             i < min((blockIdx.x + 1) * blockDim.x, payload.n_in_params);
             ++i) {
            shared.queue[i - blockIdx.x * blockDim.x] = {
                .param_id = param_ids.at(i),
                .link_idx = payload.prev_links_idx + i,
                .step = payload.step,
                .status = 0
            };
        }
        shared.queue_tail = min(blockDim.x, payload.n_in_params - blockIdx.x * blockDim.x);
        shared.queue_head = 0;
    }
    barrier.blockBarrier();  // Use traccc barrier abstraction

    // Persistent work loop using barrier.blockOr() for collective termination
    // Pattern adapted from find_tracks.ipp:141,262 - NO spin-wait
    bool has_work = true;
    while (barrier.blockOr(has_work)) {
        has_work = false;

        // Try to dequeue work item using vecmem atomic
        unsigned int my_idx =
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared.queue_head).fetch_add(1u);

        if (my_idx < shared.queue_tail) {
            has_work = true;  // Signal we have work for next iteration check

            // Process work item
            propagation_work_item& item = shared.queue[my_idx];

            // ... existing propagation logic from propagate_to_next_surface.ipp ...

            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared.completed_count).fetch_add(1u);
        }
        // Threads without work vote false in blockOr and wait at barrier
        // Loop exits when ALL threads have no work (blockOr returns false)
    }
}
```

### 7.6 Memory Requirements

| Component | Size per Block |
|-----------|---------------|
| Work queue (256 items) | 4 KB |
| Queue pointers | 16 bytes |
| Counters | 12 bytes |
| **Total** | ~4.1 KB |

**Validation (2025-12-31):** Cross-validated against codebase:
- Modern GPUs provide 48-96 KB shared memory per block
- Existing `find_tracks` kernel uses ~1.5 KB shared memory (lines 353-355)
- `propagate_to_next_surface` currently uses 0 bytes shared memory (line 515)
- GPU shared memory limits can be queried at runtime (see `gbts_seeding_algorithm.cu:784`)
- **4.1 KB is well within limits** with substantial headroom

---

## 8. Phase 4: Cooperative Stepping (Future)

### 8.1 Objective

Enable warp-level cooperation within RK4 stepping by exposing intermediate state and checkpoint/resume capabilities in detray.

### 8.2 Prerequisites

- Phase 3 implemented and benchmarked (Phases 1-2 invalidated)
- Profiling confirms RK4 computation is the bottleneck (not memory/B-field)
- Detray fork (`/dicos_ui_home/noah/detray-fork`) modifications approved

### 8.3 Detray Modifications Required

> **CRITICAL: Current Architecture Limitation**
>
> The `intermediate_state` struct is currently a **LOCAL VARIABLE** inside the `step()`
> function (`rk_stepper.ipp:670`), NOT a class member. This means:
> - There is NO existing `m_sd` member to expose
> - Significant refactoring is required to make state accessible
> - This is a ~200-300 line change to detray, not just adding accessors

#### 8.3.1 Expose Intermediate State (Requires Refactoring)

**Step 1: Add member variable to rk_stepper class**

**File:** `detray-fork/core/include/detray/propagator/rk_stepper.hpp`

```cpp
// In the rk_stepper class, add as PRIVATE member:
private:
    /// Intermediate state for RK4 stages (promoted from local in step())
    /// This enables checkpointing and cooperative stepping
    mutable intermediate_state m_sd{};

public:
    /// Get intermediate state for checkpointing
    DETRAY_HOST_DEVICE const intermediate_state& get_intermediate_state() const {
        return m_sd;
    }

    /// Set intermediate state for resume
    DETRAY_HOST_DEVICE void set_intermediate_state(const intermediate_state& sd) {
        m_sd = sd;
    }
```

**Step 2: Modify step() function to use member instead of local**

**File:** `detray-fork/core/include/detray/propagator/rk_stepper.ipp`

```cpp
// BEFORE (line 670):
intermediate_state sd{};

// AFTER:
// Remove local variable, use m_sd member instead
// All references to 'sd' must be changed to 'm_sd'
// This affects approximately 50+ locations in the step() function
```

> **Impact Assessment:**
> - Lines affected in rk_stepper.ipp: ~50-80 (all `sd.` references)
> - Lines affected in rk_stepper.hpp: ~20 (member + accessors)
> - Testing required: Full propagation validation suite
> - Risk: Medium (changes core integration algorithm)

#### 8.3.2 Partial Step API

**File:** `detray-fork/core/include/detray/propagator/rk_stepper.ipp`

```cpp
/// Execute single RK stage (for cooperative execution)
/// @param stage_id Stage number (0-3)
/// @return true if stage completed successfully
DETRAY_HOST_DEVICE bool step_single_stage(
    const unsigned int stage_id,
    const scalar_type h,
    state& stepping,
    const stepping::config& cfg);

/// Compute error estimate from current intermediate state
DETRAY_HOST_DEVICE scalar_type compute_error_from_state(
    const intermediate_state& sd,
    const scalar_type h) const;
```

### 8.4 Warp-Level Implementation

```cpp
/// Warp-cooperative RK4 stepping
/// Each warp processes tracks cooperatively
TRACCC_DEVICE void cooperative_rk_step(
    const unsigned int lane_id,      // 0-31 within warp
    const unsigned int warp_id,      // Warp index in block
    intermediate_state* warp_states, // Shared memory array [32]
    /* ... */) {

    // Each lane owns one track
    intermediate_state& my_state = warp_states[lane_id];

    // Stage 1: All lanes compute independently
    compute_rk_stage_1(my_state);
    __syncwarp();

    // Error estimation: warp-level reduction
    float my_error = estimate_error(my_state);
    float max_error = __reduce_max_sync(0xFFFFFFFF, my_error);

    // Adaptive step sizing: warp consensus
    if (max_error > tolerance) {
        // All lanes reduce step size together
        float scale = compute_scale(max_error);
        my_state.h *= scale;
    }
    __syncwarp();

    // Continue with stages 2-4...
}
```

### 8.5 Expected Complexity

| Component | Lines of Code | Risk |
|-----------|--------------|------|
| Detray state exposure | ~50 | Low |
| Partial step API | ~200 | Medium |
| Warp-level kernel | ~400 | High |
| Testing/validation | ~500 | Medium |
| **Total** | ~1150 | Medium-High |

---

## 9. Testing Strategy

### 9.1 Unit Tests

> **Note:** Phase 1-2 tests removed (invalidated). Tests focus on Phase 3.

| Test | Description | Location |
|------|-------------|----------|
| `test_work_queue` | Verify work queue push/pop | `tests/cuda/test_ckf_work_queue.cpp` |
| `test_persistent_threads` | Verify persistent thread convergence | `tests/cuda/test_ckf_persistent.cpp` |
| `test_load_balancing` | Verify work distribution | `tests/cuda/test_ckf_load_balance.cpp` |

### 9.2 Integration Tests

```bash
# Run CKF with work redistribution on ODD detector
./bin/traccc_throughput_mt_cuda \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --input-directory=odd/geant4_ttbar_mu200/ \
  --input-events=36 --processed-events=500 --cpu-threads=1
```

### 9.3 Correctness Validation

Compare track finding results between:
1. Baseline (no work redistribution)
2. Phase 3 (persistent threads)

Metrics to compare:
- Number of tracks found
- Track parameter distributions (pT, eta, phi)
- Chi-square distributions
- Efficiency vs. purity

### 9.4 Regression Tests

Add to CI pipeline:
```yaml
- name: Work Redistribution Correctness
  run: |
    ./bin/traccc_ckf_cuda_test --gtest_filter="*WorkRedistribution*"
    diff baseline_tracks.root optimized_tracks.root
```

---

## 10. Benchmarking Plan

### 10.1 Baseline Measurement

```bash
# Establish baseline performance
./bin/traccc_throughput_mt_cuda \
  --detector-file=... \
  --input-events=36 --processed-events=1000 \
  --cpu-threads=1,4,7,8
```

Record:
- Events/second at each thread count
- GPU utilization
- Memory bandwidth usage

### 10.2 Instrumentation Metrics

Enable step count histogram (already implemented):
```
=== STEP COUNT HISTOGRAM ===
Total propagations: N
Min/Max/Avg steps: X/Y/Z
Distribution by bucket...
```

### 10.3 Profiling Tools

```bash
# Nsight Compute (requires admin permissions)
ncu --set full ./bin/traccc_throughput_st_cuda ...

# If Nsight blocked, use nvprof
nvprof --metrics achieved_occupancy,warp_execution_efficiency \
  ./bin/traccc_throughput_st_cuda ...
```

### 10.4 Performance Targets

> **Note:** Phases 1-2 invalidated. Targets focus on Phase 3 (persistent threads).
>
> **Update (2025-12-31):** Added conservative estimates based on atomic overhead analysis.

| Metric | Baseline | Phase 3 (Conservative) | Phase 3 (Optimistic) | Phase 3+4 |
|--------|----------|------------------------|----------------------|-----------|
| Events/s (1 thread) | 57.82 | 69+ | 75+ | 85+ |
| Warp efficiency | ~38% | ~53% | ~70% | ~80% |
| Wasted cycles | ~62% | ~47% | ~30% | ~20% |
| Improvement | - | 20-30% | 30-50% | 40-60% |

**Conservative vs Optimistic:**
- **Conservative (20-30%)**: Accounts for atomic queue overhead, synchronization costs
- **Optimistic (30-50%)**: Assumes near-perfect work stealing with minimal contention

---

## 11. Risk Assessment

### 11.1 Technical Risks

> **Note:** Risks updated after correlation analysis invalidated Phases 1-2.

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ~~Sort key correlation weak~~ | ~~Medium~~ | ~~Low~~ | **CONFIRMED: r=-0.03, no correlation** |
| Work queue overhead > benefit | Low | Medium | Profile atomic operations |
| Shared memory pressure | Medium | Medium | Optimize queue size |
| Memory bandwidth bottleneck | Medium | High | Profile before Phase 4 |
| Correctness regression | Low | Critical | Extensive validation tests |

### 11.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase 3 complexity underestimated | Medium | Medium | Incremental implementation |
| Work-stealing pattern bugs | Medium | High | Reference find_tracks.ipp implementation |
| Detray changes rejected (Phase 4) | High | High | Validate approach with maintainers |

---

## 12. Timeline

> **Note:** Timeline updated after Phases 1-2 invalidated. Proceeding directly to Phase 3.

### Week 1: Phase 3 Foundation
- Day 1-2: Design work queue data structures
- Day 3-4: Implement shared memory queue
- Day 5: Unit tests for queue operations

### Week 2: Phase 3 Implementation
- Day 1-2: Persistent thread kernel
- Day 3-4: Work-stealing integration
- Day 5: Convergence detection

### Week 3: Phase 3 Testing + Optimization
- Day 1-2: Integration testing
- Day 3: Performance benchmarking
- Day 4: Bug fixes and edge cases
- Day 5: Documentation

### Future: Phase 4 (If Needed)
- Weeks 4-6: Detray modifications for cooperative stepping
- Contingent on Phase 3 results and maintainer approval

---

## 13. Appendix: Code References

### Key Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `device/common/include/traccc/edm/device/sort_key.hpp` | Sort key computation | 18-23 |
| `device/cuda/src/finding/combinatorial_kalman_filter.cuh` | CKF orchestration | 481-696 |
| `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | Propagation kernel | 22-161 |
| `device/common/include/traccc/finding/device/impl/find_tracks.ipp` | Load-balancing reference | 141-498 |
| `core/include/traccc/finding/finding_config.hpp` | Configuration | Full file |
| `detray-fork/core/include/detray/propagator/rk_stepper.ipp` | RK4 stepper | 606-862 |

### Data Structures

| Structure | File | Size |
|-----------|------|------|
| `bound_track_parameters` | detray | ~50 bytes |
| `propagate_to_next_surface_payload` | traccc | ~200 bytes |
| `intermediate_state` | detray | ~164 bytes |
| `candidate_link` | traccc | ~40 bytes |

### Atomic Operations

| Operation | Usage | File |
|-----------|-------|------|
| `vecmem::device_atomic_ref::fetch_add` | Queue push | find_tracks.ipp:159 |
| `vecmem::device_atomic_ref::compare_exchange_strong` | Mutex lock | find_tracks.ipp:337-340 |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-30 | 1.0 | Initial plan document |
| 2025-12-30 | 1.1 | Fixed critical issues identified during review (now obsolete - Phases 1-2 invalidated) |
| 2025-12-30 | 2.0 | **MAJOR UPDATE: Correlation analysis invalidates core assumptions** |
| | | - Added instrumentation to measure |qop| vs step count correlation |
| | | - **Finding: Pearson correlation = -0.03 (NO correlation)** |
| | | - **Finding: Higher |qop| = FEWER steps (opposite of assumption)** |
| | | - Marked Phase 1 (Pre-Sort) as INVALIDATED |
| | | - Marked Phase 2 (Two-Phase) as INVALIDATED |
| | | - Phase 3 (Persistent Threads) now RECOMMENDED primary approach |
| | | - Added Section 2: Critical Correlation Analysis Results |
| 2025-12-30 | 2.1 | Cleaned up invalidated sections: |
| | | - Fixed subsection numbering in Phase 1 (5.x) and Phase 2 (6.x) |
| | | - Removed obsolete code examples with incorrect assumptions |
| | | - Added "DO NOT IMPLEMENT" warnings to code blocks |
| | | - Condensed Phase 2 to remove 200+ lines of invalid code |
| 2025-12-30 | 2.2 | Fixed document structure and section numbering: |
| | | - Fixed duplicate Section 3 (now §3 Problem Recap, §4 Implementation Strategy) |
| | | - Fixed Phase 3 subsections (7.3-7.6) |
| | | - Fixed Phase 4 subsections (8.1-8.5) |
| | | - Renumbered Testing (§9), Benchmarking (§10), Risk (§11), Timeline (§12), Appendix (§13) |
| | | - Updated Architecture Overview diagram for Phase 3 approach |
| | | - Updated File Modification Summary for Phase 3 only |
| | | - Updated Testing Strategy tests for Phase 3 |
| | | - Updated Performance Targets table (removed Phase 1/2 columns) |
| | | - Updated Risk Assessment (removed Phase 1/2 risks) |
| | | - Updated Timeline (Phase 3 focused) |
| 2025-12-31 | 2.3 | Cross-validated with codebase, fixed code references: |
| | | - Fixed line numbers: `propagate_to_next_surface.ipp` 22-172 → 22-161 |
| | | - Fixed line numbers: `combinatorial_kalman_filter.cuh` 491-697 → 481-696 |
| | | - Fixed line numbers: `sort_key.hpp` 16-23 → 18-23 |
| | | - Updated Phase 3 code to use traccc API patterns: |
| | | - Changed `__syncthreads()` → `barrier.blockBarrier()` |
| | | - Changed raw atomics → `vecmem::device_atomic_ref` |
| | | - Clarified barrier vs blockOr usage in find_tracks.ipp references |
| | | - Updated atomic operations table with correct API names |
| 2025-12-31 | 2.4 | Cross-validated confirmation items, applied fixes: |
| | | - §7.5: Replaced spin-wait with `barrier.blockOr()` collective termination |
| | | - §7.6: Added validation note confirming 4.1 KB within GPU shared memory limits |
| | | - §10.4: Added conservative (20-30%) vs optimistic (30-50%) estimates |
| | | - Removed `active_count` tracking (unnecessary with blockOr pattern) |
| | | - Added `queue_head` initialization in setup block |

---

*This plan is a living document and will be updated as implementation progresses.*
