# Analysis: Work-Stealing Pattern and Propagator Redesign

**Date:** 2025-12-31
**Context:** Phase 3 Implementation Review for GitHub Issue #851
**Status:** Critical design issue identified

---

## Executive Summary

The Phase 3 persistent threads implementation compiles and is functionally correct, but **does not achieve the intended performance benefit**. The work-stealing pattern is incompatible with the current monolithic propagator design. Addressing warp divergence from variable RK4 step counts would require fundamental changes to the detray propagator interface.

---

## 1. Problem Statement

### 1.1 Original Goal

Reduce warp divergence in the `propagate_to_next_surface` kernel caused by variable RK4 propagation step counts (observed range: 1-31 steps). Warp divergence wastes ~62% of GPU cycles.

### 1.2 Attempted Solution

Implement a work-stealing queue pattern (adapted from `find_tracks.ipp`) where:
1. Thread 0 populates a shared memory queue with work items
2. All threads atomically dequeue and process items
3. Fast-finishing threads can "steal" work from slower threads
4. `barrier.blockOr()` provides collective termination

---

## 2. Why the Pattern Fails

### 2.1 The Synchronization Barrier Problem

The implementation uses `barrier.blockOr(has_work)` which wraps `__syncthreads_or()`:

```cpp
while (barrier.blockOr(has_work)) {
    has_work = false;
    unsigned int my_idx = atomic_fetch_add(queue_head, 1);
    if (my_idx < queue_tail) {
        has_work = true;
        process_propagation_work_item(...);  // Variable time: 1-31 RK4 steps
    }
}
```

**Critical behavior**: `__syncthreads_or()` is a **block-wide synchronization barrier**. ALL threads in the block must reach this point before ANY thread can proceed.

### 2.2 Execution Trace

```
Iteration 1:
├── barrier.blockOr(true) → ALL 128 threads sync, returns true
├── All threads do atomic fetch_add → get indices 0-127
├── All threads call process_propagation_work_item()
│   ├── Thread 0: 3 RK4 steps → finishes at T=10
│   ├── Thread 1: 25 RK4 steps → finishes at T=100
│   └── Thread 0 CANNOT proceed until Thread 1 finishes
└── ALL threads wait for slowest (Thread 1)

Iteration 2:
├── barrier.blockOr() → All threads finally sync at T=100
├── All threads fetch_add → get indices 128-255
├── All indices >= queue_tail (128) → no work
└── Loop exits

Result: Every thread processed exactly 1 item. No work was "stolen".
```

### 2.3 Comparison with find_tracks.ipp

The `find_tracks.ipp` pattern works because:

| Aspect | find_tracks.ipp | propagate_to_next_surface |
|--------|-----------------|---------------------------|
| Work generation | **Distributed** - each thread generates its own candidates | **Centralized** - thread 0 generates all items |
| Work processing | **Collective** - all threads help process all candidates | **Isolated** - each thread processes alone |
| Iterations | Multiple fill-process cycles | Single effective iteration |
| Benefit | Redistributes uneven candidate counts | None - work count equals thread count |

In `find_tracks`, if Thread A generates 10 candidates and Thread B generates 2, ALL threads collectively process all 12. Work is truly redistributed.

In `propagate_to_next_surface`, each thread takes one item and processes it alone while everyone waits.

---

## 3. Resolution Options Analysis

### Option 1: Remove Block Barrier

```cpp
// No barrier, simple loop
while (true) {
    unsigned int my_idx = atomic_fetch_add(queue_head, 1);
    if (my_idx >= queue_tail) break;
    process_propagation_work_item(...);
}
```

| Pros | Cons |
|------|------|
| Warps can proceed independently | Warp-internal divergence remains |
| Simple change | Queue has ≤128 items, minimal stealing |
| | Different termination semantics |

**Verdict**: Minor improvement for inter-warp balancing, does not address root cause.

---

### Option 2: Global Work Queue (True Persistent Threads)

```cpp
__device__ GlobalQueue global_queue;  // All tracks across all blocks

__global__ void persistent_kernel() {
    while (WorkItem* item = global_queue.try_dequeue()) {
        process(*item);
    }
}
```

| Pros | Cons |
|------|------|
| Fast blocks can steal from slow blocks | Global atomics are slower |
| Better overall GPU utilization | Significant restructuring required |
| | Still doesn't help warp divergence |

**Verdict**: Helps block-level load balancing, does not address warp divergence.

---

### Option 3: Chunked/Incremental Propagation

```cpp
struct PropagationState {
    bound_track_parameters params;
    propagator_state state;
    int steps_remaining;
    bool complete;
};

__shared__ PropagationState active_tracks[BLOCK_SIZE];

while (barrier.blockOr(has_active_track)) {
    if (has_active_track) {
        // Execute ONE RK4 step only
        bool done = do_single_rk4_step(&active_tracks[tid]);

        if (done) {
            finalize_track(&active_tracks[tid]);
            has_active_track = try_dequeue_new_track(&active_tracks[tid]);
        }
    }
    // Fast tracks (3 steps) release after 3 iterations
    // Slow tracks (25 steps) continue for 25 iterations
}
```

| Pros | Cons |
|------|------|
| **Actually addresses warp divergence** | Requires propagator interface redesign |
| Fast tracks release threads sooner | Propagator state must be persistent |
| Work naturally redistributed | Touches detray library internals |
| | Very high implementation effort |

**Verdict**: The only option that addresses the root cause, but requires fundamental propagator changes.

---

### Option 4: Revert to Original

Remove the work-stealing implementation and return to the original 1:1 thread-to-track mapping.

| Pros | Cons |
|------|------|
| Simpler code | No performance improvement |
| No false optimization claims | Original problem remains |
| Honest about limitations | |

**Verdict**: Practical if chunked propagation is not feasible.

---

## 4. Requirements for Chunked Propagation

To implement Option 3, the detray propagator would need:

### 4.1 Interface Changes

```cpp
// Current interface (monolithic)
propagator.propagate(state, actors);  // Runs to completion

// Required interface (incremental)
propagator.step(state, actors);       // Single RK4 step
bool propagator.is_complete(state);   // Check if done
```

### 4.2 State Persistence

The propagator state must be:
- Serializable to shared memory (for work redistribution)
- Resumable after interruption
- Size-bounded (shared memory is limited)

### 4.3 Actor State Handling

The 7 actor states (s0-s6) must also be:
- Persistent across steps
- Stored per-track in shared memory

### 4.4 Estimated Shared Memory Requirements

Based on detailed codebase analysis of detray propagator internals:

```
STEPPER STATE (~220 bytes):
├── Free track parameters (8 floats): 32 bytes
├── Bound track parameters (5 floats + barcode): 28 bytes
├── Jacobian transport (5x5 matrix): 100 bytes
├── Particle hypothesis (5 floats): 20 bytes
├── Path lengths (2 scalars): 8 bytes
├── Step sizes: 8 bytes
├── Trial counter: 8 bytes
└── Constraint state (4 floats): 16 bytes

NAVIGATION STATE (~424 bytes):
├── Candidate cache (10 × 40 bytes): 400 bytes
├── Status, direction, trust_level: 12 bytes
├── Volume index: 4 bytes
├── Next/Last indices: 4 bytes
└── External tolerance: 4 bytes

ACTOR STATES (~144 bytes):
├── pathlimit_aborter::state: 8 bytes
├── parameter_transporter::state: 40 bytes
├── interaction_register::state: 16 bytes
├── ckf_interactor::state: 16 bytes
├── parameter_resetter::state: 24 bytes
├── momentum_aborter::state: 16 bytes
└── ckf_aborter::state: 24 bytes

PROPAGATION CONTAINER (~20 bytes):
├── Pointers/references: 16 bytes
└── Heartbeat flags: 4 bytes

Total: ~808 bytes per track
For 128 tracks: ~101 KB (exceeds 48 KB shared memory limit)
```

This **significantly exceeds** typical shared memory limits, requiring either:
- Reduced block size (e.g., 48 tracks/block)
- Hybrid shared/global memory approach
- Spilling to global memory with caching

---

## 5. Conclusion

### 5.1 Current Implementation Status

- **Compiles**: Yes
- **Functionally correct**: Yes
- **Performance benefit**: No

### 5.2 Root Cause

The work-stealing pattern requires work to be **interruptible and redistributable**. The monolithic `propagator.propagate()` call is neither.

### 5.3 Recommendation

1. **Short term**: Revert to original 1:1 mapping or keep current code with documentation that it provides no benefit (useful as scaffolding for future work)

2. **Long term**: Investigate chunked propagation with detray maintainers. This is a significant architectural change that would benefit multiple use cases beyond this optimization.

### 5.4 Alternative Approaches (Not Requiring Propagator Changes)

- **Track sorting**: Group tracks with similar expected step counts into the same warps. However, analysis showed |qop| does not predict step count, so no good sorting key exists.

- **Warp specialization**: Assign different warps to handle different track "difficulty" levels. Requires a predictive metric that doesn't currently exist.

---

## Appendix: Code References

| File | Description |
|------|-------------|
| `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | Current implementation with work-stealing |
| `device/common/include/traccc/finding/device/impl/find_tracks.ipp` | Reference pattern (works due to different work structure) |
| `doc/work_redistribution_plan_phase3.md` | Original implementation plan |

---

*This analysis documents why the Phase 3 implementation does not achieve its performance goals and what would be required to properly address warp divergence in the propagation kernel.*
