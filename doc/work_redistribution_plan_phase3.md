# Phase 3: Persistent Threads Implementation Plan

**GitHub Issue:** [#851](https://github.com/acts-project/traccc/issues/851)
**Parent Document:** `doc/work_redistribution_plan.md`
**Branch:** `feature/work-redistribution`
**Status:** Ready for Implementation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Current Implementation Analysis](#2-current-implementation-analysis)
3. [Target Implementation Design](#3-target-implementation-design)
4. [File Modifications](#4-file-modifications)
5. [Before/After Code Comparison](#5-beforeafter-code-comparison)
6. [Implementation Steps](#6-implementation-steps)
7. [Testing Plan](#7-testing-plan)
8. [Appendix: Code References](#8-appendix-code-references)

---

## 1. Overview

### 1.1 Objective

Transform the propagation kernel from a naive 1-thread-per-track model to a persistent threads model with work-stealing queue, reducing warp divergence from ~62% wasted cycles to <30%.

### 1.2 Key Insight

The `find_tracks` kernel already implements a proven load-balanced pattern that we will adapt:

```
find_tracks.ipp (Reference)          propagate_to_next_surface.ipp (Target)
========================             ================================
Shared buffer for measurements  -->  Shared queue for propagation work items
barrier.blockOr() termination   -->  barrier.blockOr() termination
vecmem atomic push/pop          -->  vecmem atomic dequeue
~1.5 KB shared memory           -->  ~3.1 KB shared memory
```

### 1.3 Expected Outcome

| Metric | Before | After (Conservative) | After (Optimistic) |
|--------|--------|---------------------|-------------------|
| Warp efficiency | ~38% | ~53% | ~70% |
| Wasted cycles | ~62% | ~47% | ~30% |
| Throughput improvement | - | 20-30% | 30-50% |

---

## 2. Current Implementation Analysis

### 2.1 Kernel Launch Configuration

**File:** `device/cuda/src/finding/combinatorial_kalman_filter.cuh`
**Lines:** 510-516

```cpp
// CURRENT: Propagation kernel launch
const unsigned int nThreads = warp_size * 4;  // 128 threads
const unsigned int nBlocks = (n_candidates + nThreads - 1) / nThreads;
propagate_to_next_surface<...>(
    nBlocks, nThreads,
    0,  // <-- NO shared memory used
    stream, config, host_payload);
```

**Key observation:** Shared memory size is `0` - no load balancing.

### 2.2 Current Kernel Structure

**File:** `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh`
**Lines:** 20-27

```cpp
// CURRENT: Simple 1-to-1 thread mapping
template <typename propagator_t, typename bfield_t>
__global__ __launch_bounds__(128) void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    // Each thread gets ONE track based on global index
    device::propagate_to_next_surface<propagator_t, bfield_t>(
        details::global_index1(), cfg, payload);
}
```

### 2.3 Current Device Function

**File:** `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`
**Lines:** 22-159

```cpp
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const global_index_t globalIndex, const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload) {

    // Early exit for out-of-bounds threads
    if (globalIndex >= payload.n_in_params) {
        return;  // <-- PROBLEM: Thread sits idle
    }

    // ... setup code (lines 32-64) ...

    // Liveness check
    if (params_liveness.at(param_id) == 0u) {
        return;  // <-- PROBLEM: Thread sits idle
    }

    // ... propagation logic (lines 66-123) ...
    propagator.propagate(propagation, detray::tie(s0, s1, s2, s3, s4, s5, s6));

    // ... result handling (lines 125-158) ...
}
```

**Problem:** Threads that exit early OR finish quickly sit idle while other threads in the warp continue working.

### 2.4 Reference: find_tracks Load-Balanced Pattern

**File:** `device/common/include/traccc/finding/device/impl/find_tracks.ipp`
**Lines:** 141-498

```cpp
// REFERENCE: Load-balanced work loop
while (barrier.blockOr(curr_meas < num_meas ||
                       shared_payload.shared_candidates_size > 0)) {
    // Fill shared buffer with work items
    for (; curr_meas < num_meas &&
           shared_payload.shared_candidates_size < thread_id.getBlockDimX();
         curr_meas++) {
        unsigned int idx =
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared_payload.shared_candidates_size)
                .fetch_add(1u);
        shared_payload.shared_candidates[idx] = {init_meas + curr_meas,
                                                  thread_id.getLocalThreadIdX()};
    }
    barrier.blockBarrier();  // Sync after filling

    // Process work items...

    barrier.blockBarrier();  // Sync before next iteration
}
```

---

## 3. Target Implementation Design

### 3.1 New Shared Memory Structures

**New File:** `device/common/include/traccc/finding/device/propagate_shared_payload.hpp`

```cpp
#pragma once

// Note: Implementation files using this header will need:
// #include <vecmem/memory/device_atomic_ref.hpp>
// #include "traccc/device/concepts/barrier.hpp"
// #include "traccc/device/concepts/thread_id.hpp"

namespace traccc::device {

/// Work item for propagation queue
struct propagation_work_item {
    unsigned int param_id;   // Track parameter index (4 bytes)
    unsigned int link_idx;   // Link index for this track (4 bytes)
    unsigned int n_cands;    // Number of candidates so far (4 bytes)
    // Total: 12 bytes per item
};

// Compile-time verification of struct size
static_assert(sizeof(propagation_work_item) == 12,
              "propagation_work_item must be 12 bytes");

/// Shared memory payload for persistent propagation kernel
struct propagate_shared_payload {
    /// Work queue buffer (2x block size for safety)
    propagation_work_item* queue;

    /// Atomic queue head (next item to dequeue)
    unsigned int& queue_head;

    /// Queue tail (total items in queue)
    unsigned int& queue_tail;

    /// Completed propagations counter
    unsigned int& completed_count;
};

}  // namespace traccc::device
```

### 3.2 Shared Memory Layout

```
Shared Memory for 128-thread block:

EXTERN __shared__ (passed to kernel as shared_size = 3072 bytes):
┌────────────────────────────────────────────────────────┐
│ queue[256] = 256 × 12 bytes = 3072 bytes (3 KB)        │
└────────────────────────────────────────────────────────┘

STATIC __shared__ (compiler-managed, NOT in shared_size):
┌────────────────────────────────────────────────────────┐
│ queue_head: 4 bytes                                    │
│ queue_tail: 4 bytes                                    │
│ completed_count: 4 bytes                               │
└────────────────────────────────────────────────────────┘
```

**Queue Size Rationale:**
- Queue size = 2 × blockDim.x = 256 items
- Block processes at most blockDim.x (128) tracks per launch
- 2× provides safety margin and matches `find_tracks` pattern
- Actual usage: max(live_tracks_in_block) ≤ 128

### 3.3 Work Distribution Strategy

```
BEFORE (1-to-1 mapping):
Thread 0  → Track 0  (5 steps)  ████░░░░░░░░░░░░░░░░  Done, IDLE
Thread 1  → Track 1  (3 steps)  ███░░░░░░░░░░░░░░░░░  Done, IDLE
Thread 31 → Track 31 (25 steps) █████████████████████████  Still working
                                All threads WAIT for Thread 31

AFTER (work-stealing queue):
Thread 0  → Track 0  (5 steps)  ████░ → steals Track 32 (4 steps) ████░ → steals...
Thread 1  → Track 1  (3 steps)  ███░ → steals Track 33 (6 steps) ██████░ → steals...
Thread 31 → Track 31 (25 steps) █████████████████████████
                                Threads dynamically grab work until queue empty
```

---

## 4. File Modifications

### 4.1 Files to Modify

| File | Change Type | Description |
|------|-------------|-------------|
| `device/common/include/traccc/finding/device/propagate_to_next_surface.hpp` | MODIFY | Add shared payload parameter |
| `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | MODIFY | Implement work-stealing loop |
| `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh` | MODIFY | Add shared memory declarations |
| `device/cuda/src/finding/combinatorial_kalman_filter.cuh` | MODIFY | Calculate and pass shared memory size |

### 4.2 Files to Add

| File | Description |
|------|-------------|
| `device/common/include/traccc/finding/device/propagate_shared_payload.hpp` | Shared payload structure definitions |

---

## 5. Before/After Code Comparison

### 5.1 Kernel Launch (combinatorial_kalman_filter.cuh)

**BEFORE (lines 510-516):**
```cpp
const unsigned int nThreads = warp_size * 4;
const unsigned int nBlocks = (n_candidates + nThreads - 1) / nThreads;
propagate_to_next_surface<
    traccc::details::ckf_propagator_t<detector_t, bfield_t>,
    bfield_t>(nBlocks, nThreads, 0, stream, config, host_payload);
//                              ↑
//                              No shared memory
```

**AFTER:**
```cpp
const unsigned int nThreads = warp_size * 4;  // 128 threads
const unsigned int nBlocks = (n_candidates + nThreads - 1) / nThreads;

// Calculate shared memory size (extern __shared__ portion only)
// Static __shared__ variables (queue_head, queue_tail, completed_count) are
// managed by the compiler and NOT included in shared_size
const std::size_t shared_size =
    2 * nThreads * sizeof(device::propagation_work_item);  // 256 * 12 = 3072 bytes

propagate_to_next_surface<
    traccc::details::ckf_propagator_t<detector_t, bfield_t>,
    bfield_t>(nBlocks, nThreads, shared_size, stream, config, host_payload);
//                              ↑
//                              Shared memory for work queue
```

### 5.2 Kernel Function (propagate_to_next_surface_src.cuh)

**BEFORE (lines 20-27):**
```cpp
template <typename propagator_t, typename bfield_t>
__global__ __launch_bounds__(128) void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    device::propagate_to_next_surface<propagator_t, bfield_t>(
        details::global_index1(), cfg, payload);
}
```

**AFTER:**
```cpp
// Local include(s). - ADD THESE (matching find_tracks_src.cuh:14-15)
#include "../../../utils/barrier.hpp"
#include "../../../utils/thread_id.hpp"

// Project include(s). - ADD THIS
#include "traccc/finding/device/propagate_shared_payload.hpp"

template <typename propagator_t, typename bfield_t>
__global__ __launch_bounds__(128) void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    // Shared memory declarations (similar to find_tracks_src.cuh:26-33)
    // Note: These static __shared__ variables are NOT part of extern __shared__
    __shared__ unsigned int queue_head;
    __shared__ unsigned int queue_tail;
    __shared__ unsigned int completed_count;

    // Use unsigned int array for 4-byte alignment (propagation_work_item has unsigned int members)
    // Pattern adapted from find_tracks_src.cuh:29-33 (which uses unsigned long long for 8-byte alignment)
    extern __shared__ unsigned int shared_mem[];
    device::propagation_work_item* queue =
        reinterpret_cast<device::propagation_work_item*>(shared_mem);

    // Barrier and thread ID (same pattern as find_tracks_src.cuh:35-36)
    cuda::barrier barrier;
    details::thread_id1 thread_id;

    // Call device function with shared payload
    device::propagate_to_next_surface<propagator_t, bfield_t>(
        thread_id, barrier, cfg, payload,
        {queue, queue_head, queue_tail, completed_count});
}
```

### 5.3 Device Function Signature (propagate_to_next_surface.hpp)

**BEFORE (lines 98-101):**
```cpp
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    global_index_t globalIndex, const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload);
```

**AFTER:**
```cpp
// ADD THESE INCLUDES (matching find_tracks.hpp:11-12)
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"

// ADD THIS INCLUDE for propagate_shared_payload type
#include "propagate_shared_payload.hpp"

/// Original function signature (kept for backward compatibility)
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    global_index_t globalIndex, const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload);

/// New persistent threads function signature
/// Note: Use concepts:: not device::concepts:: (we're inside namespace traccc::device)
template <typename propagator_t, typename bfield_t,
          concepts::thread_id1 thread_id_t,
          concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    const propagate_shared_payload& shared_payload);
```

### 5.4 Device Function Implementation (propagate_to_next_surface.ipp)

**BEFORE (lines 22-159) - Simplified structure:**
```cpp
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const global_index_t globalIndex, const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload) {

    // [1] Bounds check - exit if out of range
    if (globalIndex >= payload.n_in_params) {
        return;
    }

    // [2] Get parameter ID from sorted order
    const unsigned int param_id = param_ids.at(globalIndex);

    // [3] Liveness check - exit if dead track
    if (params_liveness.at(param_id) == 0u) {
        return;
    }

    // [4] Setup propagator (lines 66-121)
    // ... actor states, propagator config ...

    // [5] Execute propagation
    propagator.propagate(propagation, detray::tie(s0, s1, s2, s3, s4, s5, s6));

    // [6] Handle results (lines 126-158)
    if (s6.success) {
        params[param_id] = propagation._stepping.bound_params();
        params_liveness[param_id] = 1u;
        // ... validation checks ...
    } else {
        params_liveness[param_id] = 0u;
    }

    // [7] Create tip if needed
    if (params_liveness[param_id] == 0 && n_cands >= cfg.min_track_candidates_per_track) {
        auto tip_pos = tips.push_back(link_idx);
        tip_lengths.at(tip_pos) = n_cands;
    }
}
```

**AFTER - New persistent threads implementation:**
```cpp
// ADD THIS INCLUDE (matching find_tracks.ipp:18)
#include <vecmem/memory/device_atomic_ref.hpp>

/// Helper: Process a single propagation work item
/// Extracted from original propagate_to_next_surface logic
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void process_propagation_work_item(
    const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    const propagation_work_item& item) {

    using scalar_t = typename propagator_t::detector_type::scalar_type;

    // Get data from work item
    const unsigned int param_id = item.param_id;
    const unsigned int link_idx = item.link_idx;
    const unsigned int n_cands = item.n_cands;

    // Access containers
    typename propagator_t::detector_type det(payload.det_data);
    bound_track_parameters_collection_types::device params(payload.params_view);
    vecmem::device_vector<unsigned int> params_liveness(payload.params_liveness_view);
    vecmem::device_vector<unsigned int> tips(payload.tips_view);
    vecmem::device_vector<unsigned int> tip_lengths(payload.tip_lengths_view);

    // Skip dead tracks (already filtered in queue population, but double-check)
    if (params_liveness.at(param_id) == 0u) {
        return;
    }

    // Input bound track parameter
    const bound_track_parameters<> in_par = params.at(param_id);

    // Create propagator and state
    auto prop_cfg{cfg.propagation};
    prop_cfg.navigation.estimate_scattering_noise = false;
    propagator_t propagator(prop_cfg);
    typename propagator_t::state propagation(in_par, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, in_par));
    propagation._stepping
        .template set_constraint<detray::step::constraint::e_accuracy>(
            cfg.propagation.stepping.step_constraint);

    // Setup actor states (lines 82-101 from original)
    using actor_tuple_type = typename propagator_t::actor_chain_type::actor_tuple;
    typename detray::detail::tuple_element<0, actor_tuple_type>::type::state s0{};
    typename detray::detail::tuple_element<1, actor_tuple_type>::type::state s1{};
    typename detray::detail::tuple_element<3, actor_tuple_type>::type::state s3{};
    typename detray::detail::tuple_element<2, actor_tuple_type>::type::state s2{s3};
    typename detray::detail::tuple_element<4, actor_tuple_type>::type::state s4{prop_cfg};
    typename detray::detail::tuple_element<5, actor_tuple_type>::type::state s5;
    typename detray::detail::tuple_element<6, actor_tuple_type>::type::state s6;

    // MBF smoother Jacobian handling
    if (cfg.run_mbf_smoother) {
        assert(payload.tmp_jacobian_ptr != nullptr);
        payload.tmp_jacobian_ptr[param_id] = matrix::identity<
            bound_matrix<typename propagator_t::detector_type::algebra_type>>();
        s1._full_jacobian_ptr = &payload.tmp_jacobian_ptr[param_id];
    }

    s5.min_pT(static_cast<scalar_t>(cfg.min_pT));
    s5.min_p(static_cast<scalar_t>(cfg.min_p));
    s6.min_step_length = cfg.min_step_length_for_next_surface;
    s6.max_count = cfg.max_step_counts_for_next_surface;

    // Execute propagation (THE VARIABLE-TIME OPERATION)
    propagator.propagate(propagation, detray::tie(s0, s1, s2, s3, s4, s5, s6));

    // Handle results
    if (s6.success) {
        assert(propagation._navigation.is_on_sensitive());
        assert(!propagation._stepping.bound_params().is_invalid());

        params[param_id] = propagation._stepping.bound_params();
        params_liveness[param_id] = 1u;

        const scalar theta = params[param_id].theta();
        if (theta <= 0.f || theta >= 2.f * constant<traccc::scalar>::pi) {
            TRACCC_ERROR_DEVICE("Theta is zero after propagation");
            params_liveness[param_id] = 0u;
        }
        if (!std::isfinite(params[param_id].phi())) {
            TRACCC_ERROR_DEVICE("Phi is infinite after propagation (Matrix inversion)");
            params_liveness[param_id] = 0u;
        }
        if (math::fabs(params[param_id].qop()) == 0.f) {
            TRACCC_ERROR_DEVICE("q/p is zero after propagation");
            params_liveness[param_id] = 0u;
        }
    } else {
        params_liveness[param_id] = 0u;
    }

    // Create tip if track ended
    if (params_liveness[param_id] == 0 &&
        n_cands >= cfg.min_track_candidates_per_track) {
        TRACCC_VERBOSE_DEVICE("Create tip: No next sensitive found");
        auto tip_pos = tips.push_back(link_idx);
        tip_lengths.at(tip_pos) = n_cands;
    }
}

/// Main persistent threads function
/// Note: Use concepts:: not device::concepts:: (we're inside namespace traccc::device)
template <typename propagator_t, typename bfield_t,
          concepts::thread_id1 thread_id_t,
          concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    const propagate_shared_payload& shared_payload) {

    // Access views
    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);
    vecmem::device_vector<const candidate_link> links(payload.links_view);
    vecmem::device_vector<const unsigned int> params_liveness_const(
        payload.params_liveness_view);

    // =========================================================================
    // PHASE 1: Populate work queue (thread 0 only)
    // =========================================================================
    if (thread_id.getLocalThreadIdX() == 0) {
        shared_payload.queue_head = 0;
        shared_payload.queue_tail = 0;
        shared_payload.completed_count = 0;

        // Calculate work items for this block
        const unsigned int block_start = thread_id.getBlockIdX() * thread_id.getBlockDimX();
        const unsigned int block_end = std::min(block_start + thread_id.getBlockDimX(),
                                                payload.n_in_params);

        for (unsigned int i = block_start; i < block_end; ++i) {
            const unsigned int param_id = param_ids.at(i);

            // Only enqueue live tracks
            if (params_liveness_const.at(param_id) != 0u) {
                const unsigned int link_idx = payload.prev_links_idx + param_id;
                const auto& link = links.at(link_idx);
                assert(link.step == payload.step);
                const unsigned int n_cands = link.step + 1 - link.n_skipped;

                unsigned int slot = shared_payload.queue_tail;
                shared_payload.queue[slot] = {param_id, link_idx, n_cands};
                shared_payload.queue_tail = slot + 1;
            }
        }
    }

    // All threads wait for queue population
    barrier.blockBarrier();

    // =========================================================================
    // PHASE 2: Work-stealing loop
    // Pattern: find_tracks.ipp:141,262 - barrier.blockOr() for termination
    // =========================================================================
    bool has_work = true;

    while (barrier.blockOr(has_work)) {
        has_work = false;

        // Try to dequeue a work item atomically
        // Pattern: find_tracks.ipp:156-159
        unsigned int my_idx =
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared_payload.queue_head).fetch_add(1u);

        if (my_idx < shared_payload.queue_tail) {
            // Got a work item - process it
            has_work = true;

            const propagation_work_item& item = shared_payload.queue[my_idx];

            // Execute propagation for this track
            process_propagation_work_item<propagator_t, bfield_t>(
                cfg, payload, item);

            // Increment completion counter
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared_payload.completed_count).fetch_add(1u);
        }
        // Threads without work vote 'false' in blockOr
        // Loop exits when ALL threads have no work (blockOr returns false)
    }
}
```

---

## 6. Implementation Steps

### Step 1: Create Shared Payload Header

1. Create `device/common/include/traccc/finding/device/propagate_shared_payload.hpp`
2. Define `propagation_work_item` struct (12 bytes)
3. Define `propagate_shared_payload` struct
4. Add to `device/common/CMakeLists.txt`

### Step 2: Modify Device Function Header

1. Edit `device/common/include/traccc/finding/device/propagate_to_next_surface.hpp`
2. Add include for new shared payload header
3. Add new function signature with `thread_id`, `barrier`, and `shared_payload`
4. Keep original signature for backward compatibility

### Step 3: Implement Work-Stealing Logic

1. Edit `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`
2. Add `process_propagation_work_item` helper function
3. Implement new persistent threads function
4. Keep original function for non-CUDA backends (or update all)

### Step 4: Update CUDA Kernel

1. Edit `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh`
2. Add shared memory declarations
3. Add barrier and thread_id instantiation
4. Update kernel to call new function signature

### Step 5: Update Kernel Launch

1. Edit `device/cuda/src/finding/combinatorial_kalman_filter.cuh`
2. Calculate shared memory size at line ~510
3. Pass shared memory size to kernel launch

### Step 6: Update Other Backends (Optional)

1. Apply similar changes to Alpaka backend (`device/alpaka/...`)
2. Apply similar changes to SYCL backend (`device/sycl/...`)

---

## 7. Testing Plan

### 7.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_propagation_work_item` | Verify struct size is 12 bytes |
| `test_queue_population` | Verify queue correctly populated |
| `test_atomic_dequeue` | Verify atomic dequeue works correctly |
| `test_termination` | Verify blockOr termination when queue empty |

### 7.2 Integration Tests

```bash
# Run CKF with work redistribution
./bin/traccc_throughput_mt_cuda \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --input-directory=odd/geant4_ttbar_mu200/ \
  --input-events=36 --processed-events=500 --cpu-threads=1
```

### 7.3 Correctness Validation

Compare before/after:
- Number of tracks found
- Track parameter distributions
- Chi-square distributions
- Tips created

### 7.4 Performance Benchmarks

```bash
# Baseline
git checkout main
./bin/traccc_throughput_mt_cuda ... --cpu-threads=1,4,7,8

# With work redistribution
git checkout feature/work-redistribution
./bin/traccc_throughput_mt_cuda ... --cpu-threads=1,4,7,8
```

---

## 8. Appendix: Code References

### 8.1 Key Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `device/cuda/src/finding/combinatorial_kalman_filter.cuh` | Kernel launch | 510-516 |
| `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh` | CUDA kernel | 20-27 |
| `device/common/include/traccc/finding/device/propagate_to_next_surface.hpp` | Payload struct | 27-85 |
| `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | Device function | 22-159 |
| `device/cuda/src/utils/barrier.hpp` | Barrier API | 12-26 |
| `device/common/include/traccc/finding/device/impl/find_tracks.ipp` | Reference pattern | 141-498 |
| `device/cuda/src/finding/kernels/specializations/find_tracks_src.cuh` | Reference shared mem | 26-41 |

### 8.2 Barrier API (device/cuda/src/utils/barrier.hpp)

```cpp
struct barrier {
    __device__ inline void blockBarrier() const { __syncthreads(); }
    __device__ inline bool blockOr(bool predicate) const {
        return __syncthreads_or(predicate);
    }
};
```

### 8.3 Atomic API (vecmem)

```cpp
// Atomic fetch_add pattern from find_tracks.ipp:156-159
vecmem::device_atomic_ref<unsigned int,
                          vecmem::device_address_space::local>(
    shared_variable).fetch_add(1u);
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-31 | 1.0 | Initial Phase 3 detailed plan |
| 2025-12-31 | 1.1 | Cross-validation fixes: |
| | | - §3.1: Added `static_assert` for struct size verification |
| | | - §3.2: Clarified queue size rationale, fixed memory total (4.1→3.1 KB) |
| | | - §5.2: Fixed `extern __shared__` to use `char[]` + `reinterpret_cast` pattern |
| 2025-12-31 | 1.2 | Cross-validation fixes (round 2): |
| | | - §5.2: Fixed alignment: `char[]` → `unsigned int[]` for 4-byte alignment |
| | | - §5.3, §5.4: Fixed concept namespace: `concepts::` → `device::concepts::` |
| | | - §3.1: Added includes comment for implementation files |
| | | - §5.4: Restored `TRACCC_ERROR_DEVICE`/`TRACCC_VERBOSE_DEVICE` logging macros |
| 2025-12-31 | 1.3 | Cross-validation fixes (round 3): |
| | | - §5.1: Fixed shared_size calculation (removed static __shared__ from count) |
| | | - §5.2: Added missing includes (barrier.hpp, thread_id.hpp, propagate_shared_payload.hpp) |
| | | - §5.3: Added missing include (propagate_shared_payload.hpp) |
| | | - §3.2: Clarified extern vs static shared memory distinction |
| | | - §5.4: Added missing include (vecmem/memory/device_atomic_ref.hpp) |
| 2025-12-31 | 1.4 | Cross-validation fixes (round 4): |
| | | - §5.3, §5.4: Reverted concept namespace: `device::concepts::` → `concepts::` (v1.2 regression) |
| | | - §5.3: Added concept header includes (barrier.hpp, thread_id.hpp) |
| | | - §5.4: Fixed `min` → `std::min` (matching find_tracks.ipp:557) |
| | | - §5.4: Fixed `scalar_t` → `scalar` for theta variable (matching original line 133) |
| 2025-12-31 | 1.5 | Cross-validation fixes (round 5): |
| | | - §5.4: Added missing `assert(link.step == payload.step)` in queue population (matching original line 42) |

---

*This document provides implementation-ready details for Phase 3 of the work redistribution optimization.*
