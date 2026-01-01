# Chunked Propagator Implementation Plan

**Date:** 2025-12-31
**Status:** Implementation Ready
**Reference:** `doc/chunked_propagator_redesign_plan.md` (Technical Survey)
**Issue:** GitHub #851 - Work Redistribution

---

## Executive Summary

This document provides a detailed implementation plan for chunked propagation with work redistribution. The current work-stealing implementation in `propagate_to_next_surface.ipp` (lines 273-357) provides block-level redistribution but does not checkpoint propagation state for true chunked execution.

**Goal:** Enable iteration-level checkpointing so threads completing propagation early can steal work from slower threads mid-propagation, not just at propagation boundaries.

**Current State:** Block-level work-stealing queue implemented (lines 288-356 in `propagate_to_next_surface.ipp`)

**Target State:** Iteration-level chunked propagation with state serialization (~240 bytes/track)

---

## 1. Architecture Overview

### 1.1 Current Flow (Block-Level Redistribution)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUDA Block (128 threads)                      │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Thread 0 populates work queue                         │
│           ↓                                                      │
│  Phase 2: All threads atomically dequeue work items             │
│           ↓                                                      │
│  Phase 3: Each thread runs FULL propagator.propagate()          │
│           (Variable time: some threads finish early, wait)       │
│           ↓                                                      │
│  Phase 4: blockOr() terminates when all work complete           │
└─────────────────────────────────────────────────────────────────┘
```

**File:** `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp:273-357`

### 1.2 Target Flow (Iteration-Level Redistribution)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUDA Block (128 threads)                      │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Thread 0 populates work queue                         │
│           ↓                                                      │
│  Phase 2: All threads atomically dequeue work items             │
│           ↓                                                      │
│  Phase 3: Each thread runs CHUNK_SIZE iterations                │
│           - Checkpoint state to shared memory                    │
│           - Check if propagation complete                        │
│           - If complete: dequeue next work item                  │
│           - If incomplete: continue with same track              │
│           ↓                                                      │
│  Phase 4: blockOr() terminates when all work complete           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Data Structures

**Current work item** (`propagate_shared_payload.hpp:13-18`):
```cpp
struct propagation_work_item {
    unsigned int param_id;  // 4 bytes
    unsigned int link_idx;  // 4 bytes
    unsigned int n_cands;   // 4 bytes
};  // Total: 12 bytes
```

**New chunked work item** (proposed):
```cpp
struct chunked_propagation_state {
    // === Track identification (12 bytes) ===
    unsigned int param_id;
    unsigned int link_idx;
    unsigned int n_cands;

    // === Propagation progress (8 bytes) ===
    unsigned int iteration;      // Current loop iteration
    bool is_complete;            // Propagation finished
    unsigned char padding[3];

    // === Minimal checkpoint (~220 bytes) ===
    // Track state (32 bytes)
    float bound_params[6];       // loc0, loc1, phi, theta, qop, time
    uint64_t surface_barcode;    // Current surface

    // Covariance (144 bytes) - 6x6 symmetric, store upper triangle
    float covariance[21];        // Upper triangle + diagonal

    // Navigation state (8 bytes)
    uint16_t volume_index;
    int8_t nav_next;
    int8_t nav_last;
    uint8_t trust_level;
    uint8_t nav_direction;
    uint8_t nav_status;
    uint8_t padding2;

    // CKF aborter state (12 bytes)
    unsigned int ckf_count;
    float path_from_surface;
    bool ckf_success;

    // Material interactor flags (3 bytes)
    bool do_energy_loss;
    bool do_multiple_scattering;
    bool do_covariance_transport;

    // Stepper hints (12 bytes)
    float step_size;
    float path_length;
    float abs_path_length;
};  // Total: ~240 bytes
```

---

## 2. Implementation Phases

### Phase 1: Add Iteration Counting to Detray Propagator

**Effort:** Low
**Files to modify:**
- `extern/detray/core/include/detray/propagator/propagator.hpp`

#### 2.1.1 Before (propagator.hpp:155-159)

```cpp
// Is the propagation still alive?
bool _heartbeat = false;

typename stepper_t::state _stepping;
typename navigator_t::state _navigation;
context_type _context;
```

#### 2.1.2 After

```cpp
// Is the propagation still alive?
bool _heartbeat = false;

// Iteration tracking for chunked propagation
unsigned int _iteration{0u};
unsigned int _max_iterations{0u};  // 0 = unlimited

typename stepper_t::state _stepping;
typename navigator_t::state _navigation;
context_type _context;
```

#### 2.1.3 Before (propagator.hpp:237)

```cpp
for (unsigned int i = 0; i % 2 == 0 || propagation.is_alive(); ++i) {
```

#### 2.1.4 After

```cpp
for (unsigned int i = propagation._iteration;
     (i % 2 == 0 || propagation.is_alive()) &&
     (propagation._max_iterations == 0u ||
      propagation._iteration < propagation._max_iterations);
     ++i) {

    propagation._iteration = i;
```

#### 2.1.5 New Methods to Add (propagator.hpp after line 185)

```cpp
/// @brief Set maximum iterations for chunked propagation
/// @param max_iter Maximum iterations (0 = unlimited)
DETRAY_HOST_DEVICE
void set_max_iterations(state& propagation, unsigned int max_iter) const {
    propagation._max_iterations = max_iter;
}

/// @brief Check if propagation reached iteration limit (paused, not complete)
DETRAY_HOST_DEVICE
bool is_chunk_complete(const state& propagation) const {
    return propagation._max_iterations > 0u &&
           propagation._iteration >= propagation._max_iterations &&
           propagation._navigation.is_alive();
}

/// @brief Resume propagation from checkpoint
DETRAY_HOST_DEVICE
void resume_from_iteration(state& propagation, unsigned int start_iter) const {
    propagation._iteration = start_iter;
    propagation._heartbeat = true;
}
```

---

### Phase 2: State Serialization Functions

**Effort:** Medium
**Files to create/modify:**
- `device/common/include/traccc/finding/device/propagation_checkpoint.hpp` (NEW)
- `device/common/include/traccc/finding/device/impl/propagation_checkpoint.ipp` (NEW)

#### 2.2.1 New File: propagation_checkpoint.hpp

```cpp
/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_parameters.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

/// Minimal checkpoint for chunked propagation (~240 bytes)
/// Cross-reference: doc/chunked_propagator_redesign_plan.md §5.1-5.2
struct propagation_checkpoint {
    // === Track identification (12 bytes) ===
    unsigned int param_id;
    unsigned int link_idx;
    unsigned int n_cands;

    // === Propagation progress (4 bytes) ===
    unsigned int iteration;

    // === Track parameters (32 bytes) ===
    // Reference: detray/tracks/bound_track_parameters.hpp
    scalar bound_params[6];      // loc0, loc1, phi, theta, qop, time
    uint64_t surface_barcode;

    // === Covariance upper triangle (84 bytes for float) ===
    // 6x6 symmetric matrix: store 21 elements (diagonal + upper)
    // Reference: doc/chunked_propagator_redesign_plan.md §5.1
    scalar covariance_upper[21];

    // === Navigation state (8 bytes) ===
    // Reference: detray/navigation/navigation_state.hpp:620-651
    uint16_t volume_index;
    int8_t nav_next;
    int8_t nav_last;
    uint8_t trust_level;         // 0=no_trust, 1=fair, 3=high, 4=full
    int8_t nav_direction;        // -1=backward, 1=forward
    int8_t nav_status;
    uint8_t _pad1;

    // === CKF aborter state (12 bytes) ===
    // Reference: core/include/traccc/finding/actors/ckf_aborter.hpp:25-35
    unsigned int ckf_count;
    scalar path_from_surface;
    bool ckf_success;
    uint8_t _pad2[3];

    // === Material interactor flags (4 bytes) ===
    // Reference: interaction_register.hpp:33-40
    bool do_energy_loss;
    bool do_multiple_scattering;
    bool do_covariance_transport;
    uint8_t _pad3;

    // === Stepper state (12 bytes) ===
    // Reference: detray/propagator/base_stepper.hpp:254-260
    scalar step_size;
    scalar path_length;
    scalar abs_path_length;

    // === Status flags (4 bytes) ===
    bool is_complete;
    bool is_alive;
    uint8_t _pad4[2];
};

// Verify size at compile time
static_assert(sizeof(propagation_checkpoint) <= 256,
              "propagation_checkpoint exceeds 256 bytes");

/// @brief Checkpoint propagation state for work redistribution
/// @tparam propagator_t The propagator type
/// @param propagation Current propagation state
/// @param s6 CKF aborter state
/// @param s3 Material interactor state
/// @param checkpoint Output checkpoint buffer
template <typename propagator_t, typename aborter_state_t,
          typename interactor_state_t>
TRACCC_HOST_DEVICE void checkpoint_propagation(
    const typename propagator_t::state& propagation,
    const aborter_state_t& s6,
    const interactor_state_t& s3,
    unsigned int param_id,
    unsigned int link_idx,
    unsigned int n_cands,
    propagation_checkpoint& checkpoint);

/// @brief Restore propagation state from checkpoint
/// @tparam propagator_t The propagator type
/// @param checkpoint Input checkpoint
/// @param propagation Output propagation state to restore
/// @param s6 CKF aborter state to restore
/// @param s3 Material interactor state to restore
template <typename propagator_t, typename aborter_state_t,
          typename interactor_state_t>
TRACCC_HOST_DEVICE void restore_propagation(
    const propagation_checkpoint& checkpoint,
    typename propagator_t::state& propagation,
    aborter_state_t& s6,
    interactor_state_t& s3);

}  // namespace traccc::device

#include "impl/propagation_checkpoint.ipp"
```

#### 2.2.2 New File: impl/propagation_checkpoint.ipp

```cpp
/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template <typename propagator_t, typename aborter_state_t,
          typename interactor_state_t>
TRACCC_HOST_DEVICE void checkpoint_propagation(
    const typename propagator_t::state& propagation,
    const aborter_state_t& s6,
    const interactor_state_t& s3,
    unsigned int param_id,
    unsigned int link_idx,
    unsigned int n_cands,
    propagation_checkpoint& cp) {

    // Track identification
    cp.param_id = param_id;
    cp.link_idx = link_idx;
    cp.n_cands = n_cands;

    // Propagation progress
    cp.iteration = propagation._iteration;

    // Bound track parameters
    // Reference: detray/tracks/bound_track_parameters.hpp
    const auto& bound = propagation._stepping.bound_params();
    const auto& vec = bound.vector();
    for (unsigned int i = 0; i < 6; ++i) {
        cp.bound_params[i] = getter::element(vec, i, 0u);
    }
    cp.surface_barcode = bound.surface_link().value();

    // Covariance upper triangle (6x6 symmetric)
    // Store: [0,0], [0,1], [0,2], [0,3], [0,4], [0,5],
    //              [1,1], [1,2], [1,3], [1,4], [1,5],
    //                     [2,2], [2,3], [2,4], [2,5],
    //                            [3,3], [3,4], [3,5],
    //                                   [4,4], [4,5],
    //                                          [5,5]
    const auto& cov = bound.covariance();
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 6; ++i) {
        for (unsigned int j = i; j < 6; ++j) {
            cp.covariance_upper[idx++] = getter::element(cov, i, j);
        }
    }

    // Navigation state
    // Reference: detray/navigation/navigation_state.hpp:620-651
    const auto& nav = propagation._navigation;
    cp.volume_index = static_cast<uint16_t>(nav.volume());
    cp.nav_next = static_cast<int8_t>(nav.next_index());
    cp.nav_last = static_cast<int8_t>(nav.last_index());
    cp.trust_level = static_cast<uint8_t>(nav.trust_level());
    cp.nav_direction = static_cast<int8_t>(nav.direction());
    cp.nav_status = static_cast<int8_t>(nav.status());

    // CKF aborter state
    // Reference: traccc/finding/actors/ckf_aborter.hpp:25-35
    cp.ckf_count = s6.count;
    cp.path_from_surface = s6.path_from_surface;
    cp.ckf_success = s6.success;

    // Material interactor flags
    // Reference: interaction_register.hpp modifies these at runtime
    cp.do_energy_loss = s3.do_energy_loss;
    cp.do_multiple_scattering = s3.do_multiple_scattering;
    cp.do_covariance_transport = s3.do_covariance_transport;

    // Stepper state
    // Reference: detray/propagator/base_stepper.hpp:254-260
    const auto& step = propagation._stepping;
    cp.step_size = step.step_size();
    cp.path_length = step.path_length();
    cp.abs_path_length = step.abs_path_length();

    // Status flags
    cp.is_complete = !propagation._navigation.is_alive() || s6.success;
    cp.is_alive = propagation._heartbeat;
}

template <typename propagator_t, typename aborter_state_t,
          typename interactor_state_t>
TRACCC_HOST_DEVICE void restore_propagation(
    const propagation_checkpoint& cp,
    typename propagator_t::state& propagation,
    aborter_state_t& s6,
    interactor_state_t& s3) {

    using scalar_t = typename propagator_t::detector_type::scalar_type;
    // TODO: Verify these type aliases exist in detray propagator API
    using algebra_t = typename propagator_t::detector_type::algebra_type;
    using bound_vector_t = typename bound_track_parameters<algebra_t>::vector_type;
    using bound_matrix_t = typename bound_track_parameters<algebra_t>::covariance_type;

    // Restore iteration counter
    propagation._iteration = cp.iteration;

    // Restore bound track parameters
    auto& bound = propagation._stepping.bound_params();
    bound_vector_t vec;
    for (unsigned int i = 0; i < 6; ++i) {
        getter::element(vec, i, 0u) = static_cast<scalar_t>(cp.bound_params[i]);
    }
    bound.set_vector(vec);
    bound.set_surface_link(detray::geometry::barcode{cp.surface_barcode});

    // Restore covariance from upper triangle
    bound_matrix_t cov;
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 6; ++i) {
        for (unsigned int j = i; j < 6; ++j) {
            const scalar_t val = static_cast<scalar_t>(cp.covariance_upper[idx++]);
            getter::element(cov, i, j) = val;
            if (i != j) {
                getter::element(cov, j, i) = val;  // Symmetric
            }
        }
    }
    bound.set_covariance(cov);

    // Restore navigation state
    // NOTE: Trust level set to e_fair after restore if cache not serialized
    // Reference: doc/chunked_propagator_redesign_plan.md §8.3
    // TODO: Verify set_volume() and set_fair_trust() exist in detray navigation API
    //       Only set_high_trust() and set_direction() are confirmed used in traccc
    auto& nav = propagation._navigation;
    nav.set_volume(cp.volume_index);
    // NOTE: set_fair_trust() in detray actually calls set_no_trust() internally
    // (see navigation_state.hpp:226). This forces full cache rebuild on next step.
    nav.set_fair_trust();

    // Restore CKF aborter state
    s6.count = cp.ckf_count;
    s6.path_from_surface = cp.path_from_surface;
    s6.success = cp.ckf_success;

    // Restore material interactor flags
    s3.do_energy_loss = cp.do_energy_loss;
    s3.do_multiple_scattering = cp.do_multiple_scattering;
    s3.do_covariance_transport = cp.do_covariance_transport;

    // Restore stepper state
    auto& step = propagation._stepping;
    step.set_step_size(cp.step_size);
    // NOTE: path_length and abs_path_length are not restored here.
    // They accumulate during propagation and resetting to checkpoint values
    // would require internal stepper modifications. For CKF purposes, only
    // the step_size affects the next RK4 step; path lengths are informational.

    // Restore heartbeat
    propagation._heartbeat = cp.is_alive && !cp.is_complete;
}

/// @brief Reconstruct bound track parameters from checkpoint
/// @tparam propagator_t The propagator type (used to deduce algebra type)
/// @param checkpoint The checkpoint containing serialized state
/// @return Reconstructed bound track parameters
/// @note Returns bound_track_parameters<> (default algebra) for compatibility
template <typename propagator_t>
TRACCC_HOST_DEVICE bound_track_parameters<> reconstruct_bound_params(
    const propagation_checkpoint& cp) {

    using scalar_t = typename propagator_t::detector_type::scalar_type;
    // Use default algebra to match bound_track_parameters<> return type
    using bound_vector_t = typename bound_track_parameters<>::vector_type;
    using bound_matrix_t = typename bound_track_parameters<>::covariance_type;

    // Reconstruct parameter vector
    bound_vector_t vec;
    for (unsigned int i = 0; i < 6; ++i) {
        getter::element(vec, i, 0u) = static_cast<scalar_t>(cp.bound_params[i]);
    }

    // Reconstruct covariance from upper triangle
    bound_matrix_t cov;
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 6; ++i) {
        for (unsigned int j = i; j < 6; ++j) {
            const scalar_t val = static_cast<scalar_t>(cp.covariance_upper[idx++]);
            getter::element(cov, i, j) = val;
            if (i != j) {
                getter::element(cov, j, i) = val;  // Symmetric
            }
        }
    }

    // Construct bound track parameters with default algebra
    return bound_track_parameters<>{
        detray::geometry::barcode{cp.surface_barcode}, vec, cov};
}

}  // namespace traccc::device
```

---

### Phase 3: Chunked Propagation Kernel

**Effort:** High
**Files to modify:**
- `device/common/include/traccc/finding/device/propagate_shared_payload.hpp`
- `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`
- `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh`

#### 2.3.1 Update propagate_shared_payload.hpp

**Before:**
```cpp
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
```

**After:**
```cpp
/// Chunk size for propagation iterations (namespace-level to avoid
/// issues with aggregate initialization of structs with references)
inline constexpr unsigned int PROPAGATION_CHUNK_SIZE = 10;

/// Shared memory payload for chunked propagation kernel
struct propagate_shared_payload {
    // === Work queue (original) ===
    propagation_work_item* queue;
    unsigned int& queue_head;
    unsigned int& queue_tail;
    unsigned int& completed_count;

    // === Checkpoint buffer for chunked propagation ===
    propagation_checkpoint* checkpoints;  // Per-thread checkpoint slots
};
```

#### 2.3.2 Update propagate_to_next_surface.ipp - Main Function

**Before (lines 273-357):**
```cpp
/// Main persistent threads function with work-stealing queue
template <typename propagator_t, typename bfield_t,
          concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    const propagate_shared_payload& shared_payload) {

    // ... Phase 1: Queue population (lines 288-317) ...

    barrier.blockBarrier();

    // Phase 2: Work-stealing loop
    bool has_work = true;
    while (barrier.blockOr(has_work)) {
        has_work = false;

        unsigned int my_idx =
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared_payload.queue_head)
                .fetch_add(1u);

        if (my_idx < shared_payload.queue_tail) {
            has_work = true;
            const propagation_work_item& item = shared_payload.queue[my_idx];

            // Execute FULL propagation for this track
            process_propagation_work_item<propagator_t, bfield_t>(cfg, payload, item);

            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared_payload.completed_count)
                .fetch_add(1u);
        }
    }
}
```

**After:**
```cpp
/// Main chunked propagation function with work-stealing and checkpointing
template <typename propagator_t, typename bfield_t,
          concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    const propagate_shared_payload& shared_payload) {

    using scalar_t = typename propagator_t::detector_type::scalar_type;

    // Thread-local checkpoint slot
    const unsigned int tid = thread_id.getLocalThreadIdX();
    propagation_checkpoint& my_checkpoint = shared_payload.checkpoints[tid];

    // Initialize checkpoint as empty
    my_checkpoint.is_complete = true;
    my_checkpoint.iteration = 0;

    // =========================================================================
    // PHASE 1: Populate work queue (thread 0 only) - UNCHANGED
    // =========================================================================
    if (tid == 0) {
        shared_payload.queue_head = 0;
        shared_payload.queue_tail = 0;
        shared_payload.completed_count = 0;

        const unsigned int block_start =
            thread_id.getBlockIdX() * thread_id.getBlockDimX();
        const unsigned int block_end =
            std::min(block_start + thread_id.getBlockDimX(), payload.n_in_params);

        vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);
        vecmem::device_vector<const candidate_link> links(payload.links_view);
        vecmem::device_vector<const unsigned int> params_liveness_const(
            payload.params_liveness_view);

        for (unsigned int i = block_start; i < block_end; ++i) {
            const unsigned int param_id = param_ids.at(i);
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

    barrier.blockBarrier();

    // =========================================================================
    // PHASE 2: Chunked work-stealing loop with checkpointing
    // =========================================================================
    bool has_work = true;

    while (barrier.blockOr(has_work)) {
        has_work = false;

        // Check if we have an incomplete propagation to resume
        if (!my_checkpoint.is_complete) {
            has_work = true;

            // Resume propagation from checkpoint
            process_chunked_propagation<propagator_t, bfield_t>(
                cfg, payload, my_checkpoint, PROPAGATION_CHUNK_SIZE);

            // If still incomplete, continue next iteration
            if (!my_checkpoint.is_complete) {
                continue;
            }

            // Propagation complete - update outputs and mark done
            finalize_propagation<propagator_t, bfield_t>(cfg, payload, my_checkpoint);

            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared_payload.completed_count)
                .fetch_add(1u);
        }

        // Try to dequeue a new work item
        unsigned int my_idx =
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>(
                shared_payload.queue_head)
                .fetch_add(1u);

        if (my_idx < shared_payload.queue_tail) {
            has_work = true;
            const propagation_work_item& item = shared_payload.queue[my_idx];

            // Initialize checkpoint for new track
            my_checkpoint.param_id = item.param_id;
            my_checkpoint.link_idx = item.link_idx;
            my_checkpoint.n_cands = item.n_cands;
            my_checkpoint.iteration = 0;
            my_checkpoint.is_complete = false;
            my_checkpoint.is_alive = true;
            my_checkpoint.ckf_count = 0;
            my_checkpoint.path_from_surface = 0.f;
            my_checkpoint.ckf_success = false;
            // Material interactor flags default to true (apply interactions)
            my_checkpoint.do_energy_loss = true;
            my_checkpoint.do_multiple_scattering = true;
            my_checkpoint.do_covariance_transport = true;

            // Start first chunk
            process_chunked_propagation<propagator_t, bfield_t>(
                cfg, payload, my_checkpoint, PROPAGATION_CHUNK_SIZE);

            // Check if completed in first chunk
            if (my_checkpoint.is_complete) {
                finalize_propagation<propagator_t, bfield_t>(cfg, payload, my_checkpoint);

                vecmem::device_atomic_ref<unsigned int,
                                          vecmem::device_address_space::local>(
                    shared_payload.completed_count)
                    .fetch_add(1u);
            }
        }
    }
}
```

#### 2.3.3 New Helper: process_chunked_propagation

```cpp
/// Process a chunk of propagation iterations
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void process_chunked_propagation(
    const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    propagation_checkpoint& checkpoint,
    unsigned int chunk_size) {

    using scalar_t = typename propagator_t::detector_type::scalar_type;

    // Access containers
    typename propagator_t::detector_type det(payload.det_data);
    bound_track_parameters_collection_types::device params(payload.params_view);
    vecmem::device_vector<unsigned int> params_liveness(
        payload.params_liveness_view);

    // Skip if track already dead
    if (params_liveness.at(checkpoint.param_id) == 0u) {
        checkpoint.is_complete = true;
        return;
    }

    // Get input parameters
    bound_track_parameters<> in_par;
    if (checkpoint.iteration == 0) {
        // First chunk: read from global memory
        in_par = params.at(checkpoint.param_id);
    } else {
        // Resuming: reconstruct from checkpoint
        // Note: This requires the checkpoint to have full bound params
        in_par = reconstruct_bound_params<propagator_t>(checkpoint);
    }

    // Create propagator with iteration limit
    auto prop_cfg{cfg.propagation};
    prop_cfg.navigation.estimate_scattering_noise = false;
    propagator_t propagator(prop_cfg);

    // Create propagator state
    typename propagator_t::state propagation(in_par, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, in_par));
    propagation._stepping
        .template set_constraint<detray::step::constraint::e_accuracy>(
            cfg.propagation.stepping.step_constraint);

    // Resume from checkpoint iteration
    propagation._iteration = checkpoint.iteration;
    propagation._max_iterations = checkpoint.iteration + chunk_size;

    // Setup actor states
    using actor_tuple_type =
        typename propagator_t::actor_chain_type::actor_tuple;
    typename detray::detail::tuple_element<0, actor_tuple_type>::type::state s0{};
    typename detray::detail::tuple_element<1, actor_tuple_type>::type::state s1{};
    typename detray::detail::tuple_element<3, actor_tuple_type>::type::state s3{};
    typename detray::detail::tuple_element<2, actor_tuple_type>::type::state s2{s3};
    typename detray::detail::tuple_element<4, actor_tuple_type>::type::state s4{prop_cfg};
    typename detray::detail::tuple_element<5, actor_tuple_type>::type::state s5;
    typename detray::detail::tuple_element<6, actor_tuple_type>::type::state s6;

    // Restore actor states from checkpoint (only for resumed propagation)
    // For iteration=0, use default-constructed values; for iteration>0, restore
    s6.count = checkpoint.ckf_count;
    s6.path_from_surface = checkpoint.path_from_surface;
    s6.success = checkpoint.ckf_success;
    if (checkpoint.iteration > 0) {
        // Only restore material interactor flags when resuming
        // (for iteration=0, s3 uses default values from constructor)
        s3.do_energy_loss = checkpoint.do_energy_loss;
        s3.do_multiple_scattering = checkpoint.do_multiple_scattering;
        s3.do_covariance_transport = checkpoint.do_covariance_transport;
    }

    s5.min_pT(static_cast<scalar_t>(cfg.min_pT));
    s5.min_p(static_cast<scalar_t>(cfg.min_p));
    s6.min_step_length = cfg.min_step_length_for_next_surface;
    s6.max_count = cfg.max_step_counts_for_next_surface;

    // MBF smoother handling
    if (cfg.run_mbf_smoother) {
        assert(payload.tmp_jacobian_ptr != nullptr);
        if (checkpoint.iteration == 0) {
            payload.tmp_jacobian_ptr[checkpoint.param_id] = matrix::identity<
                bound_matrix<typename propagator_t::detector_type::algebra_type>>();
        }
        s1._full_jacobian_ptr = &payload.tmp_jacobian_ptr[checkpoint.param_id];
    }

    // Execute chunk of propagation
    propagator.propagate(propagation, detray::tie(s0, s1, s2, s3, s4, s5, s6));

    // Checkpoint current state
    checkpoint_propagation<propagator_t>(
        propagation, s6, s3,
        checkpoint.param_id, checkpoint.link_idx, checkpoint.n_cands,
        checkpoint);

    // Check if propagation is truly complete (not just chunk limit)
    checkpoint.is_complete = s6.success || !propagation._navigation.is_alive();

    // Store result parameters if complete
    if (checkpoint.is_complete && s6.success) {
        params[checkpoint.param_id] = propagation._stepping.bound_params();
        params_liveness[checkpoint.param_id] = 1u;

        // Validate results
        const scalar theta = params[checkpoint.param_id].theta();
        if (theta <= 0.f || theta >= 2.f * constant<traccc::scalar>::pi) {
            params_liveness[checkpoint.param_id] = 0u;
        }
        if (!std::isfinite(params[checkpoint.param_id].phi())) {
            params_liveness[checkpoint.param_id] = 0u;
        }
        if (math::fabs(params[checkpoint.param_id].qop()) == 0.f) {
            params_liveness[checkpoint.param_id] = 0u;
        }
    } else if (checkpoint.is_complete) {
        params_liveness[checkpoint.param_id] = 0u;
    }
}
```

#### 2.3.4 New Helper: finalize_propagation

```cpp
/// Finalize propagation results (create tips if needed)
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void finalize_propagation(
    const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    const propagation_checkpoint& checkpoint) {

    vecmem::device_vector<unsigned int> params_liveness(
        payload.params_liveness_view);
    vecmem::device_vector<unsigned int> tips(payload.tips_view);
    vecmem::device_vector<unsigned int> tip_lengths(payload.tip_lengths_view);

    // Create tip if track ended with enough candidates
    if (params_liveness[checkpoint.param_id] == 0 &&
        checkpoint.n_cands >= cfg.min_track_candidates_per_track) {
        auto tip_pos = tips.push_back(checkpoint.link_idx);
        tip_lengths.at(tip_pos) = checkpoint.n_cands;
    }
}
```

---

### Phase 4: Update CUDA Kernel

**Effort:** Low
**Files to modify:**
- `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh`
- `device/cuda/src/finding/combinatorial_kalman_filter.cuh`

#### 2.4.1 Update Kernel (propagate_to_next_surface_src.cuh)

**Before:**
```cpp
template <typename propagator_t, typename bfield_t>
__global__ __launch_bounds__(128) void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    __shared__ unsigned int queue_head;
    __shared__ unsigned int queue_tail;
    __shared__ unsigned int completed_count;

    extern __shared__ unsigned int shared_mem[];
    device::propagation_work_item* queue =
        reinterpret_cast<device::propagation_work_item*>(shared_mem);

    cuda::barrier barrier;
    details::thread_id1 thread_id;

    device::propagate_to_next_surface<propagator_t, bfield_t>(
        thread_id, barrier, cfg, payload,
        {queue, queue_head, queue_tail, completed_count});
}
```

**After:**
```cpp
template <typename propagator_t, typename bfield_t>
__global__ __launch_bounds__(128) void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    // === Shared memory for work queue ===
    __shared__ unsigned int queue_head;
    __shared__ unsigned int queue_tail;
    __shared__ unsigned int completed_count;

    // Dynamic shared memory layout:
    // [0, queue_size): propagation_work_item queue
    // [queue_size, queue_size + checkpoint_size): propagation_checkpoint array
    extern __shared__ unsigned int shared_mem[];

    // queue_items = number of work items in queue (not bytes)
    const unsigned int queue_items = blockDim.x * 2;  // 2x block size for queue

    device::propagation_work_item* queue =
        reinterpret_cast<device::propagation_work_item*>(shared_mem);

    // Checkpoint buffer starts after queue
    // Calculate offset in unsigned int units: (items * bytes_per_item) / sizeof(uint)
    // Each thread gets its own checkpoint slot
    device::propagation_checkpoint* checkpoints =
        reinterpret_cast<device::propagation_checkpoint*>(
            shared_mem + (queue_items * sizeof(device::propagation_work_item) /
                         sizeof(unsigned int)));

    cuda::barrier barrier;
    details::thread_id1 thread_id;

    device::propagate_to_next_surface<propagator_t, bfield_t>(
        thread_id, barrier, cfg, payload,
        {queue, queue_head, queue_tail, completed_count, checkpoints});
}
```

#### 2.4.2 Update Shared Memory Calculation (combinatorial_kalman_filter.cuh)

**Before (lines 516-519):**
```cpp
// Calculate shared memory size for work-stealing queue
// Queue size = 2 * blockDim.x items (extern __shared__ portion)
const std::size_t shared_size =
    2 * nThreads * sizeof(device::propagation_work_item);
```

**After:**
```cpp
// Calculate shared memory size for chunked propagation
// Layout: [work queue] + [checkpoint buffer]
const std::size_t queue_size = 2 * nThreads * sizeof(device::propagation_work_item);
const std::size_t checkpoint_size = nThreads * sizeof(device::propagation_checkpoint);
const std::size_t shared_size = queue_size + checkpoint_size;

// Verify shared memory fits (48 KB default, 96 KB max on V100)
assert(shared_size <= 49152 && "Shared memory exceeds 48 KB limit");
```

---

## 3. Shared Memory Budget Analysis

Reference: `doc/chunked_propagator_redesign_plan.md` §5.5

### 3.1 Per-Block Memory Usage

| Component | Size per Thread | 128 Threads | 256 Threads |
|-----------|-----------------|-------------|-------------|
| Work queue item | 12 bytes × 2 | 3,072 bytes | 6,144 bytes |
| Checkpoint | ~240 bytes | 30,720 bytes | 61,440 bytes |
| Queue counters | 12 bytes | 12 bytes | 12 bytes |
| **Total** | | **33,804 bytes** | **67,596 bytes** |

### 3.2 Recommendation

- **128 threads/block:** Fits in 48 KB shared memory (33.8 KB used)
- **256 threads/block:** Requires 96 KB configurable shared memory

Use `cudaFuncSetAttribute()` for larger blocks:
```cpp
cudaFuncSetAttribute(
    kernels::propagate_to_next_surface<propagator_t, bfield_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    98304  // 96 KB
);
```

---

## 4. File Reference Summary

### Files to Create

| File | Purpose |
|------|---------|
| `device/common/include/traccc/finding/device/propagation_checkpoint.hpp` | Checkpoint struct definition |
| `device/common/include/traccc/finding/device/impl/propagation_checkpoint.ipp` | Serialization implementation |

### Files to Modify

| File | Changes |
|------|---------|
| `extern/detray/core/include/detray/propagator/propagator.hpp` | Add iteration tracking (lines 155, 237) |
| `device/common/include/traccc/finding/device/propagate_shared_payload.hpp` | Add checkpoint buffer pointer |
| `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | Chunked loop implementation |
| `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh` | Kernel shared memory layout |
| `device/cuda/src/finding/combinatorial_kalman_filter.cuh` | Shared memory size calculation |

### Cross-References to Technical Survey

| Implementation Section | Survey Reference |
|------------------------|------------------|
| Checkpoint struct | §5.1-5.2 (Serialization Requirements) |
| Navigation state restore | §3.1, §8.3 (Trust Level) |
| Actor states | §4 (CKF Aborter), interaction_register.hpp |
| Memory budget | §5.5 (Revised Size Estimates) |
| Loop structure | §1.2-1.3 (ANSNANSN pattern) |

---

## 5. Testing Strategy

### 5.1 Unit Tests

1. **Checkpoint/Restore correctness**
   - Checkpoint propagation state after N iterations
   - Restore and continue
   - Compare final state with non-chunked propagation

2. **Work-stealing validation**
   - Inject artificial delays in some tracks
   - Verify faster threads pick up work from slower ones
   - Measure utilization improvement

### 5.2 Integration Tests

1. **CKF output equivalence**
   - Run chunked vs non-chunked CKF
   - Compare found tracks (should be identical)

2. **Performance benchmarks**
   - ODD detector, 10K tracks
   - Measure kernel time with various CHUNK_SIZE values
   - Optimal CHUNK_SIZE likely 5-20 iterations

### 5.3 Regression Tests

1. **Existing test suites**
   - All `traccc_tests_*` must pass
   - CI integration

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Checkpoint restore introduces numerical drift | Medium | High | Validate with double precision; add tolerance tests |
| Shared memory overflow | Low | High | Compile-time size checks; fallback to smaller blocks |
| Work-stealing overhead exceeds benefit | Medium | Medium | Make CHUNK_SIZE configurable; profile extensively |
| Detray API changes break iteration tracking | Low | Medium | Minimal changes; coordinate with detray maintainers |

---

## 7. Implementation Order

1. **Week 1:** Phase 1 (Detray iteration tracking)
2. **Week 2:** Phase 2 (Checkpoint/restore functions)
3. **Week 3:** Phase 3 (Chunked kernel implementation)
4. **Week 4:** Phase 4 (CUDA integration) + Testing
5. **Week 5:** Performance tuning + Documentation

---

*This implementation plan is based on the technical survey in `doc/chunked_propagator_redesign_plan.md` and cross-validated against the traccc/detray codebase as of 2025-12-31.*
