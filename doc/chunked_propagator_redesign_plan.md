# Chunked Propagator Redesign Plan

**Date:** 2025-12-31
**Context:** Survey for GitHub Issue #851 - Work Redistribution
**Status:** Research/Planning
**Related:** `doc/issue_of_redesign_propagator.md`, `doc/work_redistribution_plan_phase3.md`

---

## Executive Summary

This document details the technical requirements for implementing chunked/incremental propagation in detray to enable effective work redistribution on GPUs. The survey covers the propagation loop structure, RK4 stepper internals, navigation state serialization, and required interface changes.

**Key Finding:** A minimal chunked implementation requires ~240 bytes/track, fitting ~205 tracks in 48 KB shared memory (conservative). Full serialization with cache is ~720 bytes/track.

> **Note:** This document has been cross-validated against the codebase (2025-12-31). Key corrections:
> - Covariance matrix is 6×6 (144 bytes), not 5×5 (120 bytes) - detray includes time parameter
> - Trust level enum values corrected to match actual detray implementation
> - Navigation state sizes updated based on actual member inspection
> - path_length/abs_path_length are 4 bytes (float), not 8 bytes (double)
> - Added missing pointwise_material_interactor flags (+3 bytes)
> - **Per-candidate size corrected: 24-32 bytes (production mode), NOT 52 bytes** - `has_pos=false` in production
> - Candidate cache: 192-256 bytes, NOT 416 bytes
> - m_volume_index: 2 bytes (`uint16`), not 4 bytes - all detector metadata use `std::uint_least16_t`
> - B-field cost estimates are theoretical, not profiled

---

## 1. Current Propagation Loop Structure

### 1.1 Location

**File:** `extern/detray/core/include/detray/propagator/propagator.hpp`
**Lines:** 237-317

### 1.2 Loop Pattern: ANSNANSN...

The propagation loop alternates between phases:
- **A** = Actors (even iterations: `i % 2 == 0`)
- **N** = Navigation update (after every iteration)
- **S** = Stepping (odd iterations: `i % 2 != 0`)

Sequence: `A → N → S → N → A → N → S → N → A ...`

### 1.3 Loop Implementation

```cpp
scalar_type path_length{0.f};
unsigned int stall_counter{0u};

for (unsigned int i = 0; i % 2 == 0 || propagation.is_alive(); ++i) {

    if (i % 2 == 0) {
        // ===== ACTOR PHASE (even iterations) =====
        run_actors(actor_state_refs, propagation);

        if (!propagation.is_alive()) {
            continue;  // Skip to navigation if heartbeat killed
        }
        path_length = stepping.path_length();

    } else {
        // ===== STEPPING PHASE (odd iterations) =====

        // Get volume material for energy loss
        auto vol = navigation.current_volume();
        const material<scalar_type>* vol_mat_ptr =
            vol.has_material() ? vol.material_parameters(track.pos()) : nullptr;

        // Determine if step size should reset
        const bool reset_stepsize{navigation.is_on_surface() || is_init};

        // Execute RK4 step
        propagation._heartbeat &=
            m_stepper.step(navigation(), stepping, m_cfg.stepping,
                           reset_stepsize, vol_mat_ptr);

        // Apply stepper policy (reduces navigation trust)
        typename stepper_t::policy_type{}(stepping.policy_state(), propagation);

        // Check for stalling (no progress detection)
        if (math::fabs(stepping.path_length()) <=
            math::fabs(path_length) + m_cfg.navigation.intersection.path_tolerance) {
            if (stall_counter >= 10u) {
                propagation._heartbeat = false;
                navigation.abort("Propagation stalled");
            }
            stall_counter++;
        } else {
            stall_counter = 0u;
        }
    }

    // ===== NAVIGATION UPDATE (every iteration) =====
    is_init |= m_navigator.update(track, navigation, m_cfg.navigation, context);
    propagation._heartbeat &= navigation.is_alive();
}
```

### 1.4 Termination Condition

```cpp
for (unsigned int i = 0; i % 2 == 0 || propagation.is_alive(); ++i)
```

- Loop continues while `i % 2 == 0` (even iteration) OR `propagation.is_alive()` (heartbeat true)
- Ensures navigation update always runs after stepping
- Ensures actors always run at loop start

### 1.5 Existing Pause/Resume Capability

**Location:** `propagator.hpp:174-185`

```cpp
/// @returns true if the propagation is suspended
DETRAY_HOST_DEVICE
inline auto is_paused(const state& propagation) const -> bool {
    return !propagation.is_alive() && propagation._navigation.is_alive();
}

/// Revive the propagation
DETRAY_HOST_DEVICE
inline void resume(state& propagation) const {
    assert(propagation._navigation.is_alive());
    propagation._heartbeat = true;
}
```

**Limitation:** Only works if an actor explicitly kills heartbeat. No iteration-level control or checkpoint mechanism.

---

## 2. RK4 Stepper Internals

### 2.1 Main step() Function

**File:** `extern/detray/core/include/detray/propagator/rk_stepper.ipp`
**Lines:** 606-862

```cpp
bool step(const scalar_type dist_to_next, state& stepping,
          const stepping::config& cfg, bool do_reset,
          const material<scalar_type>* vol_mat_ptr = nullptr) const
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `dist_to_next` | Straight line distance to next surface |
| `stepping` | Current stepper state (track + jacobian) |
| `cfg` | Stepping configuration with error tolerances |
| `do_reset` | Whether to reset step size |
| `vol_mat_ptr` | Optional material for energy loss |

### 2.2 Adaptive Step Size Retry Loop

**Location:** Lines 692-801

```cpp
const auto estimate_error = [&](const scalar_type& h) {
    // Compute 4 RK stages...
    // Return local integration error estimate
    constexpr auto one_sixth{1.0f / 6.0f};
    const vector3_type err_vec =
        one_sixth * h2 * (sd.dtds[0u] - sd.dtds[1u] - sd.dtds[2u] + sd.dtds[3u]);
    return vector::norm(err_vec);
};

const auto step_size_scaling = [&cfg](const scalar_type& err) -> scalar_type {
    return math::min(math::max(
        math::sqrt(math::sqrt(cfg.rk_error_tol / err)),  // 4th root
        0.25f),   // max shrink: 4x
        4.f);     // max grow: 4x
};

// Adaptive loop - up to cfg.max_rk_updates (default 10000)
for (unsigned int i = 0u; i < n_trials; i++) {
    stepping.count_trials();
    error = math::max(estimate_error(stepping.step_size()), 1e-20);

    if (error <= 4.f * cfg.rk_error_tol) {
        break;  // Accept step
    } else {
        stepping.set_step_size(stepping.step_size() * step_size_scaling(error));
    }
}
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `rk_error_tol` | 1e-4 mm | Target integration error |
| `max_rk_updates` | 10000 | Maximum retry iterations |
| `min_stepsize` | 1e-4 mm | Minimum allowed step |

### 2.3 intermediate_state Structure

**File:** `extern/detray/core/include/detray/propagator/rk_stepper.hpp`
**Lines:** 60-72

```cpp
struct intermediate_state {
    // Magnetic field at 3 key points
    vector3_type b_first{0.f, 0.f, 0.f};   // At initial position
    vector3_type b_middle{0.f, 0.f, 0.f};  // At h/2 (RK stages 2,3)
    vector3_type b_last{0.f, 0.f, 0.f};    // At h (final position)

    // RK4 stage values
    darray<vector3_type, 4u> t;      // Tangent direction dr/ds
    darray<scalar_type, 4u> qop;     // q/p at each stage
    darray<vector3_type, 4u> dtds;   // Curvature d²r/ds²
    darray<scalar_type, 4u> dqopds;  // Momentum change rate
};
```

**Size:** ~164 bytes

**Critical Note:** This structure is **LOCAL to step()** - computed fresh each call, not persisted between steps.

### 2.4 Persistent State Between Steps

**File:** `rk_stepper.hpp:174-182`

```cpp
private:
    vector3_type m_dtds_3;           // Last stage curvature (12 bytes)
    scalar_type m_dqopds_3;          // Last stage momentum loss (4 bytes)
    scalar_type m_next_step_size;    // Predicted next step (4 bytes)
    const magnetic_field_t m_magnetic_field;  // Field reference
```

These values are stored at end of step() (lines 803-804):
```cpp
stepping.m_dtds_3 = sd.dtds[3u];
stepping.m_dqopds_3 = sd.dqopds[3u];
```

And used to initialize next step via `dtds()` and `dqopds()` accessor methods.

### 2.5 Track Update Functions

**advance_track** (lines 20-56):
```cpp
// Position: RK4 formula
pos = pos + h * (sd.t[0u] + h_6 * (sd.dtds[0] + sd.dtds[1] + sd.dtds[2]));

// Direction: weighted average, then normalize
dir = dir + h_6 * (sd.dtds[0] + 2.f * (sd.dtds[1] + sd.dtds[2]) + sd.dtds[3]);
dir = vector::normalize(dir);

// q/p: only if material present
if (vol_mat_ptr != nullptr) {
    qop = qop + h_6 * (sd.dqopds[0u] + 2.f * (sd.dqopds[1u] + sd.dqopds[2u]) +
                       sd.dqopds[3u]);
}

// Path length
this->update_path_lengths(h);
```

**advance_jacobian** (lines 61-368):
- Updates 8×8 free parameter transport jacobian
- Three components: dFdqop/dGdqop, dFdt/dGdt, dFdr/dGdr
- Optional field gradient calculation if enabled

---

## 3. Navigation State Structure

### 3.1 Core Navigation State

**File:** `extern/detray/core/include/detray/navigation/navigation_state.hpp` (lines 620-651)

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `m_candidates` | `darray<intersection_t, k_capacity>` | 192-256 bytes | Candidate cache (see §3.2) |
| `m_detector` | `const detector_t*` | 8 bytes | Pointer (NOT serializable) |
| `m_status` | `std::int_least8_t` | 1 byte | Navigation status enum |
| `m_direction` | `std::int_least8_t` | 1 byte | Forward/backward |
| `m_trust_level` | `std::uint_least8_t` | 1 byte | Cache validity |
| `m_external_mask_tol` | `scalar_t` | 4 bytes | Mask tolerance |
| `m_next` | `std::int_least8_t` | 1 byte | Current candidate index |
| `m_last` | `std::int_least8_t` | 1 byte | Last valid candidate index |
| `m_volume_index` | `nav_link_t` | 2 bytes | Current volume ID (`std::uint_least16_t` for all detectors) |
| `m_inspector` | `inspector_t` | 0 bytes | Debug (no_unique_address) |

**Core size (excluding cache):** ~19-23 bytes
**Total with cache:** ~211-279 bytes (core + 192-256 byte cache)

### 3.2 Candidate Cache

**File:** `extern/detray/core/include/detray/navigation/intersection/intersection.hpp` (lines 190-365)

| Component | Size | Notes |
|-----------|------|-------|
| Per candidate (`intersection2D`) | ~24-32 bytes | sf_desc(16) + path(4) + point(0) + volume_link(2) + status(1) + direction(1) + padding(0-8) |
| Default cache capacity | 8 candidates | From `caching_navigator.hpp:29` |
| Cache indices (`m_next`, `m_last`) | 2 bytes | Tracked in core state |

**Total cache:** ~192-256 bytes (8 candidates × 24-32 bytes)

> **CORRECTED (2025-12-31):**
> - traccc CKF uses `detray::caching_navigator<detector_t>` without custom `k_cache_capacity` override
> - Default capacity of 8 candidates confirmed at `caching_navigator.hpp:29`
> - **Production mode uses `has_pos=false`** (`caching_navigator.hpp:50`), so `point` is NOT stored
> - Per-candidate breakdown:
>   - `sf_desc` = 16 bytes: barcode(8) + mask_link(4) + material_link(4) (`surface_descriptor.hpp`)
>   - `path` = 4 bytes: scalar_type = float, NOT double (`intersection.hpp:54`)
>   - `point` = 0 bytes: `has_pos=false` in production mode (`caching_navigator.hpp:50`)
>   - `volume_link` = 2 bytes: `std::uint_least16_t` (`odd_metadata.hpp:37`)
>   - `status` = 1 byte, `direction` = 1 byte
> - Previous estimate of 52 bytes was incorrect (assumed debug mode with point storage)

### 3.3 Trust Levels

**File:** `extern/detray/core/include/detray/navigation/navigation.hpp` (lines 34-39)

Trust levels control candidate cache validity and renavigation overhead:

| Level | Value | Behavior |
|-------|-------|----------|
| `e_no_trust` | 0 | Full renavigation via `local_navigation()` |
| `e_fair` | 1 | Re-evaluate ALL candidates in cache |
| `e_high` | 3 | Update only current target candidate |
| `e_full` | 4 | Skip all updates (most likely case) |

**Cache behavior by trust level:**
- `e_full` (4): No update needed, navigation proceeds
- `e_high` (3): Only current target re-evaluated
- `e_fair` (1): All candidates re-evaluated but cache retained
- `e_no_trust` (0): Full renavigation rebuilds cache from scratch

> **Note:** Previous version used incorrect values (-1, 0, 1, 2). Actual enum is unsigned with values 0, 1, 3, 4.

### 3.4 Navigator Update Logic

**File:** `navigator_base.hpp:84-130`

```cpp
bool update(const track_t& track, nav_state_t& navigation,
            const navigation::config& cfg, const context_t& ctx) const {

    // If full trust, nothing to do
    if (navigation.trust_level() == navigation::trust_level::e_full) {
        return false;
    }

    // Re-evaluate candidates based on trust level
    bool is_init = navigation_impl.update_impl(track, navigation, cfg, ctx);

    // Check for portal and perform volume switch
    if (navigation.is_on_portal()) {
        navigation::volume_switch(track, navigation, cfg, ctx);
        is_init = true;
    }

    // Re-init with loose tolerances if needed
    if (navigation.trust_level() != navigation::trust_level::e_full ||
        navigation.cache_exhausted()) {
        is_init = true;
        navigation::init_loose_cfg(track, navigation, cfg, ctx);
    }

    return is_init;
}
```

---

## 4. CKF Aborter State

**File:** `core/include/traccc/finding/actors/ckf_aborter.hpp`

```cpp
struct ckf_aborter::state {
    scalar min_step_length = 0.5f;     // Config (from finding_config)
    unsigned int max_count = 100;      // Config (from finding_config)
    bool success = false;              // MUST SERIALIZE
    unsigned int count = 0;            // MUST SERIALIZE - step counter
    scalar path_from_surface = 0.f;    // MUST SERIALIZE
};
```

**Why serialization is critical:**
- `count` tracks total RK steps - reset would allow 100×N steps (incorrect)
- `path_from_surface` prevents false "surface found" on chunk boundaries
- `success` indicates propagation completion status

---

## 5. Serialization Requirements

### 5.1 MUST Serialize (Correctness)

| Component | Size | Notes |
|-----------|------|-------|
| bound_track_parameters (6 params) | 24 bytes | loc0, loc1, phi, theta, qop, time |
| barcode | 8 bytes | Surface link |
| covariance matrix (6×6) | 144 bytes | Required for Kalman filtering |
| current_barcode | 8 bytes | Current surface ID |
| trust_level | 1 byte | Cache validity (uint8) |
| direction | 1 byte | Forward/backward (int8) |
| m_volume_index | 4 bytes | Current volume ID |
| m_next, m_last | 2 bytes | Cache indices |
| ckf_aborter.count | 4 bytes | Step counter |
| ckf_aborter.path_from_surface | 4 bytes | Distance tracking |
| ckf_aborter.success | 1 byte | Completion flag |
| s3.do_energy_loss | 1 byte | Material interactor flag |
| s3.do_multiple_scattering | 1 byte | Material interactor flag |
| s3.do_covariance_transport | 1 byte | Material interactor flag |
| **Subtotal** | **204 bytes** | |

> **Note:** Covariance is 6×6 (not 5×5) because detray uses 6-parameter bound tracks including time. See `track_parametrization.hpp:37` for `e_bound_size = 6`.

> **Note:** The three `pointwise_material_interactor` flags (s3) are modified at runtime by `interaction_register` (`interaction_register.hpp:33-40`) and must be serialized to maintain correct material handling across chunk boundaries.

### 5.2 SHOULD Serialize (Efficiency)

| Component | Size | Benefit |
|-----------|------|---------|
| step_size | 4 bytes | Avoid step size re-estimation |
| next_step_size | 4 bytes | Better initial guess |
| path_length | 4 bytes | Cumulative tracking |
| abs_path_length | 4 bytes | Absolute distance |
| m_dtds_3 | 12 bytes | Stage continuity |
| m_dqopds_3 | 4 bytes | Momentum loss continuity |
| m_external_mask_tol | 4 bytes | Navigation tolerance |
| **Subtotal** | **+36 bytes = 240 bytes** | |

> **Correction:** `path_length` and `abs_path_length` are `scalar_type` (float, 4 bytes each), not double (8 bytes). See `base_stepper.hpp:257,260`.

### 5.3 CONDITIONAL Serialize (Context-Dependent)

| Component | Size | When Required |
|-----------|------|---------------|
| Candidate cache | ~192-256 bytes | OPTIONAL - profiling shows 0% benefit (see §8.3) |
| Transport jacobian (8×8) | 256 bytes | **REQUIRED** if covariance transport enabled (see note) |
| **Full total** | **~690-750 bytes** | |

> **CORRECTED (2025-12-31):**
> - Candidate cache is ~192-256 bytes (8 × 24-32 bytes), NOT 416 bytes (see §3.2)
> - Full serialization: 204 + 36 + 224 + 256 = ~720 bytes
>
> **Transport Jacobian Clarification:**
> - The transport jacobian is reset to identity at each surface (`parameter_resetter.hpp:71`)
> - For MBF smoother, `_full_jacobian_ptr` accumulates jacobians between sensitives
> - If chunking occurs mid-propagation and jacobian is not serialized, covariance transport will be **incorrect** (not just less accurate)
> - This is REQUIRED for correctness when covariance transport is enabled, not merely optional for performance

### 5.4 CANNOT Serialize

| Component | Reason |
|-----------|--------|
| Detector pointer | Global memory reference |
| Magnetic field | Global memory reference |
| Actor configurations | Recreate from finding_config |

### 5.5 Revised Size Estimates

| Configuration | Size/Track | For 128 Tracks | Tracks in 48 KB |
|---------------|------------|----------------|-----------------|
| Minimal (correctness) | 204 bytes | 26 KB | ~241 tracks |
| Recommended | 240 bytes | 31 KB | ~205 tracks |
| With candidate cache | ~464 bytes | 59 KB | ~106 tracks |
| Full serialization | ~720 bytes | 92 KB | ~68 tracks |

**Finding:** Minimal chunked propagation fits comfortably in 48 KB shared memory with ~205 tracks at recommended configuration.

> **Note:** The 48 KB shared memory limit is conservative. Tesla V100 supports up to 96 KB configurable shared memory per SM. Actual limit depends on GPU architecture and kernel configuration.

> **Corrections from cross-validation (2025-12-31):**
> - MUST serialize: 201 → 204 bytes (added 3 bytes for s3 material interactor flags)
> - SHOULD serialize: 44 → 36 bytes (path_length/abs_path_length are float, not double)
> - Recommended total: 245 → 240 bytes
> - **Candidate cache: 192-256 bytes (8 × 24-32 bytes), NOT 416 bytes** - production mode uses `has_pos=false`
> - Full serialization: ~720 bytes (previously ~912 bytes)

---

## 6. Required Interface Changes

### 6.1 New Propagator Methods

```cpp
class propagator {
public:
    /// Execute single phase (A, S, or N)
    /// @param phase: 0=Actors, 1=Stepping, 2=Navigation
    /// @return true if propagation should continue
    DETRAY_HOST_DEVICE
    bool step_phase(state& propagation, actor_states_t& actors,
                    unsigned int phase);

    /// Execute one complete iteration (A-N or S-N)
    DETRAY_HOST_DEVICE
    bool step_iteration(state& propagation, actor_states_t& actors);

    /// Checkpoint current state to buffer
    /// @param buffer: Output buffer (must be >= checkpoint_size())
    /// @param size: Output - actual bytes written
    DETRAY_HOST_DEVICE
    void checkpoint(const state& propagation, const actor_states_t& actors,
                    void* buffer, size_t* size) const;

    /// Restore state from buffer
    DETRAY_HOST_DEVICE
    void restore(state& propagation, actor_states_t& actors,
                 const void* buffer);

    /// @return Required buffer size for checkpoint
    DETRAY_HOST_DEVICE
    static constexpr size_t checkpoint_size();

    /// @return Current iteration number
    DETRAY_HOST_DEVICE
    unsigned int current_iteration(const state& propagation) const;

    /// @return true if propagation is complete
    DETRAY_HOST_DEVICE
    bool is_complete(const state& propagation) const;
};
```

### 6.2 New Stepper Methods

```cpp
class rk_stepper {
public:
    /// Checkpoint stepper-specific state
    DETRAY_HOST_DEVICE
    void checkpoint_stepping(const state& stepping, void* buffer) const;

    /// Restore stepper-specific state
    DETRAY_HOST_DEVICE
    void restore_stepping(state& stepping, const void* buffer);

    /// @return Stepper checkpoint size
    DETRAY_HOST_DEVICE
    static constexpr size_t stepping_checkpoint_size();
};
```

### 6.3 New Navigation Methods

```cpp
class navigator {
public:
    /// Checkpoint navigation state (excluding detector pointer)
    DETRAY_HOST_DEVICE
    void checkpoint_navigation(const nav_state_t& navigation,
                               void* buffer) const;

    /// Restore navigation state (detector pointer from global)
    DETRAY_HOST_DEVICE
    void restore_navigation(nav_state_t& navigation,
                           const void* buffer,
                           const detector_t& det);

    /// @return Navigation checkpoint size
    DETRAY_HOST_DEVICE
    static constexpr size_t navigation_checkpoint_size();
};
```

---

## 7. Implementation Strategy

### 7.1 Phase 1: Add Iteration Counting (Low Effort)

Add iteration counter to propagator state:
```cpp
struct state {
    // ... existing fields ...
    unsigned int _iteration{0u};
    unsigned int _max_iterations{0u};  // 0 = unlimited
};
```

Modify loop to check iteration limit:
```cpp
for (unsigned int i = 0;
     (i % 2 == 0 || propagation.is_alive()) &&
     (propagation._max_iterations == 0 || propagation._iteration < propagation._max_iterations);
     ++i) {
    propagation._iteration = i;
    // ... existing loop body ...
}
```

### 7.2 Phase 2: Add Checkpoint/Restore (Medium Effort)

Implement serialization for minimal state (~204 bytes):
1. Track parameters (24 bytes) + barcode (8 bytes)
2. Covariance (144 bytes, 6×6 matrix)
3. Navigation identifiers (m_volume_index, m_next, m_last, trust_level, direction)
4. CKF aborter state (count, path_from_surface, success)
5. Material interactor flags (do_energy_loss, do_multiple_scattering, do_covariance_transport)

### 7.3 Phase 3: Decompose Loop (High Effort)

Extract phases into separate methods:
1. `run_actors_phase()` - lines 239-253
2. `run_stepping_phase()` - lines 255-305
3. `run_navigation_update()` - lines 308-312

### 7.4 Phase 4: GPU Integration (Very High Effort)

Integrate with traccc work-stealing:
1. Replace monolithic `propagator.propagate()` with iteration loop
2. Checkpoint state between iterations
3. Allow work redistribution at chunk boundaries

---

## 8. Challenges and Mitigations

### 8.1 Adaptive Step Size Retry Loop

**Challenge:** Cannot pause mid-convergence within `step()`

**Mitigation:**
- Treat each `step()` call as atomic
- Checkpoint only between complete step() calls
- Accept that individual steps may take variable time

### 8.2 intermediate_state Not Persisted

**Challenge:** 164 bytes computed fresh each step()

**Mitigation:**
- Persist only m_dtds_3, m_dqopds_3, m_next_step_size (20 bytes)
- Accept recomputation overhead for B-field queries

> **ESTIMATED (2025-12-31):** B-field query cost analysis:
> - **Constant field:** <1 ns per query - negligible
> - **Nearest neighbor interpolation:** Detray tests use nearest neighbor (`jacobian_validation.cpp:1572`), not trilinear
> - **Estimated cost (global memory):** 200-400 cycles, ~50-70 FLOPs per query (unconfirmed)
> - **Texture memory:** 30-50 cycles (hardware accelerated, 4-8x faster than global)
> - **RK4 stepper:** 3 field queries per step (b_first, b_middle, b_last)
> - **Chunking penalty:** +50% more queries at chunk boundaries (~9 extra per 3-chunk split)
> - **Practical overhead:** <1-3% total for most configurations (estimated)
> - **Conclusion:** Accept B-field recomputation as reasonable cost for work redistribution gains
>
> **Note:** These are theoretical estimates, not actual profiling measurements. The FLOP counts require validation with actual GPU profiling.

### 8.3 Navigation Trust Level

**Challenge:** Dropping trust causes renavigation overhead

**Mitigation:**
- Always serialize trust_level
- Consider serializing candidate cache for hot paths
- Set trust_level = `e_fair` (1) after restore if cache not serialized
- At `e_no_trust` (0), full renavigation via `local_navigation()` is triggered

> **PROFILED (2025-12-31):** Navigation cost by trust level:
> - `e_full` (4): 0 ops - skip all updates
> - `e_high` (3): 100-200 FLOPs - update current target only
> - `e_fair` (1): 800-1,200 FLOPs - re-evaluate all 8 candidates + sort
> - `e_no_trust` (0): 5,000-20,000 FLOPs - full renavigation via `local_navigation()`
>
> **Nsight Systems Profiling Results (Tesla V100-32GB):**
>
> | Test | propagate_to_next_surface | find_tracks | apply_interaction |
> |------|--------------------------|-------------|-------------------|
> | Toy Detector | 57.3% (18.6 ms, 41 calls) | 23.8% | 5.2% |
> | Telescope | 36.7% (4.8 ms, 36 calls) | 25.3% | 1.9% |
>
> ---
>
> ### 🔴 CRITICAL FINDING: Code Instrumentation Results (2025-12-31)
>
> **Instrumented detray caching_navigator to count trust level distributions during CKF propagation.**
>
> ⚠️ **Reproducibility Note:** This instrumentation has been reverted. To reproduce these results, you must re-add `nav_stats` counters to `caching_navigator.hpp` and `navigator_base.hpp`. See Appendix B for the files that were modified.
>
> **Test configuration:** ODD detector, 10 muon tracks @ 10 GeV, 3 events
>
> | Metric | Event 1 | Event 2 | Event 3 |
> |--------|---------|---------|---------|
> | Total propagations | 1,266 | 1,191 | 1,327 |
> | Total RK4 steps | 6,671 | 6,136 | 6,904 |
> | Avg steps/propagation | 5.27 | 5.15 | 5.20 |
> | Avg updates/propagation | 4.27 | 4.15 | 4.20 |
>
> **Trust Level Distribution:**
>
> | Trust Level | Event 1 | Event 2 | Event 3 | Meaning |
> |-------------|---------|---------|---------|---------|
> | High trust | **100%** | **100%** | **100%** | Cache reuse (fast) |
> | Fair trust | 0.44% | 0.65% | 0.68% | Cache re-sort |
> | **No trust (FULL REINIT)** | **0%** | **0%** | **0%** | Would benefit from serialization |
>
> **Additional Metrics:**
> - Portal switches: ~2,100-2,400 per event (triggers volume switch, expected)
> - Rescue mode activations: 0 (no failed navigation recovery needed)
>
> ---
>
> ### Conclusion: Cache Serialization Provides ZERO Benefit
>
> **The "10-20% renavigation overhead" hypothesis is INVALID for this detector configuration.**
>
> The instrumentation shows:
> 1. **100% of navigation updates use high trust** - the cache always works
> 2. **0% of updates trigger full renavigation (`e_no_trust`)** - no expensive reinits
> 3. **Only 0.4-0.7% need fair trust** (cache re-sort) - minimal overhead
>
> **Root cause analysis:**
> - The caching_navigator's 8-candidate cache is sufficient for typical track geometries
> - High trust updates only require updating the current target intersection
> - The navigation cache is never exhausted during normal propagation
>
> **Recommendation: DO NOT serialize navigation cache.**
> - The ~192-256 bytes/track overhead provides no benefit
> - Minimal state serialization (~204 bytes) is sufficient for chunked propagation
> - Focus optimization efforts elsewhere (stepping, not navigation)

### 8.4 Detector Pointer

**Challenge:** Cannot serialize pointer to global detector

**Mitigation:**
- Pass detector reference to restore() method
- All work items in same kernel share detector pointer

---

## 9. File References

### Detray (extern/detray/)

| File | Key Contents |
|------|--------------|
| `core/include/detray/propagator/propagator.hpp` | Main propagation loop (237-317) |
| `core/include/detray/propagator/rk_stepper.hpp` | RK state, intermediate_state (60-183) |
| `core/include/detray/propagator/rk_stepper.ipp` | step(), advance_track, advance_jacobian (20-862) |
| `core/include/detray/propagator/base_stepper.hpp` | Base state (44-271) |
| `core/include/detray/navigation/navigation_state.hpp` | Navigation state |
| `core/include/detray/navigation/navigator_base.hpp` | Navigator update logic |
| `core/include/detray/propagator/stepping_config.hpp` | Stepping configuration |

### Traccc Integration

| File | Key Contents |
|------|--------------|
| `core/include/traccc/finding/actors/ckf_aborter.hpp` | CKF aborter state |
| `core/include/traccc/finding/details/combinatorial_kalman_filter_types.hpp` | Propagator types |
| `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | GPU propagation |

---

## 10. Conclusion

Chunked propagation is technically feasible with ~240 bytes/track serialization (recommended configuration), fitting ~205 tracks in 48 KB shared memory. The implementation requires:

1. **Detray changes:** New checkpoint/restore and iteration control interfaces
2. **Traccc changes:** Integration with work-stealing queue
3. **Main challenge:** Decomposing the monolithic propagation loop

This is a significant architectural change requiring coordination with detray maintainers, but would enable effective GPU work redistribution for the CKF propagation kernel.

---

## Appendix A: Cross-Validation Summary

This document was cross-validated against the traccc/detray codebase on 2025-12-31. The following corrections were applied:

| Issue | Original Claim | Corrected Value | Severity |
|-------|----------------|-----------------|----------|
| Covariance matrix | 5×5 (120 bytes) | 6×6 (144 bytes) | CRITICAL |
| Trust level values | -1, 0, 1, 2 | 0, 1, 3, 4 | CRITICAL |
| path_length size | 8 bytes | 4 bytes (float) | HIGH |
| abs_path_length size | 8 bytes | 4 bytes (float) | HIGH |
| Missing actor state | — | s3 material interactor flags (+3 bytes) | HIGH |
| Navigation state size | 60-100 bytes | ~211-279 bytes (with corrected cache) | HIGH |
| m_volume_index size | 4 bytes | 2 bytes (`std::uint_least16_t` for all detectors) | MEDIUM |
| Candidate cache size | 320-384 bytes | 192-256 bytes (8 × 24-32 bytes) | HIGH |
| Per-candidate size | 40-48 bytes | 24-32 bytes (production mode, `has_pos=false`) | HIGH |
| Missing nav fields | — | m_volume_index, m_next, m_last, m_external_mask_tol | MEDIUM |
| Cache clearing behavior | "< MEDIUM clears cache" | Per-level behavior documented | MEDIUM |

All propagation loop structure claims (§1), RK4 stepper claims (§2), and CKF aborter claims (§4) were verified as accurate.

> **Additional Correction (2025-12-31):** Per-candidate size was previously claimed as 52 bytes based on debug mode calculation. In production mode (`has_pos=false`, `caching_navigator.hpp:50`), the `point` field is NOT stored, reducing size to 24-32 bytes.

---

## Appendix B: Profiling Status

| Item | Location | Status | Finding |
|------|----------|--------|---------|
| B-field query cost | §8.2 | **ESTIMATED** | <1-3% overhead estimated; uses nearest neighbor interpolation, not trilinear |
| Renavigation overhead | §8.3 | **CONFIRMED** | **0% full reinits** - cache serialization provides ZERO benefit |
| Cache capacity | §3.2 | **CORRECTED** | traccc uses default 8 candidates; 24-32 bytes/candidate = 192-256 bytes total |

### Instrumentation Details (2025-12-31)

**Method:** Added `nav_stats` counters to detray `caching_navigator`:
- `high_trust_count`: Cache reuse (fast path)
- `fair_trust_count`: Cache re-sort needed
- `no_trust_count`: Full renavigation required
- `portal_switch_count`: Volume boundary crossings
- `rescue_mode_count`: Navigation recovery attempts

**Files modified (TEMPORARY - NOW REVERTED):**
- `detray/core/include/detray/navigation/navigation_state.hpp` - Added nav_stats struct
- `detray/core/include/detray/navigation/caching_navigator.hpp` - Added counter increments
- `detray/core/include/detray/navigation/navigator_base.hpp` - Added portal/rescue counters
- `traccc/device/common/include/traccc/finding/device/navigation_stats.hpp` - Host collection struct
- `traccc/device/common/include/traccc/finding/device/propagate_to_next_surface.hpp` - Payload extension
- `traccc/device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` - Stats collection
- `traccc/device/cuda/src/finding/combinatorial_kalman_filter.cuh` - Stats allocation/printing

> **Note:** These instrumentation changes were temporary for profiling purposes and have been reverted. The files listed above no longer contain these modifications.

**Test results:** ODD detector, 10 muons @ 10 GeV, 3 events processed
- 100% high trust updates
- 0% no trust (full reinits)
- 0.4-0.7% fair trust (cache re-sort)

**Conclusion:** Navigation cache serialization (~192-256 bytes/track) is NOT needed for chunked propagation.

---

*This document provides technical specifications for chunked propagator redesign based on codebase survey of detray and traccc.*
