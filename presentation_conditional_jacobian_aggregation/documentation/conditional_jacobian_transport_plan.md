# Conditional Jacobian Aggregation Implementation Plan

**Date:** 2026-01-02
**Status:** ~~Planning~~ **IMPLEMENTED & PROFILED**
**Related:** GitHub #851, `register_pressure_survey.md` §4.9

---

> ## ⚠️ Post-Implementation Profiling Results
>
> **This plan's theoretical claims about register pressure reduction were NOT validated by profiling.**
>
> | Original Claim | Profiling Result |
> |----------------|------------------|
> | Save ~64 registers | **0 registers saved** (128 in all variants) |
> | Improve occupancy 16-25% → 25-50% | **No occupancy change** |
> | Skip Jacobian computation | **Jacobian still computed, only aggregation skipped** |
> | +5-15% throughput | **+18.3% validated** (apples-to-apples) |
>
> **Actual mechanism:** The benefit comes from skipping **Jacobian aggregation** (6x6 matrix multiplications + memory I/O), not from register pressure reduction. Both `bound_updater` and `parameter_transporter` compute the full Jacobian identically.
>
> See `doc/conditional_jacobian_transport_profile_report.md` for detailed analysis.

---

## Executive Summary

When `finding_config.run_mbf_smoother == false`, use `bound_updater` actor instead of `parameter_transporter` to skip Jacobian aggregation. ~~This saves ~64 registers (8×8 Jacobian matrix) and improves GPU occupancy from 16-25% to 25-50%.~~

**Actual benefit:** Skipping 6x6 matrix multiplications and global memory accesses at each surface provides **+18.3% throughput improvement**.

**Feasibility:** HIGHLY FEASIBLE - infrastructure already exists
**Effort:** 11-13 engineering hours
**Risk:** LOW - changes isolated to MBF-disabled path

---

## 1. Problem Statement

The `propagate_to_next_surface` kernel uses 7 actors, with `parameter_transporter` (s1) maintaining an 8×8 Jacobian matrix (~64 registers). When the MBF smoother is disabled, this Jacobian is never used, but the actor still consumes registers.

### Current Register Budget

| Component | Registers | Required When |
|-----------|-----------|---------------|
| Bound parameters (6 floats) | 6 | Always |
| Covariance (6×6 symmetric) | 21 | Always |
| **Jacobian (8×8)** | **~64** | **MBF only** |
| RK4 derivatives | 12 | Always |
| B-field vectors | 9 | Always |
| Navigation state | 8-12 | Always |
| Actor states (s0-s6) | 20-30 | Always |
| Temporaries | 10-20 | Always |
| **Total** | **150-180** | |

### Expected Improvement

| Metric | With Jacobian | Without Jacobian |
|--------|---------------|------------------|
| Registers | 150-180 | 86-116 |
| V100 Occupancy | 16-25% | 25-50% |
| Throughput gain | Baseline | +5-15% |

---

## 2. Current Architecture

### 2.1 Actor Chain Definition

**File:** `core/include/traccc/finding/details/combinatorial_kalman_filter_types.hpp:40-46`

```cpp
using ckf_actor_chain_t =
    detray::actor_chain<detray::pathlimit_aborter<traccc::scalar>,           // s0
                        detray::parameter_transporter<traccc::default_algebra>, // s1 ← TARGET
                        interaction_register<ckf_interactor_t>,               // s2
                        ckf_interactor_t,                                     // s3
                        detray::parameter_resetter<traccc::default_algebra>,  // s4
                        detray::momentum_aborter<traccc::scalar>,             // s5
                        ckf_aborter>;                                         // s6
```

### 2.2 Existing Conditional Checks

The codebase already has 5 runtime checks of `run_mbf_smoother`:

| Location | Purpose |
|----------|---------|
| `propagate_to_next_surface.ipp:103-115` | Jacobian initialization |
| `combinatorial_kalman_filter.cuh:178-188` | Jacobian buffer allocation |
| `combinatorial_kalman_filter.cuh:274-307` | Jacobian buffer reallocation |
| `combinatorial_kalman_filter.cuh:482-486` | Temporary buffer allocation |
| `combinatorial_kalman_filter.cuh:652-657` | Track building |

### 2.3 Jacobian Initialization (Already Conditional)

**File:** `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp:103-115`

```cpp
if (cfg.run_mbf_smoother) {
    assert(payload.tmp_jacobian_ptr != nullptr);
    payload.tmp_jacobian_ptr[param_id] = matrix::identity<
        bound_matrix<typename propagator_t::detector_type::algebra_type>>();
    s1._full_jacobian_ptr = &payload.tmp_jacobian_ptr[param_id];
}
```

### 2.4 Actor State Unpacking

**File:** `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp:82-101`

```cpp
using s0_type = detray::detail::tuple_element<0, actor_tuple_type>::type::state;  // pathlimit
using s1_type = detray::detail::tuple_element<1, actor_tuple_type>::type::state;  // parameter_transporter
using s2_type = detray::detail::tuple_element<2, actor_tuple_type>::type::state;  // interaction_register
using s3_type = detray::detail::tuple_element<3, actor_tuple_type>::type::state;  // interactor
using s4_type = detray::detail::tuple_element<4, actor_tuple_type>::type::state;  // parameter_resetter
using s5_type = detray::detail::tuple_element<5, actor_tuple_type>::type::state;  // momentum_aborter
using s6_type = detray::detail::tuple_element<6, actor_tuple_type>::type::state;  // ckf_aborter

s0_type s0{};
s1_type s1{};
s2_type s2{s3};
s3_type s3{};
s4_type s4{};
s5_type s5{cfg.min_step_length_for_surface_aborter, cfg.step_constraint,
           cfg.min_pt, cfg.min_p};
s6_type s6{&n_candidates, cfg.max_num_branches_per_seed, cfg.max_path_length};
```

### 2.5 Kernel Specialization Pattern

**File:** `device/cuda/CMakeLists.txt` (propagate_to_next_surface generation)

```cmake
foreach(DETECTOR_NAME ${TRACCC_SUPPORTED_DETECTORS})
    foreach(BFIELD_NAME ${TRACCC_CUDA_SUPPORTED_BFIELDS})
        set(GENERATED_SOURCE
            "${CMAKE_CURRENT_BINARY_DIR}/src/finding/kernels/specializations/
             propagate_to_next_surface_${DETECTOR_NAME}_${BFIELD_NAME}.cu")
        # Generate kernel via Python template substitution
    endforeach()
endforeach()
```

**Current specializations:** 9 (3 detectors × 3 bfields)

---

## 3. Implementation Approach

### 3.1 Strategy: Compile-Time Specialization

Create two kernel variants with different actor chains. Runtime dispatch based on `config.run_mbf_smoother`.

```
Finding Config (Runtime)
    |
    v
config.run_mbf_smoother check (Host)
    |
    +---> Launch MBF-enabled kernel (7 actors, 150-180 regs)
    |
    +---> Launch MBF-disabled kernel (6 actors, 86-116 regs)
```

### 3.2 Why Not Runtime if-constexpr

`if constexpr` requires compile-time constant. `run_mbf_smoother` is a runtime bool. Regular `if` inside propagate() causes branch divergence.

---

## 4. Required Changes

### 4.1 New Actor Chain Definition

**File:** `core/include/traccc/finding/details/combinatorial_kalman_filter_types.hpp`

```cpp
/// Actor chain for CKF with MBF smoother (7 actors, includes Jacobian transport)
using ckf_actor_chain_t =
    detray::actor_chain<detray::pathlimit_aborter<traccc::scalar>,
                        detray::parameter_transporter<traccc::default_algebra>,
                        interaction_register<ckf_interactor_t>,
                        ckf_interactor_t,
                        detray::parameter_resetter<traccc::default_algebra>,
                        detray::momentum_aborter<traccc::scalar>,
                        ckf_aborter>;

/// Actor chain for CKF without MBF smoother (6 actors, no Jacobian transport)
using ckf_actor_chain_no_mbf_t =
    detray::actor_chain<detray::pathlimit_aborter<traccc::scalar>,
                        // parameter_transporter OMITTED - saves ~64 registers
                        interaction_register<ckf_interactor_t>,
                        ckf_interactor_t,
                        detray::parameter_resetter<traccc::default_algebra>,
                        detray::momentum_aborter<traccc::scalar>,
                        ckf_aborter>;

/// Propagator type for CKF with MBF smoother
template <typename detector_t, typename bfield_t>
using ckf_propagator_t =
    detray::propagator<ckf_stepper_t<bfield_t>,
                       detray::caching_navigator<std::add_const_t<detector_t>>,
                       ckf_actor_chain_t>;

/// Propagator type for CKF without MBF smoother
template <typename detector_t, typename bfield_t>
using ckf_propagator_no_mbf_t =
    detray::propagator<ckf_stepper_t<bfield_t>,
                       detray::caching_navigator<std::add_const_t<detector_t>>,
                       ckf_actor_chain_no_mbf_t>;
```

### 4.2 Device Code Actor State Handling

**File:** `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`

Option A: Template specialization based on actor chain size
Option B: Preprocessor macro for MBF-enabled builds
Option C: Constexpr if with chain size detection

**Recommended: Option C**

```cpp
// Detect actor chain size at compile time
constexpr std::size_t n_actors =
    std::tuple_size<typename propagator_t::actor_chain_type::tuple_type>::value;

if constexpr (n_actors == 7) {
    // MBF-enabled path: 7 actors including parameter_transporter
    using s0_type = detray::detail::tuple_element<0, actor_tuple_type>::type::state;
    using s1_type = detray::detail::tuple_element<1, actor_tuple_type>::type::state;
    using s2_type = detray::detail::tuple_element<2, actor_tuple_type>::type::state;
    // ... s3-s6

    s1_type s1{};
    s1._full_jacobian_ptr = &payload.tmp_jacobian_ptr[param_id];
    // ...
} else {
    // MBF-disabled path: 6 actors, no parameter_transporter
    using s0_type = detray::detail::tuple_element<0, actor_tuple_type>::type::state;
    using s1_type = detray::detail::tuple_element<1, actor_tuple_type>::type::state;  // interaction_register
    using s2_type = detray::detail::tuple_element<2, actor_tuple_type>::type::state;  // interactor
    // ... s3-s5 (shifted indices)
    // No Jacobian initialization
}
```

### 4.3 CMakeLists.txt Extension

**File:** `device/cuda/CMakeLists.txt`

```cmake
set(TRACCC_MBF_VARIANTS "mbf" "no_mbf")

foreach(DETECTOR_NAME ${TRACCC_SUPPORTED_DETECTORS})
    foreach(BFIELD_NAME ${TRACCC_CUDA_SUPPORTED_BFIELDS})
        foreach(MBF_VARIANT ${TRACCC_MBF_VARIANTS})
            set(GENERATED_SOURCE
                "${CMAKE_CURRENT_BINARY_DIR}/src/finding/kernels/specializations/
                 propagate_to_next_surface_${DETECTOR_NAME}_${BFIELD_NAME}_${MBF_VARIANT}.cu")
            # Generate with ${MBF_VARIANT} substitution
        endforeach()
    endforeach()
endforeach()
```

**New specializations:** 18 (3 detectors × 3 bfields × 2 MBF variants)

### 4.4 Template File Update

**File:** `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface.cu.template`

```cpp
#if defined(TRACCC_CKF_MBF_ENABLED)
using propagator_t = traccc::details::ckf_propagator_t<${DETECTOR_NAME}::device, bfield_t>;
#else
using propagator_t = traccc::details::ckf_propagator_no_mbf_t<${DETECTOR_NAME}::device, bfield_t>;
#endif

template void propagate_to_next_surface<propagator_t, bfield_t>(...);
```

### 4.5 Runtime Dispatch

**File:** `device/cuda/src/finding/combinatorial_kalman_filter.cuh`

```cpp
// At kernel launch site (~line 509)
if (config.run_mbf_smoother) {
    propagate_to_next_surface<ckf_propagator_t<detector_t, bfield_t>, bfield_t>(
        grid_size, block_size, shared_mem_size, stream, config, payload);
} else {
    propagate_to_next_surface<ckf_propagator_no_mbf_t<detector_t, bfield_t>, bfield_t>(
        grid_size, block_size, shared_mem_size, stream, config, payload_no_mbf);
}
```

### 4.6 Payload Structure Variant

The payload structure may need a variant without `tmp_jacobian_ptr`:

```cpp
template <typename propagator_t, typename bfield_t>
struct propagate_to_next_surface_payload {
    // ... common fields ...

    // Only present for MBF-enabled propagator
    std::conditional_t<
        has_parameter_transporter_v<propagator_t>,
        bound_matrix<algebra_t>*,
        std::nullptr_t
    > tmp_jacobian_ptr;
};
```

---

## 5. Implementation Phases

### Phase 1: Preparation (2 hours)

- [ ] Define `ckf_actor_chain_no_mbf_t` in `combinatorial_kalman_filter_types.hpp`
- [ ] Define `ckf_propagator_no_mbf_t` propagator type alias
- [ ] Add compile-time actor chain size detection helper

### Phase 2: Device Code (3 hours)

- [ ] Update `propagate_to_next_surface.ipp` with `if constexpr` branching
- [ ] Handle actor state tuple unpacking for both 6 and 7 actor chains
- [ ] Update `propagate_to_next_surface.hpp` payload definition

### Phase 3: Build System (2 hours)

- [ ] Extend CMakeLists.txt with MBF variant loop
- [ ] Update `gen_kernel_specialization.py` to support MBF flag
- [ ] Create template variants or add preprocessor conditionals
- [ ] Verify all 18 specializations compile

### Phase 4: Integration (2 hours)

- [ ] Update `combinatorial_kalman_filter.cuh` kernel dispatch
- [ ] Update host `combinatorial_kalman_filter.hpp` if needed
- [ ] Ensure payload structures match kernel variants

### Phase 5: Testing & Validation (2 hours)

- [ ] Unit tests: MBF-disabled produces same track count
- [ ] Physics validation: Track parameters identical
- [ ] Profile: Verify register reduction with `nvcc -Xptxas -v`
- [ ] Benchmark: Measure throughput improvement

---

## 6. Code Reference Map

| Purpose | File | Lines |
|---------|------|-------|
| Actor chain definition | `core/include/traccc/finding/details/combinatorial_kalman_filter_types.hpp` | 40-46 |
| Jacobian init | `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | 103-115 |
| Actor state unpacking | `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` | 82-101 |
| Payload definition | `device/common/include/traccc/finding/device/propagate_to_next_surface.hpp` | ~30-80 |
| Buffer allocation | `device/cuda/src/finding/combinatorial_kalman_filter.cuh` | 178-188 |
| Kernel dispatch | `device/cuda/src/finding/combinatorial_kalman_filter.cuh` | ~509-512 |
| Kernel template | `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface.cu.template` | 1-30 |
| CMake generation | `device/cuda/CMakeLists.txt` | ~560-580 |
| Python generator | `codegen/kernel_specialization/gen_kernel_specialization.py` | 1-74 |

---

## 7. Risk Assessment

### Low Risk

- Changes isolated to MBF-disabled code path
- Physics correct (no Jacobian needed when MBF off)
- Pattern proven (detector/bfield specializations exist)
- Existing conditional checks validate approach

### Medium Risk

- Actor state tuple indexing requires careful handling
- Template metaprogramming complexity
- Compilation time increases (~2× for propagate_to_next_surface)

### Mitigation

- Add `static_assert` on actor chain size
- Comprehensive unit tests for MBF-disabled configuration
- CI validation of both variants

---

## 8. Alternatives Considered

### Alternative A: Runtime if-constexpr

**Rejected:** `run_mbf_smoother` is runtime value, cannot use `if constexpr`

### Alternative B: Single Kernel with Null Jacobian

**Rejected:** Actor s1 still instantiated, still consumes registers

### Alternative C: Upstream Detray Conditional Actor

**Rejected:** Would require detray API changes, out of scope

---

## 9. Success Criteria

### Original Criteria (Theoretical)

1. ~~**Register reduction:** MBF-disabled kernel uses <120 registers (currently 150-180)~~
2. ~~**Occupancy improvement:** V100 occupancy >40% (currently 16-25%)~~
3. **Physics correctness:** Track reconstruction identical to baseline
4. **No regression:** MBF-enabled path unchanged
5. **Build success:** All 18 kernel specializations compile

### Actual Results (Post-Profiling)

| Criterion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Register reduction | <120 registers | 128 registers | ❌ **NOT MET** |
| Occupancy improvement | >40% | No change | ❌ **NOT MET** |
| Physics correctness | Identical | Identical | ✅ **MET** |
| No regression | MBF path unchanged | Unchanged | ✅ **MET** |
| Build success | 18 specializations | 18 compile | ✅ **MET** |
| Throughput improvement | +5-15% | **+18.3%** | ✅ **EXCEEDED** |

**Conclusion:** The optimization achieved better throughput than expected (+18.3% vs +5-15%), but through a different mechanism than planned (skipped aggregation vs register reduction).

---

## 10. References

- `doc/conditional_jacobian_transport_profile_report.md` - **Profiling results showing actual mechanism**
- `doc/conditional_jacobian_transport_report.md` - Benchmark results
- `doc/register_pressure_survey.md` §4.9 - Algorithmic refactoring analysis
- `doc/register_pressure_survey.md` §6.3 - Recommendation for conditional Jacobian
- GitHub Issue #851 - Original register pressure report
- `core/include/traccc/finding/finding_config.hpp:47` - `run_mbf_smoother` definition
