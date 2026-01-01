# Symbolic Code Generation Plan

**Date:** 2026-01-01
**Related Issue:** GitHub #851 - Register Usage in Performance-Critical Kernels
**Status:** Planning Complete

---

## Executive Summary

This document details the plan for implementing symbolic code generation to reduce register pressure in traccc's GPU kernels. The goal is to achieve **15-30% register reduction** through Common Subexpression Elimination (CSE), operation fusion, and register-aware instruction scheduling.

**Key Discovery:** Detray-fork already contains SymPy-based codegen infrastructure that can be extended to traccc's Kalman filter operations.

**Target Outcome:** Improve kernel occupancy from 16-25% to 30-40% by reducing register usage from 128-203 to 80-120 registers per thread.

**Important Caveat:** Modern compilers (nvcc, clang) already perform CSE optimization. The primary benefit of symbolic codegen is not CSE alone, but:
1. **Cross-function fusion** - compilers cannot CSE across function boundaries
2. **Live variable lifetime reduction** - explicit scheduling to minimize simultaneously live values
3. **Domain-specific simplification** - exploiting matrix symmetry, sparsity patterns
4. **Measurement of actual benefit** is required before committing to full implementation

---

## Table of Contents

1. [Existing Codegen Infrastructure](#1-existing-codegen-infrastructure)
2. [Priority Targets](#2-priority-targets)
3. [Detailed Target Analysis](#3-detailed-target-analysis)
4. [CSE Opportunities](#4-cse-opportunities)
5. [Data Flow Analysis](#5-data-flow-analysis)
6. [Implementation Approach](#6-implementation-approach)
7. [Files to Generate](#7-files-to-generate)
8. [Expected Benefits](#8-expected-benefits)
9. [Risk Analysis](#9-risk-analysis)
10. [Validation and Testing](#10-validation-and-testing)
11. [Implementation Phases](#11-implementation-phases)
12. [Maintenance](#12-maintenance)
13. [References](#13-references)

---

## 1. Existing Codegen Infrastructure

### 1.1 Detray SymPy Codegen System

**Location:** `detray-fork/codegen/detray-sympy/` (relative to traccc root, verified to exist)

| Script | Lines | Output | Purpose |
|--------|-------|--------|---------|
| `gen_update_rk_transport_jacobian_impl.py` | 122 | Transport Jacobian update | RK4 Jacobian derivatives |
| `gen_full_jacobian.py` | 125 | Full Jacobian assembly | J_full = J_F2B * (D+I) * J_transport * J_B2F |
| `gen_transport_covariance_to_bound_impl.py` | 84 | Covariance transport | C_new = J_full * C * J_full^T |
| `detray_sympy/common.py` | ~100 | Shared utilities | Matrix symbols, CSE helpers, C++ printer |

**Generated C++ Headers:**
- `detray-fork/core/include/detray/propagator/codegen/update_rk_transport_jacobian.hpp`
- `detray-fork/core/include/detray/propagator/actors/codegen/full_jacobian.hpp`
- `detray-fork/core/include/detray/propagator/actors/codegen/covariance_transport.hpp`

**Verified:** These files exist in the detray-fork repository as of 2026-01-01.

### 1.2 Codegen Features Used

```python
# SymPy features leveraged:
from sympy import MatrixSymbol, symbols, cse
from sympy.printing.cxx import CXX17CodePrinter

# Common Subexpression Elimination
replacements, reduced = cse(expressions)

# C++ code generation with custom printer
printer = CXX17CodePrinter()
code = printer.doprint(expression)
```

### 1.3 Traccc Kernel Specialization (Type-Only)

**Location:** `codegen/kernel_specialization/gen_kernel_specialization.py` (73 lines)

Current system generates 44 kernel specializations via Python string templates:
- `find_tracks`: 4 detector variants
- `apply_interaction`: 4 detector variants
- `propagate_to_next_surface`: 4 detector × 3 bfield = 12 variants
- `fit_forward` / `fit_backward`: 4 detector × 3 bfield × 2 = 24 variants

**Limitation:** Type specialization only, not algorithm-level symbolic generation.

---

## 2. Priority Targets

### 2.1 Priority Matrix

| Priority | Target | Location | Impact | Call Frequency | Rationale |
|----------|--------|----------|--------|----------------|-----------|
| **P1** | Fused Kalman gain + covariance | `gain_matrix_updater.hpp:116-174` | **HIGH** | Every measurement | Cross-function fusion; compiler can't optimize |
| **P2** | Smoother 6×6 inversions | `two_filters_smoother.hpp:82-93` | HIGH | Per track | Multiple inversions; symmetry exploitation |
| **P3** | MBF Jacobian chain | `build_tracks.ipp:73-127` | MEDIUM | K× per track | Per-iteration CSE; limited by sequential nature |
| **P4** | 4×4 inverse with CSE | algebra-plugins | LOW | Occasional | Compiler likely already optimizes; validate first |
| **P5** | 2×2 inverse + determinant | algebra-plugins `hard_coded.hpp` | **LOW** | Millions/event | Only 7 ops total; minimal savings possible |

**Note on P5 (2×2 inverse):** Despite high call frequency, the 2×2 inverse is only 7 operations. Potential savings of 1-2 operations will not meaningfully impact register pressure. The compiler almost certainly already optimizes this. **Deprioritized** unless profiling shows otherwise.

**Note on P1 (Fused Kalman):** This is the highest priority because it involves multiple matrix operations across several function calls that the compiler cannot fuse. Register pressure comes from intermediate matrices kept live across these calls.

### 2.2 Measurement Dimension Flexibility

The Kalman filter supports variable measurement dimensions:

| Detector Type | Measurement Dim (D) | Matrices Affected |
|---------------|---------------------|-------------------|
| Pixel (default) | 2 | H (D×6), V (D×D), K (6×D) |
| Strip/1D | 1 | All measurement matrices |
| Future detectors | 3+ | Potential extension |

**Implication:** Generated code must be templated on measurement dimension `D`, or separate specializations generated for D=1, D=2.

```cpp
// Generated code should support:
template <std::size_t D>
TRACCC_HOST_DEVICE void kalman_gain_update(...);

// Or explicit specializations:
void kalman_gain_update_1d(...);  // D=1
void kalman_gain_update_2d(...);  // D=2 (most common)
```

### 2.3 Impact Assessment (Conservative Estimates)

| Component | Current Ops | Best Case | Likely Case | Notes |
|-----------|-------------|-----------|-------------|-------|
| Fused Kalman gain | ~45 | ~30 | ~38 | Cross-function fusion is main win |
| 6×6 inverse (LU) | ~216 | ~150 | ~180 | Symmetry helps; LU already optimized |
| 4×4 inverse | 136 | 80-90 | ~110 | Compiler may already CSE |
| Jacobian chain (per step) | 216 | 150 | ~190 | Sequential dependency limits gains |
| 2×2 inverse | 7 | 5 | 6-7 | Minimal; not worth optimizing |

**Key Insight:** Operation count reduction does not directly translate to register reduction. The primary goal is reducing **live variable lifetime**, not operation count.

---

## 3. Detailed Target Analysis

### 3.1 Kalman Gain Matrix Computation (P2)

**File:** `core/include/traccc/fitting/kalman_filter/gain_matrix_updater.hpp`
**Lines:** 116-135, 169-174

**Current Operation Sequence:**
```
1. projected_cov = C^T * H^T            (6×6 * 6×2 = 6×2)  [Line 117]
2. M = H * projected_cov + V            (2×6 * 6×2 + 2×2 = 2×2)  [Line 119]
3. K = projected_cov * M^(-1)           (6×2 * 2×2 = 6×2)  [Line 123]
4. (I-KH) = I66 - K*H                   (6×6 - 6×2*2×6 = 6×6)  [Line 131]
5. filtered_cov = (I-KH)*C*(I-KH)^T + K*V*K^T  [Lines 131-134]
6. chi2 = r^T * R^(-1) * r              (1×2 * 2×2 * 2×1 = scalar)  [Line 172-173]
```

**Data Dependencies:**
- Input: `predicted_cov` (6×6), `H` (2×6), `V` (2×2), measurements
- Output: `K` (6×2), `filtered_cov` (6×6), `chi2` (scalar)
- Intermediate: `projected_cov` (6×2), `M` (2×2), `M^(-1)` (2×2)

**Operation Count:**
- Matrix multiplications: 8
- Matrix inversions: 2 (M and R, both 2×2)
- Transposes: 2
- Additions/Subtractions: 5
- **Total: ~45 operations + inverse costs**

### 3.2 Two-Filters Smoother (P4)

**File:** `core/include/traccc/fitting/kalman_filter/two_filters_smoother.hpp`
**Lines:** 81-93, 144-232

**Smoother Covariance Composition:**
```
1. predicted_cov_inv = C^(-1)           (6×6)  [Line 82]
2. filtered_cov_inv = F^(-1)            (6×6)  [Line 84]
3. smoothed_cov_inv = C_inv + F_inv     (6×6)  [Line 89]
4. smoothed_cov = (smoothed_cov_inv)^(-1) (6×6)  [Line 93]
5. smoothed_vec = S_cov * (F_inv*F_vec + C_inv*C_vec)  [Lines 105-108]
```

**Chi-squared Calculation:**
```
6. R_smt = V - H*S_cov*H^T             (2×2)  [Line 162-164]
7. chi2_smt = y^T * R_smt^(-1) * y     (scalar)  [Line 170-172]
```

**Operation Count:** ~90 operations + 4 inverse costs (3× 6×6, 1× 2×2)

### 3.3 MBF Jacobian Chain (P5)

**File:** `device/common/include/traccc/finding/device/impl/build_tracks.ipp`
**Lines:** 73-127, 164-168

**Jacobian Composition Operations:**
```
1. accumulated_jacobian = I (6×6)                    [Line 74]
2. accumulated_jacobian = accumulated_jacobian * J   (6×6 * 6×6) [Lines 84-85]
3. small_lambda_hat = J^T * small_lambda_tilde       (6×6 * 6×1) [Line 123]
4. big_lambda_hat = J^T * big_lambda_tilde * J       (6×6 * 6×6 * 6×6) [Lines 125-126]
```

**Critical Pattern:**
```cpp
// Line 85: Multiplicative Jacobian composition
accumulated_jacobian = accumulated_jacobian * payload.jacobian_ptr[link_idx];
```

**Operation Count Per Track:** 216×K + 45 (where K = track length)

### 3.4 Hard-Coded Matrix Operations in algebra-plugins

**File:** `build/_deps/algebraplugins-src/math/generic/include/algebra/math/algorithms/matrix/inverse/hard_coded.hpp`

#### Matrix Storage Layout (CRITICAL)

algebra-plugins uses **column-major storage** (array of column vectors):

```cpp
// From storage/matrix.hpp:25
// "The matrix consists of column vectors"
template <...>
struct matrix {
    using vector_type = storage::vector<ROW, scalar_t, array_t>;
    std::array<vector_type, COL> m_storage;  // Array of columns
};

// Access pattern:
m[col][row]  // NOT m[row][col]
```

**Implication for codegen:** Generated code must use `m[col][row]` indexing, not the mathematical `m[row, col]` notation shown in documentation.

#### 2×2 Inverse (Lines 34-49)
```cpp
// Current: 7 operations (already minimal)
det = m[0][0]*m[1][1] - m[1][0]*m[0][1]  // Note: column-major indexing
inv[0][0] = m[1][1] / det
inv[1][0] = -m[1][0] / det  // Off-diagonal
inv[0][1] = -m[0][1] / det  // Off-diagonal
inv[1][1] = m[0][0] / det
```

**Verdict:** Already optimal. Compiler will fuse det computation with divisions. No codegen benefit.

#### 4×4 Inverse (Lines 51-273)
```cpp
// 16 cofactor computations, each with 6 triple-product terms
element_getter()(ret, 0, 0) =
    element_getter()(m, 1, 2) * element_getter()(m, 2, 3) * element_getter()(m, 3, 1) -
    element_getter()(m, 1, 3) * element_getter()(m, 2, 2) * element_getter()(m, 3, 1) +
    // ... 4 more terms
```

**Register Impact:** ~120 multiply-add operations. Potential CSE exists, but nvcc likely already performs this optimization. **Must validate with PTX inspection before implementing.**

### 3.5 Actor State Tuple (7 Actors)

**File:** `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`
**Lines:** 82-101

```cpp
using s0_type = tuple_element<0>::type::state;  // pathlimit_aborter
using s1_type = tuple_element<1>::type::state;  // parameter_transporter (JACOBIAN)
using s2_type = tuple_element<2>::type::state;  // interaction_register
using s3_type = tuple_element<3>::type::state;  // interactor
using s4_type = tuple_element<4>::type::state;  // parameter_resetter
using s5_type = tuple_element<5>::type::state;  // momentum_aborter
using s6_type = tuple_element<6>::type::state;  // ckf_aborter
```

**Impact:** All 7 actor states kept live simultaneously, contributing ~20-30 registers.

---

## 4. CSE Opportunities

### 4.1 4×4 Inverse CSE Targets

Identified 20+ shared 3-element products across 16 output elements:

| Product | Appears In |
|---------|------------|
| `m[1,2]*m[2,3]*m[3,1]` | ret[0,0], ret[2,1] |
| `m[1,3]*m[2,2]*m[3,0]` | ret[0,1], ret[3,0] |
| `m[0,2]*m[1,3]*m[3,1]` | ret[0,2], ret[1,2] |
| `m[0,1]*m[1,2]*m[2,0]` | ret[3,3], multiple |
| ... | (20+ total) |

**Potential Reduction:** From 96 operations down to ~50-60 with CSE.

### 4.2 Kalman Gain CSE Targets

| Expression | Computed At | Could Share With |
|------------|-------------|------------------|
| `H * projected_cov` | Line 119 | Line 157 |
| `matrix::transpose(K)` | Line 132 | Line 134 |
| `matrix::transpose(H)` | Line 117 | Multiple |
| `M^(-1)` determinant | Line 122 (inside inverse) | Line 180 |

### 4.3 Smoother CSE Targets

| Expression | Frequency | Savings |
|------------|-----------|---------|
| `H * predicted_covariance * H^T` | 2× | 12 ops |
| `matrix::transpose(C_hat)` | 2× | 36 ops |
| Identity matrix construction | 3× | 6 ops |

---

## 5. Data Flow Analysis

### 5.1 Kalman Gain Computation Data Flow

```
Current Implementation:
┌─────────────────────────────────────────────────────────────────┐
│ Input: predicted_cov (6×6), H (2×6), V (2×2), measurement (2×1) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ projected_cov = C^T × H^T                  → 6×2 intermediate   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ M = H × projected_cov + V                  → 2×2 intermediate   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ det(M) = cofactor expansion                → scalar (in inverse)│
│ M^(-1) = cofactor × (1/det)                → 2×2 intermediate   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ K = projected_cov × M^(-1)                 → 6×2 output         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ (I-KH) = I - K × H                         → 6×6 intermediate   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ filtered_cov = (I-KH) × C × (I-KH)^T + K × V × K^T → 6×6 output │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Proposed Fused Implementation

```
Symbolic Fusion:
┌─────────────────────────────────────────────────────────────────┐
│ K, filtered_cov, chi2 = fused_kalman_update(C, H, V, meas)      │
│                                                                 │
│ - Single function, shared temporaries                           │
│ - CSE across all operations                                     │
│ - Register-aware instruction ordering                           │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 MBF Jacobian Chain Data Flow

```
Per-Track Jacobian Accumulation:
┌──────────────────────────────────────────────────────────────────┐
│ accumulated_J = I (6×6)                    [Initial]             │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ For each measurement k:       │
              │   accumulated_J *= J_step[k]  │
              │   (6×6 × 6×6 = 6×6)           │
              └───────────────┬───────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ small_lambda_hat = J^T × small_lambda_tilde    (6×6 × 6×1)       │
│ big_lambda_hat = J^T × big_lambda_tilde × J    (6×6 × 6×6 × 6×6) │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Approach

### 6.1 Extend Detray's SymPy Infrastructure

Create new codegen scripts modeled after existing detray-fork infrastructure:

```python
# traccc/codegen/sympy/gen_kalman_gain.py
from sympy import MatrixSymbol, symbols, cse, eye
from sympy.printing.cxx import CXX17CodePrinter

def generate_kalman_gain():
    # Symbolic matrices
    C = MatrixSymbol('C', 6, 6)  # predicted covariance
    H = MatrixSymbol('H', 2, 6)  # measurement projection
    V = MatrixSymbol('V', 2, 2)  # measurement covariance

    # Kalman gain computation
    projected_cov = C.T * H.T
    M = H * projected_cov + V
    M_inv = M.inv()  # SymPy handles 2×2 inverse symbolically
    K = projected_cov * M_inv

    # Covariance update (Joseph form)
    I_KH = eye(6) - K * H
    filtered_cov = I_KH * C * I_KH.T + K * V * K.T

    # Apply CSE
    replacements, reduced = cse([K, filtered_cov])

    # Generate C++ code
    printer = CXX17CodePrinter()
    return generate_cpp_function(replacements, reduced, printer)
```

### 6.2 Build System Integration

Add CMake targets for codegen:

```cmake
# traccc/codegen/CMakeLists.txt
find_package(Python REQUIRED COMPONENTS Interpreter)

set(CODEGEN_SCRIPTS
    sympy/gen_kalman_gain.py
    sympy/gen_covariance_update.py
    sympy/gen_chi2_computation.py
    sympy/gen_jacobian_chain.py
)

foreach(SCRIPT ${CODEGEN_SCRIPTS})
    get_filename_component(NAME ${SCRIPT} NAME_WE)
    set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/include/traccc/codegen/${NAME}.hpp")
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND Python::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} -o ${OUTPUT_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT}
        COMMENT "Generating ${NAME}.hpp from SymPy"
    )
    list(APPEND GENERATED_HEADERS ${OUTPUT_FILE})
endforeach()

add_custom_target(traccc_codegen DEPENDS ${GENERATED_HEADERS})
```

### 6.3 Generated Code Pattern

```cpp
// Generated: core/include/traccc/fitting/kalman_filter/codegen/gain_matrix_update.hpp

#pragma once

namespace traccc::codegen {

/// Fused Kalman gain and covariance update
/// @tparam D Measurement dimension (1 or 2)
/// @tparam algebra_t Algebra type (for matrix/vector types)
template <std::size_t D, typename algebra_t>
TRACCC_HOST_DEVICE inline void kalman_gain_update(
    const typename algebra_t::template matrix_type<6, 6>& predicted_cov,
    const typename algebra_t::template matrix_type<D, 6>& H,
    const typename algebra_t::template matrix_type<D, D>& V,
    typename algebra_t::template matrix_type<6, D>& K,
    typename algebra_t::template matrix_type<6, 6>& filtered_cov) {

    using scalar_t = typename algebra_t::scalar_type;

    // IMPORTANT: algebra-plugins uses column-major storage
    // Access: matrix[col][row], NOT matrix[row][col]

    // CSE temporaries (generated by SymPy)
    // projected_cov = C^T * H^T, but computed column-wise
    const scalar_t t0 = predicted_cov[0][0] * H[0][0] + predicted_cov[0][1] * H[0][1];
    const scalar_t t1 = predicted_cov[1][0] * H[0][0] + predicted_cov[1][1] * H[0][1];
    // ... more CSE temporaries (SymPy generates optimal order) ...

    // M matrix (D×D) - measurement covariance in measurement space
    // M = H * C * H^T + V
    const scalar_t M00 = /* generated */ ;
    const scalar_t M01 = /* generated */ ;
    const scalar_t M11 = /* generated */ ;

    // D×D inverse (specialized for D=1, D=2)
    // For D=2: fused determinant computation
    const scalar_t det_inv = scalar_t(1) / (M00 * M11 - M01 * M01);
    const scalar_t Mi00 = M11 * det_inv;
    const scalar_t Mi01 = -M01 * det_inv;
    const scalar_t Mi11 = M00 * det_inv;

    // Gain matrix K (6×D) - column-major output
    K[0][0] = /* generated: first column */ ;
    K[0][1] = /* generated */ ;
    // ...

    // Covariance update (Joseph form)
    // filtered_cov = (I - K*H) * C * (I - K*H)^T + K * V * K^T
    // Key optimization: temporaries freed as soon as possible
    // ... generated code with explicit lifetime management ...
}

}  // namespace traccc::codegen
```

**Key differences from naive implementation:**
1. Uses algebra-plugins types directly (not raw pointers)
2. Column-major indexing (`m[col][row]`)
3. Templated on measurement dimension `D`
4. CSE temporaries ordered to minimize live variable count
5. Explicit comments for maintainability

---

## 7. Files to Generate

### 7.1 New Codegen Scripts

| File | Purpose | Priority |
|------|---------|----------|
| `codegen/sympy/gen_kalman_gain.py` | Kalman gain K computation | P2 |
| `codegen/sympy/gen_covariance_update.py` | Joseph covariance update | P3 |
| `codegen/sympy/gen_chi2_computation.py` | Chi-squared calculation | P3 |
| `codegen/sympy/gen_jacobian_chain.py` | MBF Jacobian accumulation | P5 |
| `codegen/sympy/gen_matrix_inverse_2x2.py` | Fused 2×2 inverse+det | P1 |
| `codegen/sympy/gen_matrix_inverse_4x4.py` | CSE-optimized 4×4 inverse | P6 |

### 7.2 Generated Headers

| Output Header | Replaces |
|---------------|----------|
| `core/include/traccc/fitting/kalman_filter/codegen/gain_matrix_update.hpp` | `gain_matrix_updater.hpp:117-135` |
| `core/include/traccc/fitting/kalman_filter/codegen/covariance_update.hpp` | `gain_matrix_updater.hpp:131-134` |
| `core/include/traccc/fitting/kalman_filter/codegen/chi2_computation.hpp` | `gain_matrix_updater.hpp:172-174` |
| `core/include/traccc/fitting/kalman_filter/codegen/smoother_update.hpp` | `two_filters_smoother.hpp:81-93` |
| `device/common/include/traccc/finding/device/codegen/jacobian_chain.hpp` | `build_tracks.ipp:73-127` |

### 7.3 Integration Points

| Original File | Line Range | Codegen Replacement |
|---------------|------------|---------------------|
| `gain_matrix_updater.hpp` | 116-135 | `codegen::kalman_gain_update()` |
| `gain_matrix_updater.hpp` | 131-134 | `codegen::covariance_update()` |
| `gain_matrix_updater.hpp` | 169-174 | `codegen::chi2_compute()` |
| `two_filters_smoother.hpp` | 81-93 | `codegen::smoother_covariance()` |
| `build_tracks.ipp` | 73-127 | `codegen::jacobian_accumulate()` |

---

## 8. Expected Benefits

### 8.1 Register Reduction Estimates (Conservative)

| Component | Current Regs (est.) | Optimistic | Realistic | Notes |
|-----------|---------------------|------------|-----------|-------|
| Fused Kalman gain + cov | 40+ | 20-25 | 30-35 | Main benefit from cross-function fusion |
| 6×6 covariance transport | 36+ | 20-25 | 28-32 | Symmetry exploitation helps |
| 8×8 Jacobian accumulation | 64 | 35-40 | 50-55 | Sequential loop limits gains |
| Actor state tuple | 20-30 | 15-18 | 18-22 | Some states must remain live |
| 2×2 inverse | 8 | 6 | 7-8 | Already optimal; skip |

**Why "Realistic" differs from "Optimistic":**
- Compiler already performs many optimizations
- Some live variables cannot be eliminated (needed for correctness)
- Loop-carried dependencies force variable lifetimes
- Register allocation is NP-hard; SymPy CSE is not optimal

### 8.2 Occupancy Improvement Projection (Conservative)

| Kernel | Current Regs | Current Occupancy | Realistic Target | Target Occupancy |
|--------|--------------|-------------------|------------------|------------------|
| `propagate_to_next_surface` | 128-203 | 16-25% | 100-140 | 25-35% |
| `fit_forward` | 128-168 | 19-25% | 100-130 | 25-35% |
| `fit_backward` | 128-168 | 19-25% | 100-130 | 25-35% |

**Note:** Achieving 50%+ occupancy would require reducing registers to <64, which is unlikely with current algorithm structure. A more realistic target is **25-35% occupancy** (up from 16-25%).

### 8.3 Operation Count vs Register Pressure

**Important distinction:** Reducing operation count does NOT directly reduce register pressure.

| Factor | Impact on Registers | CSE Helps? |
|--------|---------------------|------------|
| Live variable lifetime | HIGH | YES - if scheduled correctly |
| Intermediate matrix storage | HIGH | PARTIAL - fusion helps |
| Loop-carried dependencies | MEDIUM | NO - inherent to algorithm |
| Function call boundaries | HIGH | YES - fusion eliminates |
| Compiler spill decisions | MEDIUM | INDIRECT |

**The primary goal is live variable lifetime reduction, not operation count reduction.**

---

## 9. Risk Analysis

### 9.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Compiler already performs CSE | HIGH | Codegen provides no benefit | Validate with PTX before full implementation |
| Generated code slower than original | MEDIUM | Wasted effort | Benchmark at each phase; keep fallback |
| Numerical precision differs | LOW | Physics validation fails | Unit tests with tolerance; bit-exact where possible |
| Matrix storage mismatch | MEDIUM | Incorrect results | Verify column-major indexing in all generated code |
| SymPy expression explosion | LOW | Generation too slow | Limit matrix sizes; use incremental CSE |

### 9.2 Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Phase 1 shows no benefit | MEDIUM | Project abandoned | Define clear go/no-go criteria upfront |
| Integration complexity | MEDIUM | Delays | Compile-time switch between original/codegen |
| Detray upstream changes | LOW | Codegen out of sync | Monitor detray releases; regenerate as needed |

### 9.3 Go/No-Go Criteria

**Phase 1 must demonstrate:**
1. At least 10% register reduction in isolated benchmark
2. No performance regression (±5% tolerance)
3. Numerical equivalence (relative error < 1e-6)

**If Phase 1 fails these criteria, the project should be reconsidered.**

### 9.4 Fallback Strategy

If symbolic codegen does not provide expected benefits:

1. **Immediate fallback:** Keep original implementation; codegen disabled via compile flag
2. **Alternative approaches:**
   - Focus on `--maxrregcount` compiler flag tuning
   - Investigate CUDA 13 shared memory spilling (see `register_pressure_survey.md`)
   - Consider kernel fission (split large kernels)
3. **Document lessons learned** for future reference

---

## 10. Validation and Testing

### 10.1 Numerical Validation

**Unit Tests (per generated function):**

```cpp
TEST(CodegenKalmanGain, NumericalEquivalence) {
    // Generate random inputs
    auto predicted_cov = random_symmetric_positive_definite<6>();
    auto H = random_matrix<2, 6>();
    auto V = random_symmetric_positive_definite<2>();

    // Original implementation
    auto [K_orig, cov_orig] = original::kalman_gain_update(predicted_cov, H, V);

    // Codegen implementation
    auto [K_gen, cov_gen] = codegen::kalman_gain_update(predicted_cov, H, V);

    // Compare with tolerance
    EXPECT_MATRIX_NEAR(K_orig, K_gen, 1e-6);
    EXPECT_MATRIX_NEAR(cov_orig, cov_gen, 1e-6);
}
```

**Property-Based Tests:**
- Covariance remains symmetric positive definite
- Chi-squared is non-negative
- Kalman gain bounded appropriately

### 10.2 Register Usage Validation

**Compile-time check:**
```bash
# Compile with verbose PTX output
nvcc --ptxas-options=-v -c kernel.cu 2>&1 | grep "Used .* registers"

# Compare original vs codegen
diff <(nvcc ... original.cu) <(nvcc ... codegen.cu)
```

**Automated CI check:**
```yaml
- name: Check register usage
  run: |
    ORIG_REGS=$(nvcc --ptxas-options=-v original.cu 2>&1 | grep -oP 'Used \K\d+')
    GEN_REGS=$(nvcc --ptxas-options=-v codegen.cu 2>&1 | grep -oP 'Used \K\d+')
    if [ "$GEN_REGS" -gt "$ORIG_REGS" ]; then
      echo "ERROR: Codegen uses MORE registers ($GEN_REGS > $ORIG_REGS)"
      exit 1
    fi
```

### 10.3 Performance Benchmarking

**Micro-benchmarks:**
- Isolated function timing (Kalman gain only)
- Memory bandwidth measurement
- Occupancy measurement via Nsight Compute

**System benchmarks:**
- Full CKF throughput (events/second)
- ODD detector, ttbar events (standard test case)
- Compare original vs codegen builds

**Regression threshold:** Codegen must not be >5% slower than original.

### 10.4 PTX Inspection

Before claiming CSE benefit, inspect generated PTX:

```bash
# Generate PTX
nvcc -ptx -o kernel.ptx kernel.cu

# Look for:
# 1. Temporary register reuse
# 2. Spill loads/stores (should decrease)
# 3. Instruction count (should decrease slightly)
```

**If PTX shows no improvement, the compiler is already optimizing effectively.**

---

## 11. Implementation Phases

### Phase 1: Validation Experiment

**Objective:** Determine if symbolic codegen provides benefit over compiler optimization

**Tasks:**
1. Set up SymPy codegen infrastructure in `traccc/codegen/sympy/`
2. Implement `gen_kalman_gain.py` targeting fused Kalman gain + covariance update
3. Generate standalone test kernel (not integrated into traccc)
4. Compare PTX output: original vs codegen
5. Measure register usage with `nvcc -Xptxas -v`
6. Benchmark isolated kernel performance

**Go/No-Go Decision:**
- If register reduction < 10%: **STOP** - compiler is already effective
- If performance regression > 5%: **STOP** - codegen not beneficial
- If numerical error > 1e-6: **FIX** before proceeding

**Success Criteria (to proceed to Phase 2):**
- At least 10% register reduction demonstrated
- Performance within ±5% of original
- Numerical equivalence verified

### Phase 2: Kalman Filter Integration

**Objective:** Integrate codegen into traccc with fallback mechanism

**Tasks:**
1. Add compile-time switch: `TRACCC_USE_CODEGEN_KALMAN`
2. Generate `gain_matrix_update.hpp` with proper algebra-plugins types
3. Handle measurement dimension D=1 and D=2
4. Integrate into `gain_matrix_updater.hpp` with `#ifdef`
5. Run full test suite with codegen enabled
6. Benchmark on ODD detector, ttbar events

**Success Criteria:**
- All existing tests pass with codegen enabled
- Register reduction of 10-20% in full kernel
- Kernel occupancy improved by 5-10 percentage points
- No performance regression in full pipeline

### Phase 3: Smoother and Jacobian

**Objective:** Extend codegen to remaining high-value targets

**Tasks:**
1. Implement `gen_smoother_update.py` for two_filters_smoother
2. Implement `gen_jacobian_chain.py` for MBF smoother
3. Verify Jacobian chain benefits are limited (sequential dependency)
4. Integrate with compile-time switches
5. Full system benchmarking and profiling

**Success Criteria:**
- All Kalman filter operations support codegen
- Total register reduction of 15-25% (realistic target)
- Occupancy improved to 25-35%

### Phase 4: Optimization and Hardening

**Objective:** Refine codegen and prepare for production

**Tasks:**
1. Analyze generated code with Nsight Compute
2. Identify remaining register pressure hotspots
3. Consider live variable scheduling improvements
4. Add CI checks for register usage regression
5. Document codegen system for maintainers
6. Consider upstreaming patterns to detray

**Success Criteria:**
- Codegen enabled by default in release builds
- CI prevents register usage regression
- Documentation complete

---

## 12. Maintenance

### 12.1 Regeneration Workflow

When formulas change (e.g., new Kalman filter variant):

```bash
# 1. Modify SymPy script
vim codegen/sympy/gen_kalman_gain.py

# 2. Regenerate C++ header
python codegen/sympy/gen_kalman_gain.py -o core/include/traccc/.../codegen/gain_matrix_update.hpp

# 3. Verify numerical equivalence
ctest -R CodegenKalmanGain

# 4. Check register usage didn't regress
nvcc --ptxas-options=-v ... | grep registers

# 5. Commit both .py and .hpp
git add codegen/sympy/gen_kalman_gain.py core/include/.../gain_matrix_update.hpp
git commit -m "Update Kalman gain codegen"
```

### 12.2 Code Review Guidelines

**For codegen script changes:**
- Verify mathematical correctness (formulas match literature)
- Check SymPy expression structure (no accidental copies)
- Ensure CSE is applied (`cse()` call present)
- Verify column-major indexing in generated code

**For generated header changes:**
- Generated headers should ONLY be modified via regeneration
- Add `// GENERATED FILE - DO NOT EDIT` header
- Include generation command in header comment

### 12.3 Dependency Management

| Dependency | Version | Update Policy |
|------------|---------|---------------|
| SymPy | 1.12+ | Update annually; test regeneration |
| algebra-plugins | Latest | Check storage layout if API changes |
| detray-fork | Local | Sync codegen patterns periodically |

### 12.4 Ownership

| Component | Owner | Backup |
|-----------|-------|--------|
| SymPy scripts | [TBD] | [TBD] |
| Generated headers | Auto-generated | Regenerate from scripts |
| Integration code | traccc maintainers | Standard review process |

### 12.5 Troubleshooting

**Codegen produces incorrect results:**
1. Check matrix indexing (column-major: `m[col][row]`)
2. Verify measurement dimension template parameter
3. Compare against original implementation step-by-step

**Register usage increased after regeneration:**
1. Check if SymPy version changed (CSE behavior varies)
2. Inspect PTX for unexpected spills
3. Consider explicit variable lifetime hints

**Build fails after detray update:**
1. Check if algebra types changed
2. Verify `matrix_type` template parameters
3. Regenerate with updated type information

---

## 13. References

### 13.1 Codebase Locations

**Kalman Filter:**
- `core/include/traccc/fitting/kalman_filter/gain_matrix_updater.hpp` (188 lines)
- `core/include/traccc/fitting/kalman_filter/two_filters_smoother.hpp` (319 lines)
- `core/include/traccc/fitting/kalman_filter/kalman_fitter.hpp` (556 lines)

**Device Finding:**
- `device/common/include/traccc/finding/device/propagate_to_next_surface.hpp`
- `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` (162 lines)
- `device/common/include/traccc/finding/device/impl/build_tracks.ipp` (231 lines)

**algebra-plugins** (paths relative to build directory):
- `_deps/algebraplugins-src/math/generic/include/algebra/math/algorithms/matrix/inverse/hard_coded.hpp`
- `_deps/algebraplugins-src/math/generic/include/algebra/math/algorithms/matrix/determinant/hard_coded.hpp`
- `_deps/algebraplugins-src/storage/common/include/algebra/storage/matrix.hpp` - **Column-major storage**

**Detray Codegen** (paths relative to traccc root):
- `detray-fork/codegen/detray-sympy/gen_update_rk_transport_jacobian_impl.py`
- `detray-fork/codegen/detray-sympy/gen_full_jacobian.py`
- `detray-fork/codegen/detray-sympy/gen_transport_covariance_to_bound_impl.py`
- `detray-fork/codegen/detray-sympy/detray_sympy/common.py` - Shared utilities

### 13.2 Related Documentation

- `doc/register_pressure_survey.md` - Problem definition and solution survey
- `doc/problem_definition.md` - Warp divergence analysis (separate issue)
- GitHub Issue #851 - Original problem report

### 13.3 External Resources

- [SymPy Matrix Documentation](https://docs.sympy.org/latest/modules/matrices/matrices.html)
- [SymPy CSE Documentation](https://docs.sympy.org/latest/modules/rewriting.html#common-subexpression-detection-and-collection)
- [NVIDIA CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)
- [NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) - For PTX inspection

---

*This document provides a comprehensive plan for implementing symbolic code generation to reduce register pressure. Implementation should proceed phase by phase with validation at each step. Phase 1 serves as a go/no-go decision point - if compiler optimization is already effective, the project should be reconsidered.*
