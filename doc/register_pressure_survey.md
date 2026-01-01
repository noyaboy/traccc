# Register Pressure Solutions Survey

**Date:** 2026-01-01
**Issue:** GitHub #851 - Register Usage in Performance-Critical Kernels
**Status:** Research Complete

> **Development Note:** Detray has been forked to `detray-fork/` for local modifications.
> To use the fork, update `extern/detray/CMakeLists.txt` to point to the local path.

---

## Executive Summary

GitHub Issue #851 identifies **register pressure** as a key performance bottleneck in traccc's GPU kernels, causing occupancy as low as 10-25%. This document surveys solutions for reducing register pressure and improving kernel occupancy.

**Key Distinction:** Previous optimization work in this repository focused on **warp divergence** (~62% wasted cycles from variable RK4 step counts). Register pressure is a **separate problem** - too many registers per thread limiting concurrent warps on each SM.

**Recommendation:** Start with CUDA 13's shared memory register spilling (`enable_smem_spilling` pragma), then profile and iterate.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Previous Approaches (Warp Divergence - Different Problem)](#2-previous-approaches-warp-divergence---different-problem)
3. [Register Pressure Solutions Survey](#3-register-pressure-solutions-survey)
4. [Detailed Solution Analysis](#4-detailed-solution-analysis)
5. [Root Cause Analysis](#5-root-cause-analysis)
6. [Recommendations](#6-recommendations)
7. [References](#7-references)

---

## 1. Problem Definition

### 1.1 The Issue

From GitHub Issue #851:
> The `propagate_to_next_surface` kernel achieves only 10-25% occupancy due to excessive register consumption. The root cause is linear algebra implementations prioritizing readability over performance optimization.

### 1.2 Register Pressure vs Warp Divergence

| Problem | Cause | Symptom | Impact |
|---------|-------|---------|--------|
| **Register Pressure** | Too many registers per thread | Low occupancy (10-25%) | SM underutilization |
| **Warp Divergence** | Variable RK4 step counts (1-31) | Threads waiting for slowest | ~62% wasted cycles |

These are **independent problems** requiring different solutions. Previous work in `doc/work_redistribution_*.md` addressed warp divergence. This document addresses register pressure.

### 1.3 Occupancy Fundamentals

On NVIDIA V100 (Volta):
- 65,536 registers per SM
- Maximum 2,048 threads per SM
- For 100% occupancy: 65,536 / 2,048 = **32 registers/thread max**

Current kernel analysis (**profiled 2026-01-01**):
- `propagate_to_next_surface`: 128-203 registers/thread (measured)
- `fit_forward` / `fit_backward`: 168 registers/thread (measured)
- Maximum occupancy: 16-25% (theoretical), likely lower in practice

### 1.4 Affected Kernels

| Kernel | File | `__launch_bounds__` |
|--------|------|---------------------|
| `propagate_to_next_surface` | `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh:21` | 128 |
| `fit_forward` | `device/cuda/src/fitting/kernels/specializations/fit_forward_src.cuh:17` | 128 |
| `fit_backward` | `device/cuda/src/fitting/kernels/specializations/fit_backward_src.cuh:17` | 128 |

### 1.5 Profiled Register Usage (2026-01-01)

Compiled with `nvcc -Xptxas -v` on sm_70 (V100), CUDA 12.6:

#### Summary by Register Count

| Registers | Count | Kernel Type | Max Occupancy (V100) |
|-----------|-------|-------------|----------------------|
| **203** | 1 | `propagate_to_next_surface` (inhom B-field) | **16%** |
| **168** | 14 | `fit_forward` / `fit_backward` (all variants) | **19%** |
| **161** | 1 | `propagate_to_next_surface` | **20%** |
| **128** | 9 | `propagate_to_next_surface` / `fit_forward` | **25%** |
| **31-32** | 3 | `apply_interaction` | **100%** |
| **4** | 3 | utility kernels | **100%** |

#### Detailed Breakdown by Kernel

**Propagation Kernels (`propagate_to_next_surface`):**

| Detector | B-field | Registers | Stack (bytes) | Occupancy |
|----------|---------|-----------|---------------|-----------|
| ODD | inhom_global | **203** | 3,480 | 16% |
| ODD | inhom_texture | 168 | 48 | 19% |
| ODD | constant | 168 | 48 | 19% |
| default | inhom_global | 168 | 3,608 | 19% |
| default | inhom_texture | 168 | 3,200 | 19% |
| default | constant | 168 | 1,976 | 19% |
| telescope | inhom_global | 161 | 1,184 | 20% |
| telescope | inhom_texture | 128 | 2,920 | 25% |
| telescope | constant | 128 | 1,856 | 25% |

**Fitting Kernels (`fit_forward` / `fit_backward`):**

| Detector | B-field | Registers | Stack (bytes) | Occupancy |
|----------|---------|-----------|---------------|-----------|
| default | inhom_global | 168 | 3,608 | 19% |
| default | inhom_texture | 168 | 3,200 | 19% |
| default | constant | 128 | 3,624 | 25% |
| ODD | inhom_global | 168 | 1,904 | 19% |
| ODD | inhom_texture | 168 | 1,864 | 19% |
| ODD | constant | 128 | 2,616 | 25% |
| telescope | inhom_global | 168 | 1,400 | 19% |
| telescope | inhom_texture | 168 | 1,328 | 19% |
| telescope | constant | 168 | 1,288 | 19% |

**Interaction Kernels (`apply_interaction`):**

| Detector | Registers | Occupancy |
|----------|-----------|-----------|
| ODD | 32 | 100% |
| default | 32 | 100% |
| telescope | 31 | 100% |

#### Key Observations

1. **Worst case: 203 registers** - ODD detector with inhomogeneous global B-field
2. **Fitting consistently high**: All `fit_forward`/`fit_backward` use 128-168 registers
3. **B-field type matters**: Inhomogeneous fields add ~40 registers vs constant
4. **Detector complexity**: ODD > default > telescope in register usage
5. **Interaction kernels are fine**: Only 31-32 registers, 100% occupancy

#### Validation

| Metric | Document Estimate | Measured | Status |
|--------|-------------------|----------|--------|
| Register count | ~120+ | 128-203 | **CONFIRMED** |
| Occupancy | 10-25% | 16-25% theoretical | **CONFIRMED** |
| Register pressure issue | Yes | Yes | **VALIDATED** |

---

## 2. Previous Approaches (Warp Divergence - Different Problem)

The following approaches were investigated for **warp divergence**, not register pressure. They are documented here for completeness and to clarify they do not address issue #851.

### 2.1 Summary Table

| Approach | Outcome | Failure Reason | Documentation |
|----------|---------|----------------|---------------|
| Pre-Sort by \|qop\| | INVALIDATED | Correlation = -0.03 (none) | `doc/work_redistribution_plan.md` §2 |
| Two-Phase Separation | INVALIDATED | No predictor exists | `doc/work_redistribution_plan.md` §6 |
| Block Work-Stealing | NO BENEFIT | `__syncthreads_or()` barrier blocks all threads | `doc/issue_of_redesign_propagator.md` |
| Chunked Propagation | 5-13% SLOWER | Checkpoint serialization overhead | `doc/issue_of_chunked_propagator.md` |
| Lazy Checkpointing | 2.3% SLOWER | Branch misprediction overhead | `doc/issue_of_chunked_propagator.md` |
| Step Count Predictor | NOT VIABLE | \|eta\| best at r=-0.2028 | `doc/step_count_correlation_analysis.md` |
| Warp Specialization | BLOCKED | No predictor found | `doc/work_redistribution_approaches_summary.md` |
| Cooperative Stepping | NOT ATTEMPTED | ~1150 lines of detray changes | `doc/work_redistribution_plan.md` §8 |

### 2.2 Key Learnings

1. **Track parameters do not predict step count** - |qop|, theta, eta all show near-zero correlation
2. **Work redistribution requires interruptible work** - Monolithic `propagator.propagate()` cannot be paused efficiently
3. **Overhead exceeds benefit** - Any checkpointing/serialization costs more than potential gains

### 2.3 Why These Don't Help Register Pressure

All warp divergence approaches assume:
- Fixed register usage per thread
- Variable execution time is the problem

Register pressure solutions must:
- Reduce registers per thread
- Accept potentially longer execution time per thread
- Increase concurrent thread count to compensate

---

## 3. Register Pressure Solutions Survey

### 3.1 Solutions Overview

| Solution | Viability | Effort | Expected Benefit | Complexity |
|----------|-----------|--------|------------------|------------|
| **CUDA 13 Shared Memory Spilling** | HIGH | LOW | 7-8% speedup | Add pragma |
| **`--maxrregcount` Compiler Flag** | MEDIUM | LOW | Variable | Compiler flag |
| **`__launch_bounds__` Tuning** | MEDIUM | LOW | 5-15% | Experiment |
| **Kernel Fission** | MEDIUM | HIGH | 20-40% | Restructure |
| **Manual Shared Memory Spilling** | MEDIUM | HIGH | 10-20% | Identify hot vars |
| **Symbolic Code Generation** | HIGH | VERY HIGH | 30-50% | Long-term |
| **Mixed Precision (float16)** | LOW | MEDIUM | 10-20% | Precision risk |
| **32-bit Integer Optimization** | LOW-MEDIUM | LOW | 5-10% | Code audit |
| **Algorithmic Refactoring** | MEDIUM-HIGH | HIGH | 20-40% | Domain expertise |

### 3.2 Viability Assessment Criteria

- **HIGH**: Proven technique, low risk, clear implementation path
- **MEDIUM**: May work, requires experimentation, moderate risk
- **LOW**: Significant drawbacks or uncertain benefit

---

## 4. Detailed Solution Analysis

### 4.1 CUDA 13 Shared Memory Register Spilling (Recommended)

**New in CUDA 13.0** - Uses on-chip shared memory for register spilling instead of L2 cache.

> **Note:** CUDA 13 may not yet be released. Verify toolkit availability before implementation.
> The exact inline assembly syntax below is based on NVIDIA's blog post and may need
> adjustment when the feature becomes available.

#### Implementation

```cpp
__global__ __launch_bounds__(128) void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    // Enable shared memory spilling (CUDA 13+)
    // Note: Syntax may need verification with actual CUDA 13 release
    asm volatile(".pragma \"enable_smem_spilling\";");

    device::propagate_to_next_surface<propagator_t, bfield_t>(
        details::global_index1(), cfg, payload);
}
```

#### Benefits

- **7.76% kernel speedup** reported by NVIDIA
- **Zero spill stores/loads** to L2
- **Minimal code changes** - single pragma
- **On-chip latency** - shared memory ~5 cycles vs L2 ~200+ cycles

#### Requirements

- CUDA Toolkit 13.0 or later (verify release status: `nvcc --version`)
- Compute capability 7.0+ (Volta, Turing, Ampere, Hopper)
- Sufficient shared memory available (48KB default)

#### Caveats

- Reduces available shared memory for other uses
- May conflict with explicit shared memory allocations
- Feature availability depends on CUDA 13 release timeline
- Syntax may differ from examples - consult official documentation when available

#### References

- [NVIDIA Blog: Shared Memory Register Spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)

---

### 4.2 Compiler Register Limiting (`--maxrregcount`)

Forces compiler to limit registers per thread, spilling excess to local memory.

#### Implementation

```cmake
# Modern CMake (recommended)
target_compile_options(traccc_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>)

# Or legacy CMake style
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=32")
```

Or per-kernel:
```cpp
__global__ __launch_bounds__(128, 16) void kernel() {
    // 16 blocks/SM hint → compiler targets ~32 regs/thread
}
```

#### Expected Outcome

| Registers/Thread | Max Threads/SM | Occupancy |
|------------------|----------------|-----------|
| 32 | 2048 | 100% |
| 40 | 1638 | 80% |
| 64 | 1024 | 50% |
| 128 | 512 | 25% |

#### Caveats

- **Increased spill traffic** to local memory (L2 cache)
- **Potential performance regression** if ILP suffers
- **Must benchmark** - occupancy increase may not offset spill cost

#### When to Use

- When occupancy is severely limited (<30%)
- When kernel is memory-bound (not compute-bound)
- As baseline comparison for other optimizations

---

### 4.3 `__launch_bounds__` Tuning

Provides compiler hints about expected thread/block configuration.

#### Current Usage

```cpp
// propagate_to_next_surface_src.cuh:21
__global__ __launch_bounds__(128) void propagate_to_next_surface(...)
```

#### Optimization Options

```cpp
// Option A: Target higher occupancy
__global__ __launch_bounds__(128, 8) void kernel()  // 8 blocks/SM

// Option B: Reduce thread count for more registers
__global__ __launch_bounds__(64, 16) void kernel()  // 64 threads, 16 blocks

// Option C: Maximum occupancy hint
__global__ __launch_bounds__(256, 8) void kernel()  // 256 threads, 8 blocks
```

#### Experimentation Matrix

| `__launch_bounds__` | Expected Regs/Thread | Expected Occupancy |
|---------------------|----------------------|-------------------|
| `(128)` (current) | ~120 | 25% |
| `(128, 8)` | ~64 | 50% |
| `(128, 16)` | ~32 | 100% |
| `(64, 16)` | ~64 | 50% |
| `(256, 8)` | ~32 | 100% |

#### Implementation

Test each configuration with profiling:
```bash
nvcc --ptxas-options=-v -o kernel.o kernel.cu
# Look for "Used N registers" in output
```

---

### 4.4 Kernel Fission

Split large kernels into smaller, specialized kernels with lower register requirements.

#### Current Structure

```
propagate_to_next_surface():
├── Setup phase (container access, parameter loading)
├── Propagator state construction
├── Actor state initialization (7 actors)
├── RK4 propagation loop (variable iterations)
├── Result validation
└── Tip creation
```

#### Proposed Fission

```
Kernel 1: setup_propagation()
├── Load track parameters
├── Initialize propagator state
└── Write to intermediate buffer

Kernel 2: rk4_step() [called in loop from host]
├── Read current state
├── Execute single RK4 step
├── Check termination
└── Write updated state

Kernel 3: finalize_propagation()
├── Validate results
├── Update liveness
└── Create tips
```

#### Trade-offs

| Pro | Con |
|-----|-----|
| Lower registers per kernel | Kernel launch overhead (~5-10 μs each) |
| Better occupancy | State must be stored in global memory |
| Simpler kernels easier to optimize | More complex host-side orchestration |

#### Feasibility

- **HIGH** implementation effort
- **MEDIUM** expected benefit (20-40% if launch overhead acceptable)
- Requires intermediate state buffer (~240 bytes/track)
- Similar to chunked propagation approach but without work redistribution

---

### 4.5 Manual Shared Memory Spilling

Explicitly move hot variables from registers to shared memory.

#### Candidate Variables

Based on `propagate_to_next_surface.ipp` analysis:

| Variable | Size | Access Pattern | Spill Candidate? |
|----------|------|----------------|------------------|
| Covariance matrix (6×6) | 144 bytes | Read/write each step | YES |
| Jacobian (8×8) | 256 bytes | Accumulated | YES |
| RK4 derivatives (4×3 vectors) | 48 bytes | Per-step temporary | NO (hot) |
| Navigation cache (8 candidates) | 192-256 bytes | Infrequent access | YES |
| Track parameters (6 floats) | 24 bytes | Frequent access | NO (hot) |

#### Implementation Pattern

```cpp
__shared__ float shared_covariance[128][21];  // 21 upper-triangle elements

// In kernel:
const int tid = threadIdx.x;

// Load to shared at start
for (int i = 0; i < 21; i++) {
    shared_covariance[tid][i] = params.covariance(i);
}
__syncthreads();

// ... use shared_covariance[tid][...] instead of local array ...

// Store back at end
for (int i = 0; i < 21; i++) {
    params.set_covariance(i, shared_covariance[tid][i]);
}
```

#### Shared Memory Requirements

For 128 threads:
| Component | Size |
|-----------|------|
| Covariance (21 floats × 4 bytes × 128) | 10,752 bytes |
| Jacobian (64 floats × 4 bytes × 128) | 32,768 bytes |
| Navigation cache (~224 bytes × 128) | ~28,672 bytes |
| **Total** | ~72 KB |

**Problem:** Exceeds 48 KB default shared memory limit. Would need:
- Reduce block size to 64 threads, OR
- Use extended shared memory (96 KB on V100), OR
- Only spill subset of variables

---

### 4.6 Symbolic Code Generation

Use symbolic execution to generate optimized kernels with minimal registers.

#### Background

Referenced in issue #851: "@andiwand's symbolic execution-based code generator"

#### Approach

1. Express linear algebra symbolically (e.g., SymPy, CasADi)
2. Apply algebraic simplifications
3. Common subexpression elimination (CSE)
4. Optimal register allocation
5. Generate CUDA/C++ code

#### Example: Matrix Multiplication

Before (naive):
```cpp
for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
        for (int k = 0; k < 6; k++)
            C[i][j] += A[i][k] * B[k][j];  // 36 registers for A, 36 for B, 36 for C
```

After (symbolic, unrolled with CSE):
```cpp
// Compiler-generated optimal sequence
float t0 = A[0]*B[0];
float t1 = A[1]*B[6];
// ... minimal temporaries, maximal reuse
```

#### Implementation

- **Effort:** VERY HIGH (months of work)
- **Benefit:** 30-50% register reduction possible
- **Risk:** Maintenance burden, debugging difficulty

#### Status

- Being worked on by Yuki Asami (per issue #851)
- Long-term solution, not immediate fix

#### 4.6.1 Current Codegen Infrastructure Survey (2026-01-01)

A comprehensive survey of traccc and its dependencies reveals the following code generation patterns:

##### Existing Python-Based Kernel Specialization

| Component | Location | Purpose |
|-----------|----------|---------|
| `gen_kernel_specialization.py` | `codegen/kernel_specialization/` (73 lines) | String template substitution |
| `.cu.template` files | `device/cuda/src/*/kernels/specializations/` | 5 templates |
| CMake generation | `device/cuda/CMakeLists.txt:126-230` | 44 specialized kernels |

**How it works** (`gen_kernel_specialization.py`):
```python
# Simple string.Template substitution for:
# - ${DETECTOR_NAME} → detector type (odd_detector, default_detector, etc.)
# - ${BFIELD_NAME} → magnetic field type (const, inhom_global, inhom_texture)
# - ${MODEL} → programming model (cuda, alpaka, sycl)
```

**Generated kernels** (per CMakeLists.txt):
- `find_tracks`: 4 detector variants
- `apply_interaction`: 4 detector variants
- `propagate_to_next_surface`: 4 detector × 3 bfield = 12 variants
- `fit_forward` / `fit_backward`: 4 detector × 3 bfield × 2 = 24 variants

**Limitation**: This is **type specialization only**, not algorithm-level symbolic generation.

##### Hard-Coded Matrix Operations in algebra-plugins

The `algebra-plugins` library (dependency) uses **manually expanded** matrix formulas:

**4×4 Matrix Inverse** (`_deps/algebraplugins-src/math/generic/.../inverse/hard_coded.hpp:52-273`):
```cpp
// 16 cofactor computations, each with 6 triple-product terms
element_getter()(ret, 0, 0) =
    element_getter()(m, 1, 2) * element_getter()(m, 2, 3) * element_getter()(m, 3, 1) -
    element_getter()(m, 1, 3) * element_getter()(m, 2, 2) * element_getter()(m, 3, 1) +
    element_getter()(m, 1, 3) * element_getter()(m, 2, 1) * element_getter()(m, 3, 2) -
    // ... 3 more terms per element, 16 elements total
```

**Register impact**: ~120 multiply-add operations with NO shared temporaries.

**4×4 Determinant** (`determinant/hard_coded.hpp:40-94`):
- 48-term expanded expression
- All `ALGEBRA_HOST_DEVICE constexpr`

**Algorithm selection** (`algorithm_finder.hpp:22-57`):
- Uses C++20 `requires` clauses for compile-time size dispatch
- 2×2 and 4×4: hard-coded specializations
- Other sizes: LU decomposition (more registers)

##### Jacobian Handling - NOT Symbolically Generated

| Stage | Location | Method |
|-------|----------|--------|
| Transport Jacobian | detray `parameter_transporter` actor | Numerical (RK4 accumulation) |
| Jacobian storage | `combinatorial_kalman_filter.cuh:166-188` | `bound_matrix<>[]` buffer |
| MBF accumulation | `build_tracks.ipp:73-126` | Runtime matrix multiply chain |
| Kalman gain | `gain_matrix_updater.hpp:117-134` | `transposed_product`, `inverse` |

**Key code paths** (`propagate_to_next_surface.ipp:103-115`):
```cpp
// Initialize to identity
payload.tmp_jacobian_ptr[param_id] = matrix::identity<bound_matrix<...>>();
// Detray's parameter_transporter multiplies Jacobians during propagation
s1._full_jacobian_ptr = &payload.tmp_jacobian_ptr[param_id];
```

**MBF smoothing** (`build_tracks.ipp:123-126`):
```cpp
small_lambda_hat = matrix::transpose(accumulated_jacobian) * small_lambda_tilde;
big_lambda_hat = matrix::transpose(accumulated_jacobian) * big_lambda_tilde * accumulated_jacobian;
```

These are **runtime matrix operations**, not compile-time generated code.

##### Template Metaprogramming Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| Tuple unpacking | `propagate_to_next_surface.ipp:82-101` | 7 actor states from tuple |
| `requires` clauses | `inverse/hard_coded.hpp:36,53` | Compile-time size dispatch |
| `if constexpr` | `kalman_actor.hpp`, `magnetic_field.hpp` | Direction/type dispatch |
| `constexpr` functions | Throughout algebra-plugins | Potential compile-time eval |

**Actor state extraction** (causes register bloat):
```cpp
// propagate_to_next_surface.ipp:85-101 - 7 actor states kept live simultaneously
using s0_type = detray::detail::tuple_element<0, actor_tuple_type>::type::state;  // PathlimitAborter
using s1_type = detray::detail::tuple_element<1, actor_tuple_type>::type::state;  // ParameterTransporter
using s2_type = detray::detail::tuple_element<2, actor_tuple_type>::type::state;  // InteractionRegister
// ... s3 through s6
```

#### 4.6.2 Gap Analysis: Current vs Required for Symbolic Codegen

| Capability | Current State | Required for 30-50% Reduction |
|------------|---------------|-------------------------------|
| Symbolic math representation | None | SymPy/CasADi expression trees |
| Algebraic simplification | None | Symbolic simplification passes |
| Common subexpression elimination | Compiler-only | Explicit CSE before code emission |
| Register-aware scheduling | Compiler-driven | Custom instruction ordering |
| 6×6 covariance specialization | Uses generic N×N | Hard-coded with CSE |
| Fused Kalman formulas | Separate operations | `K = PC'(CPC'+V)^{-1}` as single unit |

#### 4.6.3 Optimization Targets for Symbolic Codegen

Based on codebase analysis, highest-impact targets:

| Target | Current Registers (est.) | Potential Savings |
|--------|--------------------------|-------------------|
| 6×6 bound covariance transport | 36+ (full matrix live) | 15-20 with row-wise CSE |
| 8×8 Jacobian accumulation | 64 (full matrix) | 20-30 with symbolic fusion |
| Kalman gain computation | 40+ (intermediates) | 15-20 with fused formula |
| 4×4 inverse (algebra-plugins) | 16+ (no temp sharing) | 5-10 with CSE |

**Example optimization** - Current 4×4 inverse cofactor:
```cpp
// Current: 6 independent triple-products per element, no sharing
m[1][2] * m[2][3] * m[3][1]  // Computed fresh
m[1][3] * m[2][2] * m[3][1]  // m[3][1] recomputed
```

**With CSE**:
```cpp
float t_31 = m[3][1];  // Shared temporary
float t_12_23 = m[1][2] * m[2][3];
float t_13_22 = m[1][3] * m[2][2];
// Reuse t_31 across multiple cofactors
```

#### 4.6.4 Implementation Roadmap

**Phase 1: Proof of Concept** (2-4 weeks)
- Target: 6×6 symmetric covariance transport
- Tool: Python + SymPy for expression generation
- Output: Single optimized C++ header

**Phase 2: Integration** (4-8 weeks)
- Integrate generated code with traccc build
- Add CMake targets for regeneration
- Benchmark against current implementation

**Phase 3: Expansion** (2-3 months)
- Extend to Jacobian operations
- Add Kalman gain fusion
- Consider detray upstream changes

**Phase 4: Full System** (3-6 months)
- RK4 stepper symbolic generation
- Register-aware instruction scheduling
- Maintenance tooling

#### 4.6.5 Conclusion

**No existing symbolic codegen infrastructure** exists in traccc or its dependencies. The current codegen is limited to type specialization via Python string templates. Implementing true symbolic code generation would require:

1. Building expression representation from scratch
2. Implementing CSE and simplification passes
3. Creating register-aware code emitters
4. Significant testing and validation effort

The "symbolic code generation" mentioned in GitHub #851 would need to be built as a new system, not extended from existing infrastructure.

---

### 4.7 Mixed Precision (float16)

Use half-precision where full precision not required.

#### Candidates

| Computation | Current | float16 OK? |
|-------------|---------|-------------|
| Track position | float32 | NO (precision critical) |
| Track direction | float32 | MAYBE |
| Covariance | float32 | NO (numerical stability) |
| B-field lookup | float32 | MAYBE |
| Intermediate RK4 | float32 | NO (error control) |

#### Implementation

```cpp
#include <cuda_fp16.h>

__half2 bfield = __floats2half2_rn(bx, by);
// ... compute in half precision ...
float result = __half2float(result_h);
```

#### Caveats

- **Precision loss** in tracking can cause physics failures
- Half-precision has limited range (±65504)
- Requires Volta+ for efficient `__half2` operations
- **NOT RECOMMENDED** for track fitting without extensive validation

---

### 4.8 32-bit Integer Optimization

Replace 64-bit integers with 32-bit where possible.

#### Problem

```cpp
// Common pattern in GPU code
size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 64-bit
```

Each 64-bit integer uses 2 registers instead of 1.

#### Solution

```cpp
// Use 32-bit explicitly
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 32-bit
```

#### Code Audit Targets

- Index calculations
- Loop counters
- Temporary arithmetic
- `size_t` usage (often unnecessary for GPU indices)

#### Expected Benefit

- 5-10% register reduction if widespread 64-bit usage
- Low effort, low risk
- Should be done regardless of other optimizations

---

### 4.9 Algorithmic Refactoring

Restructure algorithms to reduce simultaneous live variables.

#### Example: Covariance Transport

Current (all elements live):
```cpp
float cov[6][6];  // 36 registers
jacobian_transport(cov, jacobian);  // All elements modified
```

Restructured (row-by-row):
```cpp
for (int row = 0; row < 6; row++) {
    float cov_row[6];  // 6 registers
    load_row(cov_row, row);
    transform_row(cov_row, jacobian);
    store_row(cov_row, row);
}
```

#### Trade-offs

- Reduces peak register usage
- May increase memory traffic
- May reduce instruction-level parallelism
- Requires deep understanding of algorithm

---

## 5. Root Cause Analysis

### 5.1 Register-Heavy Components

Based on `propagate_to_next_surface.ipp` and detray propagator analysis:

| Component | Registers (est.) | Location |
|-----------|------------------|----------|
| Bound track parameters (6 floats) | 6 | `propagate_to_next_surface.ipp` |
| Covariance matrix (6×6 symmetric) | 21 | Implicit in track params |
| Jacobian transport (8×8) | 64 | `parameter_transporter` (s1) state |
| RK4 derivatives (4 stages × 3D) | 12 | `rk_stepper.ipp` |
| B-field vectors (3 points × 3D) | 9 | `intermediate_state` struct |
| Navigation state | 8-12 | `navigation_state.hpp` |
| Actor states (s0-s6) | 20-30 | Actor state initialization |
| Propagator temporaries | 10-20 | Various |
| **Total estimated** | **150-180** | **Requires profiling to verify** |

### 5.2 Why So Many Registers?

1. **Template instantiation** - detray uses heavy templates, increasing code size
2. **Inlining** - `DETRAY_HOST_DEVICE inline` forces inlining, keeping all locals live
3. **Complex control flow** - Adaptive RK4 loop with multiple branches
4. **No explicit spilling** - Compiler chooses suboptimal spill strategy
5. **Multiple actor states** - 7 actors each with their own state structure

### 5.3 Compiler Behavior

The NVCC compiler:
1. Inlines all `__device__` functions by default
2. Keeps all variables live across function boundaries
3. Only spills to local memory (L2) when absolutely necessary
4. Does not use shared memory for spilling (until CUDA 13)

---

## 6. Recommendations

### 6.1 Immediate Actions (Low Effort)

1. **Profile actual register usage**
   ```bash
   nvcc --ptxas-options=-v device/cuda/src/finding/kernels/specializations/propagate_to_next_surface.cu
   ```

2. **Try CUDA 13 shared memory spilling**
   - Add `asm("enable_smem_spilling");` to kernel
   - Benchmark before/after

3. **Audit 64-bit integer usage**
   - Search for `size_t`, `long`, `int64_t` in device code
   - Replace with 32-bit where safe

### 6.2 Short-term Actions (Medium Effort)

4. **Experiment with `--maxrregcount`**
   - Try values: 32, 40, 48, 64
   - Benchmark each configuration

5. **Tune `__launch_bounds__`**
   - Add minBlocksPerSM hint: `__launch_bounds__(128, 8)`
   - Experiment with thread counts: 64, 128, 256

6. **Profile with Nsight Compute**
   ```bash
   ncu --set full ./bin/traccc_throughput_st_cuda ...
   ```
   Focus on:
   - `sm__warps_active.avg.pct_of_peak_sustained_active` (occupancy)
   - `lts__t_sectors_srcunit_tex_op_read.sum` (L2 spills)

### 6.3 Long-term Actions (High Effort)

7. **Investigate kernel fission**
   - Prototype split propagation kernel
   - Measure launch overhead vs occupancy benefit

8. **Support symbolic code generation effort**
   - Coordinate with Yuki Asami's work
   - Identify highest-impact code sections

9. **Consider upstream detray changes**
   - Propose register-optimized stepper variant
   - Explore compile-time options for reduced state

### 6.4 Priority Matrix

```
                    LOW EFFORT              HIGH EFFORT
                         |                       |
    +--------------------|----------------------+
    |                    |                      |
H   | CUDA 13 smem spill | Symbolic codegen    |
I   | --maxrregcount     | Kernel fission      |
G   | launch_bounds tune |                     |
H   | 32-bit integers    |                     |
    |                    |                      |
    +--------------------|----------------------|
P   |                    |                      |
O   | Profile registers  | Manual smem spill   |
T   |                    | Algorithmic refactor|
E   |                    |                      |
N   |                    |                      |
T   +--------------------|----------------------+
I   |                    |                      |
A   |                    | Mixed precision     |
L   |                    | (NOT RECOMMENDED)   |
    |                    |                      |
L   |                    |                      |
O   |                    |                      |
W   +--------------------|----------------------+
```

---

## 7. References

### External Resources

- [NVIDIA Blog: Shared Memory Register Spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/) - CUDA 13 feature
- [Understanding CUDA Occupancy](https://medium.com/@manisharadwad/unlocking-gpu-potential-understanding-and-optimizing-cuda-occupancy-2f43ee01ad7e) - Occupancy fundamentals
- [Reducing Register Pressure](https://app.studyraid.com/en/read/11728/371500/reducing-register-pressure) - General techniques
- [AMD Lab Notes on Register Pressure](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/) - Cross-platform insights
- [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Official NVIDIA guidance

### Internal Documentation

- `doc/problem_definition.md` - Warp divergence analysis (different problem)
- `doc/work_redistribution_approaches_summary.md` - Previous optimization attempts
- `doc/chunked_propagator_redesign_plan.md` - Serialization size analysis
- GitHub Issue #851 - Original problem report

### Key Code Locations

| File | Description |
|------|-------------|
| `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface_src.cuh` | Main propagation kernel |
| `device/cuda/src/fitting/kernels/specializations/fit_forward_src.cuh` | Forward fitting kernel |
| `device/cuda/src/fitting/kernels/specializations/fit_backward_src.cuh` | Backward fitting kernel |
| `detray-fork/core/include/detray/propagator/rk_stepper.ipp` | RK4 stepper (register-heavy) |
| `detray-fork/core/include/detray/propagator/propagator.hpp` | Propagation loop |

> **Note:** Detray paths above reference the local fork. Original upstream is at `extern/detray/`.

---

## Appendix A: Register Profiling Commands

### Check Register Usage

```bash
# Compile with verbose PTX output
nvcc --ptxas-options=-v \
     -I/path/to/includes \
     -o kernel.o \
     device/cuda/src/finding/kernels/specializations/propagate_to_next_surface.cu

# Output includes:
# ptxas info    : Used N registers, M bytes smem, K bytes cmem[0]
```

### Nsight Compute Profiling

```bash
# Full kernel analysis
ncu --set full \
    --target-processes all \
    --export report.ncu-rep \
    ./bin/traccc_throughput_st_cuda \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --input-events=10 --processed-events=100 --cpu-threads=1

# View report
ncu-ui report.ncu-rep
```

### Occupancy Calculator

```bash
# CUDA Occupancy Calculator (Excel spreadsheet from NVIDIA)
# Or programmatically:
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, dynamicSmemSize);
```

---

## Appendix B: Quick Reference

### Occupancy vs Registers (V100)

| Registers/Thread | Max Threads/SM | Occupancy | Blocks (128 threads) |
|------------------|----------------|-----------|---------------------|
| 32 | 2048 | 100% | 16 |
| 40 | 1638 | 80% | 12 |
| 48 | 1365 | 67% | 10 |
| 64 | 1024 | 50% | 8 |
| 80 | 819 | 40% | 6 |
| 96 | 682 | 33% | 5 |
| 128 | 512 | 25% | 4 |
| 255 | 256 | 12.5% | 2 |

### Memory Latencies (Approximate)

| Memory Type | Latency (cycles) |
|-------------|------------------|
| Registers | 0 |
| Shared Memory | 5-10 |
| L1 Cache | 20-30 |
| L2 Cache | 200-300 |
| Global Memory | 400-800 |

---

*This document provides a comprehensive survey of register pressure solutions for GitHub Issue #851. Implementation should proceed incrementally, measuring impact at each step.*
