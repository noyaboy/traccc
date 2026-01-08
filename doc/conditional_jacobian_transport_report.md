# Conditional Jacobian Aggregation Benchmark Report

**Date:** 2026-01-03 (Updated: 2026-01-06)
**Test Environment:** Tesla V100-SXM2-32GB (compute capability 7.0)
**Dataset:** `odd/geant4_ttbar_mu200/` (36 events)
**Configuration:** 8 CPU threads, `run_mbf_smoother=false` (optimization default)

> **Note on Naming:** This optimization was originally called "Conditional Jacobian Transport" based on the assumption that it would reduce register pressure by skipping Jacobian computation. NCU profiling revealed two complementary mechanisms: (1) **register reduction** (128→96 on sm_75) enabling higher occupancy, and (2) **skipping Jacobian aggregation** (6x6 matrix multiplications) reducing instruction count. The name "Conditional Jacobian Aggregation" reflects the algorithmic change.

---

## 1. Commits Under Test

| Commit | Description | Branch |
|--------|-------------|--------|
| `a48cc783` | First round of MBF cleanup (baseline) | main |
| `25894cca` | Conditional Jacobian transport (optimization) | feature/register-presure |

---

## 2. Test Results

| Commit | Test Suite | Result |
|--------|------------|--------|
| `a48cc783` (baseline) | `traccc_test_cuda` | **710/710 PASSED** |
| `25894cca` (optimization) | `traccc_test_cuda` | **710/710 PASSED** |

---

## 3. Throughput Benchmark (Apples-to-Apples)

| Commit | Description | Throughput | Latency | Improvement |
|--------|-------------|------------|---------|-------------|
| `a48cc783` | Baseline (MBF=false) | **36.57 events/s** | 27.34 ms | - |
| `25894cca` | Optimization (MBF=false) | **43.27 events/s** | 23.11 ms | **+18.3%** |

> **Note:** Both configurations use `run_mbf_smoother=false` for a fair comparison isolating the Jacobian aggregation optimization.

---

## 4. Detailed Benchmark Output (Apples-to-Apples)

**Baseline (a48cc783, MBF=false):**
```
Throughput: Event processing  27.34 ms/event, 36.57 events/s
```

**Optimization (25894cca, MBF=false):**
```
Throughput: Event processing  23.11 ms/event, 43.27 events/s
```

---

## 5. Key Implementation Details

The optimization commit implements:

1. **New actor chain:** `ckf_actor_chain_no_mbf_t` with `bound_updater` replacing `parameter_transporter`
2. **Covariance transport:** `bound_updater` performs full Jacobian computation and covariance transport but skips Jacobian aggregation
3. **Empty state:** `bound_updater::state` is empty (no Jacobian pointer)
4. **Compile-time dispatch:** Uses `has_jacobian_transport_v<propagator_t>` type trait for kernel specialization
5. **18 kernel specializations:** 3 detectors × 3 bfields × 2 MBF variants (on/off)

### 5.1 Optimization Mechanism (from NCU profiling)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Registers (sm_75) | 128 | 96 | **-25%** |
| Theoretical Occupancy | 50% | 62.5% | **+12.5pp** |
| Achieved Occupancy | 39.3% | 48.6% | **+9.3pp** |
| Executed Instructions | 82.3M | 78.5M | **-4.6%** |
| Kernel Duration | 3.99 ms | 3.61 ms | **-9.5%** |

**Dual optimization mechanism:**
1. **Register reduction** (architecture-dependent): Enables higher occupancy and better latency hiding
2. **Skipped aggregation** (universal): Eliminates 6x6 matrix multiplication and global memory I/O

**Actor comparison:**
- `parameter_transporter`: Computes 6x6 Jacobian → multiplies with accumulated Jacobian → stores to global memory
- `bound_updater`: Computes 6x6 Jacobian → discards immediately (no aggregation)

**Savings per surface per track:**
- ~216 FLOPs (6x6 × 6x6 matrix multiplication)
- ~288 bytes memory traffic (read + write accumulated Jacobian)

See `doc/conditional_jacobian_transport_ncu_results.md` for detailed NCU profiling data.

---

## 6. Conclusion

The conditional Jacobian aggregation optimization achieves **+18.3% throughput improvement** in apples-to-apples comparison (both baseline and optimization with MBF=false) while maintaining full test correctness. All 710 CUDA tests pass on both baseline and optimization commits.

**Key achievements:**
- **+18.3%** throughput (36.57 → 43.27 events/s)
- **-25%** register usage on sm_75 (128 → 96 registers)
- **+9.3pp** achieved occupancy (39.3% → 48.6%)
- **-4.6%** executed instructions

The performance benefit comes from a **dual mechanism**: (1) architecture-dependent register reduction enabling higher occupancy, and (2) universal instruction reduction from skipping unnecessary matrix multiplications. See `doc/conditional_jacobian_transport_ncu_results.md` for detailed profiling data.
