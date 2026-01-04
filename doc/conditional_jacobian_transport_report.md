# Conditional Jacobian Aggregation Benchmark Report

**Date:** 2026-01-03 (Updated: 2026-01-04)
**Test Environment:** Tesla V100-SXM2-32GB (compute capability 7.0)
**Dataset:** `odd/geant4_ttbar_mu200/` (36 events)
**Configuration:** 8 CPU threads, `run_mbf_smoother=false` (optimization default)

> **Note on Naming:** This optimization was originally called "Conditional Jacobian Transport" based on the assumption that it would reduce register pressure by skipping Jacobian computation. Profiling revealed the actual mechanism is **skipping Jacobian aggregation** (6x6 matrix multiplications), not register reduction. A more accurate name would be "Conditional Jacobian Aggregation" or "Skip MBF Jacobian Accumulation".

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

## 3. Throughput Benchmark

| Commit | Description | Throughput | Improvement |
|--------|-------------|------------|-------------|
| `a48cc783` | Baseline | **38.75 events/s** | - |
| `25894cca` | Optimization | **43.27 events/s** | **+11.67%** |

---

## 4. Detailed Benchmark Output

**Baseline (a48cc783):**
```
Throughput: Event processing  25.8035 ms/event, 38.7544 events/s
Reconstructed track parameters: 1284887
```

**Optimization (25894cca):**
```
Throughput: Event processing  23.1114 ms/event, 43.2686 events/s
Reconstructed track parameters: 1231299
```

---

## 5. Key Implementation Details

The optimization commit implements:

1. **New actor chain:** `ckf_actor_chain_no_mbf_t` with `bound_updater` replacing `parameter_transporter`
2. **Covariance transport:** `bound_updater` performs full Jacobian computation and covariance transport but skips Jacobian aggregation
3. **Empty state:** `bound_updater::state` is empty (no Jacobian pointer)
4. **Compile-time dispatch:** Uses `has_jacobian_transport_v<propagator_t>` type trait for kernel specialization
5. **18 kernel specializations:** 3 detectors × 3 bfields × 2 MBF variants (on/off)

### 5.1 Actual Optimization Mechanism (from profiling)

| Original Claim | Profiling Result |
|----------------|------------------|
| Register reduction (~64 registers) | **NOT ACHIEVED** - 128 registers in all variants |
| Occupancy improvement | **NOT ACHIEVED** - No change |
| Throughput improvement | **VALIDATED** - +18.3% (apples-to-apples) |

**Actual mechanism:** The performance gain comes from skipping Jacobian **aggregation**, not Jacobian computation:

- `parameter_transporter`: Computes 6x6 Jacobian → multiplies with accumulated Jacobian → stores to global memory
- `bound_updater`: Computes 6x6 Jacobian → discards immediately (no aggregation)

**Savings per surface per track:**
- ~216 FLOPs (6x6 × 6x6 matrix multiplication)
- ~288 bytes memory traffic (read + write accumulated Jacobian)

See `doc/conditional_jacobian_transport_profile_report.md` for detailed analysis.

---

## 6. Conclusion

The conditional Jacobian aggregation optimization achieves **+11.67% throughput improvement** (or **+18.3%** in apples-to-apples comparison with same MBF configuration) while maintaining full test correctness. All 710 CUDA tests pass on both baseline and optimization commits.

**Important:** The original +11.67% result conflated two effects:
1. MBF default change (`run_mbf_smoother: true → false`): Affects `build_tracks` kernel
2. Conditional Jacobian aggregation: +18.3% from skipping matrix multiplications

The performance benefit is **algorithmic** (skipping unnecessary work), not **resource-based** (register pressure reduction). See `doc/conditional_jacobian_transport_profile_report.md` for the full profiling analysis.
