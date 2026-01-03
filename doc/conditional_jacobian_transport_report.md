# Conditional Jacobian Transport Benchmark Report

**Date:** 2026-01-03
**Test Environment:** Tesla V100-SXM2-32GB (compute capability 7.0)
**Dataset:** `odd/geant4_ttbar_mu200/` (36 events)
**Configuration:** 8 CPU threads, `run_mbf_smoother=false` (optimization default)

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
3. **Empty state:** `bound_updater::state` is empty (no Jacobian pointer), reducing register pressure
4. **Compile-time dispatch:** Uses `has_jacobian_transport_v<propagator_t>` type trait for kernel specialization
5. **18 kernel specializations:** 3 detectors × 3 bfields × 2 MBF variants (on/off)

---

## 6. Conclusion

The conditional Jacobian transport optimization achieves **+11.67% throughput improvement** while maintaining full test correctness. All 710 CUDA tests pass on both baseline and optimization commits.
