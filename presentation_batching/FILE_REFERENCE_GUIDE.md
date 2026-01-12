# Presentation Sources File Reference Guide

**Purpose:** This document maps each source file to the specific slides and oral script content it supports.
**Total Files:** 22 (3 presentation + 19 documentation)
**Created:** 2026-01-06

---

## Table of Contents

1. [Presentation Files](#1-presentation-files)
2. [Core Results Documentation](#2-core-results-documentation)
3. [Memory Optimization Documentation](#3-memory-optimization-documentation)
4. [Batch Size & Architecture Documentation](#4-batch-size--architecture-documentation)
5. [Bug Fix & Validation Documentation](#5-bug-fix--validation-documentation)
6. [Negative Results Documentation](#6-negative-results-documentation)
7. [Profiling & Analysis Documentation](#7-profiling--analysis-documentation)
8. [File-to-Slide Mapping Summary](#8-file-to-slide-mapping-summary)
9. [Data Flow Diagram](#9-data-flow-diagram)

---

## 1. Presentation Files

### 1.1 progress_slides.tex
- **Type:** LaTeX Beamer source
- **Size:** 36.5 KB
- **Purpose:** Main presentation slides
- **Slides:** 18 main slides + backup Q&A slides
- **Key Packages:** tikz, pgfplots, listings, fontawesome5

### 1.2 presentation_script.md
- **Type:** Markdown
- **Size:** 21.0 KB
- **Purpose:** Oral presentation script with speaker notes
- **Structure:** Slide-by-slide narration + 13 anticipated Q&A

### 1.3 progress_slides.pdf
- **Type:** Compiled PDF
- **Size:** 173.3 KB
- **Purpose:** Final presentation output

---

## 2. Core Results Documentation

### 2.1 batching_report.md
- **Size:** 12.4 KB
- **Role:** **PRIMARY AUTHORITATIVE SOURCE** for all performance claims

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 5 | 90.5% CUDA API sync time baseline |
| 10 | 93% throughput improvement (15.61 → 30.12 ev/s) |
| 14 | 1,460 tests breakdown (core:24, io:9, examples:3, cpu:714, cuda:710) |
| 16 | 68 commits, implementation timeline |
| 17 | 98% theoretical efficiency formula |

**Key Data Points:**
- Baseline commit: `5cd477ac` (15.61 ev/s)
- Optimized commit: `3ad492b5` (30.12 ev/s)
- Improvement: +93%
- Test count: 1,460 (all pass)
- Efficiency: 98% of theoretical maximum (1.55x / 1.58x)

**Script References:**
- Lines 108-112: Performance results narration
- Lines 195-205: Summary and conclusions

---

### 2.2 batching_profile_results.md
- **Size:** 8.1 KB
- **Role:** NSYS profiling validation data

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 8 | Synchronization elimination results |
| 11 | NSYS KPI cards and kernel instance comparison |

**Key Data Points:**
- cudaStreamSynchronize: 8,525 → 706 calls (-92%)
- cudaLaunchKernel: 16,390 → 3,776 (-77%)
- cudaMemcpyAsync: 7,007 → 2,162 (-69%)
- propagate_to_next_surface: 1,391 → 42 instances
- fit_forward/backward: 72 → 2 instances

**Script References:**
- Lines 117-124: NSYS results explanation
- Line 89: "cudaStreamSynchronize calls dropped by 92%"

---

### 2.3 batching_ncu_results.md
- **Size:** 7.3 KB
- **Role:** NCU kernel-level profiling data

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 13 | Occupancy table, memory throughput table, grid size scaling |

**Key Data Points:**

| Kernel | Baseline Occupancy | Optimized Occupancy | Improvement |
|--------|-------------------|---------------------|-------------|
| fit_forward | 15.4% | 38.4% | +150% |
| fit_backward | 14.9% | 29.9% | +100% |
| propagate_to_next_surface | 35.0% | 42.8% | +22% |
| find_tracks | 21.4% | 23.7% | +11% |

| Kernel | Memory Throughput Improvement |
|--------|------------------------------|
| fit_backward | +154% (96.8 → 245.5 GB/s) |
| fit_forward | +121% (61.5 → 135.7 GB/s) |
| find_tracks | +32% (43.8 → 57.7 GB/s) |

**Hardware Note:** NCU profiled on RTX 2080 Ti; throughput benchmarks on V100

**Script References:**
- Lines 140-152: Kernel-level analysis explanation
- Q11-Q13: NCU-related backup questions

---

## 3. Memory Optimization Documentation

### 3.1 CONSTANT_MEMORY_OPTIMIZATION_RESULTS.md
- **Size:** 12.0 KB
- **Role:** Constant memory optimization validation

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 7 | +3.7% constant memory improvement, combined +6.2% |

**Key Data Points:**
- Improvement: +3.7% throughput (constant memory alone)
- Combined with texture: +6.2% total
- Cycles saved: ~3,200 per measurement candidate
- Implementation: `__constant__` arrays for seed/measurement offsets
- Binary search: O(log N) with zero-latency cache

**Technical Details:**
- Global memory latency: ~400 cycles per access
- Constant memory latency: ~0 cycles (when cached)
- MAX_BATCH_SIZE: 1024 (supports up to 8191 events theoretically)

**Script References:**
- Lines 72-74: Constant memory explanation
- Q2: Event boundary enforcement explanation

---

### 3.2 TEXTURE_MEMORY_VALIDATION_RESULTS.md
- **Size:** 17.9 KB
- **Role:** Texture memory optimization for B-field

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 7 | +2.7% texture memory improvement |

**Key Data Points:**
- Improvement: +2.7% faster event processing
- Enables: Realistic inhomogeneous magnetic field at no performance cost
- Backend: `inhom_texture_bfield_backend_t`
- B-field file: 146 MB (odd-bfield.cvf)

**Physics Impact:**
- Allows 3D spatial lookups for magnetic field
- Hardware-accelerated caching for RK stepper

**Script References:**
- Lines 75-76: Texture memory explanation

---

### 3.3 OPTIMIZATION_SUMMARY.md
- **Size:** 10.1 KB
- **Role:** Consolidated optimization results

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 7 | Phase summary, efficiency metrics |

**Key Data Points:**
- Phase 1 (sync elimination): 1.55x speedup
- Phase 2 (compaction): -18.8% (rejected)
- Overall efficiency: 98% of theoretical maximum

**Note:** Thrust::seq +7.8% is documented in `batching_report.md` (Lines 162-163, 184), not this file.

**Script References:**
- Lines 77-78: Thrust::seq replacement explanation (data from batching_report.md)

---

## 4. Batch Size & Architecture Documentation

### 4.1 BATCH_SIZE_OPTIMIZATION_RESULTS.md
- **Size:** 4.9 KB
- **Role:** Batch size study results (PARTIAL - earlier study)

**⚠️ Note:** This file contains an **earlier batch size study** that found N=24 optimal (27.67 ev/s). The **final N=48 optimal** (30.12 ev/s) data is in `batching_report.md` (Lines 55-62).

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 12 | Batch size methodology (final N=48 data from batching_report.md) |

**Key Data Points (from this file - earlier study):**
| Batch Size | Throughput (ev/s) | vs Baseline |
|------------|-------------------|-------------|
| 1 (baseline) | 15.61 | - |
| 14 | 26.35 | +69% |
| 24 | 27.67 | +77% |

**Final Optimized Data (from batching_report.md):**
| Batch Size | Throughput (ev/s) | vs Baseline |
|------------|-------------------|-------------|
| **48** | **30.12** | **+93%** |

**Script References:**
- Lines 127-136: Batch size optimization explanation
- Q1: Why batch size 48 is optimal

---

### 4.2 BATCH_SIZE_SCALING_ANALYSIS.md
- **Size:** 13.2 KB
- **Role:** Scaling analysis and bug documentation

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 12 | Memory alignment bugs at N=6,8,12 |

**Key Data Points:**
- Pre-existing bugs: N=6, 8, 12 cause vecmem memory alignment errors
- N=64+: Out-of-memory issues depending on dataset
- Workaround: Avoid problematic batch sizes in production

**Script References:**
- Lines 133-134: Bug documentation
- Q10: vecmem memory alignment bugs explanation

---

### 4.3 MULTI_EVENT_BATCHING_ARCHITECTURE.md
- **Size:** 17.5 KB
- **Role:** Batching architecture design document

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 6 | ~260 lines batched CKF wrapper, offset-based indexing, batch_metadata structure |

**Key Data Points:**
- Architecture: Offset-based indexing (not event_id per element)
- Data structure: `batch_metadata` with seed_offsets and meas_offsets
- Event lookup: Binary search O(log N) via `get_event_id()`
- Implementation size: ~260 lines for batched CKF wrapper

**Example:**
```
Events: A(50 seeds), B(30 seeds), C(40 seeds)
Batched buffer: [A's 50 | B's 30 | C's 40] = 120 total
Offset array: [0, 50, 80, 120]
Lookup: seed index 75 → event B (offsets[1] ≤ 75 < offsets[2])
```

**Script References:**
- Lines 60-66: Multi-event batching architecture explanation
- Q2: Event boundary enforcement

---

## 5. Bug Fix & Validation Documentation

### 5.1 CHI2_THRESHOLD_FIX_RESULTS.md
- **Size:** 4.4 KB
- **Role:** Chi-squared bug fix documentation

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 9 | 2.75x track discrepancy, exact code fix, backward compatibility |

**Key Data Points:**
- Problem: GPU found 2.75x more tracks than CPU
- Root cause: Region-dependent chi² defaults (50, 100, 150) vs test config (10)
- Fix: Fallback to `cfg.chi2_max` when region thresholds at defaults
- Result: All 5 CUDA CKF tests pass

**Code Fix Location:**
- File: `device/common/include/traccc/finding/device/impl/find_tracks.ipp`
- Lines: 366-375

**Script References:**
- Lines 95-102: Bug fix explanation
- Q7: Most challenging part of the work

---

### 5.2 FINAL_N4_BATCHING_VALIDATION_RESULTS.md
- **Size:** 22.7 KB
- **Role:** Comprehensive validation results

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 9 | Chi-squared bug discovery context |
| 14 | Validation methodology |

**Key Data Points:**
- Initial failure: 2.75-2.84x track over-reconstruction
- Debug process: Line-by-line CPU/GPU comparison
- Resolution: Backward compatibility fix for chi² thresholds

---

## 6. Negative Results Documentation

### 6.1 phase2_compaction_analysis.md
- **Size:** 7.3 KB
- **Role:** Track compaction failure analysis

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 15 | Track compaction +18.8% overhead, 100% alive in early steps |

**Key Data Points:**
- Result: +18.8% overhead (REJECTED)
- Root cause 1: Steps 1-4 have 100% alive candidates (duplicate removal starts at step 5)
- Root cause 2: Per-step overhead: 6-11ms (allocation + kernels + sync)
- Root cause 3: 90-95% retention even after duplicate removal

**Lesson Learned:** Compaction only benefits when significant dead candidates exist

**Script References:**
- Lines 172-177: Track compaction negative result
- Q3: Why track compaction failed

---

### 6.2 phase3_pipelining_analysis.md
- **Size:** 7.3 KB
- **Role:** Pipelining feasibility analysis

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 15 | 98% efficiency already achieved |
| 17 | Pipelining +2% headroom only |

**Key Data Points:**
- Theoretical opportunity: Hide ~18ms preprocessing → +30% speedup
- Reality: Already at 98% efficiency (1.55x / 1.58x)
- Remaining headroom: Only +2%
- Decision: Document analysis, do not implement

**Script References:**
- Lines 200-201: Future work pipelining note
- Q5: Theoretical maximum speedup

---

### 6.3 SORTING_OPTIMIZATION_REVIEW.md
- **Size:** 36.5 KB
- **Role:** Sorting removal investigation

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 15 | Sorting removal negative result - improves memory coalescing |

**Key Data Points:**
- Hypothesis: Theta-based sorting is wasted work
- Finding: Sorting IMPROVES memory coalescing and cache utilization
- Recommendation: Keep sorting, remove unnecessary synchronizations instead
- Thrust already uses CUB internally (no benefit from direct CUB)

**Script References:**
- Lines 170-171: Sorting removal explanation

---

### 6.4 EXPERT_RECOMMENDATIONS_REVIEW.md
- **Size:** 28.1 KB
- **Role:** Expert guidance analysis

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 15 | Persistent kernel and kernel fusion context |

**Key Data Points:**
- Expert diagnosis: Parallelism scarcity problem, not micro-optimization
- Priority order: Batching → pruning → persistent kernel → hybrid CPU-GPU
- Persistent kernel: Not worthwhile at 98% efficiency
- Kernel fusion: High register pressure reduces occupancy

**Script References:**
- Lines 177-178: Persistent kernel and kernel fusion explanation

---

## 7. Profiling & Analysis Documentation

### 7.1 sync_audit.md
- **Size:** 9.9 KB
- **Role:** Synchronization removal safety proofs

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 8 | Sync removal methodology |
| Backup Q9 | Stream semantics safety proof |

**Key Data Points:**
- Total sync overhead: 90.5% of CUDA API time
- Syncs per event: ~54 synchronizations
- Critical bottleneck: Measurement concatenation loop (N syncs per batch)
- Hidden syncs: `get_size()` calls trigger cudaStreamSynchronize

**Safety Proof:**
- Thrust `par_nosync` provides stream-ordered execution
- CUDA guarantees in-order execution within same stream
- No explicit sync needed after Thrust sort before next kernel

**Script References:**
- Lines 86-89: Synchronization optimization explanation
- Q9: How synchronization removal was proven safe

---

### 7.2 NSYS_N4_BATCHING_PROFILING_ANALYSIS.md
- **Size:** 18.1 KB
- **Role:** Baseline profiling data

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 5 | 43% GPU idle, 15% occupancy baseline metrics |

**Key Data Points:**
- GPU idle time: ~43%
- Baseline occupancy: ~15%
- Problem: Single-event processing severely underutilizes GPU

---

### 7.3 ROOFLINE_PROFILING_RESULTS.md
- **Size:** 10.3 KB
- **Role:** Roofline analysis for theoretical limits

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| Backup Q5 | 42% DRAM utilization, 2.5x theoretical max |

**Key Data Points:**
- DRAM throughput: 42.04%
- SM throughput: 9.29%
- Kernel characterization: Latency-bound (not compute or bandwidth bound)
- Theoretical max with perfect coalescing: ~2.5x current throughput

**Script References:**
- Q5: Theoretical maximum speedup explanation

---

### 7.4 batching_ncu_guide.md
- **Size:** 7.9 KB
- **Role:** NCU profiling methodology

**Slides Supported:**
| Slide | Content Sourced |
|-------|-----------------|
| 13 | NCU profiling methodology, commit hashes |

**Key Data Points:**
- Baseline commit: `5cd477ac`
- Optimized commit: `3ad492b5`
- Hardware: RTX 2080 Ti (CC 7.5, 68 SMs)
- Reproduction commands included

---

## 8. File-to-Slide Mapping Summary

| Slide | Primary Sources | Secondary Sources |
|-------|-----------------|-------------------|
| 3 (Context) | - | batching_report.md |
| 4 (Objectives) | batching_report.md | - |
| 5 (Approach) | batching_report.md | NSYS_N4_BATCHING_PROFILING_ANALYSIS.md |
| 6 (Batching Arch) | MULTI_EVENT_BATCHING_ARCHITECTURE.md | - |
| 7 (Memory Opt) | CONSTANT_MEMORY_OPTIMIZATION_RESULTS.md, TEXTURE_MEMORY_VALIDATION_RESULTS.md | OPTIMIZATION_SUMMARY.md |
| 8 (Sync Opt) | batching_profile_results.md | sync_audit.md |
| 9 (Chi² Bug) | CHI2_THRESHOLD_FIX_RESULTS.md | FINAL_N4_BATCHING_VALIDATION_RESULTS.md |
| 10 (Results) | batching_report.md | - |
| 11 (NSYS) | batching_profile_results.md | - |
| 12 (Batch Size) | BATCH_SIZE_OPTIMIZATION_RESULTS.md | BATCH_SIZE_SCALING_ANALYSIS.md |
| 13 (NCU) | batching_ncu_results.md | batching_ncu_guide.md |
| 14 (Validation) | batching_report.md | FINAL_N4_BATCHING_VALIDATION_RESULTS.md |
| 15 (Negative) | phase2_compaction_analysis.md, SORTING_OPTIMIZATION_REVIEW.md | phase3_pipelining_analysis.md, EXPERT_RECOMMENDATIONS_REVIEW.md |
| 16 (Contributions) | batching_report.md | - |
| 17 (Summary) | batching_report.md | phase3_pipelining_analysis.md |
| Backup Q1-Q13 | Various | ROOFLINE_PROFILING_RESULTS.md, sync_audit.md |

---

## 9. Data Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │      batching_report.md             │
                    │   (PRIMARY AUTHORITATIVE SOURCE)    │
                    │  • 93% improvement                  │
                    │  • 15.61 → 30.12 ev/s              │
                    │  • 1,460 tests                      │
                    │  • 98% efficiency                   │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ NSYS Validation  │    │  NCU Validation  │    │ Memory Opt Docs  │
│                  │    │                  │    │                  │
│ batching_        │    │ batching_        │    │ CONSTANT_MEMORY_ │
│ profile_         │    │ ncu_results.md   │    │ OPTIMIZATION_    │
│ results.md       │    │                  │    │ RESULTS.md       │
│                  │    │ • Occupancy      │    │                  │
│ • 92% sync ↓     │    │ • Throughput     │    │ TEXTURE_MEMORY_  │
│ • 77% launches ↓ │    │ • Grid sizes     │    │ VALIDATION_      │
│ • Kernel counts  │    │                  │    │ RESULTS.md       │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    ┌─────────┐            ┌─────────┐            ┌─────────┐
    │ Slide 8 │            │ Slide 13│            │ Slide 7 │
    │ Slide 11│            │         │            │         │
    └─────────┘            └─────────┘            └─────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     NEGATIVE RESULTS CHAIN                       │
├──────────────────┬──────────────────┬──────────────────┬────────┤
│ phase2_          │ phase3_          │ SORTING_         │ EXPERT_│
│ compaction_      │ pipelining_      │ OPTIMIZATION_    │ RECOM- │
│ analysis.md      │ analysis.md      │ REVIEW.md        │ MENDA- │
│                  │                  │                  │ TIONS_ │
│ +18.8% overhead  │ +2% headroom     │ Sorting helps    │ REVIEW │
│ 100% alive early │ 98% already      │ coalescing       │ .md    │
└────────┬─────────┴────────┬─────────┴────────┬─────────┴───┬────┘
         │                  │                  │             │
         └──────────────────┴──────────────────┴─────────────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │  Slide 15   │
                            │  (Negative  │
                            │   Results)  │
                            └─────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      SUPPORTING DOCS                             │
├──────────────────┬──────────────────┬──────────────────┬────────┤
│ sync_audit.md    │ ROOFLINE_        │ MULTI_EVENT_     │ BATCH_ │
│                  │ PROFILING_       │ BATCHING_        │ SIZE_* │
│ Safety proofs    │ RESULTS.md       │ ARCHITECTURE.md  │ .md    │
│ for sync removal │                  │                  │        │
│                  │ 42% DRAM         │ ~260 lines       │ N=48   │
│ → Slide 8, Q9    │ 2.5x theoretical │ offset arrays    │ optimal│
│                  │                  │                  │        │
│                  │ → Backup Q5      │ → Slide 6        │→Slide12│
└──────────────────┴──────────────────┴──────────────────┴────────┘
```

---

## Appendix: Quick Reference

### Claim Verification Checklist

| Claim in Presentation | Source File | Location in File |
|-----------------------|-------------|------------------|
| 93% throughput improvement | batching_report.md | Line 14 |
| 15.61 → 30.12 ev/s | batching_report.md | Lines 14, 42, 61 |
| 90.5% CUDA API sync time | batching_report.md | Lines 44, 76, 230 |
| 92% sync reduction | batching_profile_results.md | Lines 16, 42 |
| 8,525 → 706 sync calls | batching_profile_results.md | Line 42 |
| 77% kernel launch reduction | batching_profile_results.md | Line 59 |
| 69% memory op reduction | batching_profile_results.md | Line 58 |
| +3.7% constant memory | CONSTANT_MEMORY_OPTIMIZATION_RESULTS.md | Lines 108, 119 |
| +2.7% texture memory | TEXTURE_MEMORY_VALIDATION_RESULTS.md | Line 15 |
| +7.8% thrust replacement | batching_report.md | Lines 162-163, 184 |
| +6.2% combined memory opt | CONSTANT_MEMORY_OPTIMIZATION_RESULTS.md | Line 22 |
| 3,200 cycles saved | CONSTANT_MEMORY_OPTIMIZATION_RESULTS.md | Line 152 |
| fit_forward 15.4%→38.4% | batching_ncu_results.md | Line 25 |
| fit_backward +154% throughput | batching_ncu_results.md | Line 28 |
| 2.75x track discrepancy | CHI2_THRESHOLD_FIX_RESULTS.md | Lines 11, 50 |
| +18.8% compaction overhead | phase2_compaction_analysis.md | Line 51 |
| 98% theoretical efficiency | batching_report.md | Line 255 |
| +2% pipelining headroom | phase3_pipelining_analysis.md | Lines 96-97 |
| 1,460 tests | batching_report.md | Lines 18, 275-280 |
| 68 commits | batching_report.md | Line 16 |
| N=48 optimal | batching_report.md | Lines 53, 61, 330 |
| N=6,8,12 bugs | BATCH_SIZE_SCALING_ANALYSIS.md | Throughout |
| 42% DRAM utilization | ROOFLINE_PROFILING_RESULTS.md | Line 19 |

---

*This reference guide ensures all presentation claims are traceable to source documentation.*
