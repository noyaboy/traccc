# Presentation Bundle: Conditional Jacobian Aggregation

**Created:** 2026-01-06
**Purpose:** Self-contained collection of all files necessary for the presentation

---

## Table of Contents

1. [File Inventory](#1-file-inventory)
2. [File Dependency Analysis](#2-file-dependency-analysis)
3. [Data Source Mapping](#3-data-source-mapping)
4. [Key Numbers Traceability](#4-key-numbers-traceability)
5. [Visual Dependency Graph](#5-visual-dependency-graph)
6. [Usage Guide](#6-usage-guide)

---

## 1. File Inventory

**Total: 26 files** (including this README.md)

### 1.1 Presentation Output (4 files)

| File | Description |
|------|-------------|
| `presentation/conditional_jacobian_aggregation_slides.tex` | LaTeX Beamer slides (13 slides + backup) |
| `presentation/conditional_jacobian_aggregation_slides.pdf` | Compiled PDF output |
| `presentation/conditional_jacobian_aggregation_oral_script.md` | Speaker notes for each slide + backup Q&A |
| `presentation/visualization_fix.md` | Design guidance for slide visuals |

### 1.2 Documentation (7 files)

| File | Description |
|------|-------------|
| `documentation/conditional_jacobian_transport_report.md` | Benchmark results (throughput, latency, test counts) |
| `documentation/conditional_jacobian_transport_ncu_results.md` | NCU profiling metrics (registers, occupancy, duration) |
| `documentation/conditional_jacobian_transport_plan.md` | Implementation plan with architecture details |
| `documentation/conditional_jacobian_transport_profile_report.md` | Root cause analysis, optimization mechanism |
| `documentation/conditional_jacobian_transport_profile_plan.md` | Profiling methodology |
| `documentation/conditional_jacobian_transport_ncu_guide.md` | NCU profiling commands |
| `documentation/register_pressure_survey.md` | Background on register pressure problem |

### 1.3 Code Files (5 files)

| File | Original Location |
|------|-------------------|
| `code/bound_updater.hpp` | `core/include/traccc/finding/actors/bound_updater.hpp` |
| `code/propagate_to_next_surface.ipp` | `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp` |
| `code/combinatorial_kalman_filter_types.hpp` | `core/include/traccc/finding/details/combinatorial_kalman_filter_types.hpp` |
| `code/finding_config.hpp` | `core/include/traccc/finding/finding_config.hpp` |
| `code/build_tracks.ipp` | `device/common/include/traccc/finding/device/impl/build_tracks.ipp` |

### 1.4 Build System Files (5 files)

| File | Original Location |
|------|-------------------|
| `build_system/parameter_transporter.hpp` | `build/_deps/detray-src/core/include/detray/propagator/actors/parameter_transporter.hpp` |
| `build_system/track_parametrization.hpp` | `build/_deps/detray-src/core/include/detray/definitions/track_parametrization.hpp` |
| `build_system/CMakeLists.txt` | `device/cuda/CMakeLists.txt` |
| `build_system/gen_kernel_specialization.py` | `codegen/kernel_specialization/gen_kernel_specialization.py` |
| `build_system/propagate_to_next_surface.cu.template` | `device/cuda/src/finding/kernels/specializations/propagate_to_next_surface.cu.template` |

### 1.5 Raw Profiling Data (4 files)

| File | Description |
|------|-------------|
| `raw_data/baseline_ncu_full.txt` | Full NCU output for baseline (commit a48cc783) |
| `raw_data/optimization_ncu_full.txt` | Full NCU output for optimization (commit 25894cca) |
| `raw_data/baseline_nsys.nsys-rep` | Nsight Systems trace for baseline |
| `raw_data/optimization_nsys.nsys-rep` | Nsight Systems trace for optimization |

---

## 2. File Dependency Analysis

### 2.1 Tier Structure

```
Tier 1: Primary Outputs
    └── slides.tex, oral_script.md, slides.pdf

Tier 2: Data Sources (directly referenced in slides/script)
    └── report.md, ncu_results.md, plan.md, profile_report.md, register_pressure_survey.md

Tier 3: Code Evidence (shown or referenced in slides)
    └── bound_updater.hpp, parameter_transporter.hpp, propagate_to_next_surface.ipp,
        combinatorial_kalman_filter_types.hpp, finding_config.hpp

Tier 4: Supporting/Reference (methodology, design)
    └── profile_plan.md, ncu_guide.md, visualization_fix.md, CMakeLists.txt,
        gen_kernel_specialization.py, *.cu.template, track_parametrization.hpp

Tier 5: Raw Data (ultimate source of metrics)
    └── *.txt, *.nsys-rep
```

### 2.2 Slide-to-File Mapping

| Slide | Title | Primary Data Sources |
|-------|-------|---------------------|
| 1 | Title | - |
| 2 | Outline | - |
| 3 | Register Pressure in GPU Track Finding | `register_pressure_survey.md`, `plan.md` §1 |
| 4 | Opportunity: Conditional Jacobian | `profile_report.md` §9.4 |
| 5 | Solution: Two Actor Chains | `plan.md` §2-3, `combinatorial_kalman_filter_types.hpp` |
| 6 | Key Code: Actor State Comparison | `bound_updater.hpp`, `parameter_transporter.hpp` |
| 7 | Kernel Dispatch Logic | `propagate_to_next_surface.ipp`, `plan.md` §4.3 |
| 8 | Benchmark Results | `report.md` §3-4 |
| 9 | NCU Profiling: Register Reduction Confirmed | `ncu_results.md` §2-6 |
| 10 | Optimization Mechanism Summary | `profile_report.md` §9, `ncu_results.md` §8 |
| 11 | Conclusion | `ncu_results.md`, `report.md` |
| 12 | Summary | All documentation files |
| 13 | Questions | - |
| Backup | FAQ | `profile_report.md`, `build_tracks.ipp` |

---

## 3. Data Source Mapping

### 3.1 `documentation/conditional_jacobian_transport_report.md`

**Purpose:** Primary benchmark results

| Data | Value | Used In |
|------|-------|---------|
| Baseline throughput | 36.57 events/s | Slide 8 |
| Optimized throughput | 43.27 events/s | Slide 8 |
| Throughput improvement | +18.3% | Slides 8, 10, 12, 13 |
| Baseline latency | 27.34 ms | Slide 8 |
| Optimized latency | 23.11 ms | Slide 8 |
| Latency reduction | -15.5% | Slide 8 |
| Test results | 710/710 PASSED | Slides 8, 12, 13 |
| Baseline commit | a48cc783 | Oral script |
| Optimization commit | 25894cca | Oral script |
| Dataset | odd/geant4_ttbar_mu200 | Slide 8 |
| Hardware | Tesla V100-SXM2-32GB | Slide 8 |

### 3.2 `documentation/conditional_jacobian_transport_ncu_results.md`

**Purpose:** NCU profiling metrics

| Data | Baseline | Optimized | Used In |
|------|----------|-----------|---------|
| Registers (sm_75) | 128 | 96 (-25%) | Slides 9, 11, 12 |
| Theoretical occupancy | 50% | 62.5% (+12.5pp) | Slide 9 |
| Achieved occupancy | 39.3% | 48.6% (+9.3pp) | Slides 9, 10, 12 |
| Block limit/SM | 4 | 5 | Slide 9 |
| Kernel duration | 3.99 ms | 3.61 ms (-9.5%) | Slide 9 |
| Executed instructions | 82.3M | 78.5M (-4.6%) | Slides 9, 10 |
| Memory throughput | 218 GB/s | 245 GB/s (+12%) | Slide 9 |
| Profiler | ncu 2024.1.1 | - | Slide 9 |
| Profiling hardware | RTX 2080 Ti (sm_75) | - | Slide 9 |

### 3.3 `documentation/register_pressure_survey.md`

**Purpose:** Background on register pressure problem

| Data | Value | Used In |
|------|-------|---------|
| V100 100% occupancy limit | ≤32 registers/thread | Slide 3 |
| Current kernel registers | 128-203 | Slide 3 |
| Current occupancy | 10-25% | Slide 3 |
| 7 actors analysis | s0-s6 breakdown | Slide 5, plan.md |
| Jacobian register estimate | ~64 registers | Slide 3 |

### 3.4 `documentation/conditional_jacobian_transport_plan.md`

**Purpose:** Implementation architecture

| Data | Value | Used In |
|------|-------|---------|
| Register budget breakdown | 6+21+36+65=128 | Slide 3 |
| Actor chain definition | 7 actors (s0-s6) | Slides 5, 6 |
| Kernel specializations | 18 (3×3×2) | Slide 7 |
| Type trait | `has_jacobian_transport_v` | Slide 7 |
| Two actor chains | `ckf_actor_chain_t`, `ckf_actor_chain_no_mbf_t` | Slide 5 |

### 3.5 `documentation/conditional_jacobian_transport_profile_report.md`

**Purpose:** Root cause analysis and optimization mechanism

| Data | Value | Used In |
|------|-------|---------|
| FLOPs per surface | ~216 (6×6 matrix mult) | Slides 4, 10 |
| Memory per surface | ~288 bytes | Slides 4, 10 |
| FLOPs per track (15 surfaces) | ~3,240 | Slide 10 |
| Memory per track (15 surfaces) | ~4,320 bytes | Slide 10 |
| Dual mechanism | Register reduction + Skipped aggregation | Slides 10, 11 |
| build_tracks improvement | -86.3% (MBF config effect) | Backup FAQ |

---

## 4. Key Numbers Traceability

### 4.1 Performance Metrics

| Number | Slide Location | Script Location | Source File | Source Location |
|--------|----------------|-----------------|-------------|-----------------|
| +18.3% throughput | 8, 10, 12, 13 | Lines 98, 128, 148 | `report.md` | §3, line 35 |
| 36.57 events/s | 8 | Line 98 | `report.md` | §3, line 34 |
| 43.27 events/s | 8 | Line 98 | `report.md` | §3, line 35 |
| -15.5% latency | 8 | Line 100 | `report.md` | §3, calculated |
| 710/710 tests | 8 | Line 102 | `report.md` | §2, line 26 |

### 4.2 NCU Profiling Metrics

| Number | Slide Location | Script Location | Source File | Source Location |
|--------|----------------|-----------------|-------------|-----------------|
| 128 → 96 registers | 9 | Line 111 | `ncu_results.md` | §2, line 28 |
| -25% register reduction | 9, 12 | Line 111 | `ncu_results.md` | §2, line 28 |
| 39.3% → 48.6% occupancy | 9 | Line 112 | `ncu_results.md` | §3, line 43-44 |
| +9.3pp occupancy | 10, 12 | Line 124 | `ncu_results.md` | §3, line 44 |
| 50% → 62.5% theoretical | 9 | Line 112 | `ncu_results.md` | §3, line 42 |
| -9.5% kernel duration | 9 | Line 114 | `ncu_results.md` | §5, line 71 |
| -4.6% instructions | 9, 10 | Line 114, 127 | `ncu_results.md` | §5, line 74 |
| +12% memory throughput | 9 | Line 115 | `ncu_results.md` | §4, line 56 |

### 4.3 Register Budget

| Number | Slide Location | Script Location | Source File | Source Location |
|--------|----------------|-----------------|-------------|-----------------|
| 6 (bound params) | 3 | Line 26 | `plan.md` | §1, line 46 |
| 21 (covariance) | 3 | Line 26 | `plan.md` | §1, line 47 |
| ~36 (Jacobian 6×6) | 3 | Line 27 | `plan.md` | §1, line 48 |
| ~65 (Other) | 3 | - | `plan.md` | §1, calculated |
| 128 total | 3 | Line 25 | `plan.md` | §1, line 54 |
| ≤32 for 100% occupancy | 3 | Line 24 | `survey.md` | §1.3, line 58 |

### 4.4 Optimization Mechanism

| Number | Slide Location | Script Location | Source File | Source Location |
|--------|----------------|-----------------|-------------|-----------------|
| ~216 FLOPs/surface | 4, 10 | Line 42, 130 | `profile_report.md` | §9.4 |
| ~288 bytes/surface | 4, 10 | Line 42, 130 | `profile_report.md` | §9.4 |
| ~3,240 FLOPs/track | 10 | Line 130 | `profile_report.md` | §9.5 |
| ~4,320 bytes/track | 10 | Line 130 | `profile_report.md` | §9.5 |
| 15 surfaces/track | - | Line 130 | `profile_report.md` | §9.5 |
| 18 kernel specs | 7 | Line 86 | `plan.md` | §4.3 |
| 3×3×2 formula | 7 | Line 86 | `plan.md` | §4.3 |

---

## 5. Visual Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION OUTPUTS                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  slides.tex     │  │ oral_script.md  │  │  slides.pdf     │              │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────┘              │
└───────────┼─────────────────────┼───────────────────────────────────────────┘
            │                     │
            └──────────┬──────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENTATION (Tier 2)                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Slides 3-4: Problem Context                                          │    │
│  │   • register_pressure_survey.md (V100 limits, 7 actors, occupancy)   │    │
│  │   • plan.md §1 (register budget breakdown)                           │    │
│  │   • profile_report.md §9.4 (FLOPs/bytes per surface)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Slides 5-7: Implementation                                           │    │
│  │   • plan.md §2-4 (actor chains, dispatch, kernel specs)              │    │
│  │   • profile_report.md §9.9 (mechanism explanation)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Slides 8-10: Results                                                 │    │
│  │   • report.md §3-4 (throughput, latency, tests)                      │    │
│  │   • ncu_results.md §2-6 (registers, occupancy, duration)             │    │
│  │   • profile_report.md §9.5 (per-track savings)                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Slides 11-13: Conclusion                                             │    │
│  │   • All above files (summary of key metrics)                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CODE EVIDENCE (Tier 3)                               │
│                                                                              │
│  ┌───────────────────────────┐  ┌───────────────────────────┐               │
│  │ Slide 6: Actor Comparison │  │ Slide 7: Dispatch Logic   │               │
│  │  • bound_updater.hpp      │  │  • propagate_to_next_     │               │
│  │    - struct state {}      │  │    surface.ipp            │               │
│  │    - line 70 (empty)      │  │    - if constexpr         │               │
│  │  • parameter_transporter  │  │    - line 89              │               │
│  │    .hpp                   │  │  • combinatorial_kalman_  │               │
│  │    - _full_jacobian_ptr   │  │    filter_types.hpp       │               │
│  │    - line 54              │  │    - has_jacobian_        │               │
│  └───────────────────────────┘  │      transport_v          │               │
│                                 └───────────────────────────┘               │
│                                                                              │
│  ┌───────────────────────────┐  ┌───────────────────────────┐               │
│  │ Slide 5: Configuration    │  │ Backup: build_tracks      │               │
│  │  • finding_config.hpp     │  │  • build_tracks.ipp       │               │
│  │    - run_mbf_smoother     │  │    - MBF code paths       │               │
│  │    - line 47              │  │                           │               │
│  └───────────────────────────┘  └───────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BUILD SYSTEM (Tier 4)                                   │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │ 18 Kernel Specializations (3 detectors × 3 B-fields × 2 MBF)      │      │
│  │  • CMakeLists.txt - generation logic                              │      │
│  │  • gen_kernel_specialization.py - Python template substitution    │      │
│  │  • propagate_to_next_surface.cu.template - kernel template        │      │
│  │  • track_parametrization.hpp - bound_matrix (6×6) definition      │      │
│  └───────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RAW DATA (Tier 5)                                       │
│                                                                              │
│  ┌───────────────────────────┐  ┌───────────────────────────┐               │
│  │ NCU Profiling             │  │ Nsight Systems            │               │
│  │  • baseline_ncu_full.txt  │  │  • baseline_nsys.nsys-rep │               │
│  │  • optimization_ncu_      │  │  • optimization_nsys.     │               │
│  │    full.txt               │  │    nsys-rep               │               │
│  └───────────────────────────┘  └───────────────────────────┘               │
│                                                                              │
│  Source of all metrics in ncu_results.md                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Usage Guide

### 6.1 Compiling the Slides

```bash
cd presentation_bundle/presentation
LD_PRELOAD=/path/to/libstdc++.so.6 tectonic conditional_jacobian_aggregation_slides.tex
```

Or with standard pdflatex:
```bash
pdflatex conditional_jacobian_aggregation_slides.tex
```

### 6.2 Modifying Content

1. **Update metrics:** Edit the source documentation file, then update slides.tex and oral_script.md
2. **Change visuals:** Edit TikZ/pgfplots code directly in slides.tex
3. **Add slides:** Follow existing slide structure, add corresponding oral script section

### 6.3 Verifying Data Consistency

Use this README to trace any number back to its source:
1. Find the number in "Key Numbers Traceability" section
2. Check the source file and location
3. Verify the value matches in slides.tex and oral_script.md

### 6.4 Re-generating Raw Data

```bash
# NCU profiling (baseline)
ncu --set full --print-summary per-kernel ./traccc_throughput_st_cuda \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --input-directory=odd/geant4_ttbar_mu200/ \
    --processed-events=1 --cpu-threads=1 > baseline_ncu_full.txt

# NCU profiling (optimization)
# Same command after switching to optimization commit
```

See `documentation/conditional_jacobian_transport_ncu_guide.md` for detailed commands.

---

## Appendix: Commit References

| Commit | Description | Branch |
|--------|-------------|--------|
| `a48cc783` | Baseline (MBF cleanup) | main |
| `25894cca` | Optimization (Conditional Jacobian Aggregation) | feature/register-presure |

---

*This bundle is self-contained and can be used to reproduce, verify, or modify the presentation.*
