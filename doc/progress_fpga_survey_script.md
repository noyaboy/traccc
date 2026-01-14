# FPGA Offloading Survey - Oral Presentation Script

**Duration:** ~10-12 minutes
**Date:** January 14, 2026

---

## Slide 1: Title Slide

*[30 seconds]*

Good morning/afternoon. Today I'll present my weekly progress on the FPGA offloading survey for TRACCC. This work evaluates whether we can use a hybrid GPU/FPGA architecture to accelerate particle track reconstruction.

---

## Slide 2: Outline

*[15 seconds]*

I'll cover five main topics: the survey objective and key findings, our precision validation results, GPU profiling analysis, the proposed hybrid architecture, and the critical risks we need to address.

---

## Slide 3: Survey Objective

*[1 minute]*

Our goal is to evaluate hybrid GPU/FPGA execution for TRACCC particle tracking. The target platform is the AMD Alveo V80, which has about 10,800 DSP58 slices with native single-precision floating-point support, 32 gigabytes of HBM2e memory with 819 gigabytes per second bandwidth, and PCIe Gen5 connectivity.

The key question we're trying to answer is: Can an FPGA effectively offload the latency-bound kernels that are bottlenecking GPU performance?

The diagram shows our proposed architecture with the host CPU orchestrating between the GPU handling double-precision operations and the FPGA handling single-precision DSP workloads.

---

## Slide 4: Key Finding - FPGA Offload Candidate

*[1.5 minutes]*

This is our primary finding: The `propagate_to_next_surface` kernel consumes 63 percent of total GPU time and has 93 percent warp stall cycles. This makes it an ideal FPGA offload candidate.

Why is it latency-bound? The RK4 integration involves sequential multiply-accumulate operations. The B-field lookup requires dependent memory accesses. The step count varies from 1 to 34 steps per track with no reliable predictor. This causes 62 percent of GPU cycles to be wasted due to warp divergence.

Why is FPGA ideal for this? The DSP58 blocks provide native FP32 MAC chains that can be deeply pipelined. Unlike the GPU's SIMT model, FPGA pipelines don't require warp synchronization. We can have dedicated pipelines per track, so variable step counts don't waste resources.

The bar chart shows the GPU time breakdown: propagate takes 63 percent, other FPGA-suitable kernels take 17 percent, and only about 20 percent must stay on GPU.

---

## Slide 5: FP32 vs FP64 - No Physics Difference

*[1.5 minutes]*

A critical question for FPGA offloading is whether single-precision is sufficient. We validated this with 10,000 muon tracks.

The key result: chi-squared per degree of freedom is identical at 1.0042 plus or minus 0.0020 for both FP32 and FP64. All five pull distribution sigmas differ by less than 0.003 - well within statistical fluctuations.

Looking at the table for 10 GeV muons: d0 sigma is 0.987 for both precisions, phi is 0.980 for both, and so on. The maximum difference in any parameter is just 0.002 for theta.

What does this mean for FPGA? The DSP58 native FP32 operations will produce physics-identical results to GPU FP64. No precision conversion is needed at the GPU-FPGA interface. This removes precision concerns as a blocker for FPGA development.

---

## Slide 6: GPU Kernel Time Breakdown

*[1.5 minutes]*

Let me show the detailed profiling results from Nsys analysis. The table breaks down GPU kernel time.

Propagate-to-next-surface dominates at 63 percent - this is our primary FPGA target. Build_tracks at 14.6 percent must stay on GPU due to irregular memory access patterns. The remaining kernels - apply_interaction, find_tracks, make_doublets, select_seeds, form_spacepoints, and CCL clustering - are all FPGA-suitable.

In total, about 80 percent of GPU work could potentially run on FPGA.

The propagation kernel has severe bottlenecks: 93 percent warp stall cycles, register pressure of 96 to 128 registers per thread, and L1 cache hit rate of only 46 to 54 percent. This is a classic latency-bound kernel where the SIMT execution model is penalizing us heavily.

---

## Slide 7: Proposed GPU/FPGA Partitioning

*[1.5 minutes]*

Here's the proposed hybrid architecture.

The Alveo V80 handles single-precision workloads: RK4 propagation which is 63 percent of current GPU time, B-field polynomial evaluation, chi-squared computation and threshold checking, measurement matching, CCL clustering, and seeding. With 10,848 DSP58 slices, we estimate about 98 parallel track pipelines, using HBM for geometry storage, all within 190 watts.

The GPU retains double-precision operations: covariance matrix updates which have multiplicative error accumulation, 6-by-6 matrix inversions which are condition-number sensitive, chi-squared accumulation over multiple surfaces, track deduplication which has irregular memory patterns, and final track fitting.

Expected benefits: We eliminate the 93 percent warp stalls on FPGA, potentially achieve 1.5 to 2 times throughput, with total system power around 490 watts.

---

## Slide 8: Critical Blockers and Risks

*[1.5 minutes]*

Now for the risks, and one critical blocker.

The critical blocker is per-step synchronization overhead. The Combinatorial Kalman Filter requires GPU-to-FPGA sync at each of 15 surfaces. If this overhead exceeds 200 microseconds per step, FPGA offloading is not viable. This must be validated before any FPGA development.

Our PCIe latency analysis shows: if we transfer only track parameters, overhead is 2.7 percent - acceptable. If we transfer full track state, overhead jumps to 13 percent - concerning. If we also transfer Jacobians, overhead reaches 18 percent - prohibitive.

Other risks include: DSP58 resource estimates are theoretical, XRT integration complexity is unknown, there's a Vitis HLS learning curve, and the B-field grid at 139 megabytes is too large for on-chip BRAM.

Mitigations: Cache covariance matrices in HBM2e, transfer only 24-byte parameter vectors, use V80 HBM for B-field storage, and start with just the propagation kernel for the prototype.

---

## Slide 9: Implementation Roadmap

*[1 minute]*

Here's our implementation roadmap.

Phase 1, which is current: We've completed FP32 versus FP64 precision validation, NCU and Nsys profiling analysis, and FPGA suitability assessment. Still pending is the sync barrier overhead measurement - this is the critical go/no-go decision.

Phase 2 is the prototype: RK4 propagation kernel in Vitis HLS, single-precision validation against GPU baseline, and XRT host API integration.

Phase 3 is full integration: GPU-to-FPGA data path, full CKF loop with hybrid execution, and performance benchmarking.

The immediate next step is to measure actual per-step synchronization overhead using GPU-to-Host transfers as a proxy for GPU-to-FPGA. This determines whether we proceed with FPGA development.

---

## Slide 10: Summary

*[1 minute]*

To summarize our key findings:

One: The propagate-to-next-surface kernel consumes 63 percent of GPU time with 93 percent warp stalls. This is an ideal FPGA candidate.

Two: FP32 is physics-equivalent to FP64. DSP58 native single-precision is safe for FPGA offloading.

Three: About 80 percent of GPU work is potentially FPGA-suitable.

Four: The V80 can theoretically support about 98 parallel track pipelines.

Open questions remain: What is the actual per-step sync overhead? What is the real DSP58 utilization? What are the XRT integration challenges?

Our recommendation is to proceed with sync overhead validation before committing to any FPGA development. If the overhead is acceptable, we have a promising path forward.

The full analysis is documented in survey-fpga.md, which is over 3000 lines of detailed technical analysis.

---

## Q&A Notes

**Anticipated Questions:**

1. **Why V80 specifically?**
   - Native FP32 DSP58 (no soft-logic overhead)
   - 32GB HBM2e for geometry/B-field
   - PCIe Gen5 for GPU communication
   - Available in our lab

2. **Why not just optimize GPU code?**
   - Already tried MBF optimization (+93% throughput)
   - Propagation is fundamentally latency-bound
   - SIMT model doesn't help sequential dependencies

3. **What about power consumption?**
   - V80: 190W TDP
   - Current GPU: ~300W
   - Hybrid: ~490W total, but potentially 1.5-2x throughput

4. **Timeline for prototype?**
   - Depends on sync barrier validation results
   - If viable: 2-3 months for basic RK4 kernel in Vitis HLS

---

*End of script*
