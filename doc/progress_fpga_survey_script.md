# FPGA Offloading Survey - Oral Presentation Script

**Duration:** ~8-10 minutes
**Date:** January 14, 2026

---

## Slide 1: Title Slide

*[20 seconds]*

This is my weekly progress report on the FPGA offloading survey for TRACCC.

---

## Slide 2: Completed Work Overview

*[30 seconds]*

Here's an overview of what I've completed this week. I'll go through each item and explain what it means for our FPGA offloading decision.

The main output is a comprehensive survey document - over 3000 lines of analysis in survey-fpga.md.

---

## Slide 3: FP32 vs FP64 Precision Validation

*[1 minute]*

**What I did:**
I built TRACCC with both single and double precision, ran Kalman fitter tests on 10,000 muon tracks, and compared the pull distributions and chi-squared values.

**What it means:**
The result is that FP32 and FP64 produce identical physics results. The pull sigma differs by less than 0.003, which is within statistical noise. Chi-squared per NDF is 1.0042 for both precisions.

The implication is that DSP58 native FP32 is safe for FPGA offloading - we won't lose any physics precision.

---

## Slide 4: NCU Profiling Analysis

*[1 minute]*

**What I did:**
I profiled the propagate-to-next-surface kernel with NVIDIA Nsight Compute, analyzing register pressure, cache hit rates, and warp stall reasons.

**What it means:**
93% of cycles are warp stalls - the kernel is latency-bound, not compute-bound. Register pressure is 96-128 registers per thread, which limits occupancy. L1 cache hit rate is only 46-54%, indicating unpredictable memory access.

The implication is that GPU's SIMT model is fundamentally inefficient for this workload. FPGA's pipelined execution can eliminate these stalls entirely.

---

## Slide 5: Nsys Profiling Analysis

*[1 minute]*

**What I did:**
I ran the full pipeline with Nsight Systems to measure how GPU time is distributed across kernels.

**What it means:**
propagate-to-next-surface takes 63% of GPU time - this is our primary FPGA target. build_tracks takes 14.6% but must stay on GPU due to pointer chasing. The remaining 22% is mostly FPGA-suitable.

In total, about 80% of GPU work could potentially move to FPGA.

---

## Slide 6: FPGA Suitability Assessment

*[1 minute]*

**What I did:**
I analyzed each kernel's computational pattern - MAC chains, memory access patterns, branching behavior - and mapped them to DSP58 capabilities.

**What it means:**
Good for FPGA: RK4 propagation with its MAC chains, B-field polynomial evaluation using Horner's method, matrix-vector operations that map well to systolic arrays, and CCL clustering which is a well-known FPGA pattern.

Keep on GPU: Track deduplication with irregular branching, build_tracks with pointer chasing, and 6x6 matrix inversion with complex control flow.

This gives us a clear partitioning strategy.

---

## Slide 7: Alveo V80 Resource Estimation

*[45 seconds]*

**What I did:**
I estimated DSP58 usage per track pipeline at about 110 DSP58, then calculated how many parallel pipelines we can fit on the V80.

**What it means:**
With 10,848 DSP58 slices on the V80, we can theoretically support about 98 parallel track pipelines. The 32GB HBM is sufficient for geometry and B-field storage.

This is a theoretical estimate - actual numbers require Vitis HLS synthesis to confirm.

---

## Slide 8: PCIe Latency Analysis

*[1 minute]*

**What I did:**
I measured GPU-to-Host transfer latency as a proxy for GPU-to-FPGA communication, testing different data sizes.

**What it means:**
If we transfer only the 24-byte parameter vector per track, overhead is 2.7% - acceptable. If we transfer full track state at 176 bytes, overhead jumps to 13% - concerning. Adding Jacobians pushes it to 18% - prohibitive.

The implication is we must minimize data transfer. We should cache covariance matrices on FPGA HBM and only transfer the small parameter vectors.

---

## Slide 9: Next Step - Validate Per-Step Sync Barrier

*[1.5 minutes]*

This is the critical blocker we need to address next.

The CKF algorithm requires GPU-to-FPGA synchronization at each of 15 surfaces. If this sync overhead exceeds 200 microseconds per step, FPGA offloading is simply not viable.

**What needs to be done:**
1. Instrument the CKF loop to measure actual per-step synchronization time
2. Test with GPU-to-Host sync as a proxy since we don't have FPGA hardware yet
3. Calculate total overhead compared to the 23ms baseline

**Decision criteria:**
- If overhead is less than 5%, we proceed with FPGA development
- If overhead is 5-15%, we need to consider async overlap strategies
- If overhead exceeds 15%, FPGA offloading is not viable for CKF

This is the go/no-go decision point before we invest in any FPGA development.

---

## Slide 10: Summary

*[45 seconds]*

To summarize what's completed:
1. FP32 equals FP64 for physics - DSP58 native FP32 is safe
2. 63% of GPU time is in one kernel with 93% stalls - ideal FPGA candidate
3. 80% of work is FPGA-suitable - clear partitioning strategy
4. V80 can support about 98 parallel pipelines - sufficient resources
5. PCIe overhead is 2.7% for params-only transfer - strategy defined

Next step is to measure the per-step sync barrier overhead. This is the critical go/no-go decision. If it's viable, we'll prototype the RK4 kernel in Vitis HLS.

The full documentation is in doc/survey-fpga.md.

---

## Q&A Notes

**Anticipated Questions:**

1. **Why is sync barrier so critical?**
   - CKF is iterative: propagate -> match -> update -> repeat
   - Each step needs results from previous step
   - Can't pipeline across steps without sync

2. **Can we batch multiple events to hide latency?**
   - Already doing multi-event batching on GPU (+93% throughput)
   - FPGA would need same strategy
   - Doesn't eliminate per-step sync within an event

3. **What if sync overhead is too high?**
   - Could offload only non-CKF kernels (seeding, clustering)
   - Or explore coarser-grained offloading (full propagation batch)
   - Worst case: FPGA not viable, continue GPU-only optimization

---

*End of script*
