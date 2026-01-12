# Conditional Jacobian Aggregation - Oral Script

**Duration:** ~15-20 minutes
**Slides:** `conditional_jacobian_aggregation_slides.tex`

---

## Slide 1: Title Slide

> "Good [morning/afternoon]. Today I'll be presenting our optimization work on Conditional Jacobian Aggregation for the traccc GPU track reconstruction framework. This work achieved an 18.3% throughput improvement through algorithmic refinement of the propagation kernel."

---

## Slide 2: Outline

> "I'll structure this presentation into five parts: First, I'll explain the problem context - why register pressure matters for GPU performance and the opportunity we identified. Then I'll describe our implementation approach. Next, I'll show the performance results from benchmarking and detailed kernel profiling, including our methodology for isolating the true improvement. I'll share some key lessons learned, and finally conclude with a summary."

---

## Slide 3: Register Pressure in GPU Track Finding

> "Let's start with the problem. The `propagate_to_next_surface` kernel is the most compute-intensive part of the Combinatorial Kalman Filter, taking about 63% of total GPU time.

> The challenge is that this kernel achieves only 10 to 25 percent GPU occupancy. On an NVIDIA V100, we need 32 registers or fewer per thread to achieve 100% occupancy. But our kernel uses 128 registers per thread.

> Looking at the register budget breakdown, we can see where these registers go: 6 for bound parameters, 21 for the covariance matrix, 12 for Runge-Kutta derivatives, and so on. But notice this item: approximately 36 registers for the 6-by-6 Jacobian matrix.

> The key insight is that this Jacobian is **only needed when the Multi-Branch Fit smoother is enabled**. When MBF is disabled, we're wasting significant register space."

---

## Slide 4: Opportunity: Conditional Jacobian

> "Let me explain what happens with the Jacobian during track propagation.

> As a track propagates through the detector, it encounters multiple surfaces - S1, S2, S3, and so on. At each surface, the algorithm computes a 6-by-6 Jacobian matrix representing how small changes in track parameters propagate.

> When MBF is enabled, these Jacobians are accumulated through matrix multiplication: J1, then J2 times J1, then J3 times J2 times J1, and so forth. This accumulated Jacobian is used later for backward smoothing.

> But here's the problem: when MBF is disabled, we still compute and accumulate these Jacobians - and then we throw them away!

> At each surface, we're wasting approximately 216 floating-point operations for the 6-by-6 matrix multiplication, plus 288 bytes of global memory traffic for reading and writing the accumulated Jacobian.

> This is wasted work that we can eliminate."

---

## Slide 5: Solution: Two Actor Chains

> "Our solution is to create two separate actor chains that are selected at compile time based on the MBF configuration.

> The original actor chain uses `parameter_transporter` from the detray library. This actor maintains a pointer to the accumulated Jacobian in its state structure.

> We created a new actor called `bound_updater` that has an **empty state structure** - no Jacobian pointer at all. This actor still does covariance transport, which is always needed, but it skips the Jacobian aggregation step.

> At runtime, based on the `run_mbf_smoother` configuration flag, the host code selects which propagator type to use. This propagator type is then used to instantiate the correct kernel specialization.

> The key benefit is that the compiler can generate more efficient code when it knows the Jacobian pointer doesn't exist - on some architectures, this saves 32 registers."

---

## Slide 6: Key Code: Actor State Comparison

> "Let me show you the actual code difference between these two actors.

> On the left is `parameter_transporter` from detray. Its state structure contains a pointer to the accumulated Jacobian matrix. In the operator function, after computing the full Jacobian, it performs a 6-by-6 matrix multiplication to aggregate the Jacobian, then writes the result back to global memory.

> On the right is our new `bound_updater`. Notice that the state structure is completely empty - just an empty struct. In its operator, we still compute the full Jacobian because that's needed for covariance transport, but we skip the aggregation step entirely.

> The comment in the code says it clearly: 'No Jacobian aggregation here - that's only needed for MBF smoother. This is the key difference from parameter_transporter.'

> By removing this aggregation, we save both compute and memory bandwidth at every surface the track encounters."

---

## Slide 7: Kernel Dispatch Logic

> "Here's how we implement the compile-time dispatch in the propagation kernel.

> We use C++17's `if constexpr` with a type trait called `has_jacobian_transport_v`. This trait checks whether the propagator's actor chain contains `parameter_transporter` or `bound_updater`.

> When the condition is true - meaning we have the Jacobian transport actor - we initialize the Jacobian to identity and set up the pointer in the actor state. When false, we skip this initialization entirely.

> The type trait is defined using template metaprogramming to search through the actor tuple. The key point is that this happens at compile time, so there's no runtime overhead.

> This approach results in 18 kernel specializations: 3 detector types, times 3 B-field configurations, times 2 MBF variants."

---

## Slide 8: Benchmark Results

> "Now let's look at the performance results.

> We benchmarked on a Tesla V100-SXM2 using 8 CPU threads with the ttbar mu-200 dataset - that's the Open Data Detector with 200 pileup interactions. Critically, we did an apples-to-apples comparison where both baseline and optimization use MBF=false, so we're measuring only the impact of the conditional Jacobian aggregation.

> The baseline commit is a48cc783, and the optimization commit is 25894cca.

> The baseline achieved 36.57 events per second. With our optimization, this increased to 43.27 events per second - an 18.3% improvement.

> Latency dropped from 27.34 milliseconds to 23.11 milliseconds per event - a 15.5% reduction.

> Importantly, all 710 CUDA tests pass, confirming that we haven't broken any physics correctness."

---

## Slide 9: NCU Profiling: Register Reduction Confirmed

> "To understand *why* we got this speedup, we used NVIDIA Nsight Compute version 2024.1.1 for detailed kernel profiling. We focused on the `propagate_to_next_surface` kernel - the main track finding kernel. This was done on an RTX 2080 Ti using a single event with one CPU thread to isolate GPU behavior.

> The key finding is that **register usage in `propagate_to_next_surface` dropped from 128 to 96 registers per thread** - a 25% reduction. This is exactly what we hoped for.

> This register reduction has a cascading effect: the block limit per SM increases from 4 to 5, theoretical occupancy jumps from 50% to 62.5%, and achieved occupancy improves from 39.3% to 48.6%.

> On the compute side, the kernel duration decreased by 9.5%, and total executed instructions dropped by 4.6%. Memory throughput actually increased by 12% because higher occupancy means better memory parallelism.

> So we have a dual optimization mechanism: register reduction enables higher occupancy, and skipping the aggregation reduces total instructions."

---

## Slide 10: Optimization Mechanism Summary

> "Let me summarize the two complementary mechanisms behind this optimization.

> First, register reduction: on sm_75, we go from 128 to 96 registers. This enables higher occupancy, which means better latency hiding and a 9.3% occupancy improvement.

> Second, skipped aggregation: we eliminate the 6-by-6 matrix multiplication and global memory I/O at every surface. This reduces instructions by 4.6%.

> Combined, these deliver an 18.3% throughput improvement.

> For a typical track hitting 15 surfaces, we save approximately 3,240 floating-point operations and 4,320 bytes of memory traffic. Multiply that by thousands of tracks per event, and the savings add up."

---

## Slide 11: Conclusion

> "What did we learn from this work?

> Our theoretical analysis predicted about 36 registers saved; we achieved 32 on sm_75. We expected 10 to 25 percent occupancy gain; we got 9.3 percent. But we exceeded our throughput target with 18.3% improvement.

> Multiple mechanisms contribute to performance - it's not just register reduction OR instruction reduction, it's both working together.

> And there's a trade-off: this optimization only applies when MBF smoother is disabled. For physics analyses requiring backward smoothing, the original actor chain is still needed."

---

## Slide 12: Summary

> "To summarize: Conditional Jacobian Aggregation achieves an 18.3% throughput improvement, 25% register reduction on sm_75, and 9.3% higher occupancy - while maintaining full test correctness.

> The implementation involves a new `bound_updater` actor with empty state, two actor chain variants, 18 kernel specializations, and compile-time dispatch using type traits.

> The mechanism is straightforward: skip the 6-by-6 matrix multiplication when MBF is disabled. The Jacobian is still computed for covariance transport, but we don't aggregate it.

> The key files are `bound_updater.hpp` for the new actor and `propagate_to_next_surface.ipp` for the dispatch logic.

> The main conclusion is that this optimization works through a dual mechanism: architecture-dependent register reduction plus universal instruction savings from skipped matrix operations."

---

## Slide 13: Questions

> "Thank you for your attention. I'm happy to take any questions.

> The full documentation is available in the doc directory, including detailed profiling reports and implementation plans."

---

## Backup Slides

### If asked about Jacobian computation still happening:

> "Good question. We're skipping the *aggregation* step, not the computation. The `get_full_jacobian()` function is still called because it's needed for covariance transport. A future optimization could potentially skip this computation entirely when covariance transport is also not needed, but that would require more invasive changes."

### If asked about runtime vs compile-time dispatch:

> "We chose compile-time dispatch because `if constexpr` enables dead code elimination. With runtime dispatch using a regular `if`, the compiler would still need to reserve register space for the Jacobian pointer, even if it's never used. Compile-time dispatch allows the compiler to generate completely separate kernel specializations."

### If asked about MBF smoother purpose:

> "The Multi-Branch Fit smoother performs a backward pass through the track, using the accumulated Jacobians to refine the track parameters. It's important for high-precision physics measurements. For real-time trigger applications where speed matters more than ultimate precision, it can be disabled."

### If asked about cuobjdump vs NCU discrepancy:

> "This was actually a key learning for us. The CUDA compiler generates different code for different target architectures. The sm_75 instruction set is richer than sm_70, and the compiler's register allocation heuristics differ. This is why we emphasize profiling on production hardware - paper analysis can be misleading."

### If asked about combining with multi-event batching:

> "Great question. These are complementary optimizations. Multi-event batching improves GPU utilization by processing more tracks in parallel - it achieved 93% throughput improvement. Conditional Jacobian Aggregation reduces per-thread resource usage - 18.3% improvement. They can absolutely be combined, and the benefits should be roughly additive since they address different bottlenecks."

### If asked about build_tracks improvement:

> "The 86.3% improvement in build_tracks was from changing the MBF default from true to false. When MBF is disabled, build_tracks doesn't need to process the accumulated Jacobians for backward smoothing. This is a separate effect from our Jacobian aggregation optimization - which is why we did the apples-to-apples re-benchmark to isolate our true contribution."

### If asked about warp cycles regression:

> "Yes, we observed that warp cycles per issued instruction actually increased from 47.7 to 59.9 - a 25% regression. This is counterintuitive, but it's offset by having more warps active. The net effect is still positive because the higher occupancy provides better latency hiding overall. This shows why you need to look at end-to-end metrics, not just individual counters."

---

## Timing Guide

| Section | Slides | Duration |
|---------|--------|----------|
| Introduction | 1-2 | 1-2 min |
| Problem Context | 3-4 | 3-4 min |
| Implementation | 5-7 | 4-5 min |
| Results | 8-10 | 4-5 min |
| Conclusion | 11-13 | 2-3 min |
| Q&A | - | 3-5 min |

**Total: ~16-20 minutes**

---

## Key Numbers to Remember

- **+18.3%** throughput improvement (apples-to-apples)
- **-25%** register reduction (128 → 96, on sm_75 only)
- **+9.3%** achieved occupancy
- **-4.6%** instructions executed
- **-9.5%** kernel duration
- **710/710** tests passing
- **18** kernel specializations (3 × 3 × 2)
- **~216 FLOPs** saved per surface (6×6 matrix mult)
- **~288 bytes** saved per surface (Jacobian read + write)
- **Commits:** a48cc783 (baseline), 25894cca (optimization)
- **Dataset:** odd/geant4_ttbar_mu200
- **Profiler:** ncu 2024.1.1
