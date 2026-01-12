# Presentation Script
## GPU-Accelerated Track Reconstruction Optimization
### Multi-Event Batching for the ACTS/traccc Framework

**Estimated Duration: 15-20 minutes**

---

## Slide 1: Title Slide

> Good [morning/afternoon], Professor Lai. Thank you for meeting with me today. I'd like to present my progress on GPU-accelerated track reconstruction optimization for the ACTS traccc framework. This has been my primary focus over the past several weeks, and I'm excited to share the results we've achieved.

---

## Slide 2: Outline

> I'll structure my presentation as follows: First, I'll give a brief overview of the project context and objectives. Then I'll discuss the methodology I used for profiling and optimization. The main portion will cover the implementation details of our key optimizations. I'll then present the performance results, followed by an important section on negative results—approaches that didn't work but provided valuable insights. I'll then cover our technical contributions, and finally summarize and discuss future work.

---

## Slide 3: Project Context

> Let me start with some context. traccc is a track reconstruction library that's part of the ACTS project at CERN. It's designed to be GPU-accelerated using CUDA, which is critical for processing the enormous data volumes expected from the High-Luminosity LHC.

> My work focuses specifically on the Combinatorial Kalman Filter, or CKF, which is the most compute-intensive component of the reconstruction pipeline. You can see here in the diagram that CKF sits at the core of the pipeline, taking seeds from the seeding stage and producing track candidates for fitting.

> We're particularly interested in high-pileup events—with mu equals 200, meaning 200 simultaneous proton-proton interactions per bunch crossing. This represents the challenging conditions expected at HL-LHC.

---

## Slide 4: Research Objectives

> Our primary goal is to improve the GPU throughput of the CKF algorithm for these high-pileup events.

> We have four specific objectives:
> 1. Implement multi-event batching to improve GPU utilization
> 2. Optimize memory access patterns using constant and texture memory
> 3. Reduce synchronization overhead in CUDA kernels
> 4. And critically, validate correctness across over 1,460 test cases

> Our success metric was at least 50% throughput improvement while maintaining physics accuracy. As I'll show, we significantly exceeded this target.

---

## Slide 5: Optimization Approach

> For methodology, I used systematic GPU profiling with NVIDIA's Nsight Systems and Nsight Compute tools.

> As you can see in the circular diagram, I followed an iterative development cycle: profile to identify bottlenecks, implement an optimization, then validate correctness—repeating this cycle throughout the project.

> The initial profiling revealed several key bottlenecks, shown here as large numbers. Most striking was that 90.5% of CUDA API time was spent in synchronization—the GPU was sitting idle waiting for unnecessary sync points. We also found 43% GPU idle time and only about 15% occupancy with single-event processing.

> All experiments were conducted on a Tesla V100 GPU with CUDA 12.6, using the ttbar mu-200 high-pileup dataset.

---

## Slide 6: Multi-Event Batching Architecture

> Now let me explain our core innovation: multi-event batching.

> The key insight is that processing events one at a time leaves the GPU underutilized. Instead, we process N events simultaneously in a single CKF invocation. We concatenate the seeds and measurements from multiple events into a single buffer, using offset arrays to track event boundaries.

> You can see in the diagram how events 0, 1, through N are concatenated into a single batched buffer. The offset array tells us where each event's data begins and ends—so offsets [0, 50, 80, 120] means event 0 has indices 0-49, event 1 has 50-79, and so on.

> The critical challenge is event boundary enforcement—we must ensure track candidates from one event don't accidentally link to measurements from another event. We solve this with a binary search function called `get_event_id()` that maps any global index back to its event ID.

---

## Slide 7: Memory Optimizations

> We implemented three key memory optimizations.

> First, we moved event offset arrays to CUDA constant memory. These arrays are accessed by every thread during event boundary enforcement, so putting them in constant memory gives us zero-latency cached access instead of 400 cycles for global memory. This saves approximately 3,200 cycles per measurement candidate and improved throughput by 3.7 percent. Combined with texture memory optimization, the total improvement was 6.2 percent.

> Second, we enabled texture memory for magnetic field lookups. The RK stepper does 3D spatial lookups for the B-field, and texture memory provides hardware-accelerated caching for this access pattern. This gave us a 2.7% improvement and also enabled realistic inhomogeneous magnetic fields at no performance cost.

> Third, we replaced uses of `thrust::seq`—sequential Thrust execution policy—with inline device functions. Thrust with sequential policy has significant overhead on GPU. Our inline implementations of lower_bound, find, and count are much more efficient, giving us 7.8% throughput improvement.

---

## Slide 8: Synchronization Optimization

> One of our wins came from removing unnecessary stream synchronizations.

> On the left you see the original code: after Thrust sorting, there was an explicit `str.synchronize()` call that blocked the GPU pipeline. On the right is our optimized version—we simply removed the sync.

> The key insight is that modern Thrust with the `par_nosync` policy provides stream-ordered execution. CUDA guarantees that operations within a stream execute in order. So when we launch the next kernel, it automatically waits for the sort to complete—no explicit sync needed.

> NSYS profiling confirmed the impact: cudaStreamSynchronize calls dropped by 92%—from 8,525 to just 706. This sync elimination, combined with device-side operations replacing the D→H→D transfer pattern, was the primary driver of our 93% total gain.

---

## Slide 9: Bug Fix - Chi-Squared Threshold

> I want to highlight an important bug we discovered and fixed. When we first ran tests with batching enabled, the GPU was finding 2.75 times more tracks than the CPU—an obvious correctness failure.

> The root cause was subtle: I had added region-dependent chi-squared thresholds that vary by detector region. But the existing tests only configured `cfg.chi2_max = 10`, and my code was using the new region-specific defaults of 50, 100, and 150 instead. This caused the GPU to accept far more track candidates.

> The fix was to detect when region thresholds are at their default values and fall back to the global `chi2_max`. This preserves the new feature for production while maintaining backward compatibility with existing tests.

> This experience reinforced an important lesson: when adding configuration options, always maintain backward compatibility with existing test baselines.

---

## Slide 10: Performance Results

> Now for the results. The bar chart on the left shows our throughput improvements as we scaled up the batch size and added optimizations.

> Starting from a baseline of 15.61 events per second with single-event processing, we see progressive gains as batch size increases—12% at N=4, then 69% at N=14, and 84% at N=24. The thrust::seq replacement added another boost to 91%.

> The big result is shown on the right: **+93% throughput improvement**, going from 15.61 to 30.12 events per second at batch size 48. This significantly exceeds our 50% target by 43 percentage points.

---

## Slide 11: Synchronization Reduction (NSYS)

> This slide shows the NSYS profiling results that validate our synchronization elimination strategy.

> On the left, the large KPI cards highlight the dramatic reductions. Most importantly, `cudaStreamSynchronize` calls dropped by 92%—from 8,525 to just 706. Below that, we see 77% fewer kernel launches and 69% fewer memory operations.

> On the right, the horizontal bars visualize how kernel instance counts dropped by 30 to 40 times. The propagate kernel went from 1,391 instances to just 42. The fitting kernels dropped from 72 instances each to just 2.

> The key insight here: we didn't make individual kernels faster. Instead, we eliminated the overhead between kernel calls—sync elimination was the primary optimization driver. This is why the 92% sync reduction translates to 93% throughput improvement.

---

## Slide 12: Batch Size Optimization

> This slide shows our batch size optimization study. We tested batch sizes from 1 to 64.

> The key finding is that batch size 48 is optimal for our Tesla V100-SXM2-32GB GPU, achieving 30.12 events per second. You can see the curve follows a typical optimization pattern—throughput increases rapidly at first, then shows diminishing returns as we approach the optimal point. Beyond N=48, we see a drop at N=64 due to memory pressure.

> We also discovered some pre-existing bugs: batch sizes 6, 8, and 12 cause memory alignment errors in the vecmem library. These are pre-existing issues not caused by our changes, but they constrained which batch sizes we could use. At N=64 and beyond, we hit out-of-memory issues depending on the dataset.

> Based on this analysis, we changed the default batch size from 1 to 48 in the production configuration.

---

## Slide 13: Kernel-Level Analysis (NCU)

> To validate our optimizations at the kernel level, we ran detailed profiling with NVIDIA Nsight Compute.

> The left table shows occupancy improvements. The most dramatic gains are in the fitting kernels: fit_forward improved from 15.4% to 38.4% occupancy—a 150% improvement. fit_backward doubled from 14.9% to 29.9%. The CKF kernels also improved: propagate went from 35% to 42.8%, and find_tracks from 21.4% to 23.7%.

> The key finding here is that the baseline fitting kernels were running at only 15% occupancy—meaning 85% of GPU resources were idle. This confirms that single-event processing severely under-utilizes the GPU, and batching directly addresses this problem.

> On the right, we see memory throughput improvements. fit_backward shows the highest gain at 154%—nearly tripling from 96.8 to 245.5 gigabytes per second. fit_forward improved by 121%. These improvements indicate that batching enables much better memory access patterns and bandwidth utilization.

> The grid size scaling confirms batching is working correctly: CKF kernels show 6.1x larger grids, and fitting kernels show 4.3x larger grids, corresponding to our batch-48 configuration processing multiple events per kernel invocation.

> Note that NCU profiling was performed on an RTX 2080 Ti, while throughput benchmarks were measured on the V100. The relative improvements in occupancy and memory throughput are consistent across GPU architectures.

---

## Slide 14: Validation Results

> Validation was crucial—performance gains mean nothing if we lose correctness.

> We ran the complete test suite: 1,460 tests across core, I/O, examples, CPU, and CUDA test binaries. All tests pass.

> For physics validation, we verified track count agreement within 0.1% tolerance between CPU and GPU, consistent chi-squared distributions, and no event boundary violations in batched mode. Results are identical between CPU and GPU implementations.

> We also conducted stability testing with 100+ events per configuration, multiple batch sizes, using a cold run plus measured events protocol, and achieved reproducibility within 1% variance.

---

## Slide 15: Documented Negative Results

> I want to emphasize the scientific value of documenting what *didn't* work.

> First, we tried removing theta-based sorting of measurements, hypothesizing it was wasted work. But profiling showed sorting actually *improves* memory coalescing. The sorted access pattern enables better cache utilization than random access.

> Second, we implemented track compaction to remove dead candidates between steps. But in early CKF steps, 100% of candidates are still alive, so compaction adds pure overhead—18.8% slowdown.

> Third, we tried reducing batch size thinking smaller working sets would cache better. But GPUs need massive parallelism; smaller batches just reduce utilization.

> We also explored persistent kernels and combined kernel fusion, but found we were already at 98% of theoretical efficiency—no headroom remained.

> These negative results are documented in detail and provided crucial guidance toward the optimizations that did work.

---

## Slide 16: Technical Contributions

> In terms of concrete contributions, you can see the key numbers displayed at the top: 18 new source files, approximately 3,000 lines of CUDA/C++ code, 68 commits over the development period, and about 45,000 lines of documentation.

> The key components include the batched CKF wrapper, constant memory infrastructure, device-side concatenation kernels, and inline Thrust replacements.

> We also produced extensive documentation: approximately 100 markdown documents covering optimization plans, performance analysis, safety proofs for the synchronization changes, and detailed negative results documentation.

> The methodology itself—systematic profiling, iterative development, and documenting failures—is also a contribution that could benefit future optimization work.

---

## Slide 17: Summary and Future Work

> In summary, we achieved a 93% throughput improvement, all 1,460 tests passing, and a production-ready implementation that exceeds our 50% target.

> The key innovations were multi-event batching with proper event isolation, device-side concatenation replacing the costly D→H→D transfer pattern with D2D operations, constant memory combined with stream-ordered execution, and achieving 98% of theoretical maximum efficiency.

> For future work, we'd like to explore multi-GPU scaling. We analyzed CPU-GPU pipelining and found only about 2% headroom remaining—so it's lower priority. We also want to investigate the memory alignment bugs we discovered at batch sizes 6, 8, and 12. The ultimate goal is to contribute these optimizations upstream to the ACTS project.

> The implementation work was completed between November 15 and December 21, 2025. Next steps are thesis writing and preparing publication-ready results.

> To conclude: we successfully achieved **93% throughput improvement** in GPU track reconstruction, significantly exceeding our target while maintaining full correctness across all 1,460 tests.

---

## Slide 18: Questions

> Thank you for your attention. I'm happy to answer any questions about the implementation, results, or methodology.

---

# Anticipated Questions and Answers

## Q1: Why is batch size 48 optimal? Why not higher?

> At batch size 48, we're hitting the memory bandwidth ceiling of the V100. Going higher to 64 actually decreases throughput because we start experiencing memory pressure and potentially cache thrashing. There's also the issue that some batch sizes trigger pre-existing alignment bugs, constraining our options.

## Q2: How do you ensure tracks don't cross event boundaries?

> Every time the CKF considers linking a track candidate to a measurement, we call `get_event_id()` for both the candidate's seed and the measurement. If they're from different events, we reject the link. This binary search runs in O(log N) time where N is the batch size, and with constant memory, it's essentially zero overhead.

## Q3: Why did track compaction fail? Isn't reducing warp divergence good?

> In theory, yes. But the CKF has a specific structure: in early steps (1-4), nearly 100% of candidates are still alive because duplicate removal hasn't started yet. Only in later steps do we see significant dead candidates. The overhead of compaction—buffer allocation, kernel launch, synchronization to read the count—exceeded the benefit. We'd need selective compaction only for later steps, which adds complexity.

## Q4: How confident are you in the correctness?

> Very confident. We have 1,460 automated tests covering all components. The tests include both unit tests and physics validation tests that compare GPU results to CPU reference implementations. Track counts must agree within 0.1% tolerance. We also ran extensive manual validation on the ttbar mu-200 dataset.

## Q5: What's the theoretical maximum speedup possible?

> Based on our roofline analysis, the CKF is memory-bound with about 42% DRAM utilization. The theoretical maximum with perfect memory coalescing would be roughly 2.5x our current throughput. However, the memory access pattern is fundamentally scattered due to the algorithm's nature—tracks access different detector surfaces in unpredictable order. Our 93% improvement captures most of the "easy" gains; further improvement would require algorithmic changes to the CKF itself.

## Q6: Can this work be applied to other algorithms in traccc?

> Absolutely. The batching infrastructure is generic—the `batch_metadata` structure and concatenation kernels could be reused for seeding or fitting. The constant memory and texture memory patterns are also applicable. We designed the code to be modular for this reason.

## Q7: What was the most challenging part of this work?

> The chi-squared bug was particularly challenging to diagnose. The GPU was producing "reasonable" results—just more tracks than expected. It took careful code review comparing the GPU and CPU paths line by line to identify the subtle difference in threshold handling. This taught me the importance of having exact CPU reference implementations for validation.

## Q8: How long did this take?

> The main implementation and optimization work took about 5-6 weeks, from November 15 to December 21. A significant portion of that time was spent on failed approaches—the persistent kernel work, track compaction, and sorting experiments—which ultimately guided us to the successful optimizations.

## Q9: How was the synchronization removal proven safe?

> The key is understanding CUDA stream semantics. When using Thrust with the `par_nosync` policy, operations are guaranteed to be stream-ordered. CUDA ensures that operations submitted to the same stream execute in the order they were submitted. So when we launch a kernel after a Thrust sort, the kernel automatically waits for the sort to complete—no explicit synchronization needed. We documented detailed safety proofs for each removal site in our analysis documents.

## Q10: What about the vecmem memory alignment bugs?

> These are pre-existing bugs in the vecmem library that we discovered during testing—they're not caused by our changes. The bugs are triggered at specific batch sizes (N=6, 8, and 12) and cause memory alignment errors. As a workaround, we simply avoid those batch sizes in production. Investigating and fixing these bugs upstream in vecmem is listed as future work.

## Q11: How did you validate the optimization at kernel level?

> We used NVIDIA Nsight Compute (NCU) to profile individual CUDA kernels before and after optimization. NCU provides detailed metrics like achieved occupancy, memory throughput, and cache hit rates. The NCU data confirmed our hypothesis: baseline fitting kernels were running at only 15% occupancy, meaning 85% of GPU resources were idle. After batching, occupancy increased to 30-38%, and memory throughput more than doubled for some kernels. This kernel-level evidence validates that batching directly addresses GPU under-utilization.

## Q12: Why do cache hit rates decrease but performance improves?

> This is a deliberate trade-off. With batching, we process larger working sets—multiple events worth of data instead of one. This naturally reduces L1 cache hit rates from about 48% to 43%. However, the benefit is much better memory coalescing and bandwidth utilization. Memory throughput improved by up to 154% because threads across multiple events can access memory more efficiently together. The net result is significantly higher performance despite slightly lower cache efficiency. This is a common pattern in GPU optimization: it's often better to keep the GPU busy with good memory access patterns than to optimize for cache hits alone.

## Q13: What does NCU reveal about future optimization opportunities?

> NCU shows we still have headroom. Even after optimization, fit_forward is at 38% occupancy and fit_backward at 30%—well below the theoretical 100%. The kernels are memory-bound with scattered access patterns inherent to the CKF algorithm. Further improvements would likely require algorithmic changes to improve memory locality, such as restructuring data layouts or changing how tracks access detector geometry. The NCU data provides a roadmap for these future optimizations.
