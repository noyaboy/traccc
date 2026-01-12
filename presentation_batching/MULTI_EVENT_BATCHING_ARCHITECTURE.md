# Multi-Event Batching Architecture for TRACCC CKF

**Date**: 2025-11-15
**Priority**: #1 (Expert Recommendation)
**Expected Gain**: 2-3× throughput improvement
**Risk Level**: Low-Medium (well-established pattern in HEP GPU tracking)

---

## Executive Summary

Current CKF processes events sequentially, launching kernels with adaptive grid sizing per event. This results in low GPU utilization (15-20%) because many steps have too few candidates (100-500) to fill 68 SMs on the RTX 2080 Ti.

**Multi-event batching** processes N events together in a single CKF invocation, increasing parallelism across events to maintain high GPU occupancy even when individual events have small workloads.

**Key Insight**: When Event A has 100 candidates at step 5 and Event B has 200 candidates at step 5, batching them creates 300 candidates → better GPU utilization without changing per-candidate physics.

---

## 1. High-Level Architecture

### 1.1 Current Architecture (Single Event)

```
Host Loop:
  for event in events:
    CKF(event) {
      for step in 0..max_steps:
        propagate(n_candidates)       // n varies 100-20,000
        find_tracks(n_candidates)
        apply_interaction(n_candidates)
    }
```

**Problem**: Low `n_candidates` at many steps → GPU underutilized

### 1.2 Batched Architecture (N Events)

```
Host Loop:
  for batch in batches_of_N(events):
    CKF_Batched(batch) {
      for step in 0..max_steps:
        n_total = sum(n_candidates[i] for event i in batch)
        propagate(n_total)             // Larger workload
        find_tracks(n_total)           // With event boundary checks
        apply_interaction(n_total)
    }
```

**Benefit**: `n_total` is N× larger → better GPU utilization

---

## 2. Data Structures and Memory Layout

### 2.1 Offset-Based Indexing (Chosen Approach)

Instead of adding `event_id` fields to every data structure, use **offset arrays** to track event boundaries.

```cpp
struct batch_metadata {
    unsigned int num_events;  // Batch size (N)

    // Offset arrays (size = num_events + 1)
    // offsets[i] = starting index for event i
    // offsets[i+1] - offsets[i] = count for event i
    vecmem::data::vector_view<const unsigned int> seed_offsets_view;
    vecmem::data::vector_view<const unsigned int> meas_offsets_view;

    // Per-event counts (for validation)
    std::vector<unsigned int> seeds_per_event;
    std::vector<unsigned int> meas_per_event;
};
```

**Example with N=3 events**:
```
Event A: 50 seeds,  200 measurements
Event B: 30 seeds,  150 measurements
Event C: 40 seeds,  180 measurements

Batched buffers:
  seeds:        [A's 50 seeds | B's 30 seeds | C's 40 seeds]  (total: 120)
  measurements: [A's 200 meas | B's 150 meas | C's 180 meas]  (total: 530)

Offset arrays:
  seed_offsets = [0, 50, 80, 120]
  meas_offsets = [0, 200, 350, 530]

Event lookup:
  seed index 75 belongs to event 1 (B) because offsets[1] <= 75 < offsets[2]
  meas index 400 belongs to event 2 (C) because offsets[2] <= 400 < offsets[3]
```

### 2.2 Memory Layout

**Current (Single Event)**:
```
seeds:          [s0, s1, ..., sN]
measurements:   [m0, m1, ..., mM]
params:         [p0, p1, ..., pP]
links:          [l0, l1, ..., lL]
```

**Batched (N Events)**:
```
seeds:          [evt0_seeds | evt1_seeds | ... | evtN_seeds]
measurements:   [evt0_meas  | evt1_meas  | ... | evtN_meas]
params:         [evt0_params | evt1_params | ... | evtN_params]
links:          [evt0_links  | evt1_links  | ... | evtN_links]

+ seed_offsets: [0, N0, N0+N1, ..., total_seeds]
+ meas_offsets: [0, M0, M0+M1, ..., total_meas]
```

**Advantages**:
1. **Contiguous memory**: All event data packed together for coalesced access
2. **Minimal overhead**: Only 2 offset arrays (O(N) space) instead of event_id per element (O(total_elements) space)
3. **Backward compatible**: N=1 reduces to current single-event case
4. **Cache friendly**: Sequential access within each kernel

---

## 3. Event Boundary Enforcement

The **critical invariant**: Candidates from Event A must never link to measurements from Event B.

### 3.1 Event ID Lookup (Device Helper)

```cpp
/// Binary search to find which event a global index belongs to
/// @param global_idx Index into batched buffer (e.g., parameter index)
/// @param offsets Offset array (size = num_events + 1)
/// @return Event ID (0 to num_events-1)
__device__ __forceinline__ unsigned int get_event_id(
    unsigned int global_idx,
    const vecmem::device_vector<const unsigned int>& offsets) {

    unsigned int left = 0;
    unsigned int right = offsets.size() - 2;  // Last valid event index

    while (left < right) {
        unsigned int mid = (left + right + 1) / 2;
        if (global_idx >= offsets[mid]) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }

    return left;
}
```

**Complexity**: O(log N) per lookup
**Cost**: With N=10, 3-4 integer comparisons per boundary check
**Frequency**: Once per candidate per step → acceptable overhead

### 3.2 Find Tracks Kernel Modification

```cpp
// In find_tracks kernel, before creating a link:

// 1. Determine parameter's event
unsigned int param_event_id = 0;
if (payload.num_events > 1) {
    vecmem::device_vector<const unsigned int> seed_offsets(payload.seed_offsets_view);
    param_event_id = get_event_id(owner_global_thread_id, seed_offsets);
}

// 2. For each candidate measurement, check event boundary
for (const auto& candidate_meas : compatible_measurements) {
    unsigned int meas_event_id = 0;
    if (payload.num_events > 1) {
        vecmem::device_vector<const unsigned int> meas_offsets(payload.meas_offsets_view);
        meas_event_id = get_event_id(candidate_meas.surface_link, meas_offsets);
    }

    // 3. Only create link if same event
    if (param_event_id == meas_event_id) {
        // Create candidate_link as usual
        tmp_links.at(p_offset + l_pos) = {
            .meas_idx = meas_idx,
            .seed_idx = seed_idx,
            // ...
        };
    }
    // else: silently skip (different event)
}
```

**Optimization**: Cache `param_event_id` to avoid repeated lookups for the same parameter.

---

## 4. Kernel-Level Changes

### 4.1 Propagate Kernel

**Status**: ✅ **No changes required**

The propagate kernel operates on individual candidates without cross-candidate dependencies. Batching simply increases `n_candidates`:

```cpp
// Current:
propagate_to_next_surface<<<nBlocks, nThreads>>>(n_candidates, ...);

// Batched:
propagate_to_next_surface<<<nBlocks, nThreads>>>(n_candidates_total, ...);
```

Where `n_candidates_total = sum(n_candidates[i] for i in batch)`.

The adaptive grid sizing still works:
```cpp
nBlocks = (n_candidates_total + 127) / 128;
```

### 4.2 Find Tracks Kernel

**Changes Required**:
1. Add `batch_metadata` to payload
2. Add event boundary enforcement (see Section 3.2)

```cpp
template <typename detector_t>
struct find_tracks_payload {
    // Existing fields...
    typename detector_t::const_view_type det_data;
    measurement_collection_types::const_view measurements_view;
    // ...

    // NEW: Batching metadata
    unsigned int num_events;  // 1 for single-event (backward compatible)
    vecmem::data::vector_view<const unsigned int> seed_offsets_view;
    vecmem::data::vector_view<const unsigned int> meas_offsets_view;
};
```

**Backward Compatibility**:
- When `num_events == 1`: offset arrays are `[0, total]`, event_id lookups return 0 → no overhead
- When `num_events > 1`: boundary checks activate

### 4.3 Apply Interaction Kernel

**Status**: ✅ **No changes required**

Material interaction is per-candidate, no cross-candidate dependencies.

---

## 5. Host-Side API Changes

### 5.1 Current Single-Event API

```cpp
auto result = combinatorial_kalman_filter(
    det_view,
    field,
    measurements_view,
    seeds_view,
    config,
    mr,
    copy,
    log,
    stream,
    warp_size
);
```

### 5.2 New Batched API

```cpp
/// Batched CKF processing for N events
/// @param batch_size Number of events in batch (1 = single-event)
/// @param det_views Vector of detector views (size = batch_size)
/// @param measurements_batch Batched measurements with offset metadata
/// @param seeds_batch Batched seeds with offset metadata
auto result = combinatorial_kalman_filter_batched(
    unsigned int batch_size,
    const std::vector<detector_view_type>& det_views,
    const batched_measurement_collection& measurements_batch,
    const batched_seed_collection& seeds_batch,
    const finding_config& config,
    const memory_resource& mr,
    vecmem::copy& copy,
    const Logger& log,
    stream& stream,
    unsigned int warp_size
);
```

**Helper Structure**:
```cpp
template <typename T>
struct batched_collection {
    vecmem::data::vector_view<T> data;  // Concatenated data
    vecmem::data::vector_view<const unsigned int> offsets;  // Event boundaries
    unsigned int num_events;
    std::vector<unsigned int> counts_per_event;
};
```

### 5.3 Batching Helper Function

```cpp
/// Concatenate N events' data with offset tracking
template <typename T>
batched_collection<T> create_batched_collection(
    const std::vector<vecmem::vector<T>>& per_event_data,
    const memory_resource& mr,
    vecmem::copy& copy
) {
    unsigned int num_events = per_event_data.size();
    std::vector<unsigned int> offsets(num_events + 1, 0);

    // Compute offsets
    for (unsigned int i = 0; i < num_events; i++) {
        offsets[i + 1] = offsets[i] + per_event_data[i].size();
    }

    // Allocate concatenated buffer
    unsigned int total_size = offsets[num_events];
    vecmem::vector<T> batched_data(total_size, &mr);

    // Copy data
    for (unsigned int i = 0; i < num_events; i++) {
        std::copy(per_event_data[i].begin(),
                  per_event_data[i].end(),
                  batched_data.begin() + offsets[i]);
    }

    // Create offset buffer on device
    vecmem::vector<unsigned int> offset_vec(offsets.begin(), offsets.end(), &mr);

    return {
        .data = copy.to(batched_data, mr),
        .offsets = copy.to(offset_vec, mr),
        .num_events = num_events,
        .counts_per_event = {/* counts */}
    };
}
```

---

## 6. Memory Requirements and Scaling

### 6.1 Memory Scaling

**Current (Single Event)**:
- Seeds: ~100-500 × 48 bytes = 5-24 KB
- Measurements: ~10,000 × 64 bytes = 640 KB
- Params (worst case): 100 steps × 500 seeds × 10 branches × 128 bytes = 64 MB
- Links: 100 steps × 500 × 10 × 32 bytes = 16 MB

**Total per event**: ~80 MB (worst case)

**Batched (N=10 Events)**:
- Total: ~800 MB (worst case)
- RTX 2080 Ti: 11 GB VRAM → **Fits comfortably**

**Scaling Analysis**:
```
N=5:   ~400 MB  ✅ Safe
N=10:  ~800 MB  ✅ Safe
N=20:  ~1.6 GB  ✅ Safe
N=50:  ~4 GB    ✅ Safe (leaves 7 GB for detector geometry, fields)
```

**Recommendation**: Start with N=10, tune based on memory profiling.

### 6.2 Overhead Analysis

**Additional Memory**:
- Offset arrays: 2 × (N+1) × 4 bytes = 88 bytes for N=10 → **negligible**

**Computational Overhead**:
- Event ID lookups: O(log N) = ~3-4 comparisons for N=10
- Frequency: Once per parameter per step
- Cost: ~10-20 cycles per lookup
- Compared to: RK4 integration ~1000 cycles → **<2% overhead**

---

## 7. Implementation Phases

### Phase A: Data Structure Changes (Week 1)

**Tasks**:
1. Define `batched_collection<T>` template
2. Define `batch_metadata` structure
3. Implement `create_batched_collection()` helper
4. Add batching fields to `find_tracks_payload`

**Validation**: Single-event case (N=1) compiles and runs

### Phase B: Event Boundary Enforcement (Week 1-2)

**Tasks**:
1. Implement `get_event_id()` device function
2. Modify `find_tracks` kernel with boundary checks
3. Add unit tests for offset lookups

**Validation**: Two-event batch (N=2) produces same results as two separate runs

### Phase C: Host-Side Batching (Week 2-3)

**Tasks**:
1. Implement `combinatorial_kalman_filter_batched()` wrapper
2. Modify main CKF loop to handle batched buffers
3. Update output unbatching (separate results per event)

**Validation**: N=5, N=10 batches produce identical physics results

### Phase D: Benchmarking and Tuning (Week 3-4)

**Tasks**:
1. Measure throughput for N=1, 5, 10, 20
2. Profile GPU utilization improvement
3. Tune batch size based on memory/performance trade-offs
4. Validate track efficiency remains unchanged

**Success Criteria**:
- Throughput ≥2× baseline (38 ev/s for N=10)
- Track efficiency within 0.1% of baseline
- Memory usage <7 GB

---

## 8. Risk Mitigation

### Risk 1: Cross-Event Link Bugs

**Mitigation**:
- Extensive unit tests with artificial event boundaries
- Validation: compare N=2 batch vs 2 separate runs
- Debug mode: assert no links cross event boundaries

### Risk 2: Memory Overflow

**Mitigation**:
- Dynamic batch size selection based on available VRAM
- Memory usage profiling before batching
- Fallback to smaller batch size if allocation fails

### Risk 3: Performance Regression for Small Events

**Mitigation**:
- Adaptive batching: use larger N for small events, smaller N for large events
- Single-event fast path (N=1) with zero overhead

### Risk 4: Detector Geometry Conflicts

**Assumption**: Detector geometry is shared across events (same detector for all events in batch)

**Mitigation**:
- Validate assumption in test data
- If needed: support per-event detector views (adds complexity)

---

## 9. Alternative Designs Considered

### Alternative 1: Event ID Field (Rejected)

Add `event_id` field to every candidate, link, parameter.

**Pros**: Simpler lookups (O(1) instead of O(log N))
**Cons**:
- 4 bytes per element → ~200 MB overhead for N=10
- Memory bandwidth waste
- AoS → SoA conversion needed

### Alternative 2: Per-Event Streams (Rejected)

Launch separate CUDA streams for each event.

**Pros**: Automatic event isolation
**Cons**:
- Stream synchronization overhead
- Doesn't solve GPU underutilization (streams compete for same SMs)
- Complex host-side orchestration

### Alternative 3: Persistent Kernel (Deferred to Phase 2)

Single persistent kernel with work queues across events.

**Pros**: Maximum flexibility, dynamic load balancing
**Cons**:
- Very high complexity
- 8-12 week development
- 70% failure risk (per project history)

**Decision**: Pursue batching first (lower risk, proven pattern), then consider persistent kernel in Phase 2.

---

## 10. Expected Performance Improvement

### Baseline (N=1)
- Throughput: 18.9 ev/s
- GPU utilization: 15-20%
- Median candidates/step: ~500

### Batched (N=10)
- Throughput: **38-57 ev/s** (2-3× improvement)
- GPU utilization: **40-60%** (2.5-3× improvement)
- Median candidates/step: ~5,000 (10× more parallelism)

**Why this works**:
- Small steps (100-500 candidates/event) × 10 events = 1,000-5,000 candidates → fills GPU
- Large steps (5,000+ candidates/event) × 10 events = 50,000+ candidates → saturates GPU
- Adaptive grid benefits from larger workloads

**Combined with algorithmic tuning** (1.2-1.3×):
- **Total expected**: 47-76 ev/s (2.5-4× over baseline)

---

## 11. Validation Plan

### Unit Tests
1. **Offset lookup correctness**: Verify `get_event_id()` for all boundary cases
2. **Event isolation**: Ensure no cross-event links in N=2 batch
3. **Memory safety**: Check no out-of-bounds accesses

### Integration Tests
1. **N=1 equivalence**: Batched(N=1) == Current single-event
2. **N=2 equivalence**: Batched(N=2) == 2× single-event (same tracks)
3. **Large batch**: N=20 runs without errors

### Performance Tests
1. **Throughput scaling**: Measure ev/s for N=1,5,10,20
2. **GPU utilization**: Profile with nsys for N=1 vs N=10
3. **Memory usage**: Track peak VRAM usage

### Physics Validation
1. **Track efficiency**: Compare found tracks vs truth particles
2. **Track purity**: Check for fake track rate
3. **Chi² distribution**: Verify no physics changes

---

## 12. Success Metrics

| Metric | Baseline (N=1) | Target (N=10) | Stretch Goal |
|--------|----------------|---------------|--------------|
| Throughput (ev/s) | 18.9 | 38-45 | 50-57 |
| GPU Utilization | 15-20% | 40-50% | 55-60% |
| Track Efficiency | 95.0% | ≥94.9% | ≥95.0% |
| Memory Usage | 1-2 GB | <7 GB | <5 GB |
| Development Time | - | 4 weeks | 3 weeks |

---

## 13. Implementation Checklist

- [ ] Phase A: Data structures (Week 1)
  - [ ] Define `batched_collection<T>`
  - [ ] Implement `create_batched_collection()`
  - [ ] Add fields to `find_tracks_payload`
  - [ ] Single-event (N=1) compiles

- [ ] Phase B: Event boundaries (Week 1-2)
  - [ ] Implement `get_event_id()`
  - [ ] Modify `find_tracks` kernel
  - [ ] Unit test offset lookups
  - [ ] Two-event (N=2) validation

- [ ] Phase C: Host-side (Week 2-3)
  - [ ] Implement `combinatorial_kalman_filter_batched()`
  - [ ] Modify main loop
  - [ ] Output unbatching
  - [ ] N=5,10 validation

- [ ] Phase D: Benchmarking (Week 3-4)
  - [ ] Throughput measurements
  - [ ] GPU profiling
  - [ ] Batch size tuning
  - [ ] Track efficiency validation

---

## 14. References

1. Expert recommendations: `ckf_gpu_recommendations.md`, Section 1
2. ALICE GPU tracking: Multi-event batching for CA+KF
3. Implementation plan: `PHASE1_BATCHING_IMPLEMENTATION_PLAN.md`
4. Baseline analysis: `FINAL_PROJECT_STATUS.md`

---

**Status**: Architecture design complete, ready for Phase A implementation
**Next Steps**: Begin data structure implementation (Week 1)
**Estimated Completion**: 4 weeks from start date
