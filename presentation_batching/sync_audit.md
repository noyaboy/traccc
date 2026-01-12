# Synchronization Audit for Batched CKF (N=2)

**Date:** 2025-11-18
**Commit:** 35e9675f (Phase 1 optimization branch)
**Objective:** Identify and categorize all CPU-GPU synchronization points to eliminate pipeline-blocking syncs

## Executive Summary

### Current Status
- **Total sync overhead:** 90.5% of CUDA API time (1,314 ms for 20 events)
- **Syncs per event:** ~54 synchronizations
- **Impact:** Major pipeline stalls preventing optimal GPU utilization

### Key Findings
1. **Critical bottleneck:** Measurement concatenation loop (line 398) syncs N times per batch
2. **Hidden syncs:** `get_size()` calls (lines 385-386) trigger cudaStreamSynchronize
3. **Removable syncs:** 7 high-priority sync points identified in batching path
4. **Expected gain:** 10-20% speedup from sync elimination alone

---

## Detailed Synchronization Inventory

### 1. VecMem Copy Operations (->wait())

| File | Line | Context | Call Pattern | Category | Priority |
|------|------|---------|--------------|----------|----------|
| full_chain_algorithm.cpp | 174 | Batch buffer copy | `copy(vecmem::get_data(cells), cells_copy)->wait()` | ❌ REMOVE | **CRITICAL** |
| full_chain_algorithm.cpp | 398 | Measurement concat loop | `m_copy(measurements_vec[i], event_measurements_host)->wait()` | ❌ REMOVE | **CRITICAL** |
| full_chain_algorithm.cpp | 259 | Single-event meas copy | `m_copy(measurements, measurements_host)->wait()` | ❌ REMOVE | Medium |
| full_chain_algorithm.cpp | 294 | Single-event seed copy | `host_copy(host_seeds, result)->wait()` | ✅ KEEP | - |
| full_chain_algorithm.cpp | 304 | Else branch meas copy | `m_copy(measurements, measurements_host)->wait()` | ❌ REMOVE | Low |
| full_chain_algorithm.cpp | 427 | Final track output | `host_copy(host_tracks, result)->wait()` | ✅ KEEP | - |
| full_chain_algorithm.cpp | 249 | Single-event final output | `host_copy(host_tracks, result)->wait()` | ✅ KEEP | - |

**Analysis:**
- **Line 174:** Batching path syncs immediately after copying cells to device. Should use stream ordering instead.
- **Line 398:** **HIGHEST PRIORITY** - Syncs N times in loop, creating D→H→D pattern. Should be replaced with device-side concatenation kernel.
- **Lines 294, 427, 249:** Required syncs for final output to host. Must keep.
- **Lines 259, 304:** Single-event path syncs, lower priority (not in batching path).

---

### 2. Hidden Synchronizations in VecMem

#### 2.1 get_size() Implementation

**File:** device/cuda/src/utils/get_size.hpp
**Lines:** 44, 54

```cpp
// Line 41-45
TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
    staging, data.size_ptr(), sizeof(typename T::size_type),
    cudaMemcpyDeviceToHost, stream));
TRACCC_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));  // ❌ SYNC!
return *staging;
```

**Impact:**
- Every call to `m_copy.get_size()` triggers cudaStreamSynchronize
- Used in full_chain_algorithm.cpp lines 385-386:
  ```cpp
  for (std::size_t i = 0; i < batch_size; ++i) {
      total_measurements += m_copy.get_size(measurements_vec[i]);  // ❌ SYNC!
  }
  ```

| Callsite | Line | Syncs/Batch | Category | Priority |
|----------|------|-------------|----------|----------|
| full_chain_algorithm.cpp | 385-386 | N syncs | ❌ REMOVE | **CRITICAL** |

**Action:** Store sizes in metadata during measurement creation, avoid querying device.

---

### 3. Explicit Stream Synchronizations

#### 3.1 Stream Wrapper

**File:** device/cuda/src/utils/stream.cpp
**Line:** 62

```cpp
void stream::synchronize() const {
    TRACCC_CUDA_ERROR_CHECK(cudaStreamSynchronize(m_stream->m_stream));
}
```

**Analysis:** This is the implementation of the stream synchronization method. No direct calls found in full_chain_algorithm.cpp, but may be called indirectly through VecMem.

#### 3.2 Seeding Algorithm

**File:** device/cuda/src/gbts_seeding/gbts_seeding_algorithm.cu
**Occurrences:** 26+ cudaStreamSynchronize calls

**Analysis:** Not in critical path for current optimization:
- Seeding: 4.4 ms/event (6.1% of total time)
- CKF+Fitting: 50 ms/event (69% of total time)

**Category:** ⏸️ DEFER to Phase 4

---

## Categorization Summary

### ❌ REMOVE (High Priority)

**Critical Path - Batching:**

1. **Line 174:** Batch cells copy sync
   - **Pattern:** `copy(...)->wait()`
   - **Fix:** Use stream ordering
   - **Expected gain:** ~2-4% speedup

2. **Lines 385-386:** get_size() loop syncs
   - **Pattern:** `m_copy.get_size(measurements_vec[i])` × N
   - **Fix:** Pass sizes in metadata
   - **Expected gain:** ~3-5% speedup

3. **Line 398:** Measurement concatenation loop sync
   - **Pattern:** `m_copy(...)->wait()` × N in loop
   - **Fix:** Device-side concatenation kernel
   - **Expected gain:** ~5-10% speedup

**Total expected gain from removing these 3:** **10-20% speedup**

### ✅ KEEP (Required)

1. **Line 427:** Final track output (batching path)
   - **Reason:** Host needs final results

2. **Line 249:** Final track output (single-event path)
   - **Reason:** Host needs final results

3. **Line 294:** Final seed output (single-event path)
   - **Reason:** Host needs final results

### ⚠️ INVESTIGATE

1. **VecMem internal syncs**
   - Check if `m_copy(...)->ignore()` truly avoids sync
   - Investigate Thrust algorithm syncs
   - Profile VecMem allocator behavior

2. **CUDA Graph compatibility**
   - Identify dynamic allocations preventing graph capture
   - Check if current code can be graph-ified in Phase 5

---

## Proposed Elimination Strategy

### Task 1.2: Eliminate VecMem Copy Syncs (Days 3-5)

**Target:** Line 174, 259, 304

**Approach:**
```cpp
// Before (with sync):
copy(vecmem::get_data(cells), cells_copy)->wait();

// After (stream ordering):
copy(vecmem::get_data(cells), cells_copy)->ignore();
// Kernel launch automatically waits for copy
m_clusterization(cells_copy, ...);
```

**Files to modify:**
- examples/run/cuda/full_chain_algorithm.cpp

---

### Task 1.3: Eliminate get_size() Syncs (Days 5-7)

**Target:** Lines 385-386

**Approach 1:** Pass sizes in metadata structure
```cpp
struct MeasurementBatch {
    measurement_collection_types::buffer buffer;
    std::size_t size;  // Store size when created
};
```

**Approach 2:** Use async size query with event synchronization
```cpp
// Query size asynchronously, sync only once at end
std::vector<std::size_t> sizes(batch_size);
for (...) {
    async_get_size(measurements_vec[i], &sizes[i], stream);
}
stream.synchronize();  // Single sync instead of N syncs
```

**Recommendation:** Approach 1 (metadata) is cleaner.

**Files to modify:**
- examples/run/cuda/full_chain_algorithm.cpp (measurement handling)
- Potentially: device/cuda/src/finding/* (if sizes need to be stored in kernels)

---

### Task 1.4: Replace D→H→D with D→D (Days 7-10)

**Target:** Line 398 (measurement concatenation loop)

**Current pattern (inefficient):**
```cpp
// Device → Host (with sync)
m_copy(measurements_vec[i], event_measurements_host)->wait();

// Host-side loop
for (std::size_t j = 0; j < event_measurements_host.size(); ++j) {
    concatenated_measurements_host[meas_offset + j] = event_measurements_host[j];
}

// Host → Device
m_copy.to(vecmem::get_data(concatenated_measurements_host), ...);
```

**New pattern (efficient):**
```cpp
// Device → Device (no sync until end)
concatenate_measurements_kernel<<<...>>>(
    measurements_vec.data(),
    concatenated_measurements_buffer,
    batch_size,
    offsets);
// No sync needed, next kernel uses stream ordering
```

**New kernel required:** `concatenate_measurements.cu`
- Input: Vector of measurement buffers
- Output: Single concatenated buffer
- Complexity: Simple parallel copy with offset calculation

**Files to create:**
- device/cuda/src/finding/kernels/concatenate_measurements.cu
- device/cuda/src/finding/kernels/concatenate_measurements.cuh

**Files to modify:**
- examples/run/cuda/full_chain_algorithm.cpp
- device/cuda/CMakeLists.txt

---

## Validation Plan

After each elimination:

1. **Functional correctness:**
   ```bash
   ctest -R cuda -j4
   ```

2. **Performance verification:**
   ```bash
   ./build/bin/traccc_throughput_st_cuda \
     --input-directory data/odd.bak/geant4_ttbar_mu200/ \
     --batch-size 2 --events 100
   ```

3. **NSYS profiling to verify sync reduction:**
   ```bash
   nsys profile -o after_sync_elim \
     --capture-range=cudaProfilerApi \
     --trace=cuda,nvtx \
     ./build/bin/traccc_throughput_st_cuda \
       --batch-size 2 --events 20
   ```

   **Expected:** Sync time reduced from 90.5% → <50% of CUDA API time

---

## Metrics to Track

| Metric | Baseline (35e9675f) | After Task 1.2 | After Task 1.3 | After Task 1.4 | Target |
|--------|---------------------|----------------|----------------|----------------|--------|
| Time/event (ms) | 73.2 | TBD | TBD | TBD | ~60-65 |
| Speedup vs N=1 | 1.21x | TBD | TBD | TBD | 1.33-1.41x |
| Sync time (% CUDA API) | 90.5% | TBD | TBD | TBD | <50% |
| Syncs per event | ~54 | TBD | TBD | TBD | <10 |

---

## Risk Assessment

### Low Risk
- **Lines 174, 259, 304:** Simple ->wait() → ->ignore() changes
- **Validation:** Easy to verify with existing tests

### Medium Risk
- **Lines 385-386:** Requires metadata structure changes
- **Mitigation:** Keep get_size() as fallback, use metadata when available

### High Risk
- **Line 398:** New kernel development
- **Mitigation:**
  - Prototype kernel with simple test case first
  - Validate with single-event comparison
  - Use existing Thrust primitives if possible

---

## Next Steps

1. ✅ **Completed:** Synchronization audit
2. ⏳ **Day 3-5:** Begin Task 1.2 - Eliminate simple VecMem copy syncs
3. ⏳ **Day 5-7:** Task 1.3 - Eliminate get_size() syncs
4. ⏳ **Day 7-10:** Task 1.4 - Implement device-side concatenation
5. ⏳ **Day 10:** Full validation and performance measurement

**Expected Phase 1 outcome:** 1.21x → 1.33-1.41x speedup (+10-20%)
