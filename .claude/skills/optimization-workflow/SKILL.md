---
name: optimization-workflow
description: Complete optimization workflow from profiling to verification. Includes 8 phases (0-7): prerequisites, profile, analyze, plan, implement, test, benchmark, handle results. Always verify code before making changes.
---

# Optimization Workflow

Complete workflow: **Prerequisites → Profile → Analyze → Plan → Implement → Test → Benchmark → Handle Results**

(Phase 0 + Phases 1-7 = 8 phases total)

**Primary Baseline:** 59.21 events/s @ 8 threads (Tesla V100-32GB)
**Fallback Baseline:** 51.30 events/s @ 4 threads (if 8 threads OOM)

**Key Principle:** Always READ and VERIFY code before making changes. Never assume or imagine code structure.

---

## Phase 0: PREREQUISITES

### Ensure Build Exists

```bash
cd /dicos_ui_home/noah/traccc/build
ls bin/traccc_throughput_mt_cuda 2>/dev/null || echo "BUILD REQUIRED"
```

If build doesn't exist:
```bash
cd /dicos_ui_home/noah/traccc
mkdir -p build && cd build
cmake -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DTRACCC_BUILD_CUDA=ON \
      -DTRACCC_BUILD_EXAMPLES=ON \
      ..
cmake --build . -j8
```

### Verify Baseline

Before any optimization, confirm current performance matches expected baseline:

```bash
./bin/traccc_throughput_mt_cuda \
  --detector-file=../data/geometries/odd/odd-detray_geometry_detray.json \
  --material-file=../data/geometries/odd/odd-detray_material_detray.json \
  --grid-file=../data/geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=../data/geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=true \
  --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 \
  --processed-events=500 \
  --cpu-threads=4 2>&1 | grep "Event processing"
```

**Expected:** ~51.30 events/s @ 4 threads (±5% acceptable, i.e., 48.7-53.9).
If outside this range, investigate before proceeding (GPU throttling? other processes?).

### Check GPU is Idle

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Expected:** No other processes using GPU. Wait or terminate if busy.

---

## Phase 1: PROFILE

### Run Profiling

```bash
cd /dicos_ui_home/noah/traccc/build
/usr/local/cuda-12.6/bin/nsys profile --stats=true -o profile_<name> \
  ./bin/traccc_throughput_mt_cuda \
  --detector-file=../data/geometries/odd/odd-detray_geometry_detray.json \
  --material-file=../data/geometries/odd/odd-detray_material_detray.json \
  --grid-file=../data/geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=../data/geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=true \
  --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 --processed-events=100 --cpu-threads=1
```

### Extract Stats

```bash
/usr/local/cuda-12.6/bin/nsys stats profile_<name>.nsys-rep --report cuda_gpu_kern_sum
/usr/local/cuda-12.6/bin/nsys stats profile_<name>.nsys-rep --report cuda_api_sum
```

---

## Phase 2: ANALYZE

### Identify Bottlenecks

Look for:
- **GPU kernels** with highest time % (target: >10%)
- **cudaStreamSynchronize** % (sync overhead)
- **cudaMemcpyAsync** frequency (transfer overhead)

### Find Bottleneck Source Files

**Do NOT assume files** - search for the actual kernel/function name.

nsys shows mangled C++ names. Extract the base function name and search:

```bash
# Example: if nsys shows "void traccc::cuda::kernels::propagate_to_next_surface<...>"
# Search for the function definition:
grep -rn "propagate_to_next_surface" device/ --include="*.cuh" --include="*.cu" --include="*.hpp" --include="*.ipp"

# Or search for kernel launch:
grep -rn "propagate_to_next_surface<<<" device/

# Or search for TRACCC kernel definition pattern:
grep -rn "TRACCC_.*_DEVICE.*propagate_to_next_surface" device/
```

Common locations (but ALWAYS verify via grep):
- `device/cuda/src/finding/` - CKF kernels
- `device/cuda/src/seeding/` - Seeding kernels
- `device/cuda/src/clusterization/` - CCL kernels
- `device/common/include/traccc/*/device/impl/` - Kernel implementations

### Review Code (REQUIRED)

**MUST read actual source files** of the bottleneck:

Verify:
- [ ] Read the bottleneck kernel/function code
- [ ] Understand data flow and dependencies
- [ ] Identify actual sync points and their purpose
- [ ] Check memory allocation patterns
- [ ] Check what calls this code (callers)
- [ ] Check what this code calls (callees)

### Document Findings

Create `docs/benchmarks/profiling_analysis_<name>.md` with:
- Kernel time distribution table
- API time distribution table
- Identified bottlenecks with **file:line** references
- Proposed optimizations ranked by effort/impact

---

## Phase 3: PLAN

### Create Optimization Plan

Based on **verified code analysis** (not assumptions):

1. **Target** - Specific file:line of bottleneck
2. **Current behavior** - Quote actual code snippet
3. **Proposed change** - What to modify
4. **Effort** - Low/Medium/High
5. **Expected impact** - % improvement estimate
6. **Validation method** - How to verify the impact claim
7. **Rollback plan** - How to revert if it fails
8. **Files to modify** - Exact file paths found via grep

### Optimization Categories (Estimates - Must Validate)

| Category | Effort | Estimated Impact | Validation |
|----------|--------|------------------|------------|
| Sync removal | Low | 5-15% | Re-profile sync count |
| Pinned memory | Low | 5-15% | Re-profile memcpy time |
| Kernel fusion | Medium | 10-20% | Re-profile kernel count |
| Device-side logic | Medium | 20-40% | Re-profile sync count |
| Multi-event batch | High | 30-50% | Re-profile kernel launches |

**Note:** These are estimates. Always re-profile after implementation to validate actual impact.

### New CLI Options (if applicable)

If optimization requires new configuration:
- Define new CLI option name and type
- Document default value
- Add to benchmark command in Phase 6 (BENCHMARK)

---

## Phase 4: IMPLEMENT

### Create Branch

```bash
git checkout -b optimization/<name>
```

### Save Rollback Point

```bash
# Note current commit hash
git rev-parse HEAD  # Save this hash for rollback
```

### Verify Before Changing (REQUIRED)

Before modifying any file:
1. **Read the file** using Read tool
2. **Verify line numbers** match profiling analysis
3. **Understand context** around the change
4. **Check dependencies** - what else uses this code

### Implement Changes

Make changes based on verified code, not assumptions.

### Test Incrementally (for large changes)

For multi-file changes, test after each significant change:
```bash
cmake --build . -j8 && ./bin/traccc_test_cuda
# Or filter to relevant tests (check available test names first):
# ./bin/traccc_test_cuda --gtest_list_tests | grep -i <keyword>
# ./bin/traccc_test_cuda --gtest_filter="*<keyword>*"
```

### If Implementation Fails

```bash
# Option 1: Revert all uncommitted changes
git checkout -- .

# Option 2: Reset to saved commit (if already committed)
git reset --hard <saved-commit-hash>
```

**Do NOT commit until Phase 5 tests pass.**

---

## Phase 5: TEST

### Save Profile Files (if in build/)

Before clean build, move profile files out of build/:
```bash
mv build/profile_*.nsys-rep build/profile_*.sqlite /dicos_ui_home/noah/traccc/docs/benchmarks/ 2>/dev/null || true
```

### Clean Build (REQUIRED)

```bash
cd /dicos_ui_home/noah/traccc
rm -rf build && mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DTRACCC_BUILD_CUDA=ON \
      -DTRACCC_BUILD_EXAMPLES=ON \
      ..
cmake --build . -j8
```

### Run CUDA Tests (REQUIRED)

```bash
./bin/traccc_test_cuda
```

**Must pass:** All 710 tests. If tests fail:
1. **Do NOT commit**
2. **Do NOT proceed** to benchmark
3. Fix the issue or rollback (see Phase 4)
4. Re-run tests until all pass

### Run Additional Tests (if available)

```bash
# Check for other test binaries
ls bin/traccc_test_*

# Run if exist
./bin/traccc_test_core 2>/dev/null || true
./bin/traccc_test_io 2>/dev/null || true
```

### Commit ONLY After Tests Pass

```bash
git add <files>
git commit -m "perf: <description>

<details of what was optimized and why>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 6: BENCHMARK

### Check GPU is Idle

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Run 3 Times for Statistical Significance

Run each configuration **3 times** and take **median**.

**Timeout:** If benchmark hangs >5 minutes, kill it (Ctrl+C) and investigate.
Possible causes: deadlock from optimization, GPU memory issue.

**Important:** Grep for "Event processing" specifically (not warm-up):

```bash
./bin/traccc_throughput_mt_cuda \
  --detector-file=../data/geometries/odd/odd-detray_geometry_detray.json \
  --material-file=../data/geometries/odd/odd-detray_material_detray.json \
  --grid-file=../data/geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=../data/geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=true \
  --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 \
  --processed-events=500 \
  --cpu-threads=4 \
  [NEW OPTIONS IF ADDED] 2>&1 | grep "Event processing.*events/s"
```

Run 3 times, record each "Event processing" result.

### Thread Configurations

| Threads | Baseline | When to Use |
|---------|----------|-------------|
| 1 | 25.32 | Always run (most stable) |
| 4 | 51.30 | Always run (good balance) |
| 8 | 59.21 | Run if no OOM (primary target) |

**If 8 threads OOM:** Use 4 threads as primary comparison.

### Calculate Results

```
Median = middle value of 3 runs (sort and take 2nd)
Improvement = (median - baseline) / baseline * 100%
```

### Document Results

| Threads | Baseline | Run1 | Run2 | Run3 | Median | Improvement |
|---------|----------|------|------|------|--------|-------------|
| 1 | 25.32 | ? | ? | ? | ? | ?% |
| 4 | 51.30 | ? | ? | ? | ? | ?% |
| 8 | 59.21 | ? | ? | ? | ? | ?% |

### Re-Profile to Validate Impact

```bash
# Profile after optimization
/usr/local/cuda-12.6/bin/nsys profile --stats=true -o profile_<name>_after \
  ./bin/traccc_throughput_mt_cuda ... --cpu-threads=1

# Compare to Phase 1 profile
# - Did target kernel time % decrease?
# - Did sync count/time decrease?
# - Did the expected improvement materialize?
```

---

## Phase 7: HANDLE RESULTS

### If Improvement Achieved

1. Amend commit with benchmark results (if not pushed)
2. Update `docs/benchmarks/` with new analysis
3. Proceed with PR or merge

### If No Improvement (Neutral)

1. Decide: Is the change still valuable? (cleaner code, future optimization prep)
2. If yes: Change commit message from "perf:" to "refactor:"
3. If no: Revert the commit

### If Regression (Worse Performance)

1. **Do NOT merge/push**
2. Investigate why:
   - Re-read code - did you misunderstand something?
   - Profile again - what got slower?
   - Check test results - any subtle failures?
3. Options:
   - Fix the implementation
   - Revert and try different approach
   - Abandon this optimization path

```bash
# To revert unpushed commit
git reset --soft HEAD~1  # Keep changes, undo commit
# or
git reset --hard HEAD~1  # Discard changes entirely
```

---

## Verification Checklist

### Phase 0: Prerequisites
- [ ] Build exists and works
- [ ] Baseline verified (~51.30 @ 4 threads or ~59.21 @ 8 threads)
- [ ] GPU is idle

### Phase 1-2: Profile & Analyze
- [ ] Profiling data collected
- [ ] Bottleneck files found via grep (not assumed)
- [ ] **Code actually read** (not imagined)
- [ ] Bottlenecks documented with file:line

### Phase 3: Plan
- [ ] Plan includes validation method
- [ ] Plan includes rollback strategy

### Phase 4: Implement
- [ ] Rollback point saved (commit hash noted)
- [ ] Code verified before modification
- [ ] Large changes tested incrementally

### Phase 5: Test
- [ ] Profile files saved before clean build
- [ ] Clean build succeeds
- [ ] All 710 CUDA tests pass
- [ ] **Commit made AFTER tests pass**

### Phase 6: Benchmark
- [ ] GPU idle before benchmark
- [ ] Ran 3 times, took median
- [ ] Used "Event processing" line (not warm-up)
- [ ] Re-profiled to validate impact claim

### Phase 7: Results
- [ ] Results handled appropriately (improve/neutral/regress)

---

## Current Bottlenecks

| Component | Time % | Location |
|-----------|--------|----------|
| propagate_to_next_surface | 64.3% | `combinatorial_kalman_filter.cuh:509-516` |
| cudaStreamSynchronize | 80.9% | Multiple locations |

## Reference

- `cuda-throughput-benchmark` skill for baseline details
- `docs/benchmarks/profiling_analysis_*.md` for past analyses
