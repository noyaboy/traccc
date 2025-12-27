---
name: optimization-workflow
description: Complete optimization workflow from profiling to verification. Includes profile, analyze, plan, implement, test, and benchmark phases. Always verify code before making changes.
---

# Optimization Workflow

Complete workflow: **Profile → Analyze → Plan → Implement → Test → Benchmark**

**Primary Baseline:** 59.21 events/s @ 8 threads (Tesla V100-32GB)

**Key Principle:** Always READ and VERIFY code before making changes. Never assume or imagine code structure.

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

### Review Code (REQUIRED)

**MUST read actual code files** - never assume structure:

```bash
# Key files to read
device/cuda/src/finding/combinatorial_kalman_filter.cuh
device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp
device/common/include/traccc/finding/device/impl/find_tracks.ipp
```

Verify:
- [ ] Read the bottleneck kernel/function code
- [ ] Understand data flow and dependencies
- [ ] Identify actual sync points and their purpose
- [ ] Check memory allocation patterns

### Document Findings

Create `docs/benchmarks/profiling_analysis_<name>.md` with:
- Kernel time distribution table
- API time distribution table
- Identified bottlenecks with **line numbers**
- Proposed optimizations ranked by effort/impact

---

## Phase 3: PLAN

### Create Optimization Plan

Based on **verified code analysis** (not assumptions):

1. **Target** - Specific kernel/function and line numbers
2. **Current behavior** - What the code actually does (quote code)
3. **Proposed change** - What to modify
4. **Effort** - Low/Medium/High
5. **Expected impact** - % improvement estimate
6. **Files to modify** - Exact file paths

### Optimization Categories

| Category | Effort | Impact | Examples |
|----------|--------|--------|----------|
| Sync removal | Low | 5-15% | Remove unnecessary `str.synchronize()` |
| Pinned memory | Low | 5-15% | Use `cuda::host_memory_resource` |
| Kernel fusion | Medium | 10-20% | Combine sort + propagate |
| Device-side logic | Medium | 20-40% | Keep counts on GPU |
| Multi-event batch | High | 30-50% | Process multiple events per kernel |

### New CLI Options (if applicable)

If optimization requires new configuration:
- Define new CLI option name and type
- Document default value
- Add to benchmark command in Phase 6

---

## Phase 4: IMPLEMENT

### Create Branch

```bash
git checkout -b optimization/<name>
```

### Verify Before Changing (REQUIRED)

Before modifying any file:
1. **Read the file** using Read tool
2. **Verify line numbers** match profiling analysis
3. **Understand context** around the change
4. **Check dependencies** - what else uses this code

### Implement Changes

Make changes based on verified code, not assumptions.

### Commit Changes

```bash
git add <files>
git commit -m "perf: <description>

<details of what was optimized and why>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: TEST

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

**Must pass:** All 710 tests. If tests fail, fix before proceeding.

---

## Phase 6: BENCHMARK

### Benchmark Command

Base command (adjust `--cpu-threads` and add new options if implemented):

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
  --cpu-threads=8 \
  [NEW OPTIONS IF ADDED BY IMPLEMENTATION]
```

### Add New CLI Options

If implementation added new options, include them:
```bash
  --new-option=value \
```

### Baselines (v1.0.0)

| Threads | events/s |
|---------|----------|
| 1 | 25.32 |
| 4 | 51.30 |
| **8** | **59.21** ← Primary comparison |

### Calculate Improvement

```
Improvement = (new_result - 59.21) / 59.21 * 100%
```

**Target:** Beat 59.21 events/s @ 8 threads

### Document Results

| Threads | Baseline | Optimized | Improvement |
|---------|----------|-----------|-------------|
| 1 | 25.32 | ? | ?% |
| 4 | 51.30 | ? | ?% |
| 8 | 59.21 | ? | ?% |

---

## Verification Checklist

- [ ] Phase 1: Profiling data collected
- [ ] Phase 2: **Code actually read** (not imagined)
- [ ] Phase 2: Bottlenecks documented with line numbers
- [ ] Phase 3: Plan based on verified code
- [ ] Phase 4: Code verified before modification
- [ ] Phase 5: Clean build succeeds
- [ ] Phase 5: All 710 CUDA tests pass
- [ ] Phase 6: Benchmark beats 59.21 events/s @ 8 threads
- [ ] Phase 6: Results documented in commit

---

## Current Bottlenecks

| Component | Time % | Location |
|-----------|--------|----------|
| propagate_to_next_surface | 64.3% | `combinatorial_kalman_filter.cuh:509-516` |
| cudaStreamSynchronize | 80.9% | Multiple locations |

## Reference

- `cuda-throughput-benchmark` skill for baseline details
- `docs/benchmarks/profiling_analysis_*.md` for past analyses
