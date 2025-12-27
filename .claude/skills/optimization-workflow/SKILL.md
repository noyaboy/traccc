---
name: optimization-workflow
description: Complete optimization workflow from profiling to verification. Includes profile, analyze, plan, implement, test, and benchmark phases.
---

# Optimization Workflow

Complete workflow: **Profile → Analyze → Plan → Implement → Test → Benchmark**

---

## Phase 1: PROFILE

Run profiling to collect performance data (see `cuda-throughput-benchmark` skill for full paths):

```bash
cd /dicos_ui_home/noah/traccc/build
/usr/local/cuda-12.6/bin/nsys profile --stats=true -o profile_<name> \
  ./bin/traccc_throughput_mt_cuda \
  --input-directory=../data/odd/geant4_ttbar_mu200/ \
  --input-events=36 --processed-events=100 --cpu-threads=1 \
  [other args from cuda-throughput-benchmark skill]
```

Extract stats:
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

### Review Relevant Code

Key files for CKF optimization:
- `device/cuda/src/finding/combinatorial_kalman_filter.cuh`
- `device/common/include/traccc/finding/device/impl/propagate_to_next_surface.ipp`
- `device/common/include/traccc/finding/device/impl/find_tracks.ipp`

### Document Findings

Create `docs/benchmarks/profiling_analysis_<name>.md` with:
- Kernel time distribution table
- API time distribution table
- Identified bottlenecks
- Proposed optimizations ranked by effort/impact

---

## Phase 3: PLAN

Based on analysis, create optimization plan:

1. **Identify target** - Which kernel/API to optimize
2. **Propose solution** - What change to make
3. **Estimate effort** - Low/Medium/High
4. **Estimate impact** - Expected improvement %
5. **List files to modify** - Specific files and functions

### Optimization Categories

| Category | Effort | Impact | Examples |
|----------|--------|--------|----------|
| Sync removal | Low | 5-15% | Remove unnecessary `str.synchronize()` |
| Pinned memory | Low | 5-15% | Use `cudaMallocHost` for staging buffers |
| Kernel fusion | Medium | 10-20% | Combine sort + propagate |
| Device-side logic | Medium | 20-40% | Keep counts on GPU, avoid D2H sync |
| Multi-event batch | High | 30-50% | Process multiple events per kernel |

---

## Phase 4: IMPLEMENT

### Create Branch

```bash
git checkout -b optimization/<name>
```

### Implement Changes

Follow the plan from Phase 3. Common patterns:

**Sync Removal:**
```cpp
// BEFORE
kernel<<<...>>>();
str.synchronize();  // Remove if not needed

// AFTER
kernel<<<...>>>();
// No sync - stream ordering handles dependency
```

**Pinned Memory:**
```cpp
// BEFORE
vecmem::make_unique_alloc<T>(*(mr.host));

// AFTER
vecmem::make_unique_alloc<T>(*(mr.host));  // Ensure mr.host is cuda::host_memory_resource
```

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

Use commands from `cuda-throughput-benchmark` skill.

### Quick Benchmark (1 thread)

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
  --cpu-threads=1
```

**Baseline:** 25.32 events/s

### Full Benchmark (1, 4, 8 threads)

Run with `--cpu-threads=1`, `--cpu-threads=4`, `--cpu-threads=8`

**Baselines:**
| Threads | events/s |
|---------|----------|
| 1 | 25.32 |
| 4 | 51.30 |
| 8 | 59.21 |

### Compare Results

```
Improvement = (new - baseline) / baseline * 100%
```

---

## Verification Checklist

- [ ] Profiling completed and documented
- [ ] Optimization plan created
- [ ] Implementation complete
- [ ] Clean build succeeds
- [ ] All 710 CUDA tests pass
- [ ] Benchmark shows improvement (or no regression)
- [ ] Results documented in commit message

---

## Current Bottlenecks

| Component | Time % | Notes |
|-----------|--------|-------|
| propagate_to_next_surface | 64.3% | Runge-Kutta physics |
| cudaStreamSynchronize | 80.9% | Sync overhead |

## Reference

- See `cuda-throughput-benchmark` skill for all benchmark commands
- `docs/benchmarks/profiling_analysis_*.md` for past analyses
