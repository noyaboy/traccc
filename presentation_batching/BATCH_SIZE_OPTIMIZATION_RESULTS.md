# Batch Size Optimization Results

## Summary

Benchmark testing to find the optimal batch size for the batched CKF algorithm
on geant4_ttbar_mu200 dataset using NVIDIA Tesla V100-SXM2-32GB GPU.

**Optimal Batch Size: 24**
- Average throughput: 27.67 events/s
- Improvement over no batching: **+77.3%**
- Improvement over batch-14: +5.0%
- Most stable performance (lowest variance)

## Test Configuration

- **GPU:** NVIDIA Tesla V100-SXM2-32GB
- **CUDA Version:** 12.6
- **Dataset:** geant4_ttbar_mu200 (36 events)
- **Detector:** ODD (Open Data Detector)
- **Magnetic Field:** Inhomogeneous (odd-bfield.cvf)

## Reproduction Commands

### With Batching (batch-24)
```bash
./build/bin/traccc_throughput_st_cuda \
    --batch-size 24 \
    --use-batched-api 1 \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --input-directory=odd/geant4_ttbar_mu200 \
    --input-events=36 \
    --read-bfield-from-file \
    --bfield-file=geometries/odd/odd-bfield.cvf \
    --cold-run-events 24 \
    --processed-events 96 \
    -qqq
```

### Without Batching (baseline at commit 5cd477ac)
```bash
./build/bin/traccc_throughput_st_cuda \
    --detector-file=geometries/odd/odd-detray_geometry_detray.json \
    --input-directory=odd/geant4_ttbar_mu200 \
    --input-events=36 \
    --read-bfield-from-file \
    --bfield-file=geometries/odd/odd-bfield.cvf \
    --cold-run-events 24 \
    --processed-events 96 \
    -qqq
```

## Batching vs No Batching

| Configuration | Throughput (ev/s) | Time/Event (ms) | Improvement |
|---------------|-------------------|-----------------|-------------|
| No batching (5cd477ac) | 15.61 | 64.09 | baseline |
| **Batch-24 (develop)** | **27.67** | **36.15** | **+77.3%** |

### No Batching Baseline (5 runs at commit 5cd477ac)

| Run | Throughput (ev/s) | Time/Event (ms) |
|-----|-------------------|-----------------|
| 1   | 15.33             | 65.22           |
| 2   | 15.96             | 62.64           |
| 3   | 15.76             | 63.44           |
| 4   | 15.72             | 63.60           |
| 5   | 15.26             | 65.54           |
| **Avg** | **15.61**     | **64.09**       |

## Batch Size Comparison

| Batch Size | Throughput (ev/s) | Time/Event (ms) | vs batch-14 |
|------------|-------------------|-----------------|-------------|
| 14 (prev)  | 26.35             | 37.95           | baseline    |
| 16         | 26.84             | 37.25           | +1.9%       |
| 20         | 27.18             | 36.79           | +3.2%       |
| **24**     | **27.67**         | **36.15**       | **+5.0%**   |
| 28         | 25.79             | 38.78           | -2.1%       |
| 32         | 27.57             | 36.32           | +4.6%       |
| 36         | 26.93             | 37.13           | +2.2%       |

## Repeated Test Results

### Batch-24 (5 runs)

| Run | Throughput (ev/s) | Time/Event (ms) |
|-----|-------------------|-----------------|
| 1   | 27.49             | 36.38           |
| 2   | 28.09             | 35.60           |
| 3   | 27.83             | 35.93           |
| 4   | 27.60             | 36.23           |
| 5   | 27.32             | 36.60           |
| **Avg** | **27.67**     | **36.15**       |
| Std Dev | +/-0.29       | -               |

### Batch-32 (5 runs)

| Run | Throughput (ev/s) | Time/Event (ms) |
|-----|-------------------|-----------------|
| 1   | 27.57             | 36.28           |
| 2   | 27.27             | 36.67           |
| 3   | 28.66             | 34.90           |
| 4   | 25.91             | 38.59           |
| 5   | 28.46             | 35.14           |
| **Avg** | **27.57**     | **36.32**       |
| Std Dev | +/-1.08       | -               |

## Analysis

1. **Batching provides 77% speedup** over non-batched baseline:
   - No batching: 15.61 ev/s (64.09 ms/event)
   - Batch-24: 27.67 ev/s (36.15 ms/event)

2. **Batch-24 is optimal** for this workload:
   - Highest average throughput (27.67 ev/s)
   - Most consistent results (std dev +/-0.29 vs +/-1.08 for batch-32)
   - Lower memory usage than larger batch sizes

3. **Performance trend:**
   - Throughput increases from batch-14 to batch-24
   - Dip at batch-28 (possibly due to memory alignment)
   - Recovery at batch-32 but with higher variance
   - Decline again at batch-36

4. **Recommendation:**
   - Use batch-24 for production workloads on V100
   - Provides best balance of performance and stability
   - 77% improvement over non-batched processing

## Hardware Details

```
GPU: Tesla V100-SXM2-32GB
CUDA: 12.6.77
Memory: 32GB HBM2
SM Count: 80
Architecture: Volta (sm_70)
```

## Unit Test Results

All 5 test binaries pass on the develop branch:

| Test Binary | Tests | Status |
|-------------|-------|--------|
| traccc_test_core | 24 | PASSED |
| traccc_test_examples | 3 | PASSED |
| traccc_test_io | 9 | PASSED |
| traccc_test_cpu | 714 | PASSED |
| traccc_test_cuda | 710 | PASSED |
| **Total** | **1460** | **ALL PASSED** |
