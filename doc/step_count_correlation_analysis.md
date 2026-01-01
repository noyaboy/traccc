# Step Count Correlation Analysis

## Objective
Investigate whether track parameters (theta, eta, qop) can predict propagation step count,
enabling warp-based work specialization to reduce GPU divergence.

## Dataset
- **Detector**: ODD (OpenDataDetector)
- **Events**: geant4_ttbar_mu200 (10 events)
- **Samples**: 1,057,005 track propagations

## Step Count Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 1-5 steps | 509,896 | 48.24% |
| 6-10 steps | 397,132 | 37.57% |
| 11-20 steps | 147,139 | 13.92% |
| 21-34 steps | 2,838 | 0.27% |

Mean: 6.32 steps, Max: 34 steps

## Pearson Correlation Results

| Predictor | Correlation (r) | Viable? |
|-----------|-----------------|---------|
| theta | 0.0017 | No |
| eta | -0.0023 | No |
| **\|eta\|** | **-0.2028** | No (below 0.3 threshold) |
| qop | 0.0014 | No |
| \|qop\| | -0.0225 | No |

**Best predictor**: |eta| with r = -0.2028 (negative correlation: higher |eta| = fewer steps)

## Step Count by |eta| Region

| Region | Mean Steps | % with >10 steps | Sample Count |
|--------|------------|------------------|--------------|
| Central (\|eta\| < 1.0) | 7.24 | 24.7% | 202,783 |
| Transition (1.0 ≤ \|eta\| < 2.0) | 6.99 | 17.9% | 317,281 |
| Forward (\|eta\| ≥ 2.0) | 5.58 | 7.9% | 536,941 |

## Conclusion

**No viable predictor found for warp specialization.**

While |eta| shows a weak negative correlation (r = -0.2028) with step count - meaning
central tracks require more propagation steps than forward tracks - the correlation
is below the r ≥ 0.3 threshold needed for reliable prediction.

The step count is primarily determined by:
1. **Detector geometry** between surfaces (distance, material distribution)
2. **Magnetic field configuration** encountered along the path
3. **Numerical convergence** requirements of the RK4 stepper

These factors are not predictable from initial track parameters alone.

## Implications

Warp specialization based on |eta| is not viable because:
- 25% of central tracks and 8% of forward tracks exceed 10 steps
- Within each |eta| bin, step variance remains high (e.g., central: 1-26 steps)
- Misclassification rate would be too high for effective work distribution

Alternative approaches to reduce warp divergence should focus on:
- Persistent thread pools (track-to-warp work stealing)
- Sorting tracks by observed step count from previous CKF iterations
- Geometry-aware step prediction (requires detector knowledge)
