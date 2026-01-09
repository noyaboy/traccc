# TRACCC FPGA RK4 Propagator - HLS Implementation

This directory contains a standalone HLS-ready implementation of the RK4 propagation algorithm, extracted from detray's `rk_stepper.ipp` for synthesis on AMD Alveo V80.

## Files

| File | Purpose |
|------|---------|
| `rk4_propagator.hpp` | Header with data structures and template implementations |
| `rk4_propagator.cpp` | HLS kernel implementations |
| `rk4_propagator_tb.cpp` | C-simulation testbench |
| `Makefile` | Build system for C-sim and HLS synthesis |

## Quick Start

```bash
# C-simulation (no Vitis HLS required)
make csim

# HLS synthesis (requires Vitis HLS 2024.x)
make hls PART=xcv80 CLOCK_PERIOD=2.0

# View synthesis report
make report
```

## Algorithm Overview

The RK4 (Runge-Kutta-Nystrom 4th order) integration is used for charged particle propagation through a magnetic field. The algorithm follows the equations from [Nuclear Instruments and Methods 1981](https://doi.org/10.1016/0029-554X(81)90063-X).

### Core Functions

1. **`evaluate_dtds(t, B, qop)`** - Lorentz force: `dtds = qop * (t × B)`
2. **`rk4_step()`** - Single adaptive RK4 step with error estimation
3. **`propagate_to_surface()`** - Multi-step propagation to target distance

### Data Structures

```cpp
struct TrackParams {     // 24 bytes - GPU↔FPGA transfer
    float loc0, loc1;    // Local position
    float phi, theta;    // Direction angles
    float qop;           // Charge/momentum
    float time;          // Time coordinate
};

struct FreeTrackState {  // Internal propagation state
    Vec3 pos, dir;       // Position and direction
    float qop, time;
    float path_length;
};
```

## HLS Pragmas

The implementation includes Vitis HLS pragmas for:
- **PIPELINE II=1** - Fully pipelined RK4 step execution
- **LOOP_TRIPCOUNT** - Trip count hints for timing estimation
- **INTERFACE m_axi** - AXI memory-mapped interfaces for HBM2e access

## Unit System

The implementation uses detray/ACTS native units:
- **Length**: millimeters (mm)
- **Energy/Momentum**: GeV
- **Charge**: units of elementary charge (e)
- **Magnetic Field**: GeV/(e·mm) internally

The `ConstBField` constructor accepts Tesla values and converts internally:
```cpp
// 1 Tesla = 0.000299792458 GeV/(e·mm)
constexpr float UNIT_T = 0.000299792458f;
ConstBField bfield(0.0f, 0.0f, 2.0f);  // 2 Tesla in z-direction
```

The helix radius formula in these units:
```
r (mm) = p (GeV) / (|q| × B (Tesla) × UNIT_T)
       = p / (0.000299792458 × B)
```

## Current Status

### Working
- Straight-line propagation (B=0) verified
- Circular motion with correct helix radius (0.08% error vs analytical)
- Adaptive step size with error estimation
- Direction normalization preserved
- Batch processing of 6666 tracks
- Unit conversion aligned with detray/ACTS conventions

### Known Limitations
- B-field grid interpolation not yet tested with real data
- No material effects (energy loss, scattering)
- Surface intersection simplified (assumes planar surfaces)

## Next Steps

1. **Run HLS synthesis** to get DSP58/LUT/timing estimates
2. **Create test vectors** from GPU reference runs
3. **Deploy to V80** for Phase 4 validation

## Source Reference

Extracted from:
- `detray/core/include/detray/propagator/rk_stepper.hpp`
- `detray/core/include/detray/propagator/rk_stepper.ipp`

Original algorithm reference:
- L. Bugge, J. Myrheim, "A Fast Runge-Kutta Method for Fitting Tracks in a Magnetic Field", Nuclear Instruments and Methods 179 (1981) 365-381
