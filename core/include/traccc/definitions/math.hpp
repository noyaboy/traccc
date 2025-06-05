/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL include(s).
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

// System include(s).
#include <cmath>

namespace traccc {

/// Namespace to pick up math functions from
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
namespace math = ::sycl;
#else
namespace math = std;
#endif  // SYCL

/// Namespace providing fast math functions on supported GPUs
#if defined(__CUDA_ARCH__)
namespace fast_math {
TRACCC_HOST_DEVICE inline float sqrt(float x) {
    return __fsqrt_rn(x);
}
TRACCC_HOST_DEVICE inline float fabs(float x) {
    return fabsf(x);
}
TRACCC_HOST_DEVICE inline float fma(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}
TRACCC_HOST_DEVICE inline float sin(float x) {
    return __sinf(x);
}
TRACCC_HOST_DEVICE inline float cos(float x) {
    return __cosf(x);
}
}  // namespace fast_math
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
namespace fast_math = ::sycl::native;
#else
namespace fast_math = std;
#endif

}  // namespace traccc
