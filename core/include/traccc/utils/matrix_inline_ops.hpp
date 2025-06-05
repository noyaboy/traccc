#pragma once

#include "traccc/definitions/hints.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::detail {

// Generic small matrix multiplication with loop unrolling
template <typename algebra_t, detray::dsize_type<algebra_t> M,
          detray::dsize_type<algebra_t> N, detray::dsize_type<algebra_t> P>
TRACCC_HOST_DEVICE inline auto matmul_inline(
    const detray::dmatrix<algebra_t, M, N>& A,
    const detray::dmatrix<algebra_t, N, P>& B)
    -> detray::dmatrix<algebra_t, M, P> {
    auto C = matrix::zero<detray::dmatrix<algebra_t, M, P>>();
    for (detray::dsize_type<algebra_t> i = 0u; i < M; ++i) {
        for (detray::dsize_type<algebra_t> j = 0u; j < P; ++j) {
            traccc::scalar sum = 0.f;
            TRACCC_PRAGMA_UNROLL
            for (detray::dsize_type<algebra_t> k = 0u; k < N; ++k) {
                sum += getter::element(A, i, k) * getter::element(B, k, j);
            }
            getter::element(C, i, j) = sum;
        }
    }
    return C;
}

// Matrix-vector multiplication
template <typename algebra_t, detray::dsize_type<algebra_t> M,
          detray::dsize_type<algebra_t> N>
TRACCC_HOST_DEVICE inline auto matvec_inline(
    const detray::dmatrix<algebra_t, M, N>& A,
    const detray::dmatrix<algebra_t, N, 1>& x)
    -> detray::dmatrix<algebra_t, M, 1> {
    auto y = matrix::zero<detray::dmatrix<algebra_t, M, 1>>();
    for (detray::dsize_type<algebra_t> i = 0u; i < M; ++i) {
        traccc::scalar sum = 0.f;
        TRACCC_PRAGMA_UNROLL
        for (detray::dsize_type<algebra_t> j = 0u; j < N; ++j) {
            sum += getter::element(A, i, j) * getter::element(x, j, 0u);
        }
        getter::element(y, i, 0u) = sum;
    }
    return y;
}

// Element-wise addition
template <typename algebra_t, detray::dsize_type<algebra_t> M,
          detray::dsize_type<algebra_t> N>
TRACCC_HOST_DEVICE inline auto add_inline(
    const detray::dmatrix<algebra_t, M, N>& A,
    const detray::dmatrix<algebra_t, M, N>& B)
    -> detray::dmatrix<algebra_t, M, N> {
    auto C = matrix::zero<detray::dmatrix<algebra_t, M, N>>();
    for (detray::dsize_type<algebra_t> i = 0u; i < M; ++i) {
        TRACCC_PRAGMA_UNROLL
        for (detray::dsize_type<algebra_t> j = 0u; j < N; ++j) {
            getter::element(C, i, j) =
                getter::element(A, i, j) + getter::element(B, i, j);
        }
    }
    return C;
}

// Element-wise subtraction
template <typename algebra_t, detray::dsize_type<algebra_t> M,
          detray::dsize_type<algebra_t> N>
TRACCC_HOST_DEVICE inline auto sub_inline(
    const detray::dmatrix<algebra_t, M, N>& A,
    const detray::dmatrix<algebra_t, M, N>& B)
    -> detray::dmatrix<algebra_t, M, N> {
    auto C = matrix::zero<detray::dmatrix<algebra_t, M, N>>();
    for (detray::dsize_type<algebra_t> i = 0u; i < M; ++i) {
        TRACCC_PRAGMA_UNROLL
        for (detray::dsize_type<algebra_t> j = 0u; j < N; ++j) {
            getter::element(C, i, j) =
                getter::element(A, i, j) - getter::element(B, i, j);
        }
    }
    return C;
}

}  // namespace traccc::detail
