#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)

#include <cuda_fp16.h>
#include <mma.h>

#include "traccc/definitions/primitives.hpp"

namespace traccc::cuda::utils {

namespace wmma = nvcuda::wmma;

/// Multiply two small matrices using WMMA. The matrices are expected to
/// have compile-time row/column counts given by their types.
/// The result matrix type is deduced from the template arguments.
/// The operation assumes row-major storage.

template <typename algebra_t, detray::dsize_type<algebra_t> M,
          detray::dsize_type<algebra_t> K, detray::dsize_type<algebra_t> N>
__device__ inline detray::dmatrix<algebra_t, M, N> wmma_multiply(
    const detray::dmatrix<algebra_t, M, K>& A,
    const detray::dmatrix<algebra_t, K, N>& B) {
    constexpr int TILE = 16;
    __shared__ half Ah[TILE * TILE];
    __shared__ half Bh[TILE * TILE];
    __shared__ float Ch[TILE * TILE];

    for (int idx = 0; idx < TILE * TILE; ++idx) {
        Ah[idx] = 0;
        Bh[idx] = 0;
        Ch[idx] = 0.0f;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            Ah[i * TILE + j] = __float2half(getter::element(A, i, j));
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            Bh[i * TILE + j] = __float2half(getter::element(B, i, j));
        }
    }

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;

    wmma::load_matrix_sync(a_frag, Ah, TILE);
    wmma::load_matrix_sync(b_frag, Bh, TILE);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(Ch, c_frag, TILE, wmma::mem_row_major);

    detray::dmatrix<algebra_t, M, N> C;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {

            getter::element(C, i, j) = Ch[i * TILE + j];

        }
    }
    return C;
}

}  // namespace traccc::cuda::utils

#endif  // __CUDA_ARCH__
