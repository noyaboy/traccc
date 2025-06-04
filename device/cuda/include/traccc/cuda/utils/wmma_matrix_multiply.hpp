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



    // Fill the fragments with zeros
    wmma::fill_fragment(a_frag, static_cast<half>(0));
    wmma::fill_fragment(b_frag, static_cast<half>(0));
    wmma::fill_fragment(c_frag, 0.0f);

    // Copy input matrices into the fragments
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            a_frag.x[i * TILE + j] =
                __float2half(getter::element(A, i, j));
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            b_frag.x[i * TILE + j] =
                __float2half(getter::element(B, i, j));
        }
    }

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    detray::dmatrix<algebra_t, M, N> C;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            getter::element(C, i, j) = c_frag.x[i * TILE + j];
        }
    }
    return C;
}

}  // namespace traccc::cuda::utils

#endif  // __CUDA_ARCH__
