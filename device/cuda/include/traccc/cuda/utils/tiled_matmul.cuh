/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::cuda::details {

/// Simple thread block-level tiled matrix multiplication for small matrices.
/// Each block is expected to launch with M x N threads.
///
/// @tparam M Number of rows of A and C
/// @tparam K Number of columns of A / rows of B
/// @tparam N Number of columns of B and C
template <unsigned int M, unsigned int K, unsigned int N, typename matrix_a_t,
          typename matrix_b_t, typename matrix_c_t>
__device__ inline void tiled_matmul(const matrix_a_t& A, const matrix_b_t& B,
                                    matrix_c_t& C) {
    static_assert(M <= 16 && K <= 16 && N <= 16,
                  "tiled_matmul only supports small matrices");

    // Shared memory tiles
    __shared__ traccc::scalar sA[M][K];
    __shared__ traccc::scalar sB[K][N];

    const unsigned int row = threadIdx.y;
    const unsigned int col = threadIdx.x;

    if (row < M && col < K) {
        sA[row][col] = getter::element(A, row, col);
    }
    if (row < K && col < N) {
        sB[row][col] = getter::element(B, row, col);
    }
    __syncthreads();

    if (row < M && col < N) {
        traccc::scalar acc = 0;
        for (unsigned int i = 0; i < K; ++i) {
            acc += sA[row][i] * sB[i][col];
        }
        getter::element(C, row, col) = acc;
    }
}

}  // namespace traccc::cuda::details
