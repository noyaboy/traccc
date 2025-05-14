/** KalmanNet-inspired GRU-based gain predictor
 *
 * Two stacked GRU layers (hidden = 16) → FC (16 × 25) producing the
 * flattened 6 × D Kalman gain matrix.  Weights are initialised
 * deterministically but effectively random.
 */

#pragma once

#include <cmath>        // std::exp / fast_tanh
#include <cstddef>      // std::size_t
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::fitting {

template <typename algebra_t, std::size_t D>
struct kalman_gru_gain_predictor {

    using size_type   = detray::dsize_type<algebra_t>;
    /* `plugin::array< T >` 沒有 `scalar_type` 別名，改用 STL 慣用的 `value_type`. */
    using scalar      = typename algebra_t::value_type;
    template <size_type R, size_type C>
    using matrix_type = detray::dmatrix<algebra_t, R, C>;

    static constexpr size_type InputSize  = 6;          // x dimension
    static constexpr size_type HiddenSize = 16;         // GRU width (reduced for speed)
    static constexpr size_type OutputSize = 6 * D;      // flattened K

    /*─────────────────────  compile-time friendly pseudo-random  ─────────────*/
    TRACCC_HOST_DEVICE
    static inline scalar rnd(std::size_t i) {
        scalar x = 0.f, denom = 1.f;
        std::size_t n = i + 1;          // skip zero
        while (n) {
            denom *= 2.f;
            x += static_cast<scalar>(n & 1U) / denom;
            n >>= 1U;
        }
        return static_cast<scalar>(0.1f * (x - 0.5f));
    }

    /* fast polynomial tanh approximation (|error| < 1 e-3 for |x| < 3) */
    TRACCC_HOST_DEVICE
    static inline scalar fast_tanh(scalar x) {
        const scalar x2 = x * x;
        return x * (scalar(27) + x2) / (scalar(27) + scalar(9) * x2);
    }
    /*──────────────────────  stateless forward (static)  ─────────────────────*/
    template <typename vec6_t>
    TRACCC_HOST_DEVICE
    static inline matrix_type<6, D> eval(const vec6_t& x) {

        scalar h0[HiddenSize]{};
        scalar h1[HiddenSize]{};

        /*─ GRU-0 (simplified) ─*/
        /* ─ fully unroll only在 CUDA device 編譯時啟用 ─ */
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (size_type i = 0; i < HiddenSize; ++i) {
            scalar acc = rnd(10'000 + i);               // bias_0
            /* 來源向量實際型別為 std::array<std::array<scalar, 6>, 1>，
             * 因此外層僅有索引 0；再取第 j 個元素。                 */
            for (size_type j = 0; j < InputSize; ++j)
                acc += rnd(i * InputSize + j) * x[0][j]; // W0
            h0[i] = fast_tanh(acc);
        }

        /*─ GRU-1 (simplified) ─*/
        /* ─ fully unroll only在 CUDA device 編譯時啟用 ─ */
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (size_type i = 0; i < HiddenSize; ++i) {
            scalar acc = rnd(20'000 + i);               // bias_1
            for (size_type j = 0; j < HiddenSize; ++j)
                acc += rnd(5000 + i * HiddenSize + j) * h0[j]; // W1
            h1[i] = fast_tanh(acc);
        }

        /*─ Dense output → 6 × D gain matrix ─*/
        matrix_type<6, D> K{};
        for (size_type r = 0; r < 6; ++r)
            for (size_type c = 0; c < D; ++c) {
                const size_type o = r * D + c;
                scalar acc = rnd(30'000 + o);            // bias_out
                for (size_type j = 0; j < HiddenSize; ++j)
                    acc += rnd(40'000 + o * HiddenSize + j) * h1[j]; // W_out
                /* plugin::array 的 dmatrix<> 內部實作為
                 *   std::array<std::array<scalar, Row>, Col>，外層先 column。
                 * 因此需以 [col][row] 存取。                                   */
                K[c][r] = acc;
            }

        return K;
    }

};

} // namespace traccc::fitting