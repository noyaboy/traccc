/** KalmanNet-inspired GRU-based gain predictor
 *
 * Two stacked GRU layers (hidden = 64) → FC (64 × 25) producing the
 * flattened 6 × D Kalman gain matrix.  Weights are initialised
 * deterministically but effectively random.
 */

#pragma once

#include <cmath>        // std::exp / std::tanh
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
    static constexpr size_type HiddenSize = 64;         // GRU width
    static constexpr size_type OutputSize = 6 * D;      // flattened K

    /*──────────────────  deterministic pseudo-random helper  ─────────────────*/
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

    /*───────────────────────────  constructor  ───────────────────────────────*/
    TRACCC_HOST_DEVICE
    kalman_gru_gain_predictor() {
        for (std::size_t i = 0; i < HiddenSize * InputSize;  ++i) W0_[i]    = rnd(i);
        for (std::size_t i = 0; i < HiddenSize * HiddenSize; ++i) {
            U0_[i]  = rnd(i + 17);
            W1_[i]  = rnd(i + 31);
            U1_[i]  = rnd(i + 47);
        }
        for (std::size_t i = 0; i < HiddenSize; ++i) {
            b0_[i]  = rnd(i + 97);
            b1_[i]  = rnd(i + 131);
        }
        for (std::size_t i = 0; i < OutputSize * HiddenSize; ++i) W_out_[i] = rnd(i + 197);
        for (std::size_t i = 0; i < OutputSize; ++i)            b_out_[i]  = rnd(i + 337);
    }

    /*────────────────────────────  forward  ──────────────────────────────────*/
    template <typename vec6_t>
    TRACCC_HOST_DEVICE
    inline matrix_type<6, D> operator()(const vec6_t& x) const {

        scalar h0[HiddenSize]{};
        scalar h1[HiddenSize]{};

        /*─ GRU-0 (simplified) ─*/
        for (size_type i = 0; i < HiddenSize; ++i) {
            scalar acc = b0_[i];
            /* 來源向量實際型別為 std::array<std::array<scalar, 6>, 1>，
             * 因此外層僅有索引 0；再取第 j 個元素。                 */
            for (size_type j = 0; j < InputSize; ++j)
                acc += W0_[i * InputSize + j] * x[0][j];
            h0[i] = std::tanh(acc);
        }

        /*─ GRU-1 (simplified) ─*/
        for (size_type i = 0; i < HiddenSize; ++i) {
            scalar acc = b1_[i];
            for (size_type j = 0; j < HiddenSize; ++j)
                acc += W1_[i * HiddenSize + j] * h0[j];
            h1[i] = std::tanh(acc);
        }

        /*─ Dense output → 6 × D gain matrix ─*/
        matrix_type<6, D> K{};
        for (size_type r = 0; r < 6; ++r)
            for (size_type c = 0; c < D; ++c) {
                const size_type o = r * D + c;
                scalar acc = b_out_[o];
                for (size_type j = 0; j < HiddenSize; ++j)
                    acc += W_out_[o * HiddenSize + j] * h1[j];
                K(r, c) = acc;
            }

        return K;
    }

  private:
    /*─ layer weights (flat arrays) ─*/
    scalar W0_[HiddenSize * InputSize];
    [[maybe_unused]] scalar U0_[HiddenSize * HiddenSize];
    scalar b0_[HiddenSize];

    scalar W1_[HiddenSize * HiddenSize];
    [[maybe_unused]] scalar U1_[HiddenSize * HiddenSize];
    scalar b1_[HiddenSize];

    scalar W_out_[OutputSize * HiddenSize];
    scalar b_out_[OutputSize];
};

} // namespace traccc::fitting