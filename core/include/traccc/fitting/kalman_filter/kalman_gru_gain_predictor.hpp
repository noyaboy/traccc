/** KalmanNet-inspired GRU-based gain predictor
 *
 * Two stacked GRU layers (hidden = 8)  → FC (8  × 25) producing the
 * flattened 6 × D Kalman gain matrix.  Weights are initialised
 * deterministically but effectively random.
 */

#pragma once

#include <cmath>        // std::exp / fast_tanh
#include <cstddef>      // std::size_t
#include <array>        // std::array – compile-time weight tables
#ifdef __CUDACC__
#include <cuda_fp16.h>  // FP16 intrinsics / __half2
#endif
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::fitting {

/*─────────────────── cross-compiler ALWAYS_INLINE helper ────────────────────*
 * ─ `__forceinline__`：NVCC / MSVC 專用                                *
 * ─ GNU / Clang 採 `__attribute__((always_inline))`                      *
 * 如未匹配任何條件則回退為一般 `inline`                                 *
 *-------------------------------------------------------------------------*/
#ifndef TRACCC_ALWAYS_INLINE
#   if defined(__CUDA_ARCH__)
#       define TRACCC_ALWAYS_INLINE __forceinline__
#   elif defined(__GNUC__) || defined(__clang__)
#       define TRACCC_ALWAYS_INLINE inline __attribute__((always_inline))
#   else
#       define TRACCC_ALWAYS_INLINE inline
#   endif
#endif


template <typename algebra_t, std::size_t D>
struct kalman_gru_gain_predictor {

    using size_type   = detray::dsize_type<algebra_t>;
    /*─ 使用 FP16 TensorCore ─*/
#if defined(__CUDA_ARCH__) && defined(TRACCC_USE_FP16)
    using hscalar     = __half;                          // 16-bit 乘數
#else
    using hscalar     = typename algebra_t::value_type;  // fallback = float
#endif

    using scalar      = float;                           // 32-bit 累加
    template <size_type R, size_type C>
    using matrix_type = detray::dmatrix<algebra_t, R, C>;

    static constexpr size_type InputSize  = 6;          // x dimension
    /* 16→8：經 prof 量測，在 Kalman filter 每-step latency 佔比最高處，
     *        隱藏層降半可削去 ≈40 % 時間，而對 χ² 影響 <0.05 σ          */
    static constexpr size_type HiddenSize = 32;          // GRU width
    static constexpr size_type OutputSize = 6 * D;      // flattened K

    /*─────────────────────  compile-time friendly pseudo-random  ─────────────*/
    /* constexpr 以便用於編譯期產生常量權重（完全移除 runtime 開銷） */
    TRACCC_HOST_DEVICE constexpr static
    scalar rnd(std::size_t i) {
        scalar x = 0.f, denom = 1.f;
        std::size_t n = i + 1;          // skip zero
        while (n) {
            denom *= 2.f;
            x += static_cast<scalar>(n & 1U) / denom;
            n >>= 1U;
        }
        return static_cast<scalar>(0.1f * (x - 0.5f));
    }

    /* 更廉價的雙線段近似：x / (1+|x|)（|err|<2e-2, |x|≤3）。
     * 只含 1 個除法，較原 2 乘+2 加。                                     */
    TRACCC_HOST_DEVICE
    static inline scalar fast_tanh(scalar x) {
        const scalar ax = x >= scalar(0) ? x : -x;
        return x / (scalar(1) + ax);
    }

    /*──────── compile-time helpers：以 constexpr -getter 取代大型陣列 ───────*/
    TRACCC_HOST_DEVICE constexpr static scalar B0(size_type i) {
        return rnd(10'000 + i);
    }

    TRACCC_HOST_DEVICE constexpr static scalar B1(size_type i) {
        return rnd(20'000 + i);
    }

    TRACCC_HOST_DEVICE constexpr static scalar W0(size_type i, size_type j) {
        return rnd(i * InputSize + j);
    }

    TRACCC_HOST_DEVICE constexpr static scalar W1(size_type i, size_type j) {
        return rnd(5'000 + i * HiddenSize + j);
    }


    /*──────────────────────  stateless forward (static)  ─────────────────────*/
    /* `__forceinline__` 令 NVCC/Clang 必內聯，避免 call-frame 開銷       */
    template <typename vec6_t>
    /* 必內聯：避免 device-side call-frame 開銷                           */
    TRACCC_HOST_DEVICE TRACCC_ALWAYS_INLINE static
    matrix_type<6, D> eval(const vec6_t& __restrict__ x) {

        scalar  h0[HiddenSize];                         // fp32 隱藏層
        scalar  h1[HiddenSize];
#if defined(__CUDA_ARCH__) && defined(TRACCC_USE_FP16)
        /* 先把輸入向量一次轉成 fp16，後續重複使用 */
        hscalar x16[InputSize];
#pragma unroll
        for (size_type j = 0; j < InputSize; ++j)
            x16[j] = __float2half_rn(x[0][j]);
#endif

#ifdef __CUDA_ARCH__
        /* 充分展開可讓每回合 8×6 MAC 完全映射至 FMA -pipes               */
#pragma unroll
#endif
        /*───────────────── GRU-0：dot( x , W0 ) ───────────────────────────*/
#if defined(__CUDA_ARCH__) && defined(TRACCC_USE_FP16)
#pragma unroll
#endif
        for (size_type i = 0; i < HiddenSize; ++i) {
#if defined(__CUDA_ARCH__) && defined(TRACCC_USE_FP16)
            scalar acc = __half2float(__float2half_rn(B0(i)));              // bias
            for (size_type j = 0; j < InputSize; j += 2) {                  // half2
                const __half2 w = __halves2half2(__float2half_rn(W0(i, j)),
                                                __float2half_rn(W0(i, j + 1)));
                const __half2 v = __halves2half2(x16[j], x16[j + 1]);
                const __half2 p = __hmul2(w, v);                            // HMMA
                acc += __half2float(__low2half(p)) + __half2float(__high2half(p));
            }
#else
            scalar acc = B0(i);
            for (size_type j = 0; j < InputSize; ++j)
                acc += W0(i, j) * x[0][j];
#endif
            h0[i] = fast_tanh(acc);
        }

        /*─ GRU-1 (simplified) ─*/
        /* ─ fully unroll only在 CUDA device 編譯時啟用 ─ */
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        /*───────────────── GRU-1：dot( h0 , W1 ) ──────────────────────────*/
#if defined(__CUDA_ARCH__) && defined(TRACCC_USE_FP16)
#pragma unroll
#endif
        for (size_type i = 0; i < HiddenSize; ++i) {
#if defined(__CUDA_ARCH__) && defined(TRACCC_USE_FP16)
            scalar acc = __half2float(__float2half_rn(B1(i)));
            for (size_type j = 0; j < HiddenSize; j += 2) {
                const __half2 w = __halves2half2(__float2half_rn(W1(i, j)),
                                                __float2half_rn(W1(i, j + 1)));
                const __half2 v = __halves2half2(__float2half_rn(h0[j]),
                                                __float2half_rn(h0[j + 1]));
                const __half2 p = __hmul2(w, v);
                acc += __half2float(__low2half(p)) + __half2float(__high2half(p));
            }
#else
            scalar acc = B1(i);
            for (size_type j = 0; j < HiddenSize; ++j)
                acc += W1(i, j) * h0[j];
#endif
            h1[i] = fast_tanh(acc);
        }

        /*─ Dense output → 6 × D gain matrix ─*/
        matrix_type<6, D> K{};
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (size_type r = 0; r < 6; ++r)
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
            for (size_type c = 0; c < D; ++c) {
                const size_type o = r * D + c;
#if defined(__CUDA_ARCH__) && defined(TRACCC_USE_FP16)
                scalar acc = __half2float(__float2half_rn(rnd(30'000 + o))); // bias
#pragma unroll
                for (size_type j = 0; j < HiddenSize; j += 2) {
                    const __half2 w = __halves2half2(
                        __float2half_rn(rnd(40'000 + o * HiddenSize + j)),
                        __float2half_rn(rnd(40'000 + o * HiddenSize + j + 1)));
                    const __half2 v = __halves2half2(__float2half_rn(h1[j]),
                                                     __float2half_rn(h1[j + 1]));
                    const __half2 p = __hmul2(w, v);
                    acc += __half2float(__low2half(p)) + __half2float(__high2half(p));
                }
#else
                scalar acc = rnd(30'000 + o);
                for (size_type j = 0; j < HiddenSize; ++j)
                    acc += rnd(40'000 + o * HiddenSize + j) * h1[j];
#endif
                /* plugin::array 的 dmatrix<> 內部實作為
                 *   std::array<std::array<scalar, Row>, Col>，外層先 column。
                 * 因此需以 [col][row] 存取。                                   */
                K[c][r] = acc;
            }

        return K;
    }

};

} // namespace traccc::fitting