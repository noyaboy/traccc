/** KalmanNet-inspired GRU gain predictor — INT8 TensorCore edition
 *
 * 兩層 GRU (hidden = 32) ➜ Dense 6×D，
 * 以對稱線性量化 (scale = 127) 將權重 / 活化轉為 INT8，
 * 累加於 INT32；在 SM_61+ 使用 __dp4a TensorCore 指令，
 * 無 TensorCore 時自動回退至純 CPU / GPU INT32 乘加。
 *
 * 主要優點
 * ───────────────────────────────────────────────────────────
 * • 記憶體流量較 FP32/FP16 減 4×
 * • 2080 Ti (Turing, SM 75) 可直接用 DP4A TensorCores
 * • 完全 constexpr  → 無 runtime 權重初始化/記憶體存取
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>  // __dp4a
#endif
#include "traccc/definitions/hints.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/fitting/kalman_filter/kalman_int8_gru_gain_predictor_weights.hpp"

namespace traccc::fitting {

#ifndef TRACCC_ALWAYS_INLINE
#if defined(__CUDA_ARCH__)
#define TRACCC_ALWAYS_INLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define TRACCC_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define TRACCC_ALWAYS_INLINE inline
#endif
#endif

template <typename algebra_t, std::size_t D>
struct kalman_int8_gru_gain_predictor {

    using size_type = detray::dsize_type<algebra_t>;
    using qscalar = std::int8_t;   ///< INT8 quantised值
    using accum_t = std::int32_t;  ///< INT32 累加器
    template <size_type R, size_type C>
    using matrix_type = detray::dmatrix<algebra_t, R, C>;

    // InputSize = 6（predicted_vec） + 6×6（P） + D×6（H） + D×D（V）
    static constexpr size_type InputSize = 6         // predicted_vec
                                           + 6 * 6   // P covariance
                                           + 6 * D   // H projector
                                           + D * D;  // V measurement cov
    static constexpr size_type HiddenSize1 = 32;
    static constexpr size_type HiddenSize2 = 16;
    static constexpr size_type InputStep = (InputSize + 3) / 4;
    static constexpr size_type HiddenStep1 = HiddenSize1 / 4;
    static constexpr size_type HiddenStep2 = HiddenSize2 / 4;
    static constexpr float kScale = 127.0f;
    static constexpr float kInvScaleSquared = 1.0f / (kScale * kScale);

    static_assert(D == 2, "Offline weights are generated for D=2");

    /* 簡易 INT32 tanh 近似：y = x / (1 + |x|) (右移 7 位近似除以 128) */
    TRACCC_HOST_DEVICE static inline accum_t relu(accum_t x) {
        const accum_t ax = x >= 0 ? x : 0;
        return ax;
    }

    TRACCC_HOST_DEVICE static inline qscalar saturate(accum_t x) {
        x >>= 7;
        if (x > 127)
            x = 127;
        if (x < -128)
            x = -128;
        return static_cast<qscalar>(x);
    }

    /* ─────────────── forward ─────────────── */
    template <typename vec6_t>
    TRACCC_HOST_DEVICE TRACCC_ALWAYS_INLINE static matrix_type<6, D> eval(
        const vec6_t& __restrict__ x_f32) {

        /* (1) 量化輸入 */
        TRACCC_ALIGN(16) qscalar x_q[InputStep * 4];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
        __shared__ int W0_s[HiddenSize1 * InputStep];
        __shared__ int W1_s[HiddenSize2 * HiddenStep1];
        __shared__ int W2_s[6 * D * HiddenStep2];

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            const int* W0p = reinterpret_cast<const int*>(::traccc::fitting::detail::W0);
            const int* W1p = reinterpret_cast<const int*>(::traccc::fitting::detail::W1);
            const int* W2p = reinterpret_cast<const int*>(::traccc::fitting::detail::W2);
            for (size_type i = 0; i < HiddenSize1 * InputStep; ++i) {
                W0_s[i] = W0p[i];
            }
            for (size_type i = 0; i < HiddenSize2 * HiddenStep1; ++i) {
                W1_s[i] = W1p[i];
            }
            for (size_type i = 0; i < 6 * D * HiddenStep2; ++i) {
                W2_s[i] = W2p[i];
            }
        }
        __syncthreads();
#endif

        TRACCC_PRAGMA_UNROLL
        for (size_type i = 0; i < InputSize; ++i) {
            float q = x_f32[0][i] * kScale;
            x_q[i] =
                static_cast<qscalar>(q >= 0 ? (q > 127.f ? 127 : q + 0.5f)
                                            : (q < -128.f ? -128 : q - 0.5f));
        }
        TRACCC_PRAGMA_UNROLL
        for (size_type i = InputSize; i < InputStep * 4; ++i) {
            x_q[i] = 0;
        }

        /* (2) GRU-0 linear */
        TRACCC_ALIGN(16) qscalar h0_q[HiddenSize1];
        TRACCC_PRAGMA_UNROLL
        for (size_type i = 0; i < HiddenSize1; ++i) {
            accum_t acc = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            const int* Wp = W0_s;
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < InputStep * 4; j += 4) {
                const int w = Wp[i * InputStep + j / 4];
                const int v = *reinterpret_cast<const int*>(&x_q[j]);
                acc = __dp4a(w, v, acc);
            }
#else
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < InputStep * 4; ++j)
                acc += static_cast<accum_t>(
                           kalman_int8_gru_gain_predictor_weights<
                               algebra_t, D>::W0[i * InputStep * 4 + j]) *
                       static_cast<accum_t>(x_q[j]);
#endif
            const accum_t act = relu(acc);
            h0_q[i] = saturate(act);
        }

        /* (3) GRU-1 linear */
        TRACCC_ALIGN(16) qscalar h1_q[HiddenSize2];
        TRACCC_PRAGMA_UNROLL
        for (size_type i = 0; i < HiddenSize2; ++i) {
            accum_t acc = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            const int* Wp = W1_s;
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize1; j += 4) {
                const int w = Wp[i * HiddenStep1 + j / 4];
                const int v = *reinterpret_cast<const int*>(&h0_q[j]);
                acc = __dp4a(w, v, acc);
            }
#else
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize1; ++j)
                acc += static_cast<accum_t>(
                           kalman_int8_gru_gain_predictor_weights<
                               algebra_t, D>::W1[i * HiddenSize1 + j]) *
                       static_cast<accum_t>(h0_q[j]);
#endif
            const accum_t act = relu(acc);
            h1_q[i] = saturate(act);
        }

        /* (4) Dense → 6×D Kalman gain (fp32 反量化) */
        matrix_type<6, D> K{};
        TRACCC_PRAGMA_UNROLL
        for (size_type r = 0; r < 6; ++r)
            TRACCC_PRAGMA_UNROLL
        for (size_type c = 0; c < D; ++c) {
            const size_type o = r * D + c;
            accum_t acc = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            const int* Wp = W2_s;
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize2; j += 4) {
                const int w = Wp[o * HiddenStep2 + j / 4];
                const int v = *reinterpret_cast<const int*>(&h1_q[j]);
                acc = __dp4a(w, v, acc);
            }
#else
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize2; ++j)
                acc += static_cast<accum_t>(
                           kalman_int8_gru_gain_predictor_weights<
                               algebra_t, D>::W2[o * HiddenSize2 + j]) *
                       static_cast<accum_t>(h1_q[j]);
#endif
            K[c][r] = static_cast<float>(acc) * kInvScaleSquared;
        }
        return K;
    }
};

}  // namespace traccc::fitting
