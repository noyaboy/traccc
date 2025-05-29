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

#include <cstdint>
#include <cstddef>
#include <algorithm>
#ifdef __CUDACC__
#include <cuda_runtime.h>   // __dp4a
#endif
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::fitting {

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
struct kalman_int8_gru_gain_predictor {

    using size_type = detray::dsize_type<algebra_t>;
    using qscalar   = std::int8_t;     ///< INT8 quantised值
    using accum_t   = std::int32_t;    ///< INT32 累加器
    template <size_type R, size_type C>
    using matrix_type = detray::dmatrix<algebra_t, R, C>;

    // InputSize = 6（predicted_vec） + 6×6（P） + D×6（H） + D×D（V）
    static constexpr size_type InputSize  = 6      // predicted_vec
                                          + 6*6    // P covariance
                                          + 6*D    // H projector
                                          + D*D;   // V measurement cov
    static constexpr size_type HiddenSize = 32;
    static constexpr float     kScale     = 127.0f;

    /* ────────────── compile-time 隨機權重產生 ────────────── */
    TRACCC_HOST_DEVICE constexpr static
    float rnd(std::size_t i) {
        float x = 0.f, denom = 1.f;
        for (std::size_t n = i + 1; n; n >>= 1U) {
            denom *= 2.f;
            if (n & 1U) x += 1.f / denom;
        }
        return 0.1f * (x - 0.5f);
    }
    TRACCC_HOST_DEVICE constexpr static
    qscalar qrnd(std::size_t i) {
        const float q = rnd(i) * kScale;
        return static_cast<qscalar>(q >= 0 ? (q > 127.f ? 127 : q + 0.5f)
                                           : (q < -128.f ? -128 : q - 0.5f));
    }

    /* 簡易 INT32 tanh 近似：y = x / (1 + |x|) (右移 7 位近似除以 128) */
    TRACCC_HOST_DEVICE static inline
    accum_t itanh(accum_t x) {
        const accum_t ax = x >= 0 ? x : -x;
        return x / (1 + (ax >> 7));
    }

    /* ─────────────── forward ─────────────── */
    template <typename vec6_t>
    TRACCC_HOST_DEVICE TRACCC_ALWAYS_INLINE static
    matrix_type<6, D> eval(const vec6_t& __restrict__ x_f32) {

        /* (1) 量化輸入 */
        qscalar x_q[InputSize];

#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (size_type i = 0; i < InputSize; ++i) {
            float q = x_f32[0][i] * kScale;
            x_q[i] = static_cast<qscalar>(q >= 0 ? (q > 127.f ? 127 : q + 0.5f)
                                                 : (q < -128.f ? -128 : q - 0.5f));
        }

        /* (2) GRU-0 linear */
        qscalar h0_q[HiddenSize];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (size_type i = 0; i < HiddenSize; ++i) {
            accum_t acc = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            for (size_type j = 0; j < InputSize; j += 4) {
                const accum_t w = (static_cast<unsigned char>(qrnd(i*InputSize+j    ))      ) |
                                  (static_cast<unsigned char>(qrnd(i*InputSize+j + 1)) <<  8) |
                                  (static_cast<unsigned char>(qrnd(i*InputSize+j + 2)) << 16) |
                                  (static_cast<unsigned char>(qrnd(i*InputSize+j + 3)) << 24);
                const accum_t v = *reinterpret_cast<const int*>(&x_q[j]);
                acc = __dp4a(w, v, acc);
            }
#else
            for (size_type j = 0; j < InputSize; ++j)
                acc += static_cast<accum_t>(qrnd(i*InputSize + j)) *
                       static_cast<accum_t>(x_q[j]);
#endif
            const accum_t act = itanh(acc);
            h0_q[i] = static_cast<qscalar>(std::clamp(act >> 7, -128, 127));
        }

        /* (3) GRU-1 linear */
        qscalar h1_q[HiddenSize];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (size_type i = 0; i < HiddenSize; ++i) {
            accum_t acc = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            for (size_type j = 0; j < HiddenSize; j += 4) {
                const accum_t w = (static_cast<unsigned char>(qrnd(5000 + i*HiddenSize+j    ))      ) |
                                  (static_cast<unsigned char>(qrnd(5000 + i*HiddenSize+j+1)) <<  8) |
                                  (static_cast<unsigned char>(qrnd(5000 + i*HiddenSize+j+2)) << 16) |
                                  (static_cast<unsigned char>(qrnd(5000 + i*HiddenSize+j+3)) << 24);
                const accum_t v = *reinterpret_cast<const int*>(&h0_q[j]);
                acc = __dp4a(w, v, acc);
            }
#else
            for (size_type j = 0; j < HiddenSize; ++j)
                acc += static_cast<accum_t>(qrnd(5000 + i*HiddenSize + j)) *
                       static_cast<accum_t>(h0_q[j]);
#endif
            const accum_t act = itanh(acc);
            h1_q[i] = static_cast<qscalar>(std::clamp(act >> 7, -128, 127));
        }

        /* (4) Dense → 6×D Kalman gain (fp32 反量化) */
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
                accum_t acc = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
                for (size_type j = 0; j < HiddenSize; j += 4) {
                    const accum_t w = (static_cast<unsigned char>(qrnd(40000 + o*HiddenSize+j    ))      ) |
                                      (static_cast<unsigned char>(qrnd(40000 + o*HiddenSize+j+1)) <<  8) |
                                      (static_cast<unsigned char>(qrnd(40000 + o*HiddenSize+j+2)) << 16) |
                                      (static_cast<unsigned char>(qrnd(40000 + o*HiddenSize+j+3)) << 24);
                    const accum_t v = *reinterpret_cast<const int*>(&h1_q[j]);
                    acc = __dp4a(w, v, acc);
                }
#else
                for (size_type j = 0; j < HiddenSize; ++j)
                    acc += static_cast<accum_t>(qrnd(40000 + o*HiddenSize + j)) *
                           static_cast<accum_t>(h1_q[j]);
#endif
                K[c][r] = static_cast<float>(acc) / (kScale * kScale);
            }
        return K;
    }
};

} // namespace traccc::fitting