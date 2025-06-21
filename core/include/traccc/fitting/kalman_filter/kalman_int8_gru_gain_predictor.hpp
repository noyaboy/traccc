/** KalmanNet-inspired GRU gain predictor — INT8 TensorCore edition
 *
 * 兩層 GRU (hidden = 32 → 64) ➜ Dense 6×D，
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

    // Only keep non-constant input features (see `CONST_INPUT_IDXS` in the
    // training script).  For `D=2` this results in 23 inputs.
    static constexpr size_type InputSize = 23;
    static constexpr size_type HiddenSize1 = 32;
    static constexpr size_type HiddenSize2 = 64;
    static constexpr size_type InputStep = (InputSize + 3) / 4;
    static constexpr size_type HiddenStep1 = HiddenSize1 / 4;
    static constexpr size_type HiddenStep2 = HiddenSize2 / 4;
    static constexpr float kWeightScale = 127.0f;

    static_assert(D == 2, "Offline weights are generated for D=2");

    /* 簡易 ReLU */
    TRACCC_HOST_DEVICE static inline float relu(float x) {
        return x >= 0.f ? x : 0.f;
    }

    TRACCC_HOST_DEVICE static inline qscalar quantise(float x, float inv_scale,
                                                      int zero_point) {
        float q = x * inv_scale + static_cast<float>(zero_point);
        int qi =
            q >= 0.f ? static_cast<int>(q + 0.5f) : static_cast<int>(q - 0.5f);
        if (qi > 127)
            qi = 127;
        if (qi < -128)
            qi = -128;
        return static_cast<qscalar>(qi);
    }

    /* ─────────────── forward ─────────────── */
    template <typename vec6_t>
    TRACCC_HOST_DEVICE TRACCC_ALWAYS_INLINE static matrix_type<6, D> eval(
        const vec6_t& __restrict__ x_f32) {

        using weights = kalman_int8_gru_gain_predictor_weights<algebra_t, D>;

        constexpr float s_in = weights::QuantInScale;
        constexpr float inv_s_in = 1.f / s_in;
        constexpr int zp_in = weights::QuantInZeroPoint;
        constexpr float s_fc1 = weights::FC1Scale;
        constexpr float inv_s_fc1 = 1.f / s_fc1;
        constexpr int zp_fc1 = weights::FC1ZeroPoint;
        constexpr float s_fc2 = weights::FC2Scale;
        constexpr float inv_s_fc2 = 1.f / s_fc2;
        constexpr int zp_fc2 = weights::FC2ZeroPoint;
        constexpr float s_fc3 = weights::FC3Scale;
        constexpr float inv_s_fc3 = 1.f / s_fc3;
        constexpr int zp_fc3 = weights::FC3ZeroPoint;
        constexpr float inv_w_scale = 1.f / kWeightScale;

        /* (1) 量化輸入 */
        TRACCC_ALIGN(16) qscalar x_q[InputStep * 4];

        TRACCC_PRAGMA_UNROLL
        for (size_type i = 0; i < InputSize; ++i) {
            x_q[i] = quantise(x_f32[0][i], inv_s_in, zp_in);
        }
        TRACCC_PRAGMA_UNROLL
        for (size_type i = InputSize; i < InputStep * 4; ++i) {
            x_q[i] = quantise(0.f, inv_s_in, zp_in);
        }

        /* (2) GRU-0 linear */
        TRACCC_ALIGN(16) qscalar h0_q[HiddenSize1];
        TRACCC_PRAGMA_UNROLL
        for (size_type i = 0; i < HiddenSize1; ++i) {
            int acc = 0;
            int wsum = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            const auto* W =
#ifdef __CUDA_ARCH__
                ::traccc::fitting::detail::W0
#else
                kalman_int8_gru_gain_predictor_weights<algebra_t, D>::W0
#endif
                ;
            const auto* Wp = reinterpret_cast<const int*>(W);
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < InputStep * 4; j += 4) {
                const int w = __ldg(&Wp[i * InputStep + j / 4]);
                const int v = *reinterpret_cast<const int*>(&x_q[j]);
                acc = __dp4a(w, v, acc);
                wsum = __dp4a(w, 0x01010101, wsum);
            }
#else
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < InputStep * 4; ++j) {
                const int w =
                    static_cast<int>(kalman_int8_gru_gain_predictor_weights<
                                     algebra_t, D>::W0[i * InputStep * 4 + j]);
                acc += w * static_cast<int>(x_q[j]);
                wsum += w;
            }
#endif
            float acc_f =
#ifdef __CUDA_ARCH__
                ::traccc::fitting::detail::B0[i]
#else
                kalman_int8_gru_gain_predictor_weights<algebra_t, D>::B0[i]
#endif
                + (s_in * inv_w_scale) *
                      (static_cast<float>(acc) -
                       static_cast<float>(zp_in) * static_cast<float>(wsum));
            float act = relu(acc_f);
            h0_q[i] = quantise(act, inv_s_fc1, zp_fc1);
        }

        /* (3) GRU-1 linear */
        TRACCC_ALIGN(16) qscalar h1_q[HiddenSize2];
        TRACCC_PRAGMA_UNROLL
        for (size_type i = 0; i < HiddenSize2; ++i) {
            int acc = 0;
            int wsum = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            const auto* W =
#ifdef __CUDA_ARCH__
                ::traccc::fitting::detail::W1
#else
                kalman_int8_gru_gain_predictor_weights<algebra_t, D>::W1
#endif
                ;
            const auto* Wp = reinterpret_cast<const int*>(W);
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize1; j += 4) {
                const int w = __ldg(&Wp[i * HiddenStep1 + j / 4]);
                const int v = *reinterpret_cast<const int*>(&h0_q[j]);
                acc = __dp4a(w, v, acc);
                wsum = __dp4a(w, 0x01010101, wsum);
            }
#else
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize1; ++j) {
                const int w =
                    static_cast<int>(kalman_int8_gru_gain_predictor_weights<
                                     algebra_t, D>::W1[i * HiddenSize1 + j]);
                acc += w * static_cast<int>(h0_q[j]);
                wsum += w;
            }
#endif
            float acc_f =
#ifdef __CUDA_ARCH__
                ::traccc::fitting::detail::B1[i]
#else
                kalman_int8_gru_gain_predictor_weights<algebra_t, D>::B1[i]
#endif
                + (s_fc1 * inv_w_scale) *
                      (static_cast<float>(acc) -
                       static_cast<float>(zp_fc1) * static_cast<float>(wsum));
            float act = relu(acc_f);
            h1_q[i] = quantise(act, inv_s_fc2, zp_fc2);
        }

        /* (4) Dense → 6×D Kalman gain (fp32 反量化) */
        matrix_type<6, D> K{};
        // Only the first 5 rows are predicted by the network.  The last row
        // corresponds to the removed constant outputs and is kept at zero.
        TRACCC_PRAGMA_UNROLL
        for (size_type r = 0; r < 5; ++r)
            TRACCC_PRAGMA_UNROLL
        for (size_type c = 0; c < D; ++c) {
            const size_type o = r * D + c;
            int acc = 0;
            int wsum = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
            const auto* W =
#ifdef __CUDA_ARCH__
                ::traccc::fitting::detail::W2
#else
                kalman_int8_gru_gain_predictor_weights<algebra_t, D>::W2
#endif
                ;
            const auto* Wp = reinterpret_cast<const int*>(W);
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize2; j += 4) {
                const int w = __ldg(&Wp[o * HiddenStep2 + j / 4]);
                const int v = *reinterpret_cast<const int*>(&h1_q[j]);
                acc = __dp4a(w, v, acc);
                wsum = __dp4a(w, 0x01010101, wsum);
            }
#else
            TRACCC_PRAGMA_UNROLL
            for (size_type j = 0; j < HiddenSize2; ++j) {
                const int w =
                    static_cast<int>(kalman_int8_gru_gain_predictor_weights<
                                     algebra_t, D>::W2[o * HiddenSize2 + j]);
                acc += w * static_cast<int>(h1_q[j]);
                wsum += w;
            }
#endif
            float acc_f =
#ifdef __CUDA_ARCH__
                ::traccc::fitting::detail::B2[o]
#else
                kalman_int8_gru_gain_predictor_weights<algebra_t, D>::B2[o]
#endif
                + (s_fc2 * inv_w_scale) *
                      (static_cast<float>(acc) -
                       static_cast<float>(zp_fc2) * static_cast<float>(wsum));
            qscalar q = quantise(acc_f, inv_s_fc3, zp_fc3);
            K[c][r] =
                (static_cast<float>(q) - static_cast<float>(zp_fc3)) * s_fc3;
        }
        // Pad constant outputs with zeros
        TRACCC_PRAGMA_UNROLL
        for (size_type c = 0; c < D; ++c) {
            K[c][5] = 0.f;
        }
        return K;
    }
};

}  // namespace traccc::fitting
