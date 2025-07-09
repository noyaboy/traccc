#pragma once
#include <cstdint>

#include "traccc/definitions/qualifiers.hpp"

// ------------------------------------------------------------
// ①  Weights stored in global memory for GPU execution;
//    on the host they act as regular constant arrays
// ------------------------------------------------------------

namespace traccc::fitting::detail {

// ------------------------------------------------------------
// ①  Weights stored in global memory for GPU execution;
//    on the host they act as regular constant arrays
// ------------------------------------------------------------
TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const std::int8_t W0[768] = {
    @W0@ };

TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const std::int8_t W1[2048] = {
    @W1@ };

TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const std::int8_t W2[640] = {
    @W2@ };

TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const float B0[32] = {
    @B0@ };

TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const float B1[64] = {
    @B1@ };

TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const float B2[10] = {
    @B2@ };

}  // namespace traccc::fitting::detail

// ------------------------------------------------------------
// ②  提供與舊版 *完全相同* 的介面
// ------------------------------------------------------------

namespace traccc::fitting {
template <typename algebra_t, std::size_t D>
struct kalman_int8_gru_gain_predictor_weights {
    // 以 constexpr 參考的方式把外部陣列「掛」進來
    static constexpr const auto& W0 = detail::W0;
    static constexpr const auto& W1 = detail::W1;
    static constexpr const auto& W2 = detail::W2;
    static constexpr const auto& B0 = detail::B0;
    static constexpr const auto& B1 = detail::B1;
    static constexpr const auto& B2 = detail::B2;

    // Quantisation parameters for activations
    static constexpr float QuantInScale   = @QuantInScale@;
    static constexpr int   QuantInZeroPoint = @QuantInZeroPoint@;
    static constexpr float FC1Scale       = @FC1Scale@;
    static constexpr int   FC1ZeroPoint   = @FC1ZeroPoint@;
    static constexpr float FC2Scale       = @FC2Scale@;
    static constexpr int   FC2ZeroPoint   = @FC2ZeroPoint@;
    static constexpr float FC3Scale       = @FC3Scale@;
    static constexpr int   FC3ZeroPoint   = @FC3ZeroPoint@;
};
}  // namespace traccc::fitting