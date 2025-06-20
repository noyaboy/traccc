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
TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const std::int8_t W0[768] = {0};

TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const std::int8_t W1[2048] = {0};

TRACCC_DEVICE_CONSTANT TRACCC_ALIGN(4) inline const std::int8_t W2[640] = {0};

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
};
}  // namespace traccc::fitting
