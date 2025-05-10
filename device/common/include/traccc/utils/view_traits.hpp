// =============================================================================
//  view_traits.hpp ― 提供 *view* 取得連續記憶體 pointer 之統一介面
// =============================================================================
#pragma once

#include <type_traits>

namespace traccc::device {

/// \return 指向 *view* 首元素的裸指標，假設其底層儲存連續
template <class View>
TRACCC_HOST_DEVICE inline auto contiguous_ptr(const View& v) -> decltype(v.begin()) {
    return v.begin();  // covfie::field_view / detray::dmulti_view 均支援
}

}  // namespace traccc::device
