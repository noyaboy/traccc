// =============================================================================
//  view_traits.hpp ― 為各式 *view* 提供連續記憶體指標的統一存取介面
//                  (相容 covfie::field_view, detray::dmulti_view … 等)
// =============================================================================

#pragma once

#include <type_traits>
#include <utility>

namespace traccc::device {

namespace detail {

// ────────────────────────────────────────────────────────────────────────────
//  優先序 1：具備 .data() 成員函式者
// ────────────────────────────────────────────────────────────────────────────
template <class View>
TRACCC_HOST_DEVICE inline auto contiguous_ptr_impl(const View& v, int)
    -> decltype(const_cast<View&>(v).data()) {

    return const_cast<View&>(v).data();
}

// ────────────────────────────────────────────────────────────────────────────
//  優先序 2：退而求其次，使用 .begin()
//            ※ 若 .begin() 僅於 non-const 有效，透過 const_cast 取得
// ────────────────────────────────────────────────────────────────────────────
template <class View>
TRACCC_HOST_DEVICE inline auto contiguous_ptr_impl(const View& v, ...)
    -> decltype(const_cast<View&>(v).begin()) {

    return const_cast<View&>(v).begin();
}

}  // namespace detail

// ────────────────────────────────────────────────────────────────────────────
//  公開介面：根據 SFINAE 自動挑選可行實作
// ────────────────────────────────────────────────────────────────────────────
template <class View>
TRACCC_HOST_DEVICE inline auto contiguous_ptr(const View& v) {
    // 透過整數 / 可變參數 overload trick，落實「.data() 優先、否則 .begin()」
    return detail::contiguous_ptr_impl(v, 0);
}

}  // namespace traccc::device