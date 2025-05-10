// =============================================================================
//  view_traits.hpp ― 為各式 *view* 取得連續記憶體首位址
//                    相容 covfie::field_view、detray::dmulti_view…等
// =============================================================================

#pragma once

#include <type_traits>
#include <utility>

namespace traccc::device {

// ────────────────────────────────────────────────────────────────────────────
//  SFINAE 偵測：是否具備 .data() / .begin() / operator[]
// ────────────────────────────────────────────────────────────────────────────
namespace detail {
template <typename, typename = void> struct has_data      : std::false_type {};
template <typename T>
struct has_data<T, std::void_t<decltype(std::declval<T&>().data())>>
    : std::true_type {};

template <typename, typename = void> struct has_begin     : std::false_type {};
template <typename T>
struct has_begin<T, std::void_t<decltype(std::declval<T&>().begin())>>
    : std::true_type {};

template <typename, typename = void> struct has_subscript : std::false_type {};
template <typename T>
struct has_subscript<T, std::void_t<decltype(std::declval<T&>()[0])>>
    : std::true_type {};
}  // namespace detail

// ────────────────────────────────────────────────────────────────────────────
//  取得首元素之裸指標 (`const value_type*`)
//  * .data()  >  .begin()  >  operator[]
// ────────────────────────────────────────────────────────────────────────────
template <class View>
TRACCC_HOST_DEVICE inline auto contiguous_ptr(const View& v) {

    if constexpr (detail::has_data<View>::value) {
        return const_cast<View&>(v).data();

    } else if constexpr (detail::has_begin<View>::value) {
        return &(*const_cast<View&>(v).begin());

    } else {
        static_assert(detail::has_subscript<View>::value,
                      "View type lacks .data(), .begin(), and operator[]");
        return &const_cast<View&>(v)[0];
    }
}

}  // namespace traccc::device