
#pragma once

#ifndef TRACCC_UTILS_VIEW_TRAITS_HPP
#define TRACCC_UTILS_VIEW_TRAITS_HPP

#include <type_traits>
#include <utility>
#include <iterator>

namespace traccc::device {

//──────────────────────────────────────────────────────────────────────────────
//  traits：檢測 .data() / .begin() / operator[] 是否存在
//──────────────────────────────────────────────────────────────────────────────
template <typename T, typename = void> struct has_data      : std::false_type {};
template <typename T>
struct has_data<T, std::void_t<decltype(std::declval<T&>().data())>>
    : std::true_type {};

template <typename T, typename = void> struct has_member_begin : std::false_type {};
template <typename T>
struct has_member_begin<T, std::void_t<decltype(std::declval<T&>().begin())>>
    : std::true_type {};

template <typename T, typename = void> struct has_free_begin : std::false_type {};
template <typename T>
struct has_free_begin<T, std::void_t<decltype(std::begin(std::declval<T&>()))>>
    : std::true_type {};

template <typename T, typename = void> struct has_subscript  : std::false_type {};
template <typename T>
struct has_subscript<T, std::void_t<decltype(std::declval<T&>()[0])>>
    : std::true_type {};

//──────────────────────────────────────────────────────────────────────────────
//  contiguous_ptr
//      依優先序回傳連續記憶體首位址：
//         1. .data()
//         2. .begin()  /  std::begin()
//         3. operator[]
//      * 若全都缺，仍回傳 nullptr 以保證編譯通過；呼叫端需自行確保不解參。
//──────────────────────────────────────────────────────────────────────────────
template <class View>
TRACCC_HOST_DEVICE inline const void* contiguous_ptr(const View& v) {

    if constexpr (has_data<View>::value) {
        return static_cast<const void*>(const_cast<View&>(v).data());

    } else if constexpr (has_member_begin<View>::value) {
        return static_cast<const void*>(&(*const_cast<View&>(v).begin()));

    } else if constexpr (has_free_begin<View>::value) {
        return static_cast<const void*>(&(*std::begin(const_cast<View&>(v))));

    } else if constexpr (has_subscript<View>::value) {
        return static_cast<const void*>(&const_cast<View&>(v)[0]);

    } else {
        return nullptr;  // 最後保底；不應被解參
    }
}

}  // namespace traccc::device

#endif  // TRACCC_UTILS_VIEW_TRAITS_HPP