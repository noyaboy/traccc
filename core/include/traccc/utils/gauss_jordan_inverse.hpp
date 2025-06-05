/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/hints.hpp"
#include <cmath>
#include <tuple>
#include "traccc/definitions/primitives.hpp"

namespace traccc {

/// Gauss-Jordan matrix inversion inspired by
/// "A fast parallel Gauss Jordan algorithm for matrix inversion using CUDA"
/// by Sharma et al., Computers & Structures 2013.
template <typename matrix_t>
TRACCC_HOST_DEVICE inline matrix_t gauss_jordan_inverse(matrix_t mat) {
    using size_type = std::size_t;
    // Determine matrix dimension either via a static member or tuple_size
    constexpr size_type N = [] {
        if constexpr (requires { matrix_t::RowsAtCompileTime; }) {
            return static_cast<size_type>(matrix_t::RowsAtCompileTime);
        } else {
            return static_cast<size_type>(std::tuple_size<matrix_t>::value);
        }
    }();
    matrix_t inv = matrix::identity<matrix_t>();
    TRACCC_PRAGMA_UNROLL
    for (size_type i = 0; i < N; ++i) {
        size_type pivot_row = i;
        auto pivot = getter::element(mat, i, i);
        TRACCC_PRAGMA_UNROLL
        for (size_type r = i + 1; r < N; ++r) {
            auto val = getter::element(mat, r, i);
            if (std::abs(val) > std::abs(pivot)) {
                pivot = val;
                pivot_row = r;
            }
        }
        if (pivot_row != i) {
            TRACCC_PRAGMA_UNROLL
            for (size_type c = 0; c < N; ++c) {
                auto tmp = getter::element(mat, i, c);
                getter::element(mat, i, c) = getter::element(mat, pivot_row, c);
                getter::element(mat, pivot_row, c) = tmp;
                tmp = getter::element(inv, i, c);
                getter::element(inv, i, c) = getter::element(inv, pivot_row, c);
                getter::element(inv, pivot_row, c) = tmp;
            }
        }
        auto piv_inv = 1 / getter::element(mat, i, i);
        TRACCC_PRAGMA_UNROLL
        for (size_type c = 0; c < N; ++c) {
            getter::element(mat, i, c) *= piv_inv;
            getter::element(inv, i, c) *= piv_inv;
        }
        TRACCC_PRAGMA_UNROLL
        for (size_type r = 0; r < N; ++r) {
            if (r == i) continue;
            auto factor = getter::element(mat, r, i);
            TRACCC_PRAGMA_UNROLL
            for (size_type c = 0; c < N; ++c) {
                getter::element(mat, r, c) -= factor * getter::element(mat, i, c);
                getter::element(inv, r, c) -= factor * getter::element(inv, i, c);
            }
        }
    }
    return inv;
}

} // namespace traccc

