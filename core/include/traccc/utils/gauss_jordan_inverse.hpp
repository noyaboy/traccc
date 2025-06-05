/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include <cmath>
#include "traccc/definitions/primitives.hpp"

namespace traccc {

/// Gauss-Jordan matrix inversion inspired by
/// "A fast parallel Gauss Jordan algorithm for matrix inversion using CUDA"
/// by Sharma et al., Computers & Structures 2013.
template <typename matrix_t>
TRACCC_HOST_DEVICE inline matrix_t gauss_jordan_inverse(matrix_t mat) {
    using algebra_t = typename matrix_t::algebra_type;
    using size_type = detray::dsize_type<algebra_t>;
    constexpr size_type N = matrix_t::RowsAtCompileTime;
    matrix_t inv = matrix::identity<matrix_t>();
    for (size_type i = 0; i < N; ++i) {
        size_type pivot_row = i;
        auto pivot = getter::element(mat, i, i);
        for (size_type r = i + 1; r < N; ++r) {
            auto val = getter::element(mat, r, i);
            if (std::abs(val) > std::abs(pivot)) {
                pivot = val;
                pivot_row = r;
            }
        }
        if (pivot_row != i) {
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
        for (size_type c = 0; c < N; ++c) {
            getter::element(mat, i, c) *= piv_inv;
            getter::element(inv, i, c) *= piv_inv;
        }
        for (size_type r = 0; r < N; ++r) {
            if (r == i) continue;
            auto factor = getter::element(mat, r, i);
            for (size_type c = 0; c < N; ++c) {
                getter::element(mat, r, c) -= factor * getter::element(mat, i, c);
                getter::element(inv, r, c) -= factor * getter::element(inv, i, c);
            }
        }
    }
    return inv;
}

} // namespace traccc

