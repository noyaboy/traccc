/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/status_codes.hpp"

// Detray inlcude(s)
#include <detray/geometry/shapes/line.hpp>

namespace traccc {

/// Type unrolling functor for Kalman updating
template <typename algebra_t>
struct gain_matrix_updater {

    // Type declarations
    using size_type = detray::dsize_type<algebra_t>;
    template <size_type ROWS, size_type COLS>
    using matrix_type = detray::dmatrix<algebra_t, ROWS, COLS>;
    using bound_vector_type = traccc::bound_vector<algebra_t>;
    using bound_matrix_type = traccc::bound_matrix<algebra_t>;

    template <size_type N>
    TRACCC_HOST_DEVICE static inline matrix_type<N, N> inverse_fast(
        const matrix_type<N, N>& m) {
        if constexpr (N == 1u) {
            matrix_type<1u, 1u> inv;
            getter::element(inv, 0u, 0u) =
                scalar(1) / getter::element(m, 0u, 0u);
            return inv;
        } else if constexpr (N == 2u) {
            const scalar a = getter::element(m, 0u, 0u);
            const scalar b = getter::element(m, 0u, 1u);
            const scalar c = getter::element(m, 1u, 0u);
            const scalar d = getter::element(m, 1u, 1u);
            const scalar det = a * d - b * c;
            matrix_type<2u, 2u> inv;
            getter::element(inv, 0u, 0u) = d / det;
            getter::element(inv, 0u, 1u) = -b / det;
            getter::element(inv, 1u, 0u) = -c / det;
            getter::element(inv, 1u, 1u) = a / det;
            return inv;
        } else {
            return matrix::inverse(m);
        }
    }

    /// Gain matrix updater operation
    ///
    /// @brief Based on "Application of Kalman filtering to track and vertex
    /// fitting", R.Frühwirth, NIM A
    ///
    /// @param mask_group mask group that contains the mask of surface
    /// @param index mask index of surface
    /// @param trk_state track state of the surface
    /// @param bound_params bound parameter
    ///
    /// @return true if the update succeeds
    template <typename mask_group_t, typename index_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status operator()(
        const mask_group_t& /*mask_group*/, const index_t& /*index*/,
        track_state<algebra_t>& trk_state,
        const bound_track_parameters<algebra_t>& bound_params) const {

        using shape_type = typename mask_group_t::value_type::shape;

        const auto D = trk_state.get_measurement().meas_dim;
        assert(D == 1u || D == 2u);
        kalman_fitter_status result = kalman_fitter_status::ERROR_OTHER;
        switch (D) {
            case 1u:
                result = update<1u, shape_type>(trk_state, bound_params);
                break;
            case 2u:
                result = update<2u, shape_type>(trk_state, bound_params);
                break;
            default:
                __builtin_unreachable();
        }
        return result;
    }

    template <size_type D, typename shape_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status update(
        track_state<algebra_t>& trk_state,
        const bound_track_parameters<algebra_t>& bound_params) const {

        static_assert(((D == 1u) || (D == 2u)),
                      "The measurement dimension should be 1 or 2");

        const auto meas = trk_state.get_measurement();

        // Some identity matrices
        // @TODO: Make constexpr work
        const auto I_m = matrix::identity<matrix_type<D, D>>();

        matrix_type<D, e_bound_size> H = meas.subs.template projector<D>();

        // Measurement data on surface
        const matrix_type<D, 1>& meas_local =
            trk_state.template measurement_local<D>();

        // Predicted vector of bound track parameters
        const bound_vector_type& predicted_vec = bound_params.vector();

        // Predicted covariance of bound track parameters
        const bound_matrix_type& predicted_cov = bound_params.covariance();

        // Flip the sign of projector matrix element in case the first element
        // of line measurement is negative
        if constexpr (std::is_same_v<shape_t, detray::line<true>> ||
                      std::is_same_v<shape_t, detray::line<false>>) {

            if (getter::element(predicted_vec, e_bound_loc0, 0u) < 0) {
                getter::element(H, 0u, e_bound_loc0) = -1;
            }
        }

        // Spatial resolution (Measurement covariance)
        const matrix_type<D, D> V =
            trk_state.template measurement_covariance<D>();

        const auto HT = matrix::transpose(H);
        const matrix_type<6, D> PH = predicted_cov * HT;
        const matrix_type<D, D> M = H * PH + V;
        const auto M_inv = inverse_fast(M);

        // Kalman gain matrix
        const matrix_type<6, D> K = PH * M_inv;

        const matrix_type<D, 1> proj_predicted = H * predicted_vec;

        matrix_type<6, 1> filtered_vec{};
        matrix_type<6, 6> filtered_cov{};
        matrix_type<D, 1> residual{};
        matrix_type<1, 1> chi2{};

        if constexpr (D == 1u) {
            const scalar meas_val = getter::element(meas_local, 0u, 0u);
            scalar pred_meas = scalar(0);
            for (size_type i = 0u; i < e_bound_size; ++i) {
                pred_meas += getter::element(H, 0u, i) *
                             getter::element(predicted_vec, i, 0u);
            }
            const scalar res_pred = meas_val - pred_meas;
            scalar hk = scalar(0);
            for (size_type i = 0u; i < e_bound_size; ++i) {
                getter::element(filtered_vec, i, 0u) =
                    getter::element(predicted_vec, i, 0u) +
                    getter::element(K, i, 0u) * res_pred;
                hk += getter::element(H, 0u, i) * getter::element(K, i, 0u);
            }
            for (size_type i = 0u; i < e_bound_size; ++i) {
                for (size_type j = 0u; j < e_bound_size; ++j) {
                    getter::element(filtered_cov, i, j) =
                        getter::element(predicted_cov, i, j) -
                        getter::element(K, i, 0u) * getter::element(PH, j, 0u);
                }
            }
            scalar pred_meas_f = scalar(0);
            for (size_type i = 0u; i < e_bound_size; ++i) {
                pred_meas_f += getter::element(H, 0u, i) *
                               getter::element(filtered_vec, i, 0u);
            }
            const scalar res = meas_val - pred_meas_f;
            const scalar R_val = (scalar(1) - hk) * getter::element(V, 0u, 0u);
            const scalar inv_R = scalar(1) / R_val;
            getter::element(residual, 0u, 0u) = res;
            getter::element(chi2, 0u, 0u) = res * res * inv_R;
        } else {
            const matrix_type<6, D> diff = K * (meas_local - proj_predicted);
            filtered_vec = predicted_vec + diff;
            const matrix_type<D, 6> HPC = matrix::transpose(PH);
            filtered_cov = predicted_cov - K * HPC;
            residual = meas_local - H * filtered_vec;
            const matrix_type<D, D> R = (I_m - H * K) * V;
            const matrix_type<D, D> R_inv = inverse_fast(R);
            chi2 = matrix::transpose(residual) * R_inv * residual;
        }

        // Return false if track is parallel to z-axis or phi is not finite
        const scalar theta = bound_params.theta();

        if (theta <= 0.f || theta >= constant<traccc::scalar>::pi) {
            return kalman_fitter_status::ERROR_THETA_ZERO;
        }

        if (!std::isfinite(bound_params.phi())) {
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (std::abs(bound_params.qop()) == 0.f) {
            return kalman_fitter_status::ERROR_QOP_ZERO;
        }

        if (getter::element(chi2, 0, 0) < 0.f) {
            return kalman_fitter_status::ERROR_UPDATER_CHI2_NEGATIVE;
        }

        if (!std::isfinite(getter::element(chi2, 0, 0))) {
            return kalman_fitter_status::ERROR_UPDATER_CHI2_NOT_FINITE;
        }

        // Set the track state parameters
        trk_state.filtered().set_vector(filtered_vec);
        trk_state.filtered().set_covariance(filtered_cov);
        trk_state.filtered_chi2() = getter::element(chi2, 0, 0);

        // Wrap the phi in the range of [-pi, pi]
        wrap_phi(trk_state.filtered());

        return kalman_fitter_status::SUCCESS;
    }
};

}  // namespace traccc
