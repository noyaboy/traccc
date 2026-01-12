/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/definitions/qualifiers.hpp"

// Detray include(s)
#include <detray/definitions/algebra.hpp>
#include <detray/definitions/detail/qualifiers.hpp>
#include <detray/definitions/track_parametrization.hpp>
#include <detray/geometry/tracking_surface.hpp>
#include <detray/propagator/actors/codegen/covariance_transport.hpp>
#include <detray/propagator/actors/codegen/full_jacobian.hpp>
#include <detray/propagator/base_actor.hpp>
#include <detray/propagator/detail/jacobian_engine.hpp>
#include <detray/utils/type_registry.hpp>

namespace traccc {

namespace detail {

/// Filter the masks of a detector according to the local frame type
struct select_frame {
    template <typename mask_t>
    using type = typename mask_t::local_frame;
};

}  // namespace detail

/// Actor to update bound parameters with covariance transport (no Jacobian
/// aggregation)
///
/// This actor converts free track parameters to bound parameters on the
/// current surface with proper covariance transport. Unlike parameter_
/// transporter, it does NOT aggregate the full Jacobian for MBF smoother.
///
/// Use this actor when run_mbf_smoother=false to reduce register pressure
/// by eliminating the Jacobian pointer from actor state.
///
/// Actor order in chain:
///   0. pathlimit_aborter
///   1. bound_updater         <- this actor (replaces parameter_transporter)
///   2. interaction_register
///   3. ckf_interactor
///   4. parameter_resetter
///   5. momentum_aborter
///   6. ckf_aborter
///
template <detray::concepts::algebra algebra_t>
struct bound_updater : detray::actor {

    /// @name Type definitions for the struct
    /// @{
    using scalar_type = detray::dscalar<algebra_t>;
    using transform3_type = detray::dtransform3D<algebra_t>;
    using bound_matrix_t = detray::bound_matrix<algebra_t>;
    using free_matrix_t = detray::free_matrix<algebra_t>;
    using bound_to_free_matrix_t = detray::bound_to_free_matrix<algebra_t>;
    using free_to_bound_matrix_t = detray::free_to_bound_matrix<algebra_t>;
    /// @}

    /// Empty state - no Jacobian pointer needed (saves registers vs
    /// parameter_transporter)
    struct state {};

    struct get_bound_to_free_dpos_dloc_visitor {
        template <typename frame_t>
        DETRAY_HOST_DEVICE constexpr detray::dmatrix<algebra_t, 3, 2> operator()(
            const frame_t& /*frame*/, const transform3_type& trf3,
            const detray::free_track_parameters<algebra_t>& params) const {

            return detray::detail::jacobian_engine<algebra_t>::
                template bound_to_free_jacobian_submatrix_dpos_dloc<frame_t>(
                    trf3, params.pos(), params.dir());
        }
    };

    struct get_bound_to_free_dpos_dangle_visitor {
        template <typename frame_t>
        DETRAY_HOST_DEVICE constexpr detray::dmatrix<algebra_t, 3, 2> operator()(
            const frame_t& /*frame*/, const transform3_type& trf3,
            const detray::free_track_parameters<algebra_t>& params,
            const detray::dmatrix<algebra_t, 3, 2>& ddir_dangle) const {

            return detray::detail::jacobian_engine<algebra_t>::
                template bound_to_free_jacobian_submatrix_dpos_dangle<frame_t>(
                    trf3, params.pos(), params.dir(), ddir_dangle);
        }
    };

    struct get_free_to_bound_dloc_dpos_visitor {
        template <typename frame_t, typename stepper_state_t>
        DETRAY_HOST_DEVICE constexpr detray::dmatrix<algebra_t, 2, 3> operator()(
            const frame_t& /*frame*/, const transform3_type& trf3,
            const stepper_state_t& stepping) const {

            return detray::detail::jacobian_engine<algebra_t>::
                template free_to_bound_jacobian_submatrix_dloc_dpos<frame_t>(
                    trf3, stepping().pos(), stepping().dir());
        }
    };

    /// Update bound parameters from free parameters when on a surface
    ///
    /// This performs covariance transport but skips Jacobian aggregation
    /// (which is only needed for MBF smoother).
    ///
    /// @param propagation The propagator state (navigation + stepping)
    template <typename propagator_state_t>
    TRACCC_HOST_DEVICE void operator()(state& /*unused*/,
                                       propagator_state_t& propagation) const {

        auto& stepping = propagation._stepping;
        const auto& navigation = propagation._navigation;

        // Only update when on a sensitive surface or material
        if (!(navigation.is_on_sensitive() ||
              navigation.encountered_sf_material())) {
            return;
        }

        // Geometry context
        const auto& gctx = propagation._context;

        // Current surface
        const auto sf = navigation.current_surface();

        // Get bound params reference
        auto& bound_params = stepping.bound_params();

        // Covariance is transported only when the previous surface is an
        // actual tracking surface. (i.e. This disables the covariance transport
        // from curvilinear frame)
        if (!bound_params.surface_link().is_invalid()) {
            const auto full_jacobian = get_full_jacobian(propagation);
            const bound_matrix_t old_cov = stepping.bound_params().covariance();

            detray::detail::transport_covariance_to_bound_impl(
                old_cov, full_jacobian, stepping.bound_params().covariance());

            // NOTE: No Jacobian aggregation here - that's only needed for MBF
            // smoother. This is the key difference from parameter_transporter.
        }

        // Convert free to bound vector
        bound_params.set_parameter_vector(
            sf.free_to_bound_vector(gctx, stepping()));

        // Set surface link
        bound_params.set_surface_link(sf.barcode());
    }

    template <typename propagator_state_t>
    TRACCC_HOST_DEVICE constexpr bound_matrix_t get_full_jacobian(
        propagator_state_t& propagation) const {

        // Map the surface shapes of the detector down to the common frames
        using detector_t = typename propagator_state_t::detector_type;
        using frame_registry_t =
            detray::types::mapped_registry<typename detector_t::masks,
                                           detail::select_frame>;

        const auto& stepping = propagation._stepping;
        const auto& navigation = propagation._navigation;

        // Geometry context for this track
        const auto& gctx = propagation._context;

        // Current Surface
        const auto sf = navigation.current_surface();

        // Bound track params of departure surface
        auto& bound_params = stepping.bound_params();

        // Previous surface
        detray::tracking_surface prev_sf{navigation.detector(),
                                         bound_params.surface_link()};

        // Free track params of departure surface
        const detray::free_track_parameters<algebra_t> free_params =
            prev_sf.bound_to_free_vector(gctx, bound_params);

        // Compute bound-to-free Jacobian sub-matrices
        const auto& prev_trf3 = prev_sf.transform(gctx);
        const detray::dmatrix<algebra_t, 3, 2> b2f_dpos_dloc =
            detray::types::visit<frame_registry_t,
                                 get_bound_to_free_dpos_dloc_visitor>(
                prev_sf.shape_id(), prev_trf3, free_params);

        const detray::dmatrix<algebra_t, 3, 2> b2f_ddir_dangle =
            detray::detail::jacobian_engine<algebra_t>::
                bound_to_free_jacobian_submatrix_ddir_dangle(bound_params);

        const detray::dmatrix<algebra_t, 3, 2> b2f_dpos_dangle =
            detray::types::visit<frame_registry_t,
                                 get_bound_to_free_dpos_dangle_visitor>(
                prev_sf.shape_id(), prev_trf3, free_params, b2f_ddir_dangle);

        // Compute path derivatives
        auto vol = navigation.current_volume();
        const auto vol_mat_ptr = vol.has_material()
                                     ? vol.material_parameters(stepping().pos())
                                     : nullptr;

        const auto path_to_free_derivative =
            detray::detail::jacobian_engine<algebra_t>::path_to_free_derivative(
                stepping().dir(), stepping.dtds(),
                stepping.dqopds(vol_mat_ptr));

        const auto free_to_path_derivative = sf.free_to_path_derivative(
            gctx, stepping().pos(), stepping().dir(), stepping.dtds());

        // Compute free-to-bound Jacobian sub-matrices
        const detray::dmatrix<algebra_t, 2, 3> f2b_dloc_dpos =
            detray::types::visit<frame_registry_t,
                                 get_free_to_bound_dloc_dpos_visitor>(
                sf.shape_id(), sf.transform(gctx), propagation._stepping);

        const detray::dmatrix<algebra_t, 2, 3> f2b_dangle_ddir =
            detray::detail::jacobian_engine<algebra_t>::
                free_to_bound_jacobian_submatrix_dangle_ddir(stepping().dir());

        // Use Sympy-generated full Jacobian computation
        bound_matrix_t full_jacobian;

        detray::detail::update_full_jacobian_impl(
            stepping.transport_jacobian(), b2f_dpos_dloc, b2f_ddir_dangle,
            b2f_dpos_dangle, path_to_free_derivative, free_to_path_derivative,
            f2b_dloc_dpos, f2b_dangle_ddir, full_jacobian);

        return full_jacobian;
    }
};

}  // namespace traccc
