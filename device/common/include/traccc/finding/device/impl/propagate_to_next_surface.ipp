/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/details/combinatorial_kalman_filter_types.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include <detray/plugins/algebra/array_definitions.hpp>
#include <detray/propagator/constrained_step.hpp>
#include <detray/utils/tuple_helpers.hpp>

namespace traccc::device {

template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const global_index_t globalIndex, const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload) {

    using scalar_t = propagator_t::detector_type::scalar_type;

    if (globalIndex >= payload.n_in_params) {
        return;
    }

    // Theta id
    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);

    const unsigned int param_id = param_ids.at(globalIndex);

    // Links
    vecmem::device_vector<const candidate_link> links(payload.links_view);

    const unsigned int link_idx = payload.prev_links_idx + param_id;
    const auto& link = links.at(link_idx);
    assert(link.step == payload.step);
    const unsigned int n_cands = link.step + 1 - link.n_skipped;

    // Parameter liveness
    vecmem::device_vector<unsigned int> params_liveness(
        payload.params_liveness_view);

    // tips
    vecmem::device_vector<unsigned int> tips(payload.tips_view);
    vecmem::device_vector<unsigned int> tip_lengths(payload.tip_lengths_view);

    // Detector
    typename propagator_t::detector_type det(payload.det_data);

    // Parameters
    bound_track_parameters_collection_types::device params(payload.params_view);

    if (params_liveness.at(param_id) == 0u) {
        return;
    }

    // Input bound track parameter
    const bound_track_parameters<> in_par = params.at(param_id);

    // Create propagator
    auto prop_cfg{cfg.propagation};
    prop_cfg.navigation.estimate_scattering_noise = false;
    propagator_t propagator(prop_cfg);

    // Create propagator state
    typename propagator_t::state propagation(in_par, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, in_par));
    propagation._stepping
        .template set_constraint<detray::step::constraint::e_accuracy>(
            cfg.propagation.stepping.step_constraint);

    // Actor state initialization and propagation
    // Uses if constexpr to handle both 7-actor (with Jacobian) and 6-actor
    // (without Jacobian) chains at compile time.
    using actor_tuple_type =
        typename propagator_t::actor_chain_type::actor_tuple;

    // Propagation success flag - set by ckf_aborter in each branch
    bool propagation_success = false;

    if constexpr (details::has_jacobian_transport_v<propagator_t>) {
        // 7-actor chain: includes parameter_transporter for Jacobian transport
        // Actor indices: 0=pathlimit, 1=param_transporter, 2=interaction_reg,
        //                3=interactor, 4=resetter, 5=momentum, 6=ckf_aborter

        // Pathlimit aborter
        typename detray::detail::tuple_element<0, actor_tuple_type>::type::state
            s0{};
        // Parameter transporter (Jacobian accumulation)
        typename detray::detail::tuple_element<1, actor_tuple_type>::type::state
            s1{};
        // CKF-interactor
        typename detray::detail::tuple_element<3, actor_tuple_type>::type::state
            s3{};
        // Interaction register
        typename detray::detail::tuple_element<2,
                                               actor_tuple_type>::type::state
            s2{s3};
        // Parameter resetter
        typename detray::detail::tuple_element<4,
                                               actor_tuple_type>::type::state
            s4{prop_cfg};
        // Momentum aborter
        typename detray::detail::tuple_element<5, actor_tuple_type>::type::state
            s5;
        // CKF aborter
        typename detray::detail::tuple_element<6, actor_tuple_type>::type::state
            s6;

        // Initialize Jacobian for MBF smoother
        assert(payload.tmp_jacobian_ptr != nullptr);
        payload.tmp_jacobian_ptr[param_id] = matrix::identity<
            bound_matrix<typename propagator_t::detector_type::algebra_type>>();
        s1._full_jacobian_ptr = &payload.tmp_jacobian_ptr[param_id];

        s5.min_pT(static_cast<scalar_t>(cfg.min_pT));
        s5.min_p(static_cast<scalar_t>(cfg.min_p));
        s6.min_step_length = cfg.min_step_length_for_next_surface;
        s6.max_count = cfg.max_step_counts_for_next_surface;

        // Propagate with 7 actors
        propagator.propagate(propagation,
                             detray::tie(s0, s1, s2, s3, s4, s5, s6));

        propagation_success = s6.success;
    } else {
        // 7-actor chain with bound_updater (no Jacobian transport)
        // Uses bound_updater instead of parameter_transporter for lighter
        // free-to-bound conversion without Jacobian/covariance transport.
        // Actor indices: 0=pathlimit, 1=bound_updater, 2=interaction_reg,
        //                3=interactor, 4=resetter, 5=momentum, 6=ckf_aborter

        // Pathlimit aborter
        typename detray::detail::tuple_element<0, actor_tuple_type>::type::state
            s0{};
        // Bound updater (lightweight free-to-bound conversion)
        typename detray::detail::tuple_element<1, actor_tuple_type>::type::state
            s1{};
        // CKF-interactor
        typename detray::detail::tuple_element<3, actor_tuple_type>::type::state
            s3{};
        // Interaction register
        typename detray::detail::tuple_element<2,
                                               actor_tuple_type>::type::state
            s2{s3};
        // Parameter resetter
        typename detray::detail::tuple_element<4,
                                               actor_tuple_type>::type::state
            s4{prop_cfg};
        // Momentum aborter
        typename detray::detail::tuple_element<5, actor_tuple_type>::type::state
            s5;
        // CKF aborter
        typename detray::detail::tuple_element<6, actor_tuple_type>::type::state
            s6;

        s5.min_pT(static_cast<scalar_t>(cfg.min_pT));
        s5.min_p(static_cast<scalar_t>(cfg.min_p));
        s6.min_step_length = cfg.min_step_length_for_next_surface;
        s6.max_count = cfg.max_step_counts_for_next_surface;

        // Propagate with 7 actors (bound_updater instead of param_transporter)
        propagator.propagate(propagation,
                             detray::tie(s0, s1, s2, s3, s4, s5, s6));

        propagation_success = s6.success;
    }

    // If a surface found, add the parameter for the next step
    if (propagation_success) {
        assert(propagation._navigation.is_on_sensitive());
        assert(!propagation._stepping.bound_params().is_invalid());

        params[param_id] = propagation._stepping.bound_params();
        params_liveness[param_id] = 1u;

        const scalar theta = params[param_id].theta();
        if (theta <= 0.f || theta >= 2.f * constant<traccc::scalar>::pi) {
            TRACCC_ERROR_DEVICE("Theta is zero after propagation");
            params_liveness[param_id] = 0u;
        }

        if (!std::isfinite(params[param_id].phi())) {
            TRACCC_ERROR_DEVICE(
                "Phi is infinite after propagation (Matrix inversion)");
            params_liveness[param_id] = 0u;
        }

        if (math::fabs(params[param_id].qop()) == 0.f) {
            TRACCC_ERROR_DEVICE("q/p is zero after propagation");
            params_liveness[param_id] = 0u;
        }
    } else {
        params_liveness[param_id] = 0u;
    }

    if (params_liveness[param_id] == 0 &&
        n_cands >= cfg.min_track_candidates_per_track) {
        TRACCC_VERBOSE_DEVICE("Create tip: No next sensitive found");
        auto tip_pos = tips.push_back(link_idx);
        tip_lengths.at(tip_pos) = n_cands;
    }
}

}  // namespace traccc::device
