/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../../../utils/global_index.hpp"
#include "../propagate_to_next_surface.cuh"

// Project include(s).
#include "traccc/finding/device/propagate_to_next_surface.hpp"

namespace traccc::cuda::kernels {

template <typename propagator_t, typename bfield_t>
__global__ void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    const device::global_index_t globalIndex = details::global_index1();

    if (globalIndex >= payload.n_in_params) {
        return;
    }

    // Parameter order view
    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);
    const unsigned int param_id = param_ids.at(globalIndex);

    // Track counter
    vecmem::device_vector<unsigned int> n_tracks_per_seed(
        payload.n_tracks_per_seed_view);
    vecmem::device_vector<const candidate_link> links(payload.links_view);

    const unsigned int orig_param_id =
        links.at(payload.prev_links_idx + param_id).seed_idx;

    vecmem::device_atomic_ref<unsigned int> num_tracks_per_seed(
        n_tracks_per_seed.at(orig_param_id));
    const unsigned int s_pos = num_tracks_per_seed.fetch_add(1);

    vecmem::device_vector<unsigned int> params_liveness(
        payload.params_liveness_view);

    if (s_pos >= cfg.max_num_branches_per_seed) {
        params_liveness[param_id] = 0u;
        return;
    }

    // Tips vector
    vecmem::device_vector<unsigned int> tips(payload.tips_view);

    if (links.at(payload.prev_links_idx + param_id).n_skipped >
        cfg.max_num_skipping_per_cand) {
        params_liveness[param_id] = 0u;
        tips.push_back(payload.prev_links_idx + param_id);
        return;
    }

    typename propagator_t::detector_type det(payload.det_data);
    bound_track_parameters_collection_types::device params(payload.params_view);

    if (params_liveness.at(param_id) == 0u) {
        return;
    }

    const bound_track_parameters<> in_par = params.at(param_id);

    propagator_t propagator(cfg.propagation);

    typename propagator_t::state propagation(in_par, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, in_par));
    propagation._stepping
        .template set_constraint<detray::step::constraint::e_accuracy>(
            cfg.propagation.stepping.step_constraint);

    using actor_tuple_type =
        typename propagator_t::actor_chain_type::actor_tuple;
    typename detray::detail::tuple_element<0, actor_tuple_type>::type::state
        s0{};
    typename detray::detail::tuple_element<3, actor_tuple_type>::type::state
        s3{};
    typename detray::detail::tuple_element<2, actor_tuple_type>::type::state s2{
        s3};
    typename detray::detail::tuple_element<4, actor_tuple_type>::type::state s4;
    s4.min_step_length = cfg.min_step_length_for_next_surface;
    s4.max_count = cfg.max_step_counts_for_next_surface;

    // Synchronize before heavy propagation step
    __syncthreads();

    propagator.propagate_sync(propagation, detray::tie(s0, s2, s3, s4));

    // Synchronize before writing out results
    __syncthreads();

    if (s4.success) {
        params[param_id] = propagation._stepping.bound_params();

        if (payload.step == cfg.max_track_candidates_per_track - 1) {
            tips.push_back(payload.prev_links_idx + param_id);
            params_liveness[param_id] = 0u;
        } else {
            params_liveness[param_id] = 1u;
        }
    } else {
        params_liveness[param_id] = 0u;

        if (payload.step >= cfg.min_track_candidates_per_track - 1) {
            tips.push_back(payload.prev_links_idx + param_id);
        }
    }
}

}  // namespace traccc::cuda::kernels
