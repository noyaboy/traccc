/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/finding/actors/bound_updater.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"

// Detray include(s).
#include "traccc/utils/propagation.hpp"

#include <detray/utils/tuple.hpp>

// System include(s).
#include <type_traits>

namespace traccc::details {

/// Stepper used in the Combinatorial Kalman Filter (CKF)
///
/// @tparam bfield_t The type of magnetic field to use
///
template <typename bfield_t>
using ckf_stepper_t = detray::rk_stepper<
    bfield_t, traccc::default_algebra, detray::constrained_step<traccc::scalar>,
    detray::stepper_rk_policy<traccc::scalar>, detray::stepping::void_inspector,
    static_cast<std::uint32_t>(
        detray::rk_stepper_flags::e_allow_covariance_transport)>;

/// Interactor used in the Combinatorial Kalman Filter (CKF)
using ckf_interactor_t =
    detray::pointwise_material_interactor<traccc::default_algebra>;

/// Actor chain used in the Combinatorial Kalman Filter (CKF)
///
/// This chain includes parameter_transporter for Jacobian accumulation,
/// which is required when run_mbf_smoother is enabled.
/// Uses 7 actors and ~150-180 registers.
///
using ckf_actor_chain_t =
    detray::actor_chain<detray::pathlimit_aborter<traccc::scalar>,
                        detray::parameter_transporter<traccc::default_algebra>,
                        interaction_register<ckf_interactor_t>,
                        ckf_interactor_t,
                        detray::parameter_resetter<traccc::default_algebra>,
                        detray::momentum_aborter<traccc::scalar>, ckf_aborter>;

/// Actor chain for CKF without MBF smoother (covariance transport but no
/// Jacobian aggregation)
///
/// This chain replaces parameter_transporter with bound_updater, which does
/// covariance transport but skips Jacobian aggregation (only needed for MBF).
/// The empty state (no Jacobian pointer) may reduce register pressure.
/// Use when run_mbf_smoother is disabled.
///
using ckf_actor_chain_no_mbf_t =
    detray::actor_chain<detray::pathlimit_aborter<traccc::scalar>,
                        bound_updater<traccc::default_algebra>,
                        interaction_register<ckf_interactor_t>,
                        ckf_interactor_t,
                        detray::parameter_resetter<traccc::default_algebra>,
                        detray::momentum_aborter<traccc::scalar>, ckf_aborter>;

/// Propagator type used in the Combinatorial Kalman Filter (CKF)
///
/// @tparam detector_t The detector type to use
/// @tparam bfield_t The magnetic field type to use
///
template <typename detector_t, typename bfield_t>
using ckf_propagator_t =
    detray::propagator<ckf_stepper_t<bfield_t>,
                       detray::caching_navigator<std::add_const_t<detector_t>>,
                       ckf_actor_chain_t>;

/// Propagator type for CKF without MBF smoother (no Jacobian transport)
///
/// Use this propagator when run_mbf_smoother is disabled to reduce
/// register pressure and improve GPU occupancy.
///
/// @tparam detector_t The detector type to use
/// @tparam bfield_t The magnetic field type to use
///
template <typename detector_t, typename bfield_t>
using ckf_propagator_no_mbf_t =
    detray::propagator<ckf_stepper_t<bfield_t>,
                       detray::caching_navigator<std::add_const_t<detector_t>>,
                       ckf_actor_chain_no_mbf_t>;

/// Helper to check if a type list contains parameter_transporter
/// Primary template: type not found
template <typename Target, typename Tuple>
struct contains_type : std::false_type {};

/// Specialization for detray::tuple: check first element, recurse on rest
template <typename Target, typename First, typename... Rest>
struct contains_type<Target, detray::tuple<First, Rest...>>
    : std::conditional_t<std::is_same_v<Target, First>, std::true_type,
                         contains_type<Target, detray::tuple<Rest...>>> {};

/// Base case: empty tuple
template <typename Target>
struct contains_type<Target, detray::tuple<>> : std::false_type {};

/// Type trait to detect if a propagator uses Jacobian transport
///
/// Returns true if the propagator's actor chain contains parameter_transporter
/// (the 7-actor chain), false otherwise (6-actor chain without Jacobian).
///
/// Usage in device code:
/// @code
/// if constexpr (has_jacobian_transport_v<propagator_t>) {
///     // Initialize Jacobian for MBF smoother
/// }
/// @endcode
///
template <typename propagator_t>
struct has_jacobian_transport
    : contains_type<detray::parameter_transporter<traccc::default_algebra>,
                    typename propagator_t::actor_chain_type::actor_tuple> {};

/// Helper variable template for has_jacobian_transport
template <typename propagator_t>
inline constexpr bool has_jacobian_transport_v =
    has_jacobian_transport<propagator_t>::value;

}  // namespace traccc::details
