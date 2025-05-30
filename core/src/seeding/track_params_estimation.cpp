/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/track_params_estimation.hpp"

#include "traccc/seeding/track_params_estimation_helper.hpp"

// System include(s).
#include <cassert>

namespace traccc::host {

track_params_estimation::track_params_estimation(
    vecmem::memory_resource& mr, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr(mr) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const measurement_collection_types::const_view& measurements_view,
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::seed_collection::const_view& seeds_view, const vector3& bfield,
    const std::array<traccc::scalar, traccc::e_bound_size>& stddev) const {

    // Set up the input / output objects.
    const measurement_collection_types::const_device measurements(
        measurements_view);
    const edm::spacepoint_collection::const_device spacepoints(
        spacepoints_view);
    const edm::seed_collection::const_device seeds(seeds_view);
    const edm::seed_collection::const_device::size_type num_seeds =
        seeds.size();
    output_type result(num_seeds, &m_mr.get());

    // Create a track parameters for each seed.
    for (edm::seed_collection::const_device::size_type i = 0; i < num_seeds;
         ++i) {

        TRACCC_VERBOSE("Creating track parameters for seed " << i + 1 << " / "
                                                             << num_seeds);
        TRACCC_VERBOSE("  - bottom spacepoint: "
                       << spacepoints.at(seeds.at(i).bottom_index()).global()
                       << ", radius: "
                       << spacepoints.at(seeds.at(i).bottom_index()).radius());
        TRACCC_VERBOSE("  - middle spacepoint: "
                       << spacepoints.at(seeds.at(i).middle_index()).global()
                       << ", radius: "
                       << spacepoints.at(seeds.at(i).middle_index()).radius());
        TRACCC_VERBOSE("  - top spacepoint: "
                       << spacepoints.at(seeds.at(i).top_index()).global()
                       << ", radius: "
                       << spacepoints.at(seeds.at(i).top_index()).radius());

        // Calculate the track parameter vector.
        bound_track_parameters<>& track_params = result[i];
        track_params.set_vector(
            seed_to_bound_vector(measurements, spacepoints, seeds[i], bfield));

        // Set Covariance
        for (std::size_t j = 0; j < e_bound_size; ++j) {
            getter::element(track_params.covariance(), j, j) =
                stddev[j] * stddev[j];
        }

        // Get geometry ID for bottom spacepoint
        const edm::spacepoint_collection::const_device::const_proxy_type spB =
            spacepoints.at(seeds[i].bottom_index());
        assert(spB.measurement_index_2() ==
               edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX);
        track_params.set_surface_link(
            measurements.at(spB.measurement_index_1()).surface_link);
        TRACCC_VERBOSE("  - bound track parameters: " << track_params);
    }

    // Return the result.
    return result;
}

}  // namespace traccc::host
