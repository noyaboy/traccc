/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/device/fill_sort_keys.hpp"
#include "traccc/fitting/device/fit.hpp"
#include <vecmem/containers/data/vector_view.hpp>
#include "traccc/fitting/device/soa_types.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"

// Thrust include(s).
#include <thrust/sort.h>

// System include(s).
#include <memory_resource>
#include <vector>

namespace traccc::cuda {

namespace kernels {

__global__ void fill_sort_keys(
    track_candidate_container_types::const_view track_candidates_view,
    vecmem::data::vector_view<device::sort_key> keys_view,
    vecmem::data::vector_view<unsigned int> ids_view) {

    device::fill_sort_keys(details::global_index1(), track_candidates_view,
                           keys_view, ids_view);
}

template <typename fitter_t>
__global__ void fit(const typename fitter_t::config_type cfg,
                    const device::fit_payload<fitter_t> payload) {

    device::fit<fitter_t>(details::global_index1(), cfg, payload);
}

__global__ void fill_candidate_soa(
    track_candidate_container_types::const_view track_candidates_view,
    vecmem::data::vector_view<float4> lv,
    vecmem::data::vector_view<const unsigned int> offsets) {
    const track_candidate_container_types::const_device track_candidates(
        track_candidates_view);
    vecmem::device_vector<float4> lv_dev(lv);
    vecmem::device_vector<const unsigned int> off_dev(offsets);

    unsigned int idx = details::global_index1();
    if (idx >= track_candidates.size()) return;
    auto cands = track_candidates.at(idx).items;
    unsigned int off = off_dev[idx];
    for (unsigned int i = 0; i < cands.size(); ++i) {
        const auto& c = cands[i];
        lv_dev[off + i] =
            make_float4(c.local[0], c.local[1], c.variance[0], c.variance[1]);
    }
}

}  // namespace kernels

template <typename fitter_t>
fitting_algorithm<fitter_t>::fitting_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_cfg(cfg),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

template <typename fitter_t>
track_state_container_types::buffer fitting_algorithm<fitter_t>::operator()(
    const typename fitter_t::detector_type::view_type& det_view,
    const typename fitter_t::bfield_type& field_view,
    const typename track_candidate_container_types::const_view&
        track_candidates_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Number of tracks
    const track_candidate_container_types::const_device::header_vector::
        size_type n_tracks = m_copy.get_size(track_candidates_view.headers);

    // Get the sizes of the track candidates in each track
    using jagged_buffer_size_type = track_candidate_container_types::
        const_device::item_vector::value_type::size_type;
    const std::vector<jagged_buffer_size_type> candidate_sizes =
        m_copy.get_sizes(track_candidates_view.items);

    track_state_container_types::buffer track_states_buffer{
        {n_tracks, m_mr.main},
        {candidate_sizes, m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};

    std::vector<jagged_buffer_size_type> seqs_sizes(candidate_sizes.size());
    std::transform(candidate_sizes.begin(), candidate_sizes.end(),
                   seqs_sizes.begin(),
                   [this](const jagged_buffer_size_type sz) {
                       return std::max(sz * m_cfg.barcode_sequence_size_factor,
                                       m_cfg.min_barcode_sequence_capacity);
                   });
    vecmem::data::jagged_vector_buffer<detray::geometry::barcode> seqs_buffer{
        seqs_sizes, m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable};

    m_copy.setup(track_states_buffer.headers)->ignore();
    m_copy.setup(track_states_buffer.items)->ignore();
    m_copy.setup(seqs_buffer)->ignore();

    // Calculate the number of threads and thread blocks to run the track
    // fitting
    if (n_tracks > 0) {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;

        vecmem::data::vector_buffer<device::sort_key> keys_buffer(n_tracks,
                                                                  m_mr.main);
        vecmem::data::vector_buffer<unsigned int> param_ids_buffer(n_tracks,
                                                                   m_mr.main);

        // Get key and value for sorting
        kernels::fill_sort_keys<<<nBlocks, nThreads, 0, stream>>>(
            track_candidates_view, keys_buffer, param_ids_buffer);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        // Sort the key to get the sorted parameter ids
        vecmem::device_vector<device::sort_key> keys_device(keys_buffer);
        vecmem::device_vector<unsigned int> param_ids_device(param_ids_buffer);

        thrust::sort_by_key(thrust::cuda::par_nosync(
                                std::pmr::polymorphic_allocator(&m_mr.main))
                                .on(stream),
                            keys_device.begin(), keys_device.end(),
                            param_ids_device.begin());

        // Prepare SoA layout for measurements
        std::vector<unsigned int> offsets(n_tracks + 1, 0);
        for (std::size_t i = 0; i < candidate_sizes.size(); ++i) {
            offsets[i + 1] = offsets[i] + candidate_sizes[i];
        }
        vecmem::data::vector_buffer<float4> cand_lv_buffer(offsets.back(),
                                                           m_mr.main);
        vecmem::data::vector_buffer<unsigned int> offset_buffer(offsets.size(),
                                                                m_mr.main);
        m_copy.setup(cand_lv_buffer)->ignore();
        m_copy.setup(offset_buffer)->ignore();
        vecmem::data::vector_view<const unsigned int> offsets_view(
            offsets.data(), offsets.size());
        m_copy(offsets_view, offset_buffer,
                vecmem::copy::type::host_to_device)
            ->ignore();

        // Convert device buffers into views that can be used by the kernel
        auto cand_lv_view = vecmem::get_data(cand_lv_buffer);
        auto offset_view = vecmem::get_data(offset_buffer);

        kernels::fill_candidate_soa<<<nBlocks, nThreads, 0, stream>>>(
            track_candidates_view, cand_lv_view, offset_view);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        // Run the track fitting
        device::fit_payload<fitter_t> payload{};
        payload.det_data = det_view;
        payload.field_data = field_view;
        payload.track_candidates_view = track_candidates_view;
        payload.param_ids_view = param_ids_buffer;
        payload.candidate_soa.loc_var = cand_lv_view.ptr();
        payload.candidate_soa.offsets = offset_view.ptr();
        payload.track_states_view = track_states_buffer;
        payload.barcodes_view = seqs_buffer;

        kernels::fit<fitter_t><<<nBlocks, nThreads, 0, stream>>>(m_cfg,
                                                                payload);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    m_stream.synchronize();

    return track_states_buffer;
}

// Explicit template instantiation
template class fitting_algorithm<
    fitter_for_t<traccc::default_detector::device>>;
}  // namespace traccc::cuda
