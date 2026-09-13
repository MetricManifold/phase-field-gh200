#include "pf3d/sim.cuh"
#include "pf3d/brick_classes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

namespace pf3d {
namespace {

bool checked(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "[3d] %s failed: %s\n", operation,
                 cudaGetErrorString(result));
    return false;
}

struct DeviceDeleter {
    void operator()(void* pointer) const { if (pointer) cudaFree(pointer); }
};
template<class T> using DeviceBuffer = std::unique_ptr<T, DeviceDeleter>;

template<class T>
bool allocate_buffer(DeviceBuffer<T>* buffer, std::size_t bytes) {
    T* pointer = nullptr;
    if (!checked(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes),
                 "allocate adaptive storage")) return false;
    buffer->reset(pointer);
    return true;
}

bool add_bytes(std::size_t amount, std::size_t* total) {
    if (amount > std::numeric_limits<std::size_t>::max() - *total) return false;
    *total += amount;
    return true;
}

// Growth embeds the complete source; a crop is allowed only after the GPU
// certifies zero discarded values and an empty inner guard around the new cube.
bool copy_centered(float* destination, int new_edge, const float* source,
                   int old_edge, std::int64_t old_origin_z,
                   std::int64_t new_origin_z, CellFieldStorage3D storage,
                   cudaStream_t stream) {
    int offset = 0;
    if (!destination || !source ||
        !centered_resize_offset(old_edge, new_edge, &offset)) return false;
    const int copy_edge = std::min(old_edge, new_edge);
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(const_cast<float*>(source),
        static_cast<std::size_t>(old_edge) * sizeof(float), old_edge, old_edge);
    copy.dstPtr = make_cudaPitchedPtr(destination,
        static_cast<std::size_t>(new_edge) * sizeof(float), new_edge, new_edge);
    const int source_offset = std::max(0, -offset);
    const int destination_offset = std::max(0, offset);
    CellFieldPlaneRange3D source_planes, destination_planes;
    if (!storage.retained_planes(old_edge, old_origin_z, &source_planes) ||
        !storage.retained_planes(new_edge, new_origin_z, &destination_planes))
        return false;
    // Intersect in source logical coordinates. A compact allocation stores
    // world-z planes; a cubic allocation stores local-z planes. This also
    // handles growth or cropping across the compact-storage threshold.
    const int begin_z = std::max({source_offset, source_planes.logical_begin,
                                  destination_planes.logical_begin - offset});
    const int end_z = std::min({source_offset + copy_edge,
        source_planes.logical_begin + source_planes.count,
        destination_planes.logical_begin + destination_planes.count - offset});
    if (begin_z >= end_z) return true;
    const int source_z = source_planes.stored_begin + begin_z - source_planes.logical_begin;
    const int destination_z = destination_planes.stored_begin + begin_z + offset
                            - destination_planes.logical_begin;
    copy.srcPos = make_cudaPos(static_cast<std::size_t>(source_offset) * sizeof(float),
                              source_offset, source_z);
    copy.dstPos = make_cudaPos(static_cast<std::size_t>(destination_offset) * sizeof(float),
                              destination_offset, destination_z);
    copy.extent = make_cudaExtent(static_cast<std::size_t>(copy_edge) * sizeof(float),
                                  copy_edge, end_z - begin_z);
    copy.kind = cudaMemcpyDeviceToDevice;
    return checked(cudaMemcpy3DAsync(&copy, stream), "copy centered cell field");
}

bool translate_state(CellState3D* state, int old_edge, int new_edge) {
    int offset = 0;
    if (!centered_resize_offset(old_edge, new_edge, &offset) ||
        !checked_resize_origin(state->origin_x, old_edge, new_edge, &state->origin_x) ||
        !checked_resize_origin(state->origin_y, old_edge, new_edge, &state->origin_y) ||
        !checked_resize_origin(state->origin_z, old_edge, new_edge, &state->origin_z))
        return false;
    state->Cx += static_cast<double>(offset) * state->V;
    state->Cy += static_cast<double>(offset) * state->V;
    state->Cz += static_cast<double>(offset) * state->V;
    if (!std::isfinite(state->Cx) || !std::isfinite(state->Cy) ||
        !std::isfinite(state->Cz)) return false;
    if (state->bb_hi_x >= state->bb_lo_x) {
        state->bb_lo_x += offset; state->bb_hi_x += offset;
        state->bb_lo_y += offset; state->bb_hi_y += offset;
        state->bb_lo_z += offset; state->bb_hi_z += offset;
    }
    state->pending_shift_x = state->pending_shift_y = state->pending_shift_z = 0;
    state->storage_edge = static_cast<std::uint32_t>(new_edge);
    state->flags &= ~(flag3d_bit(FLAG3D_SUPPORT_EXHAUSTED) |
                      flag3d_bit(FLAG3D_SUPPORT_EDGE));
    return true;
}

bool check_crops(float* const* fields, const CellState3D* cells,
                 const std::vector<int>& candidates, int base,
                 cudaStream_t stream, CellFieldStorage3D storage,
                 std::vector<std::uint32_t>* rejected) {
    DeviceBuffer<int> device_candidates;
    DeviceBuffer<std::uint32_t> device_rejected;
    const std::size_t bytes = candidates.size() * sizeof(int);
    if (!allocate_buffer(&device_candidates, bytes) ||
        !allocate_buffer(&device_rejected, candidates.size() * sizeof(std::uint32_t)))
        return false;
    rejected->assign(candidates.size(), 0u);
    const bool ok = checked(cudaMemcpyAsync(device_candidates.get(), candidates.data(),
            bytes, cudaMemcpyHostToDevice, stream), "upload crop candidates") &&
        checked(cudaMemsetAsync(device_rejected.get(), 0,
            rejected->size() * sizeof(std::uint32_t), stream), "clear crop results") &&
        launch_check_promoted_crops(fields, cells, device_candidates.get(),
            device_rejected.get(), static_cast<int>(candidates.size()), base, stream, storage) &&
        checked(cudaMemcpyAsync(rejected->data(), device_rejected.get(),
            rejected->size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
            "read crop results");
    // The pageable host inputs and the temporary device allocations must outlive
    // every queued copy, including a partially submitted sequence on failure.
    const bool drained = checked(cudaStreamSynchronize(stream), "finish crop check");
    return ok && drained;
}

struct PendingBrick {
    int id;
    int old_edge;
    int new_edge;
    std::size_t bytes;
    std::array<DeviceBuffer<float>, 2> fields;
};

} // namespace

bool Sim3D::resize_cells(const std::vector<std::pair<int, int>>& requested,
                         std::vector<CellState3D>* states) {
    const std::size_t count = static_cast<std::size_t>(params_.num_cells);
    if (!states || states->size() != count || step_pending_ || selection_.ids)
        return false;
    if (requested.empty()) return true;
    std::vector<unsigned char> seen(count, 0);
    std::vector<int> candidates(count, 0);
    std::vector<CellState3D> next_states = *states;
    std::vector<PendingBrick> pending;
    pending.reserve(requested.size());
    std::size_t new_bytes = 0, released_bytes = 0;
    bool shrinking = false;
    for (const auto& request : requested) {
        const int id = request.first, edge = request.second;
        if (id < 0 || id >= params_.num_cells || seen[id]++ || edge < B_ ||
            edge > maximum_support_edge_ || !valid_runtime_geometry(edge, layout_))
            return false;
        const int old_edge = cell_support_edge((*states)[id], B_);
        const bool was_promoted = old_edge > B_;
        if (old_edge == edge || old_edge < B_ ||
            (was_promoted && (!h_promoted_phi_[0][id] || !h_promoted_phi_[1][id])) ||
            !translate_state(&next_states[id], old_edge, edge)) return false;
        std::size_t bytes = 0, old_bytes = 0;
        if (!field_storage_.checked_bytes(edge, &bytes) ||
            !field_storage_.checked_bytes(old_edge, &old_bytes) ||
            (edge > B_ && (!add_bytes(bytes, &new_bytes) || !add_bytes(bytes, &new_bytes))) ||
            (was_promoted && (!add_bytes(old_bytes, &released_bytes) ||
                              !add_bytes(old_bytes, &released_bytes)))) return false;
        pending.push_back({id, old_edge, edge, bytes, {}});
        if (edge < old_edge) { candidates[id] = edge; shrinking = true; }
    }
    if (new_bytes > adaptive_budget_remaining_) {
        std::fprintf(stderr, "[3d] adaptive resize needs %.3f GiB of temporary storage; "
            "%.3f GiB remains in the HBM budget; accepted fields retained\n",
            new_bytes / 1073741824.0, adaptive_budget_remaining_ / 1073741824.0);
        return false;
    }
    if (shrinking) {
        const std::size_t probe_bytes = count * (sizeof(int) + sizeof(std::uint32_t));
        if (probe_bytes > adaptive_budget_remaining_) return false;
        std::vector<std::uint32_t> rejected;
        if (!check_crops(d_promoted_phi_[current_promoted_phi_index_], d_cells_,
                         candidates, B_, stream_, field_storage_, &rejected)) return false;
        if (std::any_of(rejected.begin(), rejected.end(),
                        [](std::uint32_t value) { return value != 0; })) return false;
    }
    // Allocate the complete replacement before changing any live pointer.
    for (auto& brick : pending)
        if (brick.new_edge > B_)
            for (auto& field : brick.fields)
                if (!allocate_buffer(&field, brick.bytes)) return false;

    bool copied = true;
    for (auto& brick : pending) {
        const float* source = brick.old_edge > B_
            ? h_promoted_phi_[current_promoted_phi_index_][brick.id]
            : current_phi() + static_cast<std::size_t>(brick.id) * brick_words_;
        const int buffers = brick.new_edge > B_ ? 2 : phi_buffers_;
        for (int buffer = 0; buffer < buffers && copied; ++buffer) {
            float* destination = brick.new_edge > B_ ? brick.fields[buffer].get()
                : d_phi_[buffer] + static_cast<std::size_t>(brick.id) * brick_words_;
            copied = checked(cudaMemsetAsync(destination, 0, brick.bytes, stream_),
                              "clear resized field") &&
                copy_centered(destination, brick.new_edge, source, brick.old_edge,
                    (*states)[brick.id].origin_z, next_states[brick.id].origin_z,
                    field_storage_, stream_);
        }
        if (!copied) break;
    }
    const bool drained = checked(cudaStreamSynchronize(stream_), "finish field resizing");
    if (!copied || !drained) return false;

    std::vector<float*> next_tables[2] = {h_promoted_phi_[0], h_promoted_phi_[1]};
    for (auto& brick : pending)
        for (int buffer = 0; buffer < 2; ++buffer)
            next_tables[buffer][brick.id] = brick.new_edge > B_
                ? brick.fields[buffer].get() : nullptr;
    std::vector<int> next_ids;
    int maximum = 0;
    // Stable ID order makes cohort order independent of the resize batch.
    for (int id = 0; id < params_.num_cells; ++id)
        if (next_states[id].storage_edge > static_cast<unsigned>(B_)) {
            next_ids.push_back(id);
            maximum = std::max(maximum, static_cast<int>(next_states[id].storage_edge));
        }
    auto install = [&](const std::vector<float*>* tables,
                       const std::vector<int>& ids,
                       const std::vector<CellState3D>& cell_states) {
        bool ok = true;
        for (int buffer = 0; buffer < 2; ++buffer)
            ok = ok && checked(cudaMemcpyAsync(d_promoted_phi_[buffer], tables[buffer].data(),
                count * sizeof(float*), cudaMemcpyHostToDevice, stream_), "install adaptive pointers");
        if (!ids.empty())
            ok = ok && checked(cudaMemcpyAsync(d_promoted_ids_, ids.data(),
                ids.size() * sizeof(int), cudaMemcpyHostToDevice, stream_), "install adaptive ids");
        ok = ok && checked(cudaMemcpyAsync(d_cells_, cell_states.data(),
            count * sizeof(CellState3D), cudaMemcpyHostToDevice, stream_), "install resized states");
        const bool complete = checked(cudaStreamSynchronize(stream_), "finish adaptive installation");
        return ok && complete;
    };
    if (!install(next_tables, next_ids, next_states)) {
        (void)install(h_promoted_phi_, h_promoted_ids_, *states);
        return false;
    }
    for (auto& brick : pending) {
        if (brick.old_edge > B_)
            for (int buffer = 0; buffer < 2; ++buffer)
                cudaFree(h_promoted_phi_[buffer][brick.id]);
        for (auto& field : brick.fields) (void)field.release();
    }
    h_promoted_phi_[0].swap(next_tables[0]);
    h_promoted_phi_[1].swap(next_tables[1]);
    h_promoted_ids_.swap(next_ids);
    states->swap(next_states);
    promoted_edge_ = maximum;
    adaptive_budget_remaining_ = adaptive_budget_remaining_ - new_bytes + released_bytes;
    required_device_bytes_ = required_device_bytes_ + new_bytes - released_bytes;
    ++recovery_events_; // The peer must reconstruct its allocations before another step.
    std::printf("[3d] resized %zu cell(s); adaptive cells %zu, maximum B=%d, "
                "field allocation delta %+.3f GiB\n", pending.size(), h_promoted_ids_.size(),
                promoted_edge_, (static_cast<double>(new_bytes) - released_bytes) / 1073741824.0);
    return true;
}

bool Sim3D::install_checkpoint_promotions(const CheckpointMeta3D& checkpoint) {
    const auto count = static_cast<std::size_t>(params_.num_cells);
    if (checkpoint.storage_edges.size() != count || !h_promoted_ids_.empty()) return false;
    std::vector<PendingBrick> pending;
    std::size_t bytes = 0;
    for (int id = 0; id < params_.num_cells; ++id) {
        const int edge = checkpoint.storage_edges[id];
        if (edge == B_) continue;
        std::size_t cell_bytes = 0;
        if (edge < B_ || edge > maximum_support_edge_ ||
            !valid_runtime_geometry(edge, layout_) ||
            !field_storage_.checked_bytes(edge, &cell_bytes) ||
            !add_bytes(cell_bytes, &bytes) || !add_bytes(cell_bytes, &bytes)) return false;
        pending.push_back({id, B_, edge, cell_bytes, {}});
    }
    if (bytes > adaptive_budget_remaining_) {
        std::fprintf(stderr, "[3d] checkpoint adaptive fields exceed the HBM budget\n");
        return false;
    }
    for (auto& brick : pending)
        for (auto& field : brick.fields)
            if (!allocate_buffer(&field, brick.bytes)) return false;
    for (auto& brick : pending) {
        for (int buffer = 0; buffer < 2; ++buffer)
            h_promoted_phi_[buffer][brick.id] = brick.fields[buffer].get();
        h_promoted_ids_.push_back(brick.id);
        promoted_edge_ = std::max(promoted_edge_, brick.new_edge);
    }
    const bool installed = upload_promoted_tables();
    const bool drained = checked(cudaStreamSynchronize(stream_), "install checkpoint storage");
    if (!installed || !drained) {
        for (auto& brick : pending)
            for (int buffer = 0; buffer < 2; ++buffer)
                h_promoted_phi_[buffer][brick.id] = nullptr;
        h_promoted_ids_.clear();
        promoted_edge_ = 0;
        return false;
    }
    for (auto& brick : pending)
        for (auto& field : brick.fields) (void)field.release();
    adaptive_budget_remaining_ -= bytes;
    required_device_bytes_ += bytes;
    return true;
}

bool Sim3D::compact_promoted_fields() {
    if (h_promoted_ids_.empty()) return true;
    const std::size_t probe_bytes = static_cast<std::size_t>(params_.num_cells) *
        (sizeof(int) + sizeof(std::uint32_t));
    if (probe_bytes > adaptive_budget_remaining_) return true;
    if (gather_state_ && !gather_state_()) return false;
    // Bounding boxes must describe the accepted field, not the previous step.
    if (!refresh_measurements(false, surface_current_, false) ||
        !checked(cudaStreamSynchronize(stream_), "measure compaction candidates")) return false;
    std::vector<CellState3D> states(params_.num_cells);
    if (!checked(cudaMemcpy(states.data(), d_cells_, states.size() * sizeof(CellState3D),
                            cudaMemcpyDeviceToHost), "read compaction states")) return false;
    std::vector<int> candidates(states.size(), 0);
    bool any = false;
    for (int id : h_promoted_ids_) {
        const auto& cell = states[id];
        const int edge = cell_support_edge(cell, B_);
        const int lo[3] = {cell.bb_lo_x, cell.bb_lo_y, cell.bb_lo_z};
        const int hi[3] = {cell.bb_hi_x, cell.bb_hi_y, cell.bb_hi_z};
        const int smaller = shrink_brick_edge(edge, B_, lo, hi);
        if (smaller < edge) { candidates[id] = smaller; any = true; }
    }
    if (!any) return true;
    std::vector<std::uint32_t> rejected;
    if (!check_crops(d_promoted_phi_[current_promoted_phi_index_], d_cells_,
                     candidates, B_, stream_, field_storage_, &rejected)) return false;
    std::vector<std::pair<int, int>> requested;
    std::size_t temporary_bytes = 0;
    for (int id : h_promoted_ids_) {
        if (!candidates[id] || rejected[id]) continue;
        std::size_t bytes = 0, pair_bytes = 0;
        if (candidates[id] > B_ &&
            (!field_storage_.checked_bytes(candidates[id], &bytes) ||
             !checked_mul_size(bytes, 2, &pair_bytes))) return false;
        // Compaction is optional: postpone a replacement whose transactional
        // allocation does not fit, rather than stopping a valid simulation.
        if (pair_bytes > adaptive_budget_remaining_ - temporary_bytes) continue;
        temporary_bytes += pair_bytes;
        requested.emplace_back(id, candidates[id]);
    }
    if (requested.empty()) return true;
    if (!resize_cells(requested, &states) || !reconstruct_current_S() ||
        !refresh_measurements(false, surface_current_, false)) return false;
    return synchronize_and_check("adaptive compaction");
}

} // namespace pf3d
