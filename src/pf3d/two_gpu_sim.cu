#include "pf3d/two_gpu_sim.cuh"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace pf3d {
namespace {

bool checked(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "[two-gpu] %s: %s\n", operation,
                 cudaGetErrorString(result));
    return false;
}

bool fail(const char* message) {
    std::fprintf(stderr, "[two-gpu] %s\n", message);
    return false;
}

// Quantized contributions are nonnegative integers. Joining disjoint cell
// cohorts therefore preserves the single-device sum exactly unless it overflows.
__global__ void add_rows(std::uint32_t* sum, const std::uint32_t* received,
                         int pitch, int ny, int lo, int rows, int zlo, int planes,
                         std::uint32_t* flags) {
    const std::size_t band_words = static_cast<std::size_t>(pitch) * rows;
    // Walk contiguous rows within each plane without per-word division/modulo.
    for (std::size_t z = blockIdx.y; z < static_cast<std::size_t>(planes); z += gridDim.y) {
        const std::size_t base = ((z + zlo) * ny + lo) * pitch;
        for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x; i < band_words;
             i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
            const std::size_t index = base + i;
            const std::uint32_t a = sum[index], b = received[index];
            if (b > 0xffffffffu - a)
                atomicAdd(flags + FLAG3D_S_OVERFLOW, 1u);
            sum[index] = a + b;
        }
    }
}

}  // namespace

TwoGpuSim3D::~TwoGpuSim3D() {
    for (int r = 0; r < 2; ++r) {
        if (!sim_[r]) continue;
        sim_[r]->distributed_step_ = {};
        sim_[r]->gather_state_ = {};
        if (!select(r)) continue;
        checked(cudaStreamSynchronize(sim_[r]->stream_), "drain owned work");
        if (exchange_streams_[r])
            checked(cudaStreamSynchronize(exchange_streams_[r]), "drain peer copies");
    }
    // Peer copies may reference either allocation. Drain both streams before
    // releasing either rank's buffers or cross-device synchronization events.
    for (int r = 0; r < 2; ++r) {
        if (!select(r)) continue;
        if (exchange_ready_[r])
            checked(cudaEventDestroy(exchange_ready_[r]), "destroy exchange-ready event");
        if (exchange_sent_[r])
            checked(cudaEventDestroy(exchange_sent_[r]), "destroy exchange-sent event");
        if (exchange_streams_[r])
            checked(cudaStreamDestroy(exchange_streams_[r]), "destroy exchange stream");
        cudaFree(d_ids_[r]);
        cudaFree(d_receive_[r]);
    }
    for (HostSnapshot* snapshot : {&downloads_[0], &downloads_[1], &upload_})
        if (snapshot->cells)
            checked(cudaFreeHost(snapshot->cells), "release pinned metadata");
}

bool TwoGpuSim3D::select(int rank) const {
    return sim_[rank] && checked(cudaSetDevice(sim_[rank]->options_.device),
                                 "select device");
}

bool TwoGpuSim3D::wait_all() const {
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !checked(cudaStreamSynchronize(sim_[r]->stream_),
                                   "wait for owned work")) return false;
    return select(0);
}

bool TwoGpuSim3D::preflight(const SimParams3D& params, RunOptions3D* options) {
    if (!params.bounded_z())
        return fail("two-device execution requires --geometry slab or channel");
    if (params.hard_wall_channel() &&
        (!params.resolved_wall_channel() || !params.wall_repulsion_active()))
        return fail("two-device channel execution requires positive resolved steric walls");
    if (options->storage_mode != StorageMode3D::Auto &&
        options->storage_mode != StorageMode3D::Throughput)
        return fail("the two-device prototype requires throughput storage");
    int count = 0;
    if (!checked(cudaGetDeviceCount(&count), "count devices")) return false;
    const int first = options->device;
    if (first < 0 || first >= count || peer_device_ < 0 || peer_device_ >= count ||
        first == peer_device_)
        return fail("two distinct visible CUDA devices are required");
    cudaDeviceProp properties[2]{};
    const int devices[2] = {first, peer_device_};
    for (int r = 0; r < 2; ++r) {
        int access = 0;
        if (!checked(cudaGetDeviceProperties(&properties[r], devices[r]), "device properties") ||
            !checked(cudaDeviceCanAccessPeer(&access, devices[r], devices[1-r]), "peer capability"))
            return false;
        if (!access) return fail("bidirectional direct CUDA peer access is required");
        if (!checked(cudaSetDevice(devices[r]), "select peer device")) return false;
        const cudaError_t enabled = cudaDeviceEnablePeerAccess(devices[1-r], 0);
        if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else if (!checked(enabled, "enable peer access")) return false;
    }
    if (properties[0].major != properties[1].major ||
        properties[0].minor != properties[1].minor ||
        properties[0].multiProcessorCount != properties[1].multiProcessorCount)
        return fail("this prototype requires two matching GPUs");
    options->storage_mode = StorageMode3D::Throughput;
    return checked(cudaSetDevice(first), "select primary device");
}

bool TwoGpuSim3D::init_fresh(const SimParams3D& params, const RunOptions3D& options) {
    if (sim_[0]) return fail("construct a new coordinator to initialize another run");
    RunOptions3D resolved = options;
    if (!preflight(params, &resolved)) return false;
    sim_[0] = std::make_unique<Sim3D>();
    return sim_[0]->init_fresh(params, resolved) && initialize();
}

bool TwoGpuSim3D::init_checkpoint(const CheckpointMeta3D& meta,
                                 const std::string& path,
                                 const RunOptions3D& options, double end) {
    if (sim_[0]) return fail("construct a new coordinator to initialize another run");
    RunOptions3D resolved = options;
    if (!preflight(meta.params, &resolved)) return false;
    sim_[0] = std::make_unique<Sim3D>();
    return sim_[0]->init_checkpoint(meta, path, resolved, end) && initialize();
}

bool TwoGpuSim3D::read_states(int rank) {
    auto& sim = *sim_[rank];
    return select(rank) && checked(cudaMemcpyAsync(downloads_[rank].cells, sim.d_cells_,
        canonical_.size() * sizeof(CellState3D), cudaMemcpyDeviceToHost,
        sim.stream_), "read cell metadata") &&
        checked(cudaStreamSynchronize(sim.stream_), "wait for cell metadata");
}

bool TwoGpuSim3D::copy_field(int id, int source, int destination) {
    if (source == destination) return true;
    auto& src = *sim_[source];
    auto& dst = *sim_[destination];
    const int edge = cell_support_edge(canonical_[static_cast<std::size_t>(id)], src.B_);
    std::size_t words = 0, bytes = 0;
    if (src.B_ != dst.B_ || src.field_storage_.z_cap != dst.field_storage_.z_cap ||
        edge < src.B_ || !src.field_storage_.checked_words(edge, &words) ||
        !checked_mul_size(words, sizeof(float), &bytes))
        return fail("invalid owned-cell storage size");
    const bool promoted = edge > src.B_;
    const float* from = promoted
        ? src.h_promoted_phi_[src.current_promoted_phi_index_][id]
        : src.current_phi() + static_cast<std::size_t>(id) * words;
    float* to = promoted
        ? dst.h_promoted_phi_[dst.current_promoted_phi_index_][id]
        : dst.current_phi() + static_cast<std::size_t>(id) * words;
    if (!from || !to) return fail("missing storage for an owned cell");
    return select(source) && checked(cudaMemcpyPeerAsync(to, dst.options_.device,
        from, src.options_.device, bytes, src.stream_),
        "transfer owned phase field");
}

bool TwoGpuSim3D::clone_peer() {
    auto& primary = *sim_[0];
    // Rebuild from the accepted canonical state after adaptive storage growth.
    // The failed timestep is retried only after the replica is ready.
    sim_[1].reset();
    sim_[1] = std::make_unique<Sim3D>();
    auto& peer = *sim_[1];
    RunOptions3D options = primary.options_;
    options.device = peer_device_;
    options.trajectory_path.clear();
    options.boundary_path.clear();
    options.boundary_interval = 0;
    options.boundary_compress = false;
    options.boundary_projection = boundary::Projection::Maximum;
    options.checkpoint_dir.clear();
    options.final_checkpoint = false;
    options.measure_shards = primary.measurement_shards_;
    options.measure_shards_supplied = true;
    if (!peer.allocate(primary.params_, options, primary.B_, false,
                       primary.measurement_shards_)) return false;
    CheckpointMeta3D meta{};
    for (const auto& cell : canonical_)
        meta.storage_edges.push_back(cell.storage_edge > 0 ?
            static_cast<int>(cell.storage_edge) : primary.B_);
    if (!peer.install_checkpoint_promotions(meta)) return false;
    // The exchange buffers survive replica reconstruction. allocate() already
    // observes their resident bytes; only this not-yet-allocated ID list needs
    // reserving again after adaptive recovery.
    if (d_receive_[1]) {
        const std::size_t bytes = canonical_.size() * sizeof(int);
        if (bytes > peer.adaptive_budget_remaining_)
            return fail("promoted owner list exceeds the replica memory budget");
        peer.adaptive_budget_remaining_ -= bytes;
        peer.required_device_bytes_ += bytes;
    }
    for (int id = 0; id < primary.params_.num_cells; ++id)
        if (!copy_field(id, 0, 1)) return false;
    if (!wait_all()) return false;
    peer.steps_done_ = primary.steps_done_;
    peer.recovery_events_ = primary.recovery_events_;
    peer.volume_current_ = primary.volume_current_;
    peer.surface_current_ = primary.surface_current_;
    if (!select(1) || !checked(cudaMemcpy(peer.d_step_, &peer.steps_done_,
         sizeof(peer.steps_done_), cudaMemcpyHostToDevice), "replicate accepted step"))
        return false;
    mirrored_recoveries_ = primary.recovery_events_;
    return select(0);
}

bool TwoGpuSim3D::synchronize_peer_growth() {
    auto& primary = *sim_[0];
    auto& peer = *sim_[1];
    if (peer.B_ != primary.B_ ||
        peer.field_storage_.z_cap != primary.field_storage_.z_cap ||
        peer.step_pending_ || primary.step_pending_ ||
        peer.phi_buffers_ != 2 || peer.S_buffers_ != 2)
        return clone_peer();
    if (!read_states(1)) return false;
    std::vector<CellState3D> peer_states(downloads_[1].cells,
                                       downloads_[1].cells + canonical_.size());
    std::vector<std::pair<int, int>> growth;
    std::size_t temporary_bytes = 0;
    for (std::size_t id = 0; id < canonical_.size(); ++id) {
        const int old_edge = cell_support_edge(peer_states[id], peer.B_);
        const int new_edge = cell_support_edge(canonical_[id], primary.B_);
        // Nonowner fields can be stale. Growth preserves them without reading
        // beyond their allocation, but they cannot certify a lossless crop.
        if (old_edge < peer.B_ || new_edge < old_edge ||
            peer_states[id].global_id != canonical_[id].global_id)
            return clone_peer();
        if (new_edge > old_edge) {
            std::size_t bytes = 0, pair_bytes = 0;
            if (!peer.field_storage_.checked_bytes(new_edge, &bytes) ||
                !checked_mul_size(bytes, 2, &pair_bytes))
                return fail("invalid peer replacement storage size");
            // A full rebuild can fit when retaining old and new bricks at
            // once would exceed the configured temporary-allocation budget.
            if (pair_bytes > peer.adaptive_budget_remaining_ - temporary_bytes)
                return clone_peer();
            temporary_bytes += pair_bytes;
            growth.emplace_back(static_cast<int>(id), new_edge);
        }
    }

    peer.selection_ = {};
    peer.owned_promoted_count_ = -1;
    if (!select(1) || !peer.resize_cells(growth, &peer_states)) return false;

    // Recovery may remeasure or retry more than the resized cells. Refresh
    // every future peer owner's accepted field, not merely the growth batch.
    // Other replicas remain unused until an ownership transfer copies them.
    std::size_t refreshed = 0;
    for (std::size_t id = 0; id < canonical_.size(); ++id) {
        const auto& cell = canonical_[id];
        const AllocatedBrick brick{cell.origin_y, cell.origin_z, cell.storage_edge};
        int owner = -1;
        if (!cell_owner_rank(brick, primary.B_, geometry_, &owner))
            return fail("invalid accepted cell ownership during peer growth");
        if (owner == 1) {
            if (!copy_field(static_cast<int>(id), 0, 1)) return false;
            ++refreshed;
        }
    }
    if (!wait_all()) return false;
    peer.steps_done_ = primary.steps_done_;
    peer.recovery_events_ = primary.recovery_events_;
    peer.volume_current_ = primary.volume_current_;
    peer.surface_current_ = primary.surface_current_;
    if (!select(1) ||
        !checked(cudaMemcpy(peer.d_step_, &peer.steps_done_,
                            sizeof(peer.steps_done_), cudaMemcpyHostToDevice),
                 "replicate accepted step after peer growth") ||
        !checked(cudaMemset(peer.d_support_requests_, 0,
                            canonical_.size() * sizeof(std::uint32_t)),
                 "clear resolved peer support requests")) return false;
    mirrored_recoveries_ = primary.recovery_events_;
    std::printf("[two-gpu] synchronized peer growth: %zu resized, %zu accepted "
                "fields refreshed; base/workspace allocations retained\n",
                growth.size(), refreshed);
    return select(0);
}

bool TwoGpuSim3D::initialize() {
    auto& primary = *sim_[0];
    const auto& layout = primary.layout_;
    geometry_ = {layout.nx, layout.ny, layout.nz, layout.pitch_x, layout.boundary_flags};
    if (validate(geometry_) != TwoRankValidation::Ok)
        return fail("unsupported bounded-z exchange geometry");
    canonical_.resize(static_cast<std::size_t>(primary.params_.num_cells));
    const std::size_t cell_bytes = canonical_.size() * sizeof(CellState3D);
    for (HostSnapshot* snapshot : {&downloads_[0], &downloads_[1], &upload_}) {
        if (!checked(cudaHostAlloc(reinterpret_cast<void**>(&snapshot->cells),
                     cell_bytes + sizeof(flags_), cudaHostAllocPortable),
                     "allocate pinned metadata")) return false;
        snapshot->flags = reinterpret_cast<std::uint32_t*>(
            snapshot->cells + canonical_.size());
    }
    if (!read_states(0)) return false;
    std::copy_n(downloads_[0].cells, canonical_.size(), canonical_.begin());
    if (!clone_peer()) return false;
    const std::size_t receive_bytes = primary.S_words_ * sizeof(std::uint32_t);
    const std::size_t id_bytes = canonical_.size() * sizeof(int);
    for (int r = 0; r < 2; ++r) {
        auto& sim = *sim_[r];
        const std::size_t reserved = receive_bytes + 2 * id_bytes;
        if (reserved > sim.adaptive_budget_remaining_)
            return fail("peer-exchange workspace exceeds the configured memory budget");
        if (!select(r) ||
            !checked(cudaMalloc(reinterpret_cast<void**>(&d_receive_[r]), receive_bytes),
                     "allocate peer receive workspace") ||
            !checked(cudaMalloc(reinterpret_cast<void**>(&d_ids_[r]), id_bytes),
                     "allocate owner list") ||
            !checked(cudaEventCreateWithFlags(&exchange_ready_[r], cudaEventDisableTiming),
                     "create exchange-ready event") ||
            !checked(cudaEventCreateWithFlags(&exchange_sent_[r], cudaEventDisableTiming),
                     "create exchange-sent event") ||
            !checked(cudaStreamCreateWithFlags(&exchange_streams_[r], cudaStreamNonBlocking),
                     "create exchange stream")) return false;
        sim.adaptive_budget_remaining_ -= reserved;
        sim.required_device_bytes_ += reserved;
    }
    if (!select(0) || !checked(cudaMemcpy(flags_.data(), primary.d_flags_,
         sizeof(flags_), cudaMemcpyDeviceToHost), "read initial integrity flags")) return false;
    primary.distributed_step_ = [this] { return distributed_step(); };
    primary.gather_state_ = [this] { return gather_state(); };
    std::printf("[two-gpu] devices %d,%d; y-partitioned whole cells; "
                "replicated storage; base measurement shards %d\n",
                primary.options_.device, peer_device_, primary.measurement_shards_);
    return select(0);
}

bool TwoGpuSim3D::enqueue_state_broadcast() {
    // Callers must have drained previous uploads before reusing this pinned
    // buffer. It stays separate from canonical_ while the CPU replans ownership.
    std::copy(canonical_.begin(), canonical_.end(), upload_.cells);
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !checked(cudaMemcpyAsync(sim_[r]->d_cells_, upload_.cells,
             canonical_.size() * sizeof(CellState3D), cudaMemcpyHostToDevice,
             sim_[r]->stream_), "replicate accepted metadata")) return false;
    return select(0);
}

bool TwoGpuSim3D::configure_ownership(bool canonical_fields) {
    if (!canonical_fields && planned_bricks_.size() == canonical_.size()) {
        bool unchanged = true;
        for (std::size_t i = 0; i < canonical_.size(); ++i) {
            const auto& cell = canonical_[i];
            const auto& brick = planned_bricks_[i];
            unchanged = unchanged && cell.origin_y == brick.origin_y &&
                cell.origin_z == brick.origin_z && cell.storage_edge == brick.storage_edge;
        }
        if (unchanged) return select(0);
    }
    std::vector<AllocatedBrick> bricks;
    std::vector<int> next_owners;
    bricks.reserve(canonical_.size());
    next_owners.reserve(canonical_.size());
    bool migrated = false;
    bool storage_changed = planned_bricks_.size() != canonical_.size();
    for (std::size_t i = 0; i < canonical_.size(); ++i) {
        const auto& cell = canonical_[i];
        const AllocatedBrick brick{cell.origin_y, cell.origin_z, cell.storage_edge};
        int rank = -1;
        if (!cell_owner_rank(brick, sim_[0]->B_, geometry_, &rank))
            return fail("invalid accepted cell ownership");
        bricks.push_back(brick);
        next_owners.push_back(rank);
        if (!storage_changed && cell.storage_edge != planned_bricks_[i].storage_edge)
            storage_changed = true;
        if (!owners_.empty() && owners_[i] != rank) {
            if (!copy_field(static_cast<int>(i), canonical_fields ? 0 : owners_[i], rank))
                return false;
            migrated = true;
        }
    }
    for (int r = 0; r < 2; ++r) {
        plans_[r] = build_seam_exchange_plan(geometry_, sim_[r]->B_, r, bricks.data(),
                                             bricks.size(), sizeof(std::uint32_t));
        if (plans_[r].status != SeamPlanStatus::Ok) return fail("invalid seam plan");
    }
    // Recentering changes exchange bounds, not the owner lists. Canonical
    // gathers still require reinstalling the selections they cleared.
    if (!canonical_fields && !migrated && !storage_changed && !owners_.empty()) {
        planned_bricks_ = std::move(bricks);
        return select(0);
    }
    if (!wait_all()) return false;
    owners_ = std::move(next_owners);
    ids_[0].clear(); ids_[1].clear();
    for (std::size_t i = 0; i < owners_.size(); ++i)
        ids_[owners_[i]].push_back(static_cast<int>(i));
    for (int r = 0; r < 2; ++r) {
        auto& sim = *sim_[r];
        if (!select(r)) return false;
        if (!sim.d_owned_promoted_ids_ &&
            !checked(cudaMalloc(reinterpret_cast<void**>(&sim.d_owned_promoted_ids_),
                                canonical_.size() * sizeof(int)), "allocate promoted owner list"))
            return false;
        if (!ids_[r].empty() && !checked(cudaMemcpyAsync(d_ids_[r], ids_[r].data(),
             ids_[r].size() * sizeof(int), cudaMemcpyHostToDevice, sim.stream_),
             "upload cell owners")) return false;
        std::vector<int> promoted;
        for (int id : ids_[r])
            if (canonical_[id].storage_edge > static_cast<std::uint32_t>(sim.B_))
                promoted.push_back(id);
        if (!promoted.empty() && !checked(cudaMemcpyAsync(sim.d_owned_promoted_ids_,
             promoted.data(), promoted.size() * sizeof(int), cudaMemcpyHostToDevice,
             sim.stream_), "upload promoted owners")) return false;
        sim.selection_ = {d_ids_[r], static_cast<int>(ids_[r].size())};
        sim.owned_promoted_count_ = static_cast<int>(promoted.size());
        if (!checked(cudaStreamSynchronize(sim.stream_), "finish ownership upload")) return false;
    }
    if (canonical_fields || migrated) {
        for (int r = 0; r < 2; ++r)
            if (!select(r) || !sim_[r]->reconstruct_current_S()) return false;
        if (!exchange_sum(false) || !wait_all() || !merge_flags()) return false;
        // Reconstruction can raise a flag on either device after the normal
        // commit check. Settle it before a healthy poll can observe this state.
        for (int f = 0; f < FLAG3D_COUNT; ++f)
            if (flag3d_is_fatal(static_cast<Flag3D>(f)) && flags_[f] != 0)
                return fail("integrity failure reconstructing accepted state; durable checkpoint retained");
    }
    if (planned_bricks_.empty()) {
        std::printf("[two-gpu] initial owners %zu/%zu; seam bytes %zu/%zu\n",
                    ids_[0].size(), ids_[1].size(),
                    plans_[0].payload_bytes, plans_[1].payload_bytes);
        std::printf("[two-gpu] split base-update CTAs/cell %d/%d\n",
                    sim_[0]->fast_base_update_shards(UpdateTilePass3D::Boundary),
                    sim_[1]->fast_base_update_shards(UpdateTilePass3D::Boundary));
    }
    planned_bricks_ = std::move(bricks);
    return select(0);
}

bool TwoGpuSim3D::exchange_sum(bool pending) {
    return start_exchange(pending) && finish_exchange(pending);
}

bool TwoGpuSim3D::start_exchange(bool pending) {
    std::uint32_t* sums[2] = {
        pending ? sim_[0]->pending_update_.S_out : sim_[0]->current_S(),
        pending ? sim_[1]->pending_update_.S_out : sim_[1]->current_S()};
    const std::size_t pitch_bytes = static_cast<std::size_t>(geometry_.pitch_x) *
                                     sizeof(std::uint32_t);
    // Each record follows the outgoing boundary tiles and consumption of the
    // previous incoming copy. Interior tiles cannot write the transmitted band.
    // Record both generations before enqueueing cross-device waits.
    for (int r = 0; r < 2; ++r)
        if (!select(r) ||
            !checked(cudaEventRecord(exchange_ready_[r], sim_[r]->stream_),
                     "record exchange readiness")) return false;
    for (int r = 0; r < 2; ++r) {
        const auto& plan = plans_[r];
        if (!select(r) ||
            !checked(cudaStreamWaitEvent(exchange_streams_[r], exchange_ready_[r], 0),
                     "wait for outgoing boundary tiles") ||
            !checked(cudaStreamWaitEvent(exchange_streams_[r], exchange_ready_[1-r], 0),
                     "wait for peer receive workspace")) return false;
        for (int b = 0; plan.payload_bytes != 0 && b < plan.bands.rows.count; ++b) {
            const auto rows = plan.bands.rows.pieces[b];
            cudaMemcpy3DPeerParms copy{};
            copy.srcPtr = make_cudaPitchedPtr(sums[r], pitch_bytes, pitch_bytes, geometry_.ny);
            copy.dstPtr = make_cudaPitchedPtr(d_receive_[1-r], pitch_bytes, pitch_bytes, geometry_.ny);
            copy.srcPos = copy.dstPos = make_cudaPos(0, rows.lo, plan.z.lo);
            copy.srcDevice = sim_[r]->options_.device;
            copy.dstDevice = sim_[1-r]->options_.device;
            copy.extent = make_cudaExtent(pitch_bytes, rows.hi - rows.lo, plan.z.hi - plan.z.lo);
            if (!checked(cudaMemcpy3DPeerAsync(&copy, exchange_streams_[r]), "exchange seam rows"))
                return false;
        }
        if (!checked(cudaEventRecord(exchange_sent_[r], exchange_streams_[r]),
                     "record completed peer copy")) return false;
    }
    return select(0);
}

bool TwoGpuSim3D::finish_exchange(bool pending) {
    std::uint32_t* sums[2] = {
        pending ? sim_[0]->pending_update_.S_out : sim_[0]->current_S(),
        pending ? sim_[1]->pending_update_.S_out : sim_[1]->current_S()};
    // Interior work precedes this join in each compute stream. Both sends must
    // finish before either receive is added: an already-merged sum
    // must never be sent back to its original contributor.
    for (int r = 0; r < 2; ++r) {
        if (!select(r) ||
            !checked(cudaStreamWaitEvent(sim_[r]->stream_, exchange_sent_[r], 0),
                     "wait for outgoing seam copy") ||
            !checked(cudaStreamWaitEvent(sim_[r]->stream_, exchange_sent_[1-r], 0),
                     "wait for incoming seam copy")) return false;
        const auto& incoming = plans_[1-r];
        if (incoming.payload_bytes == 0) continue;
        for (int b = 0; b < incoming.bands.rows.count; ++b) {
            const auto rows = incoming.bands.rows.pieces[b];
            const std::size_t band_words = static_cast<std::size_t>(geometry_.pitch_x) *
                                            (rows.hi - rows.lo);
            const unsigned x_blocks = static_cast<unsigned>(
                std::min<std::size_t>(32, (band_words + 255) / 256));
            const dim3 grid(x_blocks, std::min(32, incoming.z.hi - incoming.z.lo));
            add_rows<<<grid, 256, 0, sim_[r]->stream_>>>(sums[r], d_receive_[r],
                geometry_.pitch_x, geometry_.ny, rows.lo, rows.hi - rows.lo,
                incoming.z.lo, incoming.z.hi - incoming.z.lo, sim_[r]->d_flags_);
        }
        if (!checked(cudaGetLastError(), "merge seam rows")) return false;
    }
    return select(0);
}

bool TwoGpuSim3D::merge_flags(bool include_states) {
    // Queue both ranks before waiting. After commit the cell snapshots share
    // this barrier; no second round trip is needed to decide ownership.
    for (int r = 0; r < 2; ++r) {
        auto& sim = *sim_[r];
        if (!select(r) || !checked(cudaMemcpyAsync(downloads_[r].flags, sim.d_flags_,
             sizeof(flags_), cudaMemcpyDeviceToHost, sim.stream_),
             "read rank integrity flags")) return false;
        if (include_states && !checked(cudaMemcpyAsync(downloads_[r].cells, sim.d_cells_,
             canonical_.size() * sizeof(CellState3D), cudaMemcpyDeviceToHost,
             sim.stream_), "read committed cell metadata")) return false;
    }
    if (!wait_all()) return false;
    for (std::size_t f = 0; f < flags_.size(); ++f) {
        const std::uint64_t sum = static_cast<std::uint64_t>(flags_[f]) +
            static_cast<std::uint32_t>(downloads_[0].flags[f] - flags_[f]) +
            static_cast<std::uint32_t>(downloads_[1].flags[f] - flags_[f]);
        flags_[f] = static_cast<std::uint32_t>(std::min<std::uint64_t>(sum, 0xffffffffu));
    }
    // The wait above also completes previous uploads from this buffer. The
    // following kernels consume these flags in stream order on each device.
    std::copy(flags_.begin(), flags_.end(), upload_.flags);
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !checked(cudaMemcpyAsync(sim_[r]->d_flags_, upload_.flags,
             sizeof(flags_), cudaMemcpyHostToDevice, sim_[r]->stream_),
             "share integrity flags")) return false;
    return select(0);
}

bool TwoGpuSim3D::gather_state() {
    if (primary_canonical_) return select(0);
    if (!merge_flags(true)) return false;
    for (std::size_t id = 0; id < canonical_.size(); ++id)
        canonical_[id] = downloads_[owners_[id]].cells[id];
    for (int id : ids_[1]) if (!copy_field(id, 1, 0)) return false;
    if (!wait_all() || !enqueue_state_broadcast() || !wait_all()) return false;
    std::vector<std::uint32_t> requests(canonical_.size()), peer_requests(canonical_.size());
    if (!select(0) || !checked(cudaMemcpy(requests.data(), sim_[0]->d_support_requests_,
         requests.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read support requests") ||
        !select(1) || !checked(cudaMemcpy(peer_requests.data(), sim_[1]->d_support_requests_,
         requests.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read peer support requests"))
        return false;
    for (std::size_t id = 0; id < requests.size(); ++id) requests[id] |= peer_requests[id];
    auto& primary = *sim_[0];
    if (!select(0) || !checked(cudaMemcpy(primary.d_support_requests_, requests.data(),
         requests.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "join support requests")) return false;
    primary.selection_ = {};
    primary.owned_promoted_count_ = -1;
    if (!primary.reconstruct_current_S() ||
        !checked(cudaStreamSynchronize(primary.stream_), "assemble canonical sum")) return false;
    primary_canonical_ = true;
    return true;
}

bool TwoGpuSim3D::distributed_step() {
    auto& primary = *sim_[0];
    if (primary_canonical_) {
        if (!wait_all() || !read_states(0)) return false;
        std::copy_n(downloads_[0].cells, canonical_.size(), canonical_.begin());
        if (primary.recovery_events_ != mirrored_recoveries_ &&
            !synchronize_peer_growth()) return false;
        if (!select(0) || !checked(cudaMemcpy(flags_.data(), primary.d_flags_,
             sizeof(flags_), cudaMemcpyDeviceToHost), "read canonical flags")) return false;
        if (!select(1) || !checked(cudaMemcpy(sim_[1]->d_flags_, flags_.data(),
             sizeof(flags_), cudaMemcpyHostToDevice), "restore peer flags")) return false;
        sim_[1]->volume_current_ = primary.volume_current_;
        sim_[1]->surface_current_ = primary.surface_current_;
        if (!enqueue_state_broadcast() || !configure_ownership(true)) return false;
        primary_canonical_ = false;
    }
    for (int r = 0; r < 2; ++r) {
        const auto& plan = plans_[r];
        TileExchangeRegion3D region{};
        if (plan.payload_bytes != 0) {
            region.y_count = plan.bands.rows.count;
            for (int b = 0; b < region.y_count; ++b) {
                region.y_lo[b] = plan.bands.rows.pieces[b].lo;
                region.y_hi[b] = plan.bands.rows.pieces[b].hi;
            }
            region.z_lo = plan.z.lo;
            region.z_hi = plan.z.hi;
        }
        if (!select(r) || !sim_[r]->begin_step(&region)) return false;
    }
    if (!start_exchange(true)) return false;
    for (int r = 0; r < 2; ++r)
        if (!select(r) ||
            (sim_[r]->pending_update_.tile_pass == UpdateTilePass3D::Boundary &&
             !sim_[r]->enqueue_interior_update())) return false;
    if (!finish_exchange(true) || !merge_flags()) return false;
    bool fatal = false;
    for (int f = 0; f < FLAG3D_COUNT; ++f)
        fatal = fatal || (flag3d_is_fatal(static_cast<Flag3D>(f)) && flags_[f] != 0);
    for (int r = 0; r < 2; ++r)
        if (!select(r) || !sim_[r]->finish_step()) return false;
    if (!merge_flags(true)) return false;
    // Coordinate overflow may be discovered while committing origins, after
    // field repair. Such an unrecoverable commit must not publish any output.
    if (!fatal)
        for (int f = 0; f < FLAG3D_COUNT; ++f)
            if (flag3d_is_fatal(static_cast<Flag3D>(f)) && flags_[f] != 0)
                return fail("integrity failure during commit; durable checkpoint retained");
    if (fatal) {
        // Every rank rolled back under the same fatal flag. Only the canonical
        // solver performs adaptive recovery; the next step replicates its result.
        return gather_state() && primary.synchronize_and_check("distributed recovery");
    }
    for (std::size_t id = 0; id < canonical_.size(); ++id)
        canonical_[id] = downloads_[owners_[id]].cells[id];
    return enqueue_state_broadcast() && configure_ownership(false) && select(0);
}

bool TwoGpuSim3D::run() { return select(0) && sim_[0]->run(); }

bool TwoGpuSim3D::bench(int steps, double* milliseconds_per_step) {
    // GPU0 events include its idle waits for peer/host coordination. The shared
    // benchmark's accepted-step and recovery guards still apply.
    if (!select(0)) return false;
    if (sim_[0]->options_.bench_phases)
        std::printf("[two-gpu] primary-device phases only. Fast update phases "
                    "cover boundary tiles; repair/finalize/scatter includes "
                    "interior tiles, the exchange join, and pre-commit control; "
                    "host/output gap includes post-commit control and ownership.\n");
    return sim_[0]->bench(steps, milliseconds_per_step);
}

}  // namespace pf3d
