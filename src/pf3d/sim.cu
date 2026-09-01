#include "../../include/pf3d/sim.cuh"
#include "../../include/pf3d/measure_shards.hpp"
#include "../../include/pf3d/trajectory_header.hpp"
#include "../../include/pf3d/reference.hpp"
#include "../../include/palmieri_initializer.hpp"
#include "../../include/relaxed_centres.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace pf3d {
namespace {

constexpr std::size_t kAllocationReserve = 256u * 1024u * 1024u;
constexpr int kHostPollEvery = 16;
constexpr int kMaximumShiftPerStep = 4;
constexpr int kPhaseEventSlots = 7;
// Promoted cubes contain many more tiles than base bricks. When only a few
// cells are promoted, wider spatial sharding exposes enough independent work
// to occupy the device.
constexpr int kMaximumPromotedShards = 64;

bool gpu_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[3d] %s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool add_bytes(std::size_t value, std::size_t* total) {
    if (!total || value > std::numeric_limits<std::size_t>::max() - *total)
        return false;
    *total += value;
    return true;
}

bool multiply_bytes(std::size_t a, std::size_t b, std::size_t* out) {
    return checked_mul_size(a, b, out);
}

bool copy_centered_cube(float* destination, int destination_edge,
                        const float* source, int source_edge,
                        cudaStream_t stream) {
    if (destination == nullptr || source == nullptr ||
        destination_edge < source_edge ||
        (destination_edge - source_edge) % 2 != 0)
        return false;
    const int offset = (destination_edge - source_edge) / 2;
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(source),
        static_cast<std::size_t>(source_edge) * sizeof(float),
        source_edge, source_edge);
    copy.dstPtr = make_cudaPitchedPtr(
        destination,
        static_cast<std::size_t>(destination_edge) * sizeof(float),
        destination_edge, destination_edge);
    copy.dstPos = make_cudaPos(
        static_cast<std::size_t>(offset) * sizeof(float),
        static_cast<std::size_t>(offset),
        static_cast<std::size_t>(offset));
    copy.extent = make_cudaExtent(
        static_cast<std::size_t>(source_edge) * sizeof(float),
        source_edge, source_edge);
    copy.kind = cudaMemcpyDeviceToDevice;
    return gpu_ok(cudaMemcpy3DAsync(&copy, stream),
                  "center adaptive phase field");
}

bool any_fatal_flag(const std::vector<std::uint32_t>& flags) {
    for (int i = 0; i < FLAG3D_COUNT; ++i)
        if (flag3d_is_fatal(static_cast<Flag3D>(i)) &&
            flags[static_cast<std::size_t>(i)] != 0u)
            return true;
    return false;
}

const char* flag_name(Flag3D flag) {
    switch (flag) {
        case FLAG3D_S_OVERFLOW: return "S_overflow";
        case FLAG3D_Q_CLAMP: return "phi_squared_out_of_range";
        case FLAG3D_S_NEGATIVE: return "S_self_subtraction_negative";
        case FLAG3D_NONFINITE: return "nonfinite_field";
        case FLAG3D_V_NONPOSITIVE: return "nonpositive_volume";
        case FLAG3D_SUPPORT_EXHAUSTED: return "support_margin_exhausted";
        case FLAG3D_SUPPORT_EDGE: return "support_at_brick_edge";
        case FLAG3D_DESTINATION_CLIP: return "recentring_destination_clip";
        case FLAG3D_INVALID_GEOMETRY: return "invalid_runtime_geometry";
        case FLAG3D_SLAB_TOP_CONTACT: return "slab_top_far_field_contact";
        default: return "unknown";
    }
}

struct ModeMemory {
    StorageMode3D mode = StorageMode3D::Auto;
    int phi_buffers = 0;
    int S_buffers = 0;
    int scratch_slots = 0;
    std::size_t bytes = 0;
    bool fits = false;
};

double sphere_sphericity(double volume, double surface) {
    if (!(volume > 0.0) || !(surface > 0.0)) return 0.0;
    return std::cbrt(kPi) * std::pow(6.0 * volume, 2.0 / 3.0) / surface;
}

double hemisphere_normalized_free_surface_compactness(double volume,
                                                       double surface) {
    if (!(volume > 0.0) || !(surface > 0.0)) return 0.0;
    const double normalization =
        2.0 * kPi * std::pow(3.0 / (2.0 * kPi), 2.0 / 3.0);
    return normalization * std::pow(volume, 2.0 / 3.0) / surface;
}

std::uint64_t schedule_after(std::uint64_t current, std::uint64_t interval) {
    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    return interval == 0 || interval > maximum - current
        ? maximum : current + interval;
}

std::uint64_t schedule_grid_after(std::uint64_t current,
                                  std::uint64_t interval) {
    if (interval == 0) return std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t remainder = current % interval;
    const std::uint64_t delta = remainder == 0 ? interval
                                               : interval - remainder;
    return schedule_after(current, delta);
}

void advance_schedule(std::uint64_t* scheduled, std::uint64_t interval) {
    if (scheduled) *scheduled = schedule_after(*scheduled, interval);
}

constexpr const char* kTrajectoryColumnsPeriodic3D =
    "# columns: time cell_id x y z vx vy vz px py pz gamma v_A "
    "surface sphericity volume phi_max";

constexpr const char* kTrajectoryColumnsSlab3D =
    "# columns: time cell_id x y height vx vy vz px py pz gamma v_A "
    "free_interface_proxy hemisphere_normalized_free_surface_compactness "
    "volume phi_max";

constexpr const char* kTrajectoryColumnsResolvedChannel3D =
    "# columns: time cell_id x y height vx vy vz px py pz gamma v_A "
    "interface_measure sphericity_proxy "
    "volume phi_max wall_overlap wall_overlap_fraction "
    "outside_slit_volume physical_penetration_fraction";

const char* trajectory_columns(const SimParams3D& p) {
    return p.substrate_slab() ? kTrajectoryColumnsSlab3D
         : p.resolved_wall_channel() ? kTrajectoryColumnsResolvedChannel3D
                                     : kTrajectoryColumnsPeriodic3D;
}

}  // namespace

volatile std::sig_atomic_t Sim3D::terminate_requested = 0;

const char* storage_mode_name(StorageMode3D mode) {
    switch (mode) {
        case StorageMode3D::Auto: return "auto";
        case StorageMode3D::Throughput: return "throughput";
        case StorageMode3D::Balanced: return "balanced";
        case StorageMode3D::Compact: return "compact";
        default: return "unknown";
    }
}

bool parse_storage_mode(const std::string& text, StorageMode3D* mode) {
    if (!mode) return false;
    if (text == "auto") *mode = StorageMode3D::Auto;
    else if (text == "throughput") *mode = StorageMode3D::Throughput;
    else if (text == "balanced") *mode = StorageMode3D::Balanced;
    else if (text == "compact") *mode = StorageMode3D::Compact;
    else return false;
    return true;
}

Sim3D::~Sim3D() {
    close_trajectory();
    if (stream_) cudaStreamSynchronize(stream_);
    clear_phase_events();
    if (bench_start_) cudaEventDestroy(bench_start_);
    if (bench_stop_) cudaEventDestroy(bench_stop_);
    if (d_trajectory_) cudaFree(d_trajectory_);
    if (d_verify_S_) cudaFree(d_verify_S_);
    if (d_verify_cells_) cudaFree(d_verify_cells_);
    if (d_moment_partials_) cudaFree(d_moment_partials_);
    if (d_promoted_partials_) cudaFree(d_promoted_partials_);
    if (d_flags_) cudaFree(d_flags_);
    if (d_work_cursor_) cudaFree(d_work_cursor_);
    if (d_step_) cudaFree(d_step_);
    if (d_centres_) cudaFree(d_centres_);
    if (d_support_requests_) cudaFree(d_support_requests_);
    if (d_accepted_cells_) cudaFree(d_accepted_cells_);
    if (d_cells_) cudaFree(d_cells_);
    for (auto& table : h_promoted_phi_)
        for (float* field : table)
            if (field) cudaFree(field);
    if (d_promoted_ids_) cudaFree(d_promoted_ids_);
    for (float** table : d_promoted_phi_)
        if (table) cudaFree(table);
    if (d_scratch_) cudaFree(d_scratch_);
    for (std::uint32_t*& field : d_S_)
        if (field) cudaFree(field);
    if (d_weighted_wall_psi_sq_) cudaFree(d_weighted_wall_psi_sq_);
    if (d_wall_psi_sq_) cudaFree(d_wall_psi_sq_);
    for (float*& field : d_phi_)
        if (field) cudaFree(field);
    if (stream_) cudaStreamDestroy(stream_);
}

bool Sim3D::allocate(const SimParams3D& params,
                     const RunOptions3D& options,
                     int requested_brick_edge,
                     bool need_initial_centres,
                     std::optional<int> checkpoint_measure_shards) {
    params_ = params;
    options_ = options;
    const bool measure_shards_requested =
        options.measure_shards_supplied || options.measure_shards != 0;
    if (measure_shards_requested &&
        !valid_base_measure_policy(options.measure_shards)) {
        std::fprintf(stderr,
            "[3d] base measurement shard policy %d is outside -1..64\n",
            options.measure_shards);
        return false;
    }
    if (checkpoint_measure_shards.has_value() &&
        !valid_checkpoint_base_measure_shards(*checkpoint_measure_shards)) {
        std::fprintf(stderr,
            "[3d] checkpoint stores an invalid base measurement shard "
            "count %d\n", *checkpoint_measure_shards);
        return false;
    }
    const PromotedMeasureReduction3D reduction{
        options_.promoted_measure_shards,
        options_.promoted_measure_auto_wave_ctas};
    const bool reduction_valid = options_.promoted_measure_shards < 0
        ? (options_.promoted_measure_auto_wave_ctas == 0 ||
           valid_checkpoint_promoted_measure_reduction(reduction))
        : (valid_promoted_measure_policy(options_.promoted_measure_shards) &&
           options_.promoted_measure_auto_wave_ctas == 0);
    if (!reduction_valid) {
        std::fprintf(stderr,
            "[3d] invalid promoted-measurement reduction policy\n");
        return false;
    }
    if (!std::isfinite(options.memory_fraction) ||
        options.memory_fraction < 0.50 || options.memory_fraction > 0.99) {
        std::fprintf(stderr,
                     "[3d] memory fraction must lie in [0.50, 0.99]\n");
        return false;
    }
    B_ = requested_brick_edge > 0 ? requested_brick_edge : params.brick_edge();
    const int minimum_edge = params.brick_edge();
    if (B_ < minimum_edge || B_ % kBrickAlignment != 0 ||
        params.minimum_domain_edge() <= B_) {
        std::fprintf(stderr,
            "[3d] brick edge %d is invalid; this parameter set requires at "
            "least %d (multiple of %d), and B must be smaller than every "
            "domain extent (minimum %d)\n",
            B_, minimum_edge, kBrickAlignment, params.minimum_domain_edge());
        return false;
    }
    if (!checked_cube_size(static_cast<std::size_t>(B_), &brick_words_)) {
        std::fprintf(stderr, "[3d] B^3 overflows host size_t\n");
        return false;
    }
    if (params.Nx > std::numeric_limits<int>::max() - 31) {
        std::fprintf(stderr, "[3d] domain side is too large to align safely\n");
        return false;
    }
    layout_.nx = params.Nx;
    layout_.ny = params.Ny;
    layout_.nz = params.Nz;
    layout_.pitch_x = 32 * ((params.Nx + 31) / 32);
    layout_.boundary_flags = params.boundary_flags;
    std::size_t S_plane_words = 0;
    if (!checked_mul_size(static_cast<std::size_t>(layout_.pitch_x),
                          static_cast<std::size_t>(layout_.ny),
                          &S_plane_words) ||
        !checked_mul_size(S_plane_words,
                          static_cast<std::size_t>(layout_.nz), &S_words_)) {
        std::fprintf(stderr, "[3d] pitched aggregate-field size overflowed\n");
        return false;
    }
    if (!valid_runtime_geometry(B_, layout_)) {
        std::fprintf(stderr, "[3d] invalid brick/domain/pitch geometry\n");
        return false;
    }
    maximum_support_edge_ =
        ((params.minimum_domain_edge() - 1) / kBrickAlignment) *
        kBrickAlignment;
    // Start with a modest on-demand tier; later exhaustion grows the common
    // enlarged-cell tier up to the largest cube that fits inside the domain.
    const std::int64_t first_growth = B_ >= 128
        ? std::max<std::int64_t>(224, static_cast<std::int64_t>(B_) + 64)
        : static_cast<std::int64_t>(B_) + 64;
    const int desired_support_edge =
        first_growth >= maximum_support_edge_
            ? maximum_support_edge_
            : round_up_to_multiple(static_cast<int>(first_growth),
                                   kBrickAlignment);
    promoted_edge_ = maximum_support_edge_ > B_
        ? desired_support_edge : 0;
    promoted_words_ = 0;
    if (promoted_edge_ > 0 &&
        (!valid_runtime_geometry(promoted_edge_, layout_) ||
         !checked_cube_size(static_cast<std::size_t>(promoted_edge_),
                            &promoted_words_))) {
        std::fprintf(stderr, "[3d] adaptive support edge is invalid\n");
        return false;
    }

    if (!gpu_ok(cudaSetDevice(options.device), "cudaSetDevice")) return false;
    cudaDeviceProp property{};
    if (!gpu_ok(cudaGetDeviceProperties(&property, options.device),
                "cudaGetDeviceProperties")) return false;
    std::size_t free_bytes = 0, total_bytes = 0;
    if (!gpu_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo"))
        return false;
    const std::size_t allocation_limit = free_bytes > kAllocationReserve
        ? static_cast<std::size_t>(
              options.memory_fraction * static_cast<double>(
                  free_bytes - kAllocationReserve))
        : 0;

    int measured_inplace = 0, measured_tiled = 0;
    int fast_inplace = 0, fast_tiled = 0;
    int measurement_sharded = 0, measured_update_sharded = 0;
    int fast_update_sharded = 0;
    if (!gpu_ok(configure_tile_kernel_shared_memory(),
                "tile-kernel shared-memory limit") ||
        !gpu_ok(detail::measured_update_occupancy(
                    &measured_tiled, &measured_inplace, layout_),
                "measured-update occupancy query") ||
        !gpu_ok(detail::fast_update_occupancy(
                    &fast_tiled, &fast_inplace, layout_),
                "fast-update occupancy query") ||
        !gpu_ok(detail::sharded_occupancy(
                    &measurement_sharded, &measured_update_sharded,
                    &fast_update_sharded, layout_),
                "sharded occupancy query"))
        return false;
    if (measured_inplace <= 0 || measured_tiled <= 0 || fast_inplace <= 0 ||
        fast_tiled <= 0 || measurement_sharded <= 0 ||
        measured_update_sharded <= 0 || fast_update_sharded <= 0) {
        std::fprintf(stderr,
            "[3d] occupancy query reported a non-runnable 3D kernel\n");
        return false;
    }
    const int active_inplace = std::max(
        1, std::max(measured_inplace, fast_inplace));
    const int active_tiled = std::max(
        1, std::max(measured_tiled, fast_tiled));
    const int desired_scratch = std::min(
        params.num_cells, property.multiProcessorCount * active_inplace);
    // Avoid adding another population-wide wave when all measurement blocks
    // can fit in one. Larger populations already span waves at one CTA/cell.
    const int measured_sharded_active =
        std::min(measurement_sharded, measured_update_sharded);
    // The resolved count fixes the moment-reduction grouping. Fresh runs
    // resolve the selected policy; continuations restore the checkpoint count.
    if (!checkpoint_measure_shards.has_value()) {
        measurement_shards_ = resolve_base_measure_shards(
            measure_shards_requested ? options.measure_shards : 0,
            params.num_cells, property.multiProcessorCount,
            measured_sharded_active);
    } else {
        int resolved_measure = 0;
        const BaseMeasureResumeResult resume = resolve_base_measure_resume(
            static_cast<std::uint64_t>(*checkpoint_measure_shards),
            measure_shards_requested, options.measure_shards,
            params.num_cells, property.multiProcessorCount,
            measured_sharded_active, &resolved_measure);
        if (resume == BaseMeasureResumeResult::Mismatch) {
            std::fprintf(stderr,
                "[3d] --measure-shards %d resolves %d base measurement "
                "CTAs/cell here, but the checkpoint's effective grouping "
                "is %d; drop the option to restore the stored grouping\n",
                options.measure_shards,
                resolve_base_measure_shards(options.measure_shards,
                                            params.num_cells,
                                            property.multiProcessorCount,
                                            measured_sharded_active),
                *checkpoint_measure_shards != 0
                    ? *checkpoint_measure_shards
                    : resolve_base_measure_shards(
                          0, params.num_cells,
                          property.multiProcessorCount,
                          measured_sharded_active));
            return false;
        }
        if (resume != BaseMeasureResumeResult::Ok) {
            std::fprintf(stderr,
                "[3d] checkpoint stores an invalid base measurement shard "
                "count\n");
            return false;
        }
        measurement_shards_ = resolved_measure;
    }
    fast_base_blocks_per_sm_ = fast_update_sharded;
    // The fast path has no persistent partial-sum layout, so it can use any
    // otherwise idle capacity in one population-wide occupancy wave.
    throughput_update_shards_ = occupancy_wave_shards(
        params.num_cells, property.multiProcessorCount, fast_update_sharded,
        kMaximumPromotedShards);
    if (options.fast_base_shards > 0) {
        // Benchmark override for the fast update. The deterministic
        // measurement path keeps its own count and partial-sum layout.
        throughput_update_shards_ = std::max(
            1, std::min(options.fast_base_shards, kMaximumPromotedShards));
    }
    sm_count_ = property.multiProcessorCount;
    if (!gpu_ok(detail::promoted_fast_update_occupancy(
                    &fast_promoted_blocks_per_sm_, layout_),
                "promoted fast-update occupancy query") ||
        fast_promoted_blocks_per_sm_ <= 0)
        return false;
    if (!gpu_ok(detail::promoted_measure_occupancy(
                    &promoted_measure_blocks_per_sm_, layout_),
                "promoted measurement occupancy query") ||
        promoted_measure_blocks_per_sm_ <= 0)
        return false;
    if (options_.promoted_measure_shards < 0 &&
        options_.promoted_measure_auto_wave_ctas == 0) {
        const std::int64_t wave = static_cast<std::int64_t>(sm_count_) *
                                  promoted_measure_blocks_per_sm_;
        if (wave <= 0 || wave > std::numeric_limits<int>::max()) {
            std::fprintf(stderr,
                "[3d] promoted measurement occupancy wave is invalid\n");
            return false;
        }
        options_.promoted_measure_auto_wave_ctas = static_cast<int>(wave);
    }

    std::size_t one_cell_phi = 0, one_phi = 0, one_S = 0;
    std::size_t wall_profile_bytes = 0, channel_wall_storage_bytes = 0;
    std::size_t auxiliary = 0;
    std::size_t moment_workspace = 0, moment_records = 0;
    std::size_t adaptive_reserve_bytes = 0;
    if (!multiply_bytes(brick_words_, sizeof(float), &one_cell_phi))
        return false;
    if (!multiply_bytes(brick_words_, static_cast<std::size_t>(params.num_cells),
                        &one_phi) ||
        !multiply_bytes(one_phi, sizeof(float), &one_phi) ||
        !multiply_bytes(S_words_, sizeof(std::uint32_t), &one_S) ||
        (measurement_shards_ > 1 &&
         (!multiply_bytes(static_cast<std::size_t>(params.num_cells),
                          static_cast<std::size_t>(measurement_shards_),
                          &moment_records) ||
          !multiply_bytes(moment_records, sizeof(MomentPartial3D),
                          &moment_workspace))))
        return false;
    if (params.resolved_wall_channel() &&
        !multiply_bytes(static_cast<std::size_t>(params.Nz), sizeof(float),
                        &wall_profile_bytes))
        return false;
    if (params.resolved_wall_channel() &&
        !multiply_bytes(wall_profile_bytes, 2, &channel_wall_storage_bytes))
        return false;
    if (promoted_words_ > 0 &&
        (!multiply_bytes(promoted_words_, sizeof(float),
                         &adaptive_reserve_bytes) ||
         !multiply_bytes(adaptive_reserve_bytes, 2,
                         &adaptive_reserve_bytes)))
        return false;
    if (!add_bytes(static_cast<std::size_t>(params.num_cells) *
                       2 * sizeof(CellState3D), &auxiliary) ||
        !add_bytes(static_cast<std::size_t>(params.num_cells) *
                       (2 * sizeof(float*) + sizeof(int) +
                        sizeof(std::uint32_t)), &auxiliary) ||
        (need_initial_centres &&
         !add_bytes(static_cast<std::size_t>(params.num_cells) * sizeof(Vec3),
                    &auxiliary)) ||
        (options.strict &&
         !add_bytes(static_cast<std::size_t>(params.num_cells) *
                        sizeof(VerifyCell3D), &auxiliary)) ||
        (!options.trajectory_path.empty() &&
         !add_bytes(static_cast<std::size_t>(params.num_cells) *
                        sizeof(TrajPackedCell3D), &auxiliary)) ||
        !add_bytes(moment_workspace, &auxiliary) ||
        !add_bytes((FLAG3D_COUNT + 4) * sizeof(std::uint64_t), &auxiliary) ||
        (params.resolved_wall_channel() &&
         !add_bytes(channel_wall_storage_bytes, &auxiliary)))
        return false;

    auto assess = [&](StorageMode3D mode, bool reserve_adaptive) {
        ModeMemory result{};
        if (reserve_adaptive &&
            adaptive_reserve_bytes > allocation_limit)
            return result;
        const std::size_t mode_limit = reserve_adaptive
            ? allocation_limit - adaptive_reserve_bytes : allocation_limit;
        result.mode = mode;
        result.phi_buffers = mode == StorageMode3D::Throughput ? 2 : 1;
        result.S_buffers = mode == StorageMode3D::Compact ? 1 : 2;
        std::size_t base = auxiliary;
        std::size_t phi_pool = 0, S_pool = 0;
        if (!multiply_bytes(one_phi,
                            static_cast<std::size_t>(result.phi_buffers),
                            &phi_pool) ||
            !multiply_bytes(one_S,
                             static_cast<std::size_t>(result.S_buffers),
                            &S_pool) ||
            !add_bytes(phi_pool, &base) || !add_bytes(S_pool, &base))
            return result;
        if (mode == StorageMode3D::Throughput) {
            result.bytes = base;
            result.fits = base <= mode_limit;
            return result;
        }
        if (base >= mode_limit || one_cell_phi == 0) return result;
        int slots = static_cast<int>(std::min<std::size_t>(
            desired_scratch, (mode_limit - base) / one_cell_phi));
        if (options.scratch_slots > 0) {
            if (options.scratch_slots > desired_scratch ||
                static_cast<std::size_t>(options.scratch_slots) >
                    (mode_limit - base) / one_cell_phi)
                return result;
            slots = options.scratch_slots;
        }
        if (slots < 1) return result;
        result.scratch_slots = slots;
        result.bytes = base + static_cast<std::size_t>(slots) * one_cell_phi;
        result.fits = result.bytes <= mode_limit;
        return result;
    };

    ModeMemory selected{};
    bool adaptive_headroom_reserved = false;
    if (options.storage_mode == StorageMode3D::Auto) {
        auto select_auto = [&](bool reserve_adaptive) {
            ModeMemory choice{};
            const ModeMemory throughput = assess(
                StorageMode3D::Throughput, reserve_adaptive);
            if (throughput.fits) return throughput;
            const ModeMemory balanced = assess(
                StorageMode3D::Balanced, reserve_adaptive);
            const ModeMemory compact = assess(
                StorageMode3D::Compact, reserve_adaptive);
            const int one_cta_per_sm =
                std::min(params.num_cells, property.multiProcessorCount);
            const bool balanced_has_useful_concurrency =
                balanced.scratch_slots >= one_cta_per_sm;

            // A nearly full balanced layout can leave only a handful of
            // persistent update blocks. Compact mode is faster in that regime
            // because its smaller aggregate-field footprint restores broad
            // CTA concurrency, despite rebuilding S after each update.
            if (balanced.fits &&
                (options.scratch_slots > 0 ||
                 balanced_has_useful_concurrency || !compact.fits)) {
                choice = balanced;
            } else if (compact.fits) {
                choice = compact;
            }
            return choice;
        };
        selected = select_auto(true);
        adaptive_headroom_reserved = selected.fits;
        if (!selected.fits) selected = select_auto(false);
    } else {
        selected = assess(options.storage_mode, false);
    }
    if (!selected.fits) {
        std::fprintf(stderr,
            "[3d] requested storage cannot fit safely: one phi brick pool "
            "%.2f GiB, one S field %.2f GiB, free %.2f GiB, allocation "
            "limit %.2f GiB. Try --memory-mode compact, a larger safe "
            "--memory-fraction, a smaller valid B, or fewer cells.\n",
            static_cast<double>(one_phi) / 1073741824.0,
            static_cast<double>(one_S) / 1073741824.0,
            static_cast<double>(free_bytes) / 1073741824.0,
            static_cast<double>(allocation_limit) / 1073741824.0);
        return false;
    }
    selected_mode_ = selected.mode;
    phi_buffers_ = selected.phi_buffers;
    S_buffers_ = selected.S_buffers;
    scratch_slots_ = selected.scratch_slots;
    required_device_bytes_ = selected.bytes;
    adaptive_budget_remaining_ = allocation_limit > selected.bytes
        ? allocation_limit - selected.bytes : 0;
    update_grid_blocks_ = std::min(
        params.num_cells, property.multiProcessorCount * active_tiled);

    std::printf("--- 3D device and storage ---\n");
    std::printf("  %s  cc %d.%d  %d SMs  %.2f GiB total, %.2f GiB free\n",
                property.name, property.major, property.minor,
                property.multiProcessorCount,
                static_cast<double>(total_bytes) / 1073741824.0,
                static_cast<double>(free_bytes) / 1073741824.0);
    std::printf("  domain %d x %d x %d (pitch %d), cell brick %d^3, N=%d\n",
                params.Nx, params.Ny, params.Nz, layout_.pitch_x, B_,
                params.num_cells);
    std::printf("  mode %s: phi x%d (%.2f GiB each), S x%d (%.2f GiB each), "
                "scratch slots %d, allocation %.2f GiB (limit %.2f GiB)\n",
                storage_mode_name(selected_mode_), phi_buffers_,
                static_cast<double>(one_phi) / 1073741824.0, S_buffers_,
                static_cast<double>(one_S) / 1073741824.0,
                scratch_slots_,
                static_cast<double>(required_device_bytes_) / 1073741824.0,
                static_cast<double>(allocation_limit) / 1073741824.0);
    std::printf("  measurement CTAs/cell %d; fast throughput-update CTAs/cell %d\n",
                measurement_shards_, throughput_update_shards_);
    std::printf("  promoted fast-update occupancy %d CTA/SM (shard cap %d)\n",
                fast_promoted_blocks_per_sm_, kMaximumPromotedShards);
    if (adaptive_headroom_reserved && adaptive_reserve_bytes > 0)
        std::printf("  auto mode retains %.3f GiB for one first-tier "
                    "adaptive cell\n",
                    static_cast<double>(adaptive_reserve_bytes) /
                        1073741824.0);
    else if (promoted_edge_ > B_ &&
             adaptive_budget_remaining_ < adaptive_reserve_bytes)
        std::fprintf(stderr,
            "[3d] warning: selected storage leaves %.3f GiB adaptive "
            "headroom, below the %.3f GiB first-tier requirement\n",
            static_cast<double>(adaptive_budget_remaining_) / 1073741824.0,
            static_cast<double>(adaptive_reserve_bytes) / 1073741824.0);
    std::printf("  measured/fast sharded occupancy %d/%d CTA/SM; strict "
                "throughput updates use the measurement count; %.1f KiB "
                "moment workspace\n",
                measured_sharded_active, fast_update_sharded,
                static_cast<double>(moment_workspace) / 1024.0);
    if (selected_mode_ != StorageMode3D::Throughput &&
        scratch_slots_ < desired_scratch) {
        std::fprintf(stderr,
            "[3d] warning: HBM limits scratch concurrency to %d/%d CTAs",
            scratch_slots_, desired_scratch);
        if (scratch_slots_ < property.multiProcessorCount)
            std::fprintf(stderr, " (less than one CTA per SM)");
        std::fprintf(stderr,
            "; reduce N/B or increase --memory-fraction for higher throughput\n");
    }
    if (property.major < 9)
        std::fprintf(stderr,
            "[3d] warning: the production target is sm_90 GH200; this is sm_%d%d\n",
            property.major, property.minor);

    if (!gpu_ok(cudaStreamCreate(&stream_), "cudaStreamCreate") ||
        !gpu_ok(cudaEventCreate(&bench_start_), "cudaEventCreate(start)") ||
        !gpu_ok(cudaEventCreate(&bench_stop_), "cudaEventCreate(stop)"))
        return false;
    auto allocate_device = [&](void** pointer, std::size_t bytes,
                               const char* name) {
        if (gpu_ok(cudaMalloc(pointer, bytes), name)) return true;
        return false;
    };
    for (int i = 0; i < phi_buffers_; ++i)
        if (!allocate_device(reinterpret_cast<void**>(&d_phi_[i]), one_phi,
                             "cudaMalloc(phi)")) return false;
    for (int i = 0; i < S_buffers_; ++i)
        if (!allocate_device(reinterpret_cast<void**>(&d_S_[i]),
                             one_S,
                             "cudaMalloc(S)")) return false;
    // Diagnostics use psi_w^2, while dynamics use the same binary32 profile
    // pre-scaled by kappa_w/kappa. Keeping both immutable arrays explicit
    // prevents an aggregate buffer from being mistaken for wall storage.
    if (params.resolved_wall_channel()) {
        if (!allocate_device(reinterpret_cast<void**>(&d_wall_psi_sq_),
                             static_cast<std::size_t>(params.Nz) * sizeof(float),
                             "cudaMalloc(channel diagnostic wall field)") ||
            !allocate_device(
                reinterpret_cast<void**>(&d_weighted_wall_psi_sq_),
                static_cast<std::size_t>(params.Nz) * sizeof(float),
                "cudaMalloc(channel weighted wall field)"))
            return false;
        std::vector<float> wall_psi_sq(static_cast<std::size_t>(params.Nz));
        std::vector<float> weighted_wall(static_cast<std::size_t>(params.Nz));
        const float wall_ratio = params.wall_repulsion_active()
            ? static_cast<float>(params.wall_kappa / params.kappa) : 0.0f;
        const double profile_k = interface_k(params.wall_width);
        const double lower_wall = static_cast<double>(params.channel_padding);
        const double upper_wall = lower_wall + params.channel_height;
        for (int z = 0; z < params.Nz; ++z) {
            const double coordinate = static_cast<double>(z) + 0.5;
            const double fluid = 0.25 *
                (1.0 + std::tanh(profile_k * (coordinate - lower_wall))) *
                (1.0 - std::tanh(profile_k * (coordinate - upper_wall)));
            const double wall = 1.0 - fluid;
            wall_psi_sq[static_cast<std::size_t>(z)] =
                static_cast<float>(wall * wall);
            weighted_wall[static_cast<std::size_t>(z)] =
                wall_ratio * wall_psi_sq[static_cast<std::size_t>(z)];
        }
        if (!gpu_ok(cudaMemcpyAsync(
                        d_wall_psi_sq_, wall_psi_sq.data(),
                        wall_psi_sq.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream_),
                    "upload channel wall field"))
            return false;
        // F_w=(60 kappa_w/lambda^2) sum_i int(phi_i^2 psi_w^2).
        // The pre-scaled field lets relaxation and passive translation retain
        // the existing float addition without a per-voxel multiply.
        if (!gpu_ok(cudaMemcpyAsync(
                        d_weighted_wall_psi_sq_, weighted_wall.data(),
                        wall_profile_bytes, cudaMemcpyHostToDevice, stream_),
                    "upload weighted channel wall field"))
            return false;
    }
    if (scratch_slots_ > 0 &&
        !allocate_device(reinterpret_cast<void**>(&d_scratch_),
                          static_cast<std::size_t>(scratch_slots_) *
                              one_cell_phi,
                          "cudaMalloc(scratch)")) return false;
    const std::size_t cell_state_bytes =
        static_cast<std::size_t>(params.num_cells) * sizeof(CellState3D);
    if (!allocate_device(reinterpret_cast<void**>(&d_cells_),
                          cell_state_bytes, "cudaMalloc(cells)") ||
        !allocate_device(reinterpret_cast<void**>(&d_accepted_cells_),
                          cell_state_bytes, "cudaMalloc(accepted cells)") ||
        !allocate_device(reinterpret_cast<void**>(&d_support_requests_),
                          static_cast<std::size_t>(params.num_cells) *
                              sizeof(std::uint32_t),
                          "cudaMalloc(support requests)") ||
        !allocate_device(reinterpret_cast<void**>(&d_step_), sizeof(*d_step_),
                         "cudaMalloc(step)") ||
        !allocate_device(reinterpret_cast<void**>(&d_work_cursor_),
                         sizeof(*d_work_cursor_), "cudaMalloc(cursor)") ||
        !allocate_device(reinterpret_cast<void**>(&d_flags_),
                          FLAG3D_COUNT * sizeof(std::uint32_t),
                          "cudaMalloc(flags)"))
        return false;
    for (float**& table : d_promoted_phi_)
        if (!allocate_device(reinterpret_cast<void**>(&table),
                             static_cast<std::size_t>(params.num_cells) *
                                 sizeof(float*),
                             "cudaMalloc(promoted pointer table)"))
            return false;
    if (!allocate_device(reinterpret_cast<void**>(&d_promoted_ids_),
                         static_cast<std::size_t>(params.num_cells) *
                             sizeof(int),
                         "cudaMalloc(promoted ids)"))
        return false;
    for (auto& table : h_promoted_phi_)
        table.assign(static_cast<std::size_t>(params.num_cells), nullptr);
    h_promoted_ids_.clear();
    if (moment_workspace > 0 &&
        !allocate_device(reinterpret_cast<void**>(&d_moment_partials_),
                         moment_workspace, "cudaMalloc(moment partials)"))
        return false;
    if (!allocate_device(reinterpret_cast<void**>(&d_promoted_partials_),
                         static_cast<std::size_t>(params.num_cells) *
                             kMaximumPromotedShards * sizeof(MomentPartial3D),
                         "cudaMalloc(promoted moment partials)"))
        return false;
    if (need_initial_centres &&
        !allocate_device(reinterpret_cast<void**>(&d_centres_),
                         static_cast<std::size_t>(params.num_cells) * sizeof(Vec3),
                         "cudaMalloc(centres)"))
        return false;
    if (options.strict &&
        (!allocate_device(reinterpret_cast<void**>(&d_verify_cells_),
                          static_cast<std::size_t>(params.num_cells) *
                              sizeof(VerifyCell3D), "cudaMalloc(verify cells)") ||
         !allocate_device(reinterpret_cast<void**>(&d_verify_S_),
                          sizeof(*d_verify_S_), "cudaMalloc(verify S)")))
        return false;
    if (!options.trajectory_path.empty()) {
        if (!allocate_device(reinterpret_cast<void**>(&d_trajectory_),
                             static_cast<std::size_t>(params.num_cells) *
                                 sizeof(TrajPackedCell3D),
                             "cudaMalloc(trajectory)"))
            return false;
        h_trajectory_.resize(static_cast<std::size_t>(params.num_cells));
    }
    return gpu_ok(cudaMemsetAsync(d_promoted_phi_[0], 0,
                                  static_cast<std::size_t>(params.num_cells) *
                                      sizeof(float*), stream_),
                  "clear promoted pointer table 0") &&
           gpu_ok(cudaMemsetAsync(d_promoted_phi_[1], 0,
                                  static_cast<std::size_t>(params.num_cells) *
                                      sizeof(float*), stream_),
                  "clear promoted pointer table 1") &&
           gpu_ok(cudaMemsetAsync(d_flags_, 0,
                                  FLAG3D_COUNT * sizeof(std::uint32_t), stream_),
                  "clear flags") &&
           gpu_ok(cudaMemsetAsync(
                      d_support_requests_, 0,
                      static_cast<std::size_t>(params.num_cells) *
                          sizeof(std::uint32_t), stream_),
                  "clear support requests") &&
           gpu_ok(cudaMemsetAsync(d_step_, 0, sizeof(*d_step_), stream_),
                  "clear step");
}

bool Sim3D::init_fresh(const SimParams3D& params,
                       const RunOptions3D& options) {
    if (params.hard_wall_channel() &&
        (!params.resolved_wall_channel() || !params.wall_repulsion_active())) {
        std::fprintf(stderr,
            "[3d] fresh channel runs require positive resolved steric-wall "
            "parameters\n");
        return false;
    }
    const char* validation_error = nullptr;
    if (!validate(params, &validation_error)) {
        std::fprintf(stderr, "[3d] invalid parameters: %s\n",
                     validation_error ? validation_error : "unknown reason");
        return false;
    }
    if (!allocate(params, options, options.brick_edge, true, std::nullopt))
        return false;
    steps_done_ = 0;
    if (!initialize_fields()) return false;
    if (!initialize_common()) return false;
    return true;
}

bool Sim3D::init_checkpoint(const CheckpointMeta3D& checkpoint,
                            const std::string& path,
                            const RunOptions3D& options,
                            double t_end_override) {
    PromotedMeasureReduction3D reduction{};
    const ReductionResumeResult reduction_result =
        resolve_promoted_measure_resume(
            checkpoint.promoted_measure_reduction,
            options.promoted_measure_shards_supplied,
            options.promoted_measure_shards, &reduction);
    if (reduction_result != ReductionResumeResult::Ok) {
        std::fprintf(stderr,
            reduction_result == ReductionResumeResult::Mismatch
                ? "[3d] checkpoint reduction policy does not match the "
                  "requested promoted-measurement policy\n"
                : "[3d] checkpoint has an invalid promoted-measurement "
                  "reduction contract\n");
        return false;
    }
    RunOptions3D continued_options = options;
    continued_options.promoted_measure_shards = reduction.policy;
    continued_options.promoted_measure_auto_wave_ctas =
        reduction.auto_wave_ctas;
    if (!valid_checkpoint_base_measure_shards(
            checkpoint.base_measure_shards)) {
        std::fprintf(stderr,
            "[3d] checkpoint metadata stores an invalid base measurement "
            "shard count %d\n", checkpoint.base_measure_shards);
        return false;
    }
    if (options.brick_edge > 0 && options.brick_edge != checkpoint.brick_edge) {
        std::fprintf(stderr,
            "[3d] resume requires checkpoint B=%d; --brick-edge requested %d\n",
            checkpoint.brick_edge, options.brick_edge);
        return false;
    }
    SimParams3D continued = checkpoint.params;
    if (t_end_override >= 0.0) continued.t_end = t_end_override;
    const char* validation_error = nullptr;
    if (!validate(continued, &validation_error) ||
        continued.total_steps() < static_cast<long long>(checkpoint.step)) {
        std::fprintf(stderr,
            "[3d] continuation target is invalid or precedes checkpoint step: %s\n",
            validation_error ? validation_error : "t_end is too small");
        return false;
    }
    if (!allocate(continued, continued_options, checkpoint.brick_edge, false,
                  checkpoint.base_measure_shards))
        return false;
    steps_done_ = checkpoint.step;
    if (!install_checkpoint_promotions(checkpoint)) return false;
    CheckpointLoadView3D load{};
    load.d_cells = d_cells_;
    load.d_phi = d_phi_[0];
    load.h_promoted_phi = h_promoted_phi_[0].data();
    load.stream = stream_;
    if (!checkpoint_load_3d(path, checkpoint, load))
        return false;
    if (!h_promoted_ids_.empty()) {
        const std::size_t promoted_bytes = promoted_words_ * sizeof(float);
        for (int id : h_promoted_ids_)
            if (!gpu_ok(cudaMemcpyAsync(
                            h_promoted_phi_[1][static_cast<std::size_t>(id)],
                            h_promoted_phi_[0][static_cast<std::size_t>(id)],
                            promoted_bytes, cudaMemcpyDeviceToDevice, stream_),
                        "initialize alternate promoted checkpoint buffer"))
                return false;
    }
    current_phi_index_ = 0;
    current_S_index_ = 0;
    current_promoted_phi_index_ = 0;
    if (!gpu_ok(cudaMemcpyAsync(d_step_, &steps_done_, sizeof(steps_done_),
                                cudaMemcpyHostToDevice, stream_),
                "restore step") || !reconstruct_current_S() ||
        !refresh_measurements(false, true) ||
        !synchronize_and_check("checkpoint restore"))
        return false;
    return initialize_common();
}

bool Sim3D::initialize_fields() {
    const int N = params_.num_cells;
    std::vector<std::int64_t> global_id;
    std::vector<float> x, y, z;
    std::uint64_t initialization_hash = 0;
    if (params_.substrate_slab()) {
        // Cell-centred z planes begin at zero, so the neutral substrate face is
        // at z=-1/2. A sphere centred on that face seeds a 90-degree sessile
        // hemisphere; x/y use Palmieri's sequential 2D placement convention.
        if (options_.initial_centres_path.empty()) {
            pf::PalmieriInitDiagnostics generated{};
            try {
                pf::palmieri_sequential_centres(
                    N, static_cast<double>(params_.Nx), params_.target_radius,
                    params_.seed, &x, &y, &generated);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[3d] slab initializer failed: %s\n",
                             error.what());
                return false;
            }
            pf::PalmieriCentresCsvDiagnostics checked{};
            std::string error;
            if (!pf::palmieri_validate_centres(
                    x, y, N, static_cast<double>(params_.Nx),
                    params_.target_radius, &checked, &error)) {
                std::fprintf(stderr, "[3d] generated invalid slab centres: %s\n",
                             error.c_str());
                return false;
            }
            z.assign(x.size(), -0.5f);
            global_id = ordered_global_ids(x.size());
            initialization_hash = centre_table_fnv1a64(global_id, x, y, z);
            std::printf(
                "  slab centres %s, minimum xy-periodic distance %.4f, "
                "candidates %llu (%llu rejected), table hash %016llx\n",
                pf::kPalmieriInitializerMethod,
                checked.minimum_periodic_distance,
                static_cast<unsigned long long>(generated.candidates_drawn),
                static_cast<unsigned long long>(generated.candidates_rejected),
                static_cast<unsigned long long>(initialization_hash));
        } else {
            pf::RelaxedCentresCsvDiagnostics diagnostics{};
            std::string error;
            if (!pf::relaxed_read_centres_csv(
                    options_.initial_centres_path, N,
                    static_cast<double>(params_.Nx), &global_id,
                    &x, &y, &diagnostics, &error)) {
                std::fprintf(stderr, "[3d] %s\n", error.c_str());
                return false;
            }
            z.assign(x.size(), -0.5f);
            initialization_hash = centre_table_fnv1a64(global_id, x, y, z);
            std::printf(
                "  slab realized centres %s, minimum xy-periodic centroid "
                "distance %.4f, table hash %016llx\n",
                options_.initial_centres_path.c_str(),
                diagnostics.minimum_periodic_distance,
                static_cast<unsigned long long>(initialization_hash));
        }
    } else if (params_.hard_wall_channel()) {
        if (options_.initial_centres_path.empty()) {
            PalmieriInit3DDiagnostics generated{};
            try {
                channel3d_sequential_centres(
                    N, static_cast<double>(params_.Nx),
                    params_.accessible_channel_height(),
                    params_.target_radius, params_.seed, &x, &y, &z,
                    &generated);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[3d] channel initializer failed: %s\n",
                             error.what());
                return false;
            }
            PalmieriCentres3DCsvDiagnostics checked{};
            std::string error;
            if (!channel3d_validate_centres(
                    x, y, z, N, static_cast<double>(params_.Nx),
                    params_.accessible_channel_height(),
                    params_.target_radius, &checked, &error)) {
                std::fprintf(stderr, "[3d] generated invalid channel centres: %s\n",
                             error.c_str());
                return false;
            }
            global_id = ordered_global_ids(x.size());
            initialization_hash = checked.table_fnv1a64;
            std::printf(
                "  channel centres %s, minimum xy-periodic distance %.4f, "
                "candidates %llu (%llu rejected), table hash %016llx\n",
                kChannelInitializer3DMethod,
                checked.minimum_periodic_distance,
                static_cast<unsigned long long>(generated.candidates_drawn),
                static_cast<unsigned long long>(generated.candidates_rejected),
                static_cast<unsigned long long>(initialization_hash));
        } else {
            std::ifstream probe(options_.initial_centres_path,
                                std::ios::binary);
            std::string header;
            if (!probe || !std::getline(probe, header)) {
                std::fprintf(stderr, "[3d] cannot read channel centre CSV\n");
                return false;
            }
            pf::relaxed_normalize_csv_line(&header, true);
            if (header == "global_id,x,y") {
                const int one_layer_height = static_cast<int>(
                    std::ceil(2.0 * params_.target_radius));
                if (params_.accessible_channel_height() != one_layer_height) {
                    std::fprintf(stderr,
                        "[3d] a 2D relaxed x/y table is accepted only for the "
                        "one-layer channel H=ceil(2R)=%d; use an id,x,y,z "
                        "table for multilayer initialization\n",
                        one_layer_height);
                    return false;
                }
                pf::RelaxedCentresCsvDiagnostics diagnostics{};
                std::string error;
                if (!pf::relaxed_read_centres_csv(
                        options_.initial_centres_path, N,
                        static_cast<double>(params_.Nx), &global_id,
                        &x, &y, &diagnostics, &error, true)) {
                    std::fprintf(stderr, "[3d] %s\n", error.c_str());
                    return false;
                }
                z.assign(x.size(),
                         static_cast<float>(0.5 * one_layer_height));
                initialization_hash = centre_table_fnv1a64(
                    global_id, x, y, z);
                std::printf(
                    "  channel one-layer centres %s; transferred id/x/y only, "
                    "fresh spheres at z=H/2, minimum xy-periodic centroid "
                    "distance %.4f, table hash %016llx\n",
                    options_.initial_centres_path.c_str(),
                    diagnostics.minimum_periodic_distance,
                    static_cast<unsigned long long>(initialization_hash));
            } else {
                PalmieriCentres3DCsvDiagnostics diagnostics{};
                std::string error;
                if (!channel3d_read_centres_csv(
                        options_.initial_centres_path, N,
                        static_cast<double>(params_.Nx),
                        params_.accessible_channel_height(),
                        params_.target_radius, &x, &y, &z,
                        &diagnostics, &error)) {
                    std::fprintf(stderr, "[3d] %s\n", error.c_str());
                    return false;
                }
                global_id = ordered_global_ids(x.size());
                initialization_hash = diagnostics.table_fnv1a64;
                std::printf(
                    "  channel centres %s, minimum xy-periodic distance %.4f, "
                    "table hash %016llx\n",
                    options_.initial_centres_path.c_str(),
                    diagnostics.minimum_periodic_distance,
                    static_cast<unsigned long long>(initialization_hash));
            }
        }
    } else if (options_.initial_centres_path.empty()) {
        PalmieriInit3DDiagnostics generated{};
        try {
            palmieri3d_sequential_centres(
                N, static_cast<double>(params_.Nx), params_.target_radius,
                params_.seed, &x, &y, &z, &generated);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[3d] initializer failed: %s\n", error.what());
            return false;
        }
        PalmieriCentres3DCsvDiagnostics checked{};
        std::string error;
        if (!palmieri3d_validate_centres(
                x, y, z, N, static_cast<double>(params_.Nx),
                params_.target_radius, &checked, &error)) {
            std::fprintf(stderr, "[3d] generated invalid centres: %s\n",
                         error.c_str());
            return false;
        }
        global_id = ordered_global_ids(x.size());
        std::printf("  centres %s, minimum periodic distance %.4f, "
                    "candidates %llu (%llu rejected), table hash %016llx\n",
                    kPalmieriInitializer3DMethod,
                    checked.minimum_periodic_distance,
                    static_cast<unsigned long long>(generated.candidates_drawn),
                    static_cast<unsigned long long>(generated.candidates_rejected),
                    static_cast<unsigned long long>(checked.table_fnv1a64));
        initialization_hash = checked.table_fnv1a64;
    } else {
        PalmieriCentres3DCsvDiagnostics diagnostics{};
        std::string error;
        if (!palmieri3d_read_centres_csv(
                options_.initial_centres_path, N,
                static_cast<double>(params_.Nx), params_.target_radius,
                &x, &y, &z, &diagnostics, &error)) {
            std::fprintf(stderr, "[3d] %s\n", error.c_str());
            return false;
        }
        global_id = ordered_global_ids(x.size());
        std::printf("  centres %s, minimum periodic distance %.4f, table hash "
                    "%016llx\n", options_.initial_centres_path.c_str(),
                    diagnostics.minimum_periodic_distance,
                     static_cast<unsigned long long>(diagnostics.table_fnv1a64));
        initialization_hash = diagnostics.table_fnv1a64;
    }
    params_.initialization_hash = initialization_hash;

    std::vector<CellState3D> cells(static_cast<std::size_t>(N));
    std::vector<Vec3> centres(static_cast<std::size_t>(N));
    const int soft_count = static_cast<int>(
        std::llround(params_.cancer_fraction * static_cast<double>(N)));
    const std::vector<std::uint8_t> soft = soft_id_membership(
        N, params_.cancer_fraction, params_.seed);
    for (int i = 0; i < N; ++i) {
        CellState3D state{};
        state.global_id = global_id[static_cast<std::size_t>(i)];
        state.gamma = static_cast<float>(soft[static_cast<std::size_t>(i)] != 0
            ? params_.gamma_cancer : params_.gamma_normal);
        state.v_A = static_cast<float>(initial_active_speed(
            state.global_id, params_.seed, params_.v_A, params_.v_A_sigma));
        state.R_tgt = static_cast<float>(params_.target_radius);
        const Vec3 polarity = params_.substrate_slab()
            ? initial_planar_polarity(state.global_id,
                                      params_.polarity_stream())
            : initial_polarity(state.global_id, params_.polarity_stream());
        state.polarity_x = static_cast<float>(polarity.x);
        state.polarity_y = static_cast<float>(polarity.y);
        state.polarity_z = static_cast<float>(polarity.z);
        cells[static_cast<std::size_t>(i)] = state;
        const float stored_z = params_.resolved_wall_channel()
            ? z[static_cast<std::size_t>(i)] +
                static_cast<float>(params_.channel_padding) - 0.5f
            : z[static_cast<std::size_t>(i)];
        centres[static_cast<std::size_t>(i)] = Vec3{
            x[static_cast<std::size_t>(i)], y[static_cast<std::size_t>(i)],
            stored_z};
    }
    std::printf("  soft identities %d/%d, sampled without replacement\n",
                soft_count, N);
    if (!gpu_ok(cudaMemcpyAsync(d_cells_, cells.data(),
                                cells.size() * sizeof(CellState3D),
                                cudaMemcpyHostToDevice, stream_),
                "copy initial cell state") ||
        !gpu_ok(cudaMemcpyAsync(d_centres_, centres.data(),
                                centres.size() * sizeof(Vec3),
                                cudaMemcpyHostToDevice, stream_),
                "copy initial centres"))
        return false;

    reference::SeedRadiusCalibration calibration{};
    try {
        calibration = reference::calibrate_seed_radius(
            params_.target_radius, params_.lambda, B_);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[3d] sphere-radius calibration failed: %s\n",
                     error.what());
        return false;
    }
    const double seed_radius = calibration.seed_radius;
    if (!calibration.converged || !(seed_radius > 0.0) ||
        !std::isfinite(seed_radius)) {
        std::fprintf(stderr,
                     "[3d] discrete sphere-radius calibration did not converge\n");
        return false;
    }
    std::printf("  discrete sphere seed radius %.9g (target R %.9g, V0 %.9g)\n",
                seed_radius, params_.target_radius, params_.volume0());
    InitArgs3D init{};
    init.phi_even = d_phi_[0];
    init.phi_odd = phi_buffers_ > 1 ? d_phi_[1] : nullptr;
    init.cells = d_cells_;
    init.centres = d_centres_;
    init.N = N;
    init.B = B_;
    init.layout = layout_;
    init.lambda = static_cast<float>(params_.lambda);
    init.seed_radius = static_cast<float>(seed_radius);
    launch_initialize_spheres(init, stream_);
    if (!gpu_ok(cudaGetLastError(), "initialize-spheres launch")) return false;
    current_phi_index_ = 0;
    current_S_index_ = 0;
    if (!reconstruct_current_S() || !refresh_measurements(false, true) ||
        !synchronize_and_check("fresh initialization"))
        return false;
    cudaFree(d_centres_);
    d_centres_ = nullptr;
    return true;
}

bool Sim3D::initialize_common() {
    const std::uint64_t target = static_cast<std::uint64_t>(
        params_.total_steps());
    trajectory_every_ = options_.trajectory_interval > 0
        ? static_cast<std::uint64_t>(options_.trajectory_interval)
        : std::max<std::uint64_t>(
            1, target / static_cast<std::uint64_t>(
                std::max(1, options_.trajectory_samples)));
    if (!options_.checkpoint_dir.empty() &&
        trajectory_every_ > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr,
            "[3d] resolved trajectory interval exceeds the checkpoint limit\n");
        return false;
    }
    if (!options_.trajectory_path.empty() && !open_trajectory()) return false;
    // Seed a newly created trajectory at the initialized state. An existing
    // compatible trajectory keeps its absolute sampling grid across resumes.
    if (trajectory_file_ && trajectory_frames_ == 0 && !append_trajectory())
        return false;
    return true;
}

const float* Sim3D::current_phi() const {
    return d_phi_[current_phi_index_];
}

float* Sim3D::current_phi() {
    return d_phi_[current_phi_index_];
}

std::uint32_t* Sim3D::current_S() {
    return d_S_[current_S_index_];
}

WeightedWallField3D Sim3D::weighted_wall_field() const {
    if (!layout_.hard_wall_channel()) return {};
    return {d_weighted_wall_psi_sq_, layout_.nz};
}

bool Sim3D::upload_promoted_tables() {
    const std::size_t pointer_bytes =
        static_cast<std::size_t>(params_.num_cells) * sizeof(float*);
    for (int buffer = 0; buffer < 2; ++buffer) {
        if (!gpu_ok(cudaMemcpyAsync(d_promoted_phi_[buffer],
                                    h_promoted_phi_[buffer].data(),
                                    pointer_bytes, cudaMemcpyHostToDevice,
                                    stream_),
                    "upload promoted pointer table"))
            return false;
    }
    if (!h_promoted_ids_.empty() &&
        !gpu_ok(cudaMemcpyAsync(d_promoted_ids_, h_promoted_ids_.data(),
                                h_promoted_ids_.size() * sizeof(int),
                                cudaMemcpyHostToDevice, stream_),
                "upload promoted ids"))
        return false;
    return true;
}

bool Sim3D::allocate_promoted_fields(
    const std::vector<int>& cell_ids, std::vector<CellState3D>* states) {
    if (cell_ids.empty()) return true;
    if (!states || states->size() != static_cast<std::size_t>(params_.num_cells) ||
        promoted_edge_ <= B_ || promoted_words_ == 0) {
        std::fprintf(stderr,
            "[3d] support exhausted but no larger aligned brick fits inside "
            "the domain (base B=%d, minimum extent %d)\n", B_,
            params_.minimum_domain_edge());
        return false;
    }
    std::size_t one_field_bytes = 0, bytes_per_cell = 0, requested_bytes = 0;
    if (!multiply_bytes(promoted_words_, sizeof(float), &one_field_bytes) ||
        !multiply_bytes(one_field_bytes, 2, &bytes_per_cell) ||
        !multiply_bytes(bytes_per_cell, cell_ids.size(), &requested_bytes) ||
        requested_bytes > adaptive_budget_remaining_) {
        std::fprintf(stderr,
            "[3d] adaptive support needs %.3f GiB but only %.3f GiB remains "
            "inside the configured HBM budget; the accepted state is retained\n",
            static_cast<double>(requested_bytes) / 1073741824.0,
            static_cast<double>(adaptive_budget_remaining_) / 1073741824.0);
        return false;
    }

    struct PendingAllocation { int id; float* field[2]; };
    std::vector<PendingAllocation> pending;
    pending.reserve(cell_ids.size());
    auto release_pending = [&]() {
        for (PendingAllocation& allocation : pending)
            for (float*& field : allocation.field)
                if (field) { cudaFree(field); field = nullptr; }
    };

    const std::size_t base_words = brick_words_;
    const int offset = (promoted_edge_ - B_) / 2;
    for (int id : cell_ids) {
        if (id < 0 || id >= params_.num_cells ||
            h_promoted_phi_[0][static_cast<std::size_t>(id)] != nullptr ||
            cell_is_promoted((*states)[static_cast<std::size_t>(id)], B_)) {
            std::fprintf(stderr, "[3d] invalid duplicate promotion request\n");
            release_pending();
            return false;
        }
        PendingAllocation allocation{id, {nullptr, nullptr}};
        if (!gpu_ok(cudaMalloc(reinterpret_cast<void**>(&allocation.field[0]),
                               one_field_bytes),
                    "cudaMalloc(adaptive phi 0)")) {
            release_pending();
            return false;
        }
        if (!gpu_ok(cudaMalloc(reinterpret_cast<void**>(&allocation.field[1]),
                               one_field_bytes),
                    "cudaMalloc(adaptive phi 1)")) {
            cudaFree(allocation.field[0]);
            release_pending();
            return false;
        }
        pending.push_back(allocation);
    }

    for (PendingAllocation& allocation : pending) {
        for (float* field : allocation.field)
            if (!gpu_ok(cudaMemsetAsync(field, 0, one_field_bytes, stream_),
                        "clear adaptive phi")) {
                cudaStreamSynchronize(stream_);
                release_pending();
                return false;
            }
        const float* source = current_phi()
            + static_cast<std::size_t>(allocation.id) * base_words;
        for (float* field : allocation.field) {
            if (!copy_centered_cube(field, promoted_edge_, source, B_,
                                    stream_)) {
                cudaStreamSynchronize(stream_);
                release_pending();
                return false;
            }
        }
    }
    if (!gpu_ok(cudaStreamSynchronize(stream_),
                "adaptive support migration")) {
        release_pending();
        return false;
    }

    const std::uint32_t recovered =
        flag3d_bit(FLAG3D_SUPPORT_EXHAUSTED) |
        flag3d_bit(FLAG3D_SUPPORT_EDGE);
    for (PendingAllocation& allocation : pending) {
        const std::size_t id = static_cast<std::size_t>(allocation.id);
        CellState3D& state = (*states)[id];
        state.origin_x -= offset;
        state.origin_y -= offset;
        state.origin_z -= offset;
        state.Cx += static_cast<double>(offset) * state.V;
        state.Cy += static_cast<double>(offset) * state.V;
        state.Cz += static_cast<double>(offset) * state.V;
        if (state.bb_hi_x >= state.bb_lo_x) {
            state.bb_lo_x += offset; state.bb_hi_x += offset;
            state.bb_lo_y += offset; state.bb_hi_y += offset;
            state.bb_lo_z += offset; state.bb_hi_z += offset;
        }
        state.pending_shift_x = state.pending_shift_y =
            state.pending_shift_z = 0;
        state.storage_edge = static_cast<std::uint32_t>(promoted_edge_);
        state.flags &= ~recovered;
        h_promoted_phi_[0][id] = allocation.field[0];
        h_promoted_phi_[1][id] = allocation.field[1];
        h_promoted_ids_.push_back(allocation.id);
        allocation.field[0] = allocation.field[1] = nullptr;
    }
    if (!upload_promoted_tables() ||
        !gpu_ok(cudaMemcpyAsync(d_cells_, states->data(),
                                states->size() * sizeof(CellState3D),
                                cudaMemcpyHostToDevice, stream_),
                "install promoted cell states") ||
        !gpu_ok(cudaMemsetAsync(d_flags_ + FLAG3D_SUPPORT_EXHAUSTED, 0,
                                sizeof(std::uint32_t), stream_),
                "clear recovered support flag") ||
        !gpu_ok(cudaMemsetAsync(d_flags_ + FLAG3D_SUPPORT_EDGE, 0,
                                sizeof(std::uint32_t), stream_),
                "clear recovered support-edge advisory") ||
        !gpu_ok(cudaMemsetAsync(
                    d_support_requests_, 0,
                    static_cast<std::size_t>(params_.num_cells) *
                        sizeof(std::uint32_t), stream_),
                "clear recovered support requests") ||
        !gpu_ok(cudaStreamSynchronize(stream_),
                "install adaptive support"))
        return false;
    adaptive_budget_remaining_ -= requested_bytes;
    required_device_bytes_ += requested_bytes;
    std::printf("[3d] promoted %zu cell(s) from B=%d to B=%d "
                "(persistent adaptive storage %.3f GiB)\n",
                cell_ids.size(), B_, promoted_edge_,
                static_cast<double>(requested_bytes) / 1073741824.0);
    return true;
}

bool Sim3D::grow_promoted_fields(
    int new_edge, const std::vector<int>& additional_cell_ids,
    std::vector<CellState3D>* states) {
    const int old_edge = promoted_edge_;
    if (!states ||
        states->size() != static_cast<std::size_t>(params_.num_cells) ||
        h_promoted_ids_.empty() || new_edge <= promoted_edge_ ||
        new_edge > maximum_support_edge_ ||
        !valid_runtime_geometry(new_edge, layout_)) {
        std::fprintf(stderr,
            "[3d] no larger aligned adaptive brick fits inside minimum "
            "domain extent %d\n", params_.minimum_domain_edge());
        return false;
    }

    std::vector<int> all_ids = h_promoted_ids_;
    std::vector<unsigned char> seen(
        static_cast<std::size_t>(params_.num_cells), 0u);
    for (int id : all_ids) {
        if (id < 0 || id >= params_.num_cells ||
            seen[static_cast<std::size_t>(id)] != 0u) {
            std::fprintf(stderr, "[3d] corrupt adaptive cell table\n");
            return false;
        }
        seen[static_cast<std::size_t>(id)] = 1u;
    }
    for (int id : additional_cell_ids) {
        if (id < 0 || id >= params_.num_cells ||
            seen[static_cast<std::size_t>(id)] != 0u ||
            cell_is_promoted((*states)[static_cast<std::size_t>(id)], B_)) {
            std::fprintf(stderr, "[3d] invalid adaptive growth request\n");
            return false;
        }
        seen[static_cast<std::size_t>(id)] = 1u;
        all_ids.push_back(id);
    }

    std::size_t new_words = 0;
    std::size_t new_field_bytes = 0, new_total_bytes = 0;
    std::size_t old_field_bytes = 0, old_total_bytes = 0;
    if (!checked_cube_size(static_cast<std::size_t>(new_edge), &new_words) ||
        !multiply_bytes(new_words, sizeof(float), &new_field_bytes) ||
        !multiply_bytes(new_field_bytes, 2 * all_ids.size(),
                        &new_total_bytes) ||
        !multiply_bytes(promoted_words_, sizeof(float), &old_field_bytes) ||
        !multiply_bytes(old_field_bytes, 2 * h_promoted_ids_.size(),
                        &old_total_bytes) ||
        new_total_bytes > adaptive_budget_remaining_) {
        std::fprintf(stderr,
            "[3d] growing adaptive support to B=%d needs %.3f GiB of "
            "transactional HBM but only %.3f GiB is free; the accepted "
            "state is retained\n",
            new_edge, static_cast<double>(new_total_bytes) / 1073741824.0,
            static_cast<double>(adaptive_budget_remaining_) / 1073741824.0);
        return false;
    }

    struct NewFields { int id; float* field[2]; };
    std::vector<NewFields> pending;
    pending.reserve(all_ids.size());
    auto release_pending = [&]() {
        for (NewFields& allocation : pending)
            for (float*& field : allocation.field)
                if (field) { cudaFree(field); field = nullptr; }
    };
    for (int id : all_ids) {
        NewFields allocation{id, {nullptr, nullptr}};
        if (!gpu_ok(cudaMalloc(reinterpret_cast<void**>(&allocation.field[0]),
                               new_field_bytes),
                    "cudaMalloc(grown adaptive phi 0)")) {
            release_pending();
            return false;
        }
        if (!gpu_ok(cudaMalloc(reinterpret_cast<void**>(&allocation.field[1]),
                               new_field_bytes),
                    "cudaMalloc(grown adaptive phi 1)")) {
            cudaFree(allocation.field[0]);
            release_pending();
            return false;
        }
        pending.push_back(allocation);
    }

    std::vector<CellState3D> next_states = *states;
    std::vector<float*> next_tables[2] = {
        h_promoted_phi_[0], h_promoted_phi_[1]};
    const std::size_t base_words = brick_words_;
    for (NewFields& allocation : pending) {
        const std::size_t id = static_cast<std::size_t>(allocation.id);
        const bool was_promoted = h_promoted_phi_[0][id] != nullptr;
        const int source_edge = was_promoted ? promoted_edge_ : B_;
        const float* source = was_promoted
            ? h_promoted_phi_[current_promoted_phi_index_][id]
            : current_phi() + id * base_words;
        for (float* field : allocation.field) {
            if (!gpu_ok(cudaMemsetAsync(field, 0, new_field_bytes, stream_),
                        "clear grown adaptive phi") ||
                !copy_centered_cube(field, new_edge, source, source_edge,
                                    stream_)) {
                cudaStreamSynchronize(stream_);
                release_pending();
                return false;
            }
        }
        const int offset = (new_edge - source_edge) / 2;
        CellState3D& state = next_states[id];
        state.origin_x -= offset;
        state.origin_y -= offset;
        state.origin_z -= offset;
        state.Cx += static_cast<double>(offset) * state.V;
        state.Cy += static_cast<double>(offset) * state.V;
        state.Cz += static_cast<double>(offset) * state.V;
        if (state.bb_hi_x >= state.bb_lo_x) {
            state.bb_lo_x += offset; state.bb_hi_x += offset;
            state.bb_lo_y += offset; state.bb_hi_y += offset;
            state.bb_lo_z += offset; state.bb_hi_z += offset;
        }
        state.pending_shift_x = state.pending_shift_y =
            state.pending_shift_z = 0;
        state.storage_edge = static_cast<std::uint32_t>(new_edge);
        state.flags &= ~(flag3d_bit(FLAG3D_SUPPORT_EXHAUSTED) |
                         flag3d_bit(FLAG3D_SUPPORT_EDGE));
        next_tables[0][id] = allocation.field[0];
        next_tables[1][id] = allocation.field[1];
    }
    if (!gpu_ok(cudaStreamSynchronize(stream_),
                "grow adaptive support fields")) {
        release_pending();
        return false;
    }

    const std::size_t pointer_bytes =
        static_cast<std::size_t>(params_.num_cells) * sizeof(float*);
    const std::size_t state_bytes =
        next_states.size() * sizeof(CellState3D);
    bool installed = true;
    for (int buffer = 0; buffer < 2; ++buffer)
        installed = installed && gpu_ok(cudaMemcpyAsync(
            d_promoted_phi_[buffer], next_tables[buffer].data(), pointer_bytes,
            cudaMemcpyHostToDevice, stream_),
            "install grown adaptive pointer table");
    installed = installed && gpu_ok(cudaMemcpyAsync(
        d_promoted_ids_, all_ids.data(), all_ids.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream_), "install grown adaptive ids");
    installed = installed && gpu_ok(cudaMemcpyAsync(
        d_cells_, next_states.data(), state_bytes, cudaMemcpyHostToDevice,
        stream_), "install grown adaptive cell states");
    installed = installed && gpu_ok(cudaMemsetAsync(
        d_flags_ + FLAG3D_SUPPORT_EXHAUSTED, 0, sizeof(std::uint32_t), stream_),
        "clear recovered support flag");
    installed = installed && gpu_ok(cudaMemsetAsync(
        d_flags_ + FLAG3D_SUPPORT_EDGE, 0, sizeof(std::uint32_t), stream_),
        "clear recovered support-edge advisory");
    installed = installed && gpu_ok(cudaMemsetAsync(
        d_support_requests_, 0,
        static_cast<std::size_t>(params_.num_cells) * sizeof(std::uint32_t),
        stream_), "clear recovered support requests");
    installed = installed && gpu_ok(cudaStreamSynchronize(stream_),
                                    "install grown adaptive support");
    if (!installed) {
        // Restore the live pointer tables before releasing tentative storage.
        cudaStreamSynchronize(stream_);
        for (int buffer = 0; buffer < 2; ++buffer)
            cudaMemcpy(d_promoted_phi_[buffer], h_promoted_phi_[buffer].data(),
                       pointer_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_promoted_ids_, h_promoted_ids_.data(),
                   h_promoted_ids_.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cells_, states->data(), state_bytes,
                   cudaMemcpyHostToDevice);
        release_pending();
        return false;
    }

    for (int id : h_promoted_ids_)
        for (int buffer = 0; buffer < 2; ++buffer)
            cudaFree(h_promoted_phi_[buffer][static_cast<std::size_t>(id)]);
    h_promoted_phi_[0].swap(next_tables[0]);
    h_promoted_phi_[1].swap(next_tables[1]);
    h_promoted_ids_.swap(all_ids);
    for (NewFields& allocation : pending)
        allocation.field[0] = allocation.field[1] = nullptr;
    *states = std::move(next_states);
    promoted_edge_ = new_edge;
    promoted_words_ = new_words;
    adaptive_budget_remaining_ =
        adaptive_budget_remaining_ - new_total_bytes + old_total_bytes;
    required_device_bytes_ =
        required_device_bytes_ + new_total_bytes - old_total_bytes;
    std::printf("[3d] grew adaptive support for %zu cell(s) from B=%d to "
                "B=%d (persistent storage %.3f GiB)\n",
                h_promoted_ids_.size(), old_edge, new_edge,
                static_cast<double>(new_total_bytes) / 1073741824.0);
    return true;
}

bool Sim3D::install_checkpoint_promotions(
    const CheckpointMeta3D& checkpoint) {
    const int N = params_.num_cells;
    if (checkpoint.storage_edges.size() != static_cast<std::size_t>(N)) {
        std::fprintf(stderr,
            "[3d] checkpoint support-edge table has the wrong length\n");
        return false;
    }
    std::vector<int> ids;
    int stored_promoted_edge = 0;
    for (int n = 0; n < N; ++n) {
        const int edge = checkpoint.storage_edges[static_cast<std::size_t>(n)];
        if (edge == B_) continue;
        if (edge <= B_ || edge >= params_.minimum_domain_edge() ||
            edge % kBrickAlignment != 0 ||
            (stored_promoted_edge != 0 && stored_promoted_edge != edge)) {
            std::fprintf(stderr,
                "[3d] checkpoint contains an unsupported adaptive edge %d\n",
                edge);
            return false;
        }
        stored_promoted_edge = edge;
        ids.push_back(n);
    }
    if (ids.empty()) return upload_promoted_tables();

    promoted_edge_ = stored_promoted_edge;
    if (!checked_cube_size(static_cast<std::size_t>(promoted_edge_),
                           &promoted_words_))
        return false;
    std::size_t one_field_bytes = 0, requested_bytes = 0;
    if (!multiply_bytes(promoted_words_, sizeof(float), &one_field_bytes) ||
        !multiply_bytes(one_field_bytes, 2 * ids.size(), &requested_bytes) ||
        requested_bytes > adaptive_budget_remaining_) {
        std::fprintf(stderr,
            "[3d] checkpoint adaptive storage needs %.3f GiB but the "
            "configured HBM budget has %.3f GiB available\n",
            static_cast<double>(requested_bytes) / 1073741824.0,
            static_cast<double>(adaptive_budget_remaining_) / 1073741824.0);
        return false;
    }

    std::vector<std::pair<float*, float*>> allocations;
    allocations.reserve(ids.size());
    auto release = [&]() {
        for (auto allocation : allocations) {
            if (allocation.first) cudaFree(allocation.first);
            if (allocation.second) cudaFree(allocation.second);
        }
    };
    for (std::size_t slot = 0; slot < ids.size(); ++slot) {
        float* first = nullptr;
        float* second = nullptr;
        if (!gpu_ok(cudaMalloc(reinterpret_cast<void**>(&first),
                               one_field_bytes),
                    "cudaMalloc(checkpoint adaptive phi 0)")) {
            release();
            return false;
        }
        if (!gpu_ok(cudaMalloc(reinterpret_cast<void**>(&second),
                               one_field_bytes),
                    "cudaMalloc(checkpoint adaptive phi 1)")) {
            cudaFree(first);
            release();
            return false;
        }
        allocations.emplace_back(first, second);
    }
    // Publish only after the complete allocation succeeds. A mid-allocation
    // OOM therefore leaves no dangling host entries for the destructor.
    for (std::size_t slot = 0; slot < ids.size(); ++slot) {
        const std::size_t id = static_cast<std::size_t>(ids[slot]);
        h_promoted_phi_[0][id] = allocations[slot].first;
        h_promoted_phi_[1][id] = allocations[slot].second;
        h_promoted_ids_.push_back(ids[slot]);
    }
    if (!upload_promoted_tables()) {
        for (int id : ids) {
            h_promoted_phi_[0][static_cast<std::size_t>(id)] = nullptr;
            h_promoted_phi_[1][static_cast<std::size_t>(id)] = nullptr;
        }
        h_promoted_ids_.clear();
        release();
        return false;
    }
    adaptive_budget_remaining_ -= requested_bytes;
    required_device_bytes_ += requested_bytes;
    return true;
}

bool Sim3D::reconstruct_current_S() {
    launch_clear_S(current_S(), layout_, stream_);
    ScatterArgs3D scatter{};
    scatter.phi = current_phi();
    scatter.cells = d_cells_;
    scatter.S = current_S();
    scatter.global_flags = d_flags_;
    scatter.layout = layout_;
    scatter.N = params_.num_cells;
    scatter.B = B_;
    launch_scatter_current(scatter, stream_);
    if (!h_promoted_ids_.empty())
        launch_scatter_promoted(
            scatter, d_promoted_phi_[current_promoted_phi_index_],
            d_promoted_ids_, static_cast<int>(h_promoted_ids_.size()),
            promoted_edge_, stream_);
    return gpu_ok(cudaGetLastError(), "reconstruct S launch");
}

bool Sim3D::refresh_measurements(bool apply_tumble,
                                 bool compute_surface,
                                 bool apply_motion) {
    MeasureArgs3D measure{};
    measure.phi = current_phi();
    measure.S = current_S();
    measure.wall = weighted_wall_field();
    measure.cells = d_cells_;
    measure.global_flags = d_flags_;
    measure.step = d_step_;
    measure.layout = layout_;
    measure.N = params_.num_cells;
    measure.B = B_;
    measure.partials = d_moment_partials_;
    measure.shards = measurement_shards_;
    measure.polarity_stream = params_.polarity_stream();
    measure.aging_steps = params_.aging_steps();
    measure.p_tumble = params_.p_tumble();
    measure.motility_coeff = static_cast<float>(params_.motility());
    measure.max_shift = kMaximumShiftPerStep;
    measure.support_margin = static_cast<int>(kBrickSafetyMargin);
    measure.apply_tumble = apply_tumble;
    measure.compute_surface = compute_surface;
    if (!launch_measure_cells_only(measure, stream_)) {
        std::fprintf(stderr, "[3d] invalid base measurement launch contract\n");
        return false;
    }
    if (!phase_mark(1)) return false;
    if (!h_promoted_ids_.empty()) {
        const int promoted_count = static_cast<int>(h_promoted_ids_.size());
        const int measure_shards = promoted_measure_shard_count(promoted_count);
        if (measure_shards > 1) {
            MeasureArgs3D sharded = measure;
            sharded.shards = measure_shards;
            if (!detail::launch_measure_promoted_shards(
                    sharded, d_promoted_phi_[current_promoted_phi_index_],
                    d_promoted_ids_, promoted_count, promoted_edge_,
                    d_promoted_partials_, stream_)) {
                std::fprintf(stderr,
                    "[3d] invalid promoted measurement launch contract\n");
                return false;
            }
        } else {
            if (!launch_measure_promoted(
                    measure, d_promoted_phi_[current_promoted_phi_index_],
                    d_promoted_ids_, promoted_count, promoted_edge_, stream_)) {
                std::fprintf(stderr,
                    "[3d] invalid promoted measurement launch contract\n");
                return false;
            }
        }
    }
    if (!phase_mark(2)) return false;
    // Motion consumes the accepted-step Philox event exactly once, after all
    // base and promoted cells have refreshed V and interaction moments.
    // Adaptive recovery only remeasures the retained field and deliberately
    // preserves the accepted dynamics state bit-for-bit.
    if (apply_motion) launch_apply_cell_motion(measure, stream_);
    if (!gpu_ok(cudaGetLastError(), "measure-cells launch")) return false;
    volume_current_ = true;
    if (compute_surface) surface_current_ = true;
    return true;
}

bool Sim3D::ensure_measurements(bool require_surface, const char* context) {
    if (volume_current_ && (!require_surface || surface_current_)) return true;
    return refresh_measurements(false, require_surface) &&
           synchronize_and_check(context);
}

int Sim3D::promoted_update_shards(int promoted_count) const {
    if (promoted_count <= 0) return 1;
    if (options_.promoted_shards > 0)
        return std::min(options_.promoted_shards, kMaximumPromotedShards);
    const std::int64_t wave = static_cast<std::int64_t>(sm_count_) *
                              fast_promoted_blocks_per_sm_;
    const std::int64_t fitting = wave / promoted_count;
    return static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(kMaximumPromotedShards, fitting)));
}

int Sim3D::promoted_measure_shard_count(int promoted_count) const {
    if (promoted_count <= 0 || options_.promoted_measure_shards == 0)
        return 1;
    if (options_.promoted_measure_shards > 0)
        return std::min(options_.promoted_measure_shards,
                        kMaximumPromotedShards);
    const std::int64_t wave = options_.promoted_measure_auto_wave_ctas;
    const std::int64_t fitting = wave / promoted_count;
    return static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(kMaximumPromotedShards, fitting)));
}

int Sim3D::fast_base_update_shards() const {
    if (options_.fast_base_shards > 0)
        return std::min(options_.fast_base_shards, kMaximumPromotedShards);
    const int live_base = params_.num_cells
        - static_cast<int>(h_promoted_ids_.size());
    if (live_base <= 0) return 1;
    return occupancy_wave_shards(live_base, sm_count_,
                                 fast_base_blocks_per_sm_,
                                 kMaximumPromotedShards);
}

bool Sim3D::phase_mark(int slot) {
    if (!phase_active_) return true;
    if (slot < 0 || slot >= kPhaseEventSlots ||
        phase_recorded_ >= phase_capacity_) {
        std::fprintf(stderr, "[3d] invalid phase-timing event index\n");
        return false;
    }
    return gpu_ok(cudaEventRecord(
        phase_events_[static_cast<std::size_t>(phase_recorded_) *
                          kPhaseEventSlots + slot],
        stream_), "record phase-timing event");
}

void Sim3D::clear_phase_events() noexcept {
    phase_active_ = false;
    for (cudaEvent_t event : phase_events_)
        if (event) cudaEventDestroy(event);
    phase_events_.clear();
    phase_capacity_ = 0;
    phase_recorded_ = 0;
}

bool Sim3D::launch_one_step() {
    const bool full_moment = params_.full_moment_every > 0 &&
        steps_done_ % static_cast<std::uint64_t>(params_.full_moment_every) == 0;
    const bool measured_update = options_.strict && params_.verify_every > 0 &&
        (steps_done_ + 1u) %
            static_cast<std::uint64_t>(params_.verify_every) == 0u;
    if (!phase_mark(0)) return false;
    if (!gpu_ok(cudaMemcpyAsync(
                    d_accepted_cells_, d_cells_,
                    static_cast<std::size_t>(params_.num_cells) *
                        sizeof(CellState3D),
                    cudaMemcpyDeviceToDevice, stream_),
                "snapshot accepted cell state"))
        return false;
    if (!refresh_measurements(true, full_moment)) return false;
    const int next_S = S_buffers_ == 2 ? 1 - current_S_index_ : current_S_index_;
    const int next_phi = phi_buffers_ == 2 ? 1 - current_phi_index_
                                           : current_phi_index_;
    const int next_promoted_phi = 1 - current_promoted_phi_index_;
    if (S_buffers_ == 2) launch_clear_S(d_S_[next_S], layout_, stream_);
    if (!phase_mark(3)) return false;

    UpdateArgs3D update{};
    update.phi_in = current_phi();
    update.phi_out = phi_buffers_ == 2 ? d_phi_[next_phi] : nullptr;
    update.S_in = current_S();
    update.S_out = S_buffers_ == 2 ? d_S_[next_S] : nullptr;
    update.wall = weighted_wall_field();
    update.cells = d_cells_;
    update.global_flags = d_flags_;
    update.work_cursor = d_work_cursor_;
    update.layout = layout_;
    update.N = params_.num_cells;
    update.B = B_;
    update.dt = static_cast<float>(params_.dt);
    update.V0 = params_.volume0();
    update.volume_scale = params_.volume();
    update.bulk_scale = static_cast<float>(params_.bulk());
    update.interaction_coeff = static_cast<float>(params_.interaction());
    // Surface does not enter the evolution equations. It is measured only at
    // its requested cadence or before output that requires it.
    update.compute_surface = false;
    bool update_ok = true;
    if (phi_buffers_ == 2) {
        if (measured_update && measurement_shards_ > 1) {
            update_ok = launch_update_tiled_sharded(
                update, d_moment_partials_, measurement_shards_, stream_);
        } else if (!measured_update && fast_base_update_shards() > 1) {
            update_ok = detail::launch_update_tiled_sharded_fast(
                update, fast_base_update_shards(), stream_);
        } else if (measured_update) {
            update_ok = launch_update_tiled(
                update, update_grid_blocks_, stream_);
        } else {
            // The periodic slab-walk kernel folds z and cannot represent a
            // bounded domain. Use the boundary-aware tiled path there.
            if (params_.bounded_z()) {
                update_ok = detail::launch_update_tiled_fast(
                    update, update_grid_blocks_, stream_);
            } else {
                // Bit-identical to the tiled fast update and clears its own
                // destination while writing.
                update_ok = detail::launch_update_periodic_slab_walk_fast(
                    update, stream_);
            }
        }
    } else if (measured_update) {
        update_ok = launch_update_inplace(
            update, d_scratch_, scratch_slots_, stream_);
    } else {
        update_ok = detail::launch_update_inplace_fast(
            update, d_scratch_, scratch_slots_, stream_);
    }
    if (!update_ok) {
        std::fprintf(stderr, "[3d] invalid update launch contract\n");
        return false;
    }
    if (!phase_mark(4)) return false;
    if (!h_promoted_ids_.empty()) {
        const int promoted_count = static_cast<int>(h_promoted_ids_.size());
        const int promoted_shards =
            measured_update ? 1 : promoted_update_shards(promoted_count);
        bool promoted_ok;
        if (!measured_update && promoted_shards > 1) {
            promoted_ok = detail::launch_update_promoted_sharded_fast(
                update, d_promoted_phi_[current_promoted_phi_index_],
                d_promoted_phi_[next_promoted_phi], d_promoted_ids_,
                promoted_count, promoted_edge_, promoted_shards, stream_);
        } else {
            promoted_ok = launch_update_promoted(
                update, d_promoted_phi_[current_promoted_phi_index_],
                d_promoted_phi_[next_promoted_phi], d_promoted_ids_,
                promoted_count, promoted_edge_, measured_update, stream_);
        }
        if (!promoted_ok) {
            std::fprintf(stderr,
                         "[3d] invalid promoted update launch contract\n");
            return false;
        }
    }
    if (!phase_mark(5)) return false;
    launch_repair_after_fatal(update, stream_);
    if (!h_promoted_ids_.empty())
        launch_repair_promoted_after_fatal(
            update, d_promoted_phi_[current_promoted_phi_index_],
            d_promoted_phi_[next_promoted_phi], d_promoted_ids_,
            static_cast<int>(h_promoted_ids_.size()), promoted_edge_, stream_);
    launch_finalize_origins(d_cells_, params_.num_cells, d_flags_, stream_);
    // Field/S repair precedes this complete state rollback. Because all work
    // is in one stream, a later queued step snapshots the restored state even
    // when the host polls only every several accepted steps.
    launch_restore_cells_after_fatal(
        d_cells_, d_accepted_cells_, params_.num_cells, d_flags_,
        d_support_requests_, stream_);

    if (phi_buffers_ == 2) current_phi_index_ = next_phi;
    if (!h_promoted_ids_.empty())
        current_promoted_phi_index_ = next_promoted_phi;
    if (S_buffers_ == 2) {
        current_S_index_ = next_S;
    } else {
        // All old-S reads and in-place writes have completed in stream order.
        if (!reconstruct_current_S()) return false;
    }
    volume_current_ = measured_update;
    surface_current_ = false;
    launch_advance_step(d_step_, d_flags_, stream_);
    if (!phase_mark(6)) return false;
    if (phase_active_ && phase_recorded_ < phase_capacity_) ++phase_recorded_;
    ++steps_done_;
    return gpu_ok(cudaGetLastError(), "3D step launch sequence");
}

bool Sim3D::recover_support_exhaustion(
    const std::vector<std::uint32_t>& flags) {
    if (flags.size() != FLAG3D_COUNT ||
        flags[FLAG3D_SUPPORT_EXHAUSTED] == 0u)
        return false;
    for (int flag = 0; flag < FLAG3D_COUNT; ++flag) {
        if (flag != FLAG3D_SUPPORT_EXHAUSTED &&
            flag3d_is_fatal(static_cast<Flag3D>(flag)) &&
            flags[static_cast<std::size_t>(flag)] != 0u)
            return false;
    }

    const std::size_t count = static_cast<std::size_t>(params_.num_cells);
    std::vector<CellState3D> states(count);
    std::vector<std::uint32_t> requests(count, 0u);
    if (!gpu_ok(cudaMemcpy(states.data(), d_cells_,
                           states.size() * sizeof(CellState3D),
                           cudaMemcpyDeviceToHost),
                "read adaptive cell state") ||
        !gpu_ok(cudaMemcpy(requests.data(), d_support_requests_,
                           requests.size() * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost),
                "read adaptive support requests"))
        return false;
    std::vector<int> base_requests;
    bool promoted_exhausted = false;
    for (int n = 0; n < params_.num_cells; ++n) {
        const CellState3D& state = states[static_cast<std::size_t>(n)];
        const bool requested = requests[static_cast<std::size_t>(n)] != 0u ||
            (state.flags & flag3d_bit(FLAG3D_SUPPORT_EXHAUSTED)) != 0u;
        if (!requested) continue;
        if (cell_is_promoted(state, B_)) promoted_exhausted = true;
        else base_requests.push_back(n);
    }
    if (!promoted_exhausted && base_requests.empty()) {
        std::fprintf(stderr,
            "[3d] global support-exhaustion counter has no owning cell\n");
        return false;
    }

    bool migrated = false;
    if (promoted_exhausted) {
        if (promoted_edge_ >= maximum_support_edge_) {
            std::fprintf(stderr,
                "[3d] adaptive support B=%d reached the largest aligned "
                "cube below the minimum domain extent %d; the accepted "
                "state is retained\n",
                promoted_edge_, params_.minimum_domain_edge());
            return false;
        }
        const int increment = std::max(64, promoted_edge_ / 4);
        const std::int64_t raw_candidate =
            static_cast<std::int64_t>(promoted_edge_) + increment;
        const int candidate = raw_candidate >= maximum_support_edge_
            ? maximum_support_edge_
            : round_up_to_multiple(static_cast<int>(raw_candidate),
                                   kBrickAlignment);
        const int next_edge = std::min(candidate, maximum_support_edge_);
        migrated = grow_promoted_fields(next_edge, base_requests, &states);
    } else {
        migrated = allocate_promoted_fields(base_requests, &states);
    }
    if (!migrated ||
        !reconstruct_current_S() ||
        !refresh_measurements(false, surface_current_, false) ||
        !gpu_ok(cudaStreamSynchronize(stream_),
                "remeasure promoted support"))
        return false;

    std::vector<std::uint32_t> after(FLAG3D_COUNT, 0u);
    if (!gpu_ok(cudaMemcpy(after.data(), d_flags_,
                           after.size() * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost),
                "check promoted support"))
        return false;
    if (after[FLAG3D_SUPPORT_EXHAUSTED] != 0u) {
        // The measured support may require more than one tier. Each recursive
        // migration is transactional and advances only storage, never time.
        return recover_support_exhaustion(after);
    }
    if (any_fatal_flag(after)) {
        std::fprintf(stderr,
            "[3d] adaptive support remeasurement remained fatal\n");
        return false;
    }
    volume_current_ = true;
    ++recovery_events_;
    return true;
}

bool Sim3D::fatal_flags_present(bool print) {
    std::vector<std::uint32_t> flags(FLAG3D_COUNT, 0u);
    if (!gpu_ok(cudaMemcpy(flags.data(), d_flags_,
                           flags.size() * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost), "read fatal flags"))
        return true;
    if (!any_fatal_flag(flags)) return false;
    if (print) {
        std::fprintf(stderr, "[3d] integrity flags at accepted step %llu:\n",
                     static_cast<unsigned long long>(steps_done_));
        for (int i = 0; i < FLAG3D_COUNT; ++i)
            if (flags[static_cast<std::size_t>(i)] != 0)
                std::fprintf(stderr, "      %-34s %u\n",
                    flag_name(static_cast<Flag3D>(i)),
                    flags[static_cast<std::size_t>(i)]);
    }
    return true;
}

bool Sim3D::synchronize_and_check(const char* context) {
    if (!gpu_ok(cudaStreamSynchronize(stream_), context)) return false;
    std::uint64_t accepted = 0;
    if (!gpu_ok(cudaMemcpy(&accepted, d_step_, sizeof(accepted),
                           cudaMemcpyDeviceToHost), "read accepted step"))
        return false;
    steps_done_ = accepted;
    std::vector<std::uint32_t> flags(FLAG3D_COUNT, 0u);
    if (!gpu_ok(cudaMemcpy(flags.data(), d_flags_,
                           flags.size() * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost), "read integrity flags"))
        return false;
    if (!any_fatal_flag(flags)) return true;
    if (recover_support_exhaustion(flags)) return true;
    (void)fatal_flags_present(true);
    return false;
}

bool Sim3D::open_trajectory() {
    const std::filesystem::path path(options_.trajectory_path);
    std::error_code ec;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        std::fprintf(stderr, "[3d] cannot create trajectory directory: %s\n",
                     ec.message().c_str());
        return false;
    }
    const std::string expected_header = trajectory_header(
        params_, B_, options_.promoted_measure_shards,
        options_.promoted_measure_auto_wave_ctas, measurement_shards_,
        trajectory_every_);
    const char* const expected_columns = trajectory_columns(params_);
    if (expected_header.empty()) {
        std::fprintf(stderr, "[3d] trajectory metadata is too long\n");
        return false;
    }
    // a+ creates without truncating and also removes the exists/open race.
    trajectory_file_ = std::fopen(options_.trajectory_path.c_str(), "a+");
    if (!trajectory_file_) {
        std::fprintf(stderr, "[3d] cannot open trajectory %s\n",
                     options_.trajectory_path.c_str());
        return false;
    }
    std::rewind(trajectory_file_);
    const int first_byte = std::fgetc(trajectory_file_);
    if (first_byte == EOF && std::ferror(trajectory_file_)) {
        close_trajectory();
        return false;
    }
    const bool empty = first_byte == EOF;
    std::clearerr(trajectory_file_);
    if (empty) {
        // C update streams require a positioning operation when switching
        // from the read probe to writing.
        if (std::fseek(trajectory_file_, 0, SEEK_END) != 0) {
            close_trajectory();
            return false;
        }
        if (std::fprintf(trajectory_file_, "%s\n%s\n",
                         expected_header.c_str(), expected_columns) < 0 ||
            std::fflush(trajectory_file_) != 0) {
            close_trajectory();
            return false;
        }
        trajectory_header_written_ = true;
        return true;
    }

    std::rewind(trajectory_file_);
    char line[2048]{};
    bool line_error = false;
    auto read_line = [&]() {
        if (!std::fgets(line, sizeof(line), trajectory_file_)) return false;
        std::size_t length = std::strlen(line);
        if (length > 0 && line[length - 1] == '\n') {
            line[--length] = '\0';
            if (length > 0 && line[length - 1] == '\r') line[--length] = '\0';
        } else if (!std::feof(trajectory_file_)) {
            line_error = true;
            return false;
        }
        return true;
    };
    char first_line[sizeof(line)]{};
    if (!read_line()) {
        std::fprintf(stderr,
            "[3d] refusing to append: trajectory metadata does not match this run\n");
        close_trajectory();
        return false;
    }
    std::memcpy(first_line, line, sizeof(line));
    if (!read_line() ||
        !trajectory_metadata_compatible(first_line, line, expected_header,
                                        expected_columns)) {
        std::fprintf(stderr,
            "[3d] refusing to append: trajectory metadata does not match this run\n");
        close_trajectory();
        return false;
    }

    std::uint64_t rows = 0;
    double frame_time = 0.0, previous_frame_time = -1.0;
    while (read_line()) {
        if (line[0] == '\0') {
            std::fprintf(stderr, "[3d] blank line in trajectory payload\n");
            close_trajectory();
            return false;
        }
        double values[20]{};
        long long global_id = 0;
        char extra = '\0';
        const int parsed = params_.resolved_wall_channel()
            ? std::sscanf(
                  line,
                  "%lf %lld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
                  "%lf %lf %lf %lf %lf %lf %lf %lf %c",
                  &values[0], &global_id, &values[1], &values[2], &values[3],
                  &values[4], &values[5], &values[6], &values[7], &values[8],
                  &values[9], &values[10], &values[11], &values[12],
                  &values[13], &values[14], &values[15], &values[16],
                  &values[17], &values[18], &values[19], &extra)
            : std::sscanf(
                  line,
                  "%lf %lld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
                  "%lf %lf %lf %lf %c",
                  &values[0], &global_id, &values[1], &values[2], &values[3],
                  &values[4], &values[5], &values[6], &values[7], &values[8],
                  &values[9], &values[10], &values[11], &values[12],
                  &values[13], &values[14], &values[15], &extra);
        const int expected_values = params_.resolved_wall_channel() ? 21 : 17;
        const int finite_values = params_.resolved_wall_channel() ? 20 : 16;
        bool finite = parsed == expected_values;
        for (int k = 0; k < finite_values; ++k)
            finite = finite && std::isfinite(values[k]);
        if (!finite) {
            std::fprintf(stderr, "[3d] malformed trajectory row %llu\n",
                static_cast<unsigned long long>(rows + 1));
            close_trajectory();
            return false;
        }
        const std::size_t within = static_cast<std::size_t>(
            rows % static_cast<std::uint64_t>(params_.num_cells));
        if (within == 0) {
            frame_time = values[0];
            if (rows != 0 && !(frame_time > previous_frame_time)) {
                std::fprintf(stderr,
                    "[3d] trajectory frame times are not strictly increasing\n");
                close_trajectory();
                return false;
            }
        } else if (values[0] != frame_time) {
            std::fprintf(stderr,
                "[3d] trajectory contains a frame with mixed times\n");
            close_trajectory();
            return false;
        }
        if (global_id != static_cast<long long>(within)) {
            std::fprintf(stderr,
                "[3d] trajectory cells are not in canonical ID order\n");
            close_trajectory();
            return false;
        }
        ++rows;
        if (rows % static_cast<std::uint64_t>(params_.num_cells) == 0)
            previous_frame_time = frame_time;
    }
    if (line_error || std::ferror(trajectory_file_) ||
        rows % static_cast<std::uint64_t>(params_.num_cells) != 0) {
        std::fprintf(stderr,
            "[3d] refusing to append to a partial trajectory frame\n");
        close_trajectory();
        return false;
    }
    trajectory_frames_ = rows / static_cast<std::uint64_t>(params_.num_cells);
    if (trajectory_frames_ > 0) {
        const double step_value = previous_frame_time / params_.dt;
        if (!std::isfinite(step_value) || step_value < 0.0 ||
            step_value > 9.0e18) {
            std::fprintf(stderr, "[3d] trajectory time is outside step range\n");
            close_trajectory();
            return false;
        }
        const auto last_step = static_cast<std::uint64_t>(std::llround(step_value));
        const double reconstructed = static_cast<double>(last_step) * params_.dt;
        const double tolerance = 1.0e-10 *
            std::max(1.0, std::fabs(previous_frame_time));
        if (std::fabs(reconstructed - previous_frame_time) > tolerance ||
            last_step > steps_done_) {
            std::fprintf(stderr,
                "[3d] trajectory extends beyond the initialized state\n");
            close_trajectory();
            return false;
        }
        last_trajectory_step_ = last_step;
    }
    std::clearerr(trajectory_file_);
    if (std::fseek(trajectory_file_, 0, SEEK_END) != 0) {
        close_trajectory();
        return false;
    }
    trajectory_header_written_ = true;
    return true;
}

bool Sim3D::append_trajectory() {
    if (!trajectory_file_) return true;
    if (!ensure_measurements(true, "trajectory measurement"))
        return false;
    if (params_.resolved_wall_channel()) {
        launch_measure_wall_diagnostics(
            current_phi(),
            d_promoted_phi_[current_promoted_phi_index_], d_cells_,
            params_.num_cells, B_, layout_, d_wall_psi_sq_,
            params_.channel_height, params_.channel_padding, stream_);
        if (!gpu_ok(cudaGetLastError(), "channel wall-overlap measurement"))
            return false;
    }
    launch_pack_trajectory(d_cells_, d_trajectory_, params_.num_cells,
                           layout_, stream_, params_.channel_padding);
    if (!gpu_ok(cudaMemcpyAsync(h_trajectory_.data(), d_trajectory_,
                                h_trajectory_.size() * sizeof(TrajPackedCell3D),
                                cudaMemcpyDeviceToHost, stream_),
                "trajectory D2H") ||
        !gpu_ok(cudaStreamSynchronize(stream_), "trajectory synchronization"))
        return false;
    if (!trajectory_header_written_) return false;
    const double t = time();
    for (const TrajPackedCell3D& cell : h_trajectory_) {
        const int base_written = std::fprintf(trajectory_file_,
            "%.17g %lld %.17g %.17g %.17g %.9g %.9g %.9g %.9g %.9g %.9g "
            "%.9g %.9g %.17g %.17g %.17g %.9g",
            t, static_cast<long long>(cell.global_id),
            static_cast<double>(cell.x), static_cast<double>(cell.y),
            static_cast<double>(cell.z),
            static_cast<double>(cell.velocity_x),
            static_cast<double>(cell.velocity_y),
            static_cast<double>(cell.velocity_z),
            static_cast<double>(cell.polarity_x),
            static_cast<double>(cell.polarity_y),
            static_cast<double>(cell.polarity_z),
            static_cast<double>(cell.gamma), static_cast<double>(cell.v_A),
            static_cast<double>(cell.surface),
            params_.substrate_slab()
                ? hemisphere_normalized_free_surface_compactness(
                      static_cast<double>(cell.volume),
                      static_cast<double>(cell.surface))
                : sphere_sphericity(static_cast<double>(cell.volume),
                                    static_cast<double>(cell.surface)),
            static_cast<double>(cell.volume),
            static_cast<double>(cell.phi_max));
        const int suffix_written = params_.resolved_wall_channel()
            ? std::fprintf(
                  trajectory_file_, " %.17g %.17g %.17g %.17g\n",
                  cell.wall_overlap,
                  cell.volume > 0.0 ? cell.wall_overlap / cell.volume : 0.0,
                  cell.outside_slit_volume,
                  cell.volume > 0.0
                      ? cell.outside_slit_volume / cell.volume : 0.0)
            : std::fputc('\n', trajectory_file_);
        if (base_written < 0 || suffix_written < 0) {
            std::fprintf(stderr, "[3d] failed to write trajectory frame\n");
            return false;
        }
    }
    if (std::fflush(trajectory_file_) != 0 || std::ferror(trajectory_file_)) {
        std::fprintf(stderr, "[3d] failed to flush trajectory frame\n");
        return false;
    }
    ++trajectory_frames_;
    last_trajectory_step_ = steps_done_;
    return true;
}

void Sim3D::close_trajectory() {
    if (!trajectory_file_) return;
    std::fflush(trajectory_file_);
    std::fclose(trajectory_file_);
    trajectory_file_ = nullptr;
}

bool Sim3D::checkpoint_at_current_state(const std::string& path) {
    if (!ensure_measurements(true, "checkpoint measurement"))
        return false;
    if (trajectory_every_ > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr,
            "[3d] trajectory interval exceeds the checkpoint limit\n");
        return false;
    }
    CheckpointWriteView3D view{};
    view.params = &params_;
    view.step = steps_done_;
    view.time = time();
    view.brick_edge = B_;
    view.print_interval = options_.print_interval;
    view.trajectory_interval = static_cast<std::uint32_t>(trajectory_every_);
    view.base_measure_shards = measurement_shards_;
    view.promoted_measure_reduction = {
        options_.promoted_measure_shards,
        options_.promoted_measure_auto_wave_ctas};
    view.d_cells = d_cells_;
    view.d_phi = current_phi();
    view.h_promoted_phi =
        h_promoted_phi_[current_promoted_phi_index_].data();
    view.stream = stream_;
    return checkpoint_write_3d(path, view);
}

bool Sim3D::save_checkpoint(const std::string& path) {
    if (!synchronize_and_check("checkpoint preflight")) return false;
    return checkpoint_at_current_state(path);
}

bool Sim3D::verify(double* max_relative_volume_error,
                   std::uint32_t* maximum_S,
                   std::uint64_t* support_edge_voxels) {
    if (!max_relative_volume_error || !maximum_S || !support_edge_voxels)
        return false;
    if (!ensure_measurements(false, "verification measurement")) return false;
    if (!d_verify_cells_ &&
        !gpu_ok(cudaMalloc(reinterpret_cast<void**>(&d_verify_cells_),
                           static_cast<std::size_t>(params_.num_cells) *
                               sizeof(VerifyCell3D)),
                "cudaMalloc(verify cells)"))
        return false;
    if (!d_verify_S_ &&
        !gpu_ok(cudaMalloc(reinterpret_cast<void**>(&d_verify_S_),
                           sizeof(*d_verify_S_)),
                "cudaMalloc(verify S)"))
        return false;
    if (!gpu_ok(cudaMemsetAsync(
                    d_verify_cells_, 0,
                    static_cast<std::size_t>(params_.num_cells) *
                        sizeof(VerifyCell3D), stream_),
                "clear cell verifier") ||
        !gpu_ok(cudaMemsetAsync(d_verify_S_, 0, sizeof(*d_verify_S_), stream_),
                "clear S verifier"))
        return false;
    launch_verify_cells(current_phi(), d_cells_, d_verify_cells_,
                        params_.num_cells, B_, layout_, stream_);
    if (!h_promoted_ids_.empty())
        launch_verify_promoted(
            d_promoted_phi_[current_promoted_phi_index_], d_cells_,
            d_verify_cells_, d_promoted_ids_,
            params_.num_cells, static_cast<int>(h_promoted_ids_.size()),
            promoted_edge_, layout_, stream_);
    launch_verify_S(current_S(), layout_, d_verify_S_, stream_);
    std::vector<VerifyCell3D> cells(
        static_cast<std::size_t>(params_.num_cells));
    if (!gpu_ok(cudaMemcpyAsync(cells.data(), d_verify_cells_,
                                cells.size() * sizeof(VerifyCell3D),
                                cudaMemcpyDeviceToHost, stream_),
                "verify cells D2H") ||
        !gpu_ok(cudaMemcpyAsync(maximum_S, d_verify_S_, sizeof(*maximum_S),
                                cudaMemcpyDeviceToHost, stream_),
                "verify S D2H") ||
        !gpu_ok(cudaStreamSynchronize(stream_), "verify synchronization"))
        return false;
    *max_relative_volume_error = 0.0;
    *support_edge_voxels = 0;
    for (std::size_t index = 0; index < cells.size(); ++index) {
        const VerifyCell3D& cell = cells[index];
        if (cell.verified != 1u) {
            std::fprintf(stderr,
                         "[3d] verifier did not inspect cell slot %zu\n", index);
            return false;
        }
        const double denominator = std::max(1.0, std::fabs(cell.measured_V));
        *max_relative_volume_error = std::max(
            *max_relative_volume_error,
            std::fabs(cell.measured_V - cell.state_V) / denominator);
        *support_edge_voxels += cell.support_edge_count;
        if (cell.nonfinite_count != 0) return false;
    }
    return true;
}

bool Sim3D::run() {
    const std::uint64_t target = static_cast<std::uint64_t>(params_.total_steps());
    if (steps_done_ > target) {
        std::fprintf(stderr, "[3d] checkpoint step exceeds requested t_end\n");
        return false;
    }
    // Output may be disabled for this leg; the resolved cadence remains part
    // of subsequent checkpoints.
    const std::uint64_t trajectory_every = trajectory_file_
        ? trajectory_every_ : 0;
    std::uint64_t next_trajectory = schedule_grid_after(
        steps_done_, trajectory_every);
    std::uint64_t next_checkpoint = !options_.checkpoint_dir.empty() &&
            options_.checkpoint_interval > 0
        ? schedule_after(steps_done_, static_cast<std::uint64_t>(
              options_.checkpoint_interval))
        : std::numeric_limits<std::uint64_t>::max();
    std::uint64_t next_tagged = !options_.checkpoint_dir.empty() &&
            options_.save_interval > 0
        ? schedule_after(steps_done_, static_cast<std::uint64_t>(
              options_.save_interval))
        : std::numeric_limits<std::uint64_t>::max();
    std::uint64_t next_print = options_.print_interval > 0
        ? schedule_after(steps_done_, static_cast<std::uint64_t>(
              options_.print_interval))
        : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t verify_every = options_.strict &&
            params_.verify_every > 0
        ? static_cast<std::uint64_t>(params_.verify_every) : 0;
    const std::uint64_t verify_offset = verify_every > 0
        ? verify_every - steps_done_ % verify_every : 0;
    std::uint64_t next_verify = schedule_after(steps_done_, verify_offset);
    int since_poll = 0;

    while (steps_done_ < target && !terminate_requested) {
        if (!launch_one_step()) return false;
        ++since_poll;
        const bool event = steps_done_ >= next_trajectory ||
                           steps_done_ >= next_checkpoint ||
                           steps_done_ >= next_tagged ||
                           steps_done_ >= next_print ||
                           steps_done_ >= next_verify ||
                           steps_done_ >= target ||
                           since_poll >= kHostPollEvery;
        if (!event) continue;
        since_poll = 0;
        if (!synchronize_and_check("step synchronization")) {
            std::fprintf(stderr,
                "[3d] the failed in-memory step is diagnostic only; the last "
                "completed rolling checkpoint was not replaced\n");
            return false;
        }
        if (steps_done_ >= next_verify) {
            double volume_error = 0.0;
            std::uint32_t maximum = 0;
            std::uint64_t edges = 0;
            if (!verify(&volume_error, &maximum, &edges) ||
                volume_error > 1.0e-12 || edges != 0) {
                std::fprintf(stderr,
                    "[3d] strict verification failed: relV %.3e, edge voxels %llu\n",
                    volume_error, static_cast<unsigned long long>(edges));
                return false;
            }
            advance_schedule(&next_verify, verify_every);
        }
        if (steps_done_ >= next_print) {
            std::printf("[3d] step %llu  t %.6g  mode %s\n",
                static_cast<unsigned long long>(steps_done_), time(),
                storage_mode_name(selected_mode_));
            std::fflush(stdout);
            advance_schedule(&next_print,
                static_cast<std::uint64_t>(options_.print_interval));
        }
        if (trajectory_file_ && steps_done_ >= next_trajectory) {
            if (!append_trajectory()) return false;
            advance_schedule(&next_trajectory, trajectory_every);
        }
        if (!options_.checkpoint_dir.empty() && steps_done_ >= next_checkpoint) {
            const std::filesystem::path rolling =
                std::filesystem::path(options_.checkpoint_dir) / "checkpoint.pf3d";
            if (!checkpoint_at_current_state(rolling.string())) return false;
            advance_schedule(&next_checkpoint,
                static_cast<std::uint64_t>(options_.checkpoint_interval));
        }
        if (!options_.checkpoint_dir.empty() && steps_done_ >= next_tagged) {
            char name[80]{};
            std::snprintf(name, sizeof(name), "checkpoint_%012llu.pf3d",
                          static_cast<unsigned long long>(steps_done_));
            if (!checkpoint_at_current_state(
                    (std::filesystem::path(options_.checkpoint_dir) / name).string()))
                return false;
            advance_schedule(&next_tagged,
                static_cast<std::uint64_t>(options_.save_interval));
        }
    }

    if (!synchronize_and_check("final synchronization")) return false;
    if (trajectory_file_ && last_trajectory_step_ != steps_done_ &&
        !append_trajectory()) return false;
    if (options_.final_checkpoint && !options_.checkpoint_dir.empty()) {
        const std::filesystem::path rolling =
            std::filesystem::path(options_.checkpoint_dir) / "checkpoint.pf3d";
        if (!checkpoint_at_current_state(rolling.string())) return false;
    }
    if (terminate_requested)
        std::printf("[3d] termination requested; stopped at accepted step %llu\n",
                    static_cast<unsigned long long>(steps_done_));
    return true;
}

bool Sim3D::bench(int steps, double* milliseconds_per_step) {
    if (steps <= 0 || !milliseconds_per_step) return false;
    for (int i = 0; i < 2; ++i)
        if (!launch_one_step()) return false;
    if (!synchronize_and_check("benchmark warmup")) return false;
    const std::uint64_t guard_steps = steps_done_;
    const std::size_t guard_promoted = h_promoted_ids_.size();
    const int guard_edge = promoted_edge_;
    const std::uint64_t guard_recoveries = recovery_events_;
    if (!gpu_ok(cudaEventRecord(bench_start_, stream_), "record benchmark start"))
        return false;
    for (int i = 0; i < steps; ++i)
        if (!launch_one_step()) return false;
    if (!gpu_ok(cudaEventRecord(bench_stop_, stream_), "record benchmark stop") ||
        !gpu_ok(cudaEventSynchronize(bench_stop_), "wait benchmark stop"))
        return false;
    float elapsed = 0.0f;
    if (!gpu_ok(cudaEventElapsedTime(&elapsed, bench_start_, bench_stop_),
                "benchmark elapsed time") ||
        !synchronize_and_check("benchmark flags"))
        return false;
    *milliseconds_per_step = static_cast<double>(elapsed) / steps;
    std::printf("[3d] benchmark %d steps: %.6f ms/step (%s, B=%d, N=%d, "
                "scratch=%d)\n", steps, *milliseconds_per_step,
                storage_mode_name(selected_mode_), B_, params_.num_cells,
                scratch_slots_);
    const std::uint64_t accepted = steps_done_ - guard_steps;
    const bool window_clean =
        accepted == static_cast<std::uint64_t>(steps) &&
        h_promoted_ids_.size() == guard_promoted &&
        promoted_edge_ == guard_edge &&
        recovery_events_ == guard_recoveries;
    std::printf("[3d] bench guard: accepted %llu/%d steps, promotions "
                "%zu->%zu, promoted_edge %d->%d, recoveries +%llu; %s\n",
                static_cast<unsigned long long>(accepted), steps,
                guard_promoted, h_promoted_ids_.size(), guard_edge,
                promoted_edge_,
                static_cast<unsigned long long>(recovery_events_
                                                - guard_recoveries),
                window_clean ? "VALID"
                             : "INVALID - reject this timing window");
    if (!window_clean) return false;
    if (options_.bench_phases && !bench_phase_report(steps)) return false;
    return true;
}

bool Sim3D::bench_phase_report(int steps) {
    // Run phase timing in a second window so its events do not
    // perturb the uninstrumented timing result.
    clear_phase_events();
    phase_capacity_ = steps;
    phase_events_.assign(
        static_cast<std::size_t>(kPhaseEventSlots) * steps, nullptr);
    for (cudaEvent_t& event : phase_events_) {
        if (!gpu_ok(cudaEventCreate(&event), "phase event create")) {
            clear_phase_events();
            return false;
        }
    }
    // Validate the instrumented window independently.
    const std::uint64_t phase_guard_steps = steps_done_;
    const std::size_t phase_guard_promoted = h_promoted_ids_.size();
    const int phase_guard_edge = promoted_edge_;
    const std::uint64_t phase_guard_recoveries = recovery_events_;
    phase_active_ = true;
    for (int i = 0; i < steps; ++i) {
        if (!launch_one_step()) {
            clear_phase_events();
            return false;
        }
    }
    phase_active_ = false;
    if (!synchronize_and_check("phase-timing window")) {
        clear_phase_events();
        return false;
    }
    const std::uint64_t phase_accepted = steps_done_ - phase_guard_steps;
    const bool phase_clean =
        phase_accepted == static_cast<std::uint64_t>(steps) &&
        phase_recorded_ == steps &&
        h_promoted_ids_.size() == phase_guard_promoted &&
        promoted_edge_ == phase_guard_edge &&
        recovery_events_ == phase_guard_recoveries;
    std::printf("[3d] phase guard: accepted %llu/%d steps, recorded %d, "
                "promotions %zu->%zu, promoted_edge %d->%d, recoveries "
                "+%llu; %s\n",
                static_cast<unsigned long long>(phase_accepted), steps,
                phase_recorded_, phase_guard_promoted,
                h_promoted_ids_.size(), phase_guard_edge, promoted_edge_,
                static_cast<unsigned long long>(recovery_events_
                                                - phase_guard_recoveries),
                phase_clean ? "VALID"
                            : "INVALID - reject this phase window");
    if (!phase_clean) {
        clear_phase_events();
        return false;
    }

    static const char* const kPhaseNames[6] = {
        "base measurement", "promoted measurement", "motion+clear",
        "base update", "promoted update", "repair/finalize/scatter"};
    const int recorded = phase_recorded_;
    std::vector<float> samples(static_cast<std::size_t>(recorded));
    std::printf("[3d] phase timing over %d steps (median ms/step):\n",
                recorded);
    double accounted = 0.0;
    for (int phase = 0; phase < 6; ++phase) {
        for (int k = 0; k < recorded; ++k) {
            float ms = 0.0f;
            if (!gpu_ok(cudaEventElapsedTime(
                    &ms,
                    phase_events_[static_cast<std::size_t>(k) *
                                      kPhaseEventSlots + phase],
                    phase_events_[static_cast<std::size_t>(k) *
                                      kPhaseEventSlots + phase + 1]),
                    "read phase-timing event")) {
                clear_phase_events();
                return false;
            }
            samples[static_cast<std::size_t>(k)] = ms;
        }
        std::nth_element(samples.begin(),
                         samples.begin() + recorded / 2,
                         samples.begin() + recorded);
        const double median =
            samples[static_cast<std::size_t>(recorded / 2)];
        accounted += median;
        std::printf("      %-24s %9.4f\n", kPhaseNames[phase], median);
    }
    if (recorded > 1) {
        for (int k = 0; k + 1 < recorded; ++k) {
            float ms = 0.0f;
            if (!gpu_ok(cudaEventElapsedTime(
                    &ms,
                    phase_events_[static_cast<std::size_t>(k) *
                                      kPhaseEventSlots + 6],
                    phase_events_[static_cast<std::size_t>(k + 1) *
                                      kPhaseEventSlots]),
                    "read phase-timing gap")) {
                clear_phase_events();
                return false;
            }
            samples[static_cast<std::size_t>(k)] = ms;
        }
        std::nth_element(samples.begin(),
                         samples.begin() + (recorded - 1) / 2,
                         samples.begin() + (recorded - 1));
        const double gap =
            samples[static_cast<std::size_t>((recorded - 1) / 2)];
        std::printf("      %-24s %9.4f\n", "host/output gap", gap);
        accounted += gap;
    }
    std::printf("      %-24s %9.4f\n", "sum of medians", accounted);
    clear_phase_events();
    return true;
}

}  // namespace pf3d
