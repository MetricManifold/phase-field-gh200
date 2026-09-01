#pragma once

// CUDA state and launch interfaces for the three-dimensional solver. Its
// fields are dense x-fastest cubes, and updates stream small bricks through
// shared memory instead of staging a whole cell.

#include "params.cuh"
#include "rng.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace pf3d {

constexpr int kThreads3D = 256;
constexpr int kBrickX = 32;
constexpr int kBrickY = 16;
constexpr int kBrickZ = 8;
constexpr int kHaloX = kBrickX + 2;
constexpr int kHaloY = kBrickY + 2;
constexpr int kHaloZ = kBrickZ + 2;
constexpr int kBrickVoxels = kBrickX * kBrickY * kBrickZ;
constexpr int kHaloVoxels = kHaloX * kHaloY * kHaloZ;

// Brick kernels double-buffer halo tiles so global reads overlap stencil work.
// Launches and occupancy queries must use this extent after raising the
// per-kernel dynamic-shared-memory limit; the launch helpers do both.
constexpr std::size_t kHaloPipelineBytes =
    2u * static_cast<std::size_t>(kHaloVoxels) * sizeof(float);
cudaError_t configure_tile_kernel_shared_memory();

static_assert(kThreads3D == 256, "the deterministic reductions assume 8 warps");
static_assert(kBrickX % 4 == 0, "brick rows must support 16-byte transfers");

// Sticky integrity counters.  CellState3D::flags uses the same values as bit
// positions; global_flags, when non-null, contains one count per value.
enum Flag3D : int {
    FLAG3D_S_OVERFLOW = 0,
    FLAG3D_Q_CLAMP = 1,
    FLAG3D_S_NEGATIVE = 2,
    FLAG3D_NONFINITE = 3,
    FLAG3D_V_NONPOSITIVE = 4,
    FLAG3D_SUPPORT_EXHAUSTED = 5,
    FLAG3D_SUPPORT_EDGE = 6,
    FLAG3D_DESTINATION_CLIP = 7,
    FLAG3D_INVALID_GEOMETRY = 8,
    FLAG3D_SLAB_TOP_CONTACT = 9,
    FLAG3D_COUNT = 10
};

static_assert(FLAG3D_COUNT <= 32, "CellState3D::flags is a 32-bit bit set");

PF3D_HD constexpr std::uint32_t flag3d_bit(Flag3D flag) {
    return 1u << static_cast<unsigned>(flag);
}

// Reaching the support threshold at the brick edge is reported for sizing
// diagnostics, but is not itself evidence of lost data.  All other flags stop
// advancement and retain the last accepted state.
constexpr std::uint32_t kFatalFlagMask3D =
    flag3d_bit(FLAG3D_S_OVERFLOW)
    | flag3d_bit(FLAG3D_Q_CLAMP)
    | flag3d_bit(FLAG3D_S_NEGATIVE)
    | flag3d_bit(FLAG3D_NONFINITE)
    | flag3d_bit(FLAG3D_V_NONPOSITIVE)
    | flag3d_bit(FLAG3D_SUPPORT_EXHAUSTED)
    | flag3d_bit(FLAG3D_DESTINATION_CLIP)
    | flag3d_bit(FLAG3D_INVALID_GEOMETRY)
    | flag3d_bit(FLAG3D_SLAB_TOP_CONTACT);

PF3D_HD constexpr bool flag3d_is_fatal(Flag3D flag) {
    return (kFatalFlagMask3D & flag3d_bit(flag)) != 0u;
}

// Restart-complete per-cell state. Origins are deliberately unwrapped: they
// retain continuous cell displacement while field/S addressing wraps on each
// periodic axis (x/y only for bounded-z geometries). A checkpoint-time
// measurement may schedule recentering, so pending shifts are stored
// explicitly.
struct alignas(64) CellState3D {
    std::int64_t global_id;
    std::int64_t origin_x, origin_y, origin_z;

    float polarity_x, polarity_y, polarity_z;
    float velocity_x, velocity_y, velocity_z;
    float gamma, v_A, R_tgt;
    float phi_max;

    std::uint32_t shift_ctr;
    std::uint32_t tumble_ctr;
    std::uint32_t flags;
    std::uint32_t pad0;

    std::int32_t bb_lo_x, bb_hi_x;
    std::int32_t bb_lo_y, bb_hi_y;
    std::int32_t bb_lo_z, bb_hi_z;
    std::int32_t pending_shift_x, pending_shift_y, pending_shift_z;
    std::uint32_t pad1;

    double V;
    double Cx, Cy, Cz;
    double surface;
    double Ix, Iy, Iz;

    // Zero denotes the ordinary base brick. A larger aligned edge selects a
    // persistent adaptive-support allocation for this cell.
    std::uint32_t storage_edge;
    std::uint32_t pad2;
    // Output-only diagnostic; recomputed from phi at trajectory cadence and
    // deliberately absent from the restart contract.
    double wall_overlap;
    double outside_slit_volume;
    std::uint32_t reserved[10];
};

static_assert(sizeof(CellState3D) == 256, "CellState3D checkpoint ABI drift");
static_assert(alignof(CellState3D) == 64, "CellState3D alignment drift");
static_assert(offsetof(CellState3D, storage_edge) == 192,
              "CellState3D storage offset drift");

PF3D_HD constexpr int cell_support_edge(const CellState3D& cell,
                                        int base_edge) {
    const std::uint32_t stored = cell.storage_edge;
    return stored == 0u ? base_edge : static_cast<int>(stored);
}

PF3D_HD constexpr bool cell_is_promoted(const CellState3D& cell,
                                        int base_edge) {
    return cell_support_edge(cell, base_edge) > base_edge;
}

// Dense aggregate-field layout. Valid x values occupy [0,nx); each y row has
// pitch_x uint32 words, and planes have ny*pitch_x words. Boundary flags
// distinguish periodic volume, sessile-cap substrate, and two-wall channel.
struct SLayout3D {
    int nx, ny, nz;
    int pitch_x;
    std::uint32_t boundary_flags = kBoundaryPeriodicXYZ3D;

    __host__ __device__ constexpr std::size_t plane_words() const {
        return static_cast<std::size_t>(pitch_x) * static_cast<std::size_t>(ny);
    }
    __host__ __device__ constexpr std::size_t words() const {
        return plane_words() * static_cast<std::size_t>(nz);
    }
    __host__ __device__ constexpr bool periodic_xyz() const {
        return boundary_flags == kBoundaryPeriodicXYZ3D;
    }
    __host__ __device__ constexpr bool substrate_slab() const {
        return boundary_flags == kBoundarySubstrateSlab3D;
    }
    __host__ __device__ constexpr bool hard_wall_channel() const {
        return boundary_flags == kBoundaryHardWallChannel3D;
    }
    __host__ __device__ constexpr bool bounded_z() const {
        return substrate_slab() || hard_wall_channel();
    }
};

// Immutable channel-wall contribution already scaled by kappa_w/kappa.  Its
// plane count is part of the launch contract so a channel kernel cannot infer
// a profile from unrelated storage.
struct WeightedWallField3D {
    const float* values = nullptr;
    int planes = 0;
};

PF3D_HD constexpr bool valid_weighted_wall_field(
    const WeightedWallField3D& wall, const SLayout3D& layout) {
    return layout.hard_wall_channel()
        ? wall.values != nullptr && wall.planes == layout.nz
        : wall.values == nullptr && wall.planes == 0;
}

struct InitArgs3D {
    float* phi_even;
    // Optional in compact in-place mode; when null only phi_even is seeded.
    float* phi_odd;
    CellState3D* cells;            // gid/scalars/polarity are pre-populated
    const Vec3* centres;           // unwrapped or primary-box coordinates
    int N;
    int B;
    SLayout3D layout;
    float lambda;
    // Centered CPU-reference calibration; displaced float32 seeds are measured
    // before their first update and need not equal V0 exactly.
    float seed_radius;
};

struct ScatterArgs3D {
    const float* phi;
    CellState3D* cells;
    std::uint32_t* S;
    std::uint32_t* global_flags;
    SLayout3D layout;
    int N;
    int B;
};

// One deterministic moment reduction produced by a spatial shard.  Low-cell-
// count measurement and throughput-update kernels share this record layout;
// each finalize kernel combines the selected records in fixed shard order.
struct alignas(16) MomentPartial3D {
    double V, Cx, Cy, Cz;
    double surface;
    double Ix, Iy, Iz;
    int lo_x, hi_x, lo_y, hi_y, lo_z, hi_z;
    unsigned int phi_max_bits;
    unsigned int padding;
};

struct MeasureArgs3D {
    const float* phi;
    const std::uint32_t* S;
    WeightedWallField3D wall;
    CellState3D* cells;
    std::uint32_t* global_flags;
    const std::uint64_t* step;
    SLayout3D layout;
    int N;
    int B;
    MomentPartial3D* partials;
    int shards;
    std::uint64_t polarity_stream;
    // Number of accepted steps for which self-propulsion and tumble events are
    // suppressed.  The interaction-induced velocity remains active.
    std::uint64_t aging_steps;
    double p_tumble;
    float motility_coeff;
    int max_shift;
    int support_margin;
    // False is useful for a post-restart diagnostic measurement that must not
    // consume the step's tumble decision.
    bool apply_tumble;
    // Surface needs the already-computed gradient but adds a square root per
    // voxel; evaluate it only on the configured full-moment/output cadence.
    bool compute_surface;
};

struct UpdateArgs3D {
    // Kept mutable so the exact in-place launcher can copy its completed
    // scratch image back after all reads for that cell have finished.
    float* phi_in;
    float* phi_out;
    const std::uint32_t* S_in;
    // Optional.  When null the update performs no scatter; callers can clear
    // and reconstruct a single S buffer from phi_out afterwards.
    std::uint32_t* S_out;
    WeightedWallField3D wall;
    CellState3D* cells;
    std::uint32_t* global_flags;
    unsigned long long* work_cursor;
    SLayout3D layout;
    int N;
    int B;
    float dt;
    double V0;
    double volume_scale;
    float bulk_scale;
    float interaction_coeff;
    // Exact destination surface measurement is an additional tiled pass.
    // Normal stepping leaves this false; output paths measure the current
    // field explicitly through MeasureArgs3D::compute_surface.
    bool compute_surface;
};

struct TrajPackedCell3D {
    std::int64_t global_id;
    double x, y, z;
    double volume;
    double surface;
    double wall_overlap;
    double outside_slit_volume;
    float velocity_x, velocity_y, velocity_z;
    float polarity_x, polarity_y, polarity_z;
    float gamma, v_A;
    float phi_max;
};

struct VerifyCell3D {
    double measured_V;
    double state_V;
    float max_abs_phi;
    std::uint32_t nonfinite_count;
    std::uint32_t support_edge_count;
    std::uint32_t verified;
};

// Stable raw kernels used by direct validation tests. Hot wall-coupled kernels
// are internally specialized; normal code uses the checked launch helpers.
__global__ void k_initialize_spheres(InitArgs3D args);
__global__ void k_clear_u32(std::uint32_t* data, std::size_t words);
__global__ void k_scatter_current(ScatterArgs3D args);
__global__ void k_scatter_promoted(ScatterArgs3D args,
                                   float* const* promoted_phi,
                                   const int* promoted_ids,
                                   int promoted_count, int promoted_edge);
// Runs after every measurement launch.  Keeping stochastic state changes in a
// separate, globally guarded kernel prevents a failed measurement from
// consuming a tumble at a step that was not accepted.
__global__ void k_apply_cell_motion(MeasureArgs3D args);
__global__ void k_finalize_tiled_sharded(
    UpdateArgs3D args, const MomentPartial3D* partials, int shards_per_cell);
// Normally returns after one CTA-level flag check.  On an update-discovered
// fatal it restores a coherent output: throughput mode copies the retained
// source field, while in-place mode reconstructs S_out from committed cells.
__global__ void k_repair_after_fatal(UpdateArgs3D args);
__global__ void k_repair_promoted_after_fatal(
    UpdateArgs3D args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge);
__global__ void k_finalize_origins(CellState3D* cells, int N,
                                    const std::uint32_t* global_flags);
// Restore the complete accepted cell state after a rejected update. Support
// requests are recorded separately so adaptive recovery does not depend on
// retaining mutations made by the failed measurement.
__global__ void k_restore_cells_after_fatal(
    CellState3D* cells, const CellState3D* accepted_cells, int N,
    const std::uint32_t* global_flags, std::uint32_t* support_requests);
// Increment only when every global integrity counter is zero.
__global__ void k_advance_step(std::uint64_t* step,
                               const std::uint32_t* global_flags);
__global__ void k_pack_trajectory(const CellState3D* cells,
                                  TrajPackedCell3D* out,
                                  int N, SLayout3D layout,
                                  int channel_padding);
__global__ __launch_bounds__(kThreads3D, 1)
void k_measure_wall_diagnostics(const float* phi,
                            float* const* promoted_phi,
                            CellState3D* cells, int N, int B,
                            SLayout3D layout, const float* wall_psi_sq,
                            int channel_height, int channel_padding);
__global__ __launch_bounds__(kThreads3D, 1)
void k_verify_cells(const float* phi, const CellState3D* cells,
                    VerifyCell3D* out, int N, int B, SLayout3D layout);
__global__ __launch_bounds__(kThreads3D, 1)
void k_verify_promoted(float* const* promoted_phi,
                       const CellState3D* cells, VerifyCell3D* out,
                       const int* promoted_ids, int N, int promoted_count,
                       int promoted_edge, SLayout3D layout);
__global__ void k_verify_S(const std::uint32_t* S, std::size_t words,
                           std::uint32_t* out_max);

// Launch helpers validate the runtime brick edge before enqueueing work.
bool valid_runtime_geometry(int B, const SLayout3D& layout);
void launch_initialize_spheres(const InitArgs3D& args, cudaStream_t stream = 0);
void launch_clear_S(std::uint32_t* S, const SLayout3D& layout,
                    cudaStream_t stream = 0);
void launch_scatter_current(const ScatterArgs3D& args,
                            cudaStream_t stream = 0);
void launch_scatter_promoted(const ScatterArgs3D& args,
                             float* const* promoted_phi,
                             const int* promoted_ids, int promoted_count,
                             int promoted_edge, cudaStream_t stream = 0);
[[nodiscard]] bool launch_measure_cells(const MeasureArgs3D& args,
                                        cudaStream_t stream = 0);
// Split form used when additional support tiers must be measured before the
// single global motion/tumble launch.
[[nodiscard]] bool launch_measure_cells_only(const MeasureArgs3D& args,
                                             cudaStream_t stream = 0);
[[nodiscard]] bool launch_measure_promoted(
    const MeasureArgs3D& args, float* const* promoted_phi,
    const int* promoted_ids, int promoted_count, int promoted_edge,
    cudaStream_t stream = 0);
void launch_apply_cell_motion(const MeasureArgs3D& args,
                              cudaStream_t stream = 0);
[[nodiscard]] bool launch_update_tiled(const UpdateArgs3D& args,
                                       int grid_blocks,
                                       cudaStream_t stream = 0);
// Spatially sharded throughput path. Source tiles have disjoint destinations
// after recentering; Q5.27 additions remain exact and order-independent.
[[nodiscard]] bool launch_update_tiled_sharded(
    const UpdateArgs3D& args, MomentPartial3D* partials,
    int shards_per_cell, cudaStream_t stream = 0);
// scratch contains scratch_slots * B^3 floats.  The launcher uses exactly one
// persistent block per slot; each block owns its slot for the whole kernel.
[[nodiscard]] bool launch_update_inplace(const UpdateArgs3D& args,
                                         float* scratch, int scratch_slots,
                                         cudaStream_t stream = 0);
[[nodiscard]] bool launch_update_promoted(
    const UpdateArgs3D& args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, bool collect_moments,
    cudaStream_t stream = 0);
void launch_repair_after_fatal(const UpdateArgs3D& args,
                               cudaStream_t stream = 0);
void launch_repair_promoted_after_fatal(
    const UpdateArgs3D& args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, cudaStream_t stream = 0);
void launch_finalize_origins(CellState3D* cells, int N,
                             std::uint32_t* global_flags,
                             cudaStream_t stream = 0);
void launch_restore_cells_after_fatal(
    CellState3D* cells, const CellState3D* accepted_cells, int N,
    const std::uint32_t* global_flags, std::uint32_t* support_requests,
    cudaStream_t stream = 0);
void launch_advance_step(std::uint64_t* step,
                         const std::uint32_t* global_flags,
                         cudaStream_t stream = 0);
void launch_pack_trajectory(const CellState3D* cells, TrajPackedCell3D* out,
                            int N, const SLayout3D& layout,
                            cudaStream_t stream = 0,
                            int channel_padding = 0);
void launch_measure_wall_diagnostics(const float* phi,
                                 float* const* promoted_phi,
                                 CellState3D* cells, int N, int B,
                                 const SLayout3D& layout,
                                 const float* wall_psi_sq,
                                 int channel_height, int channel_padding,
                                 cudaStream_t stream = 0);
void launch_verify_cells(const float* phi, const CellState3D* cells,
                         VerifyCell3D* out, int N, int B,
                         const SLayout3D& layout, cudaStream_t stream = 0);
void launch_verify_promoted(float* const* promoted_phi,
                            const CellState3D* cells, VerifyCell3D* out,
                            const int* promoted_ids, int N, int promoted_count,
                            int promoted_edge, const SLayout3D& layout,
                            cudaStream_t stream = 0);
void launch_verify_S(const std::uint32_t* S, const SLayout3D& layout,
                     std::uint32_t* out_max, cudaStream_t stream = 0);

namespace detail {

// Normal stepping can defer derived moments until the next measurement.
// These launchers select kernels that retain the phase update and integrity
// behavior while compiling out post-update moment accumulation.
[[nodiscard]] bool launch_update_tiled_fast(
    const UpdateArgs3D& args, int grid_blocks, cudaStream_t stream = 0);
[[nodiscard]] bool launch_update_tiled_sharded_fast(
    const UpdateArgs3D& args, int shards_per_cell, cudaStream_t stream = 0);
[[nodiscard]] bool launch_update_inplace_fast(
    const UpdateArgs3D& args, float* scratch, int scratch_slots,
    cudaStream_t stream = 0);
// Sharded promoted throughput update: splits every pointer-backed cube among
// shards_per_cell CTAs with the base sharded update's deterministic tile
// assignment; outputs are bit-identical to launch_update_promoted's fast
// path.  Clears the destination cubes itself.
[[nodiscard]] bool launch_update_promoted_sharded_fast(
    const UpdateArgs3D& args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, int shards_per_cell,
    cudaStream_t stream = 0);
cudaError_t promoted_fast_update_occupancy(int* blocks_per_sm,
                                           const SLayout3D& layout);
// Sharded promoted measurement with an ascending-shard deterministic
// finalize.  Its reduction grouping differs from the one-CTA promoted
// measurement: runs remain deterministic and restart-exact, but are not
// bitwise comparable with the one-CTA fold.
[[nodiscard]] bool launch_measure_promoted_shards(
    const MeasureArgs3D& args, float* const* promoted_phi,
    const int* promoted_ids, int promoted_count, int promoted_edge,
    MomentPartial3D* promoted_partials, cudaStream_t stream = 0);
cudaError_t promoted_measure_occupancy(int* blocks_per_sm,
                                       const SLayout3D& layout);
// Fully periodic y-strip/rolling-plane update. It is bit-identical to
// launch_update_tiled_fast, clears phi_out as it writes, and sizes its shared
// memory from the brick edge.
[[nodiscard]] bool launch_update_periodic_slab_walk_fast(
    const UpdateArgs3D& args, cudaStream_t stream = 0);
std::size_t update_periodic_slab_walk_shared_bytes(int B);
cudaError_t measured_update_occupancy(int* tiled_blocks_per_sm,
                                      int* inplace_blocks_per_sm,
                                      const SLayout3D& layout);
cudaError_t fast_update_occupancy(int* tiled_blocks_per_sm,
                                  int* inplace_blocks_per_sm,
                                  const SLayout3D& layout);
cudaError_t sharded_occupancy(int* measurement_blocks_per_sm,
                              int* measured_update_blocks_per_sm,
                              int* fast_update_blocks_per_sm,
                              const SLayout3D& layout);

}  // namespace detail

}  // namespace pf3d
