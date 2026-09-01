#pragma once

// On-disk contract for the three-dimensional solver. All integer and IEEE-754
// fields are encoded little-endian. Records are packed so their byte offsets
// do not depend on the host ABI. The current implementation rejects big-endian
// hosts rather than attempting an implicit conversion.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ckpt3d {

constexpr std::uint32_t kMagic = 0x44334650u;  // bytes "PF3D"
constexpr std::uint16_t kCheckpointFormat = 1;  // sole supported schema
constexpr std::uint32_t kEndianMarker = 0x01020304u;
constexpr std::uint32_t kDimensions = 3;
constexpr std::uint32_t kScalarFloat32 = 1;
constexpr std::uint32_t kPhaseOrderXFastest = 1;
constexpr std::uint32_t kHeaderFlagPayloadCrc64Ecma = 1u << 0;
constexpr std::uint32_t kHeaderFlagVariableCellStorage = 1u << 1;
constexpr std::uint32_t kRequiredHeaderFlags =
    kHeaderFlagPayloadCrc64Ecma | kHeaderFlagVariableCellStorage;
constexpr std::uint32_t kBoundaryPeriodicX = 1u << 0;
constexpr std::uint32_t kBoundaryPeriodicY = 1u << 1;
constexpr std::uint32_t kBoundaryPeriodicZ = 1u << 2;
// Distinguishes a resolved two-wall channel from a substrate slab.
constexpr std::uint32_t kBoundaryChannelZ = 1u << 3;
constexpr std::uint32_t kBoundaryPeriodicXY =
    kBoundaryPeriodicX | kBoundaryPeriodicY;
constexpr std::uint32_t kBoundarySubstrateSlab = kBoundaryPeriodicXY;
constexpr std::uint32_t kBoundaryPeriodicXYZ =
    kBoundaryPeriodicX | kBoundaryPeriodicY | kBoundaryPeriodicZ;
constexpr std::uint32_t kBoundaryHardWallChannel =
    kBoundaryPeriodicXY | kBoundaryChannelZ;

constexpr bool valid_domain_geometry(std::uint32_t boundary_flags,
                                     std::int64_t nx, std::int64_t ny,
                                     std::int64_t nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return false;
    return (boundary_flags == kBoundaryPeriodicXYZ &&
            nx == ny && nx == nz) ||
           ((boundary_flags == kBoundarySubstrateSlab ||
             boundary_flags == kBoundaryHardWallChannel) && nx == ny);
}

constexpr std::int64_t storage_domain_edge(std::uint32_t boundary_flags,
                                           std::int64_t nx,
                                           std::int64_t ny,
                                           std::int64_t nz) {
    const std::int64_t xy = nx < ny ? nx : ny;
    return boundary_flags == kBoundaryHardWallChannel
        ? xy : (xy < nz ? xy : nz);
}

constexpr std::uint32_t kIntegratorExplicitEuler = 1;
constexpr std::uint32_t kStencilUnspecified = 0;
constexpr std::uint32_t kStencilSevenPoint = 1;
constexpr std::uint32_t kStencilTwentySevenPoint = 2;
constexpr std::uint32_t kRngPhilox4x32_10 = 1;

// Promoted-cell measurements may use more than one CTA per cell. The shard
// count changes the floating-point reduction grouping and is therefore part
// of the continuation contract. A checkpoint records either a fixed count or
// the occupancy wave selected for the automatic policy.
constexpr std::uint64_t kPromotedMeasurePolicyAuto = UINT64_MAX;
constexpr std::uint64_t kMaximumPromotedMeasureShards = 64;

constexpr bool valid_promoted_measure_contract(
    std::uint64_t policy, std::uint64_t auto_wave_ctas) {
    if (policy == kPromotedMeasurePolicyAuto)
        return auto_wave_ctas > 0 &&
               auto_wave_ctas <= UINT64_C(2147483647);
    return policy <= kMaximumPromotedMeasureShards && auto_wave_ctas == 0;
}

// CRC-64/ECMA-182: poly=0x42f0e1eba9ea3693, init=0, refin=false,
// refout=false, xorout=0. The checksum covers the complete file in storage
// order, with FileHeader3D::file_crc64 treated as zero while calculating it.
constexpr std::uint64_t kCrc64EcmaPolynomial =
    UINT64_C(0x42f0e1eba9ea3693);

inline const std::array<std::uint64_t, 256>& crc64_ecma_table() {
    static const std::array<std::uint64_t, 256> table = [] {
        std::array<std::uint64_t, 256> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::uint64_t value = static_cast<std::uint64_t>(i) << 56;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value & UINT64_C(0x8000000000000000)) != 0
                    ? (value << 1) ^ kCrc64EcmaPolynomial
                    : value << 1;
            }
            values[i] = value;
        }
        return values;
    }();
    return table;
}

inline std::uint64_t crc64_ecma_update(std::uint64_t crc, const void* data,
                                       std::size_t bytes) {
    const auto* input = static_cast<const std::uint8_t*>(data);
    const auto& table = crc64_ecma_table();
    for (std::size_t i = 0; i < bytes; ++i) {
        const std::uint8_t index = static_cast<std::uint8_t>(
            (crc >> 56) ^ static_cast<std::uint64_t>(input[i]));
        crc = table[index] ^ (crc << 8);
    }
    return crc;
}

inline std::uint64_t crc64_ecma(const void* data, std::size_t bytes) {
    return crc64_ecma_update(0, data, bytes);
}

// This is a corruption guard, not a supported simulation limit. Keeping B
// bounded also makes B^3 fit safely in uint64_t during input validation.
constexpr std::uint32_t kMaxBrickEdge = 4096;
constexpr std::uint32_t kStorageEdgeAlignment = 8;

constexpr bool valid_brick_edge(std::uint32_t edge) {
    return edge >= 3 && edge <= kMaxBrickEdge;
}

constexpr std::uint64_t phase_values_per_cell(std::uint32_t edge) {
    return valid_brick_edge(edge)
        ? static_cast<std::uint64_t>(edge) * edge * edge
        : 0;
}

constexpr bool valid_storage_edge(std::uint32_t edge,
                                  std::uint32_t base_edge,
                                  std::uint64_t domain_edge) {
    return valid_brick_edge(edge) && edge >= base_edge &&
           edge < domain_edge && edge % kStorageEdgeAlignment == 0;
}

#pragma pack(push, 1)

// File layout:
//   FileHeader3D
//   ParamsRecord3D
//   num_cells x { CellRecord3D, storage_edge^3 binary32 phase values }
// Each cell record stores its phase-cube edge. The parameter record includes
// the promoted-measurement reduction contract and resolved-wall parameters.
// Slot order and global IDs are preserved. Header phase counts still describe
// base B^3, and each cube is x-fastest:
// x + storage_edge*(y + storage_edge*z). The aggregate field is reconstructed.
struct FileHeader3D {
    std::uint32_t magic;                 // kMagic
    std::uint16_t version;               // kCheckpointFormat
    std::uint16_t header_bytes;          // sizeof(FileHeader3D)
    std::uint32_t endian_marker;         // kEndianMarker
    std::uint32_t dimensions;            // exactly 3
    std::uint32_t scalar_format;         // kScalarFloat32
    std::uint32_t flags;                 // kRequiredHeaderFlags
    std::int64_t step;
    double time;
    std::uint64_t num_cells;
    std::uint32_t params_bytes;          // sizeof(ParamsRecord3D)
    std::uint32_t cell_record_bytes;     // sizeof(CellRecord3D)
    std::uint32_t brick_edge;            // B, including the stencil guard
    std::uint32_t phase_order;           // kPhaseOrderXFastest
    std::uint64_t phase_values_per_cell; // base B^3
    std::uint64_t phase_bytes_per_cell;  // 4*base B^3
    // CRC-64/ECMA-182 over the canonical header (with file_crc64 zeroed),
    // parameters, cell records, and phase bricks in file order.
    std::uint64_t file_crc64;
    // Resolved base-measurement CTAs per cell. Writers record an exact count
    // in 1..64; readers interpret zero as the standard policy.
    std::uint64_t base_measure_shards;
};

// Parameter values required to interpret and continue the trajectory. Per-cell
// gamma, active speed, and target radius are repeated in CellRecord3D so a
// heterogeneous realization survives restart without sidecars.
struct ParamsRecord3D {
    std::int64_t Nx, Ny, Nz;
    double dx, dy, dz;
    std::uint32_t brick_edge;            // B; must match FileHeader3D
    // Supported geometries are the periodic XYZ cube and the periodic-XY
    // substrate slab or hard-wall channel. Bounded Nz is independent of Nx=Ny.
    std::uint32_t boundary_flags;

    double dt;
    double t_end;
    double lambda;
    double gamma_normal;
    double gamma_soft;
    double soft_fraction;
    double kappa;
    double target_radius;
    double mu;
    double v_A;
    double v_A_sigma;
    double xi;
    double tau;
    double volume_fraction;              // rho_V, except rho_A for a slab

    std::uint64_t seed;
    std::uint64_t polarity_seed;
    std::int64_t print_interval;
    std::int64_t full_moment_every;
    std::int64_t verify_every;
    std::uint32_t integrator_id;
    std::uint32_t stencil_id;
    std::uint32_t rng_id;
    // Resolved absolute-step trajectory cadence; zero means no stored cadence.
    std::uint32_t trajectory_interval;
    // Hash of the accepted initial centre table. Zero means that no identity
    // was supplied (for example, in a synthetic format test).
    std::uint64_t initialization_hash;
    // Duration of the initial passive-aging stage. It is independent of the
    // current continuation target and may therefore exceed t_end.
    double aging_time;
    // Continuation contract for promoted-cell measurement.
    // policy: 0=one CTA, 1..64=fixed CTAs/cell, UINT64_MAX=automatic.
    // The automatic policy stores its originating occupancy wave so the
    // grouping remains invariant when a checkpoint moves between devices.
    std::uint64_t promoted_measure_policy;
    std::uint64_t promoted_measure_auto_wave_ctas;
    // Resolved hard-wall channel. H is the accessible separation
    // between wall mid-surfaces; Nz=H+2*padding is the allocated height.
    std::int64_t channel_height;
    std::int64_t channel_padding;
    double wall_kappa;
    double wall_width;
};

// Future-relevant state is serialized field by field, independent of CUDA ABI.
// Moments and bounds support exact restart checks. Origins remain unwrapped so
// whole-box trajectory displacement is not lost.
struct CellRecord3D {
    std::int64_t global_id;
    std::int64_t origin_x;
    std::int64_t origin_y;
    std::int64_t origin_z;

    float polarity_x;
    float polarity_y;
    float polarity_z;
    float velocity_x;
    float velocity_y;
    float velocity_z;
    float gamma;
    float v_A;
    float target_radius;
    float phi_max;

    double volume;
    double moment_x;
    double moment_y;
    double moment_z;
    double surface;
    double interaction_x;
    double interaction_y;
    double interaction_z;

    std::uint32_t shift_counter;
    std::uint32_t tumble_counter;
    std::uint32_t state_flags;

    std::int32_t bbox_lo_x;
    std::int32_t bbox_hi_x;
    std::int32_t bbox_lo_y;
    std::int32_t bbox_hi_y;
    std::int32_t bbox_lo_z;
    std::int32_t bbox_hi_z;

    // A checkpoint-time measurement may schedule recentering for the next
    // update. Store that measured state explicitly so restart remains complete.
    std::int32_t pending_shift_x;
    std::int32_t pending_shift_y;
    std::int32_t pending_shift_z;
    // Aligned edge of this cell's serialized phase cube.
    std::uint32_t storage_edge;
    std::uint32_t reserved32;
    std::uint64_t reserved64[8];          // write as zero
};

#pragma pack(pop)

constexpr double kPolarityNormTolerance = 2.0e-5;

// Geometry-dependent cell state carried by a checkpoint. Substrate motion is
// strictly planar; periodic-XYZ and channel polarity are three-dimensional.
inline bool cell_motion_matches_geometry(const CellRecord3D& cell,
                                         std::uint32_t boundary_flags) {
    if (!std::isfinite(cell.velocity_x) ||
        !std::isfinite(cell.velocity_y) ||
        !std::isfinite(cell.velocity_z)) {
        return false;
    }
    if (boundary_flags == kBoundaryPeriodicXYZ ||
        boundary_flags == kBoundaryHardWallChannel) {
        const double norm2 =
            static_cast<double>(cell.polarity_x) * cell.polarity_x +
            static_cast<double>(cell.polarity_y) * cell.polarity_y +
            static_cast<double>(cell.polarity_z) * cell.polarity_z;
        return std::isfinite(norm2) &&
               std::fabs(norm2 - 1.0) <= kPolarityNormTolerance;
    }
    if (boundary_flags == kBoundarySubstrateSlab) {
        if (cell.polarity_z != 0.0f || cell.velocity_z != 0.0f) return false;
        const double norm2 =
            static_cast<double>(cell.polarity_x) * cell.polarity_x +
            static_cast<double>(cell.polarity_y) * cell.polarity_y;
        return std::isfinite(norm2) &&
               std::fabs(norm2 - 1.0) <= kPolarityNormTolerance;
    }
    return false;
}

// Avoid signed overflow while requiring bounded-z support within 0 <= z < Nz.
inline bool cell_support_respects_z_boundaries(
    const CellRecord3D& cell, std::uint32_t boundary_flags, std::int64_t nz) {
    if (boundary_flags == kBoundaryPeriodicXYZ) return true;
    if ((boundary_flags != kBoundarySubstrateSlab &&
         boundary_flags != kBoundaryHardWallChannel) || nz <= 0 ||
        cell.bbox_lo_z < 0 || cell.bbox_lo_z > cell.bbox_hi_z) {
        return false;
    }
    return cell.origin_z >= -static_cast<std::int64_t>(cell.bbox_lo_z) &&
           cell.origin_z < nz - static_cast<std::int64_t>(cell.bbox_hi_z);
}

static_assert(sizeof(FileHeader3D) == 96, "3D checkpoint header layout drift");
static_assert(sizeof(ParamsRecord3D) == 288, "3D parameter layout drift");
static_assert(sizeof(CellRecord3D) == 256, "3D cell-record layout drift");

static_assert(offsetof(FileHeader3D, step) == 24, "3D step offset drift");
static_assert(offsetof(FileHeader3D, num_cells) == 40,
              "3D cell-count offset drift");
static_assert(offsetof(FileHeader3D, brick_edge) == 56,
              "3D brick-edge offset drift");
static_assert(offsetof(FileHeader3D, phase_values_per_cell) == 64,
              "3D phase-count offset drift");
static_assert(offsetof(FileHeader3D, file_crc64) == 80,
              "3D file-checksum offset drift");
static_assert(offsetof(FileHeader3D, base_measure_shards) == 88,
              "3D base-measurement shard offset drift");

static_assert(offsetof(ParamsRecord3D, Nx) == 0, "3D Nx offset drift");
static_assert(offsetof(ParamsRecord3D, dx) == 24, "3D dx offset drift");
static_assert(offsetof(ParamsRecord3D, brick_edge) == 48,
              "3D parameter B offset drift");
static_assert(offsetof(ParamsRecord3D, dt) == 56, "3D dt offset drift");
static_assert(offsetof(ParamsRecord3D, seed) == 168, "3D seed offset drift");
static_assert(offsetof(ParamsRecord3D, polarity_seed) == 176,
              "3D polarity-seed offset drift");
static_assert(offsetof(ParamsRecord3D, integrator_id) == 208,
              "3D integrator offset drift");
static_assert(offsetof(ParamsRecord3D, trajectory_interval) == 220,
              "3D trajectory-interval offset drift");
static_assert(offsetof(ParamsRecord3D, initialization_hash) == 224,
              "3D initialization-hash offset drift");
static_assert(offsetof(ParamsRecord3D, aging_time) == 232,
              "3D aging-time offset drift");
static_assert(offsetof(ParamsRecord3D, promoted_measure_policy) == 240,
              "3D reduction-policy offset drift");
static_assert(offsetof(ParamsRecord3D,
                       promoted_measure_auto_wave_ctas) == 248,
              "3D reduction-wave offset drift");
static_assert(offsetof(ParamsRecord3D, channel_height) == 256,
              "3D channel-height offset drift");
static_assert(offsetof(ParamsRecord3D, wall_kappa) == 272,
              "3D wall-kappa offset drift");

static_assert(offsetof(CellRecord3D, origin_z) == 24,
              "3D origin-z offset drift");
static_assert(offsetof(CellRecord3D, polarity_x) == 32,
              "3D polarity offset drift");
static_assert(offsetof(CellRecord3D, velocity_x) == 44,
              "3D velocity offset drift");
static_assert(offsetof(CellRecord3D, volume) == 72,
              "3D volume offset drift");
static_assert(offsetof(CellRecord3D, shift_counter) == 136,
              "3D shift-counter offset drift");
static_assert(offsetof(CellRecord3D, state_flags) == 144,
              "3D state-flags offset drift");
static_assert(offsetof(CellRecord3D, bbox_lo_x) == 148,
              "3D bounding-box offset drift");
static_assert(offsetof(CellRecord3D, pending_shift_x) == 172,
              "3D pending-shift offset drift");
static_assert(offsetof(CellRecord3D, storage_edge) == 184,
              "3D storage-edge offset drift");
static_assert(offsetof(CellRecord3D, reserved32) == 188,
              "3D reserved32 offset drift");
static_assert(offsetof(CellRecord3D, reserved64) == 192,
              "3D reserved64 offset drift");

inline bool host_is_little_endian() {
    const std::uint32_t value = 1;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

}  // namespace ckpt3d
