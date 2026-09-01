#include "../../include/pf3d/checkpoint.cuh"
#include "../../include/pf3d/measure_shards.hpp"
#include "../../common/checkpoint_format_3d.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pf3d {
namespace {

constexpr std::size_t kStageTargetBytes = 64u * 1024u * 1024u;
constexpr std::size_t kCellStageTargetBytes = 1u * 1024u * 1024u;
static_assert(ckpt3d::kStorageEdgeAlignment == kBrickAlignment,
              "checkpoint and solver storage alignment must agree");
static_assert(ckpt3d::kBoundaryPeriodicX == kBoundaryPeriodicX3D &&
              ckpt3d::kBoundaryPeriodicY == kBoundaryPeriodicY3D &&
               ckpt3d::kBoundaryPeriodicZ == kBoundaryPeriodicZ3D &&
               ckpt3d::kBoundaryPeriodicXYZ == kBoundaryPeriodicXYZ3D &&
               ckpt3d::kBoundarySubstrateSlab == kBoundarySubstrateSlab3D &&
               ckpt3d::kBoundaryChannelZ == kBoundaryChannelZ3D &&
               ckpt3d::kBoundaryHardWallChannel == kBoundaryHardWallChannel3D,
              "checkpoint and solver boundary flags must agree");

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[ckpt3d] %s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool read_exact(std::FILE* file, void* dst, std::size_t bytes,
                const char* description) {
    if (bytes == 0) return true;
    if (std::fread(dst, 1, bytes, file) == bytes) return true;
    std::fprintf(stderr, "[ckpt3d] short read of %s (%zu bytes)\n",
                 description, bytes);
    return false;
}

bool write_exact(std::FILE* file, const void* src, std::size_t bytes,
                 const char* description) {
    if (bytes == 0) return true;
    if (std::fwrite(src, 1, bytes, file) == bytes) return true;
    std::fprintf(stderr, "[ckpt3d] short write of %s (%zu bytes): %s\n",
                 description, bytes, std::strerror(errno));
    return false;
}

bool checked_add(std::uintmax_t a, std::uintmax_t b, std::uintmax_t* out) {
    if (!out || b > std::numeric_limits<std::uintmax_t>::max() - a)
        return false;
    *out = a + b;
    return true;
}

bool checked_multiply(std::uintmax_t a, std::uintmax_t b,
                      std::uintmax_t* out) {
    if (!out || (a != 0 && b > std::numeric_limits<std::uintmax_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

bool open_file_size(std::FILE* file, std::uintmax_t* out) {
    if (!file || !out) return false;
#if defined(_WIN32)
    struct _stat64 status{};
    if (_fstat64(_fileno(file), &status) != 0 || status.st_size < 0)
        return false;
#else
    struct stat status{};
    if (::fstat(::fileno(file), &status) != 0 || status.st_size < 0)
        return false;
#endif
    *out = static_cast<std::uintmax_t>(status.st_size);
    return true;
}

bool seek_forward(std::FILE* file, std::uint64_t bytes) {
    if (!file || bytes > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max())) {
        return false;
    }
#if defined(_WIN32)
    return _fseeki64(file, static_cast<__int64>(bytes), SEEK_CUR) == 0;
#else
    if (bytes > static_cast<std::uint64_t>(
                    std::numeric_limits<off_t>::max())) {
        return false;
    }
    return ::fseeko(file, static_cast<off_t>(bytes), SEEK_CUR) == 0;
#endif
}

bool header_valid(const ckpt3d::FileHeader3D& h, const std::string& path) {
    auto reject = [&](const char* reason) {
        std::fprintf(stderr, "[ckpt3d] %s: %s\n", path.c_str(), reason);
        return false;
    };
    if (!ckpt3d::host_is_little_endian())
        return reject("big-endian hosts are not supported");
    if (h.magic != ckpt3d::kMagic)
        return reject("not a PF3D checkpoint (dimension/magic mismatch)");
    if (h.version != ckpt3d::kCheckpointFormat)
        return reject("unsupported PF3D checkpoint format");
    if (h.header_bytes != sizeof(ckpt3d::FileHeader3D) ||
        h.params_bytes != sizeof(ckpt3d::ParamsRecord3D) ||
        h.cell_record_bytes != sizeof(ckpt3d::CellRecord3D))
        return reject("record-size fields do not match the PF3D contract");
    if (h.endian_marker != ckpt3d::kEndianMarker ||
        h.dimensions != ckpt3d::kDimensions ||
        h.scalar_format != ckpt3d::kScalarFloat32 ||
        h.phase_order != ckpt3d::kPhaseOrderXFastest)
        return reject("incompatible scalar, dimension, endian, or phase order");
    if (h.flags != ckpt3d::kRequiredHeaderFlags)
        return reject("missing required checksum flag");
    if (!valid_stored_base_measure_shards(h.base_measure_shards))
        return reject("base measurement shard count is outside 0..64");
    if (h.step < 0 || !std::isfinite(h.time) || h.time < 0.0)
        return reject("negative step/time or non-finite time");
    if (h.num_cells == 0 ||
        h.num_cells > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        return reject("cell count is outside the supported range");
    if (!ckpt3d::valid_brick_edge(h.brick_edge) ||
        h.brick_edge % static_cast<std::uint32_t>(kBrickAlignment) != 0)
        return reject("brick edge is outside the corruption guard");
    const std::uint64_t values = ckpt3d::phase_values_per_cell(h.brick_edge);
    if (h.phase_values_per_cell != values ||
        values > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        h.phase_bytes_per_cell != values * sizeof(float))
        return reject("phase payload size is inconsistent with B^3 float32");
    return true;
}

void params_from_record(const ckpt3d::ParamsRecord3D& in,
                        std::uint64_t num_cells,
                        SimParams3D* out) {
    *out = SimParams3D{};
    out->Nx = static_cast<int>(in.Nx);
    out->Ny = static_cast<int>(in.Ny);
    out->Nz = static_cast<int>(in.Nz);
    out->dx = in.dx;
    out->dy = in.dy;
    out->dz = in.dz;
    out->boundary_flags = in.boundary_flags;
    out->num_cells = static_cast<int>(num_cells);
    out->rho = in.volume_fraction;
    out->dt = in.dt;
    out->t_end = in.t_end;
    out->lambda = in.lambda;
    out->gamma_normal = in.gamma_normal;
    out->gamma_cancer = in.gamma_soft;
    out->cancer_fraction = in.soft_fraction;
    out->kappa = in.kappa;
    out->target_radius = in.target_radius;
    out->mu = in.mu;
    out->v_A = in.v_A;
    out->v_A_sigma = in.v_A_sigma;
    out->xi = in.xi;
    out->tau = in.tau;
    out->seed = in.seed;
    out->polarity_seed = in.polarity_seed;
    out->initialization_hash = in.initialization_hash;
    out->aging_time = in.aging_time;
    out->full_moment_every = static_cast<int>(in.full_moment_every);
    out->verify_every = static_cast<int>(in.verify_every);
    out->channel_height = static_cast<int>(in.channel_height);
    out->channel_padding = static_cast<int>(in.channel_padding);
    out->wall_kappa = in.wall_kappa;
    out->wall_width = in.wall_width;
}

void params_to_record(const CheckpointWriteView3D& view,
                      ckpt3d::ParamsRecord3D* out) {
    const SimParams3D& p = *view.params;
    std::memset(out, 0, sizeof(*out));
    out->Nx = p.Nx;
    out->Ny = p.Ny;
    out->Nz = p.Nz;
    out->dx = p.dx;
    out->dy = p.dy;
    out->dz = p.dz;
    out->brick_edge = static_cast<std::uint32_t>(view.brick_edge);
    out->boundary_flags = p.boundary_flags;
    out->dt = p.dt;
    out->t_end = p.t_end;
    out->lambda = p.lambda;
    out->gamma_normal = p.gamma_normal;
    out->gamma_soft = p.gamma_cancer;
    out->soft_fraction = p.cancer_fraction;
    out->kappa = p.kappa;
    out->target_radius = p.target_radius;
    out->mu = p.mu;
    out->v_A = p.v_A;
    out->v_A_sigma = p.v_A_sigma;
    out->xi = p.xi;
    out->tau = p.tau;
    out->volume_fraction = p.rho;
    out->seed = p.seed;
    out->polarity_seed = p.polarity_stream();
    out->initialization_hash = p.initialization_hash;
    out->aging_time = p.aging_time;
    out->print_interval = view.print_interval;
    out->full_moment_every = p.full_moment_every;
    out->verify_every = p.verify_every;
    out->integrator_id = ckpt3d::kIntegratorExplicitEuler;
    out->stencil_id = ckpt3d::kStencilTwentySevenPoint;
    out->rng_id = ckpt3d::kRngPhilox4x32_10;
    out->trajectory_interval = view.trajectory_interval;
    out->promoted_measure_policy = encode_promoted_measure_policy(
        view.promoted_measure_reduction.policy);
    out->promoted_measure_auto_wave_ctas = static_cast<std::uint64_t>(
        view.promoted_measure_reduction.auto_wave_ctas);
    out->channel_height = p.channel_height;
    out->channel_padding = p.channel_padding;
    out->wall_kappa = p.wall_kappa;
    out->wall_width = p.wall_width;
}

bool params_record_valid(const ckpt3d::ParamsRecord3D& p,
                          const ckpt3d::FileHeader3D& h,
                          const std::string& path) {
    auto reject = [&](const char* reason) {
        std::fprintf(stderr, "[ckpt3d] %s: %s\n", path.c_str(), reason);
        return false;
    };
    constexpr std::int64_t kIntMax = std::numeric_limits<int>::max();
    if (p.Nx <= 0 || p.Ny <= 0 || p.Nz <= 0 ||
        p.Nx > kIntMax || p.Ny > kIntMax || p.Nz > kIntMax)
        return reject("checkpoint domain dimensions are outside the supported range");
    if (!ckpt3d::valid_domain_geometry(
            p.boundary_flags, p.Nx, p.Ny, p.Nz))
        return reject("checkpoint boundary flags and dimensions do not describe a supported geometry");
    const double scalar_parameters[] = {
        p.dx, p.dy, p.dz, p.dt, p.t_end, p.lambda, p.gamma_normal,
        p.gamma_soft, p.soft_fraction, p.kappa, p.target_radius, p.mu,
        p.v_A, p.v_A_sigma, p.xi, p.tau, p.volume_fraction, p.aging_time};
    for (double value : scalar_parameters)
        if (!std::isfinite(value))
            return reject("checkpoint contains a non-finite parameter");
    const std::int64_t minimum_domain_edge = ckpt3d::storage_domain_edge(
        p.boundary_flags, p.Nx, p.Ny, p.Nz);
    if (p.brick_edge != h.brick_edge ||
        static_cast<std::int64_t>(p.brick_edge) >= minimum_domain_edge)
        return reject("parameter and header brick edges disagree");
    if (p.integrator_id != ckpt3d::kIntegratorExplicitEuler ||
        p.stencil_id != ckpt3d::kStencilTwentySevenPoint ||
        p.rng_id != ckpt3d::kRngPhilox4x32_10)
        return reject("integrator, stencil, or RNG contract is incompatible");
    if (p.print_interval < 0 || p.print_interval > kIntMax ||
        p.full_moment_every < 0 || p.full_moment_every > kIntMax ||
        p.verify_every < 0 || p.verify_every > kIntMax)
        return reject("stored output or verification cadence is out of range");
    if (!ckpt3d::valid_promoted_measure_contract(
            p.promoted_measure_policy,
            p.promoted_measure_auto_wave_ctas)) {
        return reject("invalid promoted-measurement reduction contract");
    }
    if (p.channel_height < 0 || p.channel_height > kIntMax ||
        p.channel_padding < 0 || p.channel_padding > kIntMax ||
        !std::isfinite(p.wall_kappa) || !std::isfinite(p.wall_width))
        return reject("invalid resolved-wall checkpoint parameters");
    return true;
}

ckpt3d::CellRecord3D record_from_cell(const CellState3D& c,
                                      int storage_edge) {
    ckpt3d::CellRecord3D r{};
    r.global_id = c.global_id;
    r.origin_x = c.origin_x;
    r.origin_y = c.origin_y;
    r.origin_z = c.origin_z;
    r.polarity_x = c.polarity_x;
    r.polarity_y = c.polarity_y;
    r.polarity_z = c.polarity_z;
    r.velocity_x = c.velocity_x;
    r.velocity_y = c.velocity_y;
    r.velocity_z = c.velocity_z;
    r.gamma = c.gamma;
    r.v_A = c.v_A;
    r.target_radius = c.R_tgt;
    r.phi_max = c.phi_max;
    r.volume = c.V;
    r.moment_x = c.Cx;
    r.moment_y = c.Cy;
    r.moment_z = c.Cz;
    r.surface = c.surface;
    r.interaction_x = c.Ix;
    r.interaction_y = c.Iy;
    r.interaction_z = c.Iz;
    r.shift_counter = c.shift_ctr;
    r.tumble_counter = c.tumble_ctr;
    r.state_flags = c.flags;
    r.bbox_lo_x = c.bb_lo_x;
    r.bbox_hi_x = c.bb_hi_x;
    r.bbox_lo_y = c.bb_lo_y;
    r.bbox_hi_y = c.bb_hi_y;
    r.bbox_lo_z = c.bb_lo_z;
    r.bbox_hi_z = c.bb_hi_z;
    r.pending_shift_x = c.pending_shift_x;
    r.pending_shift_y = c.pending_shift_y;
    r.pending_shift_z = c.pending_shift_z;
    r.storage_edge = static_cast<std::uint32_t>(storage_edge);
    return r;
}

CellState3D cell_from_record(const ckpt3d::CellRecord3D& r, int base_edge) {
    CellState3D c{};
    c.global_id = r.global_id;
    c.origin_x = r.origin_x;
    c.origin_y = r.origin_y;
    c.origin_z = r.origin_z;
    c.polarity_x = r.polarity_x;
    c.polarity_y = r.polarity_y;
    c.polarity_z = r.polarity_z;
    c.velocity_x = r.velocity_x;
    c.velocity_y = r.velocity_y;
    c.velocity_z = r.velocity_z;
    c.gamma = r.gamma;
    c.v_A = r.v_A;
    c.R_tgt = r.target_radius;
    c.phi_max = r.phi_max;
    c.V = r.volume;
    c.Cx = r.moment_x;
    c.Cy = r.moment_y;
    c.Cz = r.moment_z;
    c.surface = r.surface;
    c.Ix = r.interaction_x;
    c.Iy = r.interaction_y;
    c.Iz = r.interaction_z;
    c.shift_ctr = r.shift_counter;
    c.tumble_ctr = r.tumble_counter;
    c.flags = r.state_flags;
    c.bb_lo_x = r.bbox_lo_x;
    c.bb_hi_x = r.bbox_hi_x;
    c.bb_lo_y = r.bbox_lo_y;
    c.bb_hi_y = r.bbox_hi_y;
    c.bb_lo_z = r.bbox_lo_z;
    c.bb_hi_z = r.bbox_hi_z;
    c.pending_shift_x = r.pending_shift_x;
    c.pending_shift_y = r.pending_shift_y;
    c.pending_shift_z = r.pending_shift_z;
    c.storage_edge = r.storage_edge == static_cast<std::uint32_t>(base_edge)
        ? 0u : r.storage_edge;
    return c;
}

bool record_storage_edge(const ckpt3d::CellRecord3D& r,
                         int base_edge, int domain_edge, int* out) {
    if (!out) return false;
    if (r.reserved32 != 0 ||
        !ckpt3d::valid_storage_edge(r.storage_edge,
                                    static_cast<std::uint32_t>(base_edge),
                                    static_cast<std::uint64_t>(domain_edge))) {
        return false;
    }
    *out = static_cast<int>(r.storage_edge);
    return true;
}

bool cell_record_valid(const ckpt3d::CellRecord3D& r,
                       int base_edge, int domain_edge,
                       std::uint32_t boundary_flags,
                       int domain_height, int* storage_edge = nullptr) {
    const std::uint32_t known_flags =
        FLAG3D_COUNT == 32 ? ~std::uint32_t{0}
                           : ((std::uint32_t{1} << FLAG3D_COUNT) - 1u);
    int edge = 0;
    bool reserved_zero = record_storage_edge(
        r, base_edge, domain_edge, &edge);
    for (std::uint64_t value : r.reserved64)
        reserved_zero = reserved_zero && value == 0;
    const bool bounds_valid =
        r.bbox_lo_x >= 0 && r.bbox_lo_x <= r.bbox_hi_x && r.bbox_hi_x < edge &&
        r.bbox_lo_y >= 0 && r.bbox_lo_y <= r.bbox_hi_y && r.bbox_hi_y < edge &&
        r.bbox_lo_z >= 0 && r.bbox_lo_z <= r.bbox_hi_z && r.bbox_hi_z < edge;
    const bool valid =
           ckpt3d::cell_motion_matches_geometry(r, boundary_flags) &&
           ckpt3d::cell_support_respects_z_boundaries(
               r, boundary_flags, domain_height) &&
           std::isfinite(r.volume) && r.volume > 0.0 &&
           std::isfinite(r.moment_x) && std::isfinite(r.moment_y) &&
           std::isfinite(r.moment_z) &&
           std::isfinite(r.surface) && r.surface >= 0.0 &&
           std::isfinite(r.interaction_x) && std::isfinite(r.interaction_y) &&
           std::isfinite(r.interaction_z) &&
           std::isfinite(r.gamma) && r.gamma > 0.0f &&
           std::isfinite(r.v_A) && r.v_A >= 0.0f &&
           std::isfinite(r.target_radius) && r.target_radius > 0.0f &&
           std::isfinite(r.phi_max) && r.phi_max >= 0.0f &&
           (r.state_flags & ~known_flags) == 0 &&
           (r.state_flags & kFatalFlagMask3D) == 0 && bounds_valid &&
           r.pending_shift_x >= -edge && r.pending_shift_x <= edge &&
           r.pending_shift_y >= -edge && r.pending_shift_y <= edge &&
           r.pending_shift_z >= -edge && r.pending_shift_z <= edge &&
           reserved_zero;
    if (valid && storage_edge) *storage_edge = edge;
    return valid;
}

bool sync_stream(cudaStream_t stream) {
    return cuda_ok(cudaStreamSynchronize(stream), "stream synchronization");
}

bool durable_close(std::FILE* file, const std::string& path) {
    if (std::fflush(file) != 0) {
        std::fprintf(stderr, "[ckpt3d] flush failed for %s: %s\n",
                     path.c_str(), std::strerror(errno));
        std::fclose(file);
        return false;
    }
#if defined(_WIN32)
    const int descriptor = _fileno(file);
    const intptr_t raw = _get_osfhandle(descriptor);
    if (raw == -1 || !FlushFileBuffers(reinterpret_cast<HANDLE>(raw))) {
        std::fprintf(stderr, "[ckpt3d] durable flush failed for %s\n",
                     path.c_str());
        std::fclose(file);
        return false;
    }
#else
    if (::fsync(::fileno(file)) != 0) {
        std::fprintf(stderr, "[ckpt3d] fsync failed for %s: %s\n",
                     path.c_str(), std::strerror(errno));
        std::fclose(file);
        return false;
    }
#endif
    if (std::fclose(file) == 0) return true;
    std::fprintf(stderr, "[ckpt3d] close failed for %s: %s\n",
                 path.c_str(), std::strerror(errno));
    return false;
}

bool replace_atomically(const std::string& temporary,
                         const std::string& destination) {
#if defined(_WIN32)
    if (MoveFileExA(temporary.c_str(), destination.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return true;
    std::fprintf(stderr, "[ckpt3d] replace %s failed (Windows error %lu)\n",
                 destination.c_str(), static_cast<unsigned long>(GetLastError()));
    return false;
#else
    if (::rename(temporary.c_str(), destination.c_str()) != 0) {
        std::fprintf(stderr, "[ckpt3d] rename %s -> %s failed: %s\n",
                     temporary.c_str(), destination.c_str(), std::strerror(errno));
        return false;
    }
    const std::filesystem::path target(destination);
    const std::string directory = target.parent_path().empty()
        ? std::string(".") : target.parent_path().string();
    int flags = O_RDONLY;
#if defined(O_DIRECTORY)
    flags |= O_DIRECTORY;
#endif
    const int directory_fd = ::open(directory.c_str(), flags);
    if (directory_fd < 0 || ::fsync(directory_fd) != 0) {
        const int saved_errno = errno;
        if (directory_fd >= 0) ::close(directory_fd);
        std::fprintf(stderr,
                     "[ckpt3d] checkpoint was replaced but directory fsync "
                     "failed for %s: %s\n",
                     directory.c_str(), std::strerror(saved_errno));
        return false;
    }
    ::close(directory_fd);
    return true;
#endif
}

long process_id() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<long>(::getpid());
#endif
}

bool open_unique_temporary(const std::string& destination,
                           std::string* temporary, std::FILE** file) {
    if (!temporary || !file) return false;
    static std::atomic<std::uint64_t> sequence{0};
    for (int attempt = 0; attempt < 128; ++attempt) {
        const std::uint64_t ticket =
            sequence.fetch_add(1, std::memory_order_relaxed);
        *temporary = destination + ".tmp." + std::to_string(process_id()) +
                     "." + std::to_string(ticket);
        errno = 0;
        *file = std::fopen(temporary->c_str(), "wbx");
        if (*file) return true;
        if (errno != EEXIST) {
            std::fprintf(stderr, "[ckpt3d] cannot create %s: %s\n",
                         temporary->c_str(), std::strerror(errno));
            return false;
        }
    }
    std::fprintf(stderr,
                 "[ckpt3d] could not reserve a unique temporary for %s\n",
                 destination.c_str());
    return false;
}

bool metadata_from_records(const ckpt3d::FileHeader3D& header,
                           const ckpt3d::ParamsRecord3D& params,
                           const std::string& path,
                           CheckpointMeta3D* out) {
    CheckpointMeta3D result{};
    params_from_record(params, header.num_cells, &result.params);
    result.step = static_cast<std::uint64_t>(header.step);
    result.time = header.time;
    result.brick_edge = static_cast<int>(header.brick_edge);
    result.max_storage_edge = result.brick_edge;
    result.print_interval = static_cast<int>(params.print_interval);
    result.trajectory_interval = params.trajectory_interval;
    result.promoted_measure_reduction = decode_promoted_measure_reduction(
        params.promoted_measure_policy,
        params.promoted_measure_auto_wave_ctas);
    result.base_measure_shards =
        static_cast<int>(header.base_measure_shards);
    result.file_crc64 = header.file_crc64;

    const char* validation_error = nullptr;
    if (!validate(result.params, &validation_error)) {
        std::fprintf(stderr, "[ckpt3d] %s stores invalid parameters: %s\n",
                     path.c_str(),
                     validation_error ? validation_error : "unknown reason");
        return false;
    }
    const double expected_time = static_cast<double>(result.step) *
        result.params.dt;
    const double time_tolerance = 1.0e-12 *
        std::max(1.0, std::fabs(expected_time));
    if (!std::isfinite(expected_time) ||
        std::fabs(result.time - expected_time) > time_tolerance) {
        std::fprintf(stderr,
                     "[ckpt3d] %s time does not match step multiplied by dt\n",
                     path.c_str());
        return false;
    }
    const int required_edge = result.params.brick_edge();
    const int minimum_domain_edge = result.params.minimum_domain_edge();
    if (result.brick_edge < required_edge ||
        result.brick_edge >= minimum_domain_edge ||
        result.brick_edge % kBrickAlignment != 0) {
        std::fprintf(stderr,
                     "[ckpt3d] %s stores B=%d, but these parameters require "
                     "an aligned brick of at least %d strictly inside the "
                     "minimum domain edge %d\n",
                     path.c_str(), result.brick_edge, required_edge,
                     minimum_domain_edge);
        return false;
    }
    *out = result;
    return true;
}

bool scan_storage_layout(std::FILE* file,
                         const ckpt3d::FileHeader3D& header,
                         const ckpt3d::ParamsRecord3D& params,
                         std::uintmax_t observed_size,
                         const std::string& path,
                         std::vector<int>* storage_edges,
                         int* max_storage_edge) {
    if (!file || !storage_edges || !max_storage_edge) return false;
    const std::size_t N = static_cast<std::size_t>(header.num_cells);
    storage_edges->clear();
    *max_storage_edge = static_cast<int>(header.brick_edge);
    const std::uintmax_t base_payload = static_cast<std::uintmax_t>(
        header.phase_bytes_per_cell);
    std::uintmax_t minimum_cells = 0;
    std::uintmax_t minimum_size = 0;
    std::uintmax_t prefix_size = 0;
    if (!checked_add(header.header_bytes, header.params_bytes, &prefix_size) ||
        !checked_add(sizeof(ckpt3d::CellRecord3D), base_payload,
                     &minimum_cells) ||
        !checked_multiply(minimum_cells, header.num_cells, &minimum_cells) ||
        !checked_add(prefix_size, minimum_cells, &minimum_size) ||
        minimum_size > observed_size) {
        std::fprintf(stderr,
                     "[ckpt3d] %s is too small for its cell count and base edge\n",
                     path.c_str());
        return false;
    }
    storage_edges->reserve(N);

    std::uintmax_t expected_size = 0;
    if (!checked_add(header.header_bytes, header.params_bytes,
                     &expected_size)) {
        return false;
    }
    for (std::size_t index = 0; index < N; ++index) {
        if (!checked_add(expected_size, sizeof(ckpt3d::CellRecord3D),
                         &expected_size) || expected_size > observed_size) {
            std::fprintf(stderr, "[ckpt3d] %s truncates cell record %zu\n",
                         path.c_str(), index);
            return false;
        }
        ckpt3d::CellRecord3D record{};
        if (!read_exact(file, &record, sizeof(record), "cell record"))
            return false;
        int edge = 0;
        if (!record_storage_edge(record,
                                 static_cast<int>(header.brick_edge),
                                 static_cast<int>(ckpt3d::storage_domain_edge(
                                     params.boundary_flags, params.Nx,
                                     params.Ny, params.Nz)),
                                 &edge)) {
            std::fprintf(stderr,
                         "[ckpt3d] %s has an invalid storage edge in cell %zu\n",
                         path.c_str(), index);
            return false;
        }
        if (record.global_id != static_cast<std::int64_t>(index)) {
            std::fprintf(stderr,
                         "[ckpt3d] %s cell slot %zu carries global id %lld; "
                         "expected %zu\n",
                         path.c_str(), index,
                         static_cast<long long>(record.global_id), index);
            return false;
        }
        const std::uint64_t words = ckpt3d::phase_values_per_cell(
            static_cast<std::uint32_t>(edge));
        if (words == 0 || words > std::numeric_limits<std::uint64_t>::max() /
                                    sizeof(float)) {
            return false;
        }
        const std::uint64_t bytes = words * sizeof(float);
        if (!checked_add(expected_size, static_cast<std::uintmax_t>(bytes),
                         &expected_size) || expected_size > observed_size ||
            !seek_forward(file, bytes)) {
            std::fprintf(stderr,
                         "[ckpt3d] %s truncates phase payload %zu\n",
                         path.c_str(), index);
            return false;
        }
        storage_edges->push_back(edge);
        *max_storage_edge = std::max(*max_storage_edge, edge);
    }
    if (expected_size != observed_size) {
        std::fprintf(stderr,
                     "[ckpt3d] %s has size %ju bytes; cell records require "
                     "%ju\n",
                     path.c_str(), observed_size, expected_size);
        return false;
    }
    const std::uintmax_t prefix = static_cast<std::uintmax_t>(
        header.header_bytes) + header.params_bytes;
    if (prefix > static_cast<std::uintmax_t>(LONG_MAX) ||
        std::fseek(file, static_cast<long>(prefix), SEEK_SET) != 0) {
        std::fprintf(stderr, "[ckpt3d] %s cannot rewind after probing\n",
                     path.c_str());
        return false;
    }
    return true;
}

bool read_validated_prefix(std::FILE* file, const std::string& path,
                           ckpt3d::FileHeader3D* header,
                           ckpt3d::ParamsRecord3D* params,
                           CheckpointMeta3D* metadata) {
    if (!file || !header || !params || !metadata) return false;
    // Reject format, parameter, and layout inconsistencies before any
    // checkpoint payload is copied to device state.
    std::memset(params, 0, sizeof(*params));
    if (!read_exact(file, header, sizeof(*header), "header") ||
        !header_valid(*header, path) ||
        !read_exact(file, params, header->params_bytes, "parameters") ||
        !params_record_valid(*params, *header, path)) {
        return false;
    }

    std::uintmax_t observed_size = 0;
    if (!open_file_size(file, &observed_size) ||
        !metadata_from_records(*header, *params, path, metadata) ||
        !scan_storage_layout(file, *header, *params, observed_size, path,
                             &metadata->storage_edges,
                             &metadata->max_storage_edge)) {
        return false;
    }
    return true;
}

bool params_equal(const SimParams3D& a, const SimParams3D& b) {
    return a.Nx == b.Nx && a.Ny == b.Ny && a.Nz == b.Nz &&
           a.dx == b.dx && a.dy == b.dy && a.dz == b.dz &&
           a.boundary_flags == b.boundary_flags &&
           a.num_cells == b.num_cells && a.rho == b.rho && a.dt == b.dt &&
           a.t_end == b.t_end && a.lambda == b.lambda &&
           a.target_radius == b.target_radius && a.kappa == b.kappa &&
           a.mu == b.mu && a.xi == b.xi && a.tau == b.tau &&
           a.v_A == b.v_A && a.gamma_normal == b.gamma_normal &&
           a.gamma_cancer == b.gamma_cancer &&
           a.cancer_fraction == b.cancer_fraction &&
           a.v_A_sigma == b.v_A_sigma && a.seed == b.seed &&
           a.polarity_seed == b.polarity_seed &&
           a.initialization_hash == b.initialization_hash &&
           a.aging_time == b.aging_time &&
           a.channel_height == b.channel_height &&
           a.channel_padding == b.channel_padding &&
           a.wall_kappa == b.wall_kappa &&
           a.wall_width == b.wall_width &&
           a.full_moment_every == b.full_moment_every &&
           a.verify_every == b.verify_every;
}

bool metadata_equal(const CheckpointMeta3D& a, const CheckpointMeta3D& b) {
    return a.step == b.step && a.time == b.time &&
           a.brick_edge == b.brick_edge &&
           a.max_storage_edge == b.max_storage_edge &&
           a.print_interval == b.print_interval &&
           a.trajectory_interval == b.trajectory_interval &&
           a.promoted_measure_reduction.policy ==
               b.promoted_measure_reduction.policy &&
           a.promoted_measure_reduction.auto_wave_ctas ==
               b.promoted_measure_reduction.auto_wave_ctas &&
           a.base_measure_shards == b.base_measure_shards &&
           a.file_crc64 == b.file_crc64 &&
           a.storage_edges == b.storage_edges && params_equal(a.params, b.params);
}

}  // namespace

bool checkpoint_probe_3d(const std::string& path, CheckpointMeta3D* out) {
    if (!out || path.empty()) return false;
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "[ckpt3d] cannot open %s: %s\n", path.c_str(),
                     std::strerror(errno));
        return false;
    }
    ckpt3d::FileHeader3D header{};
    ckpt3d::ParamsRecord3D params{};
    CheckpointMeta3D result{};
    const bool ok = read_validated_prefix(file, path, &header, &params, &result);
    if (std::fclose(file) != 0 && ok) {
        std::fprintf(stderr, "[ckpt3d] close failed while probing %s: %s\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }
    if (!ok) return false;
    *out = result;
    return true;
}

bool checkpoint_load_3d(const std::string& path,
                        const CheckpointMeta3D& expected,
                        const CheckpointLoadView3D& view) {
    if (!view.d_cells || !view.d_phi || expected.params.num_cells <= 0 ||
        expected.brick_edge <= 0 ||
        expected.storage_edges.size() !=
            static_cast<std::size_t>(expected.params.num_cells)) {
        return false;
    }
    for (std::size_t index = 0; index < expected.storage_edges.size(); ++index) {
        const int edge = expected.storage_edges[index];
        if (edge > expected.brick_edge &&
            (!view.h_promoted_phi || !view.h_promoted_phi[index])) {
            std::fprintf(stderr,
                         "[ckpt3d] promoted checkpoint storage was not allocated\n");
            return false;
        }
    }
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "[ckpt3d] cannot open %s: %s\n", path.c_str(),
                     std::strerror(errno));
        return false;
    }
    ckpt3d::FileHeader3D header{};
    ckpt3d::ParamsRecord3D params{};
    CheckpointMeta3D observed{};
    if (!read_validated_prefix(file, path, &header, &params, &observed)) {
        std::fclose(file);
        return false;
    }
    if (!metadata_equal(observed, expected)) {
        std::fprintf(stderr,
                     "[ckpt3d] metadata differs from the state returned by "
                     "the probe; refusing a changed checkpoint\n");
        std::fclose(file);
        return false;
    }

    const int N = expected.params.num_cells;
    const int B = expected.brick_edge;
    std::size_t base_voxels = 0;
    std::size_t maximum_voxels = 0;
    if (!checked_cube_size(static_cast<std::size_t>(B), &base_voxels) ||
        !checked_cube_size(static_cast<std::size_t>(
                               expected.max_storage_edge),
                           &maximum_voxels)) {
        std::fclose(file);
        return false;
    }
    const std::size_t phase_stage_words = std::min<std::size_t>(
        maximum_voxels,
        std::max<std::size_t>(1, kStageTargetBytes / sizeof(float)));
    const std::size_t cell_stage_capacity = std::min<std::size_t>(
        static_cast<std::size_t>(N),
        std::max<std::size_t>(1, kCellStageTargetBytes / sizeof(CellState3D)));
    // Stream phase data through bounded host buffers while extending the CRC
    // in the checkpoint's canonical byte order.
    std::vector<CellState3D> cell_stage(cell_stage_capacity);
    std::vector<float> phase_stage(phase_stage_words);
    ckpt3d::FileHeader3D canonical_header = header;
    canonical_header.file_crc64 = 0;
    std::uint64_t file_crc64 = ckpt3d::crc64_ecma(
        &canonical_header, sizeof(canonical_header));
    file_crc64 = ckpt3d::crc64_ecma_update(
        file_crc64, &params, header.params_bytes);
    int base = 0;
    while (base < N) {
        const int count = std::min<int>(N - base,
            static_cast<int>(cell_stage_capacity));
        for (int k = 0; k < count; ++k) {
            const int index = base + k;
            ckpt3d::CellRecord3D record{};
            int edge = 0;
            if (!read_exact(file, &record, sizeof(record), "cell record") ||
                !cell_record_valid(record, B,
                                   expected.params.minimum_domain_edge(),
                                   expected.params.boundary_flags,
                                   expected.params.Nz,
                                   &edge) ||
                edge != expected.storage_edges[static_cast<std::size_t>(index)] ||
                record.global_id != static_cast<std::int64_t>(index)) {
                std::fprintf(stderr, "[ckpt3d] invalid cell record %d\n", index);
                std::fclose(file);
                return false;
            }
            file_crc64 = ckpt3d::crc64_ecma_update(
                file_crc64, &record, sizeof(record));
            cell_stage[static_cast<std::size_t>(k)] = cell_from_record(
                record, B);
            std::size_t cell_voxels = 0;
            if (!checked_cube_size(static_cast<std::size_t>(edge),
                                   &cell_voxels)) {
                std::fclose(file);
                return false;
            }
            float* destination = edge == B
                ? view.d_phi + static_cast<std::size_t>(index) * base_voxels
                : view.h_promoted_phi[index];
            for (std::size_t offset = 0; offset < cell_voxels;
                 offset += phase_stage_words) {
                const std::size_t words =
                    std::min(phase_stage_words, cell_voxels - offset);
                if (!read_exact(file, phase_stage.data(), words * sizeof(float),
                                "phase brick")) {
                    std::fclose(file);
                    return false;
                }
                file_crc64 = ckpt3d::crc64_ecma_update(
                    file_crc64, phase_stage.data(), words * sizeof(float));
                if (!cuda_ok(cudaMemcpyAsync(destination + offset,
                                             phase_stage.data(),
                                             words * sizeof(float),
                                             cudaMemcpyHostToDevice, view.stream),
                             "phase payload H2D") ||
                    !sync_stream(view.stream)) {
                    std::fclose(file);
                    return false;
                }
            }
        }
        if (!cuda_ok(cudaMemcpyAsync(view.d_cells + base, cell_stage.data(),
                                     static_cast<std::size_t>(count) *
                                         sizeof(CellState3D),
                                     cudaMemcpyHostToDevice, view.stream),
                     "cell state H2D") ||
            !sync_stream(view.stream)) {
            std::fclose(file);
            return false;
        }
        base += count;
    }
    if (file_crc64 != observed.file_crc64) {
        std::fprintf(stderr,
                     "[ckpt3d] file CRC-64/ECMA mismatch for %s\n",
                     path.c_str());
        std::fclose(file);
        return false;
    }
    if (std::fclose(file) != 0) {
        std::fprintf(stderr, "[ckpt3d] close failed after loading %s: %s\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }
    return true;
}

bool checkpoint_load_3d(const std::string& path,
                        const CheckpointMeta3D& expected,
                        CellState3D* d_cells, float* d_phi,
                        cudaStream_t stream) {
    CheckpointLoadView3D view{};
    view.d_cells = d_cells;
    view.d_phi = d_phi;
    view.stream = stream;
    return checkpoint_load_3d(path, expected, view);
}

bool checkpoint_write_3d(const std::string& path,
                          const CheckpointWriteView3D& view) {
    if (path.empty() || !view.params || !view.d_cells || !view.d_phi ||
        view.params->num_cells <= 0 || view.brick_edge <= 0 ||
        !ckpt3d::host_is_little_endian()) {
        std::fprintf(stderr, "[ckpt3d] incomplete or unsupported write view\n");
        return false;
    }
    if (view.step > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) ||
        !std::isfinite(view.time) || view.time < 0.0 || view.print_interval < 0) {
        std::fprintf(stderr, "[ckpt3d] invalid step, time, or print cadence\n");
        return false;
    }
    if (!valid_checkpoint_promoted_measure_reduction(
            view.promoted_measure_reduction)) {
        std::fprintf(stderr,
            "[ckpt3d] invalid promoted-measurement reduction contract\n");
        return false;
    }
    if (view.base_measure_shards < 1 ||
        view.base_measure_shards > kMaximumBaseMeasureShards) {
        std::fprintf(stderr,
            "[ckpt3d] refusing to write a checkpoint without the resolved "
            "base measurement shard count\n");
        return false;
    }
    const char* validation_error = nullptr;
    if (!validate(*view.params, &validation_error)) {
        std::fprintf(stderr, "[ckpt3d] refusing invalid parameters: %s\n",
                     validation_error ? validation_error : "unknown reason");
        return false;
    }
    const double expected_time = static_cast<double>(view.step) *
        view.params->dt;
    const double time_tolerance = 1.0e-12 *
        std::max(1.0, std::fabs(expected_time));
    if (!std::isfinite(expected_time) ||
        std::fabs(view.time - expected_time) > time_tolerance) {
        std::fprintf(stderr,
                     "[ckpt3d] checkpoint time does not match step multiplied "
                     "by dt\n");
        return false;
    }
    const int required_edge = view.params->brick_edge();
    if (!ckpt3d::valid_brick_edge(
            static_cast<std::uint32_t>(view.brick_edge)) ||
        view.brick_edge % kBrickAlignment != 0 ||
        view.brick_edge < required_edge ||
        view.brick_edge >= view.params->minimum_domain_edge()) {
        std::fprintf(stderr,
                     "[ckpt3d] B=%d is incompatible with required B>=%d and "
                     "minimum domain edge %d\n",
                     view.brick_edge, required_edge,
                     view.params->minimum_domain_edge());
        return false;
    }
    std::size_t base_voxels = 0;
    if (!checked_cube_size(static_cast<std::size_t>(view.brick_edge),
                           &base_voxels))
        return false;

    const int N = view.params->num_cells;
    std::vector<CellState3D> cells(static_cast<std::size_t>(N));
    if (!cuda_ok(cudaMemcpyAsync(cells.data(), view.d_cells,
                                 cells.size() * sizeof(CellState3D),
                                 cudaMemcpyDeviceToHost, view.stream),
                 "cell state D2H") || !sync_stream(view.stream)) {
        return false;
    }
    std::vector<int> storage_edges(static_cast<std::size_t>(N));
    std::size_t maximum_voxels = base_voxels;
    for (int index = 0; index < N; ++index) {
        const CellState3D& state = cells[static_cast<std::size_t>(index)];
        if (state.storage_edge > static_cast<std::uint32_t>(
                                     std::numeric_limits<int>::max())) {
            std::fprintf(stderr,
                         "[ckpt3d] invalid storage edge in cell %d\n", index);
            return false;
        }
        const int edge = state.storage_edge == 0u
            ? view.brick_edge : static_cast<int>(state.storage_edge);
        const ckpt3d::CellRecord3D record = record_from_cell(state, edge);
        int checked_edge = 0;
        if (!cell_record_valid(record, view.brick_edge,
                               view.params->minimum_domain_edge(),
                               view.params->boundary_flags,
                               view.params->Nz,
                               &checked_edge) || checked_edge != edge ||
            record.global_id != static_cast<std::int64_t>(index)) {
            std::fprintf(stderr, "[ckpt3d] refusing invalid cell state %d\n",
                         index);
            return false;
        }
        if (edge > view.brick_edge &&
            (!view.h_promoted_phi || !view.h_promoted_phi[index])) {
            std::fprintf(stderr,
                         "[ckpt3d] cell %d has no promoted phase allocation\n",
                         index);
            return false;
        }
        std::size_t cell_voxels = 0;
        if (!checked_cube_size(static_cast<std::size_t>(edge),
                               &cell_voxels)) {
            return false;
        }
        storage_edges[static_cast<std::size_t>(index)] = edge;
        maximum_voxels = std::max(maximum_voxels, cell_voxels);
    }
    const std::size_t phase_stage_words = std::min<std::size_t>(
        maximum_voxels,
        std::max<std::size_t>(1, kStageTargetBytes / sizeof(float)));
    std::vector<float> phase_stage(phase_stage_words);

    std::filesystem::path destination(path);
    std::error_code ec;
    if (!destination.parent_path().empty())
        std::filesystem::create_directories(destination.parent_path(), ec);
    if (ec) {
        std::fprintf(stderr, "[ckpt3d] cannot create %s: %s\n",
                     destination.parent_path().string().c_str(),
                     ec.message().c_str());
        return false;
    }
    std::string temporary;
    std::FILE* file = nullptr;
    // Publish an exclusive sibling only after checksum backfill and a durable
    // close, leaving the previous checkpoint intact on write failure.
    if (!open_unique_temporary(path, &temporary, &file))
        return false;

    ckpt3d::FileHeader3D header{};
    header.magic = ckpt3d::kMagic;
    header.version = ckpt3d::kCheckpointFormat;
    header.header_bytes = sizeof(header);
    header.endian_marker = ckpt3d::kEndianMarker;
    header.dimensions = ckpt3d::kDimensions;
    header.scalar_format = ckpt3d::kScalarFloat32;
    header.flags = ckpt3d::kRequiredHeaderFlags;
    header.base_measure_shards =
        static_cast<std::uint64_t>(view.base_measure_shards);
    header.step = static_cast<std::int64_t>(view.step);
    header.time = view.time;
    header.num_cells = static_cast<std::uint64_t>(N);
    header.params_bytes = sizeof(ckpt3d::ParamsRecord3D);
    header.cell_record_bytes = sizeof(ckpt3d::CellRecord3D);
    header.brick_edge = static_cast<std::uint32_t>(view.brick_edge);
    header.phase_order = ckpt3d::kPhaseOrderXFastest;
    header.phase_values_per_cell = base_voxels;
    header.phase_bytes_per_cell = base_voxels * sizeof(float);
    ckpt3d::ParamsRecord3D params{};
    params_to_record(view, &params);
    if (!header_valid(header, temporary) ||
        !params_record_valid(params, header, temporary)) {
        std::fclose(file);
        std::filesystem::remove(temporary, ec);
        return false;
    }
    bool ok = write_exact(file, &header, sizeof(header), "header") &&
              write_exact(file, &params, header.params_bytes, "parameters");
    std::uint64_t file_crc64 = ckpt3d::crc64_ecma(&header, sizeof(header));
    file_crc64 = ckpt3d::crc64_ecma_update(
        file_crc64, &params, header.params_bytes);
    for (int index = 0; ok && index < N; ++index) {
        const CellState3D& state = cells[static_cast<std::size_t>(index)];
        const int edge = storage_edges[static_cast<std::size_t>(index)];
        const ckpt3d::CellRecord3D record = record_from_cell(state, edge);
        ok = write_exact(file, &record, sizeof(record), "cell record");
        if (ok) {
            file_crc64 = ckpt3d::crc64_ecma_update(
                file_crc64, &record, sizeof(record));
        }
        std::size_t cell_voxels = 0;
        ok = ok && checked_cube_size(static_cast<std::size_t>(edge),
                                     &cell_voxels);
        const float* source = edge == view.brick_edge
            ? view.d_phi + static_cast<std::size_t>(index) * base_voxels
            : view.h_promoted_phi[index];
        for (std::size_t offset = 0; ok && offset < cell_voxels;
             offset += phase_stage_words) {
            const std::size_t words =
                std::min(phase_stage_words, cell_voxels - offset);
            ok = cuda_ok(cudaMemcpyAsync(
                              phase_stage.data(), source + offset,
                              words * sizeof(float),
                              cudaMemcpyDeviceToHost, view.stream),
                          "phase payload D2H") &&
                 sync_stream(view.stream) &&
                 write_exact(file, phase_stage.data(), words * sizeof(float),
                              "phase brick");
            if (ok) {
                file_crc64 = ckpt3d::crc64_ecma_update(
                    file_crc64, phase_stage.data(), words * sizeof(float));
            }
        }
    }
    if (!ok) {
        std::fclose(file);
        std::filesystem::remove(temporary, ec);
        return false;
    }
    header.file_crc64 = file_crc64;
    if (std::fseek(file, 0, SEEK_SET) != 0 ||
        !write_exact(file, &header, sizeof(header), "checksummed header")) {
        std::fclose(file);
        std::filesystem::remove(temporary, ec);
        return false;
    }
    if (!durable_close(file, temporary) ||
        !replace_atomically(temporary, path)) {
        std::filesystem::remove(temporary, ec);
        return false;
    }
    std::printf("[ckpt3d] saved step %llu, %d cells, B=%d -> %s\n",
                static_cast<unsigned long long>(view.step), N,
                view.brick_edge, path.c_str());
    return true;
}

}  // namespace pf3d
