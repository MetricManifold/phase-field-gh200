#pragma once

#include "model_coefficients.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define PF3D_HD __host__ __device__
#define PF3D_FORCEINLINE __forceinline__
#else
#define PF3D_HD
#define PF3D_FORCEINLINE inline
#endif

namespace pf3d {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Boundary flags are part of the restart contract. Periodic XYZ is the default.
// The substrate slab and hard-wall channel are periodic only in x/y. The
// separate wall bit prevents a two-wall channel checkpoint from being mistaken
// for the sessile-cap substrate geometry.
constexpr std::uint32_t kBoundaryPeriodicX3D = 1u << 0;
constexpr std::uint32_t kBoundaryPeriodicY3D = 1u << 1;
constexpr std::uint32_t kBoundaryPeriodicZ3D = 1u << 2;
constexpr std::uint32_t kBoundaryChannelZ3D = 1u << 3;
constexpr std::uint32_t kBoundaryPeriodicXYZ3D =
    kBoundaryPeriodicX3D | kBoundaryPeriodicY3D | kBoundaryPeriodicZ3D;
constexpr std::uint32_t kBoundarySubstrateSlab3D =
    kBoundaryPeriodicX3D | kBoundaryPeriodicY3D;
constexpr std::uint32_t kBoundaryHardWallChannel3D =
    kBoundaryPeriodicX3D | kBoundaryPeriodicY3D | kBoundaryChannelZ3D;

// These are the Palmieri coefficients after including mobility M = 1/2.
// Their form is independent of spatial dimension; only the target measure
// changes from cell area to cell volume.
using pf_common::bulk_coeff;
using pf_common::interaction_coeff;
using pf_common::kNumerBulk;
using pf_common::kNumerInteraction;
using pf_common::kNumerVolume;
using pf_common::motility_coeff;
using pf_common::volume_coeff;

template <typename T>
PF3D_HD constexpr T target_sphere_volume(T radius) {
    return T(4) * T(kPi) * radius * radius * radius / T(3);
}

// Neutral 90-degree sessile cap: the slab radius is both the cap/base radius
// used for areal coverage and the radius of its hemispherical target volume.
template <typename T>
PF3D_HD constexpr T target_slab_volume(T radius) {
    return T(2) * T(kPi) * radius * radius * radius / T(3);
}

template <typename T>
PF3D_HD constexpr T target_slab_footprint_area(T radius) {
    return T(kPi) * radius * radius;
}

using pf_common::interface_k;

// Leading planar-interface correction for integral(phi^2 dV). The initializer
// subsequently solves the discrete spherical-volume condition numerically.
using pf_common::init_radius;

// Isotropic 27-point Laplacian on a unit cubic lattice. Neighbours sharing a
// face, edge, or corner have weights 14, 3, and 1 respectively.
constexpr int    kLapFaceW    = 14;
constexpr int    kLapEdgeW    = 3;
constexpr int    kLapCornerW  = 1;
constexpr int    kLapCentreW  = -128;
constexpr double kLapDenom    = 30.0;
constexpr double kLapRowSumBound = 256.0 / 30.0;

static_assert(6 * kLapFaceW + 12 * kLapEdgeW + 8 * kLapCornerW
                  + kLapCentreW == 0,
              "27-point Laplacian weights must sum to zero");
static_assert(2 * kLapFaceW + 8 * kLapEdgeW + 8 * kLapCornerW == 60,
              "27-point Laplacian must give lap(x^2) = 60/30 = 2");

// Q5.27 fixed point for S = sum_m phi_m^2, matching the 2-D solver.
constexpr float  kQScaleF     = 134217728.0f;              // 2^27
constexpr float  kQInvF       = 7.450580596923828125e-9f;  // 2^-27
constexpr double kQInvD       = 7.450580596923828125e-9;
constexpr float  kQClampPhiSq = 4.0f;

PF3D_HD PF3D_FORCEINLINE uint32_t q_of(float phi) {
    float value = phi * phi;
    if (value > kQClampPhiSq) value = kQClampPhiSq;
#if defined(__CUDA_ARCH__)
    return __float2uint_rn(value * kQScaleF);
#else
    return static_cast<uint32_t>(std::lrint(value * kQScaleF));
#endif
}

constexpr float  kSupportEps         = 1.0e-5f;
constexpr int    kBrickAlignment     = 8;
constexpr double kBrickSafetyMargin = 8.0;  // total slack across the diameter

struct BrickSizing3D {
    double effective_radius = 0.0;
    double support_tail = 0.0;
    double physical_support_extent = 0.0;
    double safety_margin = 0.0;
    double required_extent = 0.0;
    int edge = -1;
};

inline double support_tail(double lambda, double eps = kSupportEps) {
    if (!(lambda > 0.0) || !(eps > 0.0 && eps < 0.5))
        return std::numeric_limits<double>::infinity();
    return std::log(1.0 / eps - 1.0) / (2.0 * interface_k(lambda));
}

inline int round_up_to_multiple(int value, int alignment) {
    if (value <= 0 || alignment <= 0 || value >
            std::numeric_limits<int>::max() - (alignment - 1))
        return -1;
    return ((value + alignment - 1) / alignment) * alignment;
}

inline BrickSizing3D brick_sizing(double radius, double lambda,
                                  double safety_margin = kBrickSafetyMargin,
                                  double eps = kSupportEps) {
    BrickSizing3D result{};
    result.safety_margin = safety_margin;
    if (!(radius > 0.0) || !(lambda > 0.0) || !(safety_margin >= 0.0))
        return result;
    result.effective_radius = init_radius(radius, lambda);
    result.support_tail = support_tail(lambda, eps);
    result.physical_support_extent =
        2.0 * (result.effective_radius + result.support_tail);
    result.required_extent = result.physical_support_extent + safety_margin;
    if (!std::isfinite(result.required_extent) ||
        result.required_extent > static_cast<double>(std::numeric_limits<int>::max()))
        return result;
    result.edge = round_up_to_multiple(
        static_cast<int>(std::ceil(result.required_extent)), kBrickAlignment);
    return result;
}

inline int automatic_brick_edge(double radius, double lambda,
                                double safety_margin = kBrickSafetyMargin,
                                double eps = kSupportEps) {
    return brick_sizing(radius, lambda, safety_margin, eps).edge;
}

inline bool checked_mul_size(std::size_t a, std::size_t b,
                             std::size_t* out) {
    if (!out) return false;
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        return false;
    *out = a * b;
    return true;
}

inline bool checked_cube_size(std::size_t edge, std::size_t* out) {
    std::size_t square = 0;
    return checked_mul_size(edge, edge, &square) &&
           checked_mul_size(square, edge, out);
}

inline bool checked_box_size(std::size_t nx, std::size_t ny, std::size_t nz,
                             std::size_t* out) {
    std::size_t plane = 0;
    return checked_mul_size(nx, ny, &plane) &&
           checked_mul_size(plane, nz, out);
}

inline bool checked_phase_field_bytes(int num_cells, int brick_edge,
                                      int buffers, std::size_t* out) {
    if (num_cells <= 0 || brick_edge <= 0 || buffers <= 0 || !out) return false;
    std::size_t voxels = 0, words = 0, all_buffers = 0;
    return checked_cube_size(static_cast<std::size_t>(brick_edge), &voxels) &&
           checked_mul_size(voxels, static_cast<std::size_t>(num_cells), &words) &&
           checked_mul_size(words, static_cast<std::size_t>(buffers), &all_buffers) &&
           checked_mul_size(all_buffers, sizeof(float), out);
}

inline bool checked_aggregate_field_bytes(int nx, int ny, int nz, int buffers,
                                          std::size_t* out) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || buffers <= 0 || !out) return false;
    std::size_t voxels = 0, words = 0;
    return checked_box_size(static_cast<std::size_t>(nx),
                            static_cast<std::size_t>(ny),
                            static_cast<std::size_t>(nz), &voxels) &&
           checked_mul_size(voxels, static_cast<std::size_t>(buffers), &words) &&
           checked_mul_size(words, sizeof(uint32_t), out);
}

inline bool checked_aggregate_field_bytes(int side, int buffers,
                                          std::size_t* out) {
    return checked_aggregate_field_bytes(side, side, side, buffers, out);
}

inline bool checked_total_field_bytes(int num_cells, int brick_edge,
                                      int nx, int ny, int nz,
                                      int phi_buffers, int aggregate_buffers,
                                      std::size_t* out) {
    if (!out) return false;
    std::size_t phi = 0, aggregate = 0;
    if (!checked_phase_field_bytes(num_cells, brick_edge, phi_buffers, &phi) ||
        !checked_aggregate_field_bytes(nx, ny, nz, aggregate_buffers,
                                       &aggregate) ||
        aggregate > std::numeric_limits<std::size_t>::max() - phi)
        return false;
    *out = phi + aggregate;
    return true;
}

inline bool checked_total_field_bytes(int num_cells, int brick_edge, int side,
                                      int phi_buffers, int aggregate_buffers,
                                      std::size_t* out) {
    return checked_total_field_bytes(num_cells, brick_edge, side, side, side,
                                     phi_buffers, aggregate_buffers, out);
}

inline int domain_side_for(int num_cells, double radius, double rho) {
    if (num_cells <= 0 || !(radius > 0.0) || !(rho > 0.0)) return -1;
    const double volume = static_cast<double>(num_cells) *
                          target_sphere_volume(radius) / rho;
    const double side = std::ceil(std::cbrt(volume));
    if (!std::isfinite(side) || side < 1.0 ||
        side > static_cast<double>(std::numeric_limits<int>::max()))
        return -1;
    return static_cast<int>(side);
}

inline int slab_domain_side_for(int num_cells, double radius, double rho_A) {
    if (num_cells <= 0 || !(radius > 0.0) || !(rho_A > 0.0)) return -1;
    const double area = static_cast<double>(num_cells) *
                        target_slab_footprint_area(radius) / rho_A;
    const double side = std::ceil(std::sqrt(area));
    if (!std::isfinite(side) || side < 1.0 ||
        side > static_cast<double>(std::numeric_limits<int>::max()))
        return -1;
    return static_cast<int>(side);
}

inline int channel_domain_side_for(int num_cells, double radius,
                                   double rho_V, int height) {
    if (num_cells <= 0 || !(radius > 0.0) || !(rho_V > 0.0) || height <= 0)
        return -1;
    const double area = static_cast<double>(num_cells) *
                        target_sphere_volume(radius) /
                        (rho_V * static_cast<double>(height));
    const double side = std::ceil(std::sqrt(area));
    if (!std::isfinite(side) || side < 1.0 ||
        side > static_cast<double>(std::numeric_limits<int>::max()))
        return -1;
    return static_cast<int>(side);
}

inline int channel_wall_padding(double wall_width) {
    if (!(wall_width > 0.0) || !std::isfinite(wall_width) ||
        wall_width > static_cast<double>(std::numeric_limits<int>::max()) / 3.0)
        return -1;
    const double padding = std::ceil(3.0 * wall_width);
    return padding >= 1.0 &&
           padding <= static_cast<double>(std::numeric_limits<int>::max())
        ? static_cast<int>(padding) : -1;
}

struct SimParams3D {
    int Nx = 0, Ny = 0, Nz = 0;
    double dx = 1.0, dy = 1.0, dz = 1.0;
    std::uint32_t boundary_flags = kBoundaryPeriodicXYZ3D;
    int num_cells = 288;
    double rho = 0.90;  // volume fraction, except substrate-slab areal coverage
    double dt = 0.01;
    double t_end = 100.0;
    double aging_time = 0.0;
    double lambda = 7.0;
    double target_radius = 49.0;
    double kappa = 10.0;
    // A hard-wall channel stores accessible slit height H and resolved solid
    // padding outside each physical wall.
    int channel_height = 0;
    int channel_padding = 0;
    double wall_kappa = 0.0;
    double wall_width = 0.0;
    double mu = 1.0;
    double xi = 1500.0;
    double tau = 1.0e4;
    double v_A = 1.0e-2;
    double gamma_normal = 1.0;
    double gamma_cancer = 0.35;
    double cancer_fraction = 0.0;
    double v_A_sigma = 0.0;
    unsigned long long seed = 1234;
    unsigned long long polarity_seed = 0;
    unsigned long long initialization_hash = 0;
    int full_moment_every = 100;
    int verify_every = 4096;

    PF3D_HD unsigned long long polarity_stream() const {
        return polarity_seed ? polarity_seed : seed;
    }
    PF3D_HD bool substrate_slab() const {
        return boundary_flags == kBoundarySubstrateSlab3D;
    }
    PF3D_HD bool hard_wall_channel() const {
        return boundary_flags == kBoundaryHardWallChannel3D;
    }
    PF3D_HD bool periodic_xyz() const {
        return boundary_flags == kBoundaryPeriodicXYZ3D;
    }
    PF3D_HD bool bounded_z() const {
        return substrate_slab() || hard_wall_channel();
    }
    PF3D_HD bool resolved_wall_channel() const {
        return hard_wall_channel() && channel_height > 0;
    }
    PF3D_HD bool wall_repulsion_active() const {
        return resolved_wall_channel() && wall_kappa > 0.0;
    }
    PF3D_HD int accessible_channel_height() const {
        return resolved_wall_channel() ? channel_height : Nz;
    }
    PF3D_HD int minimum_domain_edge() const {
        const int xy = Nx < Ny ? Nx : Ny;
        return hard_wall_channel() ? xy : (xy < Nz ? xy : Nz);
    }
    PF3D_HD double volume0() const {
        return substrate_slab() ? target_slab_volume(target_radius)
                                : target_sphere_volume(target_radius);
    }
    PF3D_HD double bulk() const { return bulk_coeff(lambda); }
    PF3D_HD double interaction() const { return interaction_coeff(kappa, lambda); }
    PF3D_HD double motility() const { return motility_coeff(kappa, xi, lambda); }
    PF3D_HD double wall_interaction() const {
        return interaction_coeff(wall_kappa, lambda);
    }
    PF3D_HD double wall_motility() const {
        return motility_coeff(wall_kappa, xi, lambda);
    }
    PF3D_HD double volume() const { return volume_coeff(mu, volume0()); }
    PF3D_HD double dV() const { return dx * dy * dz; }
    PF3D_HD double realized_rho() const {
        const double lateral = static_cast<double>(Nx) * Ny;
        if (!(lateral > 0.0) || num_cells <= 0) return 0.0;
        if (substrate_slab())
            return static_cast<double>(num_cells) *
                   target_slab_footprint_area(target_radius) / lateral;
        const int height = hard_wall_channel() ? channel_height : Nz;
        return height > 0
            ? static_cast<double>(num_cells) *
                  target_sphere_volume(target_radius) /
                  (lateral * static_cast<double>(height))
            : 0.0;
    }
    double p_tumble() const { return -std::expm1(-dt / tau); }
    long long aging_steps() const {
        const double steps = aging_time / dt + 0.5;
        return std::isfinite(steps) && steps >= 0.0 && steps <= 9.0e18
            ? static_cast<long long>(steps) : -1LL;
    }
    int brick_edge() const {
        return automatic_brick_edge(target_radius, lambda);
    }
    long long total_steps() const {
        const double steps = t_end / dt + 0.5;
        return std::isfinite(steps) && steps >= 0.0 && steps <= 9.0e18
            ? static_cast<long long>(steps) : -1LL;
    }
};

inline bool validate(const SimParams3D& p, const char** error = nullptr) {
    auto fail = [error](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (!(p.dx == 1.0 && p.dy == 1.0 && p.dz == 1.0))
        return fail("dx, dy, and dz must all equal 1");
    if (p.periodic_xyz()) {
        if (p.Nx <= 0 || p.Nx != p.Ny || p.Nx != p.Nz)
            return fail("the periodic domain must be a positive cube");
    } else if (p.substrate_slab() || p.hard_wall_channel()) {
        if (p.Nx <= 0 || p.Ny != p.Nx || p.Nz <= 0)
            return fail("bounded-z geometry must have positive Nx=Ny and Nz");
    } else {
        return fail("boundary flags must select periodic XYZ, substrate slab, or hard-wall channel");
    }
    if (p.hard_wall_channel()) {
        if (p.channel_height <= 0 || p.channel_padding <= 0 ||
            !(p.wall_width > 0.0) || !std::isfinite(p.wall_width) ||
            !(p.wall_kappa > 0.0) || !std::isfinite(p.wall_kappa))
            return fail("resolved channel wall parameters must be positive");
        if (!(p.kappa > 0.0))
            return fail("repulsive channel walls require positive cell-cell kappa");
        const double wall_ratio = p.wall_kappa / p.kappa;
        const float wall_ratio_f = static_cast<float>(wall_ratio);
        if (!std::isfinite(wall_ratio) ||
            !std::isfinite(wall_ratio_f) || !(wall_ratio_f > 0.0f))
            return fail("wall_kappa/kappa is not representable in binary32");
        const int minimum_padding = channel_wall_padding(p.wall_width);
        if (minimum_padding <= 0 || p.channel_padding < minimum_padding)
            return fail("channel padding must be at least ceil(3*wall_width)");
        const std::int64_t allocated_height =
            static_cast<std::int64_t>(p.channel_height) +
            2 * static_cast<std::int64_t>(p.channel_padding);
        if (allocated_height != p.Nz)
            return fail("channel Nz must equal H + 2*padding");
        const double diameter = std::ceil(2.0 * p.target_radius);
        if (!std::isfinite(diameter) || p.channel_height < diameter)
            return fail("accessible channel height is smaller than one cell diameter");
    } else if (p.channel_height != 0 || p.channel_padding != 0 ||
               p.wall_kappa != 0.0 || p.wall_width != 0.0) {
        return fail("wall parameters require the hard-wall channel geometry");
    }
    if (p.num_cells <= 0) return fail("num_cells must be positive");
    if (!std::isfinite(p.rho) || !(p.rho > 0.0 && p.rho < 1.0))
        return fail("rho must be finite and lie in (0,1)");
    if (!std::isfinite(p.dt) || !std::isfinite(p.t_end) ||
        !(p.dt > 0.0) || !(p.t_end >= 0.0) || p.total_steps() < 0)
        return fail("dt and t_end do not define a valid step count");
    if (!std::isfinite(p.aging_time) ||
        !(p.aging_time >= 0.0) || p.aging_steps() < 0)
        return fail("aging_time does not define a valid step count");
    if (!std::isfinite(p.lambda) || !std::isfinite(p.target_radius) ||
        !(p.lambda > 0.0) || !(p.target_radius > 0.0))
        return fail("lambda and target radius must be finite and positive");
    if (!std::isfinite(p.kappa) || !std::isfinite(p.mu) ||
        !std::isfinite(p.xi) || p.kappa < 0.0 || p.mu < 0.0 ||
        !(p.xi > 0.0))
        return fail("kappa and mu must be non-negative and xi positive");
    if (!std::isfinite(p.tau) || !std::isfinite(p.v_A) ||
        !std::isfinite(p.v_A_sigma) || !(p.tau > 0.0) || p.v_A < 0.0 ||
        p.v_A_sigma < 0.0)
        return fail("tau must be positive and motility parameters non-negative");
    if (!std::isfinite(p.gamma_normal) ||
        !std::isfinite(p.gamma_cancer) ||
        !std::isfinite(p.cancer_fraction) ||
        !(p.gamma_normal > 0.0) || !(p.gamma_cancer > 0.0) ||
        !(p.cancer_fraction >= 0.0 && p.cancer_fraction <= 1.0))
        return fail("gamma values and cancer fraction are invalid");
    const int edge = p.brick_edge();
    if (edge <= 0 || p.minimum_domain_edge() <= edge)
        return fail("the automatic brick must fit strictly inside the domain");
    const double gamma_max = p.gamma_normal > p.gamma_cancer
        ? p.gamma_normal : p.gamma_cancer;
    if (p.dt * gamma_max * kLapRowSumBound >= 1.0)
        return fail("explicit-Euler Laplacian stability bound is violated");
    if (!std::isfinite(p.bulk()) || p.dt * gamma_max * p.bulk() >= 1.0)
        return fail("explicit-Euler bulk-reaction timestep bound is violated");
    if (!std::isfinite(p.interaction()) || !std::isfinite(p.wall_interaction()))
        return fail("cell or wall interaction coefficient is not finite");
    if (p.dt * p.wall_interaction() >= 1.0)
        return fail("explicit-Euler wall-repulsion timestep bound is violated");
    if (p.dt * (2.0 * p.mu) >= 1.0)
        return fail("explicit-Euler volume-restoring timestep bound is violated");
    const double six_sigma_speed = p.v_A_sigma == 0.0
        ? p.v_A : p.v_A * std::exp(6.0 * p.v_A_sigma);
    if (!std::isfinite(six_sigma_speed) || p.dt * six_sigma_speed >= 1.0)
        return fail("active-speed timestep bound is violated");
    if (p.kappa > 0.0) {
        const double ratio = p.interaction() / p.motility();
        if (!(std::fabs(ratio - p.xi) <= 1.0e-9 * p.xi))
            return fail("interaction/motility does not equal xi");
    }
    std::size_t bytes = 0;
    if (!checked_total_field_bytes(p.num_cells, edge, p.Nx, p.Ny, p.Nz,
                                   2, 3, &bytes))
        return fail("field memory-size calculation overflowed");
    if (error) *error = nullptr;
    return true;
}

}  // namespace pf3d
