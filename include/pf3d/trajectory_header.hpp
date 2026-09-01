#pragma once

#include "params.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace pf3d {

// Trajectory metadata line. Appending to an existing file requires this
// string to match the file's first line exactly, so every token here is a
// continuation contract: a resumed leg whose physics, geometry, or
// reduction grouping differs - including the resolved base-measurement
// shard count - is refused instead of silently extending the file. Header
// text lives in this host-only header so tests can exercise the contract
// without a device.
inline std::string trajectory_header(const SimParams3D& p, int B,
                                     int promoted_measure_policy,
                                     int promoted_measure_auto_wave_ctas,
                                     int base_measure_shards,
                                     std::uint64_t trajectory_interval) {
    char text[1536]{};
    const char* reduction = promoted_measure_policy < 0
        ? "auto-wave" : promoted_measure_policy == 0
            ? "one-cta" : "fixed";
    int written = 0;
    if (p.substrate_slab()) {
        written = std::snprintf(
            text, sizeof(text),
            "# phase_field_gh200 trajectory schema=1 dim=3 geometry=substrate-slab "
            "boundary=periodic-xy N=%d Nx=%d Ny=%d Nz=%d B=%d "
            "z_coordinate=height-above-substrate substrate_z=0 "
            "lattice_substrate_face=%a top=dirichlet-zero top_z=%d translation=planar "
            "dx=%a dt=%a aging_time=%a rho_A_target=%a rho_A_realized=%a "
            "R=%a lambda=%a kappa=%a mu=%a xi=%a "
            "tau=%a v_A=%a v_A_sigma=%a gamma_normal=%a gamma_soft=%a "
            "soft_fraction=%a seed=%llu polarity_stream=%llu init_hash=%016llx "
            "promoted_measure_reduction=%s promoted_measure_policy=%d "
            "promoted_measure_auto_wave_ctas=%d base_measure_shards=%d "
            "trajectory_interval=%llu",
            p.num_cells, p.Nx, p.Ny, p.Nz, B, -0.5, p.Nz, p.dx, p.dt,
            p.aging_time, p.rho, p.realized_rho(), p.target_radius, p.lambda,
            p.kappa, p.mu,
            p.xi, p.tau, p.v_A, p.v_A_sigma, p.gamma_normal,
            p.gamma_cancer, p.cancer_fraction,
            static_cast<unsigned long long>(p.seed),
            static_cast<unsigned long long>(p.polarity_stream()),
            static_cast<unsigned long long>(p.initialization_hash),
            reduction, promoted_measure_policy,
            promoted_measure_auto_wave_ctas, base_measure_shards,
            static_cast<unsigned long long>(trajectory_interval));
    } else if (p.resolved_wall_channel()) {
        written = std::snprintf(
            text, sizeof(text),
            "# phase_field_gh200 trajectory schema=1 dim=3 geometry=resolved-wall-channel "
            "boundary=periodic-xy outer_array_boundary=homogeneous-neumann-z N=%d "
            "Nx=%d Ny=%d allocated_Nz=%d H=%d padding=%d B=%d "
            "z_coordinate=height-above-lower-wall lower_wall_z=0 upper_wall_z=%d "
            "wall_profile=tanh-union kappa_w=%a lambda_w=%a translation=three-dimensional "
            "dx=%a dt=%a aging_time=%a rho_V_target=%a rho_V_realized=%a "
            "R=%a lambda=%a kappa=%a mu=%a xi=%a "
            "tau=%a v_A=%a v_A_sigma=%a gamma_normal=%a gamma_soft=%a "
            "soft_fraction=%a seed=%llu polarity_stream=%llu init_hash=%016llx "
            "promoted_measure_reduction=%s promoted_measure_policy=%d "
            "promoted_measure_auto_wave_ctas=%d base_measure_shards=%d "
            "trajectory_interval=%llu",
            p.num_cells, p.Nx, p.Ny, p.Nz, p.channel_height,
            p.channel_padding, B, p.channel_height, p.wall_kappa,
            p.wall_width, p.dx, p.dt, p.aging_time, p.rho, p.realized_rho(),
            p.target_radius, p.lambda, p.kappa, p.mu, p.xi, p.tau, p.v_A,
            p.v_A_sigma, p.gamma_normal, p.gamma_cancer, p.cancer_fraction,
            static_cast<unsigned long long>(p.seed),
            static_cast<unsigned long long>(p.polarity_stream()),
            static_cast<unsigned long long>(p.initialization_hash),
            reduction, promoted_measure_policy,
            promoted_measure_auto_wave_ctas, base_measure_shards,
            static_cast<unsigned long long>(trajectory_interval));
    } else {
        written = std::snprintf(
            text, sizeof(text),
            "# phase_field_gh200 trajectory schema=1 dim=3 geometry=periodic-xyz "
            "N=%d L=%d B=%d "
            "dx=%a dt=%a aging_time=%a rho_V_target=%a rho_V_realized=%a "
            "R=%a lambda=%a kappa=%a mu=%a xi=%a "
            "tau=%a v_A=%a v_A_sigma=%a gamma_normal=%a gamma_soft=%a "
            "soft_fraction=%a seed=%llu polarity_stream=%llu init_hash=%016llx "
            "promoted_measure_reduction=%s promoted_measure_policy=%d "
            "promoted_measure_auto_wave_ctas=%d base_measure_shards=%d "
            "trajectory_interval=%llu",
            p.num_cells, p.Nx, B, p.dx, p.dt, p.aging_time, p.rho,
            p.realized_rho(),
            p.target_radius, p.lambda, p.kappa, p.mu, p.xi, p.tau, p.v_A,
            p.v_A_sigma, p.gamma_normal, p.gamma_cancer, p.cancer_fraction,
            static_cast<unsigned long long>(p.seed),
            static_cast<unsigned long long>(p.polarity_stream()),
            static_cast<unsigned long long>(p.initialization_hash),
            reduction, promoted_measure_policy,
            promoted_measure_auto_wave_ctas, base_measure_shards,
            static_cast<unsigned long long>(trajectory_interval));
    }
    return written > 0 && static_cast<std::size_t>(written) < sizeof(text)
        ? std::string(text, static_cast<std::size_t>(written))
        : std::string();
}

// Production append predicate: an existing trajectory may only be extended
// when its first two lines equal this run's expected metadata and column
// contract exactly. Every continuation token above - including the resolved
// base-measurement shard count - participates through the header string.
inline bool trajectory_metadata_compatible(const char* first_line,
                                           const char* second_line,
                                           const std::string& expected_header,
                                           const char* expected_columns) {
    return first_line != nullptr && second_line != nullptr &&
           expected_columns != nullptr && expected_header == first_line &&
           std::strcmp(second_line, expected_columns) == 0;
}

}  // namespace pf3d
