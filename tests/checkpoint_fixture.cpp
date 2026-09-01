#include "checkpoint_format.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool write_bytes(std::FILE* file, const void* data, std::size_t bytes) {
    return std::fwrite(data, 1, bytes, file) == bytes;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::FILE* file = std::fopen(argv[1], "wb");
    if (!file) return 1;

    ckpt::FixedPrefix prefix{};
    prefix.magic = ckpt::MAGIC;
    prefix.version = ckpt::CHECKPOINT_FORMAT;
    prefix.step = 10;
    prefix.cur_time = 0.1;
    prefix.num_cells_local = 2;
    prefix.save_interval = 5;
    prefix.trajectory_samples = 2;
    prefix.sp_sz = sizeof(ckpt::CheckpointParamsRecord);

    ckpt::CheckpointParamsRecord params{};
    params.Nx = params.Ny = 400;
    params.dx = params.dy = 1.0;
    params.dt = 0.01;
    params.t_end = 1.0;
    params.rho = 0.9;
    params.lambda = 7.0;
    params.gamma_normal = 1.0;
    params.gamma_soft = 0.35;
    params.soft_fraction = 0.5;
    params.kappa = 10.0;
    params.target_radius = 49.0;
    params.mu = 1.0;
    params.v_A = 0.0;
    params.v_A_sigma = 0.0;
    params.xi = 1500.0;
    params.tau = 10000.0;
    params.seed = 0x12345678abcdef01ULL;
    params.polarity_seed = 0xfedcba9876543210ULL;
    params.initialization_hash = 0x13579bdf2468ace0ULL;
    params.print_interval = 100;
    params.full_moment_every = 37;
    params.verify_every = 913;
    params.save_interval = prefix.save_interval;
    params.trajectory_samples = prefix.trajectory_samples;
    params.trajectory_interval = 5;

    const std::int32_t tile_pitch = ckpt::CELL_TILE_PITCH;
    const ckpt::RankTrailer rank{1, 0, 2};
    bool ok = write_bytes(file, &prefix, sizeof(prefix)) &&
              write_bytes(file, &params, sizeof(params)) &&
              write_bytes(file, &tile_pitch, sizeof(tile_pitch)) &&
              write_bytes(file, &rank, sizeof(rank));

    for (int id = 0; ok && id < 2; ++id) {
        ckpt::CellRecordHeader cell{};
        cell.cell_id = id;
        cell.origin_x = 270 + id;
        cell.origin_y = 280 + id;
        cell.shape_class = 0;
        cell.cx = 10.25f + static_cast<float>(id);
        cell.cy = 20.5f + static_cast<float>(id);
        cell.volume_moment = 1.0;
        cell.moment_x = 76.25;
        cell.moment_y = 76.5;
        cell.perimeter = 12.0 + id;
        cell.support_lo_x = cell.support_lo_y = 76;
        cell.support_hi_x = cell.support_hi_y = 77;
        cell.phi_max = std::sqrt(0.375f);
        std::vector<float> phi(
            static_cast<std::size_t>(tile_pitch) * tile_pitch, 0.0f);
        const int x0 = 64 + cell.support_lo_x;
        const int y0 = 64 + cell.support_lo_y;
        phi[static_cast<std::size_t>(y0) * tile_pitch + x0] =
            std::sqrt(0.375f);
        phi[static_cast<std::size_t>(y0 + 1) * tile_pitch + x0] =
            std::sqrt(0.375f);
        phi[static_cast<std::size_t>(y0) * tile_pitch + x0 + 1] =
            std::sqrt(0.125f);
        phi[static_cast<std::size_t>(y0 + 1) * tile_pitch + x0 + 1] =
            std::sqrt(0.125f);
        ok = write_bytes(file, &cell, sizeof(cell)) &&
             write_bytes(file, phi.data(), phi.size() * sizeof(float));
    }

    const std::array<float, 2> polarity{0.1f, 0.2f};
    const std::array<float, 2> gamma{1.0f, 0.35f};
    const std::array<float, 2> speed{0.0f, 0.0f};
    const std::array<float, 2> radius{49.0f, 49.0f};
    const auto sidecar = [&](std::uint32_t magic,
                             const std::array<float, 2>& values) {
        const ckpt::SidecarBlockHeader header{magic, 2};
        return write_bytes(file, &header, sizeof(header)) &&
               write_bytes(file, values.data(), sizeof(values));
    };
    ok = ok && sidecar(ckpt::MAGIC_POLR, polarity) &&
         sidecar(ckpt::MAGIC_GAMA, gamma) &&
         sidecar(ckpt::MAGIC_VA_A, speed) &&
         sidecar(ckpt::MAGIC_RADI, radius);

    if (std::fclose(file) != 0) ok = false;
    return ok ? 0 : 1;
}
