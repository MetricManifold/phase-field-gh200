#include "checkpoint.cuh"
#include "trajectory.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", operation,
                 cudaGetErrorString(status));
    ++failures;
    return false;
}

std::filesystem::path unique_directory() {
    const auto stamp = std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 100; ++attempt) {
        const std::filesystem::path candidate = root /
            ("pf2d-checkpoint-" + std::to_string(stamp) + "-" +
             std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error))
            return candidate;
    }
    return {};
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        std::printf("SKIP: checkpoint staging requires a CUDA device\n");
        return 77;
    }

    const std::filesystem::path directory = unique_directory();
    expect(!directory.empty(), "unique temporary directory created");
    if (directory.empty()) return 1;
    const std::filesystem::path checkpoint = directory / "fixture.bin";

    pf::SimParams params{};
    params.Nx = params.Ny = 512;
    params.num_cells = 1;
    params.seed = 0x12345678abcdef01ULL;
    params.polarity_seed = 0xfedcba9876543210ULL;
    params.initialization_hash = 0x13579bdf2468ace0ULL;
    params.rho = 0.8123456789012345;
    params.gamma_cancer = 0.43;
    params.cancer_fraction = 0.25;
    params.v_A_sigma = 0.17;
    params.full_moment_every = 37;
    params.verify_every = 913;

    pf::CellState cell{};
    cell.global_id = 0;
    cell.gx0 = 64;
    cell.gy0 = 96;
    cell.gamma = static_cast<float>(params.gamma_normal);
    cell.v_A = static_cast<float>(params.v_A);
    cell.R_tgt = static_cast<float>(params.target_radius);
    cell.theta = 0.75f;
    cell.V = 16.0;
    cell.Cx = 1240.0;
    cell.Cy = 1240.0;
    cell.perim = 12.5;
    cell.cls = static_cast<std::uint8_t>(pf::kClassRound);
    cell.cls_written[0] = cell.cls;
    cell.cls_written[1] = cell.cls;
    cell.bb_lo_x = cell.bb_lo_y = 76;
    cell.bb_hi_x = cell.bb_hi_y = 79;
    cell.promote_ctr = 17;
    cell.phi_max = 1.0f;
    const std::uint8_t cell_class = static_cast<std::uint8_t>(pf::kClassRound);

    std::vector<float> host_phi(static_cast<std::size_t>(pf::kTileArea), 0.0f);
    for (int y = 140; y < 144; ++y)
        for (int x = 140; x < 144; ++x)
            host_phi[static_cast<std::size_t>(y) * pf::kTilePitch + x] = 1.0f;

    float* device_phi = nullptr;
    if (!cuda_ok(cudaMalloc(&device_phi, host_phi.size() * sizeof(float)),
                 "cudaMalloc(phi)") ||
        !cuda_ok(cudaMemcpy(device_phi, host_phi.data(),
                            host_phi.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(phi)")) {
        if (device_phi) cudaFree(device_phi);
        return 1;
    }

    pf::CheckpointWriteView view{};
    view.p = &params;
    view.step = 123;
    view.t = 1.23;
    view.N = 1;
    view.L = params.Nx;
    view.cell = &cell;
    view.cls = &cell_class;
    view.d_phi = device_phi;
    view.trajectory_samples = 10;
    view.trajectory_interval = 11;
    view.save_interval = 20;
    expect(pf::checkpoint_write(view, {checkpoint.string()}),
           "checkpoint writer accepts full-width seeds");

    pf::CheckpointData loaded{};
    expect(pf::checkpoint_read(checkpoint.string(), &loaded),
           "checkpoint reader loads writer output");
    expect(loaded.params.seed == params.seed,
           "placement seed round-trips without truncation");
    expect(loaded.params.polarity_seed == params.polarity_stream(),
           "resolved tumble stream round-trips without truncation");
    expect(loaded.params.initialization_hash == params.initialization_hash,
           "initialization fingerprint round-trips");
    expect(loaded.params.rho == params.rho &&
               loaded.params.gamma_cancer == params.gamma_cancer &&
               loaded.params.cancer_fraction == params.cancer_fraction &&
               loaded.params.v_A_sigma == params.v_A_sigma,
           "model and cohort metadata round-trip");
    expect(loaded.params.full_moment_every == params.full_moment_every &&
               loaded.params.verify_every == params.verify_every,
           "measurement and verification cadences round-trip");
    expect(pf::trajectory_metadata_2d(
               loaded.params, loaded.params.Nx,
               loaded.trajectory_interval) ==
               pf::trajectory_metadata_2d(
                   params, params.Nx, view.trajectory_interval),
           "trajectory append metadata survives checkpoint round-trip");
    expect(loaded.step == view.step && loaded.n == 1,
           "checkpoint state metadata round-trips");
    expect(loaded.trajectory_samples == view.trajectory_samples &&
               loaded.trajectory_interval == view.trajectory_interval,
           "trajectory schedule round-trips");
    expect(loaded.cells.size() == 1 && loaded.cells[0].theta == cell.theta,
           "per-cell polarity sidecar round-trips");
    expect(loaded.cells.size() == 1 &&
               loaded.cells[0].perimeter == cell.perim,
           "cached perimeter round-trips");
    expect(loaded.cells.size() == 1 &&
               loaded.cells[0].cls == cell.cls &&
               loaded.cells[0].promote_ctr == cell.promote_ctr &&
               loaded.cells[0].volume_moment == cell.V &&
               loaded.cells[0].moment_x == cell.Cx &&
               loaded.cells[0].moment_y == cell.Cy &&
               loaded.cells[0].support_lo_x == cell.bb_lo_x &&
               loaded.cells[0].support_hi_x == cell.bb_hi_x &&
               loaded.cells[0].support_lo_y == cell.bb_lo_y &&
               loaded.cells[0].support_hi_y == cell.bb_hi_y &&
               loaded.cells[0].phi_max == cell.phi_max,
           "adaptive state and exact moments round-trip");

    cuda_ok(cudaFree(device_phi), "cudaFree(phi)");
    std::error_code error;
    expect(std::filesystem::remove(checkpoint, error) && !error,
           "remove known checkpoint fixture");
    expect(std::filesystem::is_empty(directory, error) && !error,
           "temporary directory is empty before removal");
    expect(std::filesystem::remove(directory, error) && !error,
           "remove verified-empty temporary directory");

    if (failures != 0) {
        std::fprintf(stderr, "%d checkpoint seed test(s) failed\n", failures);
        return 1;
    }
    std::printf("2D checkpoint seed contract: PASS\n");
    return 0;
}
