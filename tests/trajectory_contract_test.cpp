#include "trajectory.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

std::filesystem::path unique_directory() {
    const auto stamp = std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 100; ++attempt) {
        const std::filesystem::path candidate = root /
            ("pf2d-trajectory-" + std::to_string(stamp) + "-" +
             std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error))
            return candidate;
    }
    return {};
}

bool write_frame(std::FILE* file, double time, bool reverse) {
    for (int row = 0; row < 2; ++row) {
        const int id = reverse ? 1 - row : row;
        if (std::fprintf(
                file,
                "%.17g %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                time, id, 10.0 + id, 20.0 + id, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.01, 1.0, 7542.0) < 0)
            return false;
    }
    return std::fflush(file) == 0;
}

}  // namespace

int main() {
    constexpr long long cadence = 5;
    pf::SimParams params{};
    params.Nx = params.Ny = 400;
    params.num_cells = 2;
    params.dt = 0.01;
    params.seed = 0x12345678abcdef01ULL;
    params.polarity_seed = 0xfedcba9876543210ULL;
    params.initialization_hash = 0x13579bdf2468ace0ULL;

    const std::filesystem::path directory = unique_directory();
    expect(!directory.empty(), "unique temporary directory created");
    if (directory.empty()) return 1;
    const std::filesystem::path trajectory = directory / "trajectory.txt";
    const std::filesystem::path partial = directory / "partial.txt";
    const std::filesystem::path reordered = directory / "reordered.txt";
    const std::filesystem::path reverse_first = directory / "reverse-first.txt";
    const std::filesystem::path precise = directory / "precise-time.txt";

    pf::TrajectoryAppendInfo2D info{};
    std::FILE* file = nullptr;
    expect(pf::open_trajectory_2d(trajectory.string(), params, params.Nx, 0,
                                 cadence, &info, &file),
           "new trajectory opens");
    expect(file != nullptr && info.frames == 0 && info.last_step == -1,
           "new trajectory state");
    if (file) {
        expect(write_frame(file, 0.1, false), "write first complete frame");
        expect(std::fclose(file) == 0, "close first trajectory handle");
    }
    file = nullptr;

    expect(pf::open_trajectory_2d(trajectory.string(), params, params.Nx, 10,
                                 cadence, &info, &file),
           "compatible complete trajectory reopens");
    expect(info.frames == 1 && info.last_step == 10,
           "existing frame and step recovered");
    if (file) std::fclose(file);
    file = nullptr;

    pf::SimParams changed = params;
    changed.polarity_seed ^= 1ULL;
    expect(!pf::open_trajectory_2d(trajectory.string(), changed, changed.Nx, 10,
                                  cadence, &info, &file) && file == nullptr,
           "changed tumble stream is rejected");

    expect(!pf::open_trajectory_2d(trajectory.string(), params, params.Nx, 10,
                                  cadence + 1, &info, &file) && file == nullptr,
           "changed trajectory cadence is rejected");

    expect(pf::open_trajectory_2d(partial.string(), params, params.Nx, 0,
                                 cadence, &info, &file),
           "partial fixture opens initially");
    if (file) {
        expect(std::fprintf(file,
                            "0.100000 0 10 20 0 0 1 0 0 0.01 1 7542\n") > 0,
               "write partial frame");
        expect(std::fclose(file) == 0, "close partial fixture");
    }
    file = nullptr;
    expect(!pf::open_trajectory_2d(partial.string(), params, params.Nx, 10,
                                  cadence, &info, &file) && file == nullptr,
           "partial frame is rejected");

    expect(pf::open_trajectory_2d(reordered.string(), params, params.Nx, 0,
                                 cadence, &info, &file),
           "ordering fixture opens initially");
    if (file) {
        expect(write_frame(file, 0.1, false), "write ordering fixture frame one");
        expect(write_frame(file, 0.2, true), "write ordering fixture frame two");
        expect(std::fclose(file) == 0, "close ordering fixture");
    }
    file = nullptr;
    expect(!pf::open_trajectory_2d(reordered.string(), params, params.Nx, 20,
                                  cadence, &info, &file) && file == nullptr,
           "cell-order change is rejected");

    expect(pf::open_trajectory_2d(reverse_first.string(), params, params.Nx, 0,
                                 cadence, &info, &file),
           "reverse-first fixture opens initially");
    if (file) {
        expect(write_frame(file, 0.1, true), "write reversed first frame");
        expect(std::fclose(file) == 0, "close reverse-first fixture");
    }
    file = nullptr;
    expect(!pf::open_trajectory_2d(reverse_first.string(), params, params.Nx,
                                  10, cadence, &info, &file) && file == nullptr,
           "noncanonical first-frame order is rejected");

    pf::SimParams precise_params = params;
    precise_params.dt = 0.00123456789;
    expect(pf::open_trajectory_2d(precise.string(), precise_params,
                                  precise_params.Nx, 0, cadence, &info, &file),
           "non-decimal time fixture opens initially");
    if (file) {
        expect(write_frame(file, precise_params.dt, false),
               "write round-trip-precision frame time");
        expect(std::fclose(file) == 0, "close precise-time fixture");
    }
    file = nullptr;
    expect(pf::open_trajectory_2d(precise.string(), precise_params,
                                  precise_params.Nx, 1, cadence, &info, &file),
           "round-trip time reconstructs its exact step");
    if (file) std::fclose(file);
    file = nullptr;

    for (const auto& path : {
             trajectory, partial, reordered, reverse_first, precise}) {
        std::error_code error;
        expect(std::filesystem::remove(path, error) && !error,
               "remove known fixture file");
    }
    std::error_code error;
    expect(std::filesystem::is_empty(directory, error) && !error,
           "temporary directory is empty before removal");
    expect(std::filesystem::remove(directory, error) && !error,
           "remove verified-empty temporary directory");

    if (failures != 0) {
        std::fprintf(stderr, "%d trajectory contract test(s) failed\n", failures);
        return 1;
    }
    std::printf("2D trajectory contract: PASS\n");
    return 0;
}
