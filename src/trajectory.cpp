#include "../include/trajectory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>

namespace pf {
namespace {

constexpr std::size_t kLineCapacity = 4096;

bool read_line(std::FILE* file, char* line, std::size_t capacity,
               bool* line_too_long) {
    if (!std::fgets(line, static_cast<int>(capacity), file)) return false;
    std::size_t length = std::strlen(line);
    if (length > 0 && line[length - 1] == '\n') {
        line[--length] = '\0';
        if (length > 0 && line[length - 1] == '\r') line[--length] = '\0';
    } else if (!std::feof(file)) {
        *line_too_long = true;
        return false;
    }
    return true;
}

void close_on_error(std::FILE** file) {
    if (*file) std::fclose(*file);
    *file = nullptr;
}

}  // namespace

std::string trajectory_metadata_2d(const SimParams& p, int side,
                                   long long trajectory_interval) {
    std::array<char, 2048> line{};
    const double realized_rho = side > 0
        ? static_cast<double>(p.num_cells) * p.area0() /
              (static_cast<double>(side) * side)
        : std::numeric_limits<double>::quiet_NaN();
    const int written = std::snprintf(
        line.data(), line.size(),
        "# trajectory_schema=1 dim=2 model=run_tumble N=%d Lx=%d Ly=%d "
        "dx=%.17g dy=%.17g dt=%.17g rho_target=%.17g "
        "rho_realized=%.17g lambda=%.17g R=%.17g kappa=%.17g mu=%.17g "
        "xi=%.17g tau=%.17g v_A=%.17g v_A_sigma=%.17g "
        "gamma_normal=%.17g gamma_soft=%.17g soft_fraction=%.17g "
        "soft_assignment=lowest_global_ids seed=%llu polarity_seed=%llu "
        "initialization_hash=%016llx full_moment=%d perim_offset=%.17g "
        "trajectory_interval=%lld",
        p.num_cells, side, side, p.dx, p.dy, p.dt, p.rho, realized_rho,
        p.lambda, p.target_radius, p.kappa, p.mu, p.xi, p.tau, p.v_A,
        p.v_A_sigma, p.gamma_normal, p.gamma_cancer, p.cancer_fraction,
        p.seed, p.polarity_stream(), p.initialization_hash,
        p.full_moment_every, kPi / interface_k(p.lambda),
        trajectory_interval);
    if (written < 0 || static_cast<std::size_t>(written) >= line.size())
        return {};
    return std::string(line.data(), static_cast<std::size_t>(written));
}

bool open_trajectory_2d(const std::string& path, const SimParams& params,
                        int side, long long current_step,
                        long long trajectory_interval,
                        TrajectoryAppendInfo2D* info, std::FILE** file) {
    if (!info || !file || path.empty() || side <= 0 || current_step < 0 ||
        trajectory_interval <= 0 || params.num_cells <= 0 ||
        !std::isfinite(params.dt) || params.dt <= 0.0)
        return false;
    *info = {};
    *file = nullptr;

    const std::string metadata = trajectory_metadata_2d(
        params, side, trajectory_interval);
    if (metadata.empty()) {
        std::fprintf(stderr, "[error] 2D trajectory metadata is too long\n");
        return false;
    }

    const std::filesystem::path output(path);
    std::error_code error;
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path(), error);
    if (error) {
        std::fprintf(stderr, "[error] cannot create trajectory directory: %s\n",
                     error.message().c_str());
        return false;
    }

    *file = std::fopen(path.c_str(), "a+");
    if (!*file) {
        std::fprintf(stderr, "[error] cannot open %s for append\n", path.c_str());
        return false;
    }
    std::rewind(*file);
    const int first_byte = std::fgetc(*file);
    if (first_byte == EOF && std::ferror(*file)) {
        std::fprintf(stderr, "[error] cannot inspect trajectory %s\n",
                     path.c_str());
        close_on_error(file);
        return false;
    }
    const bool empty = first_byte == EOF;
    std::clearerr(*file);
    if (empty) {
        // C update streams require a positioning operation when switching
        // from the read probe to writing.
        if (std::fseek(*file, 0, SEEK_END) != 0) {
            std::fprintf(stderr, "[error] cannot seek trajectory %s\n",
                         path.c_str());
            close_on_error(file);
            return false;
        }
        const bool written =
            std::fprintf(*file, "%s\n%s\n%s\n", kTrajectoryTitle2D,
                         kTrajectoryColumns2D, metadata.c_str()) >= 0 &&
            std::fflush(*file) == 0 && !std::ferror(*file);
        if (!written) {
            std::fprintf(stderr,
                         "[error] failed to write trajectory header to %s\n",
                         path.c_str());
            close_on_error(file);
            return false;
        }
        return true;
    }

    std::rewind(*file);
    std::array<char, kLineCapacity> line{};
    bool line_too_long = false;
    const char* expected[3] = {
        kTrajectoryTitle2D, kTrajectoryColumns2D, metadata.c_str()};
    for (const char* expected_line : expected) {
        if (!read_line(*file, line.data(), line.size(), &line_too_long) ||
            std::strcmp(line.data(), expected_line) != 0) {
            std::fprintf(stderr,
                "[error] refusing to append: trajectory metadata does not "
                "match this run\n");
            close_on_error(file);
            return false;
        }
    }

    std::uint64_t rows = 0;
    double frame_time = 0.0;
    double previous_frame_time = -1.0;
    while (read_line(*file, line.data(), line.size(), &line_too_long)) {
        if (line[0] == '\0') {
            std::fprintf(stderr, "[error] blank line in trajectory payload\n");
            close_on_error(file);
            return false;
        }
        double values[11]{};
        long long global_id = 0;
        char extra = '\0';
        const int parsed = std::sscanf(
            line.data(),
            "%lf %lld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %c",
            &values[0], &global_id, &values[1], &values[2], &values[3],
            &values[4], &values[5], &values[6], &values[7], &values[8],
            &values[9], &values[10], &extra);
        bool valid = parsed == 12 &&
            global_id >= std::numeric_limits<std::int32_t>::min() &&
            global_id <= std::numeric_limits<std::int32_t>::max();
        for (double value : values) valid = valid && std::isfinite(value);
        if (!valid) {
            std::fprintf(stderr, "[error] malformed trajectory row %llu\n",
                         static_cast<unsigned long long>(rows + 1));
            close_on_error(file);
            return false;
        }

        const std::size_t within = static_cast<std::size_t>(
            rows % static_cast<std::uint64_t>(params.num_cells));
        if (within == 0) {
            frame_time = values[0];
            if (rows != 0 && !(frame_time > previous_frame_time)) {
                std::fprintf(stderr,
                    "[error] trajectory frame times are not strictly increasing\n");
                close_on_error(file);
                return false;
            }
        } else if (values[0] != frame_time) {
            std::fprintf(stderr,
                         "[error] trajectory contains a frame with mixed times\n");
            close_on_error(file);
            return false;
        }
        if (global_id != static_cast<long long>(within)) {
            std::fprintf(stderr,
                         "[error] trajectory cells are not in canonical ID order\n");
            close_on_error(file);
            return false;
        }
        ++rows;
        if (rows % static_cast<std::uint64_t>(params.num_cells) == 0)
            previous_frame_time = frame_time;
    }
    if (line_too_long || std::ferror(*file) ||
        rows % static_cast<std::uint64_t>(params.num_cells) != 0) {
        std::fprintf(stderr,
                     "[error] refusing to append to a partial trajectory frame\n");
        close_on_error(file);
        return false;
    }

    info->frames = static_cast<long long>(
        rows / static_cast<std::uint64_t>(params.num_cells));
    if (info->frames > 0) {
        const double step_value = previous_frame_time / params.dt;
        if (!std::isfinite(step_value) || step_value < 0.0 ||
            step_value > 9.0e18) {
            std::fprintf(stderr, "[error] trajectory time is outside step range\n");
            close_on_error(file);
            return false;
        }
        const long long last_step = std::llround(step_value);
        const double reconstructed = static_cast<double>(last_step) * params.dt;
        const double tolerance = 1.0e-10 *
            std::max(1.0, std::fabs(previous_frame_time));
        if (std::fabs(reconstructed - previous_frame_time) > tolerance ||
            last_step > current_step) {
            std::fprintf(stderr,
                         "[error] trajectory extends beyond the initialized state\n");
            close_on_error(file);
            return false;
        }
        info->last_step = last_step;
    }

    std::clearerr(*file);
    if (std::fseek(*file, 0, SEEK_END) != 0) {
        close_on_error(file);
        return false;
    }
    return true;
}

}  // namespace pf
