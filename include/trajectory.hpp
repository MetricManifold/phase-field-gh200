#pragma once

#include "params.cuh"

#include <cstdio>
#include <string>

namespace pf {

inline constexpr const char* kTrajectoryTitle2D = "# Trajectory data";
inline constexpr const char* kTrajectoryColumns2D =
    "# Format: time cell_id x y vx vy px py theta v_A_i L_n volume";

// The metadata line is an exact append contract. It contains every model
// parameter that can change the rows or subsequent dynamics, plus the accepted
// initial-centre fingerprint.
std::string trajectory_metadata_2d(const SimParams& params, int side,
                                   long long trajectory_interval);

struct TrajectoryAppendInfo2D {
    long long frames = 0;
    long long last_step = -1;
};

// Open a trajectory without truncation. A nonempty file is accepted only when
// its metadata, columns, complete-frame structure, canonical cell-ID order,
// and final time are compatible with the initialized state.
bool open_trajectory_2d(const std::string& path, const SimParams& params,
                        int side, long long current_step,
                        long long trajectory_interval,
                        TrajectoryAppendInfo2D* info, std::FILE** file);

}  // namespace pf
