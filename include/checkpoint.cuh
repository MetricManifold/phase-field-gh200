#pragma once
// Checkpoint I/O and conversion to the simulator's active-window layout.
// The on-disk schema is defined in common/checkpoint_format.h. Writers store
// one fixed square phase-field tile and the exact adaptive state per cell.
// Readers validate that the stored support fits its class and never clip it.

#include "kernels.cuh"
#include "params.cuh"

#include "checkpoint_format.h"

#include <string>
#include <vector>

namespace pf {

constexpr int kCkptDims = 2;  // checkpoint coordinate dimensions

// Tracks parameters explicitly set on the command line. A resumed run first
// loads checkpoint parameters, then replaces only the marked fields.
struct SimOverrides {
    bool t_end = false, dt = false, v_A = false, tau = false;
    bool gamma = false, gamma_cancer = false, cancer_fraction = false;
    bool kappa = false, mu = false, xi = false, lambda = false;
    bool target_radius = false, v_A_sigma = false;
    bool seed = false, polarity_seed = false;
    bool print_interval = false, full_moment = false;
    bool verify_every = false;
    // These geometry changes are rejected on resume because stored coordinates
    // refer to the checkpoint's domain.
    bool num_cells = false, rho = false;

    // A changed assignment policy supersedes the corresponding sidecar.
    bool gamma_policy_changed() const {
        return gamma || gamma_cancer || cancer_fraction;
    }
    bool v_A_policy_changed() const { return v_A || v_A_sigma; }

    // Apply only explicitly supplied command-line values.
    void apply(SimParams& p, const SimParams& cli) const;
};

// Host-side checkpoint state in the active tile layout.
struct CkptCell {
    int32_t global_id = 0;
    // Global origin of the active window after periodic wrapping.
    int32_t origin[kCkptDims] = {0, 0};
    uint8_t cls   = 0;
    float   gamma = 0.0f, v_A = 0.0f, R_tgt = 0.0f, theta = 0.0f;
    float   vx = 0.0f, vy = 0.0f;
    double  volume_moment = 0.0, moment_x = 0.0, moment_y = 0.0;
    double  perimeter = 0.0;
    int32_t support_lo_x = 0, support_hi_x = -1;
    int32_t support_lo_y = 0, support_hi_y = -1;
    uint32_t promote_ctr = 0;
    float phi_max = 0.0f;
};

struct CheckpointData {
    SimParams params{};          // adopted from the file, before CLI overrides
    long long step    = 0;
    double    t       = 0.0;
    int       n       = 0;
    int       file_tile_pitch = 0;
    int32_t   num_ranks = 1, rank_id = 0, n_global = 0;
    int       trajectory_samples = 0;
    long long trajectory_interval = 0;
    std::vector<CkptCell> cells;      // n entries
    std::vector<float>    phi;        // n * kTileArea, exact fixed-size tiles
};

// Read and validate a checkpoint without changing its stored state.
bool checkpoint_read(const std::string& path, CheckpointData* out);

// Retain checkpoint per-cell values unless an explicit assignment policy
// requests a replacement. A passive-aging speed field is reactivated when the
// requested run has nonzero activity.
void resolve_per_cell_scalars(const SimParams& p, const SimOverrides& ov,
                              CheckpointData* d);

// Non-owning state passed to the writer. d_phi is a device pointer; the caller
// must synchronize its producing stream. Bounded staging keeps host memory
// independent of the number of cells.
struct CheckpointWriteView {
    const SimParams* p    = nullptr;
    long long        step = 0;
    double           t    = 0.0;
    int              N    = 0;
    int              L    = 0;
    const CellState* cell = nullptr;   // host copy, N entries
    const uint8_t*   cls  = nullptr;   // host copy, N entries
    const float*     d_phi = nullptr;  // device, N * kTileArea floats
    int              trajectory_samples = 0;
    long long        trajectory_interval = 0;
    int              save_interval      = 0;
};

// Copy the device field once and write each path through a temporary file and
// rename, so rolling and tagged checkpoints can share the same staging pass.
bool checkpoint_write(const CheckpointWriteView& v,
                      const std::vector<std::string>& paths);

}  // namespace pf
