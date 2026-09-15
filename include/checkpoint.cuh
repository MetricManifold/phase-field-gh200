#pragma once
// Checkpoint I/O and conversion to the simulator's active-window layout.
// The on-disk schema is defined in common/checkpoint_format.h. Writers store
// one fixed square phase-field tile and the exact adaptive state per cell.
// Readers validate that the stored support fits its class and never clip it.

#include "kernels.cuh"
#include "params.cuh"

#include "checkpoint_format.h"

#include <string>
#include <utility>
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
    bool phase_field_mobility = false;
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
    float   M_pf = static_cast<float>(kPhaseFieldM0);
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
    bool      had_mobi = false;
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

// Strictly validated per-cell phase-field mobility overrides, keyed by global
// cell id. Ids are unique; every value is finite and positive as a float.
struct PhaseFieldMobilityMap {
    std::vector<std::pair<int32_t, float>> entries;
};

// Parse one "global_id value" pair per line ('#' comments and blank lines are
// permitted). Any malformed row, non-finite or non-positive value, or
// duplicate id fails with a message; membership of the ids in the simulation
// is checked by the apply/overlay functions below before the run starts.
bool read_phase_field_mobility_map(const std::string& path,
                                   PhaseFieldMobilityMap* out,
                                   std::string* error);

// Fill out[i] with the uniform mobility, then apply map entries by global id.
// An id absent from gid[0..n), or duplicated within gid, fails with a message.
bool apply_phase_field_mobility(const PhaseFieldMobilityMap& map,
                                double uniform, const int32_t* gid, int n,
                                float* out, std::string* error);

// Apply map entries by global id on top of the existing values in inout,
// leaving unlisted cells untouched. Same id checks as above.
bool overlay_phase_field_mobility(const PhaseFieldMobilityMap& map,
                                  const int32_t* gid, int n,
                                  float* inout, std::string* error);

// Resume-time precedence, in exactly three cases:
//   1. --phase-field-mobility given (with or without a map): every cell is
//      rebased to that uniform value and map entries then override their ids.
//   2. only --phase-field-mobility-map given: map entries override their ids
//      ON TOP of the existing per-cell values (stored MOBI, or the all-0.5
//      fill of a pre-MOBI checkpoint); unlisted cells keep those values. The
//      current parameter record does not store the writing run's uniform
//      mobility (unlike gamma), so a map-only resume must never rebase
//      unlisted cells to the compiled default.
//   3. neither flag: stored MOBI values are kept; a checkpoint without MOBI
//      loads with M_i = 0.5 for every cell.
bool resolve_phase_field_mobility(const SimParams& p, const SimOverrides& ov,
                                  const PhaseFieldMobilityMap& map,
                                  bool map_given, CheckpointData* d);

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
