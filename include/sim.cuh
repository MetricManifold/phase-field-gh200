#pragma once
// Host-side simulation state, ordered kernel launches, checkpointing, and
// diagnostics.

#include "checkpoint.cuh"
#include "kernels.cuh"
#include "params.cuh"

#include <csignal>
#include <cstdio>
#include <string>
#include <vector>

namespace pf {

// Six graph steps cover both phase-field buffer parities and all three
// shared-field rotation slots.
constexpr int kGraphBody = 6;
constexpr int kMortonEvery = kGraphBody;   // aligned to the graph body
// Maximum interval between host checks of fatal device flags.
constexpr long long kFatalCheckPollEvery = 10000;

// cudaFuncGetAttributes reports stack and spills together as local memory;
// build logs remain the authoritative spill check.
constexpr size_t kLocalBytesBudget = 64;
static_assert(kFatalCheckPollEvery > 0,
              "fatal-check polling cadence must be positive");

struct RunOptions {
    bool use_graph = true;
    bool morton = false;
    bool l2_persist = true;
    bool strict = false;          // enable k_verify at verify_every cadence
    int  bench_steps = 0;         // >0: timed benchmark, no I/O
    // Trajectory sampling is independent of status-print cadence.
    int  traj_samples  = 100;     // evenly spaced samples across the run
    long long traj_interval = 0;  // steps between samples; overrides traj_samples
    std::string out_path;
    // Fresh starts only; empty selects the built-in grid-and-jitter placement.
    std::string initial_centres_path;

    // Empty ckpt_dir disables all checkpoints. Cadences use elapsed-step
    // thresholds because a graph replay advances by kGraphBody steps.
    std::string ckpt_dir;
    long long ckpt_interval = 0;  // steps between rolling <dir>/checkpoint.bin
    long long save_interval = 0;  // steps between tagged checkpoint_%08d.bin
    bool      final_checkpoint = true;   // on normal exit AND on SIGTERM
};

class Sim {
public:
    Sim() = default;
    ~Sim();
    Sim(const Sim&) = delete;
    Sim& operator=(const Sim&) = delete;

    bool init(const SimParams& p, const RunOptions& opt, int device);
    // Resume from an already resolved parameter set and checkpoint microstate.
    bool init_from_checkpoint(const SimParams& p, const CheckpointData& d,
                              const RunOptions& opt, int device);
    // False indicates a fatal device flag or an output/checkpoint failure.
    bool run();
    bool bench(int steps, double* ms_per_step);
    // Write the current state outside the regular checkpoint cadence.
    bool save_checkpoint(const std::vector<std::string>& paths);
    void report_flags() const;

    // Signal handlers set only this flag; the step loop performs orderly exit.
    static volatile std::sig_atomic_t s_terminate;
    static void request_termination(int) { s_terminate = 1; }
    bool verify(double* max_rel_V, float* max_outside, uint32_t* max_S);

    long long step() const { return steps_done_; }
    double    time() const { return (double)steps_done_ * p_.dt; }
    int       side() const { return side_; }

private:
    bool alloc_device(const SimParams& p, const RunOptions& opt, int device);
    bool configure_and_capture();
    std::vector<std::string> checkpoint_paths(bool rolling, bool tagged) const;

    StepArgs args_for_slot(int slot) const;
    void     l2_window_for_slot(int slot, const void** base, size_t* bytes,
                                float* hit) const;
    void     launch_one(int slot);
    void     print_path_report() const;
    bool     build_graph();
    bool     seed_positions(std::vector<float>& cx, std::vector<float>& cy,
                            std::vector<float>& gam, std::vector<float>& va,
                            std::vector<int32_t>& gid);
    void     print_line();
    // Stop on a fatal flag or failed flag readback.
    bool     fatal_flag_set();

    SimParams  p_{};
    RunOptions opt_{};
    int device_ = 0;
    int grid_   = 132;
    int side_   = 0;      // L
    int pitch_  = 0;      // P

    float*      d_phi_[2] = {nullptr, nullptr};
    uint32_t*   d_S_      = nullptr;     // 3 * pitch_ * side_ uint32
    size_t      s_buf_words_ = 0;
    CellState*  d_cell_   = nullptr;
    uint8_t*    d_cls_    = nullptr;
    uint32_t*   d_perm_   = nullptr;
    unsigned long long* d_cursor_ = nullptr;   // 2 slots
    unsigned long long* d_step_   = nullptr;   // 2 slots
    uint32_t*   d_flags_  = nullptr;
    double*     d_vchk_   = nullptr;
    float*      d_ochk_   = nullptr;
    uint32_t*   d_smax_   = nullptr;
    TrajPackedCell* h_traj_ = nullptr;         // pinned
    TrajPackedCell* d_traj_ = nullptr;         // device alias of h_traj_

    cudaStream_t stream_ = nullptr;
    cudaGraph_t  graph_  = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_ready_ = false;
    bool fallback_reported_ = false;
    bool fallback_no_margin_reported_ = false;

    long long steps_done_ = 0;
    size_t l2_persist_max_ = 0;
    size_t l2_window_max_  = 0;

    // Stream samples so frames written before interruption remain available.
    std::FILE* traj_fp_ = nullptr;
    long long  traj_frames_ = 0;
    long long  trajectory_every_ = 0;
    bool open_trajectory(const std::string& path);
    bool append_trajectory_frame(long long step_at);
    bool close_trajectory();
};

}  // namespace pf
