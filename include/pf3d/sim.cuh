#pragma once

#include "checkpoint.cuh"
#include "initializer.hpp"
#include "kernels.cuh"
#include "palmieri_initializer.hpp"
#include "params.cuh"
#include "reduction_mode.hpp"

#include <cuda_runtime.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace pf3d {

enum class StorageMode3D {
    Auto,
    Throughput,  // two phi + two S; direct out-of-place update
    Balanced,    // one phi + two S; exact per-CTA scratch update
    Compact      // one phi + one S; rebuild S after each update
};

const char* storage_mode_name(StorageMode3D mode);
bool parse_storage_mode(const std::string& text, StorageMode3D* mode);

struct RunOptions3D {
    StorageMode3D storage_mode = StorageMode3D::Auto;
    int brick_edge = 0;       // zero selects the model-derived minimum
    int scratch_slots = 0;    // zero selects the largest safe useful count
    double memory_fraction = 0.95;
    int device = 0;
    bool strict = false;
    bool final_checkpoint = true;
    int bench_steps = 0;
    // After the plain benchmark, repeat it with per-phase CUDA-event timing
    // and report median per-step costs of measurement, updates, and overhead.
    bool bench_phases = false;
    // Zero selects the occupancy-derived counts; positive values override for
    // benchmarking shard policies.
    int promoted_shards = 0;
    int fast_base_shards = 0;
    // Base-cell measurement CTAs per cell: 0 uses the standard at-most-four
    // wave-fitting policy, -1 raises its cap to 64, and 1..64 pins the count.
    int measure_shards = 0;
    // Distinguishes an omitted option from an explicit policy on resume.
    bool measure_shards_supplied = false;
    // Zero keeps the bitwise one-CTA promoted measurement.  A positive count
    // (or -1 for the occupancy-derived value) shards it with a deterministic
    // ascending finalize. Regrouping the reduction may change low-order
    // floating-point rounding, so its output is not bitwise comparable with
    // the one-CTA fold.
    int promoted_measure_shards = 0;
    bool promoted_measure_shards_supplied = false;
    // Resolved occupancy wave restored from an automatic-policy checkpoint.
    // Fresh runs leave this zero and derive it from the selected device.
    int promoted_measure_auto_wave_ctas = 0;
    int print_interval = 100;
    long long trajectory_interval = 0;
    int trajectory_samples = 100;
    long long checkpoint_interval = 0;
    long long save_interval = 0;
    std::string trajectory_path;
    std::string checkpoint_dir;
    std::string initial_centres_path;
};

class Sim3D {
public:
    Sim3D() = default;
    ~Sim3D();
    Sim3D(const Sim3D&) = delete;
    Sim3D& operator=(const Sim3D&) = delete;

    bool init_fresh(const SimParams3D& params, const RunOptions3D& options);
    bool init_checkpoint(const CheckpointMeta3D& checkpoint,
                         const std::string& path,
                         const RunOptions3D& options,
                         double t_end_override = -1.0);
    bool run();
    bool bench(int steps, double* milliseconds_per_step);
    bool save_checkpoint(const std::string& path);
    bool verify(double* max_relative_volume_error,
                std::uint32_t* maximum_S,
                std::uint64_t* support_edge_voxels);

    std::uint64_t step() const { return steps_done_; }
    double time() const { return static_cast<double>(steps_done_) * params_.dt; }
    int brick_edge() const { return B_; }
    StorageMode3D storage_mode() const { return selected_mode_; }

    static volatile std::sig_atomic_t terminate_requested;
    static void request_termination(int) { terminate_requested = 1; }

private:
    friend struct Sim3DTestAccess;
    bool allocate(const SimParams3D& params, const RunOptions3D& options,
                  int requested_brick_edge, bool need_initial_centres,
                  std::optional<int> checkpoint_measure_shards);
    bool initialize_common();
    bool initialize_fields();
    bool reconstruct_current_S();
    bool refresh_measurements(bool apply_tumble,
                              bool compute_surface = false,
                              bool apply_motion = true);
    bool ensure_measurements(bool require_surface, const char* context);
    bool launch_one_step();
    int promoted_update_shards(int promoted_count) const;
    int promoted_measure_shard_count(int promoted_count) const;
    int fast_base_update_shards() const;
    bool phase_mark(int slot);
    void clear_phase_events() noexcept;
    bool bench_phase_report(int steps);
    bool synchronize_and_check(const char* context);
    bool fatal_flags_present(bool print);
    bool recover_support_exhaustion(
        const std::vector<std::uint32_t>& flags);
    bool install_checkpoint_promotions(const CheckpointMeta3D& checkpoint);
    bool allocate_promoted_fields(const std::vector<int>& cell_ids,
                                  std::vector<CellState3D>* states);
    bool grow_promoted_fields(int new_edge,
                              const std::vector<int>& additional_cell_ids,
                              std::vector<CellState3D>* states);
    bool upload_promoted_tables();
    bool open_trajectory();
    bool append_trajectory();
    void close_trajectory();
    bool checkpoint_at_current_state(const std::string& path);
    const float* current_phi() const;
    float* current_phi();
    std::uint32_t* current_S();
    WeightedWallField3D weighted_wall_field() const;

    SimParams3D params_{};
    RunOptions3D options_{};
    StorageMode3D selected_mode_ = StorageMode3D::Auto;
    SLayout3D layout_{};
    int B_ = 0;
    int phi_buffers_ = 0;
    int S_buffers_ = 0;
    int scratch_slots_ = 0;
    int current_phi_index_ = 0;
    int current_S_index_ = 0;
    int current_promoted_phi_index_ = 0;
    int update_grid_blocks_ = 0;
    int measurement_shards_ = 1;
    int throughput_update_shards_ = 1;
    std::size_t brick_words_ = 0;
    std::size_t S_words_ = 0;
    std::size_t required_device_bytes_ = 0;
    std::size_t adaptive_budget_remaining_ = 0;
    std::size_t promoted_words_ = 0;
    int promoted_edge_ = 0;
    int maximum_support_edge_ = 0;
    int sm_count_ = 0;
    int fast_promoted_blocks_per_sm_ = 1;
    int promoted_measure_blocks_per_sm_ = 1;
    int fast_base_blocks_per_sm_ = 1;
    std::uint64_t recovery_events_ = 0;
    MomentPartial3D* d_promoted_partials_ = nullptr;
    // Per-phase benchmark instrumentation: seven stream events per step while
    // active, drained after the run.
    std::vector<cudaEvent_t> phase_events_;
    int phase_capacity_ = 0;
    int phase_recorded_ = 0;
    bool phase_active_ = false;
    std::uint64_t steps_done_ = 0;
    bool volume_current_ = true;
    bool surface_current_ = true;

    float* d_phi_[2] = {nullptr, nullptr};
    float** d_promoted_phi_[2] = {nullptr, nullptr};
    int* d_promoted_ids_ = nullptr;
    std::uint32_t* d_S_[2] = {nullptr, nullptr};
    float* d_wall_psi_sq_ = nullptr;
    float* d_weighted_wall_psi_sq_ = nullptr;
    float* d_scratch_ = nullptr;
    CellState3D* d_cells_ = nullptr;
    CellState3D* d_accepted_cells_ = nullptr;
    std::uint32_t* d_support_requests_ = nullptr;
    Vec3* d_centres_ = nullptr;
    std::uint64_t* d_step_ = nullptr;
    unsigned long long* d_work_cursor_ = nullptr;
    std::uint32_t* d_flags_ = nullptr;
    MomentPartial3D* d_moment_partials_ = nullptr;
    VerifyCell3D* d_verify_cells_ = nullptr;
    std::uint32_t* d_verify_S_ = nullptr;
    TrajPackedCell3D* d_trajectory_ = nullptr;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t bench_start_ = nullptr;
    cudaEvent_t bench_stop_ = nullptr;

    std::vector<TrajPackedCell3D> h_trajectory_;
    std::vector<float*> h_promoted_phi_[2];
    std::vector<int> h_promoted_ids_;
    std::FILE* trajectory_file_ = nullptr;
    bool trajectory_header_written_ = false;
    std::uint64_t trajectory_frames_ = 0;
    std::uint64_t trajectory_every_ = 0;
    std::uint64_t last_trajectory_step_ =
        static_cast<std::uint64_t>(-1);
};

}  // namespace pf3d
