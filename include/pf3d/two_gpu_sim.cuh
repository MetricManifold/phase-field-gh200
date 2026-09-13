#pragma once

#include "pf3d/sim.cuh"
#include "pf3d/substrate_planner.hpp"

#include <array>
#include <memory>

namespace pf3d {

// Exactly two peer-connected devices, with whole cells owned by a y partition.
// Physics, output scheduling, and checkpoint encoding remain in Sim3D.
class TwoGpuSim3D {
public:
    explicit TwoGpuSim3D(int peer_device = 1) : peer_device_(peer_device) {}
    ~TwoGpuSim3D();
    TwoGpuSim3D(const TwoGpuSim3D&) = delete;
    TwoGpuSim3D& operator=(const TwoGpuSim3D&) = delete;

    bool init_fresh(const SimParams3D&, const RunOptions3D&);
    bool init_checkpoint(const CheckpointMeta3D&, const std::string&,
                         const RunOptions3D&, double t_end_override = -1.0);
    bool run();
    bool bench(int steps, double* milliseconds_per_step);

private:
    bool preflight(const SimParams3D&, RunOptions3D*);
    bool initialize();
    bool clone_peer();
    bool synchronize_peer_growth();
    bool select(int rank) const;
    bool wait_all() const;
    bool read_states(int rank);
    bool enqueue_state_broadcast();
    bool copy_field(int id, int source, int destination);
    bool configure_ownership(bool canonical_fields);
    bool exchange_sum(bool pending);
    bool start_exchange(bool pending);
    bool finish_exchange(bool pending);
    bool merge_flags(bool include_states = false);
    bool gather_state();
    bool distributed_step();

    int peer_device_;
    std::array<std::unique_ptr<Sim3D>, 2> sim_;
    std::array<int*, 2> d_ids_{};
    std::array<std::uint32_t*, 2> d_receive_{};
    std::array<cudaEvent_t, 2> exchange_ready_{};
    std::array<cudaEvent_t, 2> exchange_sent_{};
    std::array<cudaStream_t, 2> exchange_streams_{};
    // One pinned allocation: cells followed by the integrity counters.
    struct HostSnapshot {
        CellState3D* cells = nullptr;
        std::uint32_t* flags = nullptr;
    };
    std::array<HostSnapshot, 2> downloads_{};
    HostSnapshot upload_{};
    std::array<std::vector<int>, 2> ids_;
    std::vector<CellState3D> canonical_;
    std::vector<int> owners_;
    std::vector<AllocatedBrick> planned_bricks_;
    std::array<SeamExchangePlan, 2> plans_{};
    std::array<std::uint32_t, FLAG3D_COUNT> flags_{};
    TwoRankSubstrateGeometry geometry_{};
    bool primary_canonical_ = true;
    std::uint64_t mirrored_recoveries_ = 0;
};

}  // namespace pf3d
