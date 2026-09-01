#pragma once

// Checkpoint I/O for the 3D solver. Records include 3D coordinates, vector
// state, adaptive field storage, and boundary-condition metadata.

#include "kernels.cuh"
#include "params.cuh"
#include "reduction_mode.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace pf3d {

struct CheckpointMeta3D {
    SimParams3D params{};
    std::uint64_t step = 0;
    double time = 0.0;
    int brick_edge = 0;
    int max_storage_edge = 0;
    int print_interval = 100;
    std::uint32_t trajectory_interval = 0;
    PromotedMeasureReduction3D promoted_measure_reduction{};
    // Resolved base-measurement CTAs per cell; readers treat zero as standard.
    int base_measure_shards = 0;
    std::uint64_t file_crc64 = 0;
    // One edge per serialized cell slot.
    std::vector<int> storage_edges;
};

// Validate the fixed metadata and per-cell storage layout before allocation.
// The complete-file checksum is verified by checkpoint_load_3d.
bool checkpoint_probe_3d(const std::string& path, CheckpointMeta3D* out);

// Load a previously probed payload directly into device storage. Phase data
// are streamed through bounded host staging memory.
struct CheckpointLoadView3D {
    CellState3D* d_cells = nullptr;
    // Base-cell payloads use d_phi + cell_index*brick_edge^3.
    float* d_phi = nullptr;
    // Host pointer table indexed by serialized cell. Required only for cells
    // whose metadata edge exceeds brick_edge.
    float* const* h_promoted_phi = nullptr;
    cudaStream_t stream = nullptr;
};

bool checkpoint_load_3d(const std::string& path,
                        const CheckpointMeta3D& expected,
                        const CheckpointLoadView3D& view);

// Convenience overload for checkpoints in which every cell uses the base
// brick edge and therefore needs no promoted allocation table.
bool checkpoint_load_3d(const std::string& path,
                        const CheckpointMeta3D& expected,
                        CellState3D* d_cells, float* d_phi,
                        cudaStream_t stream = nullptr);

struct CheckpointWriteView3D {
    const SimParams3D* params = nullptr;
    std::uint64_t step = 0;
    double time = 0.0;
    int brick_edge = 0;
    int print_interval = 100;
    std::uint32_t trajectory_interval = 0;
    PromotedMeasureReduction3D promoted_measure_reduction{};
    // Resolved base-measurement CTAs per cell (1..64), covered by the file
    // checksum.
    int base_measure_shards = 0;
    const CellState3D* d_cells = nullptr;
    const float* d_phi = nullptr;
    // Host pointer table indexed like d_cells. A non-null entry supplies the
    // cube for a cell whose CellState3D::storage_edge exceeds brick_edge.
    float* const* h_promoted_phi = nullptr;
    cudaStream_t stream = nullptr;
};

// Write through a unique temporary file and atomically replace the target only
// after every device copy, write, flush, and close has succeeded.
bool checkpoint_write_3d(const std::string& path,
                         const CheckpointWriteView3D& view);

}  // namespace pf3d
