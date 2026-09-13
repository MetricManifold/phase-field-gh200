#pragma once

#include "boundary_format_3d.h"
#include "kernels.cuh"
#include "phase_storage.hpp"

#include <memory>
#include <string>

namespace pf3d {

// Read-only output: neither projection updates moments, fields, or random state.
// Field storage changes addresses only; contours retain logical cell coordinates.
class BoundaryOutput3D {
public:
    BoundaryOutput3D();
    ~BoundaryOutput3D();
    BoundaryOutput3D(const BoundaryOutput3D&) = delete;
    BoundaryOutput3D& operator=(const BoundaryOutput3D&) = delete;

    bool open(const std::string& path, const SimParams3D& params,
              boundary::Projection projection, std::uint64_t interval,
              bool compress);
    bool capture(const float* base_phi, const float* const* promoted_phi,
                 const CellState3D* cells, int base_edge, cudaStream_t stream,
                 std::uint64_t step, double time,
                 CellFieldStorage3D storage = {});
    // Failed runs leave an unterminated stream rather than a success marker.
    bool close(bool complete = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pf3d
