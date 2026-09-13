#pragma once

#include <cstdint>

namespace pf3d::boundary {

// Projected contours retain float32 corner samples, not an inferred contact
// graph. Records are little-endian; the projection is part of the file identity.
enum class Projection : std::uint32_t { Basal = 1, Maximum = 2 };

struct FileHeader {
    char magic[8];                       // "PFB3D1", zero padded
    std::uint32_t version, header_bytes, cells, projection, boundary_flags, codec;
    std::int64_t nx, ny, nz;
    double dx, dy, dz, dt, tau, level;
    std::uint64_t interval;
    std::uint32_t square_bytes, cell_bytes;
    std::uint64_t reserved;
};

struct FrameHeader {
    char magic[8];                       // "P3BFRM1" or "P3BEND1"
    std::uint64_t step;
    double time;
    std::uint64_t raw_bytes, stored_bytes, checksum;
};

struct Cell {
    std::int64_t id, origin_x, origin_y, origin_z;
    // The projected image has one zero sample around each x/y face. Its
    // origin is the stored brick origin minus one in x/y; z is unchanged.
    std::uint32_t brick_edge, plane_edge, squares, reserved;
    float gamma, active_speed, radius, reserved_float;
};

struct Square {
    std::uint32_t xy;                    // local x in low 16 bits, y in high 16
    float phi[4];                       // lower-left, lower-right, upper-right, upper-left
};

static_assert(sizeof(FileHeader) == 128);
static_assert(sizeof(FrameHeader) == 48);
static_assert(sizeof(Cell) == 64);
static_assert(sizeof(Square) == 20);

} // namespace pf3d::boundary
