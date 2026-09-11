#pragma once
#include <cstdint>

namespace pf::boundary {
// Little-endian, independently framed snapshots. Squares retain the original
// float32 samples; contour interpolation and contact definitions stay offline.
struct FileHeader {
    char magic[8];
    uint32_t version, header_bytes, cells, side, tile_pitch, square_bytes;
    double dt, tau, level;
    uint64_t interval;
    uint32_t codec, reserved; // 0: raw; 1: zstd
};
struct FrameHeader {
    char magic[8]; // PFBFRM1 or PFBEND1
    uint64_t step;
    double time;
    uint64_t raw_bytes, stored_bytes, checksum;
};
struct Cell {
    int32_t id, origin_x, origin_y;
    float cx, cy, gamma, mobility, active_speed, radius;
    uint32_t squares;
};
struct Square {
    uint32_t xy; // x in low 16 bits, y in high 16 bits, relative to tile origin
    float phi[4]; // lower left, lower right, upper right, upper left
};
static_assert(sizeof(FileHeader) == 72);
static_assert(sizeof(FrameHeader) == 48);
static_assert(sizeof(Cell) == 40);
static_assert(sizeof(Square) == 20);
} // namespace pf::boundary
