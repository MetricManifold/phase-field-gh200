#pragma once

#include "params.cuh"

#include <cstdint>

namespace pf3d {

enum class UpdateTilePass3D { All, Boundary, Interior };

// Canonical row union and bounded z span of the outgoing aggregate field.
struct TileExchangeRegion3D {
    int y_lo[4]{};
    int y_hi[4]{};
    int y_count = 0;
    int z_lo = 0;
    int z_hi = 0;
};

PF3D_HD constexpr bool valid_update_tile_pass(UpdateTilePass3D pass) {
    return pass == UpdateTilePass3D::All
        || pass == UpdateTilePass3D::Boundary
        || pass == UpdateTilePass3D::Interior;
}

PF3D_HD constexpr bool valid_tile_exchange_region(
    const TileExchangeRegion3D& region, int ny, int nz) {
    if (ny <= 0 || nz <= 0 || region.y_count < 0 || region.y_count > 4
        || region.z_lo < 0 || region.z_hi < region.z_lo || region.z_hi > nz) {
        return false;
    }
    for (int i = 0; i < region.y_count; ++i) {
        if (region.y_lo[i] < 0 || region.y_hi[i] > ny
            || region.y_lo[i] >= region.y_hi[i]
            || (i != 0 && region.y_lo[i] <= region.y_hi[i - 1])) {
            return false;
        }
    }
    return true;
}

// Classify source voxels before recentering phi_out. The launch validates
// region; canonical y origins and local offsets must lie in [0,ny).
PF3D_HD inline bool tile_intersects_exchange_canonical(
    const TileExchangeRegion3D& region, int ny,
    int origin_y, std::int64_t origin_z,
    int y0, int z0, int extent_y, int extent_z) {
    if (ny <= 0 || origin_y < 0 || origin_y >= ny || y0 < 0 || y0 >= ny
        || region.y_count <= 0 || region.y_count > 4
        || region.z_lo >= region.z_hi || extent_y <= 0 || extent_z <= 0) {
        return false;
    }

    // Compare the origin to shifted bounds, without overflowing origin+offset.
    const std::int64_t z_lower = static_cast<std::int64_t>(region.z_lo)
        - static_cast<std::int64_t>(z0) - static_cast<std::int64_t>(extent_z);
    const std::int64_t z_upper = static_cast<std::int64_t>(region.z_hi)
        - static_cast<std::int64_t>(z0);
    if (origin_z <= z_lower || origin_z >= z_upper) return false;
    if (extent_y >= ny) return true;

    // Both sums are below 2*ny; uint32_t also covers domains near INT_MAX.
    const auto period = static_cast<std::uint32_t>(ny);
    std::uint32_t begin = static_cast<std::uint32_t>(origin_y)
        + static_cast<std::uint32_t>(y0);
    if (begin >= period) begin -= period;
    const std::uint32_t end = begin + static_cast<std::uint32_t>(extent_y);
    for (int i = 0; i < region.y_count; ++i) {
        const auto lo = static_cast<std::uint32_t>(region.y_lo[i]);
        const auto hi = static_cast<std::uint32_t>(region.y_hi[i]);
        if ((begin < hi && end > lo)
            || (end > period && end - period > lo)) return true;
    }
    return false;
}

// Arbitrary unwrapped origins and signed local offsets share the same tests
// after reducing the source tile's first y coordinate into the primary box.
PF3D_HD inline bool tile_intersects_exchange(
    const TileExchangeRegion3D& region, int ny,
    std::int64_t origin_y, std::int64_t origin_z,
    int y0, int z0, int extent_y, int extent_z) {
    if (ny <= 0) return false;
    std::int64_t begin = (origin_y % ny + static_cast<std::int64_t>(y0)) % ny;
    if (begin < 0) begin += ny;
    return tile_intersects_exchange_canonical(
        region, ny, static_cast<int>(begin), origin_z,
        0, z0, extent_y, extent_z);
}

} // namespace pf3d
