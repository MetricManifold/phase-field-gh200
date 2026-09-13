#include "pf3d/tile_partition.cuh"
#include "pf3d/update_shards.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using pf3d::TileExchangeRegion3D;

int failures = 0;
std::uint64_t comparisons = 0;
std::uint64_t canonical_comparisons = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[FAIL] %s\n", message);
        ++failures;
    }
}

TileExchangeRegion3D region_from_mask(int ny, unsigned mask,
                                    int z_lo, int z_hi) {
    TileExchangeRegion3D region{};
    region.z_lo = z_lo;
    region.z_hi = z_hi;
    for (int y = 0; y < ny;) {
        if ((mask & (1u << y)) == 0) {
            ++y;
            continue;
        }
        const int piece = region.y_count++;
        region.y_lo[piece] = y++;
        while (y < ny && (mask & (1u << y)) != 0) ++y;
        region.y_hi[piece] = y;
    }
    return region;
}

bool sent_voxel(const TileExchangeRegion3D& region, int ny,
                std::int64_t origin_y, std::int64_t origin_z,
                std::int64_t local_y, std::int64_t local_z) {
    // Reduce y before adding; compare z against translated bounds. Neither
    // calculation forms origin + offset, which may exceed int64_t.
    std::int64_t y = (origin_y % ny + local_y % ny) % ny;
    if (y < 0) y += ny;
    if (origin_z < static_cast<std::int64_t>(region.z_lo) - local_z ||
        origin_z >= static_cast<std::int64_t>(region.z_hi) - local_z)
        return false;
    for (int i = 0; i < region.y_count; ++i)
        if (y >= region.y_lo[i] && y < region.y_hi[i]) return true;
    return false;
}

bool brute_intersection(const TileExchangeRegion3D& region, int ny,
                        std::int64_t origin_y, std::int64_t origin_z,
                        int y0, int z0, int extent_y, int extent_z) {
    for (int z = 0; z < extent_z; ++z)
        for (int y = 0; y < extent_y; ++y)
            if (sent_voxel(region, ny, origin_y, origin_z,
                           static_cast<std::int64_t>(y0) + y,
                           static_cast<std::int64_t>(z0) + z))
                return true;
    return false;
}

bool compare(const TileExchangeRegion3D& region, int ny,
             std::int64_t origin_y, std::int64_t origin_z,
             int y0, int z0, int extent_y, int extent_z) {
    ++comparisons;
    const bool expected = brute_intersection(
        region, ny, origin_y, origin_z, y0, z0, extent_y, extent_z);
    const bool actual = pf3d::tile_intersects_exchange(
        region, ny, origin_y, origin_z, y0, z0, extent_y, extent_z);
    bool canonical = expected;
    if (y0 >= 0 && y0 < ny) {
        std::int64_t wrapped = origin_y % ny;
        if (wrapped < 0) wrapped += ny;
        ++canonical_comparisons;
        canonical = pf3d::tile_intersects_exchange_canonical(
            region, ny, static_cast<int>(wrapped), origin_z,
            y0, z0, extent_y, extent_z);
    }
    if (actual == expected && canonical == expected) return true;
    std::fprintf(stderr,
        "[FAIL] ny=%d origin=(%lld,%lld) tile=(%d,%d;%d,%d) "
        "z=[%d,%d) expected=%d generic=%d canonical=%d\n",
        ny, static_cast<long long>(origin_y),
        static_cast<long long>(origin_z), y0, z0, extent_y, extent_z,
        region.z_lo, region.z_hi, expected, actual, canonical);
    for (int i = 0; i < region.y_count; ++i)
        std::fprintf(stderr, "  rows=[%d,%d)\n",
                     region.y_lo[i], region.y_hi[i]);
    ++failures;
    return false;
}

void test_region_validation() {
    const auto valid = region_from_mask(8, 0x55u, 1, 5);
    expect(pf3d::valid_tile_exchange_region(valid, 8, 6),
           "four separated row pieces are valid");
    expect(pf3d::valid_tile_exchange_region({}, 8, 6),
           "empty exchange is valid");
    expect(pf3d::valid_tile_exchange_region(
               region_from_mask(8, 0xffu, 0, 6), 8, 6),
           "full domain is valid");
    expect(!pf3d::valid_tile_exchange_region(valid, 0, 6) &&
               !pf3d::valid_tile_exchange_region(valid, 8, 0) &&
               !pf3d::valid_tile_exchange_region(valid, -1, 6) &&
               !pf3d::valid_tile_exchange_region(valid, 8, -1),
           "nonpositive domain extents are rejected");
    for (int count : {-1, 5, std::numeric_limits<int>::max()}) {
        auto invalid = valid;
        invalid.y_count = count;
        expect(!pf3d::valid_tile_exchange_region(invalid, 8, 6),
               "invalid row count is rejected before indexing");
    }
    for (int mode = 0; mode < 9; ++mode) {
        auto invalid = valid;
        switch (mode) {
            case 0: invalid.y_lo[0] = -1; break;
            case 1: invalid.y_hi[3] = 9; break;
            case 2: invalid.y_hi[0] = invalid.y_lo[0]; break;
            case 3: invalid.y_hi[0] = -1; break;
            case 4: invalid.y_lo[1] = 0; break;
            case 5: invalid.y_lo[1] = invalid.y_hi[0]; break;
            case 6: invalid.z_lo = -1; break;
            case 7: invalid.z_hi = 7; break;
            case 8: invalid.z_lo = invalid.z_hi + 1; break;
        }
        expect(!pf3d::valid_tile_exchange_region(invalid, 8, 6),
               "noncanonical rows or invalid z interval are rejected");
    }
    auto unordered = valid;
    std::swap(unordered.y_lo[0], unordered.y_lo[1]);
    std::swap(unordered.y_hi[0], unordered.y_hi[1]);
    expect(!pf3d::valid_tile_exchange_region(unordered, 8, 6),
           "unordered disjoint rows are rejected");
}

void test_exhaustive_small_domains() {
    for (int ny = 1; ny <= 4; ++ny)
    for (int nz = 1; nz <= 3; ++nz)
    for (unsigned mask = 0; mask < (1u << ny); ++mask)
    for (int z_lo = 0; z_lo <= nz; ++z_lo)
    for (int z_hi = z_lo; z_hi <= nz; ++z_hi) {
        const auto region = region_from_mask(ny, mask, z_lo, z_hi);
        expect(pf3d::valid_tile_exchange_region(region, ny, nz),
               "every generated canonical union validates");
        for (int oy = -ny; oy <= ny; ++oy)
        for (int oz = -nz; oz <= nz; ++oz)
        for (int y0 = 0; y0 <= 2; ++y0)
        for (int z0 = 0; z0 <= 1; ++z0)
        for (int ey = 1; ey <= ny + 1; ++ey)
        for (int ez = 1; ez <= 3; ++ez)
            if (!compare(region, ny, oy, oz, y0, z0, ey, ez)) return;
    }
}

void test_extreme_coordinates() {
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    const std::int64_t origins[] = {lo, lo + 1, -1001, -1, 0, 1001, hi - 1, hi};
    const int offsets[] = {std::numeric_limits<int>::min(), -3, 0, 3,
                           std::numeric_limits<int>::max()};
    for (int ny : {1, 5, 8})
    for (unsigned mask : {0u, 1u, 0x55u, 0xffu}) {
        const auto region = region_from_mask(ny, mask, 1, 4);
        for (auto oy : origins)
        for (auto oz : origins)
        for (int y0 : offsets)
        for (int z0 : offsets)
        for (int ey : {-1, 0, 1, 2, 5})
        for (int ez : {-1, 0, 1, 2, 5})
            if (!compare(region, ny, oy, oz, y0, z0, ey, ez)) return;
    }
}

void test_partition_coverage() {
    for (int ny : {5, 8})
    for (unsigned mask : {0u, 0x55u, 0x81u, 0xffu})
    for (int oy : {-17, -1, 0, 4, 19})
    for (int oz : {-7, -1, 0, 4})
    for (int edge : {3, 7, 10})
    for (int tile_y : {2, 4})
    for (int tile_z : {2, 3}) {
        const auto region = region_from_mask(ny, mask, 0, 5);
        std::vector<int> visits(static_cast<std::size_t>(edge) * edge, 0);
        for (auto pass : {pf3d::UpdateTilePass3D::Boundary,
                          pf3d::UpdateTilePass3D::Interior}) {
            for (int z0 = 0; z0 < edge; z0 += tile_z)
            for (int y0 = 0; y0 < edge; y0 += tile_y) {
                const int ey = std::min(tile_y, edge - y0);
                const int ez = std::min(tile_z, edge - z0);
                const bool boundary = pf3d::tile_intersects_exchange(
                    region, ny, oy, oz, y0, z0, ey, ez);
                if (boundary != (pass == pf3d::UpdateTilePass3D::Boundary))
                    continue;
                for (int z = z0; z < z0 + ez; ++z)
                for (int y = y0; y < y0 + ey; ++y) {
                    ++visits[static_cast<std::size_t>(z) * edge + y];
                    if (sent_voxel(region, ny, oy, oz, y, z))
                        expect(boundary, "every sent voxel executes in the early pass");
                }
            }
        }
        expect(std::all_of(visits.begin(), visits.end(),
                           [](int count) { return count == 1; }),
               "early and late passes cover partial tiles exactly once");
    }
}

void test_large_canonical_domains() {
    constexpr int max_int = std::numeric_limits<int>::max();
    constexpr auto min_origin = std::numeric_limits<std::int64_t>::min();
    constexpr auto max_origin = std::numeric_limits<std::int64_t>::max();
    for (int ny : {100001, max_int - 7, max_int - 1, max_int}) {
        std::vector<TileExchangeRegion3D> regions(4);
        regions[0] = {};
        regions[1] = {{0}, {ny}, 1, 0, 16};
        regions[2] = {{0, ny - 7}, {5, ny}, 2, 3, 11};
        regions[3] = {{0, 13, ny / 2, ny - 3},
                      {1, 17, ny / 2 + 8, ny}, 4, 7, 8};
        const std::int64_t origins_y[] = {
            min_origin, min_origin + 1, -static_cast<std::int64_t>(ny) - 5,
            -1, 0, 1, ny / 2, ny - 1, max_origin};
        const std::int64_t origins_z[] = {min_origin, -7, 0, 7, max_origin};
        for (const auto& region : regions) {
            expect(pf3d::valid_tile_exchange_region(region, ny, 16),
                   "large-domain canonical region validates");
            for (auto oy : origins_y)
            for (auto oz : origins_z)
            for (int y0 : {0, 1, ny / 2, ny - 17, ny - 2})
            for (int z0 : {0, 3, 13})
            for (int ey : {1, 2, 8, 16})
            for (int ez : {1, 3, 8}) {
                // These source tiles fit a brick whose edge is below ny.
                if (static_cast<std::int64_t>(y0) + ey >= ny) continue;
                if (!compare(region, ny, oy, oz, y0, z0, ey, ez)) return;
            }
        }
    }
    const auto region = region_from_mask(8, 0xffu, 0, 6);
    for (int oy : {-1, 8, max_int})
        expect(!pf3d::tile_intersects_exchange_canonical(
                   region, 8, oy, 0, 0, 0, 1, 1),
               "noncanonical origins are rejected");
    for (int y0 : {-1, 8, max_int})
        expect(!pf3d::tile_intersects_exchange_canonical(
                   region, 8, 0, 0, y0, 0, 1, 1),
               "unbounded local offsets are rejected");
}

void test_queued_update_shards() {
    using pf3d::queued_update_shards;
    expect(queued_update_shards(12, 132, 4, 950) == 512 &&
               queued_update_shards(50, 132, 4, 950) == 512 &&
               queued_update_shards(100, 132, 4, 950) == 256,
           "N=24/100/200 evenly owned populations use the measured queued policy");
    expect(queued_update_shards(12, 132, 4, 6) == 6 &&
               queued_update_shards(12, 132, 4, 8) == 8,
           "small cubes do not launch more shards than tiles");
    for (int sms : {1, 2, 8, 132, 148, 256})
    for (int blocks : {1, 2, 4, 8})
    for (int tiles : {1, 6, 8, 17, 64, 810, 950, 2744}) {
        int previous = pf3d::kAutomaticUpdateShardCap;
        for (int cells = 1; cells <= 2048; ++cells) {
            const int target = (32 * sms * blocks + cells - 1) / cells;
            int expected = 1;
            while (expected < target) expected *= 2;
            expected = std::min({expected, tiles, 512});
            const int actual = queued_update_shards(cells, sms, blocks, tiles);
            expect(actual == expected, "queued policy matches the direct wave formula");
            expect(actual >= 1 && actual <= std::min(tiles, 512),
                   "queued count stays within the tile and automatic caps");
            expect(actual <= previous, "larger populations never need more shards");
            previous = actual;
        }
    }
    constexpr int min_int = std::numeric_limits<int>::min();
    constexpr int max_int = std::numeric_limits<int>::max();
    for (int invalid : {min_int, -1, 0}) {
        expect(queued_update_shards(invalid, 132, 4, 950) == 1 &&
                   queued_update_shards(50, invalid, 4, 950) == 1 &&
                   queued_update_shards(50, 132, invalid, 950) == 1 &&
                   queued_update_shards(50, 132, 4, invalid) == 1,
               "empty populations or invalid geometry/occupancy use one shard");
    }
    expect(queued_update_shards(max_int, 1, 1, max_int) == 1 &&
               queued_update_shards(max_int, max_int, 1, max_int) == 32,
           "large population arithmetic preserves the uncapped ratio");
    for (int tiles : {1, 3, 256, 512, max_int})
        expect(queued_update_shards(max_int, max_int, max_int, tiles)
                   == std::min(tiles, 512),
               "extreme occupancy saturates before multiplication can overflow");
}

void test_promoted_update_tiles() {
    using pf3d::capped_update_tile_count;
    using pf3d::queued_update_shards;
    expect(capped_update_tile_count(24, 32, 16, 8) == 6 &&
               capped_update_tile_count(32, 32, 16, 8) == 8,
           "small promoted cubes cap scheduling at their logical tile count");
    expect(capped_update_tile_count(224, 32, 16, 8) == 512,
           "large promoted cubes expose the full automatic scheduling cap");
    const int tiles = capped_update_tile_count(224, 32, 16, 8);
    for (int promoted : {1, 4, 24})
        expect(queued_update_shards(promoted, 132, 3, tiles) == 512,
               "sparse promoted populations can queue more than 64 fast CTAs");
    expect(queued_update_shards(64, 132, 3, tiles) == 256 &&
               queued_update_shards(200, 132, 3, tiles) == 64,
           "promoted scheduling shrinks as the promoted population grows");
    constexpr int max_int = std::numeric_limits<int>::max();
    expect(capped_update_tile_count(max_int, 1, 1, 1) == 512 &&
               capped_update_tile_count(max_int, max_int, max_int, max_int) == 1,
           "capped tile products do not overflow at integer limits");
    for (int invalid : {-1, 0})
        expect(capped_update_tile_count(invalid, 32, 16, 8) == 0 &&
                   capped_update_tile_count(224, invalid, 16, 8) == 0 &&
                   capped_update_tile_count(224, 32, invalid, 8) == 0 &&
                   capped_update_tile_count(224, 32, 16, invalid) == 0,
               "invalid tile dimensions are rejected before division");
}

}  // namespace

int main() {
    test_region_validation();
    test_exhaustive_small_domains();
    test_extreme_coordinates();
    test_partition_coverage();
    test_large_canonical_domains();
    test_queued_update_shards();
    test_promoted_update_tiles();
    if (failures != 0) return 1;
    std::printf("tile partition: %llu generic and %llu canonical oracle "
                "comparisons plus coverage/shard-policy tests passed\n",
                static_cast<unsigned long long>(comparisons),
                static_cast<unsigned long long>(canonical_comparisons));
    return 0;
}
