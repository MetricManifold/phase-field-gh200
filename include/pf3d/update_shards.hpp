#pragma once

#include <cstdint>

namespace pf3d {

constexpr int kAutomaticUpdateShardCap = 512;
constexpr int kMaximumFastUpdateShards = 1024;

// Only the capped count is needed by the scheduler. Saturating each factor
// also avoids overflowing when a caller supplies an invalidly large edge.
constexpr int capped_update_tile_count(int edge, int tile_x, int tile_y,
                                      int tile_z) {
    if (edge <= 0 || tile_x <= 0 || tile_y <= 0 || tile_z <= 0) return 0;
    int tiles = 1;
    const int tile_edges[] = {tile_x, tile_y, tile_z};
    for (int tile_edge : tile_edges) {
        const int axis = edge / tile_edge + (edge % tile_edge != 0);
        const std::int64_t product = static_cast<std::int64_t>(tiles) * axis;
        tiles = product < kAutomaticUpdateShardCap
            ? static_cast<int>(product) : kAutomaticUpdateShardCap;
    }
    return tiles;
}

// Bounded or partial tile walks have uneven work per CTA. Queued waves let the
// scheduler fill gaps; these updates do not contain floating-point reductions.
constexpr int queued_update_shards(int cells, int multiprocessors,
                                  int blocks_per_sm, int tile_count) {
    if (cells <= 0 || multiprocessors <= 0 || blocks_per_sm <= 0 || tile_count <= 0)
        return 1;
    constexpr int target_waves = 32;
    const int cap = tile_count < kAutomaticUpdateShardCap
        ? tile_count : kAutomaticUpdateShardCap;
    const std::int64_t wave =
        static_cast<std::int64_t>(multiprocessors) * blocks_per_sm;
    const std::int64_t capped_blocks = static_cast<std::int64_t>(cells) * cap;
    // Saturate before multiplying a potentially large occupancy by the target.
    if (wave >= (capped_blocks + target_waves - 1) / target_waves) return cap;
    const std::int64_t target = (wave * target_waves + cells - 1) / cells;
    int shards = 1;
    while (shards < target && shards < cap) shards *= 2;
    return shards < cap ? shards : cap;
}

} // namespace pf3d
