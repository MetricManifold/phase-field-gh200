#pragma once

#include <cstdint>

namespace pf3d {

// The number of CTAs assigned to one cell determines the grouping of its
// floating-point moment reduction. Checkpoints therefore store the resolved
// count, and a continuation restores that count exactly. The standard policy
// uses at most four CTAs per cell without adding population-wide waves when the
// population can fit in one wave.
constexpr int kBaseMeasureShardCap = 4;
constexpr int kMaximumBaseMeasureShards = 64;
constexpr int kBaseMeasureOneWave = -1;

// CLI policy: -1 fills one occupancy wave when possible (cap 64), 0 applies the
// same rule with a four-CTA cap, and 1..64 is an explicit count. If one CTA per
// cell already exceeds a wave, either automatic policy resolves to one.
constexpr bool valid_base_measure_policy(int policy) {
    return policy >= kBaseMeasureOneWave &&
           policy <= kMaximumBaseMeasureShards;
}

// Readers treat checkpoint value 0 as standard; 1..64 is an exact count.
constexpr bool valid_stored_base_measure_shards(std::uint64_t stored) {
    return stored <= static_cast<std::uint64_t>(kMaximumBaseMeasureShards);
}

constexpr bool valid_checkpoint_base_measure_shards(int stored) {
    return stored >= 0 && stored <= kMaximumBaseMeasureShards;
}

// Largest count whose population-wide block count fits in one occupancy wave,
// bounded to the supplied cap. Returns one when one block per cell already
// spans multiple waves.
constexpr int occupancy_wave_shards(int cells, int multiprocessors,
                                    int blocks_per_sm, int cap) {
    if (cells <= 0 || multiprocessors <= 0 || blocks_per_sm <= 0 || cap < 1)
        return 1;
    const std::int64_t wave_blocks =
        static_cast<std::int64_t>(multiprocessors) * blocks_per_sm;
    const std::int64_t fitting = wave_blocks / cells;
    const std::int64_t capped =
        fitting < static_cast<std::int64_t>(cap)
            ? fitting : static_cast<std::int64_t>(cap);
    return capped > 1 ? static_cast<int>(capped) : 1;
}

// Resolve a valid CLI policy against the measured-kernel occupancy.
constexpr int resolve_base_measure_shards(int policy, int cells,
                                          int multiprocessors,
                                          int blocks_per_sm) {
    if (policy > 0)
        return policy < kMaximumBaseMeasureShards
            ? policy : kMaximumBaseMeasureShards;
    return occupancy_wave_shards(cells, multiprocessors, blocks_per_sm,
                                  policy < 0 ? kMaximumBaseMeasureShards
                                             : kBaseMeasureShardCap);
}

enum class BaseMeasureResumeResult {
    Ok,
    InvalidCheckpoint,
    Mismatch
};

// Resolve the checkpoint contract on the destination device. An omitted CLI
// option restores an exact stored count; an explicit option must resolve to
// the same count. Stored zero applies the standard policy.
inline BaseMeasureResumeResult resolve_base_measure_resume(
    std::uint64_t stored, bool supplied, int policy, int cells,
    int multiprocessors, int blocks_per_sm, int* resolved) {
    if (resolved == nullptr || !valid_stored_base_measure_shards(stored))
        return BaseMeasureResumeResult::InvalidCheckpoint;
    const int effective = stored != 0
        ? static_cast<int>(stored)
        : resolve_base_measure_shards(0, cells, multiprocessors,
                                      blocks_per_sm);
    if (!supplied) {
        *resolved = effective;
        return BaseMeasureResumeResult::Ok;
    }
    if (!valid_base_measure_policy(policy))
        return BaseMeasureResumeResult::InvalidCheckpoint;
    const int requested = resolve_base_measure_shards(
        policy, cells, multiprocessors, blocks_per_sm);
    if (requested != effective)
        return BaseMeasureResumeResult::Mismatch;
    *resolved = effective;
    return BaseMeasureResumeResult::Ok;
}

}  // namespace pf3d
