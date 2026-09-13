#pragma once

#include "params.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace pf3d {

constexpr int kBrickTierQuantum = 16;
constexpr int kBrickGrowthIncrement = 32;
constexpr int kBrickShrinkMargin = 16;
constexpr std::uint64_t kBrickCompactionEvery = 1024;

PF3D_HD constexpr bool valid_brick_class_edge(int edge) {
    return edge > 0 && edge % kBrickAlignment == 0;
}

// Aim for sixteen new planes on each face. The final domain-limited tier may
// be smaller, but always retains the storage alignment and grows the cube.
PF3D_HD constexpr int next_brick_edge(int current, int maximum) {
    if (!valid_brick_class_edge(current) || maximum <= current) return 0;
    const int limit = maximum - maximum % kBrickAlignment;
    if (limit <= current) return 0;
    const std::int64_t requested = static_cast<std::int64_t>(current)
        + kBrickGrowthIncrement;
    const std::int64_t rounded =
        ((requested + kBrickTierQuantum - 1) / kBrickTierQuantum) * kBrickTierQuantum;
    return static_cast<int>(rounded < limit ? rounded : limit);
}

// Propose a centered crop, not a crop centered on the support bbox. Every
// support face must retain the spare margin; the caller must separately prove
// that the discarded shell is exactly zero. Empty/invalid bounds do not shrink.
PF3D_HD constexpr int shrink_brick_edge(
    int current, int minimum, const int lo[3], const int hi[3]) {
    if (!valid_brick_class_edge(current) || !valid_brick_class_edge(minimum)
        || minimum >= current || !lo || !hi) return current;
    int available = current;
    for (int axis = 0; axis < 3; ++axis) {
        if (lo[axis] < 0 || hi[axis] < lo[axis] || hi[axis] >= current)
            return current;
        const int upper = current - 1 - hi[axis];
        if (lo[axis] < available) available = lo[axis];
        if (upper < available) available = upper;
    }
    if (available <= kBrickShrinkMargin) return current;
    const std::int64_t needed = static_cast<std::int64_t>(current)
        - 2 * static_cast<std::int64_t>(available - kBrickShrinkMargin);
    const std::int64_t rounded =
        ((needed + kBrickTierQuantum - 1) / kBrickTierQuantum) * kBrickTierQuantum;
    if (rounded >= current) return current;
    const int candidate = static_cast<int>(rounded < minimum ? minimum : rounded);
    return candidate < current ? candidate : current;
}

// new_local = old_local + offset; new_origin = old_origin - offset.
PF3D_HD constexpr bool centered_resize_offset(int old_edge, int new_edge, int* offset) {
    if (!offset || !valid_brick_class_edge(old_edge)
        || !valid_brick_class_edge(new_edge)) return false;
    *offset = static_cast<int>((static_cast<std::int64_t>(new_edge) - old_edge) / 2);
    return true;
}

PF3D_HD constexpr bool checked_resize_origin(
    std::int64_t origin, int old_edge, int new_edge, std::int64_t* result) {
    int offset = 0;
    if (!result || !centered_resize_offset(old_edge, new_edge, &offset)) return false;
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if ((offset > 0 && origin < minimum + offset)
        || (offset < 0 && origin > maximum + offset)) return false;
    *result = origin - offset;
    return true;
}

// Signed zero is empty. Every other bit pattern, including subnormal values,
// NaNs and infinities, must prevent removal of its voxel.
PF3D_HD constexpr bool brick_shell_bits_are_zero(std::uint32_t bits) {
    return (bits & 0x7fffffffu) == 0u;
}

// Host oracle for the discarded shell only. Retained voxels still require the
// simulator's ordinary integrity checks. No phase threshold is used here.
inline bool zero_loss_centered_crop(
    const float* field, std::size_t count, int old_edge, int new_edge) {
    int resize_offset = 0;
    if (!field || new_edge > old_edge
        || !centered_resize_offset(old_edge, new_edge, &resize_offset)) return false;
    const auto edge = static_cast<std::size_t>(old_edge);
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (edge > maximum / edge || edge * edge > maximum / edge
        || count != edge * edge * edge) return false;
    const int begin = -resize_offset;
    const int end = begin + new_edge;
    for (int z = 0; z < old_edge; ++z)
        for (int y = 0; y < old_edge; ++y)
            for (int x = 0; x < old_edge; ++x) {
                if (x >= begin && x < end && y >= begin && y < end
                    && z >= begin && z < end) continue;
                const auto index = (static_cast<std::size_t>(z) * edge + y) * edge + x;
                std::uint32_t bits = 0;
                std::memcpy(&bits, field + index, sizeof(bits));
                if (!brick_shell_bits_are_zero(bits)) return false;
            }
    return true;
}

} // namespace pf3d
