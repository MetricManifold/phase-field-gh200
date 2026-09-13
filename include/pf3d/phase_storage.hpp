#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define PF3D_STORAGE_HD __host__ __device__
#else
#define PF3D_STORAGE_HD
#endif

namespace pf3d {

struct CellFieldPlaneRange3D {
    int logical_begin = 0;
    int stored_begin = 0;
    int count = 0;
};

// Logical support remains a B-cube. A positive cap stores at most z_cap
// planes; when B exceeds it, storage z is world z, without wrapping. This
// changes addresses only, not logical tile traversal or boundary conditions.
struct CellFieldStorage3D {
    int z_cap = 0;  // Zero retains cubic storage, including for raw-kernel users.

    PF3D_STORAGE_HD constexpr bool compact(int B) const {
        return z_cap > 0 && B > z_cap;
    }

    PF3D_STORAGE_HD constexpr int planes(int B) const {
        return B <= 0 || z_cap < 0 ? 0 : compact(B) ? z_cap : B;
    }

    PF3D_STORAGE_HD constexpr bool checked_words(int B, std::size_t* out) const {
        const int depth = planes(B);
        if (!out || depth == 0) return false;
        constexpr std::size_t limit = std::numeric_limits<std::size_t>::max();
        const auto edge = static_cast<std::size_t>(B);
        if (edge > limit / edge) return false;
        const std::size_t square = edge * edge;
        if (square > limit / static_cast<std::size_t>(depth)) return false;
        *out = square * static_cast<std::size_t>(depth);
        return true;
    }

    // Zero means invalid dimensions or an unrepresentable allocation size.
    PF3D_STORAGE_HD constexpr std::size_t words(int B) const {
        std::size_t result = 0;
        return checked_words(B, &result) ? result : 0;
    }

    PF3D_STORAGE_HD constexpr bool checked_bytes(int B, std::size_t* out) const {
        std::size_t count = 0;
        if (!out || !checked_words(B, &count) ||
            count > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return false;
        *out = count * sizeof(float);
        return true;
    }

    // Requires a validated allocation and an in-support point. In compact
    // storage origin_z + logical_z must also lie in [0,z_cap); ghost reflection
    // is resolved by the caller before indexing.
    PF3D_STORAGE_HD constexpr std::size_t index(
        int x, int y, int logical_z, int B, std::int64_t origin_z) const {
        const std::int64_t stored_z = compact(B)
            ? origin_z + static_cast<std::int64_t>(logical_z) : logical_z;
        return (static_cast<std::size_t>(stored_z) * static_cast<std::size_t>(B)
                + static_cast<std::size_t>(y)) * static_cast<std::size_t>(B)
                + static_cast<std::size_t>(x);
    }

    // The contiguous logical planes represented in storage. Cubic storage
    // retains all B planes; compact storage retains the intersection with the
    // bounded world-z array. Empty intersections succeed with an empty range.
    // Invalid dimensions, overflow, or a null output fail without writing it.
    PF3D_STORAGE_HD constexpr bool retained_planes(
        int B, std::int64_t origin_z, CellFieldPlaneRange3D* out) const {
        std::size_t count = 0;
        if (!out || !checked_words(B, &count)) return false;
        CellFieldPlaneRange3D result{};
        if (!compact(B)) {
            result.count = B;
        } else if (origin_z < static_cast<std::int64_t>(z_cap) &&
                   origin_z > -static_cast<std::int64_t>(B)) {
            // Nonempty intersection bounds origin_z before subtraction or
            // negation, including when the input origin is an integer limit.
            const std::int64_t begin = origin_z < 0 ? -origin_z : 0;
            const std::int64_t end = static_cast<std::int64_t>(z_cap) - origin_z;
            const std::int64_t clipped_end = end < B ? end : B;
            result.logical_begin = static_cast<int>(begin);
            result.stored_begin = static_cast<int>(origin_z + begin);
            result.count = static_cast<int>(clipped_end - begin);
        }
        *out = result;
        return true;
    }
};

} // namespace pf3d

#undef PF3D_STORAGE_HD
