#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define PF3D_ZMAP_HD __host__ __device__
#else
#define PF3D_ZMAP_HD
#endif

namespace pf3d {

// Mapping for one fixed brick origin in a bounded-z domain. Periodic callers
// retain their existing maps. Rebuild after changing the origin or brick size.
struct BoundedZMap {
    int lo = 0;
    int hi = 0;
    int origin = 0;
    bool reflect_lower = false;
    bool reflect_upper = false;

    PF3D_ZMAP_HD static constexpr BoundedZMap make(
        std::int64_t origin_z, int B, int nz, bool channel) {
        BoundedZMap map;
        if (B <= 0 || nz <= 0 || origin_z >= static_cast<std::int64_t>(nz)
            || origin_z <= -static_cast<std::int64_t>(B)) return map;

        // A nonempty intersection proves -B < origin_z < nz. All following
        // wide arithmetic is safe, including for extreme input origins.
        const std::int64_t first = origin_z < 0 ? -origin_z : 0;
        const std::int64_t end = static_cast<std::int64_t>(nz) - origin_z;
        map.lo = static_cast<int>(first);
        map.hi = static_cast<int>(end < B ? end : B);
        map.origin = static_cast<int>(origin_z);
        map.reflect_lower = origin_z <= 0;
        map.reflect_upper = channel && origin_z + static_cast<std::int64_t>(B) >= nz;
        return map;
    }

    // The aggregate caller already restricts local_z to this brick's [0,B).
    // The interval test additionally proves the signed sum lies in [0,nz).
    PF3D_ZMAP_HD constexpr bool aggregate(int local_z, int* world_z) const {
        if (local_z < lo || local_z >= hi) return false;
        *world_z = origin + local_z;
        return true;
    }

    // Accept arbitrary local offsets. Only the single reflected ghost next
    // to each enabled domain face aliases a real plane inside this brick.
    PF3D_ZMAP_HD constexpr bool fetch(int requested_z, int* source_z) const {
        if (requested_z >= lo && requested_z < hi) {
            *source_z = requested_z;
            return true;
        }
        if (reflect_lower && requested_z == lo - 1) {
            *source_z = lo;
            return true;
        }
        if (reflect_upper && requested_z == hi) {
            *source_z = hi - 1;
            return true;
        }
        return false;
    }
};

} // namespace pf3d

#undef PF3D_ZMAP_HD
