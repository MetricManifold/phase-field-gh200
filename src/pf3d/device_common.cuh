#pragma once

// Device helpers shared by the 3D kernel translation units.  These carry the
// solver's value semantics: exact Q5.27 aggregate arithmetic, sticky integrity
// flags, and the periodic index maps.  Both step implementations must produce
// identical values through them.

#include "../../include/pf3d/kernels.cuh"

#include <climits>
#include <cstdint>

namespace pf3d {

__device__ __forceinline__ std::size_t cell_words(int B) {
    return static_cast<std::size_t>(B) * static_cast<std::size_t>(B)
         * static_cast<std::size_t>(B);
}

__device__ __forceinline__ std::size_t local_index(int x, int y, int z, int B) {
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(B)
            + static_cast<std::size_t>(y)) * static_cast<std::size_t>(B)
            + static_cast<std::size_t>(x);
}

__device__ __forceinline__ int wrap_origin(std::int64_t origin, int n) {
    std::int64_t value = origin % static_cast<std::int64_t>(n);
    if (value < 0) value += n;
    return static_cast<int>(value);
}

__device__ __forceinline__ int wrap_offset(int wrapped_origin, int offset,
                                           int n) {
    int value = wrapped_origin + offset;
    if (value >= n) value -= n;
    return value;
}

__device__ __forceinline__ bool checked_world_coordinate(
    std::int64_t origin, int offset, std::int64_t* world) {
    if (world == nullptr ||
        (offset > 0 && origin > INT64_MAX - static_cast<std::int64_t>(offset)) ||
        (offset < 0 && origin < INT64_MIN - static_cast<std::int64_t>(offset)))
        return false;
    *world = origin + static_cast<std::int64_t>(offset);
    return true;
}

__device__ __forceinline__ bool checked_world_difference(
    std::int64_t world, std::int64_t origin, std::int64_t* local) {
    if (local == nullptr ||
        (origin > 0 && world < INT64_MIN + origin) ||
        (origin < 0 && world > INT64_MAX + origin))
        return false;
    *local = world - origin;
    return true;
}

// Map one local z coordinate into the dense aggregate field. Periodic runs
// retain their modulo map. Bounded-z geometries never fold a cell through a
// wall; callers skip an out-of-domain result.
__device__ __forceinline__ bool aggregate_z(
    const SLayout3D& layout, std::int64_t origin_z, int local_z,
    int* mapped_z) {
    if (layout.periodic_xyz()) {
        *mapped_z = wrap_offset(wrap_origin(origin_z, layout.nz), local_z,
                                layout.nz);
        return true;
    }
    std::int64_t world = 0;
    if (!checked_world_coordinate(origin_z, local_z, &world) ||
        world < 0 || world >= static_cast<std::int64_t>(layout.nz))
        return false;
    *mapped_z = static_cast<int>(world);
    return true;
}

// Local phase-field fetch at a z boundary. The substrate slab reflects its
// lower ghost and uses a zero far field at the top. A resolved channel reflects
// one ghost layer only at the outer allocation faces buried in solid padding.
// Periodic geometry keeps the existing local-brick zero halo.
__device__ __forceinline__ bool phase_fetch_local_z(
    const SLayout3D& layout, std::int64_t origin_z, int requested_local_z,
    int B, int* source_local_z) {
    if (layout.periodic_xyz()) {
        if (static_cast<unsigned>(requested_local_z) >=
            static_cast<unsigned>(B))
            return false;
        *source_local_z = requested_local_z;
        return true;
    }

    std::int64_t world = 0;
    if (!checked_world_coordinate(origin_z, requested_local_z, &world))
        return false;
    if (layout.hard_wall_channel() &&
        world == static_cast<std::int64_t>(layout.nz))
        world = static_cast<std::int64_t>(layout.nz) - 1;
    if (world >= static_cast<std::int64_t>(layout.nz)) return false;
    if (world < -1) return false;
    if (world == -1)
        world = 0;  // one-layer reflection about the lower face z=-1/2
    std::int64_t local = 0;
    if (!checked_world_difference(world, origin_z, &local)) return false;
    if (local < 0 || local >= static_cast<std::int64_t>(B)) return false;
    *source_local_z = static_cast<int>(local);
    return true;
}

__device__ __forceinline__ std::size_t s_index(const SLayout3D& layout,
                                                int x, int y, int z) {
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(layout.ny)
            + static_cast<std::size_t>(y))
           * static_cast<std::size_t>(layout.pitch_x)
           + static_cast<std::size_t>(x);
}

__device__ __forceinline__ void raise_flag(CellState3D* cell,
                                            std::uint32_t* global_flags,
                                            Flag3D flag) {
    atomicOr(&cell->flags, 1u << static_cast<unsigned>(flag));
    if (global_flags != nullptr)
        atomicAdd(&global_flags[static_cast<int>(flag)], 1u);
}

__device__ __forceinline__ bool fatal_global_flags_present(
    const std::uint32_t* global_flags) {
    if (global_flags == nullptr) return false;
    for (int flag = 0; flag < FLAG3D_COUNT; ++flag) {
        if (flag3d_is_fatal(static_cast<Flag3D>(flag)) &&
            global_flags[flag] != 0u)
            return true;
    }
    return false;
}

__device__ __forceinline__ bool fatal_flags_present(
    const CellState3D* cell, const std::uint32_t* global_flags) {
    return (cell->flags & kFatalFlagMask3D) != 0u
        || fatal_global_flags_present(global_flags);
}

__device__ __forceinline__ float other_field(std::uint32_t aggregate,
                                              float phi,
                                              CellState3D* cell,
                                              std::uint32_t* global_flags) {
    const std::uint32_t own = q_of(phi);
    if (aggregate >= own)
        return static_cast<float>(aggregate - own) * kQInvF;
    raise_flag(cell, global_flags, FLAG3D_S_NEGATIVE);
    return 0.0f;
}

__device__ __forceinline__ void scatter_value(std::uint32_t* S,
                                               std::size_t index,
                                               float phi,
                                               CellState3D* cell,
                                               std::uint32_t* global_flags) {
    const float square = phi * phi;
    if (square > kQClampPhiSq)
        raise_flag(cell, global_flags, FLAG3D_Q_CLAMP);
    const std::uint32_t value = q_of(phi);
    if (value == 0u) return;
    const std::uint32_t old = atomicAdd(&S[index], value);
    if (old > 0xffffffffu - value)
        raise_flag(cell, global_flags, FLAG3D_S_OVERFLOW);
}

// The 27-point evaluation over an arbitrary fetch functor: identical weights
// and summation order for every caller, so the staged layout never changes
// the arithmetic.
template <typename Fetch>
__device__ __forceinline__ void stencil27_at(Fetch fetch,
                                              float* laplacian,
                                              float* grad_x, float* grad_y,
                                              float* grad_z) {
    const float centre = fetch(0, 0, 0);
    float weighted = static_cast<float>(kLapCentreW) * centre;
#pragma unroll
    for (int dz = -1; dz <= 1; ++dz) {
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                const int distance = (dx != 0) + (dy != 0) + (dz != 0);
                if (distance == 0) continue;
                const int weight = distance == 1 ? kLapFaceW
                                 : distance == 2 ? kLapEdgeW
                                                 : kLapCornerW;
                weighted += static_cast<float>(weight) * fetch(dx, dy, dz);
            }
        }
    }
    *laplacian = weighted * static_cast<float>(1.0 / kLapDenom);
    *grad_x = 0.5f * (fetch(1, 0, 0) - fetch(-1, 0, 0));
    *grad_y = 0.5f * (fetch(0, 1, 0) - fetch(0, -1, 0));
    *grad_z = 0.5f * (fetch(0, 0, 1) - fetch(0, 0, -1));
}

// Register-resident evaluation over three staged columns (x-1, x, x+1); each
// column holds the 3x3 (y,z) neighbourhood as col[(dy+1)*3 + (dz+1)].
__device__ __forceinline__ void stencil27_cols(const float* cl,
                                                const float* cc,
                                                const float* cr,
                                                float* lap, float* gx,
                                                float* gy, float* gz) {
    auto fetch = [&](int dx, int dy, int dz) {
        const float* col = dx < 0 ? cl : dx > 0 ? cr : cc;
        return col[(dy + 1) * 3 + (dz + 1)];
    };
    stencil27_at(fetch, lap, gx, gy, gz);
}

}  // namespace pf3d
