#include "../../include/pf3d/kernels.cuh"
#include "../../include/pf3d/bounded_z_map.hpp"

#include "device_common.cuh"

#include <cuda_pipeline.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace pf3d {
namespace {

constexpr int kWarps3D = kThreads3D / 32;

using Reduction3D = MomentPartial3D;

constexpr bool recognized_boundary(const SLayout3D& layout) {
    return layout.periodic_xyz() || layout.substrate_slab() ||
           layout.hard_wall_channel();
}

// Pointer-backed cells may have different cube edges in the same launch.
// The host validates the maximum against the domain before enqueueing work.
__device__ __forceinline__ int promoted_cell_edge(
    const CellState3D& cell, int base_edge, int maximum_edge) {
    const std::uint32_t edge = cell.storage_edge;
    return edge > static_cast<std::uint32_t>(base_edge) &&
           edge <= static_cast<std::uint32_t>(maximum_edge) &&
           edge % kBrickAlignment == 0u ? static_cast<int>(edge) : 0;
}

// The resolved wall contributes (kappa_w/kappa) psi_w(z)^2 to the field that
// drives cell relaxation and passive translation.
template <bool WallCoupling>
__device__ __forceinline__ float weighted_wall_field(
    const WeightedWallField3D& wall, int world_z) {
    if constexpr (WallCoupling) return wall.values[world_z];
    return 0.0f;
}

__device__ __forceinline__ double warp_sum(double value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

__device__ __forceinline__ int warp_min(int value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        value = min(value, __shfl_down_sync(0xffffffffu, value, delta));
    return value;
}

__device__ __forceinline__ int warp_max(int value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        value = max(value, __shfl_down_sync(0xffffffffu, value, delta));
    return value;
}

__device__ __forceinline__ unsigned int warp_max_u32(unsigned int value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        value = max(value, __shfl_down_sync(0xffffffffu, value, delta));
    return value;
}

__device__ __forceinline__ Reduction3D reduce_cta(Reduction3D value,
                                                   Reduction3D* warp_values,
                                                   int B) {
    value.V       = warp_sum(value.V);
    value.Cx      = warp_sum(value.Cx);
    value.Cy      = warp_sum(value.Cy);
    value.Cz      = warp_sum(value.Cz);
    value.surface = warp_sum(value.surface);
    value.Ix      = warp_sum(value.Ix);
    value.Iy      = warp_sum(value.Iy);
    value.Iz      = warp_sum(value.Iz);
    value.lo_x = warp_min(value.lo_x); value.hi_x = warp_max(value.hi_x);
    value.lo_y = warp_min(value.lo_y); value.hi_y = warp_max(value.hi_y);
    value.lo_z = warp_min(value.lo_z); value.hi_z = warp_max(value.hi_z);
    value.phi_max_bits = warp_max_u32(value.phi_max_bits);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    Reduction3D result{};
    result.lo_x = result.lo_y = result.lo_z = B;
    result.hi_x = result.hi_y = result.hi_z = -1;
    if (threadIdx.x == 0) {
        for (int w = 0; w < kWarps3D; ++w) {
            const Reduction3D& q = warp_values[w];
            result.V += q.V; result.Cx += q.Cx; result.Cy += q.Cy;
            result.Cz += q.Cz; result.surface += q.surface;
            result.Ix += q.Ix; result.Iy += q.Iy; result.Iz += q.Iz;
            result.lo_x = min(result.lo_x, q.lo_x);
            result.hi_x = max(result.hi_x, q.hi_x);
            result.lo_y = min(result.lo_y, q.lo_y);
            result.hi_y = max(result.hi_y, q.hi_y);
            result.lo_z = min(result.lo_z, q.lo_z);
            result.hi_z = max(result.hi_z, q.hi_z);
            result.phi_max_bits = max(result.phi_max_bits, q.phi_max_bits);
        }
        warp_values[0] = result;
    }
    __syncthreads();
    return warp_values[0];
}

__device__ __forceinline__ bool load_halo(const float* tile, int B,
                                           int x0, int y0, int z0,
                                           const SLayout3D& layout,
                                           std::int64_t origin_z,
                                           const CellFieldStorage3D& storage,
                                           float* shared_phi) {
    bool any_nonzero = false;
    for (int q = static_cast<int>(threadIdx.x); q < kHaloVoxels;
         q += kThreads3D) {
        const int hx = q % kHaloX;
        const int hy = (q / kHaloX) % kHaloY;
        const int hz = q / (kHaloX * kHaloY);
        const int x = x0 + hx - 1;
        const int y = y0 + hy - 1;
        const int z = z0 + hz - 1;
        int source_z = 0;
        const float value =
            (static_cast<unsigned>(x) < static_cast<unsigned>(B) &&
             static_cast<unsigned>(y) < static_cast<unsigned>(B) &&
             phase_fetch_local_z(layout, origin_z, z, B, &source_z))
                ? tile[storage.index(x, y, source_z, B, origin_z)] : 0.0f;
        shared_phi[(hz * kHaloY + hy) * kHaloRowStride + hx] = value;
        any_nonzero = any_nonzero || value != 0.0f;
    }
    // The phase equation leaves an all-zero halo exactly zero. Skipping its
    // stencil work is therefore bitwise neutral; NaN also compares nonzero
    // and remains visible to the integrity checks below.
    return __syncthreads_or(any_nonzero ? 1 : 0) != 0;
}

__device__ __forceinline__ int halo_index(int x, int y, int z) {
    return (z * kHaloY + y) * kHaloRowStride + x;
}

__device__ __forceinline__ void stencil27(const float* shared_phi,
                                           int hx, int hy, int hz,
                                           float* laplacian,
                                           float* grad_x,
                                           float* grad_y,
                                           float* grad_z) {
    const int centre_index = halo_index(hx, hy, hz);
    const float centre = shared_phi[centre_index];
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
                weighted += static_cast<float>(weight)
                          * shared_phi[halo_index(hx + dx, hy + dy, hz + dz)];
            }
        }
    }
    *laplacian = weighted * static_cast<float>(1.0 / kLapDenom);
    *grad_x = 0.5f * (shared_phi[halo_index(hx + 1, hy, hz)]
                    - shared_phi[halo_index(hx - 1, hy, hz)]);
    *grad_y = 0.5f * (shared_phi[halo_index(hx, hy + 1, hz)]
                    - shared_phi[halo_index(hx, hy - 1, hz)]);
    *grad_z = 0.5f * (shared_phi[halo_index(hx, hy, hz + 1)]
                    - shared_phi[halo_index(hx, hy, hz - 1)]);
}

__device__ __forceinline__ void load_stencil_plane(
    const float* shared_phi, int ix, int iy, int hz, float* plane) {
#pragma unroll
    for (int y = 0; y < 3; ++y)
#pragma unroll
        for (int x = 0; x < 3; ++x)
            plane[y * 3 + x] = shared_phi[halo_index(ix + x, iy + y, hz)];
}

struct StencilValues3D {
    float centre, lap, gx, gy, gz;
};

template <int ZDistance, typename Fetch>
__device__ __forceinline__ float accumulate_stencil_plane(
    float weighted, const Fetch& fetch) {
#pragma unroll
    for (int y = 0; y < 3; ++y) {
#pragma unroll
        for (int x = 0; x < 3; ++x) {
            const int distance = ZDistance + (x != 1) + (y != 1);
            if (distance == 0) continue;
            const int weight = distance == 1 ? kLapFaceW
                             : distance == 2 ? kLapEdgeW : kLapCornerW;
            weighted += static_cast<float>(weight) * fetch(x, y);
        }
    }
    return weighted;
}

// Consume z-1 before replacing it with z+1. Two retained planes suffice,
// and the summation order stays z, then y, then x, as in stencil27.
__device__ __forceinline__ StencilValues3D advance_stencil_plane(
    const float* shared_phi, int ix, int iy, int iz,
    float* below, const float* middle) {
    const float lower_centre = below[4];
    StencilValues3D result{};
    result.centre = middle[4];
    float weighted = static_cast<float>(kLapCentreW) * result.centre;
    weighted = accumulate_stencil_plane<1>(weighted,
        [&](int x, int y) { return below[y * 3 + x]; });
    weighted = accumulate_stencil_plane<0>(weighted,
        [&](int x, int y) { return middle[y * 3 + x]; });
    weighted = accumulate_stencil_plane<1>(weighted, [&](int x, int y) {
        const float value = shared_phi[halo_index(ix + x, iy + y, iz + 2)];
        below[y * 3 + x] = value;
        return value;
    });
    result.lap = weighted * static_cast<float>(1.0 / kLapDenom);
    result.gx = 0.5f * (middle[5] - middle[3]);
    result.gy = 0.5f * (middle[7] - middle[1]);
    result.gz = 0.5f * (below[4] - lower_centre);
    return result;
}

// The measurement and surface passes use only the centred gradients, so the
// Laplacian accumulation of stencil27 is dead work there.  These expressions
// are verbatim the gradient tail of stencil27.
__device__ __forceinline__ void gradient3(const float* shared_phi,
                                           int hx, int hy, int hz,
                                           float* grad_x,
                                           float* grad_y,
                                           float* grad_z) {
    *grad_x = 0.5f * (shared_phi[halo_index(hx + 1, hy, hz)]
                    - shared_phi[halo_index(hx - 1, hy, hz)]);
    *grad_y = 0.5f * (shared_phi[halo_index(hx, hy + 1, hz)]
                    - shared_phi[halo_index(hx, hy - 1, hz)]);
    *grad_z = 0.5f * (shared_phi[halo_index(hx, hy, hz + 1)]
                    - shared_phi[halo_index(hx, hy, hz - 1)]);
}

// Two shared-memory halo slabs overlap the next tile's asynchronous load with
// the current stencil. They reproduce the synchronous loader's values exactly.
template <typename FetchZ>
__device__ __forceinline__ void prefetch_halo(const float* tile, int B,
                                               int tile_index, int tiles_x,
                                               int tiles_y,
                                               const FetchZ& fetch_z,
                                               int storage_z_offset,
                                               float* shared_buf) {
    const int tx = tile_index % tiles_x;
    const int ty = (tile_index / tiles_x) % tiles_y;
    const int tz = tile_index / (tiles_x * tiles_y);
    const int x0 = tx * kBrickX - 1;
    const int y0 = ty * kBrickY - 1;
    const int z0 = tz * kBrickZ - 1;
    constexpr int vectors_per_row = kBrickX / 4;
    constexpr int halo_rows = kHaloY * kHaloZ;
    for (int q = static_cast<int>(threadIdx.x);
         q < halo_rows * vectors_per_row; q += kThreads3D) {
        const int row = q / vectors_per_row;
        const int hx = 1 + 4 * (q % vectors_per_row);
        const int x = x0 + hx;
        const int y = y0 + row % kHaloY;
        const int z = z0 + row / kHaloY;
        float* const destination = shared_buf + row * kHaloRowStride + hx;
        int source_z = 0;
        if (static_cast<unsigned>(x) < static_cast<unsigned>(B) &&
            x + 3 < B &&
            static_cast<unsigned>(y) < static_cast<unsigned>(B) &&
            fetch_z(z, &source_z)) {
            __pipeline_memcpy_async(destination,
                &tile[local_index(x, y, source_z + storage_z_offset, B)],
                sizeof(float4));
        } else {
            *reinterpret_cast<float4*>(destination) = make_float4(0, 0, 0, 0);
        }
    }
    // Ghost columns keep the same source/reflection rules as the scalar loader.
    for (int q = static_cast<int>(threadIdx.x); q < 2 * halo_rows;
         q += kThreads3D) {
        const int row = q / 2;
        const int hx = (q % 2) * (kHaloX - 1);
        const int x = x0 + hx;
        const int y = y0 + row % kHaloY;
        const int z = z0 + row / kHaloY;
        float* const destination = shared_buf + row * kHaloRowStride + hx;
        int source_z = 0;
        if (static_cast<unsigned>(x) < static_cast<unsigned>(B) &&
            static_cast<unsigned>(y) < static_cast<unsigned>(B) &&
            fetch_z(z, &source_z)) {
            __pipeline_memcpy_async(destination,
                &tile[local_index(x, y, source_z + storage_z_offset, B)],
                sizeof(float));
        } else {
            *destination = 0.0f;
        }
    }
    __pipeline_commit();
}

__device__ __forceinline__ void prefetch_halo(const float* tile, int B,
                                               int tile_index, int tiles_x,
                                               int tiles_y,
                                               const SLayout3D& layout,
                                               std::int64_t origin_z,
                                               const CellFieldStorage3D& storage,
                                               float* shared_buf) {
    const auto fetch_z = [&](int z, int* source_z) {
        return phase_fetch_local_z(layout, origin_z, z, B, source_z);
    };
    const int storage_z_offset = storage.compact(B) ? static_cast<int>(origin_z) : 0;
    prefetch_halo(tile, B, tile_index, tiles_x, tiles_y, fetch_z,
                  storage_z_offset, shared_buf);
}

// The phase equation leaves an all-zero halo exactly zero, so skipping such a
// tile's stencil work is bitwise neutral; NaN also compares nonzero and stays
// visible to the integrity checks. Each thread scans only words it staged and
// waited on; the final collective publishes all copies before stencil reads.
__device__ __forceinline__ bool halo_any_nonzero(const float* shared_buf) {
    bool any = false;
    // Match prefetch_halo's float4-interior and scalar-ghost thread ownership.
    constexpr int rows = kHaloY * kHaloZ, vectors_per_row = kBrickX / 4;
    for (int q = static_cast<int>(threadIdx.x); q < rows * vectors_per_row;
         q += kThreads3D) {
        const int row = q / vectors_per_row, hx = 1 + 4 * (q % vectors_per_row);
        const float4 v = *reinterpret_cast<const float4*>(
            shared_buf + row * kHaloRowStride + hx);
        // Floating comparisons treat signed zero as zero and keep NaN visible.
        any = any || ((v.x != 0.0f) | (v.y != 0.0f)
                    | (v.z != 0.0f) | (v.w != 0.0f));
    }
    for (int q = static_cast<int>(threadIdx.x); q < 2 * rows; q += kThreads3D) {
        const int row = q / 2, hx = (q % 2) * (kHaloX - 1);
        any = any || shared_buf[row * kHaloRowStride + hx] != 0.0f;
    }
    return __syncthreads_or(any ? 1 : 0) != 0;
}

// Bounded-z bricks can extend beyond the allocated z array. Planes wholly
// outside that array are exact or implicit zeros,
// contribute nothing to any accumulator, write nothing, and raise nothing,
// so the walk can omit their tiles before staging halos.  Every tile with at
// least one allocated plane is kept, which preserves the reflected z=-1
// outer-array ghost: the tile containing world z=0 stages it during prefetch.
// Periodic z keeps the full range.  z is the slowest tile coordinate, so the
// live tiles form one contiguous index range and shard assignment by
// tile % shards is unchanged for every surviving tile.
__device__ __forceinline__ void live_tile_z_range(const SLayout3D& layout,
                                                   std::int64_t origin_z,
                                                   int B, int tiles_z,
                                                   int* tz_begin,
                                                   int* tz_end) {
    if (!layout.bounded_z()) {
        *tz_begin = 0;
        *tz_end = tiles_z;
        return;
    }
    const std::int64_t z_lo = -origin_z;
    const std::int64_t z_hi =
        static_cast<std::int64_t>(layout.nz) - 1 - origin_z;
    const std::int64_t live_lo = z_lo > 0 ? z_lo : 0;
    const std::int64_t live_hi =
        z_hi < static_cast<std::int64_t>(B) - 1
            ? z_hi : static_cast<std::int64_t>(B) - 1;
    if (live_lo > live_hi) {
        *tz_begin = 0;
        *tz_end = 0;
        return;
    }
    *tz_begin = static_cast<int>(live_lo / kBrickZ);
    *tz_end = static_cast<int>(live_hi / kBrickZ) + 1;
}

__device__ __forceinline__ int clamp_shift(int proposed, int limit,
                                            int bbox_lo, int bbox_hi, int B) {
    proposed = max(-limit, min(limit, proposed));
    if (bbox_hi >= bbox_lo) {
        // Every supported source voxel must remain representable after x'=x-s.
        proposed = min(proposed, bbox_lo);
        proposed = max(proposed, bbox_hi - (B - 1));
    }
    return proposed;
}

__device__ __forceinline__ Reduction3D empty_reduction(int B) {
    Reduction3D result{};
    result.lo_x = result.lo_y = result.lo_z = B;
    result.hi_x = result.hi_y = result.hi_z = -1;
    return result;
}

__device__ __forceinline__ void append_reduction(Reduction3D* total,
                                                  const Reduction3D& part) {
    total->V += part.V;
    total->Cx += part.Cx;
    total->Cy += part.Cy;
    total->Cz += part.Cz;
    total->surface += part.surface;
    total->Ix += part.Ix;
    total->Iy += part.Iy;
    total->Iz += part.Iz;
    total->lo_x = min(total->lo_x, part.lo_x);
    total->hi_x = max(total->hi_x, part.hi_x);
    total->lo_y = min(total->lo_y, part.lo_y);
    total->hi_y = max(total->hi_y, part.hi_y);
    total->lo_z = min(total->lo_z, part.lo_z);
    total->hi_z = max(total->hi_z, part.hi_z);
    total->phi_max_bits = max(total->phi_max_bits, part.phi_max_bits);
}

__device__ __forceinline__ void commit_measurement(
    const MeasureArgs3D& args, int n, int B, const Reduction3D& total) {
    CellState3D* state = &args.cells[n];
    state->V = total.V;
    state->Cx = total.Cx;
    state->Cy = total.Cy;
    state->Cz = total.Cz;
    if (args.compute_surface) state->surface = total.surface;
    state->Ix = total.Ix;
    state->Iy = total.Iy;
    state->Iz = total.Iz;
    state->bb_lo_x = total.lo_x;
    state->bb_hi_x = total.hi_x;
    state->bb_lo_y = total.lo_y;
    state->bb_hi_y = total.hi_y;
    state->bb_lo_z = total.lo_z;
    state->bb_hi_z = total.hi_z;
    state->phi_max = __uint_as_float(total.phi_max_bits);

    int shift_x = 0, shift_y = 0, shift_z = 0;
    if (total.V > 0.0 && isfinite(total.V)) {
        const double middle = 0.5 * static_cast<double>(B - 1);
        shift_x = __double2int_rn(total.Cx / total.V - middle);
        shift_y = __double2int_rn(total.Cy / total.V - middle);
        shift_z = __double2int_rn(total.Cz / total.V - middle);
        shift_x = clamp_shift(shift_x, args.max_shift,
                              total.lo_x, total.hi_x, B);
        shift_y = clamp_shift(shift_y, args.max_shift,
                              total.lo_y, total.hi_y, B);
        shift_z = clamp_shift(shift_z, args.max_shift,
                              total.lo_z, total.hi_z, B);
    } else {
        raise_flag(state, args.global_flags, FLAG3D_V_NONPOSITIVE);
    }
    state->pending_shift_x = shift_x;
    state->pending_shift_y = shift_y;
    state->pending_shift_z = shift_z;

    if (total.hi_x >= total.lo_x) {
        const int ex = total.hi_x - total.lo_x + 1;
        const int ey = total.hi_y - total.lo_y + 1;
        const int ez = total.hi_z - total.lo_z + 1;
        // Width alone misses an asymmetric tail near one face. Check the
        // bounds where the planned recentring shift will place them.
        const int lower_margin = args.support_margin / 2;
        const int upper_margin = args.support_margin - lower_margin;
        const int shifted_lo_x = total.lo_x - shift_x;
        const int shifted_hi_x = total.hi_x - shift_x;
        const int shifted_lo_y = total.lo_y - shift_y;
        const int shifted_hi_y = total.hi_y - shift_y;
        const int shifted_lo_z = total.lo_z - shift_z;
        const int shifted_hi_z = total.hi_z - shift_z;
        if (ex + args.support_margin > B ||
            ey + args.support_margin > B ||
            ez + args.support_margin > B ||
            shifted_lo_x < lower_margin ||
            shifted_hi_x >= B - upper_margin ||
            shifted_lo_y < lower_margin ||
            shifted_hi_y >= B - upper_margin ||
            shifted_lo_z < lower_margin ||
            shifted_hi_z >= B - upper_margin)
            raise_flag(state, args.global_flags, FLAG3D_SUPPORT_EXHAUSTED);
#if defined(PF_ALARMS)
        if (total.lo_x == 0 || total.hi_x == B - 1 ||
            total.lo_y == 0 || total.hi_y == B - 1 ||
            total.lo_z == 0 || total.hi_z == B - 1)
            raise_flag(state, args.global_flags, FLAG3D_SUPPORT_EDGE);
#endif
    }
}

template <bool CollectMoments, bool WallCoupling>
__device__ void process_update_cell(int n, const UpdateArgs3D& args,
                                    const float* source, float* destination,
                                    int B, float* halo_pair,
                                    Reduction3D* warp_values,
                                    int* abort_flag) {
    const std::size_t words = stored_cell_words(B, args.storage);
    CellState3D* state = &args.cells[n];

    // Measurement completes before this kernel in the same stream.  If it
    // found any invalid state, preserve the last accepted field exactly.  This
    // prevents a geometry or integrity failure from turning into partial data
    // loss while the host is waiting to observe the sticky counters.
    if (threadIdx.x == 0) {
        const int failed = fatal_flags_present(state, args.global_flags) ? 1 : 0;
        if constexpr (CollectMoments)
            warp_values[0].hi_x = failed;
        else
            *abort_flag = failed;
    }
    __syncthreads();
    int abort = 0;
    if constexpr (CollectMoments)
        abort = warp_values[0].hi_x;
    else
        abort = *abort_flag;
    if (abort != 0) {
        const int origin_x = wrap_origin(state->origin_x, args.layout.nx);
        const int origin_y = wrap_origin(state->origin_y, args.layout.ny);
        for (std::size_t q = threadIdx.x; q < words; q += kThreads3D) {
            const float value = source[q];
            destination[q] = value;
            if (args.S_out != nullptr && isfinite(value)) {
                const int x = static_cast<int>(q % static_cast<std::size_t>(B));
                const int y = static_cast<int>((q / static_cast<std::size_t>(B))
                                              % static_cast<std::size_t>(B));
                const int stored_z = static_cast<int>(
                    q / (static_cast<std::size_t>(B) * B));
                int z = 0;
                if (!logical_z_from_stored(stored_z, B, state->origin_z,
                                           args.storage, &z)) continue;
                int wz = 0;
                if (!aggregate_z(args.layout, state->origin_z, z, &wz))
                    continue;
                scatter_value(
                    args.S_out,
                    s_index(args.layout,
                            wrap_offset(origin_x, x, args.layout.nx),
                            wrap_offset(origin_y, y, args.layout.ny),
                            wz),
                    value, state, args.global_flags);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            state->pending_shift_x = 0;
            state->pending_shift_y = 0;
            state->pending_shift_z = 0;
        }
        __syncthreads();
        return;
    }

    for (std::size_t q = threadIdx.x; q < words; q += kThreads3D)
        destination[q] = 0.0f;
    __syncthreads();

    const int shift_x = state->pending_shift_x;
    const int shift_y = state->pending_shift_y;
    const int shift_z = state->pending_shift_z;
    std::int64_t destination_origin_x = 0;
    std::int64_t destination_origin_y = 0;
    std::int64_t destination_origin_z = 0;
    const bool destination_origin_valid =
        checked_world_coordinate(state->origin_x, shift_x,
                                 &destination_origin_x) &&
        checked_world_coordinate(state->origin_y, shift_y,
                                 &destination_origin_y) &&
        checked_world_coordinate(state->origin_z, shift_z,
                                 &destination_origin_z);
    if (threadIdx.x == 0 && !destination_origin_valid)
        raise_flag(state, args.global_flags, FLAG3D_INVALID_GEOMETRY);
    __syncthreads();
    if (!destination_origin_valid) return;
    const int origin_x = wrap_origin(state->origin_x, args.layout.nx);
    const int origin_y = wrap_origin(state->origin_y, args.layout.ny);
    // Load cell-wide values once because pointer aliasing prevents reliable
    // compiler hoisting inside the voxel loop.
    const float gamma = state->gamma;
    const float velocity_x = state->velocity_x;
    const float velocity_y = state->velocity_y;
    const float velocity_z = state->velocity_z;
    const float bulk = args.bulk_scale * gamma;
    const float volume = static_cast<float>(
        args.volume_scale * (args.V0 - state->V));

    [[maybe_unused]] Reduction3D local{};
    if constexpr (CollectMoments) {
        local.lo_x = local.lo_y = local.lo_z = B;
        local.hi_x = local.hi_y = local.hi_z = -1;
    }

    const int tiles_x = (B + kBrickX - 1) / kBrickX;
    const int tiles_y = (B + kBrickY - 1) / kBrickY;
    const int tiles_z = (B + kBrickZ - 1) / kBrickZ;
    int tz_begin = 0, tz_end = tiles_z;
    live_tile_z_range(args.layout, state->origin_z, B, tiles_z,
                      &tz_begin, &tz_end);
    const int tiles_per_plane = tiles_x * tiles_y;
    const int tile_begin = tz_begin * tiles_per_plane;
    const int tile_end = tz_end * tiles_per_plane;
    float* const halo0 = halo_pair + kHaloLeadingPadding;
    float* const halo1 = halo0 + kHaloBufferWords;
    int parity = 0;
    if (tile_begin < tile_end)
        prefetch_halo(source, B, tile_begin, tiles_x, tiles_y, args.layout,
                      state->origin_z, args.storage, halo0);
    // Preserve x-fastest tile order while streaming the next halo during the
    // current tile's stencil work.
    for (int tile = tile_begin; tile < tile_end; ++tile) {
        const bool more = tile + 1 < tile_end;
        if (more)
            prefetch_halo(source, B, tile + 1, tiles_x, tiles_y,
                          args.layout, state->origin_z, args.storage,
                          parity == 0 ? halo1 : halo0);
        __pipeline_wait_prior(more ? 1 : 0);
        float* const shared_phi = parity == 0 ? halo0 : halo1;
        if (halo_any_nonzero(shared_phi)) {
            const int tx = tile % tiles_x;
            const int ty = (tile / tiles_x) % tiles_y;
            const int tz = tile / (tiles_x * tiles_y);
            const int x0 = tx * kBrickX;
            const int y0 = ty * kBrickY;
            const int z0 = tz * kBrickZ;
            for (int q = static_cast<int>(threadIdx.x); q < kBrickVoxels;
                 q += kThreads3D) {
                const int ix = q % kBrickX;
                const int iy = (q / kBrickX) % kBrickY;
                const int iz = q / (kBrickX * kBrickY);
                const int x = x0 + ix, y = y0 + iy, z = z0 + iz;
                if (x >= B || y >= B || z >= B) continue;
                int wz = 0;
                if (!aggregate_z(args.layout, state->origin_z, z, &wz))
                    continue;

                const int hi = halo_index(ix + 1, iy + 1, iz + 1);
                const float centre = shared_phi[hi];
                if (!isfinite(centre)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    continue;
                }
                float lap, gx, gy, gz;
                stencil27(shared_phi, ix + 1, iy + 1, iz + 1,
                          &lap, &gx, &gy, &gz);

                // Most of a conservatively sized brick is exact zero.
                // A zero centre with a zero stencil remains zero under
                // every term in the equation, so avoid its aggregate-field
                // read and destination work.
                if (centre == 0.0f && lap == 0.0f && gx == 0.0f &&
                    gy == 0.0f && gz == 0.0f)
                    continue;

                const int wx = wrap_offset(origin_x, x, args.layout.nx);
                const int wy = wrap_offset(origin_y, y, args.layout.ny);
                const std::uint32_t aggregate =
                    args.S_in[s_index(args.layout, wx, wy, wz)];
                float other = other_field(aggregate, centre, state,
                                          args.global_flags);
                if constexpr (WallCoupling)
                    other += weighted_wall_field<true>(
                        args.wall, wz);
                const float rhs = gamma * lap
                    - bulk * centre * (1.0f - centre) * (1.0f - 2.0f * centre)
                    + volume * centre
                    - args.interaction_coeff * centre * other
                    - (velocity_x * gx + velocity_y * gy
                       + velocity_z * gz);
                const float next = centre + args.dt * rhs;
                if (!isfinite(next)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    continue;
                }
                if (args.layout.substrate_slab() &&
                    wz == args.layout.nz - 1 && fabsf(next) > kSupportEps) {
                    raise_flag(state, args.global_flags,
                               FLAG3D_SLAB_TOP_CONTACT);
                    continue;
                }

                const int dx = x - shift_x;
                const int dy = y - shift_y;
                const int dz = z - shift_z;
                if (static_cast<unsigned>(dx) >= static_cast<unsigned>(B) ||
                    static_cast<unsigned>(dy) >= static_cast<unsigned>(B) ||
                    static_cast<unsigned>(dz) >= static_cast<unsigned>(B)) {
                    if (fabsf(next) > kSupportEps)
                        raise_flag(state, args.global_flags,
                                   FLAG3D_DESTINATION_CLIP);
                    continue;
                }
                destination[field_index_at_world_z(dx, dy, dz, wz, B, args.storage)] = next;

                if (args.S_out != nullptr) {
                    // (new_origin + destination) equals (old_origin + source).
                    scatter_value(args.S_out,
                                  s_index(args.layout, wx, wy, wz), next,
                                  state, args.global_flags);
                }

                if constexpr (CollectMoments) {
                    const double square = static_cast<double>(next)
                                        * static_cast<double>(next);
                    local.V += square;
                    local.Cx += square * static_cast<double>(dx);
                    local.Cy += square * static_cast<double>(dy);
                    local.Cz += square * static_cast<double>(dz);
                    local.phi_max_bits = max(
                        local.phi_max_bits, __float_as_uint(fabsf(next)));
                    if (fabsf(next) > kSupportEps) {
                        local.lo_x = min(local.lo_x, dx);
                        local.hi_x = max(local.hi_x, dx);
                        local.lo_y = min(local.lo_y, dy);
                        local.hi_y = max(local.hi_y, dy);
                        local.lo_z = min(local.lo_z, dz);
                        local.hi_z = max(local.hi_z, dz);
                    }
                }
            }
        }
        __syncthreads();
        parity = 1 - parity;
    }

    // The surface proxy is integral |grad(phi)|.  Unlike volume and support,
    // it cannot be obtained exactly while neighbouring destination values are
    // still being produced.  Evaluate it in a separate pass only at the
    // requested full-moment cadence, matching the 2D perimeter policy.
    if constexpr (CollectMoments) if (args.compute_surface) {
        __threadfence_block();
        __syncthreads();
        int surface_tz_begin = 0, surface_tz_end = tiles_z;
        live_tile_z_range(args.layout, destination_origin_z, B, tiles_z,
                          &surface_tz_begin, &surface_tz_end);
        for (int z0 = surface_tz_begin * kBrickZ;
             z0 < surface_tz_end * kBrickZ; z0 += kBrickZ) {
            for (int y0 = 0; y0 < B; y0 += kBrickY) {
                for (int x0 = 0; x0 < B; x0 += kBrickX) {
                    if (!load_halo(destination, B, x0, y0, z0, args.layout,
                                   destination_origin_z, args.storage, halo0))
                        continue;
                    for (int q = static_cast<int>(threadIdx.x);
                         q < kBrickVoxels; q += kThreads3D) {
                        const int ix = q % kBrickX;
                        const int iy = (q / kBrickX) % kBrickY;
                        const int iz = q / (kBrickX * kBrickY);
                        if (x0 + ix >= B || y0 + iy >= B || z0 + iz >= B)
                            continue;
                        int physical_z = 0;
                        if (!aggregate_z(args.layout,
                                         destination_origin_z,
                                         z0 + iz, &physical_z))
                            continue;
                        float gx, gy, gz;
                        gradient3(halo0, ix + 1, iy + 1, iz + 1,
                                  &gx, &gy, &gz);
                        local.surface += sqrt(static_cast<double>(gx) * gx
                                            + static_cast<double>(gy) * gy
                                            + static_cast<double>(gz) * gz);
                    }
                    __syncthreads();
                }
            }
        }
    }

    if constexpr (CollectMoments) {
        const Reduction3D total = reduce_cta(local, warp_values, B);
        if (threadIdx.x == 0) {
            if (!fatal_flags_present(state, args.global_flags)) {
                state->V = total.V;
                state->Cx = total.Cx; state->Cy = total.Cy; state->Cz = total.Cz;
                state->bb_lo_x = total.lo_x; state->bb_hi_x = total.hi_x;
                state->bb_lo_y = total.lo_y; state->bb_hi_y = total.hi_y;
                state->bb_lo_z = total.lo_z; state->bb_hi_z = total.hi_z;
                state->phi_max = __uint_as_float(total.phi_max_bits);
                if (args.compute_surface) state->surface = total.surface;
            } else {
                state->pending_shift_x = 0;
                state->pending_shift_y = 0;
                state->pending_shift_z = 0;
            }
        }
    }
    __syncthreads();
}

template <UpdateTilePass3D Pass>
__device__ __forceinline__ int next_update_tile(
    int tile, int tile_end, int shards_per_cell, int tiles_x,
    int tiles_per_plane, int B, const UpdateArgs3D& args,
    int origin_y, std::int64_t origin_z) {
    if constexpr (Pass != UpdateTilePass3D::All) {
        for (; tile < tile_end; tile += shards_per_cell) {
            const int tz = tile / tiles_per_plane;
            const int ty = (tile - tz * tiles_per_plane) / tiles_x;
            const int y0 = ty * kBrickY;
            const int z0 = tz * kBrickZ;
            const bool boundary = tile_intersects_exchange_canonical(
                args.exchange_region, args.layout.ny,
                origin_y, origin_z, y0, z0,
                min(kBrickY, B - y0), min(kBrickZ, B - z0));
            if (boundary == (Pass == UpdateTilePass3D::Boundary)) break;
        }
    }
    return tile;
}

template <bool CollectMoments, bool WallCoupling,
          UpdateTilePass3D Pass = UpdateTilePass3D::All>
__device__ void process_update_shard(int n, int shard, int shards_per_cell,
                                     const UpdateArgs3D& args,
                                     const float* source, float* destination,
                                     int B,
                                     MomentPartial3D* partials,
                                     float* halo_pair,
                                     Reduction3D* warp_values,
                                     int* abort_flag) {
    CellState3D* state = &args.cells[n];

    [[maybe_unused]] Reduction3D local{};
    if constexpr (CollectMoments) {
        local.lo_x = local.lo_y = local.lo_z = B;
        local.hi_x = local.hi_y = local.hi_z = -1;
    }

    if (threadIdx.x == 0)
        *abort_flag = fatal_flags_present(state, args.global_flags) ? 1 : 0;
    __syncthreads();
    if (*abort_flag == 0) {
        const int shift_x = state->pending_shift_x;
        const int shift_y = state->pending_shift_y;
        const int shift_z = state->pending_shift_z;
        std::int64_t destination_origin_x = 0;
        std::int64_t destination_origin_y = 0;
        std::int64_t destination_origin_z = 0;
        const bool destination_origin_valid =
            checked_world_coordinate(state->origin_x, shift_x,
                                     &destination_origin_x) &&
            checked_world_coordinate(state->origin_y, shift_y,
                                     &destination_origin_y) &&
            checked_world_coordinate(state->origin_z, shift_z,
                                     &destination_origin_z);
        if (threadIdx.x == 0 && !destination_origin_valid)
            raise_flag(state, args.global_flags, FLAG3D_INVALID_GEOMETRY);
        __syncthreads();
        if (!destination_origin_valid) return;
        const int origin_x = wrap_origin(state->origin_x, args.layout.nx);
        const int origin_y = wrap_origin(state->origin_y, args.layout.ny);
        // The source origin stays fixed until every update shard completes.
        const std::int64_t origin_z = state->origin_z;
        const bool bounded_z = WallCoupling || Pass != UpdateTilePass3D::All
                            || args.layout.bounded_z();
        const auto z_map = bounded_z
            ? BoundedZMap::make(origin_z, B, args.layout.nz,
                               args.layout.hard_wall_channel())
            : BoundedZMap{};
        const auto fetch_z = [&](int z, int* source_z) {
            return bounded_z ? z_map.fetch(z, source_z)
                : phase_fetch_local_z(args.layout, origin_z, z, B, source_z);
        };
        const int storage_z_offset = args.storage.compact(B)
            ? static_cast<int>(origin_z) : 0;
        // Constant for the whole brick; see process_update_cell.
        const float gamma = state->gamma;
        const float velocity_x = state->velocity_x;
        const float velocity_y = state->velocity_y;
        const float velocity_z = state->velocity_z;
        const float bulk = args.bulk_scale * gamma;
        const float volume = static_cast<float>(
            args.volume_scale * (args.V0 - state->V));
        const int tiles_x = (B + kBrickX - 1) / kBrickX;
        const int tiles_y = (B + kBrickY - 1) / kBrickY;
        const int tiles_z = (B + kBrickZ - 1) / kBrickZ;
        const int tiles_per_plane = tiles_x * tiles_y;

        int tz_begin = 0, tz_end = tiles_z;
        live_tile_z_range(args.layout, state->origin_z, B, tiles_z,
                          &tz_begin, &tz_end);
        const int tile_begin = tz_begin * tiles_per_plane;
        const int tile_end = tz_end * tiles_per_plane;

        // The shard assignment is independent of scheduling: tile t belongs
        // to t % shards_per_cell over the unchanged global tile numbering,
        // so bounding the walk to live planes removes tiles without moving
        // any surviving tile between shards.  Source-to-destination
        // recentering is a translation, so different source tiles also have
        // disjoint outputs.
        const int begin_remainder = tile_begin % shards_per_cell;
        const int first_tile = next_update_tile<Pass>(
            tile_begin
                + (shard - begin_remainder + shards_per_cell) % shards_per_cell,
            tile_end, shards_per_cell, tiles_x, tiles_per_plane, B,
            args, origin_y, state->origin_z);
        float* const halo0 = halo_pair + kHaloLeadingPadding;
        float* const halo1 = halo0 + kHaloBufferWords;
        int parity = 0;
        if (first_tile < tile_end)
            prefetch_halo(source, B, first_tile, tiles_x, tiles_y,
                          fetch_z, storage_z_offset, halo0);
        int next_tile = first_tile;
        for (int tile = first_tile; tile < tile_end; tile = next_tile) {
            next_tile = next_update_tile<Pass>(
                tile + shards_per_cell, tile_end, shards_per_cell,
                tiles_x, tiles_per_plane, B, args, origin_y, state->origin_z);
            const bool more = next_tile < tile_end;
            if (more)
                prefetch_halo(source, B, next_tile,
                              tiles_x, tiles_y, fetch_z, storage_z_offset,
                              parity == 0 ? halo1 : halo0);
            __pipeline_wait_prior(more ? 1 : 0);
            float* const shared_phi = parity == 0 ? halo0 : halo1;
            if (!halo_any_nonzero(shared_phi)) {
                __syncthreads();
                parity = 1 - parity;
                continue;
            }
            const int tz = tile / tiles_per_plane;
            const int remainder = tile - tz * tiles_per_plane;
            const int ty = remainder / tiles_x;
            const int tx = remainder - ty * tiles_x;
            const int x0 = tx * kBrickX;
            const int y0 = ty * kBrickY;
            const int z0 = tz * kBrickZ;

            const auto update_voxel = [&](int ix, int iy, int iz,
                                          StencilValues3D values) {
                const int x = x0 + ix, y = y0 + iy, z = z0 + iz;
                if (x >= B || y >= B || z >= B) return;
                int wz = 0;
                const bool mapped = bounded_z ? z_map.aggregate(z, &wz)
                    : aggregate_z(args.layout, origin_z, z, &wz);
                if (!mapped) return;

                const int hi = halo_index(ix + 1, iy + 1, iz + 1);
                const float centre = CollectMoments ? shared_phi[hi] : values.centre;
                if (!isfinite(centre)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    return;
                }
                float lap = values.lap, gx = values.gx;
                float gy = values.gy, gz = values.gz;
                if constexpr (CollectMoments)
                    stencil27(shared_phi, ix + 1, iy + 1, iz + 1,
                              &lap, &gx, &gy, &gz);
                if (centre == 0.0f && lap == 0.0f && gx == 0.0f &&
                    gy == 0.0f && gz == 0.0f)
                    return;

                const int wx = wrap_offset(origin_x, x, args.layout.nx);
                const int wy = wrap_offset(origin_y, y, args.layout.ny);
                const std::uint32_t aggregate =
                    args.S_in[s_index(args.layout, wx, wy, wz)];
                float other = other_field(aggregate, centre, state,
                                          args.global_flags);
                if constexpr (WallCoupling)
                    other += weighted_wall_field<true>(
                        args.wall, wz);
                const float rhs = gamma * lap
                    - bulk * centre * (1.0f - centre) * (1.0f - 2.0f * centre)
                    + volume * centre
                    - args.interaction_coeff * centre * other
                    - (velocity_x * gx + velocity_y * gy
                       + velocity_z * gz);
                const float next = centre + args.dt * rhs;
                if (!isfinite(next)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    return;
                }
                if (args.layout.substrate_slab() &&
                    wz == args.layout.nz - 1 && fabsf(next) > kSupportEps) {
                    raise_flag(state, args.global_flags,
                               FLAG3D_SLAB_TOP_CONTACT);
                    return;
                }

                const int dx = x - shift_x;
                const int dy = y - shift_y;
                const int dz = z - shift_z;
                if (static_cast<unsigned>(dx) >= static_cast<unsigned>(B) ||
                    static_cast<unsigned>(dy) >= static_cast<unsigned>(B) ||
                    static_cast<unsigned>(dz) >= static_cast<unsigned>(B)) {
                    if (fabsf(next) > kSupportEps)
                        raise_flag(state, args.global_flags,
                                   FLAG3D_DESTINATION_CLIP);
                    return;
                }
                destination[field_index_at_world_z(dx, dy, dz, wz, B, args.storage)] = next;

                if (args.S_out != nullptr) {
                    // Integer Q5.27 additions are exact and order-independent;
                    // sharding changes only which CTA issues each atomic.
                    scatter_value(args.S_out,
                                  s_index(args.layout, wx, wy, wz), next,
                                  state, args.global_flags);
                }

                if constexpr (CollectMoments) {
                    const double square = static_cast<double>(next)
                                        * static_cast<double>(next);
                    local.V += square;
                    local.Cx += square * static_cast<double>(dx);
                    local.Cy += square * static_cast<double>(dy);
                    local.Cz += square * static_cast<double>(dz);
                    local.phi_max_bits = max(
                        local.phi_max_bits, __float_as_uint(fabsf(next)));
                    if (fabsf(next) > kSupportEps) {
                        local.lo_x = min(local.lo_x, dx);
                        local.hi_x = max(local.hi_x, dx);
                        local.lo_y = min(local.lo_y, dy);
                        local.hi_y = max(local.hi_y, dy);
                        local.lo_z = min(local.lo_z, dz);
                        local.hi_z = max(local.hi_z, dz);
                    }
                }
            };
            if constexpr (CollectMoments) {
                for (int q = static_cast<int>(threadIdx.x); q < kBrickVoxels;
                     q += kThreads3D) {
                    const int ix = q % kBrickX;
                    const int iy = (q / kBrickX) % kBrickY;
                    const int iz = q / (kBrickX * kBrickY);
                    update_voxel(ix, iy, iz, {});
                }
            } else {
                // A warp retains its x-row while advancing through z. Each
                // stencil reuses two XY planes and loads only the next plane.
                const int ix = static_cast<int>(threadIdx.x) % kBrickX;
                for (int iy = static_cast<int>(threadIdx.x) / kBrickX;
                     iy < kBrickY; iy += kThreads3D / kBrickX) {
                    float p0[9], p1[9];
                    load_stencil_plane(shared_phi, ix, iy, 0, p0);
                    load_stencil_plane(shared_phi, ix, iy, 1, p1);
                    const auto advance = [&](float* below,
                                              const float* middle, int iz) {
                        const auto values = advance_stencil_plane(
                            shared_phi, ix, iy, iz, below, middle);
                        update_voxel(ix, iy, iz, values);
                    };
                    for (int iz = 0; iz < kBrickZ; iz += 2) {
                        advance(p0, p1, iz);
                        if (iz + 1 < kBrickZ) advance(p1, p0, iz + 1);
                    }
                }
            }
            __syncthreads();
            parity = 1 - parity;
        }
    }

    if constexpr (CollectMoments) {
        const Reduction3D total = reduce_cta(local, warp_values, B);
        if (threadIdx.x == 0) {
            partials[static_cast<std::size_t>(n) * shards_per_cell + shard] =
                total;
        }
    } else {
        __syncthreads();
    }
}

}  // namespace

__global__ void k_initialize_spheres(InitArgs3D args) {
    const int n = static_cast<int>(blockIdx.x);
    if (n >= args.N) return;
    const int B = args.B;
    const std::size_t words = stored_cell_words(B, args.storage);
    CellState3D* state = &args.cells[n];
    const Vec3 centre = args.centres[n];

    if (threadIdx.x == 0) {
        state->origin_x = __double2ll_rn(centre.x - 0.5 * static_cast<double>(B - 1));
        state->origin_y = __double2ll_rn(centre.y - 0.5 * static_cast<double>(B - 1));
        state->origin_z = __double2ll_rn(centre.z - 0.5 * static_cast<double>(B - 1));
        state->shift_ctr = state->tumble_ctr = 0u;
        state->flags = 0u;
        state->pad0 = state->pad1 = state->pad2 = 0u;
        state->pending_shift_x = state->pending_shift_y = state->pending_shift_z = 0;
        state->V = state->Cx = state->Cy = state->Cz = 0.0;
        state->surface = state->Ix = state->Iy = state->Iz = 0.0;
        state->wall_overlap = state->outside_slit_volume = 0.0;
        state->phi_max = 0.0f;
        for (unsigned& value : state->reserved) value = 0u;
    }
    __syncthreads();

    float* even = args.phi_even + static_cast<std::size_t>(n) * words;
    float* odd = args.phi_odd != nullptr
        ? args.phi_odd + static_cast<std::size_t>(n) * words : nullptr;
    const float k = interface_k(args.lambda);
    for (std::size_t q = threadIdx.x; q < words; q += blockDim.x) {
        const int x = static_cast<int>(q % static_cast<std::size_t>(B));
        const int y = static_cast<int>((q / static_cast<std::size_t>(B))
                                      % static_cast<std::size_t>(B));
        const int stored_z = static_cast<int>(q / (static_cast<std::size_t>(B) * B));
        int z = 0;
        if (!logical_z_from_stored(stored_z, B, state->origin_z, args.storage, &z)) {
            even[q] = 0.0f;
            if (odd != nullptr) odd[q] = 0.0f;
            continue;
        }
        int physical_z = 0;
        const bool in_domain = aggregate_z(
            args.layout, state->origin_z, z, &physical_z);
        const double dx = static_cast<double>(state->origin_x + x) - centre.x;
        const double dy = static_cast<double>(state->origin_y + y) - centre.y;
        const double dz = static_cast<double>(state->origin_z + z) - centre.z;
        const float radius = static_cast<float>(sqrt(dx * dx + dy * dy + dz * dz));
        even[q] = in_domain
            ? 0.5f * (1.0f - tanhf(k * (radius - args.seed_radius)))
            : 0.0f;
        if (odd != nullptr) odd[q] = 0.0f;
    }
}

__global__ void k_clear_u32(std::uint32_t* data, std::size_t words) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
         q < words; q += stride)
        data[q] = 0u;
}

__global__ void k_clear_selected_base_out(UpdateArgs3D args,
                                          int blocks_per_cell) {
    const int slot = static_cast<int>(blockIdx.x) / blocks_per_cell;
    const int part = static_cast<int>(blockIdx.x) % blocks_per_cell;
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;
    const std::size_t words = stored_cell_words(args.B, args.storage);
    float* out = args.phi_out + static_cast<std::size_t>(n) * words;
    const std::size_t stride =
        static_cast<std::size_t>(blocks_per_cell) * blockDim.x;
    for (std::size_t q = static_cast<std::size_t>(part) * blockDim.x
                       + threadIdx.x; q < words; q += stride)
        out[q] = 0.0f;
}

__global__ void k_scatter_current(ScatterArgs3D args) {
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;
    const int B = args.B;
    const std::size_t words = stored_cell_words(B, args.storage);
    CellState3D* state = &args.cells[n];
    const float* tile = args.phi + static_cast<std::size_t>(n) * words;
    const int ox = wrap_origin(state->origin_x, args.layout.nx);
    const int oy = wrap_origin(state->origin_y, args.layout.ny);

    // One warp per contiguous x-row: the (y, z) decomposition and wrapped row
    // base are computed once per row instead of per voxel, and the exact
    // Q5.27 additions are order-independent, so the scatter is unchanged.
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warps = static_cast<int>(blockDim.x) >> 5;
    const int rows = B * B;
    for (int row = warp; row < rows; row += warps) {
        const int z = row / B;
        const int y = row - z * B;
        const int wy = wrap_offset(oy, y, args.layout.ny);
        int wz = 0;
        if (!aggregate_z(args.layout, state->origin_z, z, &wz)) continue;
        const float* grow = tile + field_index_at_world_z(0, y, z, wz, B, args.storage);
        const std::size_t row_base = s_index(args.layout, 0, wy, wz);
        for (int x = lane; x < B; x += 32) {
            const float value = grow[x];
            if (!isfinite(value)) {
                raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                continue;
            }
            const int wx = wrap_offset(ox, x, args.layout.nx);
            scatter_value(args.S, row_base + static_cast<std::size_t>(wx),
                          value, state, args.global_flags);
        }
    }
}

__global__ void k_scatter_promoted(ScatterArgs3D args,
                                   float* const* promoted_phi,
                                   const int* promoted_ids,
                                   int promoted_count, int promoted_edge) {
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= args.N || promoted_phi[n] == nullptr) return;
    const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
    if (B == 0) return;
    CellState3D* state = &args.cells[n];
    const float* tile = promoted_phi[n];
    const int ox = wrap_origin(state->origin_x, args.layout.nx);
    const int oy = wrap_origin(state->origin_y, args.layout.ny);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warps = static_cast<int>(blockDim.x) >> 5;
    const int rows = B * B;
    for (int row = warp; row < rows; row += warps) {
        const int z = row / B;
        const int y = row - z * B;
        const int wy = wrap_offset(oy, y, args.layout.ny);
        int wz = 0;
        if (!aggregate_z(args.layout, state->origin_z, z, &wz)) continue;
        const float* grow = tile + field_index_at_world_z(0, y, z, wz, B, args.storage);
        const std::size_t row_base = s_index(args.layout, 0, wy, wz);
        for (int x = lane; x < B; x += 32) {
            const float value = grow[x];
            if (!isfinite(value)) {
                raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                continue;
            }
            const int wx = wrap_offset(ox, x, args.layout.nx);
            scatter_value(args.S, row_base + static_cast<std::size_t>(wx),
                          value, state, args.global_flags);
        }
    }
}

__global__ void k_check_promoted_crops(
    float* const* phi, const CellState3D* cells, const int* candidate_edges,
    std::uint32_t* rejected, int N, int base_edge, CellFieldStorage3D storage) {
    const int n = static_cast<int>(blockIdx.x);
    if (n >= N) return;
    const int target = candidate_edges[n];
    if (target == 0) return;
    constexpr int kCropGuard = 8;
    const int B = promoted_cell_edge(cells[n], base_edge, INT_MAX);
    const bool valid = B > 0 && target >= base_edge && target < B &&
        target > 2 * kCropGuard && target % kBrickAlignment == 0 &&
        phi[n] != nullptr &&
        static_cast<std::size_t>(B) <=
            SIZE_MAX / static_cast<std::size_t>(B) /
            static_cast<std::size_t>(B) / sizeof(float);
    if (!valid) {
        if (threadIdx.x == 0) atomicOr(rejected + n, 1u);
        return;
    }

    const int offset = (B - target) / 2;
    const int lo = offset + kCropGuard;
    const int hi = offset + target - kCropGuard;
    const std::size_t words = cell_words(B);
    bool reject = false;
    for (std::size_t q = threadIdx.x; q < words; q += blockDim.x) {
        const float value = logical_field_value(phi[n], q, B, cells[n].origin_z, storage);
        const int x = static_cast<int>(q % B);
        const int y = static_cast<int>((q / B) % B);
        const int z = static_cast<int>(q / (static_cast<std::size_t>(B) * B));
        const bool outside = x < lo || x >= hi || y < lo || y >= hi ||
                             z < lo || z >= hi;
        // Inspect the logical cube; omitted domain-exterior planes are zero.
        // Exact zeros also protect eight retained planes at each crop face.
        // A support threshold cannot justify discarding nonzero field values.
        if (!isfinite(value) || (outside && value != 0.0f)) {
            reject = true;
            break;
        }
    }
    const int any_rejected = __syncthreads_or(reject ? 1 : 0);
    if (threadIdx.x == 0 && any_rejected) atomicOr(rejected + n, 1u);
}

template <bool WallCoupling>
__device__ void measure_one_cell(const MeasureArgs3D& args, int n, int B,
                                 const float* tile, float* halo_pair,
                                 Reduction3D* warp_values) {
    CellState3D* state = &args.cells[n];
    const int ox = wrap_origin(state->origin_x, args.layout.nx);
    const int oy = wrap_origin(state->origin_y, args.layout.ny);

    // The origin is fixed throughout this measurement, including halo staging.
    const std::int64_t origin_z = state->origin_z;
    const bool bounded_z = args.layout.bounded_z();
    const auto z_map = bounded_z
        ? BoundedZMap::make(origin_z, B, args.layout.nz,
                           args.layout.hard_wall_channel())
        : BoundedZMap{};
    const auto fetch_z = [&](int z, int* source_z) {
        return bounded_z ? z_map.fetch(z, source_z)
            : phase_fetch_local_z(args.layout, origin_z, z, B, source_z);
    };
    const int storage_z_offset = args.storage.compact(B)
        ? static_cast<int>(origin_z) : 0;

    Reduction3D local{};
    local.lo_x = local.lo_y = local.lo_z = B;
    local.hi_x = local.hi_y = local.hi_z = -1;

    const int tiles_x = (B + kBrickX - 1) / kBrickX;
    const int tiles_y = (B + kBrickY - 1) / kBrickY;
    const int tiles_z = (B + kBrickZ - 1) / kBrickZ;
    int tz_begin = 0, tz_end = tiles_z;
    live_tile_z_range(args.layout, origin_z, B, tiles_z,
                      &tz_begin, &tz_end);
    const int tiles_per_plane = tiles_x * tiles_y;
    const int tile_begin = tz_begin * tiles_per_plane;
    const int tile_end = tz_end * tiles_per_plane;
    float* const halo0 = halo_pair + kHaloLeadingPadding;
    float* const halo1 = halo0 + kHaloBufferWords;
    int parity = 0;
    if (tile_begin < tile_end)
        prefetch_halo(tile, B, tile_begin, tiles_x, tiles_y, fetch_z,
                      storage_z_offset, halo0);
    // X-fastest tile order fixes each thread's voxel accumulation sequence.
    for (int t = tile_begin; t < tile_end; ++t) {
        const bool more = t + 1 < tile_end;
        if (more)
            prefetch_halo(tile, B, t + 1, tiles_x, tiles_y,
                          fetch_z, storage_z_offset,
                          parity == 0 ? halo1 : halo0);
        __pipeline_wait_prior(more ? 1 : 0);
        float* const shared_phi = parity == 0 ? halo0 : halo1;
        if (halo_any_nonzero(shared_phi)) {
            const int tx = t % tiles_x;
            const int ty = (t / tiles_x) % tiles_y;
            const int tz = t / (tiles_x * tiles_y);
            const int x0 = tx * kBrickX;
            const int y0 = ty * kBrickY;
            const int z0 = tz * kBrickZ;
            for (int q = static_cast<int>(threadIdx.x); q < kBrickVoxels;
                 q += kThreads3D) {
                const int ix = q % kBrickX;
                const int iy = (q / kBrickX) % kBrickY;
                const int iz = q / (kBrickX * kBrickY);
                const int x = x0 + ix, y = y0 + iy, z = z0 + iz;
                if (x >= B || y >= B || z >= B) continue;
                int wz = 0;
                const bool mapped = bounded_z ? z_map.aggregate(z, &wz)
                    : aggregate_z(args.layout, origin_z, z, &wz);
                if (!mapped)
                    continue;
                const float centre = shared_phi[halo_index(ix + 1, iy + 1,
                                                           iz + 1)];
                if (!isfinite(centre)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    continue;
                }
                float gx, gy, gz;
                gradient3(shared_phi, ix + 1, iy + 1, iz + 1,
                          &gx, &gy, &gz);
                const int wx = wrap_offset(ox, x, args.layout.nx);
                const int wy = wrap_offset(oy, y, args.layout.ny);
                float other = centre != 0.0f
                    ? other_field(
                        args.S[s_index(args.layout, wx, wy, wz)], centre,
                        state, args.global_flags)
                    : 0.0f;
                if constexpr (WallCoupling)
                    other += weighted_wall_field<true>(
                        args.wall, wz);
                const double square = static_cast<double>(centre)
                                    * static_cast<double>(centre);
                local.V += square;
                local.Cx += square * static_cast<double>(x);
                local.Cy += square * static_cast<double>(y);
                local.Cz += square * static_cast<double>(z);
                if (args.compute_surface)
                    local.surface += sqrt(static_cast<double>(gx) * gx
                                        + static_cast<double>(gy) * gy
                                        + static_cast<double>(gz) * gz);
                local.Ix += static_cast<double>(centre * gx * other);
                local.Iy += static_cast<double>(centre * gy * other);
                local.Iz += static_cast<double>(centre * gz * other);
                local.phi_max_bits = max(local.phi_max_bits,
                                         __float_as_uint(fabsf(centre)));
                if (fabsf(centre) > kSupportEps) {
                    if (args.layout.substrate_slab() &&
                        wz == args.layout.nz - 1)
                        raise_flag(state, args.global_flags,
                                   FLAG3D_SLAB_TOP_CONTACT);
                    local.lo_x = min(local.lo_x, x);
                    local.hi_x = max(local.hi_x, x);
                    local.lo_y = min(local.lo_y, y);
                    local.hi_y = max(local.hi_y, y);
                    local.lo_z = min(local.lo_z, z);
                    local.hi_z = max(local.hi_z, z);
                }
            }
        }
        __syncthreads();
        parity = 1 - parity;
    }

    const Reduction3D total = reduce_cta(local, warp_values, B);
    if (threadIdx.x == 0) commit_measurement(args, n, B, total);
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 3)
void k_measure_cells_impl(MeasureArgs3D args) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;
    const std::size_t words = stored_cell_words(args.B, args.storage);
    const float* tile = args.phi + static_cast<std::size_t>(n) * words;
    measure_one_cell<WallCoupling>(args, n, args.B, tile, halo_pair,
                                   warp_values);
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 1)
void k_measure_promoted_impl(MeasureArgs3D args,
                             float* const* promoted_phi,
                             const int* promoted_ids,
                             int promoted_count, int promoted_edge) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= args.N || promoted_phi[n] == nullptr) return;
    const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
    if (B == 0) return;
    measure_one_cell<WallCoupling>(args, n, B, promoted_phi[n],
                                   halo_pair, warp_values);
}

// Sharded promoted measurement: each pointer-backed cube is measured by
// args.shards CTAs over the same deterministic brick assignment as the base
// sharded measurement, with per-shard partials folded in ascending shard
// order by the finalize kernel.  The reduction grouping differs from the
// one-CTA promoted measurement, so runs using this path are deterministic
// and restart-exact but not bitwise comparable with the one-CTA fold.
template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 3)
void k_measure_promoted_shards_impl(MeasureArgs3D args,
                                    float* const* promoted_phi,
                                    const int* promoted_ids,
                                    int promoted_count, int promoted_edge,
                                    MomentPartial3D* promoted_partials) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    const int block = static_cast<int>(blockIdx.x);
    const int slot = block / args.shards;
    const int shard = block - slot * args.shards;
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= args.N || promoted_phi[n] == nullptr) return;
    const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
    if (B == 0) return;
    CellState3D* state = &args.cells[n];
    const float* tile = promoted_phi[n];
    const int ox = wrap_origin(state->origin_x, args.layout.nx);
    const int oy = wrap_origin(state->origin_y, args.layout.ny);

    const std::int64_t origin_z = state->origin_z;
    const bool bounded_z = args.layout.bounded_z();
    const auto z_map = bounded_z
        ? BoundedZMap::make(origin_z, B, args.layout.nz,
                           args.layout.hard_wall_channel())
        : BoundedZMap{};
    const auto fetch_z = [&](int z, int* source_z) {
        return bounded_z ? z_map.fetch(z, source_z)
            : phase_fetch_local_z(args.layout, origin_z, z, B, source_z);
    };
    const int storage_z_offset = args.storage.compact(B)
        ? static_cast<int>(origin_z) : 0;

    Reduction3D local = empty_reduction(B);
    const int bricks_x = (B + kBrickX - 1) / kBrickX;
    const int bricks_y = (B + kBrickY - 1) / kBrickY;
    const int bricks_z = (B + kBrickZ - 1) / kBrickZ;
    int tz_begin = 0, tz_end = bricks_z;
    live_tile_z_range(args.layout, origin_z, B, bricks_z,
                      &tz_begin, &tz_end);
    const int bricks_per_plane = bricks_x * bricks_y;
    const int brick_begin = tz_begin * bricks_per_plane;
    const int brick_end = tz_end * bricks_per_plane;
    const int begin_remainder = brick_begin % args.shards;
    const int first_brick = brick_begin
        + (shard - begin_remainder + args.shards) % args.shards;
    float* const halo0 = halo_pair + kHaloLeadingPadding;
    float* const halo1 = halo0 + kHaloBufferWords;
    int parity = 0;
    if (first_brick < brick_end)
        prefetch_halo(tile, B, first_brick, bricks_x, bricks_y, fetch_z,
                      storage_z_offset, halo0);
    for (int brick = first_brick; brick < brick_end; brick += args.shards) {
        const bool more = brick + args.shards < brick_end;
        if (more)
            prefetch_halo(tile, B, brick + args.shards, bricks_x, bricks_y,
                          fetch_z, storage_z_offset,
                          parity == 0 ? halo1 : halo0);
        __pipeline_wait_prior(more ? 1 : 0);
        float* const shared_phi = parity == 0 ? halo0 : halo1;
        if (halo_any_nonzero(shared_phi)) {
            const int bx = brick % bricks_x;
            const int by = (brick / bricks_x) % bricks_y;
            const int bz = brick / (bricks_x * bricks_y);
            const int x0 = bx * kBrickX;
            const int y0 = by * kBrickY;
            const int z0 = bz * kBrickZ;
            for (int q = static_cast<int>(threadIdx.x); q < kBrickVoxels;
                 q += kThreads3D) {
                const int ix = q % kBrickX;
                const int iy = (q / kBrickX) % kBrickY;
                const int iz = q / (kBrickX * kBrickY);
                const int x = x0 + ix, y = y0 + iy, z = z0 + iz;
                if (x >= B || y >= B || z >= B) continue;
                int wz = 0;
                const bool mapped = bounded_z ? z_map.aggregate(z, &wz)
                    : aggregate_z(args.layout, origin_z, z, &wz);
                if (!mapped)
                    continue;
                const float centre = shared_phi[halo_index(ix + 1, iy + 1,
                                                           iz + 1)];
                if (!isfinite(centre)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    continue;
                }
                float gx, gy, gz;
                gradient3(shared_phi, ix + 1, iy + 1, iz + 1,
                          &gx, &gy, &gz);
                const int wx = wrap_offset(ox, x, args.layout.nx);
                const int wy = wrap_offset(oy, y, args.layout.ny);
                float other = centre != 0.0f
                    ? other_field(args.S[s_index(args.layout, wx, wy, wz)],
                                  centre, state, args.global_flags)
                    : 0.0f;
                if constexpr (WallCoupling)
                    other += weighted_wall_field<true>(
                        args.wall, wz);
                const double square = static_cast<double>(centre)
                                    * static_cast<double>(centre);
                local.V += square;
                local.Cx += square * static_cast<double>(x);
                local.Cy += square * static_cast<double>(y);
                local.Cz += square * static_cast<double>(z);
                if (args.compute_surface)
                    local.surface += sqrt(static_cast<double>(gx) * gx
                                        + static_cast<double>(gy) * gy
                                        + static_cast<double>(gz) * gz);
                local.Ix += static_cast<double>(centre * gx * other);
                local.Iy += static_cast<double>(centre * gy * other);
                local.Iz += static_cast<double>(centre * gz * other);
                local.phi_max_bits = max(local.phi_max_bits,
                                         __float_as_uint(fabsf(centre)));
                if (fabsf(centre) > kSupportEps) {
                    if (args.layout.substrate_slab() &&
                        wz == args.layout.nz - 1)
                        raise_flag(state, args.global_flags,
                                   FLAG3D_SLAB_TOP_CONTACT);
                    local.lo_x = min(local.lo_x, x);
                    local.hi_x = max(local.hi_x, x);
                    local.lo_y = min(local.lo_y, y);
                    local.hi_y = max(local.hi_y, y);
                    local.lo_z = min(local.lo_z, z);
                    local.hi_z = max(local.hi_z, z);
                }
            }
        }
        __syncthreads();
        parity = 1 - parity;
    }

    const Reduction3D total = reduce_cta(local, warp_values, B);
    if (threadIdx.x == 0) {
        promoted_partials[static_cast<std::size_t>(slot) * args.shards
                          + shard] = total;
    }
}

__global__ void k_finalize_measure_promoted(
    MeasureArgs3D args, const int* promoted_ids, int promoted_count,
    int promoted_edge, const MomentPartial3D* promoted_partials) {
    const int slot = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= args.N) return;
    const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
    if (B == 0) return;
    Reduction3D total = empty_reduction(B);
    const MomentPartial3D* parts =
        promoted_partials + static_cast<std::size_t>(slot) * args.shards;
    for (int shard = 0; shard < args.shards; ++shard)
        append_reduction(&total, parts[shard]);
    commit_measurement(args, n, B, total);
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 3)
void k_measure_cell_shards_impl(MeasureArgs3D args) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    const int block = static_cast<int>(blockIdx.x);
    const int slot = block / args.shards;
    const int shard = block - slot * args.shards;
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;

    const int B = args.B;
    const std::size_t words = stored_cell_words(B, args.storage);
    CellState3D* state = &args.cells[n];
    const float* tile = args.phi + static_cast<std::size_t>(n) * words;
    const int ox = wrap_origin(state->origin_x, args.layout.nx);
    const int oy = wrap_origin(state->origin_y, args.layout.ny);
    // Recentering commits after measurement; every tile uses this same origin.
    const std::int64_t origin_z = state->origin_z;
    const bool bounded_z = args.layout.bounded_z();
    const auto z_map = bounded_z
        ? BoundedZMap::make(origin_z, B, args.layout.nz,
                           args.layout.hard_wall_channel())
        : BoundedZMap{};
    const auto fetch_z = [&](int z, int* source_z) {
        return bounded_z ? z_map.fetch(z, source_z)
            : phase_fetch_local_z(args.layout, origin_z, z, B, source_z);
    };
    const int storage_z_offset = args.storage.compact(B)
        ? static_cast<int>(origin_z) : 0;

    Reduction3D local = empty_reduction(B);
    const int bricks_x = (B + kBrickX - 1) / kBrickX;
    const int bricks_y = (B + kBrickY - 1) / kBrickY;
    const int bricks_z = (B + kBrickZ - 1) / kBrickZ;
    int tz_begin = 0, tz_end = bricks_z;
    live_tile_z_range(args.layout, origin_z, B, bricks_z,
                      &tz_begin, &tz_end);
    const int bricks_per_plane = bricks_x * bricks_y;
    const int brick_begin = tz_begin * bricks_per_plane;
    const int brick_end = tz_end * bricks_per_plane;
    const int begin_remainder = brick_begin % args.shards;
    const int first_brick = brick_begin
        + (shard - begin_remainder + args.shards) % args.shards;
    float* const halo0 = halo_pair + kHaloLeadingPadding;
    float* const halo1 = halo0 + kHaloBufferWords;
    int parity = 0;
    if (first_brick < brick_end)
        prefetch_halo(tile, B, first_brick, bricks_x, bricks_y, fetch_z,
                      storage_z_offset, halo0);
    for (int brick = first_brick; brick < brick_end; brick += args.shards) {
        const bool more = brick + args.shards < brick_end;
        if (more)
            prefetch_halo(tile, B, brick + args.shards, bricks_x, bricks_y,
                          fetch_z, storage_z_offset,
                          parity == 0 ? halo1 : halo0);
        __pipeline_wait_prior(more ? 1 : 0);
        float* const shared_phi = parity == 0 ? halo0 : halo1;
        if (halo_any_nonzero(shared_phi)) {
            const int bx = brick % bricks_x;
            const int by = (brick / bricks_x) % bricks_y;
            const int bz = brick / (bricks_x * bricks_y);
            const int x0 = bx * kBrickX;
            const int y0 = by * kBrickY;
            const int z0 = bz * kBrickZ;
            for (int q = static_cast<int>(threadIdx.x); q < kBrickVoxels;
                 q += kThreads3D) {
                const int ix = q % kBrickX;
                const int iy = (q / kBrickX) % kBrickY;
                const int iz = q / (kBrickX * kBrickY);
                const int x = x0 + ix, y = y0 + iy, z = z0 + iz;
                if (x >= B || y >= B || z >= B) continue;
                int wz = 0;
                const bool mapped = bounded_z ? z_map.aggregate(z, &wz)
                    : aggregate_z(args.layout, origin_z, z, &wz);
                if (!mapped)
                    continue;
                const float centre = shared_phi[halo_index(ix + 1, iy + 1,
                                                           iz + 1)];
                if (!isfinite(centre)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    continue;
                }
                float gx, gy, gz;
                gradient3(shared_phi, ix + 1, iy + 1, iz + 1,
                          &gx, &gy, &gz);
                const int wx = wrap_offset(ox, x, args.layout.nx);
                const int wy = wrap_offset(oy, y, args.layout.ny);
                float other = centre != 0.0f
                    ? other_field(args.S[s_index(args.layout, wx, wy, wz)],
                                  centre, state, args.global_flags)
                    : 0.0f;
                if constexpr (WallCoupling)
                    other += weighted_wall_field<true>(
                        args.wall, wz);
                const double square = static_cast<double>(centre)
                                    * static_cast<double>(centre);
                local.V += square;
                local.Cx += square * static_cast<double>(x);
                local.Cy += square * static_cast<double>(y);
                local.Cz += square * static_cast<double>(z);
                if (args.compute_surface)
                    local.surface += sqrt(static_cast<double>(gx) * gx
                                        + static_cast<double>(gy) * gy
                                        + static_cast<double>(gz) * gz);
                local.Ix += static_cast<double>(centre * gx * other);
                local.Iy += static_cast<double>(centre * gy * other);
                local.Iz += static_cast<double>(centre * gz * other);
                local.phi_max_bits = max(local.phi_max_bits,
                                         __float_as_uint(fabsf(centre)));
                if (fabsf(centre) > kSupportEps) {
                    if (args.layout.substrate_slab() &&
                        wz == args.layout.nz - 1)
                        raise_flag(state, args.global_flags,
                                   FLAG3D_SLAB_TOP_CONTACT);
                    local.lo_x = min(local.lo_x, x);
                    local.hi_x = max(local.hi_x, x);
                    local.lo_y = min(local.lo_y, y);
                    local.hi_y = max(local.hi_y, y);
                    local.lo_z = min(local.lo_z, z);
                    local.hi_z = max(local.hi_z, z);
                }
            }
        }
        __syncthreads();
        parity = 1 - parity;
    }

    const Reduction3D total = reduce_cta(local, warp_values, B);
    if (threadIdx.x == 0) {
        args.partials[static_cast<std::size_t>(n) * args.shards + shard] = total;
    }
}

__global__ void k_finalize_measure_cell_shards(MeasureArgs3D args) {
    const int slot = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;
    Reduction3D total = empty_reduction(args.B);
    const MomentPartial3D* parts =
        args.partials + static_cast<std::size_t>(n) * args.shards;
    for (int shard = 0; shard < args.shards; ++shard)
        append_reduction(&total, parts[shard]);
    commit_measurement(args, n, args.B, total);
}

__global__ void k_apply_cell_motion(MeasureArgs3D args) {
    const int slot = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    CellState3D* state = &args.cells[n];
    if (fatal_flags_present(state, args.global_flags)) return;

    const unsigned long long global_step = args.step != nullptr ? *args.step : 0ull;
    const bool active = global_step >= args.aging_steps;
    Vec3 polarity{state->polarity_x, state->polarity_y, state->polarity_z};
    bool tumbled = false;
    if (args.apply_tumble && active) {
        const TumbleSample sample = args.layout.substrate_slab()
            ? tumble_sample_planar(
                  global_step - args.aging_steps, state->global_id,
                  args.polarity_stream, args.p_tumble, polarity)
            : tumble_sample(
                  global_step - args.aging_steps, state->global_id,
                  args.polarity_stream, args.p_tumble, polarity);
        polarity = sample.polarity;
        tumbled = sample.tumbled;
    }
    const float px = static_cast<float>(polarity.x);
    const float py = static_cast<float>(polarity.y);
    const float pz = args.layout.substrate_slab()
        ? 0.0f : static_cast<float>(polarity.z);
    const float active_speed = active ? state->v_A : 0.0f;
    const float vx = active_speed * px
                   + args.motility_coeff * static_cast<float>(state->Ix);
    const float vy = active_speed * py
                   + args.motility_coeff * static_cast<float>(state->Iy);
    // Substrate-slab translation is projected onto x/y; the phase field still
    // relaxes and deforms in all three spatial dimensions.
    const float vz = args.layout.substrate_slab()
        ? 0.0f
        : active_speed * pz
            + args.motility_coeff * static_cast<float>(state->Iz);
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz) ||
        !isfinite(vx) || !isfinite(vy) || !isfinite(vz)) {
        raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
        return;
    }
    state->polarity_x = px;
    state->polarity_y = py;
    state->polarity_z = pz;
    state->velocity_x = vx;
    state->velocity_y = vy;
    state->velocity_z = vz;
    if (tumbled) ++state->tumble_ctr;
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 1)
void k_update_tiled_impl(UpdateArgs3D args) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    __shared__ int cell_number;

    for (;;) {
        if (threadIdx.x == 0) {
            const unsigned long long ticket = atomicAdd(args.work_cursor, 1ull);
            cell_number = ticket < static_cast<unsigned long long>(
                args.selection.selected_count(args.N))
                ? args.selection.cell_index(static_cast<int>(ticket)) : -1;
        }
        __syncthreads();
        const int n = cell_number;
        if (n < 0) break;
        if (cell_is_promoted(args.cells[n], args.B)) continue;
        const std::size_t words = stored_cell_words(args.B, args.storage);
        const float* source =
            args.phi_in + static_cast<std::size_t>(n) * words;
        float* destination = args.phi_out + static_cast<std::size_t>(n) * words;
        process_update_cell<true, WallCoupling>(
            n, args, source, destination, args.B, halo_pair, warp_values,
            nullptr);
    }
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 4)
void k_update_tiled_fast_impl(UpdateArgs3D args) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ int cell_number;
    __shared__ int abort_flag;

    for (;;) {
        if (threadIdx.x == 0) {
            const unsigned long long ticket = atomicAdd(args.work_cursor, 1ull);
            cell_number = ticket < static_cast<unsigned long long>(
                args.selection.selected_count(args.N))
                ? args.selection.cell_index(static_cast<int>(ticket)) : -1;
        }
        __syncthreads();
        const int n = cell_number;
        if (n < 0) break;
        if (cell_is_promoted(args.cells[n], args.B)) continue;
        const std::size_t words = stored_cell_words(args.B, args.storage);
        const float* source =
            args.phi_in + static_cast<std::size_t>(n) * words;
        float* destination = args.phi_out + static_cast<std::size_t>(n) * words;
        process_update_cell<false, WallCoupling>(
            n, args, source, destination, args.B, halo_pair, nullptr,
            &abort_flag);
    }
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 1)
void k_update_tiled_sharded_impl(UpdateArgs3D args,
                                 MomentPartial3D* partials,
                                 int shards_per_cell) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    __shared__ int abort_flag;
    const int block = static_cast<int>(blockIdx.x);
    const int slot = block / shards_per_cell;
    const int shard = block - slot * shards_per_cell;
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;
    const std::size_t words = stored_cell_words(args.B, args.storage);
    process_update_shard<true, WallCoupling>(
        n, shard, shards_per_cell, args,
        args.phi_in + static_cast<std::size_t>(n) * words,
        args.phi_out + static_cast<std::size_t>(n) * words,
        args.B, partials, halo_pair, warp_values, &abort_flag);
}

template <bool WallCoupling, UpdateTilePass3D Pass = UpdateTilePass3D::All>
__global__ __launch_bounds__(kThreads3D, 3)
void k_update_tiled_sharded_fast_impl(UpdateArgs3D args,
                                      int shards_per_cell) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ int abort_flag;
    const int block = static_cast<int>(blockIdx.x);
    const int slot = block / shards_per_cell;
    const int shard = block - slot * shards_per_cell;
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;
    const std::size_t words = stored_cell_words(args.B, args.storage);
    process_update_shard<false, WallCoupling, Pass>(
        n, shard, shards_per_cell, args,
        args.phi_in + static_cast<std::size_t>(n) * words,
        args.phi_out + static_cast<std::size_t>(n) * words,
        args.B, nullptr, halo_pair, nullptr, &abort_flag);
}

// The promoted throughput update splits every pointer-backed cube among
// shards_per_cell CTAs with the same deterministic tile assignment as the
// base sharded update: tile t belongs to t % shards_per_cell.  The phase
// arithmetic is process_update_shard's, so outputs are bit-identical to the
// one-CTA promoted update; the destination cubes must be cleared first
// because shards skip absent voxels instead of zero-filling.
__global__ void k_clear_promoted_out(UpdateArgs3D args,
                                     float* const* promoted_phi_out,
                                     const int* promoted_ids,
                                     int promoted_count, int promoted_edge,
                                     int blocks_per_cube) {
    const int block = static_cast<int>(blockIdx.x);
    const int slot = block / blocks_per_cube;
    const int part = block - slot * blocks_per_cube;
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= args.N) return;
    float* out = promoted_phi_out[n];
    if (out == nullptr) return;
    const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
    if (B == 0) return;
    const std::size_t words = stored_cell_words(B, args.storage);
    const std::size_t stride =
        static_cast<std::size_t>(blocks_per_cube) * blockDim.x;
    for (std::size_t q = static_cast<std::size_t>(part) * blockDim.x
                       + threadIdx.x;
         q < words; q += stride)
        out[q] = 0.0f;
}

template <bool WallCoupling, UpdateTilePass3D Pass = UpdateTilePass3D::All>
__global__ __launch_bounds__(kThreads3D, 3)
void k_update_promoted_sharded_fast_impl(
    UpdateArgs3D args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, int shards_per_cell) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ int abort_flag;
    const int block = static_cast<int>(blockIdx.x);
    const int slot = block / shards_per_cell;
    const int shard = block - slot * shards_per_cell;
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= args.N || promoted_phi_in[n] == nullptr ||
        promoted_phi_out[n] == nullptr ||
        promoted_phi_in[n] == promoted_phi_out[n])
        return;
    const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
    if (B == 0) return;
    process_update_shard<false, WallCoupling, Pass>(
        n, shard, shards_per_cell, args, promoted_phi_in[n],
        promoted_phi_out[n], B, nullptr, halo_pair, nullptr,
        &abort_flag);
}

__global__ void k_finalize_tiled_sharded(
    UpdateArgs3D args, const MomentPartial3D* partials, int shards_per_cell) {
    const int slot = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= args.selection.selected_count(args.N)) return;
    const int n = args.selection.cell_index(slot);
    if (cell_is_promoted(args.cells[n], args.B)) return;

    Reduction3D total{};
    total.lo_x = total.lo_y = total.lo_z = args.B;
    total.hi_x = total.hi_y = total.hi_z = -1;
    const std::size_t base = static_cast<std::size_t>(n) * shards_per_cell;
    // This serial, ascending-shard fold makes the next-step moments invariant
    // to CTA scheduling.  Each shard's intra-CTA reduction is deterministic as
    // well; only the grouping differs from the one-CTA implementation.
    for (int shard = 0; shard < shards_per_cell; ++shard) {
        const Reduction3D part = partials[base + shard];
        total.V += part.V;
        total.Cx += part.Cx;
        total.Cy += part.Cy;
        total.Cz += part.Cz;
        total.surface += part.surface;
        total.Ix += part.Ix;
        total.Iy += part.Iy;
        total.Iz += part.Iz;
        total.lo_x = min(total.lo_x, part.lo_x);
        total.hi_x = max(total.hi_x, part.hi_x);
        total.lo_y = min(total.lo_y, part.lo_y);
        total.hi_y = max(total.hi_y, part.hi_y);
        total.lo_z = min(total.lo_z, part.lo_z);
        total.hi_z = max(total.hi_z, part.hi_z);
        total.phi_max_bits = max(total.phi_max_bits, part.phi_max_bits);
    }

    CellState3D* state = &args.cells[n];
    if (!fatal_flags_present(state, args.global_flags)) {
        state->V = total.V;
        state->Cx = total.Cx;
        state->Cy = total.Cy;
        state->Cz = total.Cz;
        state->bb_lo_x = total.lo_x;
        state->bb_hi_x = total.hi_x;
        state->bb_lo_y = total.lo_y;
        state->bb_hi_y = total.hi_y;
        state->bb_lo_z = total.lo_z;
        state->bb_hi_z = total.hi_z;
        state->phi_max = __uint_as_float(total.phi_max_bits);
    } else {
        state->pending_shift_x = 0;
        state->pending_shift_y = 0;
        state->pending_shift_z = 0;
    }
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 1)
void k_update_inplace_impl(UpdateArgs3D args, float* scratch,
                           int scratch_slots) {
    if (static_cast<int>(blockIdx.x) >= scratch_slots) return;
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    __shared__ int cell_number;
    const std::size_t words = stored_cell_words(args.B, args.storage);
    float* block_scratch = scratch + static_cast<std::size_t>(blockIdx.x) * words;

    for (;;) {
        if (threadIdx.x == 0) {
            const unsigned long long ticket = atomicAdd(args.work_cursor, 1ull);
            cell_number = ticket < static_cast<unsigned long long>(
                args.selection.selected_count(args.N))
                ? args.selection.cell_index(static_cast<int>(ticket)) : -1;
        }
        __syncthreads();
        const int n = cell_number;
        if (n < 0) break;
        if (cell_is_promoted(args.cells[n], args.B)) continue;
        const float* source =
            args.phi_in + static_cast<std::size_t>(n) * words;
        process_update_cell<true, WallCoupling>(
            n, args, source, block_scratch, args.B, halo_pair, warp_values,
            nullptr);

        // phi_in is still intact here.  Do not commit this cell if its update
        // raised a fatal alarm.  A fatal raised concurrently by another CTA
        // is also observed whenever it is visible before this check.
        if (threadIdx.x == 0) {
            const bool failed =
                fatal_flags_present(&args.cells[n], args.global_flags);
            if (failed) {
                args.cells[n].pending_shift_x = 0;
                args.cells[n].pending_shift_y = 0;
                args.cells[n].pending_shift_z = 0;
            }
            cell_number = failed ? -1 : n;
        }
        __syncthreads();
        if (cell_number >= 0) {
            float* destination =
                args.phi_in + static_cast<std::size_t>(n) * words;
            for (std::size_t q = threadIdx.x; q < words; q += kThreads3D)
                destination[q] = block_scratch[q];
        }
        __syncthreads();
    }
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 4)
void k_update_inplace_fast_impl(UpdateArgs3D args, float* scratch,
                                int scratch_slots) {
    if (static_cast<int>(blockIdx.x) >= scratch_slots) return;
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ int cell_number;
    __shared__ int abort_flag;
    const std::size_t words = stored_cell_words(args.B, args.storage);
    float* block_scratch = scratch + static_cast<std::size_t>(blockIdx.x) * words;

    for (;;) {
        if (threadIdx.x == 0) {
            const unsigned long long ticket = atomicAdd(args.work_cursor, 1ull);
            cell_number = ticket < static_cast<unsigned long long>(
                args.selection.selected_count(args.N))
                ? args.selection.cell_index(static_cast<int>(ticket)) : -1;
        }
        __syncthreads();
        const int n = cell_number;
        if (n < 0) break;
        if (cell_is_promoted(args.cells[n], args.B)) continue;
        const float* source =
            args.phi_in + static_cast<std::size_t>(n) * words;
        process_update_cell<false, WallCoupling>(
            n, args, source, block_scratch, args.B, halo_pair, nullptr,
            &abort_flag);

        if (threadIdx.x == 0) {
            const bool failed =
                fatal_flags_present(&args.cells[n], args.global_flags);
            if (failed) {
                args.cells[n].pending_shift_x = 0;
                args.cells[n].pending_shift_y = 0;
                args.cells[n].pending_shift_z = 0;
            }
            cell_number = failed ? -1 : n;
        }
        __syncthreads();
        if (cell_number >= 0) {
            float* destination =
                args.phi_in + static_cast<std::size_t>(n) * words;
            for (std::size_t q = threadIdx.x; q < words; q += kThreads3D)
                destination[q] = block_scratch[q];
        }
        __syncthreads();
    }
}

template <bool WallCoupling>
__global__ __launch_bounds__(kThreads3D, 1)
void k_update_promoted_impl(UpdateArgs3D args,
                            float* const* promoted_phi_in,
                            float* const* promoted_phi_out,
                            const int* promoted_ids,
                            int promoted_count, int promoted_edge,
                            bool collect_moments) {
    extern __shared__ __align__(16) float halo_pair[];
    __shared__ Reduction3D warp_values[kWarps3D];
    __shared__ int abort_flag;
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= args.N || promoted_phi_in[n] == nullptr ||
        promoted_phi_out[n] == nullptr ||
        promoted_phi_in[n] == promoted_phi_out[n])
        return;
    const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
    if (B == 0) return;
    if (collect_moments) {
        process_update_cell<true, WallCoupling>(
            n, args, promoted_phi_in[n], promoted_phi_out[n], B,
            halo_pair, warp_values, nullptr);
    } else {
        process_update_cell<false, WallCoupling>(
            n, args, promoted_phi_in[n], promoted_phi_out[n], B,
            halo_pair, nullptr, &abort_flag);
    }
}

__global__ void k_repair_after_fatal(UpdateArgs3D args) {
    __shared__ int repair;
    if (threadIdx.x == 0)
        repair = fatal_global_flags_present(args.global_flags) ? 1 : 0;
    __syncthreads();
    if (repair == 0) return;

    if (args.S_out != nullptr) {
        for (std::size_t q = threadIdx.x; q < args.layout.words();
             q += blockDim.x)
            args.S_out[q] = 0u;
        __syncthreads();
    }

    const std::size_t words = stored_cell_words(args.B, args.storage);
    for (int slot = 0; slot < args.selection.selected_count(args.N); ++slot) {
        const int n = args.selection.cell_index(slot);
        CellState3D* state = &args.cells[n];
        if (cell_is_promoted(*state, args.B)) continue;
        float* source = args.phi_in + static_cast<std::size_t>(n) * words;
        if (args.phi_out != nullptr) {
            float* output = args.phi_out + static_cast<std::size_t>(n) * words;
            for (std::size_t q = threadIdx.x; q < words; q += blockDim.x)
                output[q] = source[q];
            if (threadIdx.x == 0) {
                state->pending_shift_x = 0;
                state->pending_shift_y = 0;
                state->pending_shift_z = 0;
            }
            __syncthreads();
        }

        if (args.S_out != nullptr) {
            // In-place cells that committed before another CTA failed retain
            // their pending shift until finalize; include it in world mapping.
            std::int64_t effective_x = 0, effective_y = 0, effective_z = 0;
            const int sx = args.phi_out == nullptr ? state->pending_shift_x : 0;
            const int sy = args.phi_out == nullptr ? state->pending_shift_y : 0;
            const int sz = args.phi_out == nullptr ? state->pending_shift_z : 0;
            const bool effective_origin_valid =
                checked_world_coordinate(state->origin_x, sx, &effective_x) &&
                checked_world_coordinate(state->origin_y, sy, &effective_y) &&
                checked_world_coordinate(state->origin_z, sz, &effective_z);
            if (!effective_origin_valid) {
                if (threadIdx.x == 0)
                    raise_flag(state, args.global_flags,
                               FLAG3D_INVALID_GEOMETRY);
                __syncthreads();
                continue;
            }
            const int ox = wrap_origin(effective_x, args.layout.nx);
            const int oy = wrap_origin(effective_y, args.layout.ny);
            for (std::size_t q = threadIdx.x; q < words; q += blockDim.x) {
                const float value = source[q];
                if (!isfinite(value)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    continue;
                }
                const int x = static_cast<int>(q % static_cast<std::size_t>(args.B));
                const int y = static_cast<int>((q / static_cast<std::size_t>(args.B))
                                              % static_cast<std::size_t>(args.B));
                const int stored_z = static_cast<int>(
                    q / (static_cast<std::size_t>(args.B) * args.B));
                int z = 0;
                if (!logical_z_from_stored(stored_z, args.B, effective_z,
                                           args.storage, &z)) continue;
                int wz = 0;
                if (!aggregate_z(args.layout, effective_z, z, &wz)) continue;
                scatter_value(
                    args.S_out,
                    s_index(args.layout,
                            wrap_offset(ox, x, args.layout.nx),
                            wrap_offset(oy, y, args.layout.ny),
                            wz),
                    value, state, args.global_flags);
            }
            __syncthreads();
        }
    }
}

__global__ void k_repair_promoted_after_fatal(
    UpdateArgs3D args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge) {
    __shared__ int repair;
    if (threadIdx.x == 0)
        repair = fatal_global_flags_present(args.global_flags) ? 1 : 0;
    __syncthreads();
    if (repair == 0) return;

    for (int slot = 0; slot < promoted_count; ++slot) {
        const int n = promoted_ids[slot];
        if (n < 0 || n >= args.N || promoted_phi_in[n] == nullptr ||
            promoted_phi_out[n] == nullptr)
            continue;
        const int B = promoted_cell_edge(args.cells[n], args.B, promoted_edge);
        if (B == 0) continue;
        const std::size_t words = stored_cell_words(B, args.storage);
        CellState3D* state = &args.cells[n];
        const float* source = promoted_phi_in[n];
        float* output = promoted_phi_out[n];
        for (std::size_t q = threadIdx.x; q < words; q += blockDim.x)
            output[q] = source[q];
        if (threadIdx.x == 0) {
            state->pending_shift_x = 0;
            state->pending_shift_y = 0;
            state->pending_shift_z = 0;
        }
        __syncthreads();

        if (args.S_out != nullptr) {
            const int ox = wrap_origin(state->origin_x, args.layout.nx);
            const int oy = wrap_origin(state->origin_y, args.layout.ny);
            for (std::size_t q = threadIdx.x; q < words; q += blockDim.x) {
                const float value = source[q];
                if (!isfinite(value)) {
                    raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                    continue;
                }
                const int x = static_cast<int>(q % B);
                const int y = static_cast<int>((q / B) % B);
                const int stored_z = static_cast<int>(
                    q / (static_cast<std::size_t>(B) * B));
                int z = 0;
                if (!logical_z_from_stored(stored_z, B, state->origin_z,
                                           args.storage, &z)) continue;
                int wz = 0;
                if (!aggregate_z(args.layout, state->origin_z, z, &wz))
                    continue;
                scatter_value(
                    args.S_out,
                    s_index(args.layout,
                            wrap_offset(ox, x, args.layout.nx),
                            wrap_offset(oy, y, args.layout.ny),
                            wz),
                    value, state, args.global_flags);
            }
            __syncthreads();
        }
    }
}

__global__ void k_finalize_origins(CellState3D* cells, int N,
                                   std::uint32_t* global_flags) {
    const int n = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;
    CellState3D* state = &cells[n];
    // A cell that committed before another CTA raised a global fatal still
    // owns a consistently recentered in-place field.  Only its own fatal bit
    // suppresses the matching origin change; repair cleared uncommitted shifts.
    if ((state->flags & kFatalFlagMask3D) != 0u) {
        state->pending_shift_x = 0;
        state->pending_shift_y = 0;
        state->pending_shift_z = 0;
        return;
    }
    const int sx = state->pending_shift_x;
    const int sy = state->pending_shift_y;
    const int sz = state->pending_shift_z;
    std::int64_t next_x = 0, next_y = 0, next_z = 0;
    if (!checked_world_coordinate(state->origin_x, sx, &next_x) ||
        !checked_world_coordinate(state->origin_y, sy, &next_y) ||
        !checked_world_coordinate(state->origin_z, sz, &next_z)) {
        raise_flag(state, global_flags, FLAG3D_INVALID_GEOMETRY);
        state->pending_shift_x = state->pending_shift_y =
            state->pending_shift_z = 0;
        return;
    }
    state->origin_x = next_x;
    state->origin_y = next_y;
    state->origin_z = next_z;
    if (sx != 0 || sy != 0 || sz != 0) ++state->shift_ctr;
    state->pending_shift_x = state->pending_shift_y = state->pending_shift_z = 0;
}

__global__ void k_restore_cells_after_fatal(
    CellState3D* cells, const CellState3D* accepted_cells, int N,
    const std::uint32_t* global_flags, std::uint32_t* support_requests) {
    if (!fatal_global_flags_present(global_flags)) return;
    const int n = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const std::uint32_t failed_flags = cells[n].flags;
    if ((failed_flags & flag3d_bit(FLAG3D_SUPPORT_EXHAUSTED)) != 0u &&
        support_requests != nullptr)
        support_requests[n] = 1u;
    cells[n] = accepted_cells[n];
}

__global__ void k_advance_step(std::uint64_t* step,
                               const std::uint32_t* global_flags) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (!fatal_global_flags_present(global_flags)) ++(*step);
}

__global__ void k_pack_trajectory(const CellState3D* cells,
                                  TrajPackedCell3D* out,
                                  int N, SLayout3D layout,
                                  int channel_padding) {
    const int n = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const CellState3D state = cells[n];
    const double inv_volume = state.V > 0.0 ? 1.0 / state.V : 0.0;
    TrajPackedCell3D record{};
    record.global_id = state.global_id;
    record.x = static_cast<double>(state.origin_x) + state.Cx * inv_volume;
    record.y = static_cast<double>(state.origin_y) + state.Cy * inv_volume;
    record.z = static_cast<double>(state.origin_z) + state.Cz * inv_volume;
    // Bounded-z trajectories report height above the lower face. The solver's
    // cell-centred planes are at z=0,1,... and that face is z=-1/2.
    if (layout.substrate_slab()) {
        record.z += 0.5;
    } else if (layout.hard_wall_channel()) {
        record.z += 0.5 - static_cast<double>(channel_padding);
    }
    record.velocity_x = state.velocity_x;
    record.velocity_y = state.velocity_y;
    record.velocity_z = state.velocity_z;
    record.polarity_x = state.polarity_x;
    record.polarity_y = state.polarity_y;
    record.polarity_z = state.polarity_z;
    record.volume = state.V;
    record.surface = state.surface;
    record.wall_overlap = state.wall_overlap;
    record.outside_slit_volume = state.outside_slit_volume;
    record.gamma = state.gamma;
    record.v_A = state.v_A;
    record.phi_max = state.phi_max;
    out[n] = record;
}

__global__ __launch_bounds__(kThreads3D, 1)
void k_measure_wall_diagnostics(const float* phi,
                            float* const* promoted_phi,
                            CellState3D* cells, int N, int B,
                            SLayout3D layout, const float* wall_psi_sq,
                            int channel_height, int channel_padding,
                            CellFieldStorage3D storage) {
    const int n = static_cast<int>(blockIdx.x);
    if (n >= N) return;
    __shared__ double overlap_by_warp[kWarps3D];
    __shared__ double outside_by_warp[kWarps3D];
    CellState3D* state = &cells[n];
    const int edge = cell_support_edge(*state, B);
    const float* tile = edge == B
        ? phi + static_cast<std::size_t>(n) * stored_cell_words(B, storage)
        : promoted_phi != nullptr ? promoted_phi[n] : nullptr;
    double overlap = 0.0;
    double outside = 0.0;
    if (tile != nullptr && wall_psi_sq != nullptr) {
        const std::size_t words = cell_words(edge);
        for (std::size_t q = threadIdx.x; q < words; q += kThreads3D) {
            const int z = static_cast<int>(
                q / (static_cast<std::size_t>(edge) * edge));
            int world_z = 0;
            if (!aggregate_z(layout, state->origin_z, z, &world_z)) continue;
            const double value = static_cast<double>(
                logical_field_value(tile, q, edge, state->origin_z, storage));
            const double phi_sq = value * value;
            overlap += phi_sq * static_cast<double>(wall_psi_sq[world_z]);
            if (world_z < channel_padding ||
                world_z >= channel_padding + channel_height)
                outside += phi_sq;
        }
    }
    overlap = warp_sum(overlap);
    outside = warp_sum(outside);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        overlap_by_warp[warp] = overlap;
        outside_by_warp[warp] = outside;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double total_overlap = 0.0, total_outside = 0.0;
        for (int w = 0; w < kWarps3D; ++w) {
            total_overlap += overlap_by_warp[w];
            total_outside += outside_by_warp[w];
        }
        state->wall_overlap = total_overlap;
        state->outside_slit_volume = total_outside;
    }
}

__device__ void verify_one_cell(const float* tile, const CellState3D* cells,
                                 VerifyCell3D* out, int n, int B,
                                 SLayout3D layout,
                                 const CellFieldStorage3D& storage,
                                 double* volume_by_warp,
                                unsigned int* max_by_warp,
                                unsigned int* bad_by_warp,
                                unsigned int* edge_by_warp) {
    const std::size_t words = cell_words(B);
    double volume = 0.0;
    unsigned int max_bits = 0u, nonfinite = 0u, edge = 0u;
    for (std::size_t q = threadIdx.x; q < words; q += kThreads3D) {
        const float value = logical_field_value(tile, q, B, cells[n].origin_z, storage);
        if (!isfinite(value)) {
            ++nonfinite;
            continue;
        }
        const int z = static_cast<int>(
            q / (static_cast<std::size_t>(B) * B));
        int physical_z = 0;
        const bool in_domain = aggregate_z(
            layout, cells[n].origin_z, z, &physical_z);
        if (!in_domain) {
            edge += fabsf(value) > kSupportEps;
            continue;
        }
        volume += static_cast<double>(value) * static_cast<double>(value);
        max_bits = max(max_bits, __float_as_uint(fabsf(value)));
        if (fabsf(value) > kSupportEps) {
            const int x = static_cast<int>(q % static_cast<std::size_t>(B));
            const int y = static_cast<int>((q / static_cast<std::size_t>(B))
                                          % static_cast<std::size_t>(B));
            edge += (x == 0 || x == B - 1 || y == 0 || y == B - 1 ||
                     z == 0 || z == B - 1 ||
                     (layout.substrate_slab() &&
                      physical_z == layout.nz - 1));
        }
    }
    volume = warp_sum(volume);
    max_bits = warp_max_u32(max_bits);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        nonfinite += __shfl_down_sync(0xffffffffu, nonfinite, delta);
        edge += __shfl_down_sync(0xffffffffu, edge, delta);
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        volume_by_warp[warp] = volume;
        max_by_warp[warp] = max_bits;
        bad_by_warp[warp] = nonfinite;
        edge_by_warp[warp] = edge;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        VerifyCell3D result{};
        for (int w = 0; w < kWarps3D; ++w) {
            result.measured_V += volume_by_warp[w];
            result.max_abs_phi = fmaxf(result.max_abs_phi,
                                       __uint_as_float(max_by_warp[w]));
            result.nonfinite_count += bad_by_warp[w];
            result.support_edge_count += edge_by_warp[w];
        }
        result.state_V = cells[n].V;
        result.verified = 1u;
        out[n] = result;
    }
}

__global__ __launch_bounds__(kThreads3D, 1)
void k_verify_cells(const float* phi, const CellState3D* cells,
                    VerifyCell3D* out, int N, int B, SLayout3D layout,
                    CellFieldStorage3D storage) {
    __shared__ double volume_by_warp[kWarps3D];
    __shared__ unsigned int max_by_warp[kWarps3D];
    __shared__ unsigned int bad_by_warp[kWarps3D];
    __shared__ unsigned int edge_by_warp[kWarps3D];
    const int n = static_cast<int>(blockIdx.x);
    if (n >= N || cell_is_promoted(cells[n], B)) return;
    const std::size_t words = stored_cell_words(B, storage);
    verify_one_cell(phi + static_cast<std::size_t>(n) * words, cells, out,
                    n, B, layout, storage, volume_by_warp, max_by_warp, bad_by_warp,
                    edge_by_warp);
}

__global__ __launch_bounds__(kThreads3D, 1)
void k_verify_promoted(float* const* promoted_phi,
                       const CellState3D* cells, VerifyCell3D* out,
                       const int* promoted_ids, int N, int promoted_count,
                       int promoted_edge, SLayout3D layout,
                       CellFieldStorage3D storage) {
    __shared__ double volume_by_warp[kWarps3D];
    __shared__ unsigned int max_by_warp[kWarps3D];
    __shared__ unsigned int bad_by_warp[kWarps3D];
    __shared__ unsigned int edge_by_warp[kWarps3D];
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= promoted_count) return;
    const int n = promoted_ids[slot];
    if (n < 0 || n >= N || promoted_phi[n] == nullptr) return;
    const int B = promoted_cell_edge(cells[n], 0, promoted_edge);
    if (B == 0) return;
    verify_one_cell(promoted_phi[n], cells, out, n, B, layout, storage,
                    volume_by_warp, max_by_warp, bad_by_warp, edge_by_warp);
}

__global__ void k_verify_S(const std::uint32_t* S, std::size_t words,
                           std::uint32_t* out_max) {
    std::uint32_t maximum = 0u;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
         q < words; q += stride)
        maximum = max(maximum, S[q]);
    atomicMax(out_max, maximum);
}

namespace {

// The brick-walking kernels need their dynamic-shared-memory limit raised
// above the 48 KiB default before the first launch or occupancy query.
thread_local bool tile_kernels_configured = false;

cudaError_t configure_current_tile_kernels() {
    const void* kernels[] = {
        reinterpret_cast<const void*>(k_measure_cells_impl<false>),
        reinterpret_cast<const void*>(k_measure_cells_impl<true>),
        reinterpret_cast<const void*>(k_measure_promoted_impl<false>),
        reinterpret_cast<const void*>(k_measure_promoted_impl<true>),
        reinterpret_cast<const void*>(k_measure_cell_shards_impl<false>),
        reinterpret_cast<const void*>(k_measure_cell_shards_impl<true>),
        reinterpret_cast<const void*>(k_update_tiled_impl<false>),
        reinterpret_cast<const void*>(k_update_tiled_impl<true>),
        reinterpret_cast<const void*>(k_update_tiled_fast_impl<false>),
        reinterpret_cast<const void*>(k_update_tiled_fast_impl<true>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_impl<false>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_impl<true>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_fast_impl<false>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_fast_impl<true>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_fast_impl<
            false, UpdateTilePass3D::Boundary>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_fast_impl<
            false, UpdateTilePass3D::Interior>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_fast_impl<
            true, UpdateTilePass3D::Boundary>),
        reinterpret_cast<const void*>(k_update_tiled_sharded_fast_impl<
            true, UpdateTilePass3D::Interior>),
        reinterpret_cast<const void*>(k_update_inplace_impl<false>),
        reinterpret_cast<const void*>(k_update_inplace_impl<true>),
        reinterpret_cast<const void*>(k_update_inplace_fast_impl<false>),
        reinterpret_cast<const void*>(k_update_inplace_fast_impl<true>),
        reinterpret_cast<const void*>(k_update_promoted_impl<false>),
        reinterpret_cast<const void*>(k_update_promoted_impl<true>),
        reinterpret_cast<const void*>(
            k_update_promoted_sharded_fast_impl<false>),
        reinterpret_cast<const void*>(
            k_update_promoted_sharded_fast_impl<true>),
        reinterpret_cast<const void*>(k_update_promoted_sharded_fast_impl<
            false, UpdateTilePass3D::Boundary>),
        reinterpret_cast<const void*>(k_update_promoted_sharded_fast_impl<
            false, UpdateTilePass3D::Interior>),
        reinterpret_cast<const void*>(k_update_promoted_sharded_fast_impl<
            true, UpdateTilePass3D::Boundary>),
        reinterpret_cast<const void*>(k_update_promoted_sharded_fast_impl<
            true, UpdateTilePass3D::Interior>),
        reinterpret_cast<const void*>(k_measure_promoted_shards_impl<false>),
        reinterpret_cast<const void*>(k_measure_promoted_shards_impl<true>),
    };
    for (const void* kernel : kernels) {
        const cudaError_t error = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kHaloPipelineBytes));
        if (error != cudaSuccess) return error;
    }
    tile_kernels_configured = true;
    return cudaSuccess;
}

cudaError_t set_tile_kernel_smem_limit() {
    return tile_kernels_configured ? cudaSuccess
                                  : configure_current_tile_kernels();
}

bool valid_update_partition(const UpdateArgs3D& args) {
    if (args.tile_pass == UpdateTilePass3D::All) return true;
    return valid_update_tile_pass(args.tile_pass)
        && args.layout.bounded_z()
        && args.phi_out != nullptr && args.S_out != nullptr
        && valid_tile_exchange_region(args.exchange_region,
                                      args.layout.ny, args.layout.nz);
}

bool valid_runtime_field(int B, const SLayout3D& layout,
                         const CellFieldStorage3D& storage) {
    return valid_runtime_geometry(B, layout) && valid_field_storage(storage, B, layout);
}

bool checked_stored_field_bytes(int N, int B, const CellFieldStorage3D& storage,
                                std::size_t* bytes) {
    std::size_t one = 0;
    return N > 0 && storage.checked_bytes(B, &one) &&
           checked_mul_size(static_cast<std::size_t>(N), one, bytes);
}

cudaError_t clear_base_output(const UpdateArgs3D& args,
                              std::size_t phi_bytes, cudaStream_t stream) {
    if (args.selection.ids == nullptr)
        return cudaMemsetAsync(args.phi_out, 0, phi_bytes, stream);
    constexpr int kClearBlocksPerCell = 64;
    if (args.selection.count >
        std::numeric_limits<int>::max() / kClearBlocksPerCell)
        return cudaErrorInvalidValue;
    k_clear_selected_base_out<<<args.selection.count * kClearBlocksPerCell,
                                kThreads3D, 0, stream>>>(
        args, kClearBlocksPerCell);
    return cudaPeekAtLastError();
}

// The exchange partition changes scheduling, not the wall term or stencil.
// Both passes use the same geometry specialization as the unsplit update.
template <bool WallCoupling>
void enqueue_base_fast_pass(const UpdateArgs3D& args, int count,
                            int shards_per_cell, cudaStream_t stream) {
    const int blocks = count * shards_per_cell;
    if (args.tile_pass == UpdateTilePass3D::Boundary)
        k_update_tiled_sharded_fast_impl<WallCoupling,
                                         UpdateTilePass3D::Boundary>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, shards_per_cell);
    else if (args.tile_pass == UpdateTilePass3D::Interior)
        k_update_tiled_sharded_fast_impl<WallCoupling,
                                         UpdateTilePass3D::Interior>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, shards_per_cell);
    else
        k_update_tiled_sharded_fast_impl<WallCoupling>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, shards_per_cell);
}

template <bool WallCoupling>
void enqueue_promoted_fast_pass(
    const UpdateArgs3D& args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, int shards_per_cell,
    cudaStream_t stream) {
    const int blocks = promoted_count * shards_per_cell;
    if (args.tile_pass == UpdateTilePass3D::Boundary)
        k_update_promoted_sharded_fast_impl<WallCoupling,
                                            UpdateTilePass3D::Boundary>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, promoted_phi_in, promoted_phi_out, promoted_ids,
                promoted_count, promoted_edge, shards_per_cell);
    else if (args.tile_pass == UpdateTilePass3D::Interior)
        k_update_promoted_sharded_fast_impl<WallCoupling,
                                            UpdateTilePass3D::Interior>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, promoted_phi_in, promoted_phi_out, promoted_ids,
                promoted_count, promoted_edge, shards_per_cell);
    else
        k_update_promoted_sharded_fast_impl<WallCoupling>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, promoted_phi_in, promoted_phi_out, promoted_ids,
                promoted_count, promoted_edge, shards_per_cell);
}

}  // namespace

cudaError_t configure_tile_kernel_shared_memory() {
    // Called during allocation with its device current. Explicit configuration
    // must not be skipped because this host thread previously used another GPU.
    return configure_current_tile_kernels();
}

bool valid_runtime_geometry(int B, const SLayout3D& layout) {
    return B > 0 && B % kBrickAlignment == 0
        && layout.nx > B && layout.ny > B
        && (layout.hard_wall_channel() ? layout.nz > 0 : layout.nz > B)
        && layout.pitch_x >= layout.nx && layout.pitch_x % 4 == 0
        && recognized_boundary(layout);
}

void launch_initialize_spheres(const InitArgs3D& args, cudaStream_t stream) {
    if (args.N <= 0 || args.phi_even == nullptr ||
        args.cells == nullptr || args.centres == nullptr || args.B <= 0 ||
        args.B % kBrickAlignment != 0 || !(args.lambda > 0.0f) ||
        !(args.seed_radius > 0.0f) ||
        !valid_runtime_field(args.B, args.layout, args.storage))
        return;
    k_initialize_spheres<<<args.N, kThreads3D, 0, stream>>>(args);
}

void launch_clear_S(std::uint32_t* S, const SLayout3D& layout,
                    cudaStream_t stream) {
    if (S == nullptr || layout.words() == 0) return;
    const int blocks = static_cast<int>(std::min<std::size_t>(
        4096, (layout.words() + kThreads3D - 1) / kThreads3D));
    k_clear_u32<<<std::max(1, blocks), kThreads3D, 0, stream>>>(S, layout.words());
}

void launch_scatter_current(const ScatterArgs3D& args, cudaStream_t stream) {
    const int count = args.selection.selected_count(args.N);
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        !args.selection.valid(args.N) || count == 0 ||
        args.phi == nullptr || args.cells == nullptr || args.S == nullptr)
        return;
    k_scatter_current<<<count, kThreads3D, 0, stream>>>(args);
}

void launch_scatter_promoted(const ScatterArgs3D& args,
                             float* const* promoted_phi,
                             const int* promoted_ids, int promoted_count,
                             int promoted_edge, cudaStream_t stream) {
    if (!valid_runtime_field(promoted_edge, args.layout, args.storage) || args.N <= 0 ||
        promoted_phi == nullptr || promoted_ids == nullptr ||
        promoted_count <= 0 || args.cells == nullptr || args.S == nullptr)
        return;
    k_scatter_promoted<<<promoted_count, kThreads3D, 0, stream>>>(
        args, promoted_phi, promoted_ids, promoted_count, promoted_edge);
}

bool launch_check_promoted_crops(
    float* const* phi, const CellState3D* cells, const int* candidate_edges,
    std::uint32_t* rejected, int N, int base_edge, cudaStream_t stream,
    CellFieldStorage3D storage) {
    std::size_t bytes = 0;
    if (phi == nullptr || cells == nullptr || candidate_edges == nullptr ||
        rejected == nullptr || N <= 0 || base_edge <= 0 ||
        base_edge % kBrickAlignment != 0 || !storage.checked_bytes(base_edge, &bytes))
        return false;
    k_check_promoted_crops<<<N, kThreads3D, 0, stream>>>(
        phi, cells, candidate_edges, rejected, N, base_edge, storage);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_measure_cells_only(const MeasureArgs3D& args,
                               cudaStream_t stream) {
    if (!args.selection.valid(args.N)) return false;
    const int count = args.selection.selected_count(args.N);
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        args.phi == nullptr || args.S == nullptr || args.cells == nullptr ||
        !valid_weighted_wall_field(args.wall, args.layout) ||
        ((args.apply_tumble || args.aging_steps != 0u) && args.step == nullptr))
        return false;
    if (count == 0) return true;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    const bool sharded = args.shards > 1 && args.shards <= 64 &&
                         args.partials != nullptr;
    if (sharded) {
        if (wall_coupling)
            k_measure_cell_shards_impl<true>
                <<<count * args.shards, kThreads3D,
                   kHaloPipelineBytes, stream>>>(args);
        else
            k_measure_cell_shards_impl<false>
                <<<count * args.shards, kThreads3D,
                   kHaloPipelineBytes, stream>>>(args);
        k_finalize_measure_cell_shards<<<(count + 127) / 128,
                                         128, 0, stream>>>(args);
    } else {
        if (wall_coupling)
            k_measure_cells_impl<true>
                <<<count, kThreads3D, kHaloPipelineBytes, stream>>>(args);
        else
            k_measure_cells_impl<false>
                <<<count, kThreads3D, kHaloPipelineBytes, stream>>>(args);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_measure_cells(const MeasureArgs3D& args, cudaStream_t stream) {
    if (!launch_measure_cells_only(args, stream)) return false;
    launch_apply_cell_motion(args, stream);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_measure_promoted(const MeasureArgs3D& args,
                             float* const* promoted_phi,
                             const int* promoted_ids, int promoted_count,
                             int promoted_edge, cudaStream_t stream) {
    if (!valid_runtime_field(promoted_edge, args.layout, args.storage) || args.N <= 0 ||
        promoted_phi == nullptr || promoted_ids == nullptr ||
        promoted_count <= 0 || args.S == nullptr || args.cells == nullptr ||
        !valid_weighted_wall_field(args.wall, args.layout) ||
        ((args.apply_tumble || args.aging_steps != 0u) && args.step == nullptr))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_measure_promoted_impl<true>
            <<<promoted_count, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, promoted_phi, promoted_ids, promoted_count,
                promoted_edge);
    else
        k_measure_promoted_impl<false>
            <<<promoted_count, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, promoted_phi, promoted_ids, promoted_count,
                promoted_edge);
    return cudaPeekAtLastError() == cudaSuccess;
}

void launch_apply_cell_motion(const MeasureArgs3D& args,
                              cudaStream_t stream) {
    const int count = args.selection.selected_count(args.N);
    if (args.N <= 0 || args.cells == nullptr || args.global_flags == nullptr ||
        !args.selection.valid(args.N) || count == 0 ||
        ((args.apply_tumble || args.aging_steps != 0u) && args.step == nullptr))
        return;
    k_apply_cell_motion<<<(count + 127) / 128, 128, 0, stream>>>(args);
}

bool launch_update_tiled(const UpdateArgs3D& args, int grid_blocks,
                         cudaStream_t stream) {
    if (args.tile_pass != UpdateTilePass3D::All) return false;
    if (args.N <= 0 || !args.selection.valid(args.N)) return false;
    const int count = args.selection.selected_count(args.N);
    if (count == 0) return true;
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        args.phi_in == nullptr || args.phi_out == nullptr ||
        args.phi_in == args.phi_out || args.S_in == nullptr ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.cells == nullptr || args.work_cursor == nullptr || grid_blocks <= 0 ||
        !valid_weighted_wall_field(args.wall, args.layout))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    if (cudaMemsetAsync(args.work_cursor, 0, sizeof(unsigned long long),
                        stream) != cudaSuccess)
        return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_update_tiled_impl<true>
            <<<grid_blocks, kThreads3D, kHaloPipelineBytes, stream>>>(args);
    else
        k_update_tiled_impl<false>
            <<<grid_blocks, kThreads3D, kHaloPipelineBytes, stream>>>(args);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_update_tiled_sharded(const UpdateArgs3D& args,
                                 MomentPartial3D* partials,
                                 int shards_per_cell,
                                 cudaStream_t stream) {
    if (args.tile_pass != UpdateTilePass3D::All) return false;
    if (args.N <= 0 || !args.selection.valid(args.N)) return false;
    const int count = args.selection.selected_count(args.N);
    if (count == 0) return true;
    std::size_t phi_bytes = 0;
    const bool grid_fits = shards_per_cell > 1 &&
        args.N <= std::numeric_limits<int>::max() / shards_per_cell;
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        args.phi_in == nullptr || args.phi_out == nullptr ||
        args.phi_in == args.phi_out || args.S_in == nullptr ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.cells == nullptr || args.compute_surface || partials == nullptr ||
        !grid_fits || !valid_weighted_wall_field(args.wall, args.layout) ||
        !checked_stored_field_bytes(args.N, args.B, args.storage, &phi_bytes))
        return false;

    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    // Recentered writes are disjoint. Clear only the selected destination
    // cubes when another device owns the remaining cell slots.
    if (clear_base_output(args, phi_bytes, stream) != cudaSuccess)
        return false;
    const int blocks = count * shards_per_cell;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_update_tiled_sharded_impl<true>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, partials, shards_per_cell);
    else
        k_update_tiled_sharded_impl<false>
            <<<blocks, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, partials, shards_per_cell);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    k_finalize_tiled_sharded<<<(count + 127) / 128, 128, 0, stream>>>(
        args, partials, shards_per_cell);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_update_inplace(const UpdateArgs3D& args, float* scratch,
                           int scratch_slots, cudaStream_t stream) {
    if (args.tile_pass != UpdateTilePass3D::All) return false;
    if (args.N <= 0 || !args.selection.valid(args.N)) return false;
    const int count = args.selection.selected_count(args.N);
    if (count == 0) return true;
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        args.phi_in == nullptr || scratch == nullptr || scratch_slots <= 0 ||
        scratch == args.phi_in ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.S_in == nullptr || args.cells == nullptr ||
        args.work_cursor == nullptr ||
        !valid_weighted_wall_field(args.wall, args.layout))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    if (cudaMemsetAsync(args.work_cursor, 0, sizeof(unsigned long long),
                        stream) != cudaSuccess)
        return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_update_inplace_impl<true>
            <<<scratch_slots, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, scratch, scratch_slots);
    else
        k_update_inplace_impl<false>
            <<<scratch_slots, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, scratch, scratch_slots);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_update_promoted(
    const UpdateArgs3D& args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, bool collect_moments,
    cudaStream_t stream) {
    if (args.tile_pass != UpdateTilePass3D::All) return false;
    if (!valid_runtime_field(promoted_edge, args.layout, args.storage) || args.N <= 0 ||
        promoted_phi_in == nullptr || promoted_phi_out == nullptr ||
        promoted_phi_in == promoted_phi_out || promoted_ids == nullptr ||
        promoted_count <= 0 || args.S_in == nullptr || args.cells == nullptr ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        !valid_weighted_wall_field(args.wall, args.layout))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_update_promoted_impl<true>
            <<<promoted_count, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, promoted_phi_in, promoted_phi_out, promoted_ids,
                promoted_count, promoted_edge, collect_moments);
    else
        k_update_promoted_impl<false>
            <<<promoted_count, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, promoted_phi_in, promoted_phi_out, promoted_ids,
                promoted_count, promoted_edge, collect_moments);
    return cudaPeekAtLastError() == cudaSuccess;
}

namespace detail {

bool launch_update_promoted_sharded_fast(
    const UpdateArgs3D& args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, int shards_per_cell,
    cudaStream_t stream) {
    const bool partial = args.tile_pass != UpdateTilePass3D::All;
    const bool grid_fits = shards_per_cell >= (partial ? 1 : 2) &&
        promoted_count <= std::numeric_limits<int>::max() / shards_per_cell;
    if (!valid_update_partition(args) ||
        !valid_runtime_field(promoted_edge, args.layout, args.storage) || args.N <= 0 ||
        promoted_phi_in == nullptr || promoted_phi_out == nullptr ||
        promoted_phi_in == promoted_phi_out || promoted_ids == nullptr ||
        promoted_count <= 0 || args.S_in == nullptr || args.cells == nullptr ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.compute_surface || !grid_fits ||
        !valid_weighted_wall_field(args.wall, args.layout))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    // Interior completes the boundary pass's destination; it must not clear it.
    if (args.tile_pass != UpdateTilePass3D::Interior) {
        constexpr int kClearBlocksPerCube = 64;
        if (promoted_count >
            std::numeric_limits<int>::max() / kClearBlocksPerCube) return false;
        k_clear_promoted_out<<<promoted_count * kClearBlocksPerCube, kThreads3D,
                               0, stream>>>(args, promoted_phi_out, promoted_ids,
                                            promoted_count, promoted_edge,
                                            kClearBlocksPerCube);
        if (cudaPeekAtLastError() != cudaSuccess) return false;
    }
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        enqueue_promoted_fast_pass<true>(
            args, promoted_phi_in, promoted_phi_out, promoted_ids,
            promoted_count, promoted_edge, shards_per_cell, stream);
    else
        enqueue_promoted_fast_pass<false>(
            args, promoted_phi_in, promoted_phi_out, promoted_ids,
            promoted_count, promoted_edge, shards_per_cell, stream);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_measure_promoted_shards(
    const MeasureArgs3D& args, float* const* promoted_phi,
    const int* promoted_ids, int promoted_count, int promoted_edge,
    MomentPartial3D* promoted_partials, cudaStream_t stream) {
    const bool grid_fits = args.shards > 1 &&
        promoted_count <= std::numeric_limits<int>::max() / args.shards;
    if (!valid_runtime_field(promoted_edge, args.layout, args.storage) || args.N <= 0 ||
        promoted_phi == nullptr || promoted_ids == nullptr ||
        promoted_count <= 0 || args.S == nullptr || args.cells == nullptr ||
        promoted_partials == nullptr || !grid_fits ||
        !valid_weighted_wall_field(args.wall, args.layout))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_measure_promoted_shards_impl<true>
            <<<promoted_count * args.shards, kThreads3D,
               kHaloPipelineBytes, stream>>>(
                args, promoted_phi, promoted_ids, promoted_count,
                promoted_edge, promoted_partials);
    else
        k_measure_promoted_shards_impl<false>
            <<<promoted_count * args.shards, kThreads3D,
               kHaloPipelineBytes, stream>>>(
                args, promoted_phi, promoted_ids, promoted_count,
                promoted_edge, promoted_partials);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    k_finalize_measure_promoted<<<(promoted_count + 127) / 128, 128, 0,
                                  stream>>>(
        args, promoted_ids, promoted_count, promoted_edge,
        promoted_partials);
    return cudaPeekAtLastError() == cudaSuccess;
}

cudaError_t promoted_measure_occupancy(int* blocks_per_sm,
                                       const SLayout3D& layout) {
    if (blocks_per_sm == nullptr || !recognized_boundary(layout))
        return cudaErrorInvalidValue;
    const cudaError_t limited = set_tile_kernel_smem_limit();
    if (limited != cudaSuccess) return limited;
    const bool wall_coupling = layout.hard_wall_channel();
    return wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              blocks_per_sm, k_measure_promoted_shards_impl<true>,
              kThreads3D, kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              blocks_per_sm, k_measure_promoted_shards_impl<false>,
              kThreads3D, kHaloPipelineBytes);
}

bool launch_update_tiled_fast(const UpdateArgs3D& args, int grid_blocks,
                              cudaStream_t stream) {
    if (args.tile_pass != UpdateTilePass3D::All) return false;
    if (args.N <= 0 || !args.selection.valid(args.N)) return false;
    const int count = args.selection.selected_count(args.N);
    if (count == 0) return true;
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        args.phi_in == nullptr || args.phi_out == nullptr ||
        args.phi_in == args.phi_out || args.S_in == nullptr ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.cells == nullptr || args.work_cursor == nullptr ||
        args.compute_surface || grid_blocks <= 0 ||
        !valid_weighted_wall_field(args.wall, args.layout))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    if (cudaMemsetAsync(args.work_cursor, 0, sizeof(unsigned long long),
                        stream) != cudaSuccess)
        return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_update_tiled_fast_impl<true>
            <<<grid_blocks, kThreads3D, kHaloPipelineBytes, stream>>>(args);
    else
        k_update_tiled_fast_impl<false>
            <<<grid_blocks, kThreads3D, kHaloPipelineBytes, stream>>>(args);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_update_tiled_sharded_fast(const UpdateArgs3D& args,
                                      int shards_per_cell,
                                      cudaStream_t stream) {
    const bool partial = args.tile_pass != UpdateTilePass3D::All;
    if (!valid_update_partition(args) || (partial && shards_per_cell < 1)
        || args.N <= 0 || !args.selection.valid(args.N)) return false;
    const int count = args.selection.selected_count(args.N);
    if (count == 0) return true;
    std::size_t phi_bytes = 0;
    const bool grid_fits = shards_per_cell >= (partial ? 1 : 2) &&
        args.N <= std::numeric_limits<int>::max() / shards_per_cell;
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        args.phi_in == nullptr || args.phi_out == nullptr ||
        args.phi_in == args.phi_out || args.S_in == nullptr ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.cells == nullptr || args.compute_surface || !grid_fits ||
        !valid_weighted_wall_field(args.wall, args.layout) ||
        !checked_stored_field_bytes(args.N, args.B, args.storage, &phi_bytes))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    if (args.tile_pass != UpdateTilePass3D::Interior
        && clear_base_output(args, phi_bytes, stream) != cudaSuccess)
        return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        enqueue_base_fast_pass<true>(args, count, shards_per_cell, stream);
    else
        enqueue_base_fast_pass<false>(args, count, shards_per_cell, stream);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_update_inplace_fast(const UpdateArgs3D& args, float* scratch,
                                int scratch_slots, cudaStream_t stream) {
    if (args.tile_pass != UpdateTilePass3D::All) return false;
    if (args.N <= 0 || !args.selection.valid(args.N)) return false;
    const int count = args.selection.selected_count(args.N);
    if (count == 0) return true;
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        args.phi_in == nullptr || scratch == nullptr || scratch_slots <= 0 ||
        scratch == args.phi_in ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.S_in == nullptr || args.cells == nullptr ||
        args.work_cursor == nullptr || args.compute_surface ||
        !valid_weighted_wall_field(args.wall, args.layout))
        return false;
    if (set_tile_kernel_smem_limit() != cudaSuccess) return false;
    if (cudaMemsetAsync(args.work_cursor, 0, sizeof(unsigned long long),
                        stream) != cudaSuccess)
        return false;
    const bool wall_coupling = args.layout.hard_wall_channel();
    if (wall_coupling)
        k_update_inplace_fast_impl<true>
            <<<scratch_slots, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, scratch, scratch_slots);
    else
        k_update_inplace_fast_impl<false>
            <<<scratch_slots, kThreads3D, kHaloPipelineBytes, stream>>>(
                args, scratch, scratch_slots);
    return cudaPeekAtLastError() == cudaSuccess;
}

cudaError_t promoted_fast_update_occupancy(int* blocks_per_sm,
                                           const SLayout3D& layout) {
    if (blocks_per_sm == nullptr || !recognized_boundary(layout))
        return cudaErrorInvalidValue;
    const cudaError_t limited = set_tile_kernel_smem_limit();
    if (limited != cudaSuccess) return limited;
    const bool wall_coupling = layout.hard_wall_channel();
    return wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              blocks_per_sm, k_update_promoted_sharded_fast_impl<true>,
              kThreads3D, kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              blocks_per_sm, k_update_promoted_sharded_fast_impl<false>,
              kThreads3D, kHaloPipelineBytes);
}

cudaError_t measured_update_occupancy(int* tiled_blocks_per_sm,
                                      int* inplace_blocks_per_sm,
                                      const SLayout3D& layout) {
    if (tiled_blocks_per_sm == nullptr || inplace_blocks_per_sm == nullptr ||
        !recognized_boundary(layout))
        return cudaErrorInvalidValue;
    cudaError_t status = set_tile_kernel_smem_limit();
    if (status != cudaSuccess) return status;
    const bool wall_coupling = layout.hard_wall_channel();
    status = wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              tiled_blocks_per_sm, k_update_tiled_impl<true>, kThreads3D,
              kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              tiled_blocks_per_sm, k_update_tiled_impl<false>, kThreads3D,
              kHaloPipelineBytes);
    if (status != cudaSuccess) return status;
    return wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              inplace_blocks_per_sm, k_update_inplace_impl<true>, kThreads3D,
              kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              inplace_blocks_per_sm, k_update_inplace_impl<false>, kThreads3D,
              kHaloPipelineBytes);
}

cudaError_t fast_update_occupancy(int* tiled_blocks_per_sm,
                                  int* inplace_blocks_per_sm,
                                  const SLayout3D& layout) {
    if (tiled_blocks_per_sm == nullptr || inplace_blocks_per_sm == nullptr ||
        !recognized_boundary(layout))
        return cudaErrorInvalidValue;
    cudaError_t status = set_tile_kernel_smem_limit();
    if (status != cudaSuccess) return status;
    const bool wall_coupling = layout.hard_wall_channel();
    status = wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              tiled_blocks_per_sm, k_update_tiled_fast_impl<true>,
              kThreads3D, kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              tiled_blocks_per_sm, k_update_tiled_fast_impl<false>,
              kThreads3D, kHaloPipelineBytes);
    if (status != cudaSuccess) return status;
    return wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              inplace_blocks_per_sm, k_update_inplace_fast_impl<true>,
              kThreads3D, kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              inplace_blocks_per_sm, k_update_inplace_fast_impl<false>,
              kThreads3D, kHaloPipelineBytes);
}

cudaError_t sharded_occupancy(int* measurement_blocks_per_sm,
                              int* measured_update_blocks_per_sm,
                              int* fast_update_blocks_per_sm,
                              const SLayout3D& layout) {
    if (measurement_blocks_per_sm == nullptr ||
        measured_update_blocks_per_sm == nullptr ||
        fast_update_blocks_per_sm == nullptr || !recognized_boundary(layout))
        return cudaErrorInvalidValue;
    cudaError_t status = set_tile_kernel_smem_limit();
    if (status != cudaSuccess) return status;
    const bool wall_coupling = layout.hard_wall_channel();
    status = wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              measurement_blocks_per_sm, k_measure_cell_shards_impl<true>,
              kThreads3D, kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              measurement_blocks_per_sm, k_measure_cell_shards_impl<false>,
              kThreads3D, kHaloPipelineBytes);
    if (status != cudaSuccess) return status;
    status = wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              measured_update_blocks_per_sm,
              k_update_tiled_sharded_impl<true>, kThreads3D,
              kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              measured_update_blocks_per_sm,
              k_update_tiled_sharded_impl<false>, kThreads3D,
              kHaloPipelineBytes);
    if (status != cudaSuccess) return status;
    return wall_coupling
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              fast_update_blocks_per_sm,
              k_update_tiled_sharded_fast_impl<true>, kThreads3D,
              kHaloPipelineBytes)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              fast_update_blocks_per_sm,
              k_update_tiled_sharded_fast_impl<false>, kThreads3D,
              kHaloPipelineBytes);
}

}  // namespace detail

void launch_repair_after_fatal(const UpdateArgs3D& args,
                               cudaStream_t stream) {
    if (!valid_runtime_field(args.B, args.layout, args.storage) || args.N <= 0 ||
        !args.selection.valid(args.N) ||
        args.phi_in == nullptr || args.cells == nullptr ||
        args.global_flags == nullptr)
        return;
    k_repair_after_fatal<<<1, kThreads3D, 0, stream>>>(args);
}

void launch_repair_promoted_after_fatal(
    const UpdateArgs3D& args, float* const* promoted_phi_in,
    float* const* promoted_phi_out, const int* promoted_ids,
    int promoted_count, int promoted_edge, cudaStream_t stream) {
    if (!valid_runtime_field(promoted_edge, args.layout, args.storage) || args.N <= 0 ||
        promoted_phi_in == nullptr || promoted_phi_out == nullptr ||
        promoted_ids == nullptr || promoted_count <= 0 ||
        args.cells == nullptr || args.global_flags == nullptr)
        return;
    k_repair_promoted_after_fatal<<<1, kThreads3D, 0, stream>>>(
        args, promoted_phi_in, promoted_phi_out, promoted_ids,
        promoted_count, promoted_edge);
}

void launch_finalize_origins(CellState3D* cells, int N,
                             std::uint32_t* global_flags,
                             cudaStream_t stream) {
    if (cells == nullptr || N <= 0) return;
    k_finalize_origins<<<(N + 127) / 128, 128, 0, stream>>>(
        cells, N, global_flags);
}

void launch_restore_cells_after_fatal(
    CellState3D* cells, const CellState3D* accepted_cells, int N,
    const std::uint32_t* global_flags, std::uint32_t* support_requests,
    cudaStream_t stream) {
    if (cells == nullptr || accepted_cells == nullptr || N <= 0 ||
        global_flags == nullptr)
        return;
    k_restore_cells_after_fatal<<<(N + 127) / 128, 128, 0, stream>>>(
        cells, accepted_cells, N, global_flags, support_requests);
}

void launch_advance_step(std::uint64_t* step,
                         const std::uint32_t* global_flags,
                         cudaStream_t stream) {
    if (step != nullptr)
        k_advance_step<<<1, 1, 0, stream>>>(step, global_flags);
}

void launch_pack_trajectory(const CellState3D* cells, TrajPackedCell3D* out,
                            int N, const SLayout3D& layout,
                            cudaStream_t stream, int channel_padding) {
    if (cells == nullptr || out == nullptr || N <= 0) return;
    k_pack_trajectory<<<(N + 127) / 128, 128, 0, stream>>>(
        cells, out, N, layout, channel_padding);
}

void launch_measure_wall_diagnostics(const float* phi,
                                 float* const* promoted_phi,
                                 CellState3D* cells, int N, int B,
                                 const SLayout3D& layout,
                                 const float* wall_psi_sq,
                                 int channel_height, int channel_padding,
                                 cudaStream_t stream, CellFieldStorage3D storage) {
    if (!phi || !cells || N <= 0 || !valid_field_storage(storage, B, layout)) return;
    k_measure_wall_diagnostics<<<N, kThreads3D, 0, stream>>>(
        phi, promoted_phi, cells, N, B, layout, wall_psi_sq,
        channel_height, channel_padding, storage);
}

void launch_verify_cells(const float* phi, const CellState3D* cells,
                         VerifyCell3D* out, int N, int B,
                         const SLayout3D& layout, cudaStream_t stream,
                         CellFieldStorage3D storage) {
    if (phi == nullptr || cells == nullptr || out == nullptr || N <= 0 ||
        B <= 0 || B % kBrickAlignment != 0 || !valid_field_storage(storage, B, layout))
        return;
    k_verify_cells<<<N, kThreads3D, 0, stream>>>(
        phi, cells, out, N, B, layout, storage);
}

void launch_verify_promoted(float* const* promoted_phi,
                            const CellState3D* cells, VerifyCell3D* out,
                            const int* promoted_ids, int N, int promoted_count,
                            int promoted_edge, const SLayout3D& layout,
                            cudaStream_t stream, CellFieldStorage3D storage) {
    if (promoted_phi == nullptr || cells == nullptr || out == nullptr ||
        promoted_ids == nullptr || N <= 0 || promoted_count <= 0 ||
        promoted_edge <= 0 || promoted_edge % kBrickAlignment != 0 ||
        !valid_field_storage(storage, promoted_edge, layout))
        return;
    k_verify_promoted<<<promoted_count, kThreads3D, 0, stream>>>(
        promoted_phi, cells, out, promoted_ids, N, promoted_count,
        promoted_edge, layout, storage);
}

void launch_verify_S(const std::uint32_t* S, const SLayout3D& layout,
                     std::uint32_t* out_max, cudaStream_t stream) {
    if (S == nullptr || out_max == nullptr || layout.words() == 0) return;
    cudaMemsetAsync(out_max, 0, sizeof(std::uint32_t), stream);
    const int blocks = static_cast<int>(std::min<std::size_t>(
        4096, (layout.words() + kThreads3D - 1) / kThreads3D));
    k_verify_S<<<std::max(1, blocks), kThreads3D, 0, stream>>>(
        S, layout.words(), out_max);
}

}  // namespace pf3d
