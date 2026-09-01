// Fully periodic fast update: one CTA per cell walks y-strips through a four-plane
// shared-memory ring while cp.async loads the next plane. A register stencil
// reduces shared loads from 27 to 9 per voxel. Every destination is written
// once, and arithmetic order matches k_update_tiled_fast bit for bit. This
// rolling-plane path requires periodic z and is not the substrate boundary
// implementation.
#include "../../include/pf3d/kernels.cuh"

#include "device_common.cuh"

#include <cuda_pipeline.h>

#include <cstdint>

namespace pf3d {
namespace {

constexpr int kSlabStripY = 16;                  // destination rows per strip
constexpr int kSlabSrcRows = kSlabStripY + 2;    // staged source rows
constexpr int kSlabRing = 4;                     // strip-planes in flight

// Stage one source strip-plane (kSlabSrcRows rows of B+2 values at the
// recentring offset) into a ring slot; absent source voxels become exact
// zeros.  Row stride in shared is B+2.
__device__ __forceinline__ void slab_prefetch_plane(const float* source, int B,
                                                     int src_z, int src_y0,
                                                     int src_x0, float* slot) {
    const int row_len = B + 2;
    const int elements = kSlabSrcRows * row_len;
    const bool z_ok = static_cast<unsigned>(src_z) < static_cast<unsigned>(B);
    for (int q = static_cast<int>(threadIdx.x); q < elements;
         q += kThreads3D) {
        const int row = q / row_len;
        const int col = q - row * row_len;
        const int y = src_y0 + row;
        const int x = src_x0 + col;
        if (z_ok &&
            static_cast<unsigned>(y) < static_cast<unsigned>(B) &&
            static_cast<unsigned>(x) < static_cast<unsigned>(B)) {
            __pipeline_memcpy_async(&slot[q],
                                    &source[local_index(x, y, src_z, B)],
                                    sizeof(float));
        } else {
            slot[q] = 0.0f;
        }
    }
    __pipeline_commit();
}

}  // namespace

namespace detail {

__global__ __launch_bounds__(kThreads3D, 3)
void k_update_periodic_slab_walk_fast(UpdateArgs3D args) {
    extern __shared__ float ring[];  // kSlabRing * kSlabSrcRows * (B+2)
    const int n = static_cast<int>(blockIdx.x);
    if (n >= args.N) return;
    if (cell_is_promoted(args.cells[n], args.B)) return;
    const int B = args.B;
    const std::size_t words = cell_words(B);
    CellState3D* state = &args.cells[n];
    const float* source = args.phi_in + static_cast<std::size_t>(n) * words;
    float* destination = args.phi_out + static_cast<std::size_t>(n) * words;

    __shared__ int abort_flag;
    if (threadIdx.x == 0)
        abort_flag = fatal_flags_present(state, args.global_flags) ? 1 : 0;
    __syncthreads();
    const int origin_x = wrap_origin(state->origin_x, args.layout.nx);
    const int origin_y = wrap_origin(state->origin_y, args.layout.ny);
    const int origin_z = wrap_origin(state->origin_z, args.layout.nz);
    if (abort_flag != 0) {
        // Preserve the last accepted field exactly; see process_update_cell.
        for (std::size_t q = threadIdx.x; q < words; q += kThreads3D) {
            const float value = source[q];
            destination[q] = value;
            if (args.S_out != nullptr && isfinite(value)) {
                const int x = static_cast<int>(q % static_cast<std::size_t>(B));
                const int y = static_cast<int>((q / static_cast<std::size_t>(B))
                                              % static_cast<std::size_t>(B));
                const int z = static_cast<int>(
                    q / (static_cast<std::size_t>(B) * B));
                scatter_value(
                    args.S_out,
                    s_index(args.layout,
                            wrap_offset(origin_x, x, args.layout.nx),
                            wrap_offset(origin_y, y, args.layout.ny),
                            wrap_offset(origin_z, z, args.layout.nz)),
                    value, state, args.global_flags);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            state->pending_shift_x = 0;
            state->pending_shift_y = 0;
            state->pending_shift_z = 0;
        }
        return;
    }

    const int shift_x = state->pending_shift_x;
    const int shift_y = state->pending_shift_y;
    const int shift_z = state->pending_shift_z;
    const float gamma = state->gamma;
    const float velocity_x = state->velocity_x;
    const float velocity_y = state->velocity_y;
    const float velocity_z = state->velocity_z;
    const float bulk = args.bulk_scale * gamma;
    const float volume = static_cast<float>(
        args.volume_scale * (args.V0 - state->V));

    const int row_len = B + 2;
    const int plane_words = kSlabSrcRows * row_len;
    const int runs_per_row = 16;             // threads per destination row
    const int run = (B + runs_per_row - 1) / runs_per_row;
    const int row_of_thread = static_cast<int>(threadIdx.x) / runs_per_row;
    const int run_of_thread = static_cast<int>(threadIdx.x) % runs_per_row;
    const int x_begin = run_of_thread * run;

    for (int y0 = 0; y0 < B; y0 += kSlabStripY) {
        const int src_y0 = y0 + shift_y - 1;
        const int src_x0 = shift_x - 1;
        for (int p = -1; p <= 1; ++p) {
            const int src_z = shift_z + p;
            slab_prefetch_plane(source, B, src_z, src_y0, src_x0,
                                ring + (((src_z % kSlabRing) + kSlabRing)
                                        % kSlabRing) * plane_words);
        }
        for (int dz = 0; dz < B; ++dz) {
            const int ahead = dz + shift_z + 2;
            slab_prefetch_plane(source, B, ahead, src_y0, src_x0,
                                ring + (((ahead % kSlabRing) + kSlabRing)
                                        % kSlabRing) * plane_words);
            __pipeline_wait_prior(1);
            __syncthreads();

            const int dy = y0 + row_of_thread;
            // Threads whose whole run lies beyond the brick must not touch
            // the ring: their column preload would index past the staged
            // rows of the last plane slot.
            if (dy < B && x_begin < B) {
                const int ring_row = row_of_thread + 1;
                const float* plane0 = ring
                    + ((((dz + shift_z - 1) % kSlabRing) + kSlabRing)
                       % kSlabRing) * plane_words;
                const float* plane1 = ring
                    + ((((dz + shift_z) % kSlabRing) + kSlabRing)
                       % kSlabRing) * plane_words;
                const float* plane2 = ring
                    + ((((dz + shift_z + 1) % kSlabRing) + kSlabRing)
                       % kSlabRing) * plane_words;
                const float* planes[3] = { plane0, plane1, plane2 };
                const std::size_t dest_row = local_index(0, dy, dz, B);
                const int wy = wrap_offset(origin_y, dy + shift_y,
                                           args.layout.ny);
                const int wz = wrap_offset(origin_z, dz + shift_z,
                                           args.layout.nz);
                const bool src_yz_ok =
                    static_cast<unsigned>(dy + shift_y)
                        < static_cast<unsigned>(B) &&
                    static_cast<unsigned>(dz + shift_z)
                        < static_cast<unsigned>(B);
                const std::size_t s_row = s_index(args.layout, 0, wy, wz);
                if (!src_yz_ok) {
                    // The whole source row is absent: exact zeros, as the
                    // tiled path's destination clear provided.
                    for (int i = 0; i < run; ++i) {
                        const int x = x_begin + i;
                        if (x >= B) break;
                        destination[dest_row + x] = 0.0f;
                    }
                } else {
                    // Rotate three statically named register columns; prefetch
                    // the aggregate row one triple ahead of the stencil.
                    auto load_col = [&](float* col, int c) {
#pragma unroll
                        for (int dzc = 0; dzc < 3; ++dzc)
#pragma unroll
                            for (int dyc = 0; dyc < 3; ++dyc)
                                col[dyc * 3 + dzc] =
                                    planes[dzc][(ring_row - 1 + dyc) * row_len
                                                + c];
                    };
                    auto agg_index = [&](int x) {
                        const int sx = x + shift_x;
                        const int safe =
                            static_cast<unsigned>(sx)
                                    < static_cast<unsigned>(B) ? sx : 0;
                        return s_row + static_cast<std::size_t>(
                            wrap_offset(origin_x, safe, args.layout.nx));
                    };
                    auto step = [&](const float* wl, const float* wc,
                                    float* wr, int x,
                                    std::uint32_t aggregate) {
                        if (x >= B) return;
                        load_col(wr, x + 2);
                        const int sx = x + shift_x;
                        if (static_cast<unsigned>(sx)
                                >= static_cast<unsigned>(B)) {
                            destination[dest_row + x] = 0.0f;
                            return;
                        }
                        const float centre = wc[4];  // (dy=0,dz=0)
                        if (!isfinite(centre)) {
                            raise_flag(state, args.global_flags,
                                       FLAG3D_NONFINITE);
                            destination[dest_row + x] = 0.0f;
                            return;
                        }
                        float lap, gx, gy, gz;
                        stencil27_cols(wl, wc, wr, &lap, &gx, &gy, &gz);
                        const float other = other_field(
                            aggregate, centre, state, args.global_flags);
                        const float rhs = gamma * lap
                            - bulk * centre * (1.0f - centre)
                                  * (1.0f - 2.0f * centre)
                            + volume * centre
                            - args.interaction_coeff * centre * other
                            - (velocity_x * gx + velocity_y * gy
                               + velocity_z * gz);
                        const float next = centre + args.dt * rhs;
                        if (!isfinite(next)) {
                            raise_flag(state, args.global_flags,
                                       FLAG3D_NONFINITE);
                            destination[dest_row + x] = 0.0f;
                            return;
                        }
                        destination[dest_row + x] = next;
                        if (args.S_out != nullptr) {
                            scatter_value(args.S_out, agg_index(x), next,
                                          state, args.global_flags);
                        }
                    };
                    float c0[9], c1[9], c2[9];
                    load_col(c0, x_begin);
                    load_col(c1, x_begin + 1);
                    // Steps past the thread's run belong to the next thread;
                    // executing them would scatter those voxels' aggregate
                    // contributions twice.
                    for (int i = 0; i < run; i += 3) {
                        const int x0i = x_begin + i;
                        const std::uint32_t s0 = args.S_in[agg_index(x0i)];
                        const std::uint32_t s1 = x0i + 1 < B
                            ? args.S_in[agg_index(x0i + 1)] : 0u;
                        const std::uint32_t s2 = x0i + 2 < B
                            ? args.S_in[agg_index(x0i + 2)] : 0u;
                        step(c0, c1, c2, x0i, s0);
                        if (i + 1 < run) step(c1, c2, c0, x0i + 1, s1);
                        if (i + 2 < run) step(c2, c0, c1, x0i + 2, s2);
                    }
                }
            }
            __syncthreads();
        }
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // On recentering, detect supported source values shifted outside the brick
    // even though those voxels have no destination write.
    if (shift_x != 0 || shift_y != 0 || shift_z != 0) {
        auto fetch_global = [&](int x, int y, int z) {
            return (static_cast<unsigned>(x) < static_cast<unsigned>(B) &&
                    static_cast<unsigned>(y) < static_cast<unsigned>(B) &&
                    static_cast<unsigned>(z) < static_cast<unsigned>(B))
                ? source[local_index(x, y, z, B)] : 0.0f;
        };
        for (std::size_t q = threadIdx.x; q < words; q += kThreads3D) {
            const int x = static_cast<int>(q % static_cast<std::size_t>(B));
            const int y = static_cast<int>((q / static_cast<std::size_t>(B))
                                          % static_cast<std::size_t>(B));
            const int z = static_cast<int>(
                q / (static_cast<std::size_t>(B) * B));
            const bool clipped =
                static_cast<unsigned>(x - shift_x)
                    >= static_cast<unsigned>(B) ||
                static_cast<unsigned>(y - shift_y)
                    >= static_cast<unsigned>(B) ||
                static_cast<unsigned>(z - shift_z)
                    >= static_cast<unsigned>(B);
            if (!clipped) continue;
            auto fetch = [&](int ddx, int ddy, int ddz) {
                return fetch_global(x + ddx, y + ddy, z + ddz);
            };
            const float centre = fetch(0, 0, 0);
            if (!isfinite(centre)) {
                raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                continue;
            }
            float lap, gx, gy, gz;
            stencil27_at(fetch, &lap, &gx, &gy, &gz);
            if (centre == 0.0f && lap == 0.0f && gx == 0.0f &&
                gy == 0.0f && gz == 0.0f)
                continue;
            const int wx = wrap_offset(origin_x, x, args.layout.nx);
            const int wy = wrap_offset(origin_y, y, args.layout.ny);
            const int wz = wrap_offset(origin_z, z, args.layout.nz);
            const std::uint32_t aggregate =
                args.S_in[s_index(args.layout, wx, wy, wz)];
            const float other = other_field(aggregate, centre, state,
                                            args.global_flags);
            const float rhs = gamma * lap
                - bulk * centre * (1.0f - centre) * (1.0f - 2.0f * centre)
                + volume * centre
                - args.interaction_coeff * centre * other
                - (velocity_x * gx + velocity_y * gy + velocity_z * gz);
            const float next = centre + args.dt * rhs;
            if (!isfinite(next)) {
                raise_flag(state, args.global_flags, FLAG3D_NONFINITE);
                continue;
            }
            if (fabsf(next) > kSupportEps)
                raise_flag(state, args.global_flags, FLAG3D_DESTINATION_CLIP);
        }
    }
}

std::size_t update_periodic_slab_walk_shared_bytes(int B) {
    return static_cast<std::size_t>(kSlabRing) * kSlabSrcRows
         * static_cast<std::size_t>(B + 2) * sizeof(float);
}

bool launch_update_periodic_slab_walk_fast(const UpdateArgs3D& args,
                                           cudaStream_t stream) {
    // This rolling-plane kernel folds z and is therefore fully periodic only.
    if (!args.layout.periodic_xyz() ||
        !valid_weighted_wall_field(args.wall, args.layout) ||
        !valid_runtime_geometry(args.B, args.layout) || args.N <= 0 ||
        args.phi_in == nullptr || args.phi_out == nullptr ||
        args.phi_in == args.phi_out || args.S_in == nullptr ||
        (args.S_out != nullptr && args.S_out == args.S_in) ||
        args.cells == nullptr || args.compute_surface)
        return false;
    const std::size_t shared =
        update_periodic_slab_walk_shared_bytes(args.B);
    static int configured_for = -1;
    if (configured_for != args.B) {
        if (cudaFuncSetAttribute(
                reinterpret_cast<const void*>(
                    k_update_periodic_slab_walk_fast),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared)) != cudaSuccess)
            return false;
        configured_for = args.B;
    }
    k_update_periodic_slab_walk_fast<<<args.N, kThreads3D, shared, stream>>>(
        args);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace detail
}  // namespace pf3d
