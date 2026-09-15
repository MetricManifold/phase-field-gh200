#pragma once
// Compile-time observation specializations. Disabled paths have no observer
// arithmetic, memory traffic, reductions, or per-step observer kernel.
#include "kernels.cuh"

namespace pf {
namespace velocity {
// The existing scalar broadcast reserves room for these three aligned doubles.
constexpr int kGeometryWord = 32;
struct Geometry {
    double twice_inverse_volume, cx, cy;
};
static_assert(kGeometryWord * sizeof(float) % alignof(Geometry) == 0);
static_assert(kGeometryWord * sizeof(float) + sizeof(Geometry) <= kBcastCtrlN * 4);

__device__ __forceinline__ void broadcast_geometry(float* broadcast, const CellState& cell) {
    auto* g = reinterpret_cast<Geometry*>(broadcast + kGeometryWord);
    g->twice_inverse_volume = 2.0 / cell.V;
    g->cx = cell.Cx / cell.V;
    g->cy = cell.Cy / cell.V;
}

template <bool Enabled> struct RelaxMoments {
    __device__ explicit RelaxMoments(const float*) {}
    template <typename... Args> __device__ void pixel(Args...) {}
    __device__ void finish(const StepArgs&, int, double*) {}
};

template <> struct RelaxMoments<true> {
    double gx = 0.0, gy = 0.0, kx = 0.0, ky = 0.0;
    const volatile Geometry* geometry;

    __device__ explicit RelaxMoments(const float* broadcast)
        : geometry(reinterpret_cast<const Geometry*>(broadcast + kGeometryWord)) {}

    __device__ __forceinline__ void pixel(int x, int y, float p, float e, float w, float north,
                                          float south, float ne, float nw, float se, float sw,
                                          float other, float gamma, float bulk,
                                          float repulsion) {
        const double v = p;
        const double lap =
            (4.0 * (double(north) + south + e + w) + double(ne) + nw + se + sw - 20.0 * v) /
            6.0;
        const double weight = geometry->twice_inverse_volume * v;
        const double interfacial =
            weight * (double(gamma) * lap - double(bulk) * v * (1.0 - v) * (1.0 - 2.0 * v));
        gx += (double(x) - geometry->cx) * interfacial;
        gy += (double(y) - geometry->cy) * interfacial;
        const double overlap = -weight * double(repulsion) * v * double(other);
        kx += (double(x) - geometry->cx) * overlap;
        ky += (double(y) - geometry->cy) * overlap;
    }

    __device__ __forceinline__ void finish(const StepArgs& state, int cell, double* reduction) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
        for (int delta = 16; delta; delta >>= 1) {
            gx += __shfl_down_sync(0xffffffffu, gx, delta);
            gy += __shfl_down_sync(0xffffffffu, gy, delta);
            kx += __shfl_down_sync(0xffffffffu, kx, delta);
            ky += __shfl_down_sync(0xffffffffu, ky, delta);
        }
        if (lane == 0) {
            double* r = reduction + warp * kRedSlots;
            r[0] = gx;
            r[1] = gy;
            r[2] = kx;
            r[3] = ky;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            auto& m = state.moments[cell];
            const double mrel = static_cast<double>(state.cell[cell].M_pf) /
                                kPhaseFieldM0;
            for (int j = 0; j < kSpatialValues; ++j) {
                double rate = 0.0;
                for (int w = 0; w < kWarpsPerBlock; ++w)
                    rate += reduction[w * kRedSlots + j];
                // Only the passive interfacial and overlap rates live here.
                // The uniform area source has exactly zero centroid moment.
                if (mrel != 1.0) rate *= mrel;
                if (m.spatial_count > 1)
                    m.correction[j] +=
                        linear_correction(state.physical_dt, m.last_interval, m.held[j], rate);
                m.held[j] = rate;
            }
        }
        __syncthreads();
    }
};

// Called by lane zero after the physics velocity reduction, before phi changes.
// Broadcast words 11..15 are the existing v_A, cos(theta), sin(theta), vx, vy.
template <bool Enabled, bool Spatial>
__device__ __forceinline__ void
record_advection(const StepArgs& state, int cell, const float* broadcast,
                 const double* reduction, double force_x, double force_y) {
    if constexpr (Enabled) {
        auto& c = state.cell[cell];
        auto& a = c.advection;
        const double ax = double(broadcast[11]) * broadcast[12];
        const double ay = double(broadcast[11]) * broadcast[13];
        const double ix = double(state.mot_coeff) * force_x;
        const double iy = double(state.mot_coeff) * force_y;
        const double dt = state.physical_dt;
        if constexpr (Spatial) {
            auto& m = state.moments[cell];
            begin_spatial_sample(m, a, dt);
            // Bx = sum(phi(x,y)*phi(x+1,y))/V (and likewise in y).
            // This same-cell neighbor overlap is the centered-gradient
            // integration-by-parts factor, not overlap with another cell.
            double bx = 0.0, by = 0.0;
            for (int w = 0; w < kWarpsPerBlock; ++w) {
                bx += reduction[w * kRedSlots + 2];
                by += reduction[w * kRedSlots + 3];
            }
            m.held[kSpatialValues] = bx / c.V;
            m.held[kSpatialValues + 1] = by / c.V;
        }
        a.q[kActive] += dt * ax;
        a.q[kActive + 1] += dt * ay;
        a.q[kInteraction] += dt * ix;
        a.q[kInteraction + 1] += dt * iy;
        a.q[kRounding] += dt * (double(broadcast[14]) - ax - ix);
        a.q[kRounding + 1] += dt * (double(broadcast[15]) - ay - iy);
        ++a.count;
    }
}
} // namespace velocity
} // namespace pf
