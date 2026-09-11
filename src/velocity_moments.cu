#include "../include/velocity_moments.cuh"
#include <cmath>

namespace pf {
using velocity::MomentAccum;
namespace {
constexpr int NT = 256, NW = NT / 32;

__device__ float phi_at(const float* tile, const ShapeClass& sc, int x, int y) {
    if ((unsigned)x >= (unsigned)sc.wx || (unsigned)y >= (unsigned)sc.wy)
        return 0.0f;
    return tile[(sc.ty0 + y) * kTilePitch + sc.tx0 + x];
}
__device__ float other_at(const StepArgs& A, const CellState& c, int x, int y, float p) {
    const int gx = (c.gx0 + x) % A.L, gy = (c.gy0 + y) % A.L;
    const uint32_t all = A.S_rd[(size_t)gy * A.P + gx], self = q_of(p);
    // The production kernel handles invalid shared-field state. An invalid
    // observer value is retained as NaN so analysis also refuses the sample.
    return all >= self ? (float)(all - self) * kQInvF : nanf("");
}
template <int K> __device__ void sum_block(double (&v)[K], double* shared) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        for (int d = 16; d; d >>= 1)
            v[j] += __shfl_down_sync(0xffffffffu, v[j], d);
        if (lane == 0)
            shared[j * NW + warp] = v[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
            double s = 0.;
            for (int w = 0; w < NW; ++w)
                s += shared[j * NW + w];
            shared[j * NW] = s;
        }
    }
    __syncthreads();
}

__global__ void observe(StepArgs A, MomentAccum* out, double physical_dt) {
    const int n = blockIdx.x;
    const CellState& c = A.cell[n];
    const ShapeClass sc = class_of(c.cls);
    const float* tile = A.phi_in + (size_t)n * kTileArea;
    __shared__ double sums[8 * NW];
    __shared__ double velocity[4];
    double force[2] = {0., 0.};
    for (int k = threadIdx.x; k < sc.wx * sc.wy; k += NT) {
        const int x = k % sc.wx, y = k / sc.wx;
        const float p = phi_at(tile, sc, x, y);
        const float gx = .5f * (phi_at(tile, sc, x + 1, y) - phi_at(tile, sc, x - 1, y));
        const float gy = .5f * (phi_at(tile, sc, x, y + 1) - phi_at(tile, sc, x, y - 1));
        const float so = other_at(A, c, x, y, p);
        // Same float product as the production interaction-force integrand.
        force[0] += (double)(p * gx * so);
        force[1] += (double)(p * gy * so);
    }
    sum_block(force, sums);
    if (threadIdx.x == 0) {
        const unsigned long long step = *A.step_rd;
        const Philox4 r = philox4x32_10(
            (uint32_t)step, (uint32_t)(step >> 32), (uint32_t)c.global_id, 0xA5A5A5A5u,
            (uint32_t)A.polarity_seed, (uint32_t)(A.polarity_seed >> 32));
        float theta = c.theta;
        if (philox_uniform53(r.v[0], r.v[1]) < A.p_tumble)
            theta = (float)(2.0 * kPi * philox_uniform53(r.v[2], r.v[3]));
        float sn, co;
        sincosf(theta, &sn, &co);
        velocity[0] = (double)c.v_A * (double)co;
        velocity[1] = (double)c.v_A * (double)sn;
        velocity[2] = (double)A.mot_coeff * sums[0];
        velocity[3] = (double)A.mot_coeff * sums[NW];
    }
    __syncthreads();
    const double cx = c.Cx / c.V, cy = c.Cy / c.V;
    const double dw = (double)(A.bulk_scale * c.gamma);
    const double vol = (double)(float)(A.vol_scale * (A.A0 - c.V));
    // The continuum components use these same rounded model coefficients,
    // but their variational arithmetic and moments are evaluated in double.
    const double vx = (double)(float)(velocity[0] + velocity[2]);
    const double vy = (double)(float)(velocity[1] + velocity[3]);
    double moments[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
    for (int k = threadIdx.x; k < sc.wx * sc.wy; k += NT) {
        const int x = k % sc.wx, y = k / sc.wx;
        const double p = (double)phi_at(tile, sc, x, y);
        const double e = phi_at(tile, sc, x + 1, y), w = phi_at(tile, sc, x - 1, y);
        const double nn = phi_at(tile, sc, x, y + 1), s = phi_at(tile, sc, x, y - 1);
        const double ne = phi_at(tile, sc, x + 1, y + 1), nw = phi_at(tile, sc, x - 1, y + 1);
        const double se = phi_at(tile, sc, x + 1, y - 1), sw = phi_at(tile, sc, x - 1, y - 1);
        const double lap = (4. * (nn + s + e + w) + ne + nw + se + sw - 20. * p) / 6.;
        const double gx = .5 * (e - w), gy = .5 * (nn - s);
        const double so = (double)other_at(A, c, x, y, (float)p);
        const double source[4] = {(double)c.gamma * lap - dw * p * (1. - p) * (1. - 2. * p),
                                  -(double)A.rep_coeff * p * so, vol * p, -(vx * gx + vy * gy)};
        const double wx = 2. * ((double)x - cx) * p / c.V;
        const double wy = 2. * ((double)y - cy) * p / c.V;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            moments[2 * j] += wx * source[j];
            moments[2 * j + 1] += wy * source[j];
        }
    }
    sum_block(moments, sums);
    if (threadIdx.x == 0) {
        MomentAccum& z = out[n];
        for (int j = 0; j < 4; ++j)
            z.q[j] += physical_dt * velocity[j];
        for (int j = 0; j < 6; ++j)
            z.q[4 + j] += physical_dt * sums[j * NW];
        z.q[10] += physical_dt * (sums[6 * NW] - velocity[0] - velocity[2]);
        z.q[11] += physical_dt * (sums[7 * NW] - velocity[1] - velocity[3]);
        ++z.count;
    }
}
} // namespace

void launch_velocity_moments(const StepArgs& A, MomentAccum* out, double physical_dt,
                             cudaStream_t stream) {
    observe<<<A.N, NT, 0, stream>>>(A, out, physical_dt);
}

} // namespace pf
