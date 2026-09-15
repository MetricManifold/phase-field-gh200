// Independent CPU/GPU centroid-quotient oracle, plus bitwise observation
// on/off checks on every shared and fallback geometry with forced tumbles.
#include "../include/checkpoint.cuh"
#include "../include/velocity_moments.cuh"
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace pf;
using velocity::MomentAccum;
#define GPU(call)                                                                              \
    do {                                                                                       \
        auto e = (call);                                                                       \
        if (e != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e));                    \
            std::exit(2);                                                                      \
        }                                                                                      \
    } while (0)

static CheckpointData synthetic(bool variable_mobility) {
    CheckpointData d;
    d.n = kNumClasses;
    d.params.num_cells = d.n;
    d.params.Nx = d.params.Ny = 701;
    d.step = 9000037;
    d.t = d.step * d.params.dt;
    d.cells.resize(d.n);
    d.phi.assign((size_t)d.n * kTileArea, 0.f);
    for (int i = 0; i < d.n; ++i) {
        auto& c = d.cells[i];
        c.global_id = 17 + i * 13;
        c.cls = (uint8_t)i;
        const auto sc = class_of(c.cls);
        c.origin[0] = (680 + i * 100 - sc.wx / 2 + 701) % 701;
        c.origin[1] = (11 + i * 130 - sc.wy / 2 + 701) % 701;
        c.gamma = i % 2 ? .35f : 1.f;
        c.M_pf = variable_mobility ? .25f + .125f * (i % 5) : .5f;
        c.v_A = .01f;
        c.R_tgt = 49.f;
        c.theta = .4f + float(i);
        for (int y = 0; y < sc.wy; ++y)
            for (int x = 0; x < sc.wx; ++x) {
                double X = (x - .5 * sc.wx) / 18., Y = (y - .5 * sc.wy) / 14.;
                double z = .79 * std::exp(-.5 * (X * X + Y * Y)) * (1. + .08 * X * Y + .06 * X);
                d.phi[(size_t)i * kTileArea + (sc.ty0 + y) * kTilePitch + sc.tx0 + x] =
                    (float)z;
            }
    }
    return d;
}

struct ReferenceMetrics {
    double max_force, max_moment, max_quotient, max_area, max_euler, max_half;
};
static ReferenceMetrics check_reference(const CheckpointData& d,
                                        const std::vector<CellState>& cs,
                                        const std::vector<uint32_t>& S, const StepArgs& A,
                                        const std::vector<MomentAccum>& q, bool synth) {
    const auto& p = d.params;
    const int N = d.n, L = A.L, P = A.P;
    const auto step = static_cast<unsigned long long>(d.step);
    double max_force = 0., max_moment = 0., max_quotient = 0., max_area = 0., max_euler = 0.,
           max_half = 0.;
    for (int i = 0; i < N; ++i) {
        if (q[i].count != 1)
            std::exit(5);
        const auto& c = cs[i];
        const auto sc = class_of(c.cls);
        auto f = [&](int x, int y) -> double {
            if (x < 0 || y < 0 || x >= sc.wx || y >= sc.wy)
                return 0.;
            return d.phi[(size_t)i * kTileArea + (sc.ty0 + y) * kTilePitch + sc.tx0 + x];
        };
        auto other = [&](int x, int y) -> float {
            uint32_t self = q_of((float)f(x, y));
            uint32_t all = S[(size_t)((c.gy0 + y) % L) * P + (c.gx0 + x) % L];
            if (all < self)
                std::exit(6);
            return (float)(all - self) * kQInvF;
        };
        double fx = 0, fy = 0;
        for (int y = 0; y < sc.wy; ++y)
            for (int x = 0; x < sc.wx; ++x) {
                float v = (float)f(x, y), ox = other(x, y);
                float gx = .5f * ((float)f(x + 1, y) - (float)f(x - 1, y));
                float gy = .5f * ((float)f(x, y + 1) - (float)f(x, y - 1));
                fx += (double)(v * gx * ox);
                fy += (double)(v * gy * ox);
            }
        auto rng = philox4x32_10((uint32_t)step, (uint32_t)(step >> 32), (uint32_t)c.global_id,
                                 0xA5A5A5A5u, (uint32_t)A.polarity_seed,
                                 (uint32_t)(A.polarity_seed >> 32));
        float theta = c.theta;
        if (philox_uniform53(rng.v[0], rng.v[1]) < A.p_tumble)
            theta = (float)(2 * kPi * philox_uniform53(rng.v[2], rng.v[3]));
        double ax = (double)c.v_A * (double)std::cos(theta),
               ay = (double)c.v_A * (double)std::sin(theta);
        double ix = (double)A.mot_coeff * fx, iy = (double)A.mot_coeff * fy;
        double expected[4] = {ax, ay, ix, iy};
        for (int k = 0; k < 4; ++k)
            max_force = std::fmax(max_force, std::fabs(q[i].q[k] / p.dt - expected[k]));
        // Use observed advection only for the source terms; force itself was
        // checked independently above (CPU/GPU sinf can differ by one ulp).
        double vx = (float)((q[i].q[0] + q[i].q[2]) / p.dt),
               vy = (float)((q[i].q[1] + q[i].q[3]) / p.dt);
        const double center[2] = {c.Cx / c.V, c.Cy / c.V};
        double mom[4][2] = {};
        double plus[4][3] = {}, minus[4][3] = {};
        // Synthetic fields deliberately overlap strongly; their source is
        // larger, requiring a smaller quotient perturbation.
        double evolved[2][3] = {};
        const double eps = synth ? 1e-6 : 1e-3;
        for (int y = 0; y < sc.wy; ++y)
            for (int x = 0; x < sc.wx; ++x) {
                double v = f(x, y), gx = (f(x + 1, y) - f(x - 1, y)) / 2.,
                       gy = (f(x, y + 1) - f(x, y - 1)) / 2.;
                double lap = -20. * v / 6.;
                for (int yy = -1; yy <= 1; ++yy)
                    for (int xx = -1; xx <= 1; ++xx)
                        if (xx || yy)
                            lap += f(x + xx, y + yy) * ((xx && yy) ? 1. / 6. : 2. / 3.);
                double mobility = (double)(c.M_pf / (float)kPhaseFieldM0);
                double dw = (float)(A.bulk_scale * c.gamma);
                double sources[4] = {
                    mobility * (c.gamma * lap - dw * v * (1 - v) * (1 - 2 * v)),
                    -mobility * A.rep_coeff * v * (double)other(x, y),
                    mobility * (float)(A.vol_scale * (A.A0 - c.V)) * v, -vx * gx - vy * gy};
                double total = 0.;
                for (int k = 0; k < 4; ++k) {
                    total += sources[k];
                    mom[k][0] += 2 * (x - center[0]) * v * sources[k] / c.V;
                    mom[k][1] += 2 * (y - center[1]) * v * sources[k] / c.V;
                    double vp = v + eps * sources[k], vm = v - eps * sources[k];
                    plus[k][0] += vp * vp;
                    plus[k][1] += x * vp * vp;
                    plus[k][2] += y * vp * vp;
                    minus[k][0] += vm * vm;
                    minus[k][1] += x * vm * vm;
                    minus[k][2] += y * vm * vm;
                }
                for (int h = 0; h < 2; ++h) {
                    double vn = v + p.dt / (h + 1) * total;
                    evolved[h][0] += vn * vn;
                    evolved[h][1] += x * vn * vn;
                    evolved[h][2] += y * vn * vn;
                }
            }
        for (int k = 0; k < 4; ++k)
            for (int xy = 0; xy < 2; ++xy) {
                double target = q[i].q[4 + 2 * k + xy] / p.dt;
                if (k == 3)
                    target += (q[i].q[xy] + q[i].q[2 + xy]) / p.dt;
                max_moment = std::fmax(max_moment, std::fabs(target - mom[k][xy]));
                double derivative =
                    (plus[k][xy + 1] / plus[k][0] - minus[k][xy + 1] / minus[k][0]) / (2 * eps);
                max_quotient = std::fmax(max_quotient, std::fabs(derivative - mom[k][xy]));
                if (k == 2)
                    max_area = std::fmax(max_area, std::fabs(target));
            }
        for (int xy = 0; xy < 2; ++xy) {
            double total = 0;
            for (int k = 0; k < 4; ++k)
                total += mom[k][xy];
            max_euler = std::fmax(
                max_euler,
                std::fabs((evolved[0][xy + 1] / evolved[0][0] - center[xy]) / p.dt - total));
            max_half = std::fmax(
                max_half,
                std::fabs((evolved[1][xy + 1] / evolved[1][0] - center[xy]) / (p.dt / 2) -
                          total));
        }
    }
    return {max_force, max_moment, max_quotient, max_area, max_euler, max_half};
}

static double check_euler_update(const CheckpointData& d, const std::vector<CellState>& before,
                                const std::vector<CellState>& after,
                                const std::vector<uint32_t>& aggregate,
                                const std::vector<float>& updated, const StepArgs& args) {
    // Independent double-precision stencil/RHS on the same pre-step float field.
    // Compare source-window interior points after resolving the integer shift;
    // output outside that window is covered by the separate storage tests.
    double worst = 0.;
    for (int i = 0; i < d.n; ++i) {
        const auto& cell = before[i];
        const auto old_shape = class_of(cell.cls), new_shape = class_of(after[i].cls);
        const auto wrap = [&](int x) { return (x % args.L + args.L) % args.L; };
        int sx = after[i].gx0 - cell.gx0, sy = after[i].gy0 - cell.gy0;
        if (sx > args.L / 2) sx -= args.L;
        if (sx < -args.L / 2) sx += args.L;
        if (sy > args.L / 2) sy -= args.L;
        if (sy < -args.L / 2) sy += args.L;
        const auto field = [&](int x, int y) -> double {
            return d.phi[(size_t)i * kTileArea +
                         (old_shape.ty0 + y) * kTilePitch + old_shape.tx0 + x];
        };
        const double m = (double)cell.M_pf / kPhaseFieldM0;
        const double bulk = (float)(args.bulk_scale * cell.gamma);
        const double area = (float)(args.vol_scale * (args.A0 - cell.V));
        for (int y = 0; y < new_shape.wy; ++y) {
            for (int x = 0; x < new_shape.wx; ++x) {
                const int ox = x + sx, oy = y + sy;
                if (ox < 1 || oy < 1 || ox >= old_shape.wx - 1 || oy >= old_shape.wy - 1)
                    continue;
                const double p = field(ox, oy);
                const double e = field(ox + 1, oy), w = field(ox - 1, oy);
                const double n = field(ox, oy + 1), s = field(ox, oy - 1);
                const double lap = (4. * (e + w + n + s) - 20. * p +
                    field(ox - 1, oy - 1) + field(ox + 1, oy - 1) +
                    field(ox - 1, oy + 1) + field(ox + 1, oy + 1)) / 6.;
                const uint32_t total = aggregate[(size_t)wrap(cell.gy0 + oy) * args.P +
                                                  wrap(cell.gx0 + ox)];
                const float others = (float)(total - q_of((float)p)) * kQInvF;
                const double passive = cell.gamma * lap - bulk * p * (1. - p) * (1. - 2. * p)
                    + area * p - args.rep_coeff * p * others;
                const double advect = after[i].vx * .5 * (e - w) + after[i].vy * .5 * (n - s);
                const double expected = p + args.dt * (m * passive - advect);
                const float actual = updated[(size_t)i * kTileArea +
                    (new_shape.ty0 + y) * kTilePitch + new_shape.tx0 + x];
                worst = std::fmax(worst, std::fabs(actual - expected));
            }
        }
    }
    return worst;
}

static bool check_fused_paths(const CheckpointData& d, const std::vector<CellState>& cs,
                              const std::vector<uint32_t>& S, const std::vector<MomentAccum>& q,
                              StepArgs A, unsigned long long* dt, MomentAccum* dq) {
    const auto& p = d.params;
    const int N = d.n;
    auto* dc = A.cell;
    const auto step = static_cast<unsigned long long>(d.step);
    bool ok = true;
    double fused_error = 0.;
    {
        // Exercise all six actual evolution paths, including promotion,
        // fallback, periodic wrapping, forced tumbles and heterogeneous stiffness.
        float* next;
        uint32_t *sc, *cl, *flags, *perm;
        uint8_t* classes;
        unsigned long long* cursors;
        GPU(cudaMalloc(&next, d.phi.size() * 4));
        GPU(cudaMalloc(&sc, S.size() * 4));
        GPU(cudaMalloc(&cl, S.size() * 4));
        GPU(cudaMalloc(&flags, FLAG_COUNT * 4));
        GPU(cudaMalloc(&perm, N * 4));
        GPU(cudaMalloc(&classes, N));
        GPU(cudaMalloc(&cursors, 16));
        std::vector<uint32_t> order(N);
        std::vector<uint8_t> cls(N);
        for (int i = 0; i < N; ++i) {
            order[i] = i;
            cls[i] = cs[i].cls;
        }
        GPU(cudaMemcpy(perm, order.data(), N * 4, cudaMemcpyHostToDevice));
        A.phi_out = next;
        A.S_sc = sc;
        A.S_cl = cl;
        A.flags = flags;
        A.perm = perm;
        A.cell_cls = classes;
        A.cursor_use = cursors;
        A.cursor_clear = cursors + 1;
        A.step_wr = dt + 1;
        A.parity_out = 1;
        A.physical_dt = p.dt;
        A.full_moment_every = 100;
        A.clear_ahead_words = S.size();
        configure_k_step_smem();
        auto reset = [&]() {
            GPU(cudaMemset(next, 0, d.phi.size() * 4));
            GPU(cudaMemset(sc, 0, S.size() * 4));
            GPU(cudaMemset(cl, 0, S.size() * 4));
            GPU(cudaMemset(flags, 0, FLAG_COUNT * 4));
            GPU(cudaMemset(cursors, 0, 16));
            GPU(cudaMemset(dq, 0, N * sizeof(MomentAccum)));
            GPU(cudaMemcpy(dc, cs.data(), N * sizeof(CellState), cudaMemcpyHostToDevice));
            GPU(cudaMemcpy(classes, cls.data(), N, cudaMemcpyHostToDevice));
            GPU(cudaMemcpy(dt, &step, 8, cudaMemcpyHostToDevice));
        };
        reset();
        A.moments = nullptr;
        launch_step(A, k_step_grid(0), 0, nullptr, 0, 0);
        GPU(cudaDeviceSynchronize());
        std::vector<float> offphi(d.phi.size()), onphi(d.phi.size());
        std::vector<CellState> offcell(N), oncell(N);
        GPU(cudaMemcpy(offphi.data(), next, d.phi.size() * 4, cudaMemcpyDeviceToHost));
        GPU(cudaMemcpy(offcell.data(), dc, N * sizeof(CellState), cudaMemcpyDeviceToHost));
        const double euler_error = check_euler_update(d, cs, offcell, S, offphi, A);
        std::printf("independent_interior_euler_max_abs_error=%.17g\n", euler_error);
        ok = ok && euler_error < 2e-6;
        reset();
        A.moments = dq;
        launch_step(A, k_step_grid(0), 0, nullptr, 0, 0, VelocityStep::Spatial);
        GPU(cudaDeviceSynchronize());
        GPU(cudaMemcpy(onphi.data(), next, d.phi.size() * 4, cudaMemcpyDeviceToHost));
        GPU(cudaMemcpy(oncell.data(), dc, N * sizeof(CellState), cudaMemcpyDeviceToHost));
        std::vector<MomentAccum> fused(N);
        GPU(cudaMemcpy(fused.data(), dq, N * sizeof(MomentAccum), cudaMemcpyDeviceToHost));
        std::vector<uint32_t> checks(FLAG_COUNT);
        GPU(cudaMemcpy(checks.data(), flags, FLAG_COUNT * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < FLAG_COUNT; ++i)
            if (flag_is_fatal(i) && checks[i])
                ok = false;
        for (int i = 0; i < N; ++i) {
            const auto& c = oncell[i];
            auto& z = fused[i];
            const double* v = c.advection.q;
            z.count = c.advection.count;
            for (int j = 0; j < 4; ++j) {
                z.q[j] = v[j];
                z.q[4 + j] = p.dt * z.held[j];
            }
            for (int xy = 0; xy < 2; ++xy)
                z.q[10 + xy] =
                    (z.held[4 + xy] - 1.) * (v[xy] + v[2 + xy]) + z.held[4 + xy] * v[4 + xy];
            for (double& a : oncell[i].advection.q)
                a = 0.;
            oncell[i].advection.count = 0;
        }
        if (std::memcmp(offphi.data(), onphi.data(), d.phi.size() * 4) ||
            std::memcmp(offcell.data(), oncell.data(), N * sizeof(CellState)))
            ok = false;
        for (int i = 0; i < N; ++i) {
            if (fused[i].count != 1)
                ok = false;
            for (int j = 0; j < 12; ++j)
                fused_error = std::fmax(fused_error, std::fabs(fused[i].q[j] - q[i].q[j]));
        }
        ok = ok && fused_error < 2e-9;
        reset();
        A.moments = dq;
        launch_step(A, k_step_grid(0), 0, nullptr, 0, 0, VelocityStep::Advection);
        GPU(cudaDeviceSynchronize());
        GPU(cudaMemcpy(onphi.data(), next, d.phi.size() * 4, cudaMemcpyDeviceToHost));
        GPU(cudaMemcpy(oncell.data(), dc, N * sizeof(CellState), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            if (oncell[i].advection.count != 1)
                ok = false;
            for (int j = 0; j < 4; ++j)
                if (std::fabs(oncell[i].advection.q[j] - q[i].q[j]) > 2e-9)
                    ok = false;
            for (double& a : oncell[i].advection.q)
                a = 0.;
            oncell[i].advection.count = 0;
        }
        if (std::memcmp(offphi.data(), onphi.data(), d.phi.size() * 4) ||
            std::memcmp(offcell.data(), oncell.data(), N * sizeof(CellState)))
            ok = false;

        GPU(cudaFree(next));
        GPU(cudaFree(sc));
        GPU(cudaFree(cl));
        GPU(cudaFree(flags));
        GPU(cudaFree(perm));
        GPU(cudaFree(classes));
        GPU(cudaFree(cursors));
        std::printf("fused_all_classes_readonly_and_reference=%s max_error=%.17g\n",
                    ok ? "PASS" : "FAIL", fused_error);
    }
    return ok;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: probe CHECKPOINT|--synthetic|--synthetic-mobility\n");
        return 2;
    }
    CheckpointData d;
    bool variable_mobility = std::string(argv[1]) == "--synthetic-mobility";
    bool synth = variable_mobility || std::string(argv[1]) == "--synthetic";
    if (synth)
        d = synthetic(variable_mobility);
    else if (!checkpoint_read(argv[1], &d))
        return 2;
    auto p = d.params;
    const int L = p.Nx, P = s_pitch_for(L), N = d.n;
    std::vector<CellState> cs(N);
    std::vector<uint32_t> S((size_t)L * P, 0);
    for (int i = 0; i < N; ++i) {
        const auto& c = d.cells[i];
        auto& z = cs[i];
        const auto sc = class_of(c.cls);
        z.gx0 = c.origin[0];
        z.gy0 = c.origin[1];
        z.cls = c.cls;
        z.global_id = c.global_id;
        z.cls_written[0] = z.cls_written[1] = c.cls;
        z.gamma = c.gamma;
        z.M_pf = c.M_pf;
        z.v_A = c.v_A;
        z.theta = c.theta;
        for (int y = 0; y < sc.wy; ++y)
            for (int x = 0; x < sc.wx; ++x) {
                const float v =
                    d.phi[(size_t)i * kTileArea + (sc.ty0 + y) * kTilePitch + sc.tx0 + x];
                const double w = (double)v * v;
                z.V += w;
                z.Cx += w * x;
                z.Cy += w * y;
                auto& all = S[(size_t)((z.gy0 + y) % L) * P + (z.gx0 + x) % L];
                uint64_t next = (uint64_t)all + q_of(v);
                if (next > UINT32_MAX)
                    return 3;
                all = (uint32_t)next;
            }
    }
    StepArgs A{};
    A.N = N;
    A.L = L;
    A.P = P;
    A.dt = (float)p.dt;
    A.A0 = p.area0();
    A.vol_scale = p.volume();
    A.bulk_scale = (float)p.bulk();
    A.rep_coeff = (float)p.interaction();
    A.mot_coeff = (float)p.motility();
    A.polarity_seed = p.polarity_stream();
    A.p_tumble = synth ? 1. : p.p_tumble();
    float* dp;
    CellState* dc;
    uint32_t* ds;
    unsigned long long* dt;
    MomentAccum* dq;
    GPU(cudaMalloc(&dp, d.phi.size() * 4));
    GPU(cudaMalloc(&dc, N * sizeof(CellState)));
    GPU(cudaMalloc(&ds, S.size() * 4));
    GPU(cudaMalloc(&dt, 16));
    GPU(cudaMalloc(&dq, N * sizeof(MomentAccum)));
    unsigned long long step = (unsigned long long)d.step;
    GPU(cudaMemcpy(dp, d.phi.data(), d.phi.size() * 4, cudaMemcpyHostToDevice));
    GPU(cudaMemcpy(dc, cs.data(), N * sizeof(CellState), cudaMemcpyHostToDevice));
    GPU(cudaMemcpy(ds, S.data(), S.size() * 4, cudaMemcpyHostToDevice));
    GPU(cudaMemcpy(dt, &step, 8, cudaMemcpyHostToDevice));
    GPU(cudaMemset(dq, 0, N * sizeof(MomentAccum)));
    A.phi_in = dp;
    A.cell = dc;
    A.S_rd = ds;
    A.step_rd = dt;
    launch_velocity_moments(A, dq, p.dt, 0);
    GPU(cudaDeviceSynchronize());
    std::vector<MomentAccum> q(N);
    GPU(cudaMemcpy(q.data(), dq, N * sizeof(MomentAccum), cudaMemcpyDeviceToHost));
    std::vector<float> phi_after(d.phi.size());
    std::vector<CellState> c_after(N);
    std::vector<uint32_t> s_after(S.size());
    GPU(cudaMemcpy(phi_after.data(), dp, d.phi.size() * 4, cudaMemcpyDeviceToHost));
    GPU(cudaMemcpy(c_after.data(), dc, N * sizeof(CellState), cudaMemcpyDeviceToHost));
    GPU(cudaMemcpy(s_after.data(), ds, S.size() * 4, cudaMemcpyDeviceToHost));
    if (std::memcmp(phi_after.data(), d.phi.data(), d.phi.size() * 4) ||
        std::memcmp(cs.data(), c_after.data(), N * sizeof(CellState)) || S != s_after)
        return 4;
    const auto metrics = check_reference(d, cs, S, A, q, synth);
    const double max_force = metrics.max_force;
    const double max_moment = metrics.max_moment;
    const double max_quotient = metrics.max_quotient;
    const double max_area = metrics.max_area;
    const double max_euler = metrics.max_euler;
    const double max_half = metrics.max_half;
    // Count and accumulation repeat without advancing physics: detects missing
    // channel writes and confirms the observer has no hidden RNG/state changes.
    launch_velocity_moments(A, dq, p.dt, 0);
    GPU(cudaDeviceSynchronize());
    std::vector<MomentAccum> twice(N);
    GPU(cudaMemcpy(twice.data(), dq, N * sizeof(MomentAccum), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        if (twice[i].count != 2)
            return 7;
        for (int j = 0; j < 12; ++j)
            if (twice[i].q[j] != 2 * q[i].q[j])
                return 8;
    }
    bool ok = max_force < 3e-9 && max_moment < 2e-8 && max_area < 2e-8 && max_quotient < 1e-5;
    if (max_euler > 1e-8)
        ok = ok && max_half < .65 * max_euler;
    if (synth)
        ok = check_fused_paths(d, cs, S, q, A, dt, dq) && ok;
    std::printf("{\"status\":\"%s\",\"N\":%d,\"synthetic_all_classes\":%s,\"read_only\":true,"
                "\"repeat_accumulation\":true,\"max_force_error\":%.17g,\"max_component_"
                "error\":%.17g,\"max_quotient_error\":%.17g,\"max_area\":%.17g,\"euler_error\":"
                "%.17g,\"half_dt_error\":%.17g}\n",
                ok ? "PASS" : "FAIL", N, synth ? "true" : "false", max_force, max_moment,
                max_quotient, max_area, max_euler, max_half);
    GPU(cudaFree(dp));
    GPU(cudaFree(dc));
    GPU(cudaFree(ds));
    GPU(cudaFree(dt));
    GPU(cudaFree(dq));
    return ok ? 0 : 9;
}
