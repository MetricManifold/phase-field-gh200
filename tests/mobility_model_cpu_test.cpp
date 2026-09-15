// Host tests for the cell-specific Allen-Cahn phase-field mobility.
// phase_field_rhs is the exact per-pixel expression both update kernels
// execute, so these checks pin the shipped arithmetic:
//   1. the default path (M_i = M0) reproduces the pre-mobility expression
//      bit for bit;
//   2. M_i/M0 scales every passive variational term and never advection;
//   3. two cells advance with their own multipliers in a two-cell field step;
//   4. (gamma=1, M=0.5) and (gamma=0.35, M=0.5/0.35) give the same
//      interfacial increment to floating-point accuracy;
//   5. the explicit-Euler guard uses max_i[(M_i/M0)*gamma_i].
#include "kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace pf;

static_assert(sizeof(CellState) == 192, "CellState grew");
static_assert(alignof(CellState) == 64, "CellState alignment changed");
static_assert(offsetof(CellState, M_pf) == 188,
              "M_pf must sit in the final reserved word");
static_assert(offsetof(CellState, reserved) == 120,
              "reserved diagnostics moved");

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "[FAIL] %s\n", what);
        ++failures;
    }
}

bool same_bits(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
}

// The pre-mobility per-pixel expression, copied verbatim from the update
// kernels as they stood before this change.
float legacy_rhs(float lap, float phi, float So, float gx, float gy,
                 float gam, float dwC, float volC, float repC,
                 float vx, float vy) {
    return gam * lap
         - dwC * (phi * (1.0f - phi) * (1.0f - 2.0f * phi))
         + volC * phi
         - repC * phi * So
         - (vx * gx + vy * gy);
}

struct Px {
    float lap, phi, So, gx, gy;
};

const Px kPixels[] = {
    {0.125f, 0.5f, 0.25f, 0.03f, -0.02f},
    {-0.75f, 0.9999f, 0.0f, -0.4f, 0.4f},
    {2.5e-3f, 1.0e-5f, 0.9f, 1.0e-4f, -1.0e-4f},
    {-3.1f, 0.31f, 1.7f, 0.11f, 0.07f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {-1.9e-4f, 0.62f, 3.9f, -0.9f, -0.8f},
};
const float kCoeffSets[][6] = {
    // gam, dwC, volC, repC, vx, vy
    {1.0f, 0.612244879f, 1.3e-4f, 12.2448978f, 0.01f, -0.02f},
    {0.35f, 0.214285716f, -2.0e-4f, 12.2448978f, -0.004f, 0.006f},
    {2.0f, 1.5f, 0.5f, 0.0f, 0.0f, 0.3f},
};

void test_default_path_bitwise() {
    for (const Px& q : kPixels)
        for (const auto& c : kCoeffSets) {
            const float a = phase_field_rhs(q.lap, q.phi, q.So, q.gx, q.gy,
                                            c[0], c[1], c[2], c[3], c[4], c[5],
                                            1.0f);
            const float b = legacy_rhs(q.lap, q.phi, q.So, q.gx, q.gy,
                                       c[0], c[1], c[2], c[3], c[4], c[5]);
            expect(same_bits(a, b),
                   "default path must equal the legacy expression bitwise");
        }
}

void test_passive_scaling_advection_unchanged() {
    const float mrels[] = {0.5f, 1.0f / 0.35f, 2.857143f, 4.0f};
    for (const Px& q : kPixels)
        for (const auto& c : kCoeffSets)
            for (const float mrel : mrels) {
                // Passive-only value at mrel = 1 is the legacy expression.
                const float p1 = phase_field_rhs(q.lap, q.phi, q.So, q.gx,
                                                 q.gy, c[0], c[1], c[2], c[3],
                                                 0.0f, 0.0f, 1.0f);
                // The scaled path applies mrel as one multiplication of the
                // complete passive sum.
                const float pm = phase_field_rhs(q.lap, q.phi, q.So, q.gx,
                                                 q.gy, c[0], c[1], c[2], c[3],
                                                 0.0f, 0.0f, mrel);
                expect(same_bits(pm, mrel * p1),
                       "passive terms must scale by exactly mrel");
                // Advection is subtracted outside the mobility factor.
                const float adv = c[4] * q.gx + c[5] * q.gy;
                const float full = phase_field_rhs(q.lap, q.phi, q.So, q.gx,
                                                   q.gy, c[0], c[1], c[2],
                                                   c[3], c[4], c[5], mrel);
                expect(same_bits(full, pm - adv),
                       "advection must stay outside the mobility factor");
            }

    // With every passive coefficient zero the result is pure advection and
    // must be independent of mrel, bit for bit.
    for (const Px& q : kPixels) {
        const float a1 = phase_field_rhs(q.lap, q.phi, q.So, q.gx, q.gy,
                                         0.0f, 0.0f, 0.0f, 0.0f,
                                         0.013f, -0.021f, 1.0f);
        const float a2 = phase_field_rhs(q.lap, q.phi, q.So, q.gx, q.gy,
                                         0.0f, 0.0f, 0.0f, 0.0f,
                                         0.013f, -0.021f, 1.0f / 0.35f);
        expect(same_bits(a1, a2), "advection must not depend on M_i");
    }

    // Each passive term in isolation carries the multiplier.
    const float mrel = 1.0f / 0.35f;
    const Px q = kPixels[3];
    expect(same_bits(phase_field_rhs(q.lap, q.phi, 0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, mrel),
                     mrel * (1.0f * q.lap)),
           "interfacial term must scale by mrel");
    expect(same_bits(phase_field_rhs(0.0f, q.phi, 0.0f, 0.0f, 0.0f,
                                     0.0f, 0.61f, 0.0f, 0.0f, 0.0f, 0.0f,
                                     mrel),
                     mrel * (-0.61f * (q.phi * (1.0f - q.phi)
                                       * (1.0f - 2.0f * q.phi)))),
           "double-well term must scale by mrel");
    expect(same_bits(phase_field_rhs(0.0f, q.phi, 0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 2.5e-4f, 0.0f, 0.0f, 0.0f,
                                     mrel),
                     mrel * (2.5e-4f * q.phi)),
           "area-constraint term must scale by mrel");
    expect(same_bits(phase_field_rhs(0.0f, q.phi, q.So, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 12.2448978f, 0.0f,
                                     0.0f, mrel),
                     mrel * (-12.2448978f * q.phi * q.So)),
           "overlap-repulsion term must scale by mrel");
}

// Host mirror of the kernels' aggregate-field subtraction.
float s_other_host(uint32_t qS, float phi_self) {
    const uint32_t qc = q_of(phi_self);
    return qS >= qc ? (float)(qS - qc) * kQInvF : 0.0f;
}

// One explicit-Euler step of one cell on an L x L periodic lattice using the
// kernel stencil, gradients, quantized S, and phase_field_rhs.
void step_field(const std::vector<float>& phi, const std::vector<uint32_t>& S,
                int L, float gam, float dwC, float volC, float repC,
                float vx, float vy, float mrel, float dt,
                std::vector<float>* out) {
    out->assign((size_t)L * L, 0.0f);
    auto at = [&](int x, int y) {
        return phi[(size_t)((y + L) % L) * L + (size_t)((x + L) % L)];
    };
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
            const float c = at(x, y);
            const float lap = ((float)kLapEdgeW * (at(x, y + 1) + at(x, y - 1)
                                                   + at(x + 1, y) + at(x - 1, y))
                             + (float)kLapDiagW * (at(x + 1, y + 1) + at(x - 1, y + 1)
                                                   + at(x + 1, y - 1) + at(x - 1, y - 1))
                             + (float)kLapCentreW * c)
                            * (float)(1.0 / kLapDenom);
            const float gx = 0.5f * (at(x + 1, y) - at(x - 1, y));
            const float gy = 0.5f * (at(x, y + 1) - at(x, y - 1));
            const float So = s_other_host(S[(size_t)y * L + x], c);
            const float rhs = phase_field_rhs(lap, c, So, gx, gy, gam, dwC,
                                              volC, repC, vx, vy, mrel);
            (*out)[(size_t)y * L + x] = c + dt * rhs;
        }
}

std::vector<float> tanh_disc(int L, float cx, float cy, float R, float lam) {
    std::vector<float> phi((size_t)L * L, 0.0f);
    const float k = (float)interface_k((double)lam);
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
            const float dx = (float)x - cx, dy = (float)y - cy;
            const float r = std::sqrt(dx * dx + dy * dy);
            phi[(size_t)y * L + x] =
                0.5f * (1.0f - std::tanh(k * (r - R)));
        }
    return phi;
}

void test_two_cell_multipliers() {
    const int L = 64;
    const float dt = 0.01f;
    const float bulk = (float)bulk_coeff(7.0);
    const float rep = (float)interaction_coeff(10.0, 7.0);
    const std::vector<float> pa = tanh_disc(L, 24.0f, 32.0f, 10.0f, 7.0f);
    const std::vector<float> pb = tanh_disc(L, 42.0f, 32.0f, 10.0f, 7.0f);

    // Aggregate field S = sum_m q(phi_m), as the scatter kernels build it.
    std::vector<uint32_t> S((size_t)L * L, 0u);
    for (size_t i = 0; i < S.size(); ++i) S[i] = q_of(pa[i]) + q_of(pb[i]);

    // Cell a is a normal cell (M = M0); cell b is the soft-cell control
    // M = 0.5/0.35 with gamma = 0.35, exactly as the run pipeline derives it:
    // M_pf stored as binary32, mrel = M_pf * (1/M0).
    const float gam_a = 1.0f, mrel_a = 0.5f * 2.0f;
    const float gam_b = 0.35f;
    const float M_b = (float)(kPhaseFieldM0 / 0.35);
    const float mrel_b = M_b * (float)(1.0 / kPhaseFieldM0);
    const float volC = 1.0e-4f, vx = 0.01f, vy = -0.005f;

    std::vector<float> na, nb;
    step_field(pa, S, L, gam_a, bulk * gam_a, volC, rep, vx, vy, mrel_a, dt,
               &na);
    step_field(pb, S, L, gam_b, bulk * gam_b, volC, rep, vx, vy, mrel_b, dt,
               &nb);

    // Independent expectations: the legacy expression for cell a and the
    // explicitly scaled expression for cell b.
    int checked_a = 0, checked_b = 0;
    for (int y = 1; y < L - 1; ++y)
        for (int x = 1; x < L - 1; ++x) {
            const size_t i = (size_t)y * L + x;
            auto stencil = [&](const std::vector<float>& p, float* lap,
                              float* gx, float* gy) {
                const float c = p[i];
                *lap = ((float)kLapEdgeW * (p[i + L] + p[i - L] + p[i + 1]
                                            + p[i - 1])
                      + (float)kLapDiagW * (p[i + L + 1] + p[i + L - 1]
                                            + p[i - L + 1] + p[i - L - 1])
                      + (float)kLapCentreW * c) * (float)(1.0 / kLapDenom);
                *gx = 0.5f * (p[i + 1] - p[i - 1]);
                *gy = 0.5f * (p[i + L] - p[i - L]);
            };
            float lap, gx, gy;
            stencil(pa, &lap, &gx, &gy);
            const float ea = pa[i] + dt * legacy_rhs(
                lap, pa[i], s_other_host(S[i], pa[i]), gx, gy, gam_a,
                bulk * gam_a, volC, rep, vx, vy);
            if (!same_bits(na[i], ea) && ++checked_a == 1)
                expect(false, "cell a must advance with the legacy default "
                              "path");
            stencil(pb, &lap, &gx, &gy);
            const float eb = pb[i] + dt * (mrel_b * (
                gam_b * lap
                - (bulk * gam_b) * (pb[i] * (1.0f - pb[i])
                                    * (1.0f - 2.0f * pb[i]))
                + volC * pb[i]
                - rep * pb[i] * s_other_host(S[i], pb[i]))
                - (vx * gx + vy * gy));
            if (!same_bits(nb[i], eb) && ++checked_b == 1)
                expect(false, "cell b must advance with its own M_i/M0");
        }
    expect(checked_a == 0 && checked_b == 0,
           "two-cell step applied a wrong per-cell multiplier somewhere");

    // A distinct multiplier must actually change cell b's passive update.
    std::vector<float> nb_default;
    step_field(pb, S, L, gam_b, bulk * gam_b, volC, rep, vx, vy, 1.0f, dt,
               &nb_default);
    expect(nb != nb_default,
           "mrel != 1 must change the soft cell's one-step result");
}

void test_gamma_mobility_equivalence() {
    // Area constraint, overlap, and advection disabled: the pairs
    // (gamma=1, M=0.5) and (gamma=0.35, M=0.5/0.35) must produce the same
    // interfacial + double-well increment up to float rounding.
    const int L = 64;
    const float dt = 0.01f;
    const float bulk = (float)bulk_coeff(7.0);
    const std::vector<float> phi = tanh_disc(L, 32.0f, 32.0f, 12.0f, 7.0f);
    const std::vector<uint32_t> S0((size_t)L * L, 0u);

    std::vector<float> ref, soft;
    step_field(phi, S0, L, 1.0f, bulk * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
               1.0f, dt, &ref);
    const float M_s = (float)(kPhaseFieldM0 / 0.35);
    const float mrel_s = M_s * (float)(1.0 / kPhaseFieldM0);
    step_field(phi, S0, L, 0.35f, bulk * 0.35f, 0.0f, 0.0f, 0.0f, 0.0f,
               mrel_s, dt, &soft);

    double max_rel = 0.0, max_abs = 0.0, max_inc = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double da = (double)ref[i] - (double)phi[i];
        const double db = (double)soft[i] - (double)phi[i];
        max_inc = std::fmax(max_inc, std::fabs(da));
        const double diff = std::fabs(da - db);
        max_abs = std::fmax(max_abs, diff);
        if (std::fabs(da) > 1.0e-12)
            max_rel = std::fmax(max_rel, diff / std::fabs(da));
    }
    std::printf("gamma_mobility_equivalence: max increment %.3e, max abs "
                "diff %.3e, max rel diff %.3e\n", max_inc, max_abs, max_rel);
    expect(max_inc > 1.0e-6, "the interfacial increment must be nonzero");
    // (1/0.35)*(0.35*x) differs from x by a few ulp; equality is to float
    // rounding, not bitwise.
    expect(max_rel < 1.0e-5,
           "gamma*M compensation must reproduce the interfacial increment");
}

void test_effective_stiffness_guard() {
    SimParams p;
    p.dt = 0.01;
    const float g[2] = {1.0f, 0.35f};
    const float m[2] = {0.5f, (float)(kPhaseFieldM0 / 0.35)};
    expect(validate_effective_stiffness(p, g, m, 2),
           "the matched (gamma, M) control must pass at dt = 0.01");

    // max_i[(M_i/M0)*gamma_i] = 3 makes dt = 0.06 exceed the safety limit
    // even though dt*gamma alone would pass.
    SimParams p2;
    p2.dt = 0.06;
    const float g2[2] = {1.0f, 1.0f};
    const float m2[2] = {0.5f, 1.5f};
    std::fprintf(stderr, "(expected fatal message follows)\n");
    expect(!validate_effective_stiffness(p2, g2, m2, 2),
           "the guard must use max_i[(M_i/M0)*gamma_i]");
    expect(validate_effective_stiffness(p2, g2, m2, 1),
           "dt = 0.06 with every M_i = M0 must still pass");

    const float bad_m[1] = {0.0f};
    const float g1[1] = {1.0f};
    std::fprintf(stderr, "(expected fatal message follows)\n");
    expect(!validate_effective_stiffness(p, g1, bad_m, 1),
           "non-positive per-cell mobility must be fatal");
}

void test_validate_map_downgrade() {
    // dt*(uniform M/M0)*gamma exceeds the limit, but a supplied map may
    // lower every resolved per-cell pair below it. With a map the uniform
    // interfacial fatal becomes a warning and the binding, still-fatal check
    // is validate_effective_stiffness on the resolved values.
    SimParams p;
    p.Nx = p.Ny = 400;
    p.dt = 0.04;
    p.gamma_normal = 1.0;
    p.phase_field_mobility = 2.0;   // (M/M0)*gamma = 4 -> cfl ~ 1.07
    std::fprintf(stderr, "(expected fatal message follows)\n");
    expect(!validate(p, false),
           "without a map the uniform interfacial check must stay fatal");
    std::fprintf(stderr, "(expected warning follows)\n");
    expect(validate(p, true),
           "with a map the uniform interfacial check must downgrade to a "
           "warning");
    // The resolved pairs remain fatally guarded either way.
    const float g[1] = {1.0f};
    const float m_ok[1] = {0.5f};
    const float m_bad[1] = {2.0f};
    expect(validate_effective_stiffness(p, g, m_ok, 1),
           "an all-0.5 map resolves below the limit at dt = 0.04");
    std::fprintf(stderr, "(expected fatal message follows)\n");
    expect(!validate_effective_stiffness(p, g, m_bad, 1),
           "resolved per-cell pairs above the limit must remain fatal");
}

}  // namespace

int main() {
    test_default_path_bitwise();
    test_passive_scaling_advection_unchanged();
    test_two_cell_multipliers();
    test_gamma_mobility_equivalence();
    test_effective_stiffness_guard();
    test_validate_map_downgrade();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("MOBILITY_MODEL_CPU_TEST_PASS");
    return 0;
}
