#include "velocity_moments.h"
#include "velocity_format.h"
#include "velocity_sampling.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {
void require(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}
bool near(double a, double b) {
    return std::fabs(a - b) <= 2e-11 * (1.0 + std::fabs(b));
}
double rate(int channel, long long step) {
    return (channel + 1) * (0.25 + 0.03125 * double(step));
}
void check_interval(int stride, long long start, int dense, int steps) {
    using namespace pf::velocity;
    pf::VelocitySampling policy;
    policy.configure(true, false, stride);
    policy.start_run(start, dense);
    MomentAccum m{}, endpoint{};
    AdvectionAccum a{};
    double expected[kIntegralValues]{};
    constexpr double dt = 0.01;
    // Constant geometry but strongly changing velocity: rapid direction changes
    // must remain exact even when spatial observations are sparse.
    const double beta[2] = {0.875, 0.75};
    unsigned long long samples = 0;
    for (int n = 0; n < steps; ++n) {
        if (policy.spatial_sample(int((start + n) % policy.graph_period()), start + n, false)) {
            begin_spatial_sample(m, a, dt);
            for (int j = 0; j < kSpatialValues; ++j) {
                const double next = rate(j, n);
                if (m.spatial_count > 1)
                    m.correction[j] += linear_correction(dt, m.last_interval, m.held[j], next);
                m.held[j] = next;
            }
            m.held[4] = beta[0];
            m.held[5] = beta[1];
            ++samples;
        }
        for (int xy = 0; xy < 2; ++xy) {
            const double active = ((n / 7) % 2 ? -1.0 : 1.0) * (xy + 1);
            const double interaction = double(n % 5) / 8.0;
            const double rounding = double(n % 3 - 1) * 1e-7;
            a.q[kActive + xy] += dt * active;
            a.q[kInteraction + xy] += dt * interaction;
            a.q[kRounding + xy] += dt * rounding;
            expected[kActive + xy] += dt * active;
            expected[kInteraction + xy] += dt * interaction;
            expected[kAdvectionCorrection + xy] +=
                dt * ((beta[xy] - 1.0) * (active + interaction) + beta[xy] * rounding);
        }
        for (int j = 0; j < kSpatialValues; ++j)
            expected[kInterfacial + j] += dt * rate(j, n);
        ++a.count;
        policy.advanced();
    }
    for (int j = 0; j < kSpatialValues; ++j)
        endpoint.q[kInterfacial + j] = dt * rate(j, steps);
    const MomentAccum original = m;
    MomentAccum snapshot = m;
    finish_snapshot(snapshot, a, endpoint, dt);
    require(std::memcmp(&m, &original, sizeof(m)) == 0, "readback changed accumulator");
    require(snapshot.count == static_cast<unsigned>(steps), "step count mismatch");
    require(snapshot.spatial_count == samples, "sample count mismatch");
    for (int j = 0; j < kIntegralValues; ++j)
        require(near(snapshot.q[j], expected[j]), "discrete integral mismatch");
}

void check_graph_phases(int stride) {
    pf::VelocitySampling policy;
    policy.configure(true, false, stride);
    require(policy.graph_count() <= 6, "unbounded graph phase count");
    for (int start = 0; start < 2 * policy.graph_period(); start += policy.graph_steps()) {
        const int phase = policy.graph_index(start);
        require(phase >= 0 && phase < policy.graph_count(), "invalid graph phase");
        for (int offset = 0; offset < policy.graph_steps(); ++offset) {
            const int captured = phase * policy.graph_steps() + offset;
            const int absolute = start + offset;
            require(captured % 2 == absolute % 2, "graph selected wrong phi buffers");
            require(captured % 3 == absolute % 3, "graph selected wrong shared-field buffers");
            require(captured % 6 == absolute % 6, "graph shifted Morton cadence");
            require(policy.spatial_sample(captured, 0, true) == (absolute % stride == 0),
                    "graph shifted spatial observations");
        }
    }
}
} // namespace

int main() {
    for (int stride = 1; stride <= 1000; ++stride) check_graph_phases(stride);
    for (int stride : {1, 3, 6, 25, 100, 997})
        for (long long start : {0LL, 37LL, 9000037LL})
            for (int dense : {0, 17, 100})
                for (int steps : {0, 1, 2, 37, 100, 107, 603})
                    check_interval(stride, start, dense, steps);
    pf::VelocitySampling s;
    s.configure(true, false, 100);
    s.start_run(37, 17);
    require(s.graph_period() == 300 && s.graph_steps() == 100 && s.graph_count() == 3,
            "graphs do not cover the periodic slots at spatial cadence");
    require(!s.can_replay(600), "first sample must prime an unaligned invocation");
    // Capture ignores both dense startup and the not-yet-primed flag.
    require(!s.spatial_sample(1, 37, true), "startup leaked into captured graph");
    require(s.spatial_sample(0, 37, true), "captured spatial slot is missing");
    s.advanced();
    require(!s.can_replay(53) && s.can_replay(54), "dense transition is off by one");
    s.configure(false, false, 100);
    require(s.graph_steps() == 6 && s.graph_count() == 1 && s.can_replay(0),
            "disabled policy altered physics graph");
    require(sizeof(pf::velocity::FileHeader) == 56, "wire header size changed");
    std::puts("PASS: 378 integration cases, exact changing advection, startup, 1000 graph strides.");
}
