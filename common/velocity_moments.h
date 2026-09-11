#pragma once
// Runtime observation state and discrete integration. No checkpoint fields.
// Array offsets are x/y pairs in the documented PFVMOM4 channel order.
#include <cstdint>

#ifdef __CUDACC__
#define PF_VELOCITY_HD __host__ __device__ __forceinline__
#else
#define PF_VELOCITY_HD inline
#endif

namespace pf {
namespace velocity {
constexpr int kActive = 0;
constexpr int kInteraction = 2;
constexpr int kInterfacial = 4;
constexpr int kOverlap = 6;
constexpr int kArea = 8;
constexpr int kAdvectionCorrection = 10;
constexpr int kIntegralValues = 12;
constexpr int kSpatialValues = 4;
constexpr int kRounding = 4; // in AdvectionAccum only

struct AdvectionAccum {
    double q[6]; // exact integrated A, I, and stored-float-velocity rounding
    unsigned long long count;
};
static_assert(sizeof(AdvectionAccum) == 56, "runtime advection layout changed");

struct MomentAccum {
    double q[kIntegralValues];
    unsigned long long count;        // used by the independent dense observer
    double held[kSpatialValues + 2]; // G/K rates, then normalized x/y overlaps
    unsigned long long spatial_count;
    unsigned long long last_sample;
    unsigned long long last_interval;
    double last_totals[4]; // A+I x/y, then rounding x/y at last spatial sample
    double correction[kSpatialValues];
};

PF_VELOCITY_HD void advection_totals(const AdvectionAccum& a, double (&v)[4]) {
    v[0] = a.q[kActive] + a.q[kInteraction];
    v[1] = a.q[kActive + 1] + a.q[kInteraction + 1];
    v[2] = a.q[kRounding];
    v[3] = a.q[kRounding + 1];
}

// The geometry B is held between samples; the actual velocity (including
// tumbles and rounding) is integrated every solver step:
// L = (B-1)*delta(Q_A+Q_I) + B*delta(Q_rounding).
PF_VELOCITY_HD void add_advection_correction(MomentAccum& m, const double (&totals)[4]) {
    for (int xy = 0; xy < 2; ++xy) {
        const double beta = m.held[kSpatialValues + xy];
        m.q[kAdvectionCorrection + xy] += (beta - 1.0) * (totals[xy] - m.last_totals[xy]) +
                                          beta * (totals[2 + xy] - m.last_totals[2 + xy]);
    }
}

// Close the preceding sample's left-held interval before replacing its rates.
// Counts are relative to this recorder invocation, not the checkpoint step.
PF_VELOCITY_HD void begin_spatial_sample(MomentAccum& m, const AdvectionAccum& a, double dt) {
    double totals[4];
    advection_totals(a, totals);
    const auto interval = a.count - m.last_sample;
    if (m.spatial_count) {
        for (int j = 0; j < kSpatialValues; ++j)
            m.q[kInterfacial + j] += dt * double(interval) * m.held[j];
        add_advection_correction(m, totals);
    }
    for (int j = 0; j < 4; ++j)
        m.last_totals[j] = totals[j];
    m.last_interval = interval;
    m.last_sample = a.count;
    ++m.spatial_count;
}

// Target the discrete PRE-step sum, not a continuous trapezoidal integral.
// For S steps, linear interpolation contributes dt*(S-1)/2*(g_end-g_start).
PF_VELOCITY_HD double linear_correction(double dt, unsigned long long steps, double first,
                                        double last) {
    return steps ? dt * 0.5 * double(steps - 1) * (last - first) : 0.0;
}

// Host-side readback only: finish a copy, leaving device accumulators intact.
// endpoint contains dt times the instantaneous independently measured rate.
PF_VELOCITY_HD void finish_snapshot(MomentAccum& m, const AdvectionAccum& a,
                                    const MomentAccum& endpoint, double dt) {
    m.count = a.count;
    for (int j = 0; j < kInterfacial; ++j)
        m.q[j] = a.q[j];
    if (!m.spatial_count)
        return;
    const auto tail = a.count - m.last_sample;
    for (int j = 0; j < kSpatialValues; ++j) {
        m.q[kInterfacial + j] += m.correction[j] + dt * double(tail) * m.held[j];
        m.q[kInterfacial + j] +=
            linear_correction(dt, tail, m.held[j], endpoint.q[kInterfacial + j] / dt);
    }
    double totals[4];
    advection_totals(a, totals);
    add_advection_correction(m, totals);
}
} // namespace velocity
} // namespace pf
#undef PF_VELOCITY_HD
