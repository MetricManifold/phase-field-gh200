#pragma once
// Physical parameters, discretization constants, and GPU tile geometry shared
// by host checks and device updates.

#include <cuda_runtime.h>

#include "model_coefficients.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace pf {

constexpr double kPi = 3.14159265358979323846;

// Physics: Palmieri et al. 2015, Sci Rep 5:11745, Eq. (S15). Mobility M = 1/2.
// The coefficient helpers include the mobility factor and return the values
// used directly in dphi/dt:
//
//   dphi_n/dt = gamma_n * lap(phi_n)
//             - bulk_coeff(lambda) * gamma_n * phi(1-phi)(1-2phi)
//             + volume_coeff(mu, A0) * (A0 - V_n) * phi
//             - interaction_coeff(kappa, lambda) * phi_n * sum_{m!=n} phi_m^2
//             - (vx * dphi/dx + vy * dphi/dy)
//
//   v_n = v_A * p_hat_n
//       + motility_coeff(kappa, xi, lambda)
//         * integral( phi_n * grad(phi_n) * sum_{m!=n} phi_m^2 dA )
//
// The interaction energy sums ordered pairs, giving 60*kappa/lambda^2 after
// applying M = 1/2. Both compile-time and runtime checks enforce
// interaction_coeff/motility_coeff = xi.
using pf_common::bulk_coeff;
using pf_common::interaction_coeff;
using pf_common::kNumerBulk;
using pf_common::kNumerInteraction;
using pf_common::kNumerVolume;
using pf_common::motility_coeff;
using pf_common::volume_coeff;

template <typename T>
__host__ __device__ constexpr T target_area(T radius) {
    return T(kPi) * radius * radius;
}

// The stationary planar interface is phi(d) = [1-tanh(k d)]/2 with
// k = sqrt(7.5)/lambda; its phi and phi^2 tails decay as exp(-2kd) and
// exp(-4kd), respectively.
using pf_common::interface_k;

// Shift the seeded interface radius by 1/(2k), the leading-order correction
// that makes integral(phi^2 dA) equal to the target area pi*R^2.
using pf_common::init_radius;

// Compile-time checks of interaction_coeff/motility_coeff = xi.
namespace detail {
constexpr bool ratio_is_xi(double kappa, double xi, double lambda) {
    const double a = interaction_coeff(kappa, lambda);
    const double b = motility_coeff(kappa, xi, lambda);
    const double r = a / b;
    return r > xi * (1.0 - 1e-12) && r < xi * (1.0 + 1e-12);
}
static_assert(ratio_is_xi(10.0, 1500.0, 7.0),
              "interaction_coeff/motility_coeff must equal xi (reference set)");
static_assert(ratio_is_xi(1.0, 1.0, 1.0), "invariant broken at unit scale");
static_assert(ratio_is_xi(3.7, 977.0, 5.25), "invariant broken off-lattice");
static_assert(ratio_is_xi(0.125, 65536.0, 0.5), "invariant broken at extremes");
}  // namespace detail

// Numerics. h = dx = dy = 1 throughout; validate() refuses to run otherwise.
// 9-point isotropic McLellan Laplacian:
//   lap = (4*(N+S+E+W) + (NE+NW+SE+SW) - 20*c) / (6 h^2)
// The weights sum to zero and give lap(x^2)=2. The exact spectral radius is
// 16/3; kLapRowSumBound is the conservative absolute row-sum bound 40/6.
constexpr int    kLapEdgeW   = 4;
constexpr int    kLapDiagW   = 1;
constexpr int    kLapCentreW = -20;
constexpr double kLapDenom   = 6.0;

static_assert(4 * kLapEdgeW + 4 * kLapDiagW + kLapCentreW == 0,
              "9-point Laplacian weights must sum to zero");
static_assert(kLapEdgeW * (0 + 0 + 1 + 1) + kLapDiagW * (1 + 1 + 1 + 1)
                  + kLapCentreW * 0 == 12,
              "9-point Laplacian must give lap(x^2) == 12/6 == 2 exactly at h = 1");
static_assert(kLapDenom == 6.0, "lap(x^2) == 2 requires the denominator 6");

constexpr double kLapRowSumBound = (4.0 * kLapEdgeW + 4.0 * kLapDiagW
                                    - kLapCentreW) / kLapDenom;   // 40/6

// Q5.27 unsigned fixed point for the shared field S = sum_m phi_m^2.
//   q = round(phi^2 * 2^27), value = q * 2^-27, range [0, 32), quantum 7.45e-9.
// Each contribution is limited to phi^2 <= 4; atomic overflow is detected and
// recorded as a fatal flag.
constexpr float  kQScaleF     = 134217728.0f;                 // 2^27
constexpr float  kQInvF       = 7.45058059692382812e-09f;     // 2^-27, exact
constexpr double kQInvD       = 7.45058059692382812e-09;
constexpr float  kQClampPhiSq = 4.0f;

__host__ __device__ __forceinline__ uint32_t q_of(float phi) {
    float s = phi * phi;                 // one multiply: no FMA contraction
    s = fminf(s, kQClampPhiSq);
#if defined(__CUDA_ARCH__)
    return __float2uint_rn(s * kQScaleF);
#else
    return (uint32_t)std::lrint(s * kQScaleF);  // round-to-nearest-even, as _rn
#endif
}

// Support threshold for the bounding box that drives the shape-class decision.
constexpr float kSupportEps = 1e-5f;

// Each cell owns a fixed square phase-field tile and an active rectangular
// window selected from kClasses. Values outside the active window are exactly
// zero, which permits synthesis of the one-pixel stencil halo. Shared-memory
// classes cover the common geometries; the last class uses the tile interior
// directly from global memory.
constexpr int kTilePitch = 288;
constexpr int kLargeClassEdge = 224;
constexpr int kTileArea  = kTilePitch * kTilePitch;
static_assert(kTilePitch == 288 && kLargeClassEdge == 224,
              "the fixed support layout changed");
static_assert(kTilePitch % 32 == 0,
              "tile rows must remain 128-byte aligned in float storage");

struct ShapeClass {
    int wx, wy, tx0, ty0;
};

// Classes 1 and 2 accommodate one-axis elongation; class 3 accommodates
// moderate growth on both axes. Class 4 stages phi but reads S globally. The
// fallback stages neither field and leaves a one-pixel stencil ring in the
// fixed tile.
constexpr int kNumClasses = 6;
constexpr ShapeClass kClasses[kNumClasses] = {
    {144, 144, 64, 64},   // 0: round
    {176, 144, 32, 64},   // 1: wide
    {144, 176, 64, 32},   // 2: tall
    {160, 160, 32, 32},   // 3: larger square
    {kLargeClassEdge, kLargeClassEdge, 32, 32},
                           // 4: large (phi in shared memory; S global)
    {kTilePitch - 2, kTilePitch - 2, 1, 1},
                           // 5: fallback (phi and S in global memory)
};

constexpr int kClassRound = 0;
constexpr int kClassWide  = 1;
constexpr int kClassTall  = 2;
constexpr int kClassBig   = 3;
constexpr int kClassLarge = 4;
constexpr int kClassFallback = 5;
static_assert(kNumClasses == 6, "class_of() below enumerates exactly 6 classes");
// k_step explicitly dispatches every valid class and rejects other ids.
static_assert(kClassFallback == kNumClasses - 1,
              "the global fallback must remain last in the class table");

// Pin the shared-class geometry independently of the fallback.
static_assert(kClasses[0].wx == 144 && kClasses[0].wy == 144 &&
              kClasses[0].tx0 == 64 && kClasses[0].ty0 == 64,
              "class 0 geometry changed");
static_assert(kClasses[1].wx == 176 && kClasses[1].wy == 144 &&
              kClasses[1].tx0 == 32 && kClasses[1].ty0 == 64,
              "class 1 geometry changed");
static_assert(kClasses[2].wx == 144 && kClasses[2].wy == 176 &&
              kClasses[2].tx0 == 64 && kClasses[2].ty0 == 32,
              "class 2 geometry changed");
static_assert(kClasses[3].wx == 160 && kClasses[3].wy == 160 &&
              kClasses[3].tx0 == 32 && kClasses[3].ty0 == 32,
              "class 3 geometry changed");
static_assert(kClasses[kClassLarge].wx == kLargeClassEdge &&
              kClasses[kClassLarge].wy == kLargeClassEdge &&
              kClasses[kClassLarge].tx0 == 32 &&
              kClasses[kClassLarge].ty0 == 32,
              "largest shared-phi class must retain its aligned origin");
static_assert(kClasses[kClassFallback].wx == 286 &&
              kClasses[kClassFallback].wy == 286 &&
              kClasses[kClassFallback].tx0 == 1 &&
              kClasses[kClassFallback].ty0 == 1,
              "fallback must be the guarded 286x286 tile interior");

// Template specialization removes shared-field choices from the runtime path.
constexpr bool class_stages_S(int c) { return c >= 0 && c <= kClassBig; }
constexpr bool class_stages_phi(int c) { return c != kClassFallback; }
template <int CLS>
inline constexpr bool kStagesS = class_stages_S(CLS);

// Literal selections allow runtime device lookup while keeping class geometry
// available to the compiler as immediate constants.
__host__ __device__ __forceinline__ constexpr ShapeClass class_of(int c) {
    if (c == kClassFallback) return kClasses[kClassFallback];
    return ShapeClass{
        c == 1 ? kClasses[1].wx  : c == 2 ? kClasses[2].wx
               : c == 3 ? kClasses[3].wx  : c == 4 ? kClasses[4].wx
               : kClasses[0].wx,
        c == 1 ? kClasses[1].wy  : c == 2 ? kClasses[2].wy
               : c == 3 ? kClasses[3].wy  : c == 4 ? kClasses[4].wy
               : kClasses[0].wy,
        c == 1 ? kClasses[1].tx0 : c == 2 ? kClasses[2].tx0
               : c == 3 ? kClasses[3].tx0 : c == 4 ? kClasses[4].tx0
               : kClasses[0].tx0,
        c == 1 ? kClasses[1].ty0 : c == 2 ? kClasses[2].ty0
               : c == 3 ? kClasses[3].ty0 : c == 4 ? kClasses[4].ty0
               : kClasses[0].ty0};
}

__host__ __device__ __forceinline__ constexpr bool class_contains_support(
    int c, int ex, int ey, int slack) {
    const ShapeClass sc = class_of(c);
    return ex + slack <= sc.wx && ey + slack <= sc.wy;
}

__host__ __device__ __forceinline__ constexpr int class_support_capacity(
    int c, int slack) {
    return class_of(c).wx - slack;
}

// Minimum-area class containing both support extents plus slack; -1 if none
// fits.
__host__ __device__ __forceinline__ constexpr int class_containing(int ex, int ey,
                                                                   int slack) {
    int best = -1, best_area = 0;
    for (int c = 0; c < kNumClasses; ++c) {
        const ShapeClass sc = class_of(c);
        const int area = sc.wx * sc.wy;
        if (class_contains_support(c, ex, ey, slack) &&
            (best < 0 || area < best_area)) {
            best = c;
            best_area = area;
        }
    }
    return best;
}

// Asymmetric class hysteresis: promote promptly and demote only after a dwell.
// Slack is measured against the candidate window on both axes.
constexpr int kPromoteSlack   = 8;     // leave a class when extent + 8 > its W
constexpr int kDemoteSlack    = 20;    // enter a smaller class with this margin
constexpr int kDemoteDwell    = 200;   // consecutive checks before demotion
constexpr int kMaxShiftPerStep = 4;    // maximum recentring shift per axis

// Block shape and shared-memory layout.
// One 768-thread block processes one cell. Dynamic shared memory contains:
//   [0)                 red_s   : double[24][8]   fp64 reduction slots
//   [kRedBytes)         bcast_s : 128 words       CTA-wide scalar broadcast
//   [+kBcastBytes)      128 B reserved
//   [kScalarBytes)      phi_s   : float (WY+2) rows x phi_pitch(WX)
//   [+phi bytes)        S_s     : uint32 WY rows x WX (staged classes only)
// Four floats of left padding align each phi row for 16-byte cp.async copies;
// the extra rows and columns hold the synthesized stencil halo.
constexpr int kBlockThreads  = 768;
constexpr int kWarpsPerBlock = kBlockThreads / 32;    // 24
constexpr int kStripRows     = 16;
constexpr int kPipeStages    = 3;
constexpr int kPhiPadLeft    = 4;

// Place loop-control words at the end of the broadcast region. With no static
// allocation, the dynamic region begins at the aligned shared-memory base.
constexpr int kBcastCtrlN   = 124;
constexpr int kBcastCtrlCls = 125;

constexpr int kRedSlots   = 8;
constexpr int kRedBytes   = kWarpsPerBlock * kRedSlots * (int)sizeof(double);  // 1536
constexpr int kBcastWords = 128;
constexpr int kBcastBytes = kBcastWords * 4;                                   // 512
constexpr int kMbarBytes  = 128;                                               // reserved
constexpr int kScalarBytes = kRedBytes + kBcastBytes + kMbarBytes;             // 2176

static_assert(kScalarBytes % 16 == 0, "phi_s must land 16 B aligned");

constexpr int align_up(int v, int a) { return ((v + a - 1) / a) * a; }

constexpr int phi_pitch(int wx) { return align_up(wx + kPhiPadLeft + 1, 4); }
constexpr int phi_bytes(int wx, int wy) { return (wy + 2) * phi_pitch(wx) * 4; }
constexpr int s_bytes(int wx, int wy) { return wy * wx * 4; }
constexpr int class_smem(int wx, int wy) {
    return kScalarBytes + phi_bytes(wx, wy) + s_bytes(wx, wy);
}
constexpr int class_smem_large(int wx, int wy) {
    return kScalarBytes + phi_bytes(wx, wy);
}
constexpr int class_smem_of(int c) {
    return !class_stages_phi(c) ? kScalarBytes
         : class_stages_S(c) ? class_smem(kClasses[c].wx, kClasses[c].wy)
                             : class_smem_large(kClasses[c].wx, kClasses[c].wy);
}

constexpr int cmax(int a, int b) { return a > b ? a : b; }

constexpr int smem_raw_all() {
    int m = 0;
    for (int c = 0; c < kNumClasses; ++c) m = cmax(m, class_smem_of(c));
    return m;
}
constexpr int smem_raw_staged_only() {
    int m = 0;
    for (int c = 0; c < kNumClasses; ++c)
        if (class_stages_S(c)) m = cmax(m, class_smem_of(c));
    return m;
}
constexpr int kSmemRaw = smem_raw_all();
constexpr int kSmemBytes = align_up(kSmemRaw, 128);
constexpr int kLargeClassSmemRaw = class_smem_of(kClassLarge);
constexpr int kLargeClassSmemBytes = align_up(kLargeClassSmemRaw, 128);
constexpr int kExpectedLargeClassSmemRaw = 211904;

static_assert(kLargeClassSmemRaw == kExpectedLargeClassSmemRaw,
              "largest shared-phi class memory calculation changed");
static_assert(kLargeClassSmemBytes == 211968,
              "largest shared-phi class alignment changed");
static_assert(kSmemRaw == 213440 && kSmemBytes == 213504,
              "shared-class launch memory changed");

// The largest shared-phi class must remain below the requirement of the
// staged classes so it does not reduce occupancy or alter the launch request.
static_assert(kSmemRaw == smem_raw_staged_only(),
              "the large class raised kSmemRaw: it must fit inside the budget "
              "the staged classes already set, or it is not free");
static_assert(class_smem_of(kClassLarge) < smem_raw_staged_only(),
              "the large class must be strictly cheaper than the staged "
              "classes, otherwise there is no reason for it to skip S_s");
static_assert(class_smem_of(kClassFallback) == kScalarBytes,
              "the fallback must not reserve a shared phi or S field");

// Per-block opt-in shared-memory limit on sm_90; the per-SM limit is 233472 B.
constexpr int kSmemPerBlockOptinSm90 = 232448;
static_assert(kSmemBytes <= kSmemPerBlockOptinSm90,
              "dynamic shared memory request exceeds the sm_90 per-block "
              "opt-in maximum");
constexpr int kSmemLaunchMarginSm90 = kSmemPerBlockOptinSm90 - kSmemBytes;
constexpr int kLargeClassMarginToStaged = kSmemRaw - kLargeClassSmemRaw;
static_assert(kSmemLaunchMarginSm90 == 18944,
              "unexpected sm_90 per-block shared-memory margin");
static_assert(kLargeClassMarginToStaged == 1536,
              "unexpected large-to-staged shared-memory margin");
static_assert(kLargeClassSmemBytes <= kSmemPerBlockOptinSm90,
              "largest shared-phi class exceeds the sm_90 opt-in maximum");

// sm_90 limits used by the startup occupancy report.
constexpr int kSmemPerSmSm90       = 233472;
constexpr int kRegsPerSmSm90       = 65536;
constexpr int kMaxThreadsPerSmSm90 = 2048;

// Per-class structural invariants.
namespace detail {
constexpr bool class_ok(ShapeClass c) {
    return c.wy % kStripRows == 0            // whole strips, no partial tail
        && c.wx % 4 == 0                     // 16 B cp.async chunks on phi
        && c.tx0 % 32 == 0 && c.ty0 % 32 == 0  // 128 B aligned tile rows
        && c.tx0 >= 1 && c.ty0 >= 1          // room for the zero ring
        && c.tx0 + c.wx <= kTilePitch - 1
        && c.ty0 + c.wy <= kTilePitch - 1
        && phi_pitch(c.wx) % 4 == 0;
}
// Apply the same source-row alignment and zero-border rules to every class.
static_assert(class_ok(kClasses[0]), "shape class 0 violates a layout rule");
static_assert(class_ok(kClasses[1]), "shape class 1 violates a layout rule");
static_assert(class_ok(kClasses[2]), "shape class 2 violates a layout rule");
static_assert(class_ok(kClasses[3]), "shape class 3 violates a layout rule");
static_assert(class_ok(kClasses[4]), "shape class 4 violates a layout rule");
static_assert(kNumClasses == 6, "add a layout assertion for each new class");
constexpr bool fallback_class_ok(ShapeClass c) {
    return c.wx == kTilePitch - 2 && c.wy == kTilePitch - 2
        && c.tx0 == 1 && c.ty0 == 1;
}
static_assert(fallback_class_ok(kClasses[kClassFallback]),
              "fallback must leave a one-pixel tile ring");
static_assert(kClasses[0].wx == kClasses[0].wy, "class 0 must be the square one");

// Check 16-byte alignment of every cp.async destination row.
constexpr bool cpasync_dst_ok(ShapeClass c) {
    return (kScalarBytes % 16 == 0) && ((phi_pitch(c.wx) * 4) % 16 == 0)
        && ((kPhiPadLeft * 4) % 16 == 0);
}
static_assert(cpasync_dst_ok(kClasses[0]), "class 0 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[1]), "class 1 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[2]), "class 2 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[3]), "class 3 breaks cp.async dst alignment");
static_assert(cpasync_dst_ok(kClasses[4]), "class 4 breaks cp.async dst alignment");

// A class change must not narrow either axis below the round window, or it could
// truncate support along the axis that did not trigger the change.
constexpr bool class_not_narrower(ShapeClass c) {
    return c.wx >= kClasses[0].wx && c.wy >= kClasses[0].wy;
}
static_assert(class_not_narrower(kClasses[1]),
              "shape class 1 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[2]),
              "shape class 2 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[3]),
              "shape class 3 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[4]),
              "shape class 4 is narrower than class 0 on one axis: promoting "
              "into it would truncate the support on that axis");
static_assert(class_not_narrower(kClasses[5]),
              "fallback is narrower than class 0");

// The largest shared-phi class must contain classes 0--3 on both axes.
constexpr bool dominated_by_large(ShapeClass c) {
    return c.wx <= kClasses[kClassLarge].wx && c.wy <= kClasses[kClassLarge].wy;
}
static_assert(dominated_by_large(kClasses[0]), "class 0 is not covered by the large class");
static_assert(dominated_by_large(kClasses[1]), "class 1 is not covered by the large class");
static_assert(dominated_by_large(kClasses[2]), "class 2 is not covered by the large class");
static_assert(dominated_by_large(kClasses[3]), "class 3 is not covered by the large class");

constexpr bool dominated_by_fallback(ShapeClass c) {
    return c.wx <= kClasses[kClassFallback].wx
        && c.wy <= kClasses[kClassFallback].wy;
}
static_assert(dominated_by_fallback(kClasses[kClassLarge]),
              "fallback must contain the largest shared-memory class");

// It must also have the largest area so minimum-area selection prefers cheaper
// staged classes whenever they fit.
constexpr bool smaller_area_than_large(ShapeClass c) {
    return c.wx * c.wy < kClasses[kClassLarge].wx * kClasses[kClassLarge].wy;
}
static_assert(smaller_area_than_large(kClasses[0]), "class 0 is not smaller in area than the large class");
static_assert(smaller_area_than_large(kClasses[1]), "class 1 is not smaller in area than the large class");
static_assert(smaller_area_than_large(kClasses[2]), "class 2 is not smaller in area than the large class");
static_assert(smaller_area_than_large(kClasses[3]), "class 3 is not smaller in area than the large class");
static_assert(kClasses[kClassLarge].wx * kClasses[kClassLarge].wy
                  < kClasses[kClassFallback].wx * kClasses[kClassFallback].wy,
              "fallback must be more expensive than shared-phi classes");
}  // namespace detail

// Sticky device flags. All except FLAG_SUPPORT_CLIP invalidate the run.
enum : int {
    FLAG_S_OVERFLOW = 0,   // Q5.27 accumulator wrapped
    FLAG_Q_CLAMP    = 1,   // a single contribution hit phi^2 > 4
    FLAG_SUPPORT_CLIP = 2, // support bbox touched the rect edge
    FLAG_CLASS_EXHAUSTED = 3,  // support exceeded the tile interior
    FLAG_S_NEGATIVE = 4,   // q_S < q_of(phi_n): counted, never floored silently
    FLAG_NONFINITE  = 5,   // phi went non-finite
    FLAG_V_NONPOS   = 6,   // carried V <= 0, recentring skipped
    // The dispatch received a class id for which no kernel specialization exists.
    FLAG_CLASS_UNSUPPORTED = 7,
    FLAG_COUNT      = 8
};
// Fatal checks are always enabled. PF_ALARMS additionally records when the
// phi > kSupportEps bounding box touches a window edge. That event is advisory;
// failure to fit any shape class is a separate, always-fatal condition.
#if defined(PF_NO_ALARMS)
#error "fatal integrity checks cannot be disabled; omit PF_ALARMS to disable only the support-edge diagnostic"
#endif

#define PF_FATAL_ADD(flags, idx) atomicAdd(&(flags)[idx], 1u)
#define PF_FATAL_OR(flags, idx)  atomicOr(&(flags)[idx], 1u)

#if defined(PF_ALARMS)
#define PF_ADVISORY_ADD(flags, idx) atomicAdd(&(flags)[idx], 1u)
#define PF_SUPPORT_CLIP_ENABLED 1
#else
#define PF_ADVISORY_ADD(flags, idx) ((void)0)
#define PF_SUPPORT_CLIP_ENABLED 0
#endif

// The device buffer and every host readback use this dense range.
static_assert(FLAG_CLASS_UNSUPPORTED < FLAG_COUNT, "flag index out of range");

constexpr bool flag_is_fatal(int i) {
    return i >= 0 && i < FLAG_COUNT && i != FLAG_SUPPORT_CLIP;
}
static_assert(!flag_is_fatal(FLAG_SUPPORT_CLIP),
              "support_clip is the sole advisory flag");
static_assert(flag_is_fatal(FLAG_S_OVERFLOW), "S_overflow must be fatal");
static_assert(flag_is_fatal(FLAG_Q_CLAMP), "q_clamp must be fatal");
static_assert(flag_is_fatal(FLAG_CLASS_EXHAUSTED),
              "class_exhausted must be fatal");
static_assert(flag_is_fatal(FLAG_S_NEGATIVE),
              "S_other_negative must be fatal");
static_assert(flag_is_fatal(FLAG_NONFINITE), "phi_nonfinite must be fatal");
static_assert(flag_is_fatal(FLAG_V_NONPOS), "V_nonpositive must be fatal");
static_assert(flag_is_fatal(FLAG_CLASS_UNSUPPORTED),
              "class_unsupported must be fatal");

inline const char* flag_name(int i) {
    switch (i) {
        case FLAG_S_OVERFLOW:      return "S_overflow";
        case FLAG_Q_CLAMP:         return "q_clamp";
        case FLAG_SUPPORT_CLIP:    return "support_clip";
        case FLAG_CLASS_EXHAUSTED: return "class_exhausted";
        case FLAG_S_NEGATIVE:      return "S_other_negative";
        case FLAG_NONFINITE:       return "phi_nonfinite";
        case FLAG_V_NONPOS:        return "V_nonpositive";
        case FLAG_CLASS_UNSUPPORTED: return "class_unsupported";
        default:                   return "unused";
    }
}

// Simulation parameters. Host doubles reduce roundoff in coefficient derivation
// and time accumulation; the kernel takes float copies where required.
struct SimParams {
    int    Nx = 0, Ny = 0;          // domain side, pixels (must be equal)
    double dx = 1.0, dy = 1.0;      // must both be exactly 1
    int    num_cells = 288;
    double rho = 0.90;              // packing fraction, sets Nx from N and R
    double dt = 0.01;
    double t_end = 100.0;
    double lambda = 7.0;
    double target_radius = 49.0;
    double kappa = 10.0;
    double mu = 1.0;
    double xi = 1500.0;
    double tau = 1.0e4;
    double v_A = 1.0e-2;
    double gamma_normal = 1.0;
    double gamma_cancer = 0.35;
    double cancer_fraction = 0.0;   // fraction of cells given gamma_cancer
    double v_A_sigma = 0.0;         // lognormal spread on per-cell v_A
    unsigned long long seed = 1234;
    // Independent stream for initial polarity and tumbles; zero reuses seed.
    // Paired simulations can therefore share reorientation events while using
    // independent placements.
    unsigned long long polarity_seed = 0;
    // FNV-1a fingerprint of the accepted binary32 centre table. It is output
    // metadata and checkpoint provenance, not an input to the dynamics.
    unsigned long long initialization_hash = 0;
    int    print_interval = 100;
    int    full_moment_every = 100;
    int    verify_every = 4096;

    // Stream used for the initial angle and every tumble event.
    __host__ __device__ unsigned long long polarity_stream() const {
        return polarity_seed ? polarity_seed : seed;
    }

    __host__ __device__ double area0() const {
        return target_area(target_radius);
    }
    __host__ __device__ double bulk() const { return bulk_coeff(lambda); }
    __host__ __device__ double interaction() const {
        return interaction_coeff(kappa, lambda);
    }
    __host__ __device__ double motility() const {
        return motility_coeff(kappa, xi, lambda);
    }
    __host__ __device__ double volume() const { return volume_coeff(mu, area0()); }
    __host__ __device__ double dA() const { return dx * dy; }
    __host__ __device__ double realized_rho() const {
        return Nx > 0 && Ny > 0
            ? static_cast<double>(num_cells) * area0() /
                  (static_cast<double>(Nx) * Ny)
            : 0.0;
    }

    // Per-step tumble probability, evaluated in double with expm1 to avoid
    // cancellation when dt/tau is small (1.33% bias at the default parameters).
    double p_tumble() const { return -std::expm1(-dt / tau); }

    // Return -1 when the rounded step count does not fit in int64; validate()
    // rejects it before the simulation starts.
    long long total_steps() const {
        const double n = t_end / dt + 0.5;
        return (n >= 0.0 && n <= 9.0e18) ? (long long)n : -1LL;
    }
};

// Domain side for N cells of area A0 at packing fraction rho.
inline int domain_side_for(int n, double radius, double rho) {
    const double a = (double)n * target_area(radius) / rho;
    const double side = std::ceil(std::sqrt(a));
    return std::isfinite(side) && side >= 1.0 &&
                   side <= static_cast<double>(std::numeric_limits<int>::max())
        ? static_cast<int>(side) : -1;
}

// Row pitch of an S buffer, in uint32: 32-word (128 B) aligned rows. Return -1
// rather than overflowing when a corrupt checkpoint supplies an extreme side.
inline constexpr int s_pitch_for(int side) {
    if (side <= 0) return -1;
    const long long pitch = 32LL *
        ((static_cast<long long>(side) + 31LL) / 32LL);
    return pitch <= std::numeric_limits<int>::max()
        ? static_cast<int>(pitch) : -1;
}
static_assert(s_pitch_for(1) == 32, "aligned pitch rounds up");
static_assert(s_pitch_for(2147483616) == 2147483616,
              "largest representable aligned pitch changed");
static_assert(s_pitch_for(2147483617) == -1,
              "aligned pitch overflow must be rejected");

// Startup validation. Returns false and prints an actionable message rather
// than asserting, so invalid input cannot silently produce invalid output.
inline bool validate(const SimParams& p) {
    bool ok = true;
    auto fail = [&](const char* msg) {
        std::fprintf(stderr, "[fatal] %s\n", msg);
        ok = false;
    };

    if (!(p.dx == 1.0 && p.dy == 1.0)) {
        std::fprintf(stderr,
            "[fatal] dx and dy must both be exactly 1.0 (got %.17g, %.17g).\n"
            "        The 9-point Laplacian, the 1/(2h) gradients and the\n"
            "        dA = 1 quadrature are all hard-coded for h = 1. Rescale\n"
            "        lambda and R instead of changing h.\n", p.dx, p.dy);
        ok = false;
    }
    if (p.Nx != p.Ny) {
        std::fprintf(stderr,
            "[fatal] Nx must equal Ny (got %d x %d). The shared field S uses a\n"
            "        single side length L for both axes.\n", p.Nx, p.Ny);
        ok = false;
    }
    if (p.Nx <= 0) {
        fail("domain side is invalid or exceeds the supported integer range");
    } else if (s_pitch_for(p.Nx) < 0) {
        fail("domain side is too large for the aligned shared-field pitch");
    }
    if (p.num_cells <= 0) fail("num_cells must be positive");
    if (!std::isfinite(p.lambda) || !(p.lambda > 0.0))
        fail("lambda must be finite and positive");
    if (!std::isfinite(p.target_radius) || !(p.target_radius > 0.0))
        fail("radius must be finite and positive");
    if (!std::isfinite(p.dt) || !(p.dt > 0.0))
        fail("dt must be finite and positive");
    if (!std::isfinite(p.t_end) || p.t_end < 0.0)
        fail("t_end must be finite and non-negative");
    if (p.total_steps() < 0) {
        std::fprintf(stderr,
            "[fatal] t_end/dt = %.6g/%.6g does not fit in a 64-bit step count.\n"
            "        Reduce t_end or increase dt (the limit is 9e18 steps).\n",
            p.t_end, p.dt);
        ok = false;
    }
    if (!std::isfinite(p.xi) || !(p.xi > 0.0))
        fail("xi must be finite and positive");
    if (!std::isfinite(p.tau) || !(p.tau > 0.0))
        fail("tau must be finite and positive");
    if (!std::isfinite(p.kappa) || p.kappa < 0.0)
        fail("kappa must be finite and non-negative");
    if (!std::isfinite(p.mu) || p.mu < 0.0)
        fail("mu must be finite and non-negative");
    if (!std::isfinite(p.v_A) || p.v_A < 0.0 ||
        !std::isfinite(p.v_A_sigma) || p.v_A_sigma < 0.0)
        fail("v_A and v_A_sigma must be finite and non-negative");
    if (!std::isfinite(p.gamma_normal) ||
        !std::isfinite(p.gamma_cancer) ||
        !(p.gamma_normal > 0.0) || !(p.gamma_cancer > 0.0))
        fail("both gamma values must be finite and positive");
    if (!std::isfinite(p.cancer_fraction) ||
        !(p.cancer_fraction >= 0.0 && p.cancer_fraction <= 1.0))
        fail("cancer_fraction must be finite and lie in [0,1]");
    if (!std::isfinite(p.rho) || !(p.rho > 0.0 && p.rho < 1.0))
        fail("rho must be finite and lie in (0,1)");
    if (p.print_interval < 0 || p.full_moment_every < 0 ||
        p.verify_every < 0)
        fail("output and verification intervals must be non-negative");

    // The rect must fit strictly inside the domain, otherwise two rect pixels
    // alias onto one global pixel and q_S - q_of(phi_n) stops being an exact
    // self-subtraction.
    int wmax = 0;
    for (int c = 0; c < kNumClasses; ++c) {
        wmax = wmax > kClasses[c].wx ? wmax : kClasses[c].wx;
        wmax = wmax > kClasses[c].wy ? wmax : kClasses[c].wy;
    }
    if (p.Nx > 0 && p.Nx <= wmax) {
        std::fprintf(stderr,
            "[fatal] domain side %d must exceed the largest rect dimension %d.\n"
            "        Increase N or decrease rho so the periodic domain is larger.\n",
            p.Nx, wmax);
        ok = false;
    }

    // Conservative explicit-Euler guard using the Laplacian row-sum bound.
    const double gmax = p.gamma_normal > p.gamma_cancer ? p.gamma_normal
                                                        : p.gamma_cancer;
    const double cfl = p.dt * gmax * kLapRowSumBound;
    if (cfl >= 1.0) {
        std::fprintf(stderr,
            "[fatal] dt*gamma*laplacian_bound = %.4f; the configured safety "
            "limit is 1. Reduce dt below %.5g.\n",
            cfl, 1.0 / (gmax * kLapRowSumBound));
        ok = false;
    } else if (cfl > 0.5) {
        std::fprintf(stderr,
            "[warn] dt*gamma*laplacian_bound = %.4f; margin to the configured "
            "safety limit is under 2x.\n", cfl);
    }

    // Runtime check that interaction_coeff/motility_coeff equals xi.
    const double ratio = p.interaction() / p.motility();
    if (!(std::fabs(ratio - p.xi) <= 1e-9 * p.xi)) {
        std::fprintf(stderr,
            "[fatal] interaction_coeff/motility_coeff = %.17g but xi = %.17g.\n"
            "        One of the two coefficients has the wrong numerical factor.\n",
            ratio, p.xi);
        ok = false;
    }
    return ok;
}

inline void print_params(const SimParams& p, int side, int pitch) {
    std::printf("--- simulation configuration ---\n");
    std::printf("  cells            %d\n", p.num_cells);
    std::printf("  domain           %d x %d  (rho target/realized = %.6g / %.6g)\n",
                side, side, p.rho, p.realized_rho());
    std::printf("  S pitch          %d uint32  (%.2f MB/buffer, x3)\n",
                pitch, (double)pitch * side * 4.0 / 1048576.0);
    std::printf("  R, lambda        %.4g, %.4g   (init R_eff = %.6g)\n",
                p.target_radius, p.lambda,
                init_radius(p.target_radius, p.lambda));
    std::printf("  dt, t_end        %.6g, %.6g  (%lld steps)\n",
                p.dt, p.t_end, p.total_steps());
    std::printf("  kappa, mu, xi    %.6g, %.6g, %.6g\n", p.kappa, p.mu, p.xi);
    std::printf("  gamma n/c        %.6g / %.6g  (cancer fraction %.4g)\n",
                p.gamma_normal, p.gamma_cancer, p.cancer_fraction);
    std::printf("  v_A, tau         %.6g, %.6g\n", p.v_A, p.tau);
    std::printf("  A0               %.6f\n", p.area0());
    std::printf("  bulk  30/lambda^2             %.9g\n", p.bulk());
    std::printf("  inter 60*kappa/lambda^2       %.9g\n", p.interaction());
    std::printf("  motil 60*kappa/(xi*lambda^2)  %.9g\n", p.motility());
    std::printf("  inter/motil                    %.9g   (must equal xi)\n",
                p.interaction() / p.motility());
    std::printf("  vol   2mu/A0     %.9g\n", p.volume());
    std::printf("  p_tumble         %.9e   (-expm1(-dt/tau))\n", p.p_tumble());
    std::printf("  fatal checks     ENABLED\n");
    std::printf("  support-edge diagnostic  %s\n",
                PF_SUPPORT_CLIP_ENABLED
                    ? "ENABLED (-DPF_ALARMS; advisory)"
                    : "DISABLED (default)");
    std::printf("  support layout   fixed tile %d (guarded support %d px/axis, "
                "physical interior %d)\n",
                kTilePitch,
                class_support_capacity(kClassFallback, kPromoteSlack),
                kClasses[kClassFallback].wx);
    std::printf("  shared memory/block  %d B of %d B opt-in max\n",
                kSmemBytes, kSmemPerBlockOptinSm90);
    std::printf("  largest shared-phi class  %d: %d x %d; "
                "%d B shared-memory margin\n",
                kClassLarge, kClasses[kClassLarge].wx, kClasses[kClassLarge].wy,
                kSmemLaunchMarginSm90);
    std::printf("  fallback class   %d: %d x %d at (1,1), phi and S global\n",
                kClassFallback, kClasses[kClassFallback].wx,
                kClasses[kClassFallback].wy);
    std::printf("  step kernels     shared-class update then fallback filter "
                "(%d threads; %d / %d B shared memory)\n",
                kBlockThreads, kSmemBytes, kScalarBytes);
}

}  // namespace pf
