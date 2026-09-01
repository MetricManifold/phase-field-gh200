#pragma once

// Dimension-independent coefficients in the phase-field model. Spatial
// discretization and target area/volume definitions remain solver-specific.

#if defined(__CUDACC__)
#define PF_COMMON_HD __host__ __device__
#define PF_COMMON_INLINE __forceinline__
#else
#define PF_COMMON_HD
#define PF_COMMON_INLINE inline
#endif

namespace pf_common {

constexpr int kNumerBulk = 30;
constexpr int kNumerInteraction = 60;
constexpr int kNumerVolume = 2;

static_assert(kNumerInteraction == 2 * kNumerBulk,
              "the ordered-pair interaction factor must remain 60 = 2*30");

template <typename T>
PF_COMMON_HD constexpr T bulk_coeff(T lambda) {
    return T(kNumerBulk) / (lambda * lambda);
}

template <typename T>
PF_COMMON_HD constexpr T interaction_coeff(T kappa, T lambda) {
    return T(kNumerInteraction) * kappa / (lambda * lambda);
}

// Defining this coefficient from the interaction term preserves their exact
// ratio in both solvers.
template <typename T>
PF_COMMON_HD constexpr T motility_coeff(T kappa, T xi, T lambda) {
    return interaction_coeff(kappa, lambda) / xi;
}

template <typename T>
PF_COMMON_HD constexpr T volume_coeff(T mu, T target_measure) {
    return T(kNumerVolume) * mu / target_measure;
}

template <typename T>
PF_COMMON_HD PF_COMMON_INLINE T interface_k(T lambda) {
    return T(2.7386127875258306) / lambda;  // sqrt(7.5)/lambda
}

template <typename T>
PF_COMMON_HD PF_COMMON_INLINE T init_radius(T radius, T lambda) {
    return radius + T(0.5) / interface_k(lambda);
}

}  // namespace pf_common

#undef PF_COMMON_INLINE
#undef PF_COMMON_HD
