#include "model_coefficients.cuh"
#include "philox.cuh"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

bool check_philox(std::array<std::uint32_t, 4> counter,
                  std::array<std::uint32_t, 2> key,
                  std::array<std::uint32_t, 4> expected) {
    const pf_common::Philox4 result = pf_common::philox4x32_10(
        counter[0], counter[1], counter[2], counter[3], key[0], key[1]);
    return std::memcmp(result.v, expected.data(), sizeof(result.v)) == 0;
}

template <typename T>
bool check_coefficients(T lambda, T kappa, T xi, T mu, T target) {
    return same_bits(pf_common::bulk_coeff(lambda),
                     T(30) / (lambda * lambda)) &&
           same_bits(pf_common::interaction_coeff(kappa, lambda),
                     T(60) * kappa / (lambda * lambda)) &&
           same_bits(pf_common::motility_coeff(kappa, xi, lambda),
                     (T(60) * kappa / (lambda * lambda)) / xi) &&
           same_bits(pf_common::volume_coeff(mu, target),
                     T(2) * mu / target) &&
           same_bits(pf_common::interface_k(lambda),
                     T(2.7386127875258306) / lambda) &&
           same_bits(pf_common::init_radius(target, lambda),
                     target + T(0.5) /
                         (T(2.7386127875258306) / lambda));
}

}  // namespace

int main() {
    static_assert(pf_common::kNumerBulk == 30);
    static_assert(pf_common::kNumerInteraction == 60);
    static_assert(pf_common::kNumerVolume == 2);

    const bool coefficients =
        check_coefficients<float>(7.0f, 10.0f, 1500.0f, 1.0f, 7542.0f) &&
        check_coefficients<float>(5.25f, 3.7f, 977.0f, 0.125f, 11.75f) &&
        check_coefficients<double>(7.0, 10.0, 1500.0, 1.0, 7542.0) &&
        check_coefficients<double>(0.5, 0.125, 65536.0, 3.7, 0.75);

    const bool philox =
        check_philox({0u, 0u, 0u, 0u}, {0u, 0u},
                     {0x6627E8D5u, 0xE169C58Du, 0xBC57AC4Cu, 0x9B00DBD8u}) &&
        check_philox({1u, 2u, 3u, 4u}, {5u, 6u},
                     {0xC0C839BCu, 0x889C87C5u, 0x61986739u, 0x2D4623D0u}) &&
        check_philox({0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu},
                     {0xFFFFFFFFu, 0xFFFFFFFFu},
                     {0x408F276Du, 0x41C83B0Eu, 0xA20BC7C6u, 0x6D5451FDu});

    if (!coefficients || !philox) {
        std::fprintf(stderr, "common primitive golden-vector test failed\n");
        return 1;
    }
    std::printf("common primitive golden-vector test passed\n");
    return 0;
}
