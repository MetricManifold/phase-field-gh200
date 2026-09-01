#pragma once

// Philox4x32-10 primitive shared by the 2D and 3D solvers. Each solver keeps
// its own counter domains and maps these words to physical random variables.

#include <cstdint>

#if defined(__CUDACC__)
#define PF_COMMON_HD __host__ __device__
#define PF_COMMON_FORCEINLINE __forceinline__
#else
#define PF_COMMON_HD
#define PF_COMMON_FORCEINLINE inline
#endif

namespace pf_common {

struct Philox4 {
    std::uint32_t v[4];
};

PF_COMMON_HD PF_COMMON_FORCEINLINE std::uint32_t mulhi32(
    std::uint32_t a, std::uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b)) >> 32);
#endif
}

PF_COMMON_HD PF_COMMON_FORCEINLINE Philox4 philox4x32_10(
    std::uint32_t c0, std::uint32_t c1, std::uint32_t c2, std::uint32_t c3,
    std::uint32_t k0, std::uint32_t k1) {
    constexpr std::uint32_t kMultiplier0 = 0xD2511F53u;
    constexpr std::uint32_t kMultiplier1 = 0xCD9E8D57u;
    constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
    constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k0 += kWeyl0;
            k1 += kWeyl1;
        }
        const std::uint32_t hi0 = mulhi32(kMultiplier0, c0);
        const std::uint32_t lo0 = kMultiplier0 * c0;
        const std::uint32_t hi1 = mulhi32(kMultiplier1, c2);
        const std::uint32_t lo1 = kMultiplier1 * c2;
        const std::uint32_t n0 = hi1 ^ c1 ^ k0;
        const std::uint32_t n1 = lo1;
        const std::uint32_t n2 = hi0 ^ c3 ^ k1;
        const std::uint32_t n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    return Philox4{{c0, c1, c2, c3}};
}

// Uniform variate on [0,1) using the high 53 bits of two output words.
PF_COMMON_HD PF_COMMON_FORCEINLINE double philox_uniform53(
    std::uint32_t a, std::uint32_t b) {
    const std::uint64_t mantissa =
        ((static_cast<std::uint64_t>(a) << 32) |
         static_cast<std::uint64_t>(b)) >> 11;
    return static_cast<double>(mantissa) * (1.0 / 9007199254740992.0);
}

}  // namespace pf_common

#undef PF_COMMON_FORCEINLINE
#undef PF_COMMON_HD
