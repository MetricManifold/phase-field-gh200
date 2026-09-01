#pragma once

// Reproducible cell-identity utilities shared by three-dimensional fresh
// starts. Spatial placement is defined in palmieri_initializer.hpp.

#include "rng.cuh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pf3d {

inline constexpr std::uint32_t kInitDomainSoftSelection3D = 0x33544653u;

namespace initializer_detail {

inline std::uint64_t selection_word(std::uint64_t draw,
                                    std::uint32_t domain,
                                    std::uint32_t attempt,
                                    std::uint64_t seed) {
    const Philox4 random = philox4x32_10(
        static_cast<std::uint32_t>(draw),
        static_cast<std::uint32_t>(draw >> 32), domain, attempt,
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32));
    return (static_cast<std::uint64_t>(random.v[0]) << 32) | random.v[1];
}

// Rejection before modulo makes the result uniform for every bound. The
// attempt is part of the counter-based draw, so no mutable RNG state is hidden.
inline std::uint64_t bounded_selection_word(std::uint64_t draw,
                                            std::uint32_t domain,
                                            std::uint64_t seed,
                                            std::uint64_t bound) {
    if (bound == 0)
        throw std::invalid_argument("selection bound must be positive");
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
    for (std::uint32_t attempt = 0;; ++attempt) {
        const std::uint64_t random = selection_word(draw, domain, attempt, seed);
        if (random >= threshold) return random % bound;
        if (attempt == std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("bounded Philox selection did not converge");
    }
}

inline std::vector<int> sample_without_replacement(int population, int count,
                                                   std::uint64_t seed,
                                                   std::uint32_t domain) {
    if (population < 0 || count < 0 || count > population)
        throw std::invalid_argument("invalid without-replacement sample size");
    std::vector<int> values(static_cast<std::size_t>(population));
    std::iota(values.begin(), values.end(), 0);
    for (int i = 0; i < count; ++i) {
        const std::uint64_t remaining =
            static_cast<std::uint64_t>(population - i);
        const int selected = i + static_cast<int>(bounded_selection_word(
            static_cast<std::uint64_t>(i), domain, seed, remaining));
        std::swap(values[static_cast<std::size_t>(i)],
                  values[static_cast<std::size_t>(selected)]);
    }
    values.resize(static_cast<std::size_t>(count));
    return values;
}

inline void fnv_byte(std::uint64_t* hash, std::uint8_t byte) {
    *hash ^= byte;
    *hash *= UINT64_C(0x00000100000001b3);
}

}  // namespace initializer_detail

// Soft identities are sampled independently of placement. Matched control and
// soft-cell simulations can therefore share a centre table and seed without
// coupling positions to the requested number of soft cells.
inline std::vector<std::uint8_t> soft_id_membership(
    int n, double fraction, std::uint64_t seed) {
    if (n < 1 || !std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
        throw std::invalid_argument("invalid soft-cell selection parameters");
    const int count = static_cast<int>(
        std::llround(fraction * static_cast<double>(n)));
    std::vector<std::uint8_t> membership(static_cast<std::size_t>(n), 0u);
    const std::vector<int> selected =
        initializer_detail::sample_without_replacement(
            n, count, seed, kInitDomainSoftSelection3D);
    for (int id : selected) membership[static_cast<std::size_t>(id)] = 1u;
    return membership;
}

// Hash the logical centre table as little-endian int64/float32 words,
// independent of host padding and CSV formatting.
inline std::uint64_t centre_table_fnv1a64(
    const std::vector<std::int64_t>& global_id,
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& z) {
    if (global_id.size() != x.size() || x.size() != y.size() ||
        y.size() != z.size()) {
        return 0;
    }
    std::uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::uint64_t id_bits = static_cast<std::uint64_t>(global_id[i]);
        for (unsigned shift = 0; shift < 64; shift += 8) {
            initializer_detail::fnv_byte(
                &hash, static_cast<std::uint8_t>(id_bits >> shift));
        }
        const float values[3] = {x[i], y[i], z[i]};
        for (float value : values) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (unsigned shift = 0; shift < 32; shift += 8) {
                initializer_detail::fnv_byte(
                    &hash, static_cast<std::uint8_t>(bits >> shift));
            }
        }
    }
    return hash;
}

inline std::vector<std::int64_t> ordered_global_ids(std::size_t count) {
    std::vector<std::int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), std::int64_t{0});
    return ids;
}

}  // namespace pf3d
