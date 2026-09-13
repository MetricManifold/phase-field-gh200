#include "pf3d/bounded_z_map.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
constexpr int sentinel = -1234567;
std::uint64_t fetch_cases = 0;
std::uint64_t aggregate_cases = 0;

// Independent branch-for-branch oracle from device_common.cuh.
// It retains the original checked world-coordinate operations, not the map's
// interval or ghost-alias derivation.
bool checked_sum(std::int64_t origin, int offset, std::int64_t* world) {
    if ((offset > 0 && origin > maximum - static_cast<std::int64_t>(offset))
        || (offset < 0 && origin < minimum - static_cast<std::int64_t>(offset)))
        return false;
    *world = origin + static_cast<std::int64_t>(offset);
    return true;
}

bool original_fetch(std::int64_t origin, int requested, int B, int nz,
                    bool channel, int* source) {
    std::int64_t world = 0;
    if (!checked_sum(origin, requested, &world)) return false;
    if (channel && world == static_cast<std::int64_t>(nz)) world = nz - 1;
    if (world >= static_cast<std::int64_t>(nz) || world < -1) return false;
    if (world == -1) world = 0;
    if ((origin > 0 && world < minimum + origin)
        || (origin < 0 && world > maximum + origin)) return false;
    const std::int64_t local = world - origin;
    if (local < 0 || local >= static_cast<std::int64_t>(B)) return false;
    *source = static_cast<int>(local);
    return true;
}

bool original_aggregate(std::int64_t origin, int local, int nz, int* result) {
    std::int64_t world = 0;
    if (!checked_sum(origin, local, &world) || world < 0 || world >= nz) return false;
    *result = static_cast<int>(world);
    return true;
}

[[noreturn]] void fail(const char* operation, std::int64_t origin, int B,
                      int nz, int q, bool channel) {
    std::cerr << operation << " mismatch: origin=" << origin << " B=" << B
              << " nz=" << nz << " q=" << q << " channel=" << channel << '\n';
    std::exit(1);
}

void check(std::int64_t origin, int B, int nz, bool channel, std::int64_t offset) {
    if (offset < std::numeric_limits<int>::min()
        || offset > std::numeric_limits<int>::max()) return;
    const int q = static_cast<int>(offset);
    const auto map = pf3d::BoundedZMap::make(origin, B, nz, channel);
    int expected = sentinel, actual = sentinel;
    const bool expected_fetch = original_fetch(origin, q, B, nz, channel, &expected);
    const bool actual_fetch = map.fetch(q, &actual);
    ++fetch_cases;
    if (expected_fetch != actual_fetch || expected != actual)
        fail("fetch", origin, B, nz, q, channel);

    // aggregate_z's existing voxel callers first require 0 <= q < B.
    if (q >= 0 && q < B) {
        expected = actual = sentinel;
        const bool expected_aggregate = original_aggregate(origin, q, nz, &expected);
        const bool actual_aggregate = map.aggregate(q, &actual);
        ++aggregate_cases;
        if (expected_aggregate != actual_aggregate || expected != actual)
            fail("aggregate", origin, B, nz, q, channel);
    }
}

void check_endpoints(std::int64_t origin, int B, int nz, bool channel) {
    const auto map = pf3d::BoundedZMap::make(origin, B, nz, channel);
    for (const std::int64_t q : {
            static_cast<std::int64_t>(std::numeric_limits<int>::min()),
            static_cast<std::int64_t>(std::numeric_limits<int>::min()) + 1,
            -static_cast<std::int64_t>(B) - 1, -static_cast<std::int64_t>(B),
            std::int64_t{-2}, std::int64_t{-1}, std::int64_t{0}, std::int64_t{1},
            static_cast<std::int64_t>(B) - 1, static_cast<std::int64_t>(B),
            static_cast<std::int64_t>(B) + 1,
            static_cast<std::int64_t>(map.lo) - 1, static_cast<std::int64_t>(map.lo),
            static_cast<std::int64_t>(map.hi) - 1, static_cast<std::int64_t>(map.hi),
            static_cast<std::int64_t>(std::numeric_limits<int>::max()) - 1,
            static_cast<std::int64_t>(std::numeric_limits<int>::max())})
        check(origin, B, nz, channel, q);
}

} // namespace

int main() {
    for (int B = 1; B <= 16; ++B)
        for (int nz = 1; nz <= 16; ++nz)
            for (const bool channel : {false, true})
                for (std::int64_t origin = -32; origin <= 32; ++origin) {
                    for (int q = -B - 2; q <= B + 2; ++q)
                        check(origin, B, nz, channel, q);
                    check_endpoints(origin, B, nz, channel);
                }

    // Real brick sizes, a channel thinner than its brick, exact face contacts,
    // and signed integer limits. No large arrays are allocated.
    for (const int B : {1, 2, 7, 8, 16, 144, 224, 512, std::numeric_limits<int>::max()})
        for (const int nz : {1, 2, 7, 98, 288, std::numeric_limits<int>::max()})
            for (const bool channel : {false, true})
                for (const std::int64_t origin : {
                        minimum, minimum + 1, minimum + std::numeric_limits<int>::max(),
                        -static_cast<std::int64_t>(B) - 1, -static_cast<std::int64_t>(B),
                        -static_cast<std::int64_t>(B) + 1, std::int64_t{-1}, std::int64_t{0},
                        std::int64_t{1}, static_cast<std::int64_t>(nz) - B - 1,
                        static_cast<std::int64_t>(nz) - B,
                        static_cast<std::int64_t>(nz) - 1, static_cast<std::int64_t>(nz),
                        static_cast<std::int64_t>(nz) + 1,
                        maximum - std::numeric_limits<int>::max(), maximum - 1, maximum})
                    check_endpoints(origin, B, nz, channel);

    for (const auto dimensions : {std::array<int, 2>{0, 288}, {224, 0}, {-1, 288}, {224, -1}}) {
        const auto map = pf3d::BoundedZMap::make(0, dimensions[0], dimensions[1], true);
        int untouched = sentinel;
        if (map.lo != map.hi || map.aggregate(0, &untouched)
            || map.fetch(-1, &untouched) || map.fetch(0, &untouched)
            || untouched != sentinel) return 1;
    }

    std::cout << "PASS bounded-z oracle: " << fetch_cases << " arbitrary-offset fetches, "
              << aggregate_cases << " in-brick aggregate mappings; substrate/channel; "
              << "integer extremes and unchanged failure outputs\n";
}
