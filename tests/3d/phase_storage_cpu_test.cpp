#include "pf3d/phase_storage.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using pf3d::CellFieldPlaneRange3D;
using pf3d::CellFieldStorage3D;
constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
constexpr auto maximum = std::numeric_limits<std::int64_t>::max();

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL phase storage: " << message << '\n';
        std::exit(1);
    }
}

bool same_range(const CellFieldPlaneRange3D& a, const CellFieldPlaneRange3D& b) {
    return a.logical_begin == b.logical_begin &&
           a.stored_begin == b.stored_begin && a.count == b.count;
}

void check_small_layouts() {
    for (int cap = 0; cap <= 32; ++cap) {
        const CellFieldStorage3D storage{cap};
        for (int B = 1; B <= 32; ++B) {
            const int depth = cap > 0 && cap < B ? cap : B;
            require(storage.planes(B) == depth, "plane count");
            require(storage.compact(B) == (depth < B), "compact threshold");
            const std::size_t expected_words = static_cast<std::size_t>(B) * B * depth;
            require(storage.words(B) == expected_words, "word count");
            std::size_t bytes = 0;
            require(storage.checked_bytes(B, &bytes) &&
                    bytes == expected_words * sizeof(float), "byte count");
            for (std::int64_t origin = -48; origin <= 48; ++origin) {
                CellFieldPlaneRange3D actual{}, expected{};
                require(storage.retained_planes(B, origin, &actual), "valid range");
                for (int z = 0; z < B; ++z) {
                    const std::int64_t world = origin + z;
                    const bool retained = depth == B || (world >= 0 && world < cap);
                    if (!retained) continue;
                    const int stored_z = depth == B ? z : static_cast<int>(world);
                    if (expected.count == 0) {
                        expected.logical_begin = z;
                        expected.stored_begin = stored_z;
                    }
                    ++expected.count;
                    for (const int x : {0, B - 1}) {
                        for (const int y : {0, B - 1}) {
                            const std::size_t expected_index =
                                (static_cast<std::size_t>(stored_z) * B + y) * B + x;
                            const auto actual_index = storage.index(x, y, z, B, origin);
                            require(actual_index == expected_index &&
                                    actual_index < expected_words, "in-bounds address");
                        }
                    }
                }
                require(same_range(actual, expected), "retained interval oracle");
            }
        }
    }
}

void check_limits() {
    const CellFieldStorage3D compact{140}, cubic{};
    for (const std::int64_t origin : {minimum, minimum + 1, maximum - 1, maximum}) {
        CellFieldPlaneRange3D range{7, 8, 9};
        require(compact.retained_planes(160, origin, &range) &&
                same_range(range, {}), "extreme compact origin is empty");
        require(cubic.retained_planes(160, origin, &range) &&
                same_range(range, {0, 0, 160}), "cubic ignores world origin");
        require(cubic.index(159, 159, 159, 160, origin) == cubic.words(160) - 1,
                "cubic index does not add extreme origin");
    }
    const std::int64_t origins[] = {-160, -159, -20, 0, 139, 140};
    const CellFieldPlaneRange3D expected[] = {
        {}, {159, 0, 1}, {20, 0, 140}, {0, 0, 140}, {0, 139, 1}, {}};
    for (std::size_t i = 0; i < sizeof(origins) / sizeof(origins[0]); ++i) {
        CellFieldPlaneRange3D range{};
        require(compact.retained_planes(160, origins[i], &range) &&
                same_range(range, expected[i]), "intersection endpoints");
    }
    constexpr std::size_t untouched = 123;
    for (const CellFieldStorage3D storage : {CellFieldStorage3D{-1}, cubic, compact}) {
        for (const int B : {-1, 0}) {
            std::size_t out = untouched;
            CellFieldPlaneRange3D range{7, 8, 9};
            require(storage.planes(B) == 0 && storage.words(B) == 0 &&
                    !storage.checked_words(B, &out) &&
                    !storage.checked_bytes(B, &out) && out == untouched &&
                    !storage.retained_planes(B, 0, &range) &&
                    same_range(range, {7, 8, 9}), "invalid dimensions preserve outputs");
        }
    }
    std::size_t out = untouched;
    require(!CellFieldStorage3D{-1}.checked_words(160, &out) && out == untouched,
            "negative cap rejected");
    require(!compact.checked_words(160, nullptr) &&
            !compact.checked_bytes(160, nullptr) &&
            !compact.retained_planes(160, 0, nullptr), "null outputs rejected");
    const int largest = std::numeric_limits<int>::max();
    require(!cubic.checked_words(largest, &out) && out == untouched &&
            cubic.words(largest) == 0, "cube overflow rejected");
    if constexpr (sizeof(std::size_t) == 8) {
        require(CellFieldStorage3D{4}.checked_words(largest, &out),
                "large representable word count");
        out = untouched;
        require(!CellFieldStorage3D{4}.checked_bytes(largest, &out) && out == untouched,
                "float-byte overflow rejected");
        require(!CellFieldStorage3D{5}.checked_words(largest, &out) && out == untouched,
                "capped word overflow rejected");
    }
}

void check_resize(int old_edge, int new_edge, int cap) {
    const CellFieldStorage3D storage{cap};
    const int old_origin = (cap - old_edge) / 2;
    const int new_origin = (cap - new_edge) / 2;
    std::vector<float> old_field(storage.words(old_edge), 0.0f);
    std::vector<float> new_field(storage.words(new_edge), 0.0f);
    const int offset = (new_edge - old_edge) / 2;
    CellFieldPlaneRange3D old_range{}, new_range{};
    require(storage.retained_planes(old_edge, old_origin, &old_range) &&
            storage.retained_planes(new_edge, new_origin, &new_range), "resize ranges");
    const int first_world = std::max(old_origin + old_range.logical_begin,
                                     new_origin + new_range.logical_begin);
    const int end_world = std::min(old_origin + old_range.logical_begin + old_range.count,
                                   new_origin + new_range.logical_begin + new_range.count);
    // Populate a shared interior column with distinct exactly representable
    // values, then copy through the two layouts without changing world z.
    for (int world = first_world; world < end_world; ++world) {
        const int old_z = world - old_origin;
        const int new_z = world - new_origin;
        const int old_x = old_edge / 2, old_y = old_edge / 2;
        const int new_x = old_x + offset, new_y = old_y + offset;
        const float value = static_cast<float>(world + 100);
        const auto old_index = storage.index(old_x, old_y, old_z, old_edge, old_origin);
        const auto new_index = storage.index(new_x, new_y, new_z, new_edge, new_origin);
        old_field[old_index] = value;
        new_field[new_index] = old_field[old_index];
        require(new_z == old_z + offset && new_field[new_index] == value,
                "centered resize preserves physical voxel");
    }
    require(first_world < end_world, "resize exercises retained planes");
}

void check_packing() {
    constexpr int B = 24;
    const CellFieldStorage3D cube{}, storage{16};
    for (const int origin : {-30, -20, -4, 0, 7, 16}) {
        std::vector<float> source(cube.words(B), 0.0f);
        std::vector<float> packed(storage.words(B), 0.0f);
        std::vector<float> restored(cube.words(B), 0.0f);
        for (int z = 0; z < B; ++z) {
            if (origin + z < 0 || origin + z >= 16) continue;
            for (int y = 0; y < B; ++y)
                for (int x = 0; x < B; ++x) {
                    const auto q = cube.index(x, y, z, B, origin);
                    source[q] = static_cast<float>(q + 1);
                }
        }
        CellFieldPlaneRange3D range{};
        require(storage.retained_planes(B, origin, &range), "packing range");
        const std::size_t plane = B * B;
        std::copy_n(source.begin() + range.logical_begin * plane, range.count * plane,
                    packed.begin() + range.stored_begin * plane);
        std::copy_n(packed.begin() + range.stored_begin * plane, range.count * plane,
                    restored.begin() + range.logical_begin * plane);
        require(source == restored, "cubic checkpoint packing round trip");
        if (range.count > 1) {
            const int z = range.logical_begin + 1;
            require(storage.index(1, 1, z, B, origin) ==
                    storage.index(1, 1, z - 1, B, origin + 1),
                    "z recentering keeps the same stored world plane");
        }
    }
}

} // namespace

int main() {
    static_assert(CellFieldStorage3D{}.words(24) == 24u * 24u * 24u);
    static_assert(CellFieldStorage3D{16}.words(24) == 24u * 24u * 16u);
    static_assert(CellFieldStorage3D{16}.index(0, 0, 4, 24, -4) == 0);
    check_small_layouts();
    check_limits();
    check_packing();
    for (const int old_edge : {8, 16, 24, 32})
        for (const int new_edge : {8, 16, 24, 32})
            check_resize(old_edge, new_edge, 16);
    const std::size_t saved = 64u * 2u * sizeof(float) *
        (CellFieldStorage3D{}.words(160) - CellFieldStorage3D{140}.words(160));
    require(saved == 262144000u && saved / (1024u * 1024u) == 250u,
            "N64 two-buffer savings are 250 MiB");
    std::cout << "PASS phase storage: cubic/capped addressing, retained ranges, "
                 "integer limits, checked sizes, centered resize, and 250 MiB saving\n";
}
