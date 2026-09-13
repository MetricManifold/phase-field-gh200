#include "pf3d/brick_classes.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

std::size_t index(int x, int y, int z, int edge) {
    return (static_cast<std::size_t>(z) * edge + y) * edge + x;
}

void growth_policy() {
    using pf3d::next_brick_edge;
    expect(next_brick_edge(144, 512) == 176, "144 grows to 176");
    expect(next_brick_edge(176, 512) == 208, "176 grows to 208");
    expect(next_brick_edge(208, 512) == 240, "208 grows to 240");
    expect(next_brick_edge(152, 512) == 192, "non-tier base rounds upward");
    expect(next_brick_edge(224, 250) == 248, "last tier respects aligned domain limit");
    expect(next_brick_edge(224, 231) == 0, "no aligned growth remains");
    expect(next_brick_edge(224, 224) == 0, "maximum-sized brick cannot grow");
    expect(next_brick_edge(145, 512) == 0, "misaligned current edge rejected");
    expect(next_brick_edge(0, 512) == 0, "zero current edge rejected");
    constexpr int maximum = std::numeric_limits<int>::max();
    expect(next_brick_edge(maximum - 31, maximum) == maximum - 7,
           "growth arithmetic does not overflow int");
    expect(next_brick_edge(maximum - 7, maximum) == 0, "aligned integer ceiling is terminal");
    expect(pf3d::kBrickCompactionEvery == 1024, "compaction has an accepted-step cadence");
}

void shrink_policy() {
    const int small_lo[3] = {80, 80, 80}, small_hi[3] = {143, 143, 143};
    expect(pf3d::shrink_brick_edge(224, 144, small_lo, small_hi) == 144,
           "compact support can return to the base");
    const int medium_lo[3] = {32, 32, 32}, medium_hi[3] = {191, 191, 191};
    expect(pf3d::shrink_brick_edge(224, 144, medium_lo, medium_hi) == 192,
           "all support faces retain sixteen spare planes");
    const int offcentre_lo[3] = {8, 80, 80}, offcentre_hi[3] = {100, 143, 143};
    expect(pf3d::shrink_brick_edge(224, 144, offcentre_lo, offcentre_hi) == 224,
           "small but off-center support cannot be clipped by a centered crop");
    const int aligned_lo[3] = {64, 64, 64}, aligned_hi[3] = {159, 159, 159};
    expect(pf3d::shrink_brick_edge(224, 152, aligned_lo, aligned_hi) == 152,
           "minimum base need not be a multiple of sixteen");
    const int empty_hi[3] = {-1, -1, -1};
    expect(pf3d::shrink_brick_edge(224, 144, small_lo, empty_hi) == 224,
           "empty support metadata does not authorize shrinking");
    expect(pf3d::shrink_brick_edge(224, 144, nullptr, small_hi) == 224,
           "missing support bounds do not authorize shrinking");
    constexpr int largest = std::numeric_limits<int>::max() - 7;
    const int extreme_lo[3] = {17, 17, 17};
    const int extreme_hi[3] = {largest - 18, largest - 18, largest - 18};
    expect(pf3d::shrink_brick_edge(largest, 144, extreme_lo, extreme_hi) == largest,
           "rounding above the integer ceiling cannot create a negative crop");

    for (int current : {24, 32, 144, 152, 176, 224, 256})
        for (int minimum : {8, 24, 144, 152}) {
            if (minimum > current) continue;
            for (int shell = 0; shell < current / 2; ++shell) {
                const int lo[3] = {shell, shell, shell};
                const int hi[3] = {current - 1 - shell, current - 1 - shell,
                                   current - 1 - shell};
                const int candidate = pf3d::shrink_brick_edge(current, minimum, lo, hi);
                expect(candidate >= minimum && candidate <= current
                           && pf3d::valid_brick_class_edge(candidate),
                       "candidate is aligned and bounded");
                if (candidate == current) continue;
                const int removed = (current - candidate) / 2;
                expect(lo[0] - removed >= pf3d::kBrickShrinkMargin
                           && hi[0] - removed < candidate - pf3d::kBrickShrinkMargin,
                       "centered crop retains the required spare margin");
            }
        }
}

void resize_coordinates() {
    int offset = 91;
    expect(pf3d::centered_resize_offset(144, 176, &offset) && offset == 16,
           "growth has the centered positive local offset");
    expect(pf3d::centered_resize_offset(176, 144, &offset) && offset == -16,
           "shrink has the centered negative local offset");
    expect(!pf3d::centered_resize_offset(145, 176, &offset), "misaligned resize rejected");
    constexpr auto low = std::numeric_limits<std::int64_t>::min();
    constexpr auto high = std::numeric_limits<std::int64_t>::max();
    std::int64_t result = 91;
    expect(!pf3d::checked_resize_origin(low, 144, 176, &result) && result == 91,
           "growth rejects origin underflow without changing output");
    expect(pf3d::checked_resize_origin(low + 16, 144, 176, &result) && result == low,
           "lowest representable grown origin accepted");
    result = 91;
    expect(!pf3d::checked_resize_origin(high, 176, 144, &result) && result == 91,
           "shrink rejects origin overflow without changing output");
    expect(pf3d::checked_resize_origin(high - 16, 176, 144, &result) && result == high,
           "highest representable shrunken origin accepted");
    for (int old_edge : {24, 144, 152, 224})
        for (int new_edge : {24, 144, 152, 224}) {
            pf3d::centered_resize_offset(old_edge, new_edge, &offset);
            for (std::int64_t origin : {-1000, -1, 0, 1000}) {
                expect(pf3d::checked_resize_origin(origin, old_edge, new_edge, &result),
                       "ordinary origin rebasing succeeds");
                for (int local = 0; local < old_edge; ++local) {
                    const int shifted = local + offset;
                    if (shifted < 0 || shifted >= new_edge) continue;
                    expect(origin + local == result + shifted,
                           "retained voxel world coordinate is invariant");
                }
            }
        }
}

void exact_zero_shell() {
    constexpr int old_edge = 32, new_edge = 16;
    std::vector<float> field(old_edge * old_edge * old_edge, 0.0f);
    field[index(16, 16, 16, old_edge)] = 1.0f;
    const auto eligible = [&] {
        return pf3d::zero_loss_centered_crop(field.data(), field.size(), old_edge, new_edge);
    };
    expect(eligible(), "nonzero retained interior and zero shell are eligible");
    const std::array<std::array<int, 3>, 6> faces = {{
        {0, 16, 16}, {31, 16, 16}, {16, 0, 16},
        {16, 31, 16}, {16, 16, 0}, {16, 16, 31}}};
    const std::array<std::uint32_t, 8> patterns = {
        1u, 0x80000001u, 0x33800000u, 0xb3800000u,
        0x7f800000u, 0xff800000u, 0x7fc00001u, 0x7f800001u};
    for (const auto& xyz : faces) {
        float& voxel = field[index(xyz[0], xyz[1], xyz[2], old_edge)];
        for (const auto bits : patterns) {
            std::memcpy(&voxel, &bits, sizeof(bits));
            expect(!eligible(), "every discarded nonzero/subnormal/nonfinite voxel blocks shrinking");
        }
        const std::uint32_t negative_zero = 0x80000000u;
        std::memcpy(&voxel, &negative_zero, sizeof(negative_zero));
        expect(eligible(), "signed zero is an empty shell voxel");
        voxel = 0.0f;
    }
    expect(!pf3d::zero_loss_centered_crop(nullptr, field.size(), old_edge, new_edge),
           "missing phase field rejected");
    expect(!pf3d::zero_loss_centered_crop(field.data(), field.size() - 1, old_edge, new_edge),
           "truncated phase field rejected before reading");
    expect(!pf3d::zero_loss_centered_crop(field.data(), field.size(), old_edge, 48),
           "crop oracle does not authorize growth");
    expect(!pf3d::zero_loss_centered_crop(field.data(), field.size(),
                                        std::numeric_limits<int>::max() - 7, 16),
           "unrepresentable cube byte count rejected before reading");
}

} // namespace

int main() {
    growth_policy();
    shrink_policy();
    resize_coordinates();
    exact_zero_shell();
    std::printf("%s brick classes: %d checks, %d failures\n",
                failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
