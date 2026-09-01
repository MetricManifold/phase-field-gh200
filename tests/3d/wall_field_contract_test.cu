#include "pf3d/kernels.cuh"

#include <cstdio>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[FAIL] %s\n", message);
        ++failures;
    }
}

pf3d::SLayout3D layout(std::uint32_t boundary) {
    pf3d::SLayout3D result{};
    result.nx = 64;
    result.ny = 64;
    result.nz = 48;
    result.pitch_x = 64;
    result.boundary_flags = boundary;
    return result;
}

void wall_field_contract() {
    float sample = 0.0f;
    const pf3d::SLayout3D periodic =
        layout(pf3d::kBoundaryPeriodicXYZ3D);
    const pf3d::SLayout3D slab =
        layout(pf3d::kBoundarySubstrateSlab3D);
    const pf3d::SLayout3D channel =
        layout(pf3d::kBoundaryHardWallChannel3D);

    expect(pf3d::valid_weighted_wall_field({}, periodic),
           "periodic geometry accepts an absent wall field");
    expect(pf3d::valid_weighted_wall_field({}, slab),
           "substrate geometry accepts an absent wall field");
    expect(!pf3d::valid_weighted_wall_field({&sample, periodic.nz}, periodic),
           "periodic geometry rejects a wall field");
    expect(!pf3d::valid_weighted_wall_field({&sample, slab.nz}, slab),
           "substrate geometry rejects a wall field");

    expect(pf3d::valid_weighted_wall_field({&sample, channel.nz}, channel),
           "channel geometry accepts an exact wall field");
    expect(!pf3d::valid_weighted_wall_field({}, channel),
           "channel geometry rejects an absent wall field");
    expect(!pf3d::valid_weighted_wall_field({nullptr, channel.nz}, channel),
           "channel geometry rejects a null wall pointer");
    expect(!pf3d::valid_weighted_wall_field({&sample, 0}, channel),
           "channel geometry rejects an empty wall field");
    expect(!pf3d::valid_weighted_wall_field({&sample, channel.nz - 1}, channel),
           "channel geometry rejects a short wall field");
    expect(!pf3d::valid_weighted_wall_field({&sample, channel.nz + 1}, channel),
           "channel geometry rejects a long wall field");
}

}  // namespace

int main() {
    wall_field_contract();
    if (failures == 0)
        std::printf("wall_field_contract: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
