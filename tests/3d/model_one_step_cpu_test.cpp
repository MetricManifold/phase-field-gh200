#include "pf3d/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace ref = pf3d::reference;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[FAIL] %s\n", message);
        ++failures;
    }
}

bool close(double a, double b, double abs_tolerance = 1.0e-11,
           double rel_tolerance = 1.0e-10) {
    return std::fabs(a - b) <=
        abs_tolerance + rel_tolerance * std::max(std::fabs(a), std::fabs(b));
}

ref::DenseState empty_state(int side, int cells) {
    ref::DenseState state;
    state.side = side;
    state.cells.resize(static_cast<std::size_t>(cells));
    const std::size_t words = static_cast<std::size_t>(side) * side * side;
    state.phi.assign(words * static_cast<std::size_t>(cells), 0.0);
    return state;
}

ref::SubstrateState empty_substrate_state(int side, int height, int cells) {
    ref::SubstrateState state;
    state.side = side;
    state.height = height;
    state.cells.resize(static_cast<std::size_t>(cells));
    const std::size_t words = static_cast<std::size_t>(side) * side * height;
    state.phi.assign(words * static_cast<std::size_t>(cells), 0.0);
    return state;
}

double volume_of_buffer(const std::vector<double>& phi, std::size_t begin,
                        std::size_t words) {
    double volume = 0.0;
    for (std::size_t q = 0; q < words; ++q)
        volume += phi[begin + q] * phi[begin + q];
    return volume;
}

void test_stencil() {
    ref::DenseState state = empty_state(9, 1);
    std::fill(state.phi.begin(), state.phi.end(), 3.25);
    expect(close(ref::laplacian_at(state, 0, 4, 4, 4), 0.0),
           "27-point stencil annihilates a constant");
    for (int z = 0; z < state.side; ++z)
        for (int y = 0; y < state.side; ++y)
            for (int x = 0; x < state.side; ++x)
                state.phi[state.index(0, x, y, z)] =
                    static_cast<double>(x * x + y * y + z * z);
    expect(close(ref::laplacian_at(state, 0, 4, 4, 4), 6.0),
           "27-point stencil gives lap(x^2+y^2+z^2)=6");
}

void test_substrate_ghost_law() {
    ref::SubstrateState state = empty_substrate_state(5, 4, 1);
    for (int z = 0; z < state.height; ++z)
        for (int y = 0; y < state.side; ++y)
            for (int x = 0; x < state.side; ++x)
                state.phi[state.index(0, x, y, z)] =
                    static_cast<double>(100 * z + 10 * y + x);

    expect(ref::substrate_field_at(state, 0, 2, 3, -1) ==
               ref::substrate_field_at(state, 0, 2, 3, 0),
           "substrate ghost plane -1 reflects exactly onto plane 0");
    expect(ref::substrate_field_at(state, 0, -1, 2, 1) ==
               ref::substrate_field_at(state, 0, 4, 2, 1) &&
               ref::substrate_field_at(state, 0, 2, 5, 1) ==
               ref::substrate_field_at(state, 0, 2, 0, 1),
           "substrate slab remains periodic in x and y");
    expect(ref::substrate_field_at(state, 0, 2, 3, state.height) == 0.0,
           "substrate slab uses a zero upper ghost plane");

    std::fill(state.phi.begin(), state.phi.end(), 2.5);
    expect(ref::substrate_laplacian_at(state, 0, 2, 2, 0) == 0.0,
           "neutral bottom ghost preserves a constant stencil exactly");
}

void test_substrate_calibration_symmetry() {
    constexpr int edge = 24;
    constexpr int half_height = edge / 2;
    constexpr double radius = 4.0;
    constexpr double lambda = 2.0;
    const ref::SeedRadiusCalibration full =
        ref::calibrate_seed_radius(radius, lambda, edge, 1.0e-11, 80);
    const ref::SeedRadiusCalibration slab =
        ref::calibrate_substrate_seed_radius(
            radius, lambda, edge, 1.0e-11, 80);
    expect(full.converged && slab.converged &&
               slab.seed_radius == full.seed_radius &&
               slab.iterations == full.iterations &&
               slab.target == 0.5 * full.target,
           "neutral hemisphere and full sphere have the same calibrated radius");

    ref::DenseState sphere = empty_state(edge, 1);
    ref::SubstrateState hemisphere =
        empty_substrate_state(edge, half_height, 1);
    const double centre = 0.5 * static_cast<double>(edge - 1);
    ref::seed_periodic_sphere(&sphere, 0, {centre, centre, centre},
                              full.seed_radius, lambda);
    ref::seed_substrate_hemisphere(&hemisphere, 0, centre, centre,
                                   full.seed_radius, lambda);

    double lower_volume = 0.0;
    double upper_volume = 0.0;
    bool profiles_match = true;
    for (int z = 0; z < half_height; ++z) {
        const int lower_z = half_height - 1 - z;
        const int upper_z = half_height + z;
        for (int y = 0; y < edge; ++y) {
            for (int x = 0; x < edge; ++x) {
                const double lower = sphere.phi[
                    sphere.index(0, x, y, lower_z)];
                const double upper = sphere.phi[
                    sphere.index(0, x, y, upper_z)];
                const double retained = hemisphere.phi[
                    hemisphere.index(0, x, y, z)];
                profiles_match = profiles_match &&
                    lower == upper && upper == retained;
                lower_volume += lower * lower;
                upper_volume += upper * upper;
            }
        }
    }
    expect(profiles_match,
           "hemisphere voxels are exact mirrors about the face z=-1/2");
    expect(close(lower_volume, upper_volume, 1.0e-13, 1.0e-13) &&
               close(upper_volume,
                     ref::substrate_volume_of(hemisphere, 0),
                     1.0e-13, 1.0e-13),
           "the mirrored half-volumes agree within reduction roundoff");
    expect(close(upper_volume, slab.measured_volume, 1.0e-13, 1.0e-13) &&
               slab.relative_error <= 1.0e-11,
           "hemisphere calibration reaches the half-sphere target volume");
}

void test_substrate_one_step() {
    constexpr int side = 24;
    constexpr int height = 12;
    constexpr double radius = 4.0;
    constexpr double lambda = 2.0;
    const ref::SeedRadiusCalibration calibrated =
        ref::calibrate_substrate_seed_radius(
            radius, lambda, side, 1.0e-11, 80);
    ref::SubstrateState state = empty_substrate_state(side, height, 2);
    state.cells[0].active_speed = 0.2;
    state.cells[0].polarity = {0.6, 0.8, 9.0};
    state.cells[1].active_speed = 0.0;
    ref::seed_substrate_hemisphere(
        &state, 0, 9.5, 11.5, calibrated.seed_radius, lambda);
    ref::seed_substrate_hemisphere(
        &state, 1, 14.5, 11.5, calibrated.seed_radius, lambda);

    ref::ModelParameters model;
    model.dt = 1.0e-4;
    model.lambda = lambda;
    model.target_radius = radius;
    model.kappa = 10.0;
    model.mu = 1.0;
    const ref::OneStepResult step =
        ref::one_euler_step_substrate(state, model);
    expect(step.velocity[0].z == 0.0 && step.velocity[1].z == 0.0,
           "substrate motion has exactly zero normal velocity");
    expect(close(step.velocity[0].x,
                 0.2 * state.cells[0].polarity.x +
                 60.0 * model.kappa /
                     (model.xi * model.lambda * model.lambda) *
                     step.interaction_integral[0].x),
           "substrate motion retains active and interaction velocity in-plane");
    expect(std::fabs(step.interaction_integral[0].z) > 1.0e-12,
           "one-step gate exercises a discarded normal interaction force");
    expect(step.phi_next.size() == state.phi.size() &&
               std::all_of(step.phi_next.begin(), step.phi_next.end(),
                           [](double value) { return std::isfinite(value); }),
           "substrate one-step reference produces a finite complete field");
    for (int cell = 0; cell < 2; ++cell) {
        const std::size_t begin = static_cast<std::size_t>(cell)
                                * state.domain_words();
        bool changed = false;
        for (std::size_t q = 0; q < state.domain_words(); ++q)
            changed = changed || step.phi_next[begin + q] != state.phi[begin + q];
        expect(changed, "substrate one-step reference advances every cell");
    }
}

void test_hard_wall_channel_one_step() {
    constexpr int side = 24;
    constexpr int height = 12;
    constexpr double radius = 4.0;
    constexpr double lambda = 2.0;
    ref::SubstrateState state = empty_substrate_state(side, height, 1);
    state.neutral_top = true;
    state.three_dimensional_motion = true;
    state.cells[0].active_speed = 0.2;
    state.cells[0].polarity = {0.0, 0.0, 1.0};
    ref::seed_channel_sphere(
        &state, 0, {11.5, 11.5, 5.5}, radius, lambda);

    expect(ref::substrate_field_at(state, 0, 11, 11, -1) ==
               ref::substrate_field_at(state, 0, 11, 11, 0) &&
           ref::substrate_field_at(state, 0, 11, 11, height) ==
               ref::substrate_field_at(state, 0, 11, 11, height - 1),
           "channel reflects the phase field at both wall faces");

    ref::ModelParameters model;
    model.dt = 1.0e-4;
    model.lambda = lambda;
    model.target_radius = radius;
    const ref::OneStepResult step =
        ref::one_euler_step_substrate(state, model);
    expect(close(step.velocity[0].z, 0.2),
           "channel retains active motion normal to the walls");
    expect(step.phi_next.size() == state.phi.size() &&
               std::all_of(step.phi_next.begin(), step.phi_next.end(),
                           [](double value) { return std::isfinite(value); }),
           "channel one-step reference produces a finite complete field");
    expect(!std::equal(step.phi_next.begin(), step.phi_next.end(),
                       state.phi.begin()),
           "channel one-step reference advances the field");

    // Add the resolved static wall using the same half-grid coordinates as
    // production. A centered cell has no preferred wall-normal direction.
    constexpr double lower_wall = 3.0;
    constexpr double upper_wall = 9.0;
    const double profile_k = std::sqrt(7.5) / lambda;
    state.wall_psi_sq.resize(height);
    for (int z = 0; z < height; ++z) {
        const double coordinate = static_cast<double>(z) + 0.5;
        const double fluid = 0.25 *
            (1.0 + std::tanh(profile_k * (coordinate - lower_wall))) *
            (1.0 - std::tanh(profile_k * (coordinate - upper_wall)));
        const double wall = 1.0 - fluid;
        state.wall_psi_sq[static_cast<std::size_t>(z)] = wall * wall;
    }
    model.wall_kappa = model.kappa;
    const ref::OneStepResult repulsive =
        ref::one_euler_step_substrate(state, model);
    expect(close(repulsive.velocity[0].z, step.velocity[0].z,
                 1.0e-11, 1.0e-10),
           "symmetric resolved walls exert no net force at the midplane");
    expect(!std::equal(repulsive.phi_next.begin(), repulsive.phi_next.end(),
                       step.phi_next.begin()),
           "resolved wall enters the Allen--Cahn update");

    auto passive_wall_velocity = [&](double physical_centre_z) {
        ref::SubstrateState shifted = empty_substrate_state(side, height, 1);
        shifted.neutral_top = true;
        shifted.three_dimensional_motion = true;
        shifted.wall_psi_sq = state.wall_psi_sq;
        ref::seed_channel_sphere(
            &shifted, 0,
            {11.5, 11.5, physical_centre_z - 0.5},
            2.0, lambda);
        ref::ModelParameters shifted_model = model;
        shifted_model.target_radius = 2.0;
        return ref::one_euler_step_substrate(
            shifted, shifted_model).velocity[0].z;
    };
    const double lower_velocity = passive_wall_velocity(4.0);
    const double middle_velocity = passive_wall_velocity(6.0);
    const double upper_velocity = passive_wall_velocity(8.0);
    expect(lower_velocity > 1.0e-9 && upper_velocity < -1.0e-9,
           "lower and upper walls drive passive cells into the slit");
    expect(close(lower_velocity, -upper_velocity, 1.0e-11, 1.0e-9) &&
               std::fabs(middle_velocity) <= 1.0e-11,
           "the two-wall force is antisymmetric about the channel midplane");
}

void test_calibration_and_isolated_cell() {
    constexpr int side = 24;
    constexpr double radius = 4.0, lambda = 2.0;
    const ref::SeedRadiusCalibration calibrated =
        ref::calibrate_seed_radius(radius, lambda, side, 1.0e-11, 80);
    expect(calibrated.converged && calibrated.relative_error <= 1.0e-11,
           "seed-radius bisection reaches its requested volume tolerance");
    expect(calibrated.seed_radius > radius,
           "phi^2 volume calibration shifts the half-height radius outward");

    ref::DenseState state = empty_state(side, 1);
    state.cells[0].active_speed = 0.0;
    ref::seed_periodic_sphere(&state, 0, {11.5, 11.5, 11.5},
                              calibrated.seed_radius, lambda);
    ref::ModelParameters model;
    model.dt = 1.0e-4;
    model.lambda = lambda;
    model.target_radius = radius;
    model.kappa = 10.0;
    model.mu = 1.0;
    const ref::OneStepResult step = ref::one_euler_step(state, model);
    expect(close(step.velocity[0].x, 0.0) && close(step.velocity[0].y, 0.0) &&
               close(step.velocity[0].z, 0.0),
           "an isolated passive sphere has zero velocity");
    expect(std::isfinite(step.volume[0]) && step.volume[0] > 0.0,
           "isolated sphere has finite positive volume");

    // Cubic symmetry is exact for a sphere centered at half-grid coordinates.
    const std::size_t qx = state.index(0, 14, 12, 12);
    const std::size_t qy = state.index(0, 12, 14, 12);
    const std::size_t qz = state.index(0, 12, 12, 14);
    expect(close(step.phi_next[qx], step.phi_next[qy], 1.0e-13, 1.0e-13) &&
               close(step.phi_next[qy], step.phi_next[qz], 1.0e-13, 1.0e-13),
           "isolated one-step update preserves axis symmetry");

    // With surface and overlap terms disabled, the volume penalty alone moves
    // both under- and over-filled profiles toward V0.
    model.kappa = 0.0;
    model.mu = 1.0;
    state.cells[0].gamma = 0.0;
    ref::seed_periodic_sphere(&state, 0, {11.5, 11.5, 11.5},
                              calibrated.seed_radius - 0.5, lambda);
    const double under_before = ref::volume_of(state, 0);
    const ref::OneStepResult under = ref::one_euler_step(state, model);
    const double under_after = volume_of_buffer(under.phi_next, 0,
                                                state.domain_words());
    expect(under_before < ref::target_sphere_volume(radius) &&
               under_after > under_before,
           "volume penalty grows an under-filled isolated cell");
    ref::seed_periodic_sphere(&state, 0, {11.5, 11.5, 11.5},
                              calibrated.seed_radius + 0.5, lambda);
    const double over_before = ref::volume_of(state, 0);
    const ref::OneStepResult over = ref::one_euler_step(state, model);
    const double over_after = volume_of_buffer(over.phi_next, 0,
                                               state.domain_words());
    expect(over_before > ref::target_sphere_volume(radius) &&
               over_after < over_before,
           "volume penalty shrinks an over-filled isolated cell");
}

ref::OneStepResult pair_step(ref::Vector3 displacement,
                             ref::DenseState* state_out = nullptr) {
    constexpr int side = 28;
    constexpr double radius = 5.0, lambda = 2.0;
    const auto calibrated = ref::calibrate_seed_radius(radius, lambda, side,
                                                        1.0e-10, 70);
    ref::DenseState state = empty_state(side, 2);
    state.cells[0].active_speed = state.cells[1].active_speed = 0.0;
    const ref::Vector3 first{10.0, 10.0, 10.0};
    const ref::Vector3 second{first.x + displacement.x,
                              first.y + displacement.y,
                              first.z + displacement.z};
    ref::seed_periodic_sphere(&state, 0, first, calibrated.seed_radius, lambda);
    ref::seed_periodic_sphere(&state, 1, second, calibrated.seed_radius, lambda);
    ref::ModelParameters model;
    model.dt = 1.0e-4;
    model.lambda = lambda;
    model.target_radius = radius;
    model.kappa = 10.0;
    model.mu = 1.0;
    if (state_out) *state_out = state;
    return ref::one_euler_step(state, model);
}

void test_pair_repulsion_and_permutations() {
    const ref::OneStepResult x = pair_step({7.0, 0.0, 0.0});
    const ref::OneStepResult y = pair_step({0.0, 7.0, 0.0});
    const ref::OneStepResult z = pair_step({0.0, 0.0, 7.0});
    expect(x.velocity[0].x < 0.0 && x.velocity[1].x > 0.0,
           "overlap repels both cells along x");
    expect(y.velocity[0].y < 0.0 && y.velocity[1].y > 0.0,
           "overlap repels both cells along y");
    expect(z.velocity[0].z < 0.0 && z.velocity[1].z > 0.0,
           "overlap repels both cells along z");
    expect(close(x.velocity[0].x, y.velocity[0].y, 1.0e-10, 1.0e-9) &&
               close(y.velocity[0].y, z.velocity[0].z, 1.0e-10, 1.0e-9),
           "axis-permuted pairs have equal repulsive velocity");
    expect(std::fabs(x.velocity[0].y) < 1.0e-12 &&
               std::fabs(x.velocity[0].z) < 1.0e-12,
           "axis-aligned pair has no transverse drift");

    const ref::OneStepResult diagonal = pair_step({4.0, 4.0, 4.0});
    expect(diagonal.velocity[0].x < 0.0 && diagonal.velocity[0].y < 0.0 &&
               diagonal.velocity[0].z < 0.0 &&
               diagonal.velocity[1].x > 0.0 && diagonal.velocity[1].y > 0.0 &&
               diagonal.velocity[1].z > 0.0,
           "body-diagonal overlap repels in every component");
    expect(close(diagonal.velocity[0].x, diagonal.velocity[0].y,
                 1.0e-10, 1.0e-9) &&
               close(diagonal.velocity[0].y, diagonal.velocity[0].z,
                     1.0e-10, 1.0e-9),
           "body-diagonal repulsion respects permutation symmetry");
}

void test_no_overlap_and_exact_subtraction() {
    ref::DenseState state = empty_state(9, 2);
    state.cells[0].active_speed = state.cells[1].active_speed = 0.0;
    state.phi[state.index(0, 2, 2, 2)] = 1.0;
    state.phi[state.index(1, 6, 6, 6)] = 1.0;
    ref::ModelParameters model;
    model.dt = 1.0e-4;
    model.target_radius = 1.0;
    const ref::OneStepResult step = ref::one_euler_step(state, model);
    for (int cell = 0; cell < 2; ++cell) {
        expect(step.interaction_integral[static_cast<std::size_t>(cell)].x == 0.0 &&
                   step.interaction_integral[static_cast<std::size_t>(cell)].y == 0.0 &&
                   step.interaction_integral[static_cast<std::size_t>(cell)].z == 0.0,
               "disjoint compact fields have exactly zero interaction");
    }
    const std::size_t own = ref::domain_index(9, 2, 2, 2);
    expect(ref::other_field(step.aggregate_q[own], 1.0) == 0.0,
           "Q5.27 self-subtraction is exact");
}

void test_periodic_translation() {
    ref::DenseState original;
    const ref::OneStepResult a = pair_step({7.0, 0.0, 0.0}, &original);
    ref::DenseState shifted = empty_state(original.side, 2);
    shifted.cells = original.cells;
    constexpr int sx = 9, sy = -6, sz = 5;
    for (int cell = 0; cell < 2; ++cell)
        for (int z = 0; z < original.side; ++z)
            for (int y = 0; y < original.side; ++y)
                for (int x = 0; x < original.side; ++x)
                    shifted.phi[shifted.index(cell,
                        ref::wrap(x + sx, original.side),
                        ref::wrap(y + sy, original.side),
                        ref::wrap(z + sz, original.side))] =
                        original.phi[original.index(cell, x, y, z)];
    ref::ModelParameters model;
    model.dt = 1.0e-4;
    model.lambda = 2.0;
    model.target_radius = 5.0;
    model.kappa = 10.0;
    model.mu = 1.0;
    const ref::OneStepResult b = ref::one_euler_step(shifted, model);
    for (int cell = 0; cell < 2; ++cell) {
        expect(close(a.velocity[static_cast<std::size_t>(cell)].x,
                     b.velocity[static_cast<std::size_t>(cell)].x) &&
                   close(a.velocity[static_cast<std::size_t>(cell)].y,
                         b.velocity[static_cast<std::size_t>(cell)].y) &&
                   close(a.velocity[static_cast<std::size_t>(cell)].z,
                         b.velocity[static_cast<std::size_t>(cell)].z),
               "periodic translation preserves cell velocity");
        for (int z = 0; z < original.side; ++z)
            for (int y = 0; y < original.side; ++y)
                for (int x = 0; x < original.side; ++x) {
                    const double expected = a.phi_next[original.index(cell, x, y, z)];
                    const double actual = b.phi_next[shifted.index(cell,
                        ref::wrap(x + sx, original.side),
                        ref::wrap(y + sy, original.side),
                        ref::wrap(z + sz, original.side))];
                    if (!close(expected, actual, 2.0e-12, 2.0e-11)) {
                        expect(false, "periodic translation commutes with one Euler step");
                        return;
                    }
                }
    }
}

}  // namespace

int main() {
    test_stencil();
    test_substrate_ghost_law();
    test_substrate_calibration_symmetry();
    test_substrate_one_step();
    test_hard_wall_channel_one_step();
    test_calibration_and_isolated_cell();
    test_pair_repulsion_and_permutations();
    test_no_overlap_and_exact_subtraction();
    test_periodic_translation();
    if (failures != 0) {
        std::fprintf(stderr, "%d 3-D one-step CPU-reference test(s) failed\n",
                     failures);
        return 1;
    }
    std::printf("3-D one-step CPU-reference checks passed\n");
    return 0;
}
