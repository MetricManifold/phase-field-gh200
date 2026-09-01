#pragma once

// Independent, deterministic CPU evaluation of one three-dimensional model
// step. It favors explicit operation order and diagnostic clarity over speed.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pf3d::reference {

// Kept local rather than imported from the production parameter header so the
// independent reference can detect an accidental coefficient or stencil change.
constexpr double kReferencePi = 3.141592653589793238462643383279502884;
constexpr double kReferenceQScale = 134217728.0;  // 2^27
constexpr double kReferenceQInv = 1.0 / kReferenceQScale;

inline double target_sphere_volume(double radius) {
    return 4.0 * kReferencePi * radius * radius * radius / 3.0;
}

inline double target_slab_volume(double radius) {
    return 0.5 * target_sphere_volume(radius);
}

inline bool checked_multiply(std::size_t a, std::size_t b, std::size_t* out) {
    if (!out || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

inline bool checked_cube(std::size_t edge, std::size_t* out) {
    std::size_t square = 0;
    return checked_multiply(edge, edge, &square) &&
           checked_multiply(square, edge, out);
}

struct Vector3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

struct CellParameters {
    double gamma = 1.0;
    double active_speed = 0.0;
    Vector3 polarity{1.0, 0.0, 0.0};
};

struct ModelParameters {
    double dt = 0.01;
    double lambda = 7.0;
    double target_radius = 49.0;
    double kappa = 10.0;
    double wall_kappa = 0.0;
    double mu = 1.0;
    double xi = 1500.0;
};

// Cell-major, x-fastest full-domain fields. Full periodic fields keep the
// reference independent of production brick allocation and recentering.
struct DenseState {
    int side = 0;
    std::vector<CellParameters> cells;
    std::vector<double> phi;

    std::size_t domain_words() const {
        if (side <= 0) return 0;
        std::size_t words = 0;
        return checked_cube(static_cast<std::size_t>(side), &words)
            ? words : 0;
    }

    std::size_t index(int cell, int x, int y, int z) const {
        return static_cast<std::size_t>(cell) * domain_words()
             + (static_cast<std::size_t>(z) * static_cast<std::size_t>(side)
                + static_cast<std::size_t>(y)) * static_cast<std::size_t>(side)
             + static_cast<std::size_t>(x);
    }
};

// Independent dense CPU reference for a bounded-z domain periodic in x and y.
// By default, the lower boundary is neutral and the upper boundary is the
// zero-field far boundary used by the sessile-cap substrate. Tests may make
// the upper boundary neutral as well. A nonempty wall_psi_sq adds resolved
// steric-wall coupling.
struct SubstrateState {
    int side = 0;
    int height = 0;
    bool neutral_top = false;
    bool three_dimensional_motion = false;
    std::vector<CellParameters> cells;
    std::vector<double> phi;
    // Optional psi_w(z)^2 for a resolved steric-wall channel. An empty vector
    // retains the neutral-lower-boundary reference problem.
    std::vector<double> wall_psi_sq;

    std::size_t domain_words() const {
        if (side <= 0 || height <= 0) return 0;
        std::size_t area = 0, words = 0;
        return checked_multiply(static_cast<std::size_t>(side),
                                static_cast<std::size_t>(side), &area) &&
               checked_multiply(area, static_cast<std::size_t>(height), &words)
            ? words : 0;
    }

    std::size_t index(int cell, int x, int y, int z) const {
        return static_cast<std::size_t>(cell) * domain_words()
             + (static_cast<std::size_t>(z) * static_cast<std::size_t>(side)
                + static_cast<std::size_t>(y)) * static_cast<std::size_t>(side)
             + static_cast<std::size_t>(x);
    }
};

struct OneStepResult {
    std::vector<double> phi_next;
    std::vector<uint32_t> aggregate_q;
    std::vector<double> volume;
    std::vector<Vector3> interaction_integral;
    std::vector<Vector3> velocity;
};

inline int wrap(int coordinate, int side) {
    coordinate %= side;
    return coordinate < 0 ? coordinate + side : coordinate;
}

inline std::size_t domain_index(int side, int x, int y, int z) {
    x = wrap(x, side);
    y = wrap(y, side);
    z = wrap(z, side);
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(side)
            + static_cast<std::size_t>(y)) * static_cast<std::size_t>(side)
         + static_cast<std::size_t>(x);
}

// Round a non-negative Q5.27 value to nearest, ties to even, independently of
// the host floating-point rounding mode.
inline uint32_t quantize_phi_sq(double phi) {
    double square = phi * phi;
    if (square > 4.0) square = 4.0;
    const double scaled = square * kReferenceQScale;
    uint64_t rounded = static_cast<uint64_t>(std::floor(scaled));
    const double fraction = scaled - static_cast<double>(rounded);
    if (fraction > 0.5 || (fraction == 0.5 && (rounded & 1u) != 0u))
        ++rounded;
    return static_cast<uint32_t>(rounded);
}

inline void validate_state(const DenseState& state) {
    if (state.side < 3 || state.cells.empty())
        throw std::invalid_argument("the CPU reference requires cells on a side >= 3");
    const std::size_t words = state.domain_words();
    std::size_t expected = 0;
    if (words == 0 ||
        !checked_multiply(words, state.cells.size(), &expected) ||
        state.phi.size() != expected)
        throw std::invalid_argument("CPU reference phase-field dimensions are invalid");
    for (double value : state.phi)
        if (!std::isfinite(value))
            throw std::invalid_argument("CPU reference phase field is non-finite");
}

inline void validate_model(const ModelParameters& model) {
    if (!(model.dt > 0.0) || !(model.lambda > 0.0) ||
        !(model.target_radius > 0.0) || model.kappa < 0.0 ||
        model.wall_kappa < 0.0 || model.mu < 0.0 ||
        !(model.xi > 0.0))
        throw std::invalid_argument("CPU reference model parameters are invalid");
    if (model.wall_kappa > 0.0 && !(model.kappa > 0.0))
        throw std::invalid_argument(
            "CPU wall reference requires positive cell-cell kappa");
    const double wall_reaction =
        60.0 * model.wall_kappa / (model.lambda * model.lambda);
    if (!std::isfinite(wall_reaction) ||
        model.dt * wall_reaction >= 1.0)
        throw std::invalid_argument(
            "CPU wall reference violates the explicit reaction bound");
}

inline double field_at(const DenseState& state, int cell,
                       int x, int y, int z) {
    const std::size_t local = domain_index(state.side, x, y, z);
    return state.phi[static_cast<std::size_t>(cell) * state.domain_words() + local];
}

inline int stencil_weight(int dx, int dy, int dz) {
    const int distance = (dx != 0) + (dy != 0) + (dz != 0);
    if (distance == 0) return -128;
    if (distance == 1) return 14;
    if (distance == 2) return 3;
    return 1;
}

// The nested z/y/x order and one accumulator define the reference operation order.
inline double laplacian_at(const DenseState& state, int cell,
                           int x, int y, int z) {
    double weighted = 0.0;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                weighted += static_cast<double>(stencil_weight(dx, dy, dz))
                          * field_at(state, cell, x + dx, y + dy, z + dz);
    return weighted / 30.0;
}

inline Vector3 gradient_at(const DenseState& state, int cell,
                           int x, int y, int z) {
    Vector3 gradient;
    gradient.x = 0.5 * (field_at(state, cell, x + 1, y, z)
                      - field_at(state, cell, x - 1, y, z));
    gradient.y = 0.5 * (field_at(state, cell, x, y + 1, z)
                      - field_at(state, cell, x, y - 1, z));
    gradient.z = 0.5 * (field_at(state, cell, x, y, z + 1)
                      - field_at(state, cell, x, y, z - 1));
    return gradient;
}

inline std::vector<uint32_t> aggregate_field_q(const DenseState& state) {
    validate_state(state);
    const std::size_t words = state.domain_words();
    std::vector<uint32_t> aggregate(words, 0u);
    for (std::size_t cell = 0; cell < state.cells.size(); ++cell) {
        const std::size_t base = cell * words;
        for (std::size_t voxel = 0; voxel < words; ++voxel) {
            const uint32_t value = quantize_phi_sq(state.phi[base + voxel]);
            if (value > std::numeric_limits<uint32_t>::max() - aggregate[voxel])
                throw std::overflow_error("CPU reference Q5.27 aggregate overflow");
            aggregate[voxel] += value;
        }
    }
    return aggregate;
}

inline double other_field(uint32_t aggregate, double own_phi) {
    const uint32_t own = quantize_phi_sq(own_phi);
    if (aggregate < own)
        throw std::logic_error("CPU reference exact self-subtraction failed");
    return static_cast<double>(aggregate - own) * kReferenceQInv;
}

inline double volume_of(const DenseState& state, int cell) {
    const std::size_t words = state.domain_words();
    const std::size_t base = static_cast<std::size_t>(cell) * words;
    double volume = 0.0;
    for (std::size_t voxel = 0; voxel < words; ++voxel) {
        const double value = state.phi[base + voxel];
        volume += value * value;
    }
    return volume;
}

inline OneStepResult one_euler_step(const DenseState& state,
                                    const ModelParameters& model) {
    validate_state(state);
    validate_model(model);
    const int side = state.side;
    const int count = static_cast<int>(state.cells.size());
    const std::size_t words = state.domain_words();
    OneStepResult result;
    result.phi_next.assign(state.phi.size(), 0.0);
    result.aggregate_q = aggregate_field_q(state);
    result.volume.resize(state.cells.size());
    result.interaction_integral.resize(state.cells.size());
    result.velocity.resize(state.cells.size());

    for (int cell = 0; cell < count; ++cell)
        result.volume[static_cast<std::size_t>(cell)] = volume_of(state, cell);

    // Velocity is evaluated completely from phi^n before any cell is advanced.
    for (int cell = 0; cell < count; ++cell) {
        Vector3 integral{};
        for (int z = 0; z < side; ++z) {
            for (int y = 0; y < side; ++y) {
                for (int x = 0; x < side; ++x) {
                    const std::size_t local = domain_index(side, x, y, z);
                    const double phi = field_at(state, cell, x, y, z);
                    const double other = other_field(result.aggregate_q[local], phi);
                    const Vector3 gradient = gradient_at(state, cell, x, y, z);
                    integral.x += phi * gradient.x * other;
                    integral.y += phi * gradient.y * other;
                    integral.z += phi * gradient.z * other;
                }
            }
        }
        result.interaction_integral[static_cast<std::size_t>(cell)] = integral;
        const CellParameters& parameters = state.cells[static_cast<std::size_t>(cell)];
        const double coefficient = 60.0 * model.kappa
                                 / (model.xi * model.lambda * model.lambda);
        result.velocity[static_cast<std::size_t>(cell)] = Vector3{
            parameters.active_speed * parameters.polarity.x + coefficient * integral.x,
            parameters.active_speed * parameters.polarity.y + coefficient * integral.y,
            parameters.active_speed * parameters.polarity.z + coefficient * integral.z};
    }

    const double volume0 = target_sphere_volume(model.target_radius);
    const double volume_scale = 2.0 * model.mu / volume0;
    const double bulk_scale = 30.0 / (model.lambda * model.lambda);
    const double repulsion = 60.0 * model.kappa
                           / (model.lambda * model.lambda);
    for (int cell = 0; cell < count; ++cell) {
        const CellParameters& parameters = state.cells[static_cast<std::size_t>(cell)];
        const Vector3 velocity = result.velocity[static_cast<std::size_t>(cell)];
        const double volume_term = volume_scale
                                 * (volume0 - result.volume[static_cast<std::size_t>(cell)]);
        const double bulk = bulk_scale * parameters.gamma;
        const std::size_t base = static_cast<std::size_t>(cell) * words;
        for (int z = 0; z < side; ++z) {
            for (int y = 0; y < side; ++y) {
                for (int x = 0; x < side; ++x) {
                    const std::size_t local = domain_index(side, x, y, z);
                    const double phi = state.phi[base + local];
                    const double other = other_field(result.aggregate_q[local], phi);
                    const double laplacian = laplacian_at(state, cell, x, y, z);
                    const Vector3 gradient = gradient_at(state, cell, x, y, z);
                    const double rhs = parameters.gamma * laplacian
                        - bulk * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
                        + volume_term * phi
                        - repulsion * phi * other
                        - (velocity.x * gradient.x + velocity.y * gradient.y
                           + velocity.z * gradient.z);
                    result.phi_next[base + local] = phi + model.dt * rhs;
                }
            }
        }
    }
    return result;
}

inline void validate_substrate_state(const SubstrateState& state) {
    if (state.side < 3 || state.height < 2 || state.cells.empty())
        throw std::invalid_argument(
            "the substrate CPU reference requires side >= 3 and height >= 2");
    const std::size_t words = state.domain_words();
    std::size_t expected = 0;
    if (words == 0 ||
        !checked_multiply(words, state.cells.size(), &expected) ||
        state.phi.size() != expected)
        throw std::invalid_argument(
            "substrate CPU reference phase-field dimensions are invalid");
    for (double value : state.phi)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "substrate CPU reference phase field is non-finite");
    if (!state.wall_psi_sq.empty()) {
        if (state.wall_psi_sq.size() != static_cast<std::size_t>(state.height))
            throw std::invalid_argument(
                "substrate CPU wall field has the wrong height");
        for (double value : state.wall_psi_sq)
            if (!std::isfinite(value) || value < 0.0 || value > 1.0)
                throw std::invalid_argument(
                    "substrate CPU wall field lies outside [0,1]");
    }
}

inline std::size_t substrate_domain_index(int side, int x, int y, int z) {
    x = wrap(x, side);
    y = wrap(y, side);
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(side)
            + static_cast<std::size_t>(y)) * static_cast<std::size_t>(side)
         + static_cast<std::size_t>(x);
}

// The lower ghost reflects onto plane zero. Substrate states use a zero upper
// ghost; channel states reflect it onto the last physical plane.
inline double substrate_field_at(const SubstrateState& state, int cell,
                                 int x, int y, int z) {
    std::int64_t source_z = static_cast<std::int64_t>(z);
    if (source_z < -1) return 0.0;
    if (source_z == -1) source_z = 0;
    if (state.neutral_top && source_z == state.height)
        source_z = state.height - 1;
    if (source_z >= static_cast<std::int64_t>(state.height)) return 0.0;
    const std::size_t local = substrate_domain_index(
        state.side, x, y, static_cast<int>(source_z));
    return state.phi[static_cast<std::size_t>(cell) * state.domain_words()
                   + local];
}

inline double substrate_laplacian_at(const SubstrateState& state, int cell,
                                     int x, int y, int z) {
    double weighted = 0.0;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                weighted += static_cast<double>(stencil_weight(dx, dy, dz))
                          * substrate_field_at(
                                state, cell, x + dx, y + dy, z + dz);
    return weighted / 30.0;
}

inline Vector3 substrate_gradient_at(const SubstrateState& state, int cell,
                                     int x, int y, int z) {
    return Vector3{
        0.5 * (substrate_field_at(state, cell, x + 1, y, z)
             - substrate_field_at(state, cell, x - 1, y, z)),
        0.5 * (substrate_field_at(state, cell, x, y + 1, z)
             - substrate_field_at(state, cell, x, y - 1, z)),
        0.5 * (substrate_field_at(state, cell, x, y, z + 1)
             - substrate_field_at(state, cell, x, y, z - 1))};
}

inline std::vector<uint32_t> aggregate_substrate_field_q(
    const SubstrateState& state) {
    validate_substrate_state(state);
    const std::size_t words = state.domain_words();
    std::vector<uint32_t> aggregate(words, 0u);
    for (std::size_t cell = 0; cell < state.cells.size(); ++cell) {
        const std::size_t base = cell * words;
        for (std::size_t voxel = 0; voxel < words; ++voxel) {
            const uint32_t value = quantize_phi_sq(state.phi[base + voxel]);
            if (value > std::numeric_limits<uint32_t>::max() - aggregate[voxel])
                throw std::overflow_error(
                    "substrate CPU reference Q5.27 aggregate overflow");
            aggregate[voxel] += value;
        }
    }
    return aggregate;
}

inline double substrate_volume_of(const SubstrateState& state, int cell) {
    const std::size_t words = state.domain_words();
    const std::size_t base = static_cast<std::size_t>(cell) * words;
    double volume = 0.0;
    for (std::size_t voxel = 0; voxel < words; ++voxel) {
        const double value = state.phi[base + voxel];
        volume += value * value;
    }
    return volume;
}

inline OneStepResult one_euler_step_substrate(
    const SubstrateState& state, const ModelParameters& model) {
    validate_substrate_state(state);
    validate_model(model);
    const int side = state.side;
    const int height = state.height;
    const int count = static_cast<int>(state.cells.size());
    const std::size_t words = state.domain_words();
    OneStepResult result;
    result.phi_next.assign(state.phi.size(), 0.0);
    result.aggregate_q = aggregate_substrate_field_q(state);
    result.volume.resize(state.cells.size());
    result.interaction_integral.resize(state.cells.size());
    result.velocity.resize(state.cells.size());

    for (int cell = 0; cell < count; ++cell)
        result.volume[static_cast<std::size_t>(cell)] =
            substrate_volume_of(state, cell);

    for (int cell = 0; cell < count; ++cell) {
        Vector3 integral{};
        for (int z = 0; z < height; ++z) {
            for (int y = 0; y < side; ++y) {
                for (int x = 0; x < side; ++x) {
                    const std::size_t local =
                        substrate_domain_index(side, x, y, z);
                    const double phi = substrate_field_at(
                        state, cell, x, y, z);
                    double other = other_field(
                        result.aggregate_q[local], phi);
                    if (model.wall_kappa > 0.0 &&
                        !state.wall_psi_sq.empty())
                        other += (model.wall_kappa / model.kappa) *
                                 state.wall_psi_sq[static_cast<std::size_t>(z)];
                    const Vector3 gradient = substrate_gradient_at(
                        state, cell, x, y, z);
                    integral.x += phi * gradient.x * other;
                    integral.y += phi * gradient.y * other;
                    integral.z += phi * gradient.z * other;
                }
            }
        }
        result.interaction_integral[static_cast<std::size_t>(cell)] = integral;
        const CellParameters& parameters =
            state.cells[static_cast<std::size_t>(cell)];
        const double coefficient = 60.0 * model.kappa
                                 / (model.xi * model.lambda * model.lambda);
        result.velocity[static_cast<std::size_t>(cell)] = Vector3{
            parameters.active_speed * parameters.polarity.x
                + coefficient * integral.x,
            parameters.active_speed * parameters.polarity.y
                + coefficient * integral.y,
            state.three_dimensional_motion
                ? parameters.active_speed * parameters.polarity.z
                    + coefficient * integral.z
                : 0.0};
    }

    const double volume0 = state.three_dimensional_motion
        ? target_sphere_volume(model.target_radius)
        : target_slab_volume(model.target_radius);
    const double volume_scale = 2.0 * model.mu / volume0;
    const double bulk_scale = 30.0 / (model.lambda * model.lambda);
    const double repulsion = 60.0 * model.kappa
                           / (model.lambda * model.lambda);
    for (int cell = 0; cell < count; ++cell) {
        const CellParameters& parameters =
            state.cells[static_cast<std::size_t>(cell)];
        const Vector3 velocity = result.velocity[static_cast<std::size_t>(cell)];
        const double volume_term = volume_scale
            * (volume0 - result.volume[static_cast<std::size_t>(cell)]);
        const double bulk = bulk_scale * parameters.gamma;
        const std::size_t base = static_cast<std::size_t>(cell) * words;
        for (int z = 0; z < height; ++z) {
            for (int y = 0; y < side; ++y) {
                for (int x = 0; x < side; ++x) {
                    const std::size_t local =
                        substrate_domain_index(side, x, y, z);
                    const double phi = state.phi[base + local];
                    double other = other_field(
                        result.aggregate_q[local], phi);
                    if (model.wall_kappa > 0.0 &&
                        !state.wall_psi_sq.empty())
                        other += (model.wall_kappa / model.kappa) *
                                 state.wall_psi_sq[static_cast<std::size_t>(z)];
                    const double laplacian = substrate_laplacian_at(
                        state, cell, x, y, z);
                    const Vector3 gradient = substrate_gradient_at(
                        state, cell, x, y, z);
                    const double rhs = parameters.gamma * laplacian
                        - bulk * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
                        + volume_term * phi
                        - repulsion * phi * other
                         - (velocity.x * gradient.x
                            + velocity.y * gradient.y
                            + velocity.z * gradient.z);
                    result.phi_next[base + local] = phi + model.dt * rhs;
                }
            }
        }
    }
    return result;
}

inline double minimum_image(double displacement, double side) {
    return displacement - side * std::floor(displacement / side + 0.5);
}

inline void seed_periodic_sphere(DenseState* state, int cell, Vector3 centre,
                                 double seed_radius, double lambda) {
    if (!state || cell < 0 || cell >= static_cast<int>(state->cells.size()) ||
        !(seed_radius >= 0.0) || !(lambda > 0.0))
        throw std::invalid_argument("invalid periodic sphere seed request");
    const int side = state->side;
    const std::size_t words = state->domain_words();
    const std::size_t base = static_cast<std::size_t>(cell) * words;
    const double k = std::sqrt(7.5) / lambda;
    for (int z = 0; z < side; ++z) {
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                const double dx = minimum_image(static_cast<double>(x) - centre.x,
                                                static_cast<double>(side));
                const double dy = minimum_image(static_cast<double>(y) - centre.y,
                                                static_cast<double>(side));
                const double dz = minimum_image(static_cast<double>(z) - centre.z,
                                                static_cast<double>(side));
                const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                state->phi[base + domain_index(side, x, y, z)] =
                    0.5 * (1.0 - std::tanh(k * (radius - seed_radius)));
            }
        }
    }
}

inline void seed_substrate_hemisphere(SubstrateState* state, int cell,
                                      double centre_x, double centre_y,
                                      double seed_radius, double lambda) {
    if (!state || cell < 0 || cell >= static_cast<int>(state->cells.size()) ||
        !(seed_radius >= 0.0) || !(lambda > 0.0) ||
        !std::isfinite(centre_x) || !std::isfinite(centre_y))
        throw std::invalid_argument("invalid substrate hemisphere seed request");
    validate_substrate_state(*state);
    const int side = state->side;
    const int height = state->height;
    const std::size_t base = static_cast<std::size_t>(cell)
                           * state->domain_words();
    const double k = std::sqrt(7.5) / lambda;
    for (int z = 0; z < height; ++z) {
        const double dz = static_cast<double>(z) + 0.5;
        for (int y = 0; y < side; ++y) {
            const double dy = minimum_image(
                static_cast<double>(y) - centre_y,
                static_cast<double>(side));
            for (int x = 0; x < side; ++x) {
                const double dx = minimum_image(
                    static_cast<double>(x) - centre_x,
                    static_cast<double>(side));
                const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                state->phi[base + substrate_domain_index(side, x, y, z)] =
                    0.5 * (1.0 - std::tanh(k * (radius - seed_radius)));
            }
        }
    }
}

inline void seed_channel_sphere(SubstrateState* state, int cell,
                                Vector3 centre, double seed_radius,
                                double lambda) {
    if (!state || cell < 0 || cell >= static_cast<int>(state->cells.size()) ||
        !(seed_radius >= 0.0) || !(lambda > 0.0) ||
        !std::isfinite(centre.x) || !std::isfinite(centre.y) ||
        !std::isfinite(centre.z))
        throw std::invalid_argument("invalid hard-wall channel sphere seed request");
    validate_substrate_state(*state);
    const int side = state->side;
    const std::size_t base = static_cast<std::size_t>(cell)
                           * state->domain_words();
    const double k = std::sqrt(7.5) / lambda;
    for (int z = 0; z < state->height; ++z) {
        const double dz = static_cast<double>(z) - centre.z;
        for (int y = 0; y < side; ++y) {
            const double dy = minimum_image(
                static_cast<double>(y) - centre.y, static_cast<double>(side));
            for (int x = 0; x < side; ++x) {
                const double dx = minimum_image(
                    static_cast<double>(x) - centre.x,
                    static_cast<double>(side));
                const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                state->phi[base + substrate_domain_index(side, x, y, z)] =
                    0.5 * (1.0 - std::tanh(k * (distance - seed_radius)));
            }
        }
    }
}

// Evaluate the discrete phi^2 volume after shifting the diffuse sphere away
// from the half-grid centre used by the radius calibration. Production rounds
// each brick origin to the nearest lattice site, so every realized centre is
// represented by one offset in [-0.5,0.5]^3. Individual cells are not
// recalibrated after placement.
inline double seeded_volume_in_brick_at_offset(int edge, double seed_radius,
                                               double lambda,
                                               Vector3 centre_offset) {
    if (edge < 3 || !(seed_radius >= 0.0) || !(lambda > 0.0))
        throw std::invalid_argument("invalid seed-radius calibration geometry");
    const auto valid_offset = [](double value) {
        return std::isfinite(value) && value >= -0.5 && value <= 0.5;
    };
    if (!valid_offset(centre_offset.x) ||
        !valid_offset(centre_offset.y) ||
        !valid_offset(centre_offset.z))
        throw std::invalid_argument("seed centre offset lies outside [-0.5,0.5]");
    const double centre = 0.5 * static_cast<double>(edge - 1);
    const double centre_x = centre + centre_offset.x;
    const double centre_y = centre + centre_offset.y;
    const double centre_z = centre + centre_offset.z;
    const double k = std::sqrt(7.5) / lambda;
    double volume = 0.0;
    for (int z = 0; z < edge; ++z) {
        const double dz = static_cast<double>(z) - centre_z;
        for (int y = 0; y < edge; ++y) {
            const double dy = static_cast<double>(y) - centre_y;
            for (int x = 0; x < edge; ++x) {
                const double dx = static_cast<double>(x) - centre_x;
                const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double phi =
                    0.5 * (1.0 - std::tanh(k * (radius - seed_radius)));
                volume += phi * phi;
            }
        }
    }
    return volume;
}

inline double seeded_volume_in_brick(int edge, double seed_radius,
                                     double lambda) {
    return seeded_volume_in_brick_at_offset(
        edge, seed_radius, lambda, Vector3{});
}

// Positive-z half of the same even brick used by the spherical calibration.
// With the wall at z=-1/2, every retained plane z has a bitwise-identical
// mirror at -z-1 in the full sphere.
inline double seeded_substrate_volume_in_brick(int edge, double seed_radius,
                                               double lambda) {
    if (edge < 4 || (edge & 1) != 0 || !(seed_radius >= 0.0) ||
        !(lambda > 0.0))
        throw std::invalid_argument(
            "substrate calibration requires a positive even brick edge");
    const double centre = 0.5 * static_cast<double>(edge - 1);
    const double k = std::sqrt(7.5) / lambda;
    double volume = 0.0;
    for (int z = edge / 2; z < edge; ++z) {
        const double dz = static_cast<double>(z) - centre;
        for (int y = 0; y < edge; ++y) {
            const double dy = static_cast<double>(y) - centre;
            for (int x = 0; x < edge; ++x) {
                const double dx = static_cast<double>(x) - centre;
                const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double phi =
                    0.5 * (1.0 - std::tanh(k * (radius - seed_radius)));
                volume += phi * phi;
            }
        }
    }
    return volume;
}

struct SeedRadiusCalibration {
    double seed_radius = 0.0;
    double measured_volume = 0.0;
    double target = 0.0;
    double relative_error = std::numeric_limits<double>::infinity();
    int iterations = 0;
    bool converged = false;
};

inline SeedRadiusCalibration calibrate_seed_radius(
    double target_radius_value, double lambda, int brick_edge,
    double relative_tolerance = 1.0e-10, int max_iterations = 80) {
    if (!(target_radius_value > 0.0) || !(lambda > 0.0) || brick_edge < 3 ||
        !(relative_tolerance > 0.0) || max_iterations <= 0)
        throw std::invalid_argument("invalid seed-radius calibration request");
    SeedRadiusCalibration result;
    result.target = target_sphere_volume(target_radius_value);
    double lower = 0.0;
    double upper = static_cast<double>(brick_edge);
    const double lower_volume = seeded_volume_in_brick(brick_edge, lower, lambda);
    const double upper_volume = seeded_volume_in_brick(brick_edge, upper, lambda);
    if (lower_volume > result.target || upper_volume < result.target)
        throw std::invalid_argument("target volume is not bracketed by the brick");

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        const double measured = seeded_volume_in_brick(brick_edge, midpoint, lambda);
        const double relative_error = std::fabs(measured - result.target) / result.target;
        result.seed_radius = midpoint;
        result.measured_volume = measured;
        result.relative_error = relative_error;
        result.iterations = iteration;
        if (relative_error <= relative_tolerance) {
            result.converged = true;
            break;
        }
        if (measured < result.target)
            lower = midpoint;
        else
            upper = midpoint;
    }
    return result;
}

inline SeedRadiusCalibration calibrate_substrate_seed_radius(
    double target_radius_value, double lambda, int brick_edge,
    double relative_tolerance = 1.0e-10, int max_iterations = 80) {
    if ((brick_edge & 1) != 0)
        throw std::invalid_argument(
            "substrate calibration requires an even brick edge");
    // The neutral hemisphere is exactly one mirror half of the full discrete
    // sphere. Its target and every bisection ordinate are scaled by the same
    // factor, so it must use the identical calibrated radius.
    SeedRadiusCalibration result = calibrate_seed_radius(
        target_radius_value, lambda, brick_edge,
        relative_tolerance, max_iterations);
    result.target = target_slab_volume(target_radius_value);
    result.measured_volume = seeded_substrate_volume_in_brick(
        brick_edge, result.seed_radius, lambda);
    result.relative_error = std::fabs(result.measured_volume - result.target)
                          / result.target;
    result.converged = result.relative_error <= relative_tolerance;
    return result;
}

}  // namespace pf3d::reference
