#pragma once

#include "params.cuh"
#include "philox.cuh"

#include <cmath>
#include <cstdint>

namespace pf3d {

using Philox4 = pf_common::Philox4;
using pf_common::philox4x32_10;
using pf_common::philox_uniform53;

struct Vec3 {
    double x, y, z;
};

PF3D_HD inline Vec3 isotropic_unit_vector(double axial_u, double azimuth_u) {
    const double z = 2.0 * axial_u - 1.0;
    const double radial_sq = 1.0 - z * z;
    const double radial = sqrt(radial_sq > 0.0 ? radial_sq : 0.0);
    const double azimuth = 2.0 * kPi * azimuth_u;
    return Vec3{radial * cos(azimuth), radial * sin(azimuth), z};
}

// Unit vector tangent to the substrate.  Keeping this separate from the
// isotropic sampler makes the stochastic contract of each geometry explicit:
// both use Philox counters, but a substrate-bound cell draws one angle rather
// than a direction on the sphere.
PF3D_HD inline Vec3 planar_unit_vector(double azimuth_u) {
    const double azimuth = 2.0 * kPi * azimuth_u;
    return Vec3{cos(azimuth), sin(azimuth), 0.0};
}

constexpr uint32_t kInitialPolarityDomain = 0x504F4C49u;  // "POLI"
constexpr uint32_t kInitialActiveSpeedDomain = 0x56414C4Fu;  // "VALO"
constexpr uint32_t kTumbleEventDomain     = 0x54554D45u;  // "TUME"
constexpr uint32_t kTumbleVectorDomain    = 0x54554D56u;  // "TUMV"

PF3D_HD inline Vec3 initial_polarity(int64_t global_id,
                                     unsigned long long stream) {
    const uint64_t gid = static_cast<uint64_t>(global_id);
    const Philox4 random = philox4x32_10(
        static_cast<uint32_t>(gid), static_cast<uint32_t>(gid >> 32),
        kInitialPolarityDomain, 0u,
        static_cast<uint32_t>(stream), static_cast<uint32_t>(stream >> 32));
    return isotropic_unit_vector(
        philox_uniform53(random.v[0], random.v[1]),
        philox_uniform53(random.v[2], random.v[3]));
}

PF3D_HD inline Vec3 initial_planar_polarity(
    int64_t global_id, unsigned long long stream) {
    const uint64_t gid = static_cast<uint64_t>(global_id);
    const Philox4 random = philox4x32_10(
        static_cast<uint32_t>(gid), static_cast<uint32_t>(gid >> 32),
        kInitialPolarityDomain, 0u,
        static_cast<uint32_t>(stream), static_cast<uint32_t>(stream >> 32));
    return planar_unit_vector(philox_uniform53(random.v[2], random.v[3]));
}

// Per-cell lognormal active speed. The supplied speed is the median; sigma is
// the standard deviation of log(v_A/median). A zero sigma consumes no random
// value and returns the median exactly.
PF3D_HD inline double initial_active_speed(int64_t global_id,
                                           unsigned long long seed,
                                           double median,
                                           double sigma) {
    if (!(sigma > 0.0)) return median;
    const uint64_t gid = static_cast<uint64_t>(global_id);
    const Philox4 random = philox4x32_10(
        static_cast<uint32_t>(gid), static_cast<uint32_t>(gid >> 32),
        kInitialActiveSpeedDomain, 0u,
        static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    double u1 = philox_uniform53(random.v[0], random.v[1]);
    if (u1 < 1.0e-300) u1 = 1.0e-300;
    const double u2 = philox_uniform53(random.v[2], random.v[3]);
    const double normal = sqrt(-2.0 * log(u1)) * cos(2.0 * kPi * u2);
    return median * exp(sigma * normal);
}

PF3D_HD inline Vec3 tumble_polarity(unsigned long long step, int64_t global_id,
                                    unsigned long long stream) {
    const uint64_t gid = static_cast<uint64_t>(global_id);
    const Philox4 event = philox4x32_10(
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        static_cast<uint32_t>(gid),
        static_cast<uint32_t>(gid >> 32) ^ kTumbleEventDomain,
        static_cast<uint32_t>(stream), static_cast<uint32_t>(stream >> 32));
    const Philox4 vector = philox4x32_10(
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        static_cast<uint32_t>(gid),
        static_cast<uint32_t>(gid >> 32) ^ kTumbleVectorDomain,
        static_cast<uint32_t>(stream), static_cast<uint32_t>(stream >> 32));
    return isotropic_unit_vector(
        philox_uniform53(event.v[2], event.v[3]),
        philox_uniform53(vector.v[0], vector.v[1]));
}

PF3D_HD inline Vec3 tumble_planar_polarity(
    unsigned long long step, int64_t global_id,
    unsigned long long stream) {
    const uint64_t gid = static_cast<uint64_t>(global_id);
    const Philox4 vector = philox4x32_10(
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        static_cast<uint32_t>(gid),
        static_cast<uint32_t>(gid >> 32) ^ kTumbleVectorDomain,
        static_cast<uint32_t>(stream), static_cast<uint32_t>(stream >> 32));
    return planar_unit_vector(philox_uniform53(vector.v[0], vector.v[1]));
}

PF3D_HD inline bool tumble_occurs(unsigned long long step, int64_t global_id,
                                  unsigned long long stream,
                                  double probability) {
    const uint64_t gid = static_cast<uint64_t>(global_id);
    const Philox4 random = philox4x32_10(
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        static_cast<uint32_t>(gid),
        static_cast<uint32_t>(gid >> 32) ^ kTumbleEventDomain,
        static_cast<uint32_t>(stream), static_cast<uint32_t>(stream >> 32));
    return philox_uniform53(random.v[0], random.v[1]) < probability;
}

struct TumbleSample {
    Vec3 polarity;
    bool tumbled;
};

PF3D_HD inline TumbleSample tumble_sample(unsigned long long step,
                                          int64_t global_id,
                                          unsigned long long stream,
                                          double probability,
                                          Vec3 current_polarity) {
    const bool event = tumble_occurs(step, global_id, stream, probability);
    return TumbleSample{
        event ? tumble_polarity(step, global_id, stream) : current_polarity,
        event};
}

PF3D_HD inline TumbleSample tumble_sample_planar(
    unsigned long long step, int64_t global_id,
    unsigned long long stream, double probability,
    Vec3 current_polarity) {
    const bool event = tumble_occurs(step, global_id, stream, probability);
    return TumbleSample{
        event ? tumble_planar_polarity(step, global_id, stream)
              : current_polarity,
        event};
}

}  // namespace pf3d
