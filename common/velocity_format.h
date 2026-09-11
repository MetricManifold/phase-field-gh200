#pragma once
// PFVMOM4 is an explicitly versioned little-endian output, not a checkpoint.
// A frame is int64 step followed by cells * kColumns float64 values.
// All components are cumulative integrals since start_step, except columns
// 22..25, which contain instantaneous endpoint Gx, Gy, Kx, Ky.
#include <cstddef>
#include <cstdint>

namespace pf {
namespace velocity {
constexpr std::uint32_t kColumns = 26;
constexpr std::uint32_t kDiscreteLinearQuadrature = 2;
constexpr int kIntegralColumn = 9;
constexpr int kSpatialCountColumn = 21;
constexpr int kEndpointColumn = 22;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8]; // PFVMOM4 followed by NUL
    std::uint32_t cells;
    std::uint32_t columns;
    double dt;
    std::int64_t start_step;
    double side;
    std::uint32_t spatial_stride;
    std::uint32_t quadrature;
    std::uint64_t dense_start;
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 56, "PFVMOM4 header layout changed");
static_assert(offsetof(FileHeader, dense_start) == 48, "PFVMOM4 offset changed");
} // namespace velocity
} // namespace pf
