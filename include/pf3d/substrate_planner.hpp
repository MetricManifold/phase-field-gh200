#pragma once

// Whole-cell ownership and conservative peer-exchange bounds for bounded-z
// geometries split in y, using a complete accepted-state snapshot.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "pf3d/params.cuh"

namespace pf3d {

struct TwoRankSubstrateGeometry {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int pitch_x = 0;  // uint32 words per world-y row
    std::uint32_t boundary_flags = 0;
};

enum class TwoRankValidation {
    Ok,
    NonPositiveExtent,
    NonSquareLateralDomain,
    UnexpectedPitch,
    WrongBoundaryFlags
};

inline TwoRankValidation validate(const TwoRankSubstrateGeometry& geometry) {
    if (geometry.nx <= 0 || geometry.ny <= 0 || geometry.nz <= 0 ||
        geometry.pitch_x <= 0) {
        return TwoRankValidation::NonPositiveExtent;
    }
    if (geometry.nx != geometry.ny)
        return TwoRankValidation::NonSquareLateralDomain;
    const std::int64_t expected_pitch =
        32LL * ((static_cast<std::int64_t>(geometry.nx) + 31LL) / 32LL);
    if (expected_pitch > std::numeric_limits<int>::max() ||
        geometry.pitch_x != expected_pitch) {
        return TwoRankValidation::UnexpectedPitch;
    }
    if (geometry.boundary_flags != kBoundarySubstrateSlab3D &&
        geometry.boundary_flags != kBoundaryHardWallChannel3D)
        return TwoRankValidation::WrongBoundaryFlags;
    return TwoRankValidation::Ok;
}

struct RowInterval {
    int lo = 0;
    int hi = 0;  // exclusive
};

inline bool empty(const RowInterval& interval) {
    return interval.hi <= interval.lo;
}

struct WrappedRows {
    std::array<RowInterval, 2> pieces{};
    int count = 0;
};

struct RowSet {
    std::array<RowInterval, 4> pieces{};
    int count = 0;
    int total_rows = 0;
};

struct RankOwnership {
    int y_lo = 0;
    int y_hi = 0;  // exclusive
};

inline RankOwnership rank_ownership(const TwoRankSubstrateGeometry& geometry,
                                    int rank) {
    if (validate(geometry) != TwoRankValidation::Ok ||
        (rank != 0 && rank != 1)) {
        return {};
    }
    const int cut = geometry.ny / 2;
    return rank == 0 ? RankOwnership{0, cut}
                     : RankOwnership{cut, geometry.ny};
}

namespace substrate_planner_detail {

inline WrappedRows band_around(int center, std::int64_t half_width,
                               int domain_rows) {
    WrappedRows band{};
    if (domain_rows <= 0 || half_width <= 0) return band;
    const std::int64_t full_threshold =
        static_cast<std::int64_t>(domain_rows) - domain_rows / 2;
    if (half_width >= full_threshold) {
        band.pieces[0] = {0, domain_rows};
        band.count = 1;
        return band;
    }

    const std::int64_t lo = static_cast<std::int64_t>(center) - half_width;
    const std::int64_t hi = static_cast<std::int64_t>(center) + half_width;
    if (lo < 0) {
        band.pieces[0] = {0, static_cast<int>(hi)};
        band.pieces[1] = {static_cast<int>(domain_rows + lo), domain_rows};
        band.count = 2;
    } else if (hi > domain_rows) {
        band.pieces[0] = {static_cast<int>(lo), domain_rows};
        band.pieces[1] = {0, static_cast<int>(hi - domain_rows)};
        band.count = 2;
    } else {
        band.pieces[0] = {static_cast<int>(lo), static_cast<int>(hi)};
        band.count = 1;
    }
    return band;
}

inline RowSet canonical_union(const WrappedRows& first,
                              const WrappedRows& second) {
    RowSet result{};
    std::array<RowInterval, 4> pending{};
    int pending_count = 0;
    for (int i = 0; i < first.count; ++i)
        if (!empty(first.pieces[i])) pending[pending_count++] = first.pieces[i];
    for (int i = 0; i < second.count; ++i)
        if (!empty(second.pieces[i])) pending[pending_count++] = second.pieces[i];

    for (int i = 1; i < pending_count; ++i) {
        const RowInterval key = pending[i];
        int j = i - 1;
        while (j >= 0 && pending[j].lo > key.lo) {
            pending[j + 1] = pending[j];
            --j;
        }
        pending[j + 1] = key;
    }
    for (int i = 0; i < pending_count; ++i) {
        const RowInterval interval = pending[i];
        if (result.count == 0 ||
            interval.lo > result.pieces[result.count - 1].hi) {
            result.pieces[result.count++] = interval;
        } else if (interval.hi > result.pieces[result.count - 1].hi) {
            result.pieces[result.count - 1].hi = interval.hi;
        }
    }
    for (int i = 0; i < result.count; ++i)
        result.total_rows += result.pieces[i].hi - result.pieces[i].lo;
    return result;
}

inline bool intersects(const WrappedRows& wrapped, const RowSet& rows) {
    for (int i = 0; i < wrapped.count; ++i) {
        for (int j = 0; j < rows.count; ++j) {
            if (wrapped.pieces[i].lo < rows.pieces[j].hi &&
                rows.pieces[j].lo < wrapped.pieces[i].hi) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace substrate_planner_detail

struct SeamBands {
    WrappedRows interior_cut;
    WrappedRows periodic_cut;
    RowSet rows;
    int domain_rows = 0;
    std::int64_t half_width = 0;
    bool valid = false;
    bool full_domain = false;
};

inline SeamBands seam_band_union(const TwoRankSubstrateGeometry& geometry,
                                 std::int64_t half_width) {
    SeamBands bands{};
    if (validate(geometry) != TwoRankValidation::Ok || half_width < 0)
        return bands;
    bands.valid = true;
    bands.domain_rows = geometry.ny;
    bands.half_width = half_width;
    bands.interior_cut = substrate_planner_detail::band_around(
        geometry.ny / 2, half_width, geometry.ny);
    bands.periodic_cut = substrate_planner_detail::band_around(
        0, half_width, geometry.ny);
    bands.rows = substrate_planner_detail::canonical_union(
        bands.interior_cut, bands.periodic_cut);
    bands.full_domain = bands.rows.total_rows == geometry.ny;
    return bands;
}

// Minimal host projection of the CellState3D fields needed by the planner.
// A zero storage_edge resolves to the runtime-selected base edge.
struct AllocatedBrick {
    std::int64_t origin_y = 0;
    std::int64_t origin_z = 0;
    std::uint32_t storage_edge = 0;
};

inline bool resolve_storage_edge(const AllocatedBrick& cell,
                                 std::uint32_t runtime_base_edge,
                                 int* resolved_edge) {
    if (resolved_edge == nullptr || runtime_base_edge == 0 ||
        runtime_base_edge > static_cast<std::uint32_t>(
                                std::numeric_limits<int>::max())) {
        return false;
    }
    const std::uint32_t edge =
        cell.storage_edge == 0 ? runtime_base_edge : cell.storage_edge;
    if (edge == 0 ||
        edge > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    *resolved_edge = static_cast<int>(edge);
    return true;
}

struct ZEnvelope {
    int lo = 0;
    int hi = 0;  // exclusive
};

inline bool empty(const ZEnvelope& envelope) {
    return envelope.hi <= envelope.lo;
}

inline bool cell_allocated_z_envelope(const AllocatedBrick& cell,
                                      std::uint32_t runtime_base_edge,
                                      int domain_height,
                                      ZEnvelope* envelope,
                                      std::int64_t max_origin_displacement = 0) {
    if (envelope == nullptr) return false;
    *envelope = {};
    int edge = 0;
    if (domain_height <= 0 || max_origin_displacement < 0 ||
        !resolve_storage_edge(cell, runtime_base_edge, &edge)) {
        return false;
    }

    const std::int64_t origin_z = cell.origin_z;
    const std::int64_t edge64 = edge;
    const std::int64_t allocated_hi =
        origin_z > std::numeric_limits<std::int64_t>::max() - edge64
            ? std::numeric_limits<std::int64_t>::max()
            : origin_z + edge64;
    const std::int64_t lo =
        origin_z < std::numeric_limits<std::int64_t>::min() +
                       max_origin_displacement
            ? std::numeric_limits<std::int64_t>::min()
            : origin_z - max_origin_displacement;
    const std::int64_t hi =
        allocated_hi > std::numeric_limits<std::int64_t>::max() -
                           max_origin_displacement
            ? std::numeric_limits<std::int64_t>::max()
            : allocated_hi + max_origin_displacement;
    const std::int64_t clamped_lo = lo > 0 ? lo : 0;
    const std::int64_t clamped_hi =
        hi < domain_height ? hi : static_cast<std::int64_t>(domain_height);
    if (clamped_hi <= clamped_lo) return true;
    if (clamped_lo > std::numeric_limits<int>::max()) return true;
    envelope->lo = static_cast<int>(clamped_lo);
    envelope->hi = static_cast<int>(clamped_hi);
    return true;
}

inline bool cell_wrapped_y(const AllocatedBrick& cell,
                           std::uint32_t runtime_base_edge, int domain_rows,
                           WrappedRows* wrapped) {
    if (wrapped == nullptr) return false;
    *wrapped = {};
    int edge = 0;
    if (domain_rows <= 0 ||
        !resolve_storage_edge(cell, runtime_base_edge, &edge)) {
        return false;
    }
    if (edge >= domain_rows) {
        wrapped->pieces[0] = {0, domain_rows};
        wrapped->count = 1;
        return true;
    }
    const std::int64_t origin_y = cell.origin_y;
    std::int64_t start = origin_y % domain_rows;
    if (start < 0) start += domain_rows;
    const std::int64_t end = start + edge;
    if (end <= domain_rows) {
        wrapped->pieces[0] = {static_cast<int>(start), static_cast<int>(end)};
        wrapped->count = 1;
    } else {
        wrapped->pieces[0] = {static_cast<int>(start), domain_rows};
        wrapped->pieces[1] = {0, static_cast<int>(end - domain_rows)};
        wrapped->count = 2;
    }
    return true;
}

enum class CellSeamClass { Invalid, BandContributing, DeepInterior };

inline CellSeamClass classify_cell(const AllocatedBrick& cell,
                                   std::uint32_t runtime_base_edge,
                                   const TwoRankSubstrateGeometry& geometry,
                                   const SeamBands& bands) {
    if (validate(geometry) != TwoRankValidation::Ok || !bands.valid ||
        bands.domain_rows != geometry.ny) {
        return CellSeamClass::Invalid;
    }
    WrappedRows cell_rows{};
    if (!cell_wrapped_y(cell, runtime_base_edge, geometry.ny, &cell_rows))
        return CellSeamClass::Invalid;
    return substrate_planner_detail::intersects(cell_rows, bands.rows)
               ? CellSeamClass::BandContributing
               : CellSeamClass::DeepInterior;
}

// Ownership follows the midpoint at the planning snapshot. It may stay frozen
// while the exchange plan's displacement bound and storage edges remain valid.
inline bool cell_owner_rank(const AllocatedBrick& cell,
                            std::uint32_t runtime_base_edge,
                            const TwoRankSubstrateGeometry& geometry,
                            int* owner) {
    if (owner == nullptr || validate(geometry) != TwoRankValidation::Ok)
        return false;
    int edge = 0;
    if (!resolve_storage_edge(cell, runtime_base_edge, &edge)) return false;
    const std::int64_t origin_y = cell.origin_y;
    std::int64_t midpoint = origin_y % geometry.ny;
    if (midpoint < 0) midpoint += geometry.ny;
    midpoint = (midpoint + edge / 2) % geometry.ny;
    *owner = midpoint < geometry.ny / 2 ? 0 : 1;
    return true;
}

inline bool pitched_payload_bytes(int rows, int pitch_x, int z_extent,
                                  std::size_t element_bytes,
                                  std::size_t* bytes) {
    if (bytes == nullptr) return false;
    *bytes = 0;
    if (rows <= 0 || pitch_x <= 0 || z_extent <= 0 || element_bytes == 0)
        return false;
    std::size_t row_words = 0;
    std::size_t volume_words = 0;
    return checked_mul_size(static_cast<std::size_t>(rows),
                            static_cast<std::size_t>(pitch_x), &row_words) &&
           checked_mul_size(row_words, static_cast<std::size_t>(z_extent),
                            &volume_words) &&
           checked_mul_size(volume_words, element_bytes, bytes);
}

enum class SeamPlanStatus {
    Ok,
    InvalidGeometry,
    InvalidRank,
    InvalidBaseEdge,
    InvalidElementSize,
    InvalidCellArray,
    InvalidCell,
    InvalidDisplacement,
    SizeOverflow
};

struct SeamExchangePlan {
    SeamPlanStatus status = SeamPlanStatus::InvalidGeometry;
    SeamBands bands{};
    ZEnvelope z{};
    std::size_t owned_cells = 0;
    std::size_t contributing_cells = 0;
    std::size_t deep_interior_cells = 0;
    std::size_t payload_bytes = 0;  // one direction, one GPU
};

// Freeze ownership from this snapshot until any origin moves more than the
// supplied bound in y or z, or any storage edge changes. Both ranks must use
// the same snapshot and bound; a zero bound retains the single-step plan.
inline SeamExchangePlan build_seam_exchange_plan(
    const TwoRankSubstrateGeometry& geometry,
    std::uint32_t runtime_base_edge, int owner_rank,
    const AllocatedBrick* all_cells, std::size_t cell_count,
    std::size_t element_bytes,
    std::int64_t max_origin_displacement = 0) {
    SeamExchangePlan plan{};
    if (validate(geometry) != TwoRankValidation::Ok) return plan;
    if (owner_rank != 0 && owner_rank != 1) {
        plan.status = SeamPlanStatus::InvalidRank;
        return plan;
    }
    int base_edge = 0;
    AllocatedBrick base{};
    if (!resolve_storage_edge(base, runtime_base_edge, &base_edge)) {
        plan.status = SeamPlanStatus::InvalidBaseEdge;
        return plan;
    }
    // Match SimParams3D::minimum_domain_edge: a channel brick may extend past
    // both z faces, but must not overlap its own periodic image in x or y.
    const int minimum_extent =
        geometry.boundary_flags == kBoundaryHardWallChannel3D
            ? geometry.nx
            : (geometry.nz < geometry.nx ? geometry.nz : geometry.nx);
    if (base_edge < 3 || base_edge % kBrickAlignment != 0 ||
        base_edge >= minimum_extent) {
        plan.status = SeamPlanStatus::InvalidBaseEdge;
        return plan;
    }
    if (element_bytes == 0) {
        plan.status = SeamPlanStatus::InvalidElementSize;
        return plan;
    }
    if (cell_count != 0 && all_cells == nullptr) {
        plan.status = SeamPlanStatus::InvalidCellArray;
        return plan;
    }
    if (max_origin_displacement < 0) {
        plan.status = SeamPlanStatus::InvalidDisplacement;
        return plan;
    }

    int max_edge = base_edge;
    for (std::size_t i = 0; i < cell_count; ++i) {
        int edge = 0;
        int resolved_owner = -1;
        if (!resolve_storage_edge(all_cells[i], runtime_base_edge, &edge) ||
            edge < base_edge || edge % kBrickAlignment != 0 ||
            edge >= minimum_extent) {
            plan.status = SeamPlanStatus::InvalidCell;
            return plan;
        }
        if (!cell_owner_rank(all_cells[i], runtime_base_edge, geometry,
                             &resolved_owner)) {
            plan.status = SeamPlanStatus::InvalidCell;
            return plan;
        }
        if (edge > max_edge) max_edge = edge;
    }
    const std::int64_t static_half_width =
        (static_cast<std::int64_t>(max_edge) + 1) / 2;
    if (max_origin_displacement > std::numeric_limits<std::int64_t>::max() -
                                      static_half_width) {
        plan.status = SeamPlanStatus::SizeOverflow;
        return plan;
    }
    const std::int64_t half_width = static_half_width + max_origin_displacement;
    plan.bands = seam_band_union(geometry, half_width);
    if (!plan.bands.valid) return plan;

    bool have_z = false;
    for (std::size_t i = 0; i < cell_count; ++i) {
        int resolved_owner = -1;
        if (!cell_owner_rank(all_cells[i], runtime_base_edge, geometry,
                             &resolved_owner)) {
            plan.status = SeamPlanStatus::InvalidCell;
            return plan;
        }
        if (resolved_owner != owner_rank) continue;
        ++plan.owned_cells;
        // A cell initially away from a seam can reach it before replanning.
        // Include every owned swept box when ownership lasts multiple steps.
        const CellSeamClass classification = max_origin_displacement == 0
            ? classify_cell(all_cells[i], runtime_base_edge, geometry, plan.bands)
            : CellSeamClass::BandContributing;
        if (classification == CellSeamClass::Invalid) {
            plan.status = SeamPlanStatus::InvalidCell;
            return plan;
        }
        if (classification == CellSeamClass::DeepInterior) {
            ++plan.deep_interior_cells;
            continue;
        }
        ++plan.contributing_cells;
        ZEnvelope cell_z{};
        if (!cell_allocated_z_envelope(all_cells[i], runtime_base_edge,
                                       geometry.nz, &cell_z,
                                       max_origin_displacement)) {
            plan.status = SeamPlanStatus::InvalidCell;
            return plan;
        }
        if (empty(cell_z)) {
            plan.status = SeamPlanStatus::InvalidCell;
            return plan;
        }
        if (!have_z) {
            plan.z = cell_z;
            have_z = true;
        } else {
            if (cell_z.lo < plan.z.lo) plan.z.lo = cell_z.lo;
            if (cell_z.hi > plan.z.hi) plan.z.hi = cell_z.hi;
        }
    }

    if (plan.contributing_cells != 0 && have_z) {
        if (!pitched_payload_bytes(plan.bands.rows.total_rows,
                                   geometry.pitch_x, plan.z.hi - plan.z.lo,
                                   element_bytes, &plan.payload_bytes)) {
            plan.status = SeamPlanStatus::SizeOverflow;
            return plan;
        }
    }
    plan.status = SeamPlanStatus::Ok;
    return plan;
}

}  // namespace pf3d
