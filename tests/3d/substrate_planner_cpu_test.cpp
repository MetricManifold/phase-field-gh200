#include "pf3d/substrate_planner.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>

namespace {

constexpr int kSide = 916;
constexpr int kHeight = 288;
constexpr int kPitch = 928;
constexpr std::uint32_t kBaseEdge = 152;

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[FAIL] %s\n", message);
        ++failures;
    }
}

pf3d::TwoRankSubstrateGeometry geometry(int side = kSide) {
    return {side, side, kHeight, 32 * ((side + 31) / 32),
            pf3d::kBoundarySubstrateSlab3D};
}

pf3d::TwoRankSubstrateGeometry channel_geometry(int side = 733) {
    return {side, side, 140, 32 * ((side + 31) / 32),
            pf3d::kBoundaryHardWallChannel3D};
}

bool interval_is(const pf3d::RowInterval& interval, int lo, int hi) {
    return interval.lo == lo && interval.hi == hi;
}

bool contains(const pf3d::RowSet& rows, int row) {
    for (int i = 0; i < rows.count; ++i) {
        if (row >= rows.pieces[i].lo && row < rows.pieces[i].hi) return true;
    }
    return false;
}

bool same_plan(const pf3d::SeamExchangePlan& a,
               const pf3d::SeamExchangePlan& b) {
    if (a.status != b.status || a.z.lo != b.z.lo || a.z.hi != b.z.hi ||
        a.owned_cells != b.owned_cells ||
        a.contributing_cells != b.contributing_cells ||
        a.deep_interior_cells != b.deep_interior_cells ||
        a.payload_bytes != b.payload_bytes ||
        a.bands.valid != b.bands.valid ||
        a.bands.full_domain != b.bands.full_domain ||
        a.bands.domain_rows != b.bands.domain_rows ||
        a.bands.half_width != b.bands.half_width ||
        a.bands.rows.count != b.bands.rows.count ||
        a.bands.rows.total_rows != b.bands.rows.total_rows ||
        a.bands.interior_cut.count != b.bands.interior_cut.count ||
        a.bands.periodic_cut.count != b.bands.periodic_cut.count)
        return false;
    for (std::size_t i = 0; i < a.bands.rows.pieces.size(); ++i)
        if (!interval_is(a.bands.rows.pieces[i],
                         b.bands.rows.pieces[i].lo, b.bands.rows.pieces[i].hi))
            return false;
    for (std::size_t i = 0; i < a.bands.interior_cut.pieces.size(); ++i)
        if (!interval_is(a.bands.interior_cut.pieces[i],
                         b.bands.interior_cut.pieces[i].lo,
                         b.bands.interior_cut.pieces[i].hi) ||
            !interval_is(a.bands.periodic_cut.pieces[i],
                         b.bands.periodic_cut.pieces[i].lo,
                         b.bands.periodic_cut.pieces[i].hi))
            return false;
    return true;
}

void test_geometry_and_ownership() {
    const auto even = geometry();
    expect(pf3d::validate(even) == pf3d::TwoRankValidation::Ok,
           "production geometry is valid");
    expect(interval_is({pf3d::rank_ownership(even, 0).y_lo,
                        pf3d::rank_ownership(even, 0).y_hi}, 0, 458),
           "rank zero owns the lower half");
    expect(interval_is({pf3d::rank_ownership(even, 1).y_lo,
                        pf3d::rank_ownership(even, 1).y_hi}, 458, 916),
           "rank one owns the upper half");

    const auto odd = geometry(917);
    expect(pf3d::validate(odd) == pf3d::TwoRankValidation::Ok,
           "odd lateral size is supported");
    expect(pf3d::rank_ownership(odd, 0).y_hi == 458 &&
               pf3d::rank_ownership(odd, 1).y_lo == 458 &&
               pf3d::rank_ownership(odd, 1).y_hi == 917,
           "odd row is assigned to rank one");

    auto invalid = even;
    invalid.nx = 0;
    expect(pf3d::validate(invalid) ==
               pf3d::TwoRankValidation::NonPositiveExtent,
           "zero extent is rejected");
    invalid = even;
    invalid.nx = 915;
    expect(pf3d::validate(invalid) ==
               pf3d::TwoRankValidation::NonSquareLateralDomain,
           "non-square slab is rejected");
    invalid = even;
    invalid.pitch_x = 916;
    expect(pf3d::validate(invalid) ==
               pf3d::TwoRankValidation::UnexpectedPitch,
           "non-runtime pitch is rejected");
    invalid = even;
    invalid.boundary_flags = pf3d::kBoundaryPeriodicXYZ3D;
    expect(pf3d::validate(invalid) ==
               pf3d::TwoRankValidation::WrongBoundaryFlags,
           "periodic-z geometry is rejected");
    expect(pf3d::validate(channel_geometry()) == pf3d::TwoRankValidation::Ok,
           "two-hard-wall geometry is supported");
    for (std::uint32_t flags : {0u, pf3d::kBoundaryPeriodicX3D,
             pf3d::kBoundaryChannelZ3D,
             pf3d::kBoundaryPeriodicXYZ3D | pf3d::kBoundaryChannelZ3D,
             pf3d::kBoundaryHardWallChannel3D | 16u}) {
        invalid = even;
        invalid.boundary_flags = flags;
        expect(pf3d::validate(invalid) ==
                   pf3d::TwoRankValidation::WrongBoundaryFlags,
               "incomplete, conflicting, or unknown boundary flags are rejected");
    }
    expect(pf3d::rank_ownership(even, 2).y_hi == 0,
           "invalid rank has no ownership");
}

void test_seam_union() {
    const auto bands = pf3d::seam_band_union(geometry(), 76);
    expect(bands.valid && !bands.full_domain,
           "B=152 seam bands are valid and disjoint");
    expect(bands.rows.count == 3 && bands.rows.total_rows == 304,
           "B=152 canonical union has 304 rows");
    expect(interval_is(bands.rows.pieces[0], 0, 76) &&
               interval_is(bands.rows.pieces[1], 382, 534) &&
               interval_is(bands.rows.pieces[2], 840, 916),
           "B=152 union contains the two periodic cuts once");

    const auto zero = pf3d::seam_band_union(geometry(), 0);
    expect(zero.valid && zero.rows.count == 0 && zero.rows.total_rows == 0,
           "zero-width bands are an empty valid union");
    expect(!pf3d::seam_band_union(geometry(), -1).valid,
           "negative half-width is rejected");

    const auto near_full = pf3d::seam_band_union(geometry(200), 49);
    expect(!near_full.full_domain && near_full.rows.total_rows == 196,
           "one row below the overlap threshold leaves four rows unsent");
    const auto full = pf3d::seam_band_union(geometry(200), 50);
    expect(full.full_domain && full.rows.count == 1 &&
               interval_is(full.rows.pieces[0], 0, 200),
           "full-domain fallback has one canonical interval");
}

void test_edge_resolution_and_z_envelopes() {
    pf3d::AllocatedBrick base{};
    base.origin_z = -76;
    pf3d::ZEnvelope envelope{};
    expect(pf3d::cell_allocated_z_envelope(base, kBaseEdge, kHeight,
                                            &envelope) &&
               envelope.lo == 0 && envelope.hi == 76,
           "base sentinel resolves to the runtime B=152 envelope");
    expect(pf3d::cell_allocated_z_envelope(base, 160, kHeight, &envelope) &&
               envelope.lo == 0 && envelope.hi == 84,
           "explicit runtime base edge overrides the model default");

    pf3d::AllocatedBrick promoted = base;
    promoted.origin_z = -112;
    promoted.storage_edge = 224;
    expect(pf3d::cell_allocated_z_envelope(promoted, kBaseEdge, kHeight,
                                             &envelope) &&
               envelope.lo == 0 && envelope.hi == 112,
           "promoted edge overrides the base sentinel");

    pf3d::AllocatedBrick recentered{};
    recentered.origin_z = 15;
    expect(pf3d::cell_allocated_z_envelope(recentered, kBaseEdge, kHeight,
                                            &envelope) &&
               envelope.lo == 15 && envelope.hi == 167,
           "accepted vertical recenter changes the nonzero z envelope");

    pf3d::AllocatedBrick above{};
    above.origin_z = std::numeric_limits<std::int64_t>::max() - 1;
    expect(pf3d::cell_allocated_z_envelope(above, kBaseEdge, kHeight,
                                            &envelope) &&
               pf3d::empty(envelope),
           "extreme positive origin is safely outside the domain");
    pf3d::AllocatedBrick below{};
    below.origin_z = std::numeric_limits<std::int64_t>::min();
    expect(pf3d::cell_allocated_z_envelope(below, kBaseEdge, kHeight,
                                            &envelope) &&
               pf3d::empty(envelope),
           "extreme negative origin is safely outside the domain");
    expect(!pf3d::cell_allocated_z_envelope(base, 0, kHeight, &envelope),
           "an unresolved base sentinel fails closed");

    pf3d::AllocatedBrick lower_clip{};
    lower_clip.origin_z = -10;
    expect(pf3d::cell_allocated_z_envelope(
               lower_clip, kBaseEdge, kHeight, &envelope) &&
               envelope.lo == 0 && envelope.hi == 142,
           "allocated z envelope clips at the substrate face");
    pf3d::AllocatedBrick upper_clip{};
    upper_clip.origin_z = 200;
    expect(pf3d::cell_allocated_z_envelope(
               upper_clip, kBaseEdge, kHeight, &envelope) &&
               envelope.lo == 200 && envelope.hi == 288,
           "allocated z envelope clips at the upper face");

}

void test_wrapping_classification_and_owner() {
    pf3d::AllocatedBrick wrapped{};
    wrapped.origin_y = -50;
    pf3d::WrappedRows rows{};
    expect(pf3d::cell_wrapped_y(wrapped, 100, kSide, &rows) &&
               rows.count == 2 && interval_is(rows.pieces[0], 866, 916) &&
               interval_is(rows.pieces[1], 0, 50),
           "negative origin wraps into two y intervals");

    const auto bands = pf3d::seam_band_union(geometry(), 76);
    pf3d::AllocatedBrick left_touch{};
    left_touch.origin_y = 230;  // [230,382), exactly below the interior band
    expect(pf3d::classify_cell(left_touch, kBaseEdge, geometry(), bands) ==
               pf3d::CellSeamClass::DeepInterior,
           "half-open contact below a seam is deep interior");
    pf3d::AllocatedBrick overlap = left_touch;
    overlap.origin_y = 231;
    expect(pf3d::classify_cell(overlap, kBaseEdge, geometry(), bands) ==
               pf3d::CellSeamClass::BandContributing,
           "one-row overlap contributes to the seam");
    expect(pf3d::classify_cell(overlap, 0, geometry(), bands) ==
               pf3d::CellSeamClass::Invalid,
           "unresolved cell cannot be classified as safe");
    expect(pf3d::classify_cell(overlap, kBaseEdge, geometry(917), bands) ==
               pf3d::CellSeamClass::Invalid,
           "bands from another geometry fail closed");

    int owner = -1;
    pf3d::AllocatedBrick before_cut{};
    before_cut.origin_y = 381;  // midpoint 457
    expect(pf3d::cell_owner_rank(before_cut, kBaseEdge, geometry(), &owner) &&
               owner == 0,
           "brick midpoint below the interior cut belongs to rank zero");
    before_cut.origin_y = 382;  // midpoint 458, exact tie
    expect(pf3d::cell_owner_rank(before_cut, kBaseEdge, geometry(), &owner) &&
               owner == 1,
           "interior-cut tie belongs to rank one");
    before_cut.origin_y = 840;  // midpoint wraps to row zero
    expect(pf3d::cell_owner_rank(before_cut, kBaseEdge, geometry(), &owner) &&
               owner == 0,
           "periodic-cut tie wraps to rank zero");

}

void test_complete_plans() {
    pf3d::AllocatedBrick initial[3]{};
    initial[0].origin_y = 300;
    initial[1].origin_y = 200;
    initial[2].origin_y = 850;
    for (auto& cell : initial) cell.origin_z = -76;
    const auto base = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, initial, 3, sizeof(std::uint32_t));
    expect(base.status == pf3d::SeamPlanStatus::Ok &&
               base.bands.half_width == 76 &&
               base.bands.rows.total_rows == 304,
           "base plan derives its band from the resolved runtime edge");
    expect(base.contributing_cells == 2 && base.deep_interior_cells == 1,
           "base plan classifies every supplied cell");
    expect(base.z.lo == 0 && base.z.hi == 76 &&
               base.payload_bytes == 85762048ULL,
           "base plan derives the exact 81.7890625 MiB payload");

    pf3d::AllocatedBrick elevated{};
    elevated.origin_y = 300;
    elevated.origin_z = 15;
    const auto elevated_plan = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, &elevated, 1, sizeof(std::uint32_t));
    expect(elevated_plan.status == pf3d::SeamPlanStatus::Ok &&
               elevated_plan.z.lo == 15 && elevated_plan.z.hi == 167 &&
               elevated_plan.payload_bytes == 171524096ULL,
           "plan follows an accepted nonzero vertical origin");

    pf3d::AllocatedBrick mixed[2]{};
    mixed[0].origin_y = 300;
    mixed[0].origin_z = -76;
    mixed[1].origin_y = 200;
    mixed[1].origin_z = -112;
    mixed[1].storage_edge = 224;
    const auto promoted = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, mixed, 2, sizeof(std::uint32_t));
    expect(promoted.status == pf3d::SeamPlanStatus::Ok &&
               promoted.bands.half_width == 112 &&
               promoted.bands.rows.total_rows == 448 &&
               promoted.z.lo == 0 && promoted.z.hi == 112 &&
               promoted.payload_bytes == 186253312ULL,
           "mixed plan expands to the E=224 allocated envelope");

    pf3d::AllocatedBrick edge280{};
    edge280.origin_y = 300;
    edge280.origin_z = -140;
    edge280.storage_edge = 280;
    const auto promoted280 = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, &edge280, 1, sizeof(std::uint32_t));
    expect(promoted280.status == pf3d::SeamPlanStatus::Ok &&
               promoted280.bands.rows.total_rows == 560 &&
               promoted280.z.hi == 140 &&
               promoted280.payload_bytes == 291020800ULL,
           "E=280 plan derives the 277.5390625 MiB payload");

    bool all_contributing = true;
    const auto bands280 = pf3d::seam_band_union(geometry(), 140);
    for (int origin = 0; origin < kSide; ++origin) {
        edge280.origin_y = origin;
        if (pf3d::classify_cell(edge280, kBaseEdge, geometry(), bands280) !=
            pf3d::CellSeamClass::BandContributing) {
            all_contributing = false;
            break;
        }
    }
    expect(all_contributing,
           "every E=280 origin contributes to one of the two seam bands");

    const auto empty_plan = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, nullptr, 0, sizeof(std::uint32_t));
    expect(empty_plan.status == pf3d::SeamPlanStatus::Ok &&
               empty_plan.payload_bytes == 0,
           "an empty ownership cohort sends no payload");
    expect(pf3d::build_seam_exchange_plan(
               geometry(), kBaseEdge, 0, nullptr, 0, 0).status ==
               pf3d::SeamPlanStatus::InvalidElementSize,
           "zero-byte elements are rejected for an empty cohort");
    expect(pf3d::build_seam_exchange_plan(
               geometry(), kBaseEdge, 0, nullptr, 1,
               sizeof(std::uint32_t)).status ==
               pf3d::SeamPlanStatus::InvalidCellArray,
           "a missing non-empty cohort is rejected");
    pf3d::AllocatedBrick invalid{};
    invalid.storage_edge = std::numeric_limits<std::uint32_t>::max();
    expect(pf3d::build_seam_exchange_plan(
               geometry(), kBaseEdge, 0, &invalid, 1,
               sizeof(std::uint32_t)).status ==
               pf3d::SeamPlanStatus::InvalidCell,
           "an unrepresentable storage edge is rejected");

    const auto other_rank = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 1, initial, 3, sizeof(std::uint32_t));
    expect(other_rank.status == pf3d::SeamPlanStatus::Ok &&
               other_rank.owned_cells == 0 && other_rank.payload_bytes == 0,
           "the planner derives an empty cohort for the other GPU");
    expect(pf3d::build_seam_exchange_plan(
               geometry(), kBaseEdge, 2, nullptr, 0,
               sizeof(std::uint32_t)).status ==
               pf3d::SeamPlanStatus::InvalidRank,
           "only the two supported owner ranks are accepted");

    pf3d::AllocatedBrick unaligned{};
    unaligned.storage_edge = 153;
    expect(pf3d::build_seam_exchange_plan(
               geometry(), kBaseEdge, 0, &unaligned, 1,
               sizeof(std::uint32_t)).status ==
               pf3d::SeamPlanStatus::InvalidCell,
           "unaligned storage edges are rejected");

    // Both directions must use the global maximum edge. Otherwise the rank-1
    // base cell would appear deep at h=76 even though it overlaps rank 0's
    // promoted field on rows [534,597).
    pf3d::AllocatedBrick asymmetric[2]{};
    asymmetric[0].origin_y = 317;
    asymmetric[0].origin_z = -140;
    asymmetric[0].storage_edge = 280;
    asymmetric[1].origin_y = 534;
    asymmetric[1].origin_z = -76;
    const auto rank_zero = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, asymmetric, 2, sizeof(std::uint32_t));
    const auto rank_one = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 1, asymmetric, 2, sizeof(std::uint32_t));
    expect(rank_zero.status == pf3d::SeamPlanStatus::Ok &&
               rank_one.status == pf3d::SeamPlanStatus::Ok &&
               rank_zero.bands.half_width == 140 &&
               rank_one.bands.half_width == 140,
           "asymmetric ranks agree on the global promoted band width");
    expect(rank_one.owned_cells == 1 && rank_one.contributing_cells == 1,
           "rank-one base cell is retained in the widened seam exchange");

}

void test_exhaustive_seam_coverage() {
    bool plans_valid = true;
    bool remote_rows_covered = true;
    const int sides[] = {916, 917};
    const int edges[] = {152, 224, 280};
    for (int side : sides) {
        const auto domain = geometry(side);
        const int cut = side / 2;
        for (int edge : edges) {
            for (int origin = 0; origin < side; ++origin) {
                pf3d::AllocatedBrick cell{};
                cell.origin_y = origin;
                cell.origin_z = -edge / 2;
                cell.storage_edge = edge == 152
                                        ? 0u
                                        : static_cast<std::uint32_t>(edge);
                int owner = -1;
                if (!pf3d::cell_owner_rank(cell, kBaseEdge, domain, &owner)) {
                    plans_valid = false;
                    continue;
                }
                const auto plan = pf3d::build_seam_exchange_plan(
                    domain, kBaseEdge, owner, &cell, 1,
                    sizeof(std::uint32_t));
                if (plan.status != pf3d::SeamPlanStatus::Ok ||
                    plan.owned_cells != 1) {
                    plans_valid = false;
                    continue;
                }

                pf3d::WrappedRows position{};
                if (!pf3d::cell_wrapped_y(cell, kBaseEdge, side,
                                          &position)) {
                    plans_valid = false;
                    continue;
                }
                bool reaches_other_rank = false;
                for (int piece = 0; piece < position.count; ++piece) {
                    for (int row = position.pieces[piece].lo;
                         row < position.pieces[piece].hi; ++row) {
                        const int spatial_owner = row < cut ? 0 : 1;
                        if (spatial_owner != owner) {
                            reaches_other_rank = true;
                            if (!contains(plan.bands.rows, row)) {
                                remote_rows_covered = false;
                            }
                        }
                    }
                }
                if (reaches_other_rank && plan.contributing_cells != 1)
                    plans_valid = false;
            }
        }
    }
    expect(plans_valid, "all odd/even edge plans remain valid");
    expect(remote_rows_covered,
           "all allocated rows crossing either ownership cut are exchanged");
}

void test_payload_arithmetic() {
    std::size_t bytes = 0;
    expect(pf3d::pitched_payload_bytes(304, kPitch, kHeight, 4, &bytes) &&
               bytes == 324993024ULL,
           "full-z seam payload is exact");
    expect(pf3d::pitched_payload_bytes(kSide, kPitch, kHeight, 4, &bytes) &&
               bytes == 979255296ULL,
           "full-S payload is exact");
    expect(!pf3d::pitched_payload_bytes(0, kPitch, 76, 4, &bytes) && bytes == 0,
           "zero rows are rejected");
    expect(!pf3d::pitched_payload_bytes(
               std::numeric_limits<int>::max(),
               std::numeric_limits<int>::max(),
               std::numeric_limits<int>::max(), 4, &bytes) && bytes == 0,
           "payload overflow is rejected");
}

void test_epoch_plans() {
    pf3d::AllocatedBrick cells[2]{{300, -76, 0}, {200, 80, 0}};
    const auto zero = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, cells, 2, sizeof(std::uint32_t), 0);
    const auto swept = pf3d::build_seam_exchange_plan(
        geometry(), kBaseEdge, 0, cells, 2, sizeof(std::uint32_t), 16);
    expect(zero.z.hi == 76 && zero.deep_interior_cells == 1,
           "zero displacement retains the contributing-only z envelope");
    expect(swept.status == pf3d::SeamPlanStatus::Ok &&
               swept.bands.half_width == 92 && swept.z.lo == 0 &&
               swept.z.hi == 248 && swept.contributing_cells == 2,
           "epoch envelope includes an initially deep elevated cell");
    for (int displacement : {0, 4, 16, 32}) {
        const auto no_owner = pf3d::build_seam_exchange_plan(
            geometry(), kBaseEdge, 1, cells, 2, sizeof(std::uint32_t),
            displacement);
        expect(no_owner.status == pf3d::SeamPlanStatus::Ok &&
                   no_owner.owned_cells == 0 && no_owner.payload_bytes == 0 &&
                   pf3d::empty(no_owner.z),
               "a zero-owner epoch has no transfer or z envelope");
    }
    expect(pf3d::build_seam_exchange_plan(
               geometry(), kBaseEdge, 0, cells, 2, sizeof(std::uint32_t), -1)
               .status == pf3d::SeamPlanStatus::InvalidDisplacement,
           "a negative displacement bound is rejected");
    expect(pf3d::build_seam_exchange_plan(
               geometry(), kBaseEdge, 0, cells, 2, sizeof(std::uint32_t),
               std::numeric_limits<std::int64_t>::max())
               .status == pf3d::SeamPlanStatus::SizeOverflow,
           "a seam half-width overflow is rejected");
    const auto full = pf3d::build_seam_exchange_plan(
        geometry(917), kBaseEdge, 0, cells, 2, sizeof(std::uint32_t), 1000);
    expect(full.status == pf3d::SeamPlanStatus::Ok &&
               full.bands.full_domain && full.z.lo == 0 &&
               full.z.hi == kHeight &&
               full.payload_bytes == static_cast<std::size_t>(917) *
                   kPitch * kHeight * sizeof(std::uint32_t),
           "a large displacement safely falls back to the full padded domain");

    pf3d::ZEnvelope z{};
    const std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
    const pf3d::AllocatedBrick below{
        0, std::numeric_limits<std::int64_t>::min(), 0};
    const pf3d::AllocatedBrick above{0, maximum, 0};
    expect(pf3d::cell_allocated_z_envelope(
               below, kBaseEdge, kHeight, &z, maximum) &&
               z.lo == 0 && z.hi == 151,
           "swept z bounds handle the minimum int64 origin without overflow");
    expect(pf3d::cell_allocated_z_envelope(
               above, kBaseEdge, kHeight, &z, maximum) &&
               z.lo == 0 && z.hi == kHeight,
           "swept z bounds saturate positive int64 overflow before clipping");
    expect(!pf3d::cell_allocated_z_envelope(
               cells[0], kBaseEdge, kHeight, &z, -1),
           "the z-envelope helper rejects a negative displacement");
}

void test_exhaustive_epoch_coverage() {
    bool valid = true, covered_y = true, covered_z = true, zero_unchanged = true;
    bool crossed_periodic = false, crossed_interior = false;
    for (int side : {916, 917}) {
        const auto domain = geometry(side);
        for (int edge : {152, 224, 280}) {
            for (int origin = 0; origin < side; ++origin) {
                const pf3d::AllocatedBrick cell{
                    origin, -edge / 2, static_cast<std::uint32_t>(edge)};
                int frozen_owner = -1;
                valid = pf3d::cell_owner_rank(
                    cell, kBaseEdge, domain, &frozen_owner) && valid;
                const auto original = pf3d::build_seam_exchange_plan(
                    domain, kBaseEdge, frozen_owner, &cell, 1,
                    sizeof(std::uint32_t));
                for (int displacement : {0, 4, 16, 32}) {
                    const auto plan = pf3d::build_seam_exchange_plan(
                        domain, kBaseEdge, frozen_owner, &cell, 1,
                        sizeof(std::uint32_t), displacement);
                    valid = plan.status == pf3d::SeamPlanStatus::Ok && valid;
                    if (displacement == 0)
                        zero_unchanged = same_plan(original, plan) && zero_unchanged;
                    for (int dy = -displacement; dy <= displacement; ++dy) {
                        auto translated = cell;
                        translated.origin_y += dy;
                        int moved_owner = -1;
                        valid = pf3d::cell_owner_rank(
                            translated, kBaseEdge, domain, &moved_owner) && valid;
                        if (moved_owner != frozen_owner) {
                            const int old_midpoint = (origin + edge / 2) % side;
                            if (old_midpoint + dy < 0 || old_midpoint + dy >= side)
                                crossed_periodic = true;
                            else
                                crossed_interior = true;
                        }
                        pf3d::WrappedRows rows{};
                        valid = pf3d::cell_wrapped_y(
                            translated, kBaseEdge, side, &rows) && valid;
                        for (int piece = 0; piece < rows.count; ++piece)
                            for (int row = rows.pieces[piece].lo;
                                 row < rows.pieces[piece].hi; ++row)
                                if ((row < side / 2 ? 0 : 1) != frozen_owner)
                                    covered_y = contains(plan.bands.rows, row) &&
                                                covered_y;
                    }
                }
            }
        }
    }
    // Axis-aligned swept boxes are Cartesian products: exhaustive independent
    // y and z translations cover their combinations without a voxel-volume loop.
    for (int edge : {152, 224, 280}) {
        for (int origin_z = 1 - edge; origin_z < kHeight; ++origin_z) {
            const pf3d::AllocatedBrick cell{
                400, origin_z, static_cast<std::uint32_t>(edge)};
            for (int displacement : {0, 4, 16, 32}) {
                pf3d::ZEnvelope swept{};
                valid = pf3d::cell_allocated_z_envelope(
                    cell, kBaseEdge, kHeight, &swept, displacement) && valid;
                for (int dz = -displacement; dz <= displacement; ++dz) {
                    auto translated = cell;
                    translated.origin_z += dz;
                    pf3d::ZEnvelope current{};
                    valid = pf3d::cell_allocated_z_envelope(
                        translated, kBaseEdge, kHeight, &current) && valid;
                    if (!pf3d::empty(current))
                        covered_z = current.lo >= swept.lo &&
                                    current.hi <= swept.hi && covered_z;
                }
            }
        }
    }
    expect(valid, "all translated odd/even base/promoted epoch plans are valid");
    expect(zero_unchanged, "explicit zero displacement leaves every plan field unchanged");
    expect(covered_y && crossed_periodic && crossed_interior,
           "frozen-owner transfer bands cover all bounded translations across both cuts");
    expect(covered_z, "swept z envelopes cover every bounded translated physical slice");
}

void test_mixed_epoch_coverage() {
    bool valid = true, covered = true;
    for (int side : {916, 917}) {
        const auto domain = geometry(side);
        const pf3d::AllocatedBrick cells[] = {
            {side / 2 - 77, -76, 0}, {side / 2 - 111, 60, 224},
            {side - 77, 200, 0}, {side - 139, -140, 280}};
        for (int displacement : {0, 4, 16, 32}) {
            const pf3d::SeamExchangePlan plans[] = {
                pf3d::build_seam_exchange_plan(
                    domain, kBaseEdge, 0, cells, 4, sizeof(std::uint32_t), displacement),
                pf3d::build_seam_exchange_plan(
                    domain, kBaseEdge, 1, cells, 4, sizeof(std::uint32_t), displacement)};
            valid = plans[0].status == pf3d::SeamPlanStatus::Ok &&
                    plans[1].status == pf3d::SeamPlanStatus::Ok &&
                    plans[0].bands.half_width == 140 + displacement &&
                    plans[1].bands.half_width == 140 + displacement && valid;
            for (const auto& cell : cells) {
                int owner = -1;
                valid = pf3d::cell_owner_rank(cell, kBaseEdge, domain, &owner) && valid;
                const auto& plan = plans[owner];
                for (int delta = -displacement; delta <= displacement; ++delta) {
                    auto translated = cell;
                    translated.origin_y += delta;
                    translated.origin_z += delta;
                    pf3d::WrappedRows rows{};
                    pf3d::ZEnvelope z{};
                    valid = pf3d::cell_wrapped_y(
                        translated, kBaseEdge, side, &rows) && valid;
                    valid = pf3d::cell_allocated_z_envelope(
                        translated, kBaseEdge, kHeight, &z) && valid;
                    if (!pf3d::empty(z))
                        covered = z.lo >= plan.z.lo && z.hi <= plan.z.hi && covered;
                    for (int piece = 0; piece < rows.count; ++piece)
                        for (int row = rows.pieces[piece].lo; row < rows.pieces[piece].hi; ++row)
                            if ((row < side / 2 ? 0 : 1) != owner)
                                covered = contains(plan.bands.rows, row) && covered;
                }
            }
        }
    }
    expect(valid && covered,
           "asymmetric mixed-edge cohorts share a safe frozen-owner y/z envelope");
}

void test_channel_plans() {
    const auto domain = channel_geometry();
    const pf3d::AllocatedBrick cells[] = {
        {286, -10, 0},       // B=160 spans both z faces; midpoint at the y cut.
        {-112, -42, 224}};   // Promoted brick crosses the periodic y cut.
    const auto lower = pf3d::build_seam_exchange_plan(
        domain, 160, 0, cells, 2, sizeof(std::uint32_t));
    const auto upper = pf3d::build_seam_exchange_plan(
        domain, 160, 1, cells, 2, sizeof(std::uint32_t));
    expect(lower.status == pf3d::SeamPlanStatus::Ok &&
               upper.status == pf3d::SeamPlanStatus::Ok &&
               lower.owned_cells == 1 && upper.owned_cells == 1 &&
               lower.contributing_cells == 1 && upper.contributing_cells == 1,
           "channel accepts base and promoted bricks taller than Nz");
    expect(lower.z.lo == 0 && lower.z.hi == domain.nz &&
               upper.z.lo == 0 && upper.z.hi == domain.nz,
           "both ranks clamp oversized bricks to the physical aggregate domain");
    expect(lower.bands.rows.total_rows == 448 &&
               upper.bands.rows.total_rows == 448 &&
               lower.payload_bytes == 448ULL * 736 * 140 * 4 &&
               upper.payload_bytes == lower.payload_bytes,
           "channel exchange spans all pitched x rows, including periodic x images");
    expect(lower.bands.rows.count == 3 &&
               interval_is(lower.bands.rows.pieces[0], 0, 112) &&
               interval_is(lower.bands.rows.pieces[1], 254, 478) &&
               interval_is(lower.bands.rows.pieces[2], 621, 733),
           "odd channel geometry retains disjoint interior and periodic seams");

    auto substrate = domain;
    substrate.boundary_flags = pf3d::kBoundarySubstrateSlab3D;
    expect(pf3d::build_seam_exchange_plan(
               substrate, 160, 0, cells, 2, sizeof(std::uint32_t)).status ==
               pf3d::SeamPlanStatus::InvalidBaseEdge,
           "substrate retains its prior minimum-domain-edge guard");
    const auto small_domain = channel_geometry(160);
    expect(pf3d::build_seam_exchange_plan(
               small_domain, 160, 0, nullptr, 0, sizeof(std::uint32_t)).status ==
               pf3d::SeamPlanStatus::InvalidBaseEdge,
           "base edge equal to a periodic side is rejected");
    const pf3d::AllocatedBrick too_wide{0, -10, 736};
    expect(pf3d::build_seam_exchange_plan(
               domain, 160, 0, &too_wide, 1, sizeof(std::uint32_t)).status ==
               pf3d::SeamPlanStatus::InvalidCell,
           "promoted edge exceeding a periodic side is rejected");

    for (int edge : {160, 224}) {
        for (int origin_z : {-edge + 1, -10, 0, 130}) {
            const pf3d::AllocatedBrick cell{
                366 - edge / 2, origin_z, static_cast<std::uint32_t>(edge)};
            const auto plan = pf3d::build_seam_exchange_plan(
                domain, 160, 1, &cell, 1, sizeof(std::uint32_t));
            const int expected_lo = origin_z < 0 ? 0 : origin_z;
            const int expected_hi = origin_z + edge > domain.nz
                ? domain.nz : origin_z + edge;
            expect(plan.status == pf3d::SeamPlanStatus::Ok &&
                       plan.z.lo == expected_lo && plan.z.hi == expected_hi,
                   "channel clips partial base/promoted bricks without z wrapping");
        }
    }
}

void test_channel_swept_coverage() {
    bool valid = true, covered_y = true, covered_z = true;
    bool wrapped_y = false, partial_band = false, below = false, above = false;
    for (int side : {733, 734}) {
        const auto domain = channel_geometry(side);
        for (int edge : {160, 192, 224, 280}) {
            for (int origin_y = -edge; origin_y < side; ++origin_y) {
                for (int displacement : {0, 16}) {
                    const pf3d::AllocatedBrick cell{
                        origin_y, (domain.nz - edge) / 2,
                        static_cast<std::uint32_t>(edge)};
                    int owner = -1;
                    valid = pf3d::cell_owner_rank(cell, 160, domain, &owner) && valid;
                    const auto plan = pf3d::build_seam_exchange_plan(
                        domain, 160, owner, &cell, 1,
                        sizeof(std::uint32_t), displacement);
                    valid = plan.status == pf3d::SeamPlanStatus::Ok && valid;
                    partial_band = partial_band || !plan.bands.full_domain;
                    for (int dy : {-displacement, 0, displacement}) {
                        auto moved = cell;
                        moved.origin_y += dy;
                        pf3d::WrappedRows rows{};
                        valid = pf3d::cell_wrapped_y(moved, 160, side, &rows) && valid;
                        wrapped_y = wrapped_y || rows.count == 2;
                        for (int piece = 0; piece < rows.count; ++piece) {
                            for (int row = rows.pieces[piece].lo;
                                 row < rows.pieces[piece].hi; ++row) {
                                if ((row < side / 2 ? 0 : 1) != owner)
                                    covered_y = plan.contributing_cells == 1 &&
                                                contains(plan.bands.rows, row) && covered_y;
                            }
                        }
                    }
                }
            }
            for (int origin_z = 1 - edge; origin_z < domain.nz; ++origin_z) {
                const pf3d::AllocatedBrick cell{
                    side / 2 - edge / 2, origin_z,
                    static_cast<std::uint32_t>(edge)};
                for (int displacement : {0, 16}) {
                    const auto plan = pf3d::build_seam_exchange_plan(
                        domain, 160, 1, &cell, 1,
                        sizeof(std::uint32_t), displacement);
                    valid = plan.status == pf3d::SeamPlanStatus::Ok && valid;
                    for (int dz = -displacement; dz <= displacement; ++dz) {
                        const int lo = origin_z + dz > 0 ? origin_z + dz : 0;
                        const int hi = origin_z + dz + edge < domain.nz
                            ? origin_z + dz + edge : domain.nz;
                        if (hi > lo)
                            covered_z = lo >= plan.z.lo && hi <= plan.z.hi && covered_z;
                        below = below || origin_z + dz < 0;
                        above = above || origin_z + dz + edge > domain.nz;
                    }
                }
            }
        }
    }
    expect(valid && wrapped_y && partial_band && covered_y,
           "channel plans cover wrapped y rows across both partial exchange seams");
    expect(covered_z && below && above,
           "channel plans cover every live z slice during bounded motion past either face");
}

}  // namespace

int main() {
    test_geometry_and_ownership();
    test_seam_union();
    test_edge_resolution_and_z_envelopes();
    test_wrapping_classification_and_owner();
    test_complete_plans();
    test_exhaustive_seam_coverage();
    test_payload_arithmetic();
    test_epoch_plans();
    test_exhaustive_epoch_coverage();
    test_mixed_epoch_coverage();
    test_channel_plans();
    test_channel_swept_coverage();
    if (failures == 0)
        std::printf("substrate_planner_cpu: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
