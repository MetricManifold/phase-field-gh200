#pragma once

#include "checkpoint_format_3d.h"

#include <cstdint>

namespace pf3d {

// Numerical contract for the promoted-cell moment reduction. A negative
// policy selects the automatic shard count; its resolved occupancy wave is
// stored so restart does not depend on the destination device.
struct PromotedMeasureReduction3D {
    int policy = 0;          // 0=one CTA, 1..64=fixed, -1=automatic
    int auto_wave_ctas = 0;  // positive only for a resolved automatic policy
};

constexpr int kPromotedMeasureAuto = -1;
constexpr int kMaximumPromotedMeasureShards =
    static_cast<int>(ckpt3d::kMaximumPromotedMeasureShards);

constexpr bool valid_promoted_measure_policy(int policy) {
    return policy >= kPromotedMeasureAuto &&
           policy <= kMaximumPromotedMeasureShards;
}

constexpr bool valid_fresh_promoted_measure_reduction(
    const PromotedMeasureReduction3D& reduction) {
    return valid_promoted_measure_policy(reduction.policy) &&
           reduction.auto_wave_ctas == 0;
}

constexpr bool valid_checkpoint_promoted_measure_reduction(
    const PromotedMeasureReduction3D& reduction) {
    return valid_promoted_measure_policy(reduction.policy) &&
           (reduction.policy == kPromotedMeasureAuto
                ? reduction.auto_wave_ctas > 0
                : reduction.auto_wave_ctas == 0);
}

constexpr std::uint64_t encode_promoted_measure_policy(int policy) {
    return policy == kPromotedMeasureAuto
        ? ckpt3d::kPromotedMeasurePolicyAuto
        : static_cast<std::uint64_t>(policy);
}

constexpr PromotedMeasureReduction3D decode_promoted_measure_reduction(
    std::uint64_t policy, std::uint64_t auto_wave_ctas) {
    return {
        policy == ckpt3d::kPromotedMeasurePolicyAuto
            ? kPromotedMeasureAuto : static_cast<int>(policy),
        static_cast<int>(auto_wave_ctas)};
}

enum class ReductionResumeResult {
    Ok,
    InvalidCheckpoint,
    Mismatch,
    InvalidRequest
};

// Restore the stored grouping unless an explicit policy change is authorized.
// A changed automatic policy resolves on the destination device; requesting
// the same automatic policy always preserves the checkpoint's stored wave.
constexpr ReductionResumeResult resolve_promoted_measure_resume(
    const PromotedMeasureReduction3D& stored,
    bool option_was_supplied, int requested_policy,
    PromotedMeasureReduction3D* resolved, bool allow_regroup = false) {
    if (!resolved ||
        !valid_checkpoint_promoted_measure_reduction(stored)) {
        return ReductionResumeResult::InvalidCheckpoint;
    }
    if ((allow_regroup && !option_was_supplied) ||
        (option_was_supplied && !valid_promoted_measure_policy(requested_policy)))
        return ReductionResumeResult::InvalidRequest;
    if (option_was_supplied && requested_policy != stored.policy) {
        if (!allow_regroup) return ReductionResumeResult::Mismatch;
        *resolved = {requested_policy, 0};
    } else {
        *resolved = stored;
    }
    return ReductionResumeResult::Ok;
}

}  // namespace pf3d
