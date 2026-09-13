#include "pf3d/reduction_mode.hpp"

#include <cstdio>
#include <initializer_list>

namespace {

int failures = 0;
int checks = 0;

void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

using Reduction = pf3d::PromotedMeasureReduction3D;
using Result = pf3d::ReductionResumeResult;

bool equal(Reduction a, Reduction b) {
    return a.policy == b.policy && a.auto_wave_ctas == b.auto_wave_ctas;
}

void restores_and_explicit_noops() {
    // A different device must not replace a checkpoint's resolved auto wave.
    for (Reduction stored : {Reduction{4, 0}, Reduction{64, 0},
                             Reduction{-1, 264}}) {
        Reduction resolved{17, 123};
        expect(pf3d::resolve_promoted_measure_resume(
                   stored, false, 0, &resolved) == Result::Ok &&
                   equal(resolved, stored),
               "omitted option restores stored 4, 64, or automatic wave");
        for (bool allow_regroup : {false, true}) {
            resolved = {17, 123};
            expect(pf3d::resolve_promoted_measure_resume(
                       stored, true, stored.policy, &resolved,
                       allow_regroup) == Result::Ok && equal(resolved, stored),
                   "same policy preserves the complete stored contract");
        }
    }
}

void authorized_changes() {
    const Reduction contracts[] = {{0, 0}, {1, 0}, {4, 0}, {64, 0}, {-1, 264}};
    for (Reduction stored : contracts) {
        for (Reduction requested : contracts) {
            if (stored.policy == requested.policy) continue;
            Reduction resolved{17, 123};
            expect(pf3d::resolve_promoted_measure_resume(
                       stored, true, requested.policy, &resolved) ==
                       Result::Mismatch && equal(resolved, {17, 123}),
                   "default-false API rejects changes without modifying output");
            expect(pf3d::resolve_promoted_measure_resume(
                       stored, true, requested.policy, &resolved, false) ==
                       Result::Mismatch && equal(resolved, {17, 123}),
                   "explicit false keeps the old mismatch guard");
            expect(pf3d::resolve_promoted_measure_resume(
                       stored, true, requested.policy, &resolved, true) ==
                       Result::Ok && equal(resolved, {requested.policy, 0}),
                   "authorized fixed/auto regroup selects policy and clears wave");
        }
    }
}

void invalid_contracts_and_requests() {
    const Reduction invalid[] = {{-2, 0}, {65, 0}, {4, 1}, {64, -1},
                                 {-1, 0}, {-1, -1}};
    for (Reduction stored : invalid) {
        for (bool allow_regroup : {false, true}) {
            Reduction resolved{17, 123};
            expect(pf3d::resolve_promoted_measure_resume(
                       stored, true, 64, &resolved, allow_regroup) ==
                       Result::InvalidCheckpoint && equal(resolved, {17, 123}),
                   "regroup never legitimizes an invalid stored contract");
        }
    }
    for (bool allow_regroup : {false, true}) {
        expect(pf3d::resolve_promoted_measure_resume(
                   {4, 0}, true, 64, nullptr, allow_regroup) ==
                   Result::InvalidCheckpoint,
               "null output is rejected");
        for (int requested : {-2, 65}) {
            Reduction resolved{17, 123};
            expect(pf3d::resolve_promoted_measure_resume(
                       {4, 0}, true, requested, &resolved, allow_regroup) ==
                       Result::InvalidRequest && equal(resolved, {17, 123}),
                   "explicit invalid request fails even with authorization");
        }
    }
    for (Reduction stored : {Reduction{4, 0}, Reduction{64, 0},
                             Reduction{-1, 264}}) {
        Reduction resolved{17, 123};
        expect(pf3d::resolve_promoted_measure_resume(
                   stored, false, stored.policy, &resolved, true) ==
                   Result::InvalidRequest && equal(resolved, {17, 123}),
               "authorization requires an explicitly supplied policy");
    }
    Reduction resolved{};
    expect(pf3d::resolve_promoted_measure_resume(
               {4, 0}, false, 65, &resolved) == Result::Ok &&
               equal(resolved, {4, 0}),
           "an unsupplied option is ignored without regroup authorization");
}

}  // namespace

int main() {
    restores_and_explicit_noops();
    authorized_changes();
    invalid_contracts_and_requests();
    std::printf("promoted_regroup_cpu: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
