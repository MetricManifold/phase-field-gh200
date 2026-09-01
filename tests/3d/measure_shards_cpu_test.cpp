#include "pf3d/measure_shards.hpp"
#include "pf3d/trajectory_header.hpp"
#include "checkpoint_format_3d.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[FAIL] %s\n", message);
        ++failures;
    }
}

// GH200 configuration used to exercise the occupancy-derived policies.
constexpr int kSMs = 132;
constexpr int kOcc = 2;

void policy_validation() {
    expect(pf3d::valid_base_measure_policy(-1), "policy -1 valid");
    expect(pf3d::valid_base_measure_policy(0), "policy 0 valid");
    expect(pf3d::valid_base_measure_policy(1), "policy 1 valid");
    expect(pf3d::valid_base_measure_policy(64), "policy 64 valid");
    expect(!pf3d::valid_base_measure_policy(-2), "policy -2 invalid");
    expect(!pf3d::valid_base_measure_policy(65), "policy 65 invalid");
}

void default_resolution_unchanged() {
    // Compare with the original at-most-four formula for every population.
    for (int cells = 1; cells <= 1024; ++cells) {
        const std::int64_t wave = static_cast<std::int64_t>(kSMs) * kOcc;
        std::int64_t expected = wave / cells;
        if (expected > 4) expected = 4;
        if (expected < 1) expected = 1;
        if (pf3d::resolve_base_measure_shards(0, cells, kSMs, kOcc) !=
            static_cast<int>(expected)) {
            expect(false, "default resolution deviates from the at-most-four "
                          "policy");
            return;
        }
    }
    expect(pf3d::resolve_base_measure_shards(0, 24, kSMs, kOcc) == 4,
           "default N=24 resolves 4");
    expect(pf3d::resolve_base_measure_shards(0, 100, kSMs, kOcc) == 2,
           "default N=100 resolves 2");
    expect(pf3d::resolve_base_measure_shards(0, 200, kSMs, kOcc) == 1,
           "default N=200 resolves 1");
}

void wave_and_fixed_resolution() {
    expect(pf3d::resolve_base_measure_shards(-1, 24, kSMs, kOcc) == 11,
           "wave-fitting policy at N=24 resolves 11");
    expect(pf3d::resolve_base_measure_shards(-1, 100, kSMs, kOcc) == 2,
           "wave-fitting policy at N=100 resolves 2");
    expect(pf3d::resolve_base_measure_shards(-1, 300, kSMs, kOcc) == 1,
           "wave-fitting policy floors at 1 for multi-wave populations");
    expect(pf3d::resolve_base_measure_shards(-1, 1, kSMs, kOcc) == 64,
           "wave-fitting policy caps at 64");
    expect(pf3d::resolve_base_measure_shards(1, 24, kSMs, kOcc) == 1,
           "fixed 1 resolves 1");
    expect(pf3d::resolve_base_measure_shards(64, 24, kSMs, kOcc) == 64,
           "fixed 64 resolves 64");
}

void checkpoint_value_validation() {
    expect(!pf3d::valid_checkpoint_base_measure_shards(-2),
           "checkpoint value -2 rejected");
    expect(!pf3d::valid_checkpoint_base_measure_shards(-1),
           "checkpoint value -1 rejected");
    expect(pf3d::valid_checkpoint_base_measure_shards(0),
           "checkpoint value 0 valid");
    expect(pf3d::valid_checkpoint_base_measure_shards(1),
           "checkpoint value 1 valid");
    expect(pf3d::valid_checkpoint_base_measure_shards(64),
           "checkpoint value 64 valid");
    expect(!pf3d::valid_checkpoint_base_measure_shards(65),
           "checkpoint value 65 rejected");
}

void stored_validation() {
    expect(pf3d::valid_stored_base_measure_shards(0),
           "stored 0 selects the standard policy");
    expect(pf3d::valid_stored_base_measure_shards(11), "stored 11 valid");
    expect(pf3d::valid_stored_base_measure_shards(64), "stored 64 valid");
    expect(!pf3d::valid_stored_base_measure_shards(65), "stored 65 rejected");
    expect(!pf3d::valid_stored_base_measure_shards(1ull << 40),
           "stored huge value rejected");
}

void resume_rules() {
    using pf3d::BaseMeasureResumeResult;
    int resolved = 0;

    // Omitted option restores a stored count exactly.
    expect(pf3d::resolve_base_measure_resume(11, false, 0, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Ok && resolved == 11,
           "omitted option restores stored 11");
    // ...also on hardware whose derived occupancy differs, so an automatic
    // policy can never silently regroup a continuation.
    expect(pf3d::resolve_base_measure_resume(11, false, 0, 24, kSMs, 1,
                                             &resolved) ==
               BaseMeasureResumeResult::Ok && resolved == 11,
           "different occupancy cannot regroup a stored count");
    // Stored zero resolves the standard policy.
    expect(pf3d::resolve_base_measure_resume(0, false, 0, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Ok && resolved == 4,
           "stored zero resolves the at-most-four policy");

    // Explicit matching overrides succeed.
    expect(pf3d::resolve_base_measure_resume(11, true, 11, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Ok && resolved == 11,
           "explicit matching fixed count succeeds");
    expect(pf3d::resolve_base_measure_resume(11, true, -1, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Ok && resolved == 11,
           "matching one-wave request succeeds");
    expect(pf3d::resolve_base_measure_resume(0, true, 0, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Ok && resolved == 4,
           "explicit 0 matches a stored standard policy");
    expect(pf3d::resolve_base_measure_resume(0, true, 4, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Ok && resolved == 4,
           "explicit 4 matches the effective standard count at N=24");

    // Incompatible overrides fail before any step.
    expect(pf3d::resolve_base_measure_resume(11, true, 4, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Mismatch,
           "explicit 4 against stored 11 fails");
    expect(pf3d::resolve_base_measure_resume(11, true, 0, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Mismatch,
           "explicit 0 against stored 11 fails");
    expect(pf3d::resolve_base_measure_resume(0, true, -1, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::Mismatch,
           "one-wave request against stored standard policy fails at N=24");
    expect(pf3d::resolve_base_measure_resume(11, true, -1, 24, kSMs, 1,
                                             &resolved) ==
               BaseMeasureResumeResult::Mismatch,
           "one-wave request on different occupancy fails loudly");

    // Corrupt inputs.
    expect(pf3d::resolve_base_measure_resume(65, false, 0, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::InvalidCheckpoint,
           "stored 65 rejected");
    expect(pf3d::resolve_base_measure_resume(11, true, -2, 24, kSMs, kOcc,
                                             &resolved) ==
               BaseMeasureResumeResult::InvalidCheckpoint,
           "invalid requested policy rejected");
}

void checksum_covers_stored_count() {
    ckpt3d::FileHeader3D a{};
    a.magic = ckpt3d::kMagic;
    a.file_crc64 = 0;
    a.base_measure_shards = 4;
    ckpt3d::FileHeader3D b = a;
    b.base_measure_shards = 11;
    expect(ckpt3d::crc64_ecma(&a, sizeof(a)) !=
               ckpt3d::crc64_ecma(&b, sizeof(b)),
           "base-measurement shard count participates in the header checksum");
}

pf3d::SimParams3D channel_params() {
    pf3d::SimParams3D p{};
    p.num_cells = 24;
    p.Nx = 449; p.Ny = 449; p.Nz = 140;
    p.boundary_flags = pf3d::kBoundaryHardWallChannel3D;
    p.channel_height = 98;
    p.channel_padding = 21;
    p.wall_kappa = 10.0;
    p.wall_width = 7.0;
    return p;
}

void trajectory_contract() {
    pf3d::SimParams3D periodic{};
    periodic.num_cells = 24;
    periodic.Nx = periodic.Ny = periodic.Nz = 479;
    periodic.boundary_flags = pf3d::kBoundaryPeriodicXYZ3D;

    pf3d::SimParams3D slab = periodic;
    slab.boundary_flags = pf3d::kBoundarySubstrateSlab3D;

    const pf3d::SimParams3D channel = channel_params();

    const pf3d::SimParams3D* all[] = {&periodic, &slab, &channel};
    const char* names[] = {"periodic", "slab", "channel"};
    const char* contracts[] = {
        "schema=1 dim=3 geometry=periodic-xyz",
        "schema=1 dim=3 geometry=substrate-slab",
        "schema=1 dim=3 geometry=resolved-wall-channel"};
    for (int g = 0; g < 3; ++g) {
        const std::string with_11 =
            pf3d::trajectory_header(*all[g], 152, -1, 264, 11, 17);
        const std::string with_4 =
            pf3d::trajectory_header(*all[g], 152, -1, 264, 4, 17);
        char message[128]{};
        std::snprintf(message, sizeof(message),
                      "%s header records the resolved count", names[g]);
        expect(with_11.find(" base_measure_shards=11") != std::string::npos,
               message);
        std::snprintf(message, sizeof(message),
                      "%s header records trajectory cadence", names[g]);
        expect(with_11.find(" trajectory_interval=17") != std::string::npos,
               message);
        std::snprintf(message, sizeof(message),
                      "%s header identifies the current geometry contract",
                      names[g]);
        expect(with_11.find(contracts[g]) != std::string::npos, message);
        std::snprintf(message, sizeof(message),
                      "%s headers with different counts differ (append "
                      "refused by the exact-match contract)", names[g]);
        expect(!with_11.empty() && !with_4.empty() && with_11 != with_4,
               message);
    }
}

void append_predicate() {
    const pf3d::SimParams3D channel = channel_params();
    const std::string header =
        pf3d::trajectory_header(channel, 152, -1, 264, 11, 17);
    const char* columns = "# columns: t id x";
    expect(pf3d::trajectory_metadata_compatible(header.c_str(), columns,
                                                header, columns),
           "append predicate accepts an exact metadata match");
    const std::string other =
        pf3d::trajectory_header(channel, 152, -1, 264, 4, 17);
    expect(!pf3d::trajectory_metadata_compatible(other.c_str(), columns,
                                                 header, columns),
           "append predicate refuses a different resolved count");
    const std::string other_cadence =
        pf3d::trajectory_header(channel, 152, -1, 264, 11, 19);
    expect(!pf3d::trajectory_metadata_compatible(
               other_cadence.c_str(), columns, header, columns),
           "append predicate refuses a different trajectory cadence");
    const std::string token = " base_measure_shards=11";
    std::string without_token = header;
    without_token.erase(without_token.find(token), token.size());
    expect(!pf3d::trajectory_metadata_compatible(without_token.c_str(), columns,
                                                 header, columns),
           "append predicate refuses a header without the shard count");
    expect(!pf3d::trajectory_metadata_compatible(header.c_str(),
                                                 "# columns: other",
                                                 header, columns),
           "append predicate refuses changed columns");
    expect(!pf3d::trajectory_metadata_compatible(nullptr, columns, header,
                                                 columns),
           "append predicate refuses a missing header line");
}

}  // namespace

int main() {
    policy_validation();
    default_resolution_unchanged();
    wave_and_fixed_resolution();
    checkpoint_value_validation();
    stored_validation();
    resume_rules();
    checksum_covers_stored_count();
    trajectory_contract();
    append_predicate();
    if (failures == 0) std::printf("measure_shards_cpu: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
