// Host tests for phase-field-mobility persistence and the strict map
// interface: the MOBI sidecar reader, the all-default load of pre-mobility
// checkpoints, resume precedence, and map validation. When a CUDA device is
// present the real writer -> reader round trip also runs (checkpoint I/O only;
// no kernels are launched). Includes the checkpoint implementation to exercise
// its sidecar parser directly without advancing the simulation.
#include "../src/checkpoint.cu"

#include <cstdio>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace pf;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "[FAIL] %s\n", what);
        ++failures;
    }
}

constexpr int kN = 2;
constexpr int32_t kIds[kN] = {0, 1};   // current checkpoint canonical ids
constexpr float kSoftM = 1.42857146f;  // (float)(0.5/0.35)

std::vector<float> blob_tile() {
    std::vector<float> t((size_t)kTileArea, 0.0f);
    for (int y = 70; y <= 120; ++y)
        for (int x = 70; x <= 120; ++x)
            t[(size_t)y * kTilePitch + x] = 0.5f;   // class-0 window, exact
    return t;
}

void params_record(ckpt::CheckpointParamsRecord* sp) {
    SimParams p;
    p.Nx = p.Ny = 320;
    p.num_cells = kN;
    p.target_radius = 20.0;
    p.t_end = 1.0;
    params_to_record(p, 100, 0, 0, sp);
}

bool put(std::FILE* f, const void* p, size_t n) {
    return std::fwrite(p, 1, n, f) == n;
}

bool put_sidecar(std::FILE* f, uint32_t magic, const float* v, int n) {
    const ckpt::SidecarBlockHeader sh{magic, n};
    return put(f, &sh, sizeof(sh)) && put(f, v, (size_t)n * sizeof(float));
}

// Emit a minimal current-format checkpoint; mobi == nullptr omits the MOBI block.
// ids defaults to kIds; a caller may pass duplicated ids to model corruption.
bool write_synthetic(const std::string& path, const float* mobi,
                     const int32_t* ids = kIds) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    ckpt::FixedPrefix pre{};
    pre.magic = ckpt::MAGIC;
    pre.version = ckpt::CHECKPOINT_FORMAT;
    pre.step = 0;
    pre.cur_time = 0.0;
    pre.num_cells_local = kN;
    pre.trajectory_samples = 100;
    pre.sp_sz = (uint32_t)sizeof(ckpt::CheckpointParamsRecord);
    ckpt::CheckpointParamsRecord sp{};
    params_record(&sp);
    const int32_t pitch = kTilePitch;
    const ckpt::RankTrailer tr{1, 0, kN};

    bool ok = put(f, &pre, sizeof(pre)) && put(f, &sp, sizeof(sp)) &&
              put(f, &pitch, sizeof(pitch)) && put(f, &tr, sizeof(tr));
    const std::vector<float> tile = blob_tile();
    for (int i = 0; ok && i < kN; ++i) {
        ckpt::CellRecordHeader rec{};
        rec.cell_id = ids[i];
        rec.origin_x = 10 + 60 * i;
        rec.origin_y = 20;
        rec.shape_class = kClassRound;
        rec.volume_moment = 51.0 * 51.0 * 0.25;
        rec.moment_x = rec.moment_y = 31.0 * rec.volume_moment;
        rec.support_lo_x = rec.support_lo_y = 6;
        rec.support_hi_x = rec.support_hi_y = 56;
        rec.phi_max = 0.5f;
        rec.cx = static_cast<float>(rec.origin_x + 95);
        rec.cy = static_cast<float>(rec.origin_y + 95);
        ok = put(f, &rec, sizeof(rec)) &&
             put(f, tile.data(), tile.size() * sizeof(float));
    }
    const float theta[kN] = {0.25f, 1.75f};
    const float gamma[kN] = {1.0f, 0.35f};
    const float va[kN]    = {0.01f, 0.01f};
    const float radi[kN]  = {20.0f, 20.0f};
    ok = ok && put_sidecar(f, ckpt::MAGIC_POLR, theta, kN) &&
         put_sidecar(f, ckpt::MAGIC_GAMA, gamma, kN) &&
         put_sidecar(f, ckpt::MAGIC_VA_A, va, kN) &&
         put_sidecar(f, ckpt::MAGIC_RADI, radi, kN);
    if (ok && mobi) ok = put_sidecar(f, ckpt::MAGIC_MOBI, mobi, kN);
    return std::fclose(f) == 0 && ok;
}

std::string write_map(const char* name, const char* text) {
    std::FILE* f = std::fopen(name, "wb");
    if (!f) {
        expect(false, "temporary map opened");
        return name;
    }
    std::fwrite(text, 1, std::strlen(text), f);
    std::fclose(f);
    return name;
}

void test_map_parser() {
    PhaseFieldMobilityMap m;
    std::string err;

    const std::string good = write_map("mobility_tmp_good.tsv",
        "# soft-cell control\n"
        "\n"
        "7\t1.4285715\n"
        "0 0.5\n");
    expect(read_phase_field_mobility_map(good, &m, &err) &&
               m.entries.size() == 2 && m.entries[0].first == 7 &&
               m.entries[1].first == 0 && m.entries[1].second == 0.5f,
           "a valid map with comments and blank lines must parse");

    const struct { const char* name; const char* text; const char* why; }
    bad[] = {
        {"mobility_tmp_b1.tsv", "1 2 3\n", "trailing token"},
        {"mobility_tmp_b2.tsv", "x 1.0\n", "non-numeric id"},
        {"mobility_tmp_b3.tsv", "1 abc\n", "non-numeric value"},
        {"mobility_tmp_b4.tsv", "1 0.5\n1 0.6\n", "duplicate id"},
        {"mobility_tmp_b5.tsv", "1 -0.5\n", "negative value"},
        {"mobility_tmp_b6.tsv", "1 0\n", "zero value"},
        {"mobility_tmp_b7.tsv", "1 nan\n", "NaN value"},
        {"mobility_tmp_b8.tsv", "1 inf\n", "infinite value"},
        {"mobility_tmp_b9.tsv", "1 1e-60\n", "binary32 underflow to zero"},
        {"mobility_tmp_b10.tsv", "-3 0.5\n", "negative id"},
        {"mobility_tmp_b11.tsv", "1.5 0.5\n", "non-integer id"},
        {"mobility_tmp_b12.tsv", "7\n", "missing value"},
        {"mobility_tmp_b13.tsv", "", "empty map"},
        {"mobility_tmp_b14.tsv", "1 0.5\rjunk\n", "token after carriage return"},
        {"mobility_tmp_b15.tsv", "1 1e99\n", "binary32 overflow"},
    };
    for (const auto& c : bad) {
        write_map(c.name, c.text);
        if (read_phase_field_mobility_map(c.name, &m, &err)) {
            std::fprintf(stderr, "[FAIL] parser accepted %s\n", c.why);
            ++failures;
        }
    }
}

void test_apply() {
    const int32_t gid[3] = {5, 9, 2};
    float out[3];
    std::string err;
    PhaseFieldMobilityMap empty;
    expect(apply_phase_field_mobility(empty, kPhaseFieldM0, gid, 3, out,
                                      &err) &&
               out[0] == 0.5f && out[1] == 0.5f && out[2] == 0.5f,
           "the default configuration must give every cell M = 0.5");

    PhaseFieldMobilityMap m;
    m.entries = {{9, kSoftM}, {2, 0.75f}};
    expect(apply_phase_field_mobility(m, kPhaseFieldM0, gid, 3, out, &err) &&
               out[0] == 0.5f && out[1] == kSoftM && out[2] == 0.75f,
           "map entries must override by arbitrary global id");

    PhaseFieldMobilityMap unknown;
    unknown.entries = {{3, 1.0f}};
    expect(!apply_phase_field_mobility(unknown, kPhaseFieldM0, gid, 3, out,
                                       &err),
           "an id outside the simulation must be rejected");

    // The overlay leaves unlisted cells exactly as it found them.
    float inout[3] = {0.7f, kSoftM, 0.9f};
    PhaseFieldMobilityMap one;
    one.entries = {{9, 1.25f}};
    expect(overlay_phase_field_mobility(one, gid, 3, inout, &err) &&
               inout[0] == 0.7f && inout[1] == 1.25f && inout[2] == 0.9f,
           "the overlay must touch only listed ids");

    // A duplicated global id among the cells is corruption: a map entry
    // would silently update only one of the records.
    const int32_t dup_gid[3] = {5, 9, 5};
    PhaseFieldMobilityMap hits_dup;
    hits_dup.entries = {{5, 1.0f}};
    expect(!apply_phase_field_mobility(hits_dup, kPhaseFieldM0, dup_gid, 3,
                                       out, &err) &&
               err.find("duplicated") != std::string::npos,
           "duplicated simulated global ids must be rejected by apply");
    expect(!overlay_phase_field_mobility(hits_dup, dup_gid, 3, inout, &err),
           "duplicated simulated global ids must be rejected by overlay");
}

void test_reader_and_precedence() {
    const std::string plain = "mobility_tmp_plain.bin";
    const std::string with  = "mobility_tmp_mobi.bin";
    const std::string bad   = "mobility_tmp_badmobi.bin";
    const std::string dup   = "mobility_tmp_dupids.bin";
    // Stored values deliberately distinguishable from the compiled default
    // 0.5, so a wrongful rebase of unlisted cells cannot pass unnoticed.
    const float mobi[kN] = {0.7f, kSoftM};
    const float badm[kN] = {0.5f, 0.0f};
    const int32_t dup_ids[kN] = {7, 7};
    expect(write_synthetic(plain, nullptr), "synthetic checkpoint written");
    expect(write_synthetic(with, mobi), "synthetic MOBI checkpoint written");
    expect(write_synthetic(bad, badm), "synthetic bad-MOBI file written");
    expect(write_synthetic(dup, mobi, dup_ids),
           "synthetic duplicate-id checkpoint written");

    CheckpointData d;
    expect(checkpoint_read(plain, &d) && !d.had_mobi &&
               d.cells[0].M_pf == 0.5f && d.cells[1].M_pf == 0.5f,
           "a pre-mobility checkpoint must load with M_i = 0.5 everywhere");

    CheckpointData dm;
    expect(checkpoint_read(with, &dm) && dm.had_mobi &&
               dm.cells[0].M_pf == 0.7f && dm.cells[1].M_pf == kSoftM,
           "MOBI sidecar values must load per cell");
    expect(dm.cells[1].gamma == 0.35f,
           "existing GAMA reading must be unaffected");

    CheckpointData db;
    std::fprintf(stderr, "(expected MOBI rejection follows)\n");
    expect(!checkpoint_read(bad, &db),
           "a non-positive stored mobility must be rejected");

    // Case 3: plain resume keeps stored MOBI values.
    SimParams p = dm.params;
    SimOverrides ov;
    PhaseFieldMobilityMap none;
    expect(resolve_phase_field_mobility(p, ov, none, false, &dm) &&
               dm.cells[0].M_pf == 0.7f && dm.cells[1].M_pf == kSoftM,
           "a plain resume must keep stored per-cell values");

    // Case 2: a map ALONE overlays only its listed ids; the unlisted cell
    // must keep its stored 0.7, not be rebased to the compiled default 0.5
    // (the current-format record does not store the writing run's uniform).
    PhaseFieldMobilityMap m;
    m.entries = {{1, 1.25f}};
    SimOverrides ov_map_only;
    SimParams p_def = dm.params;   // uniform at the 0.5 default, as on resume
    expect(resolve_phase_field_mobility(p_def, ov_map_only, m, true, &dm) &&
               dm.cells[0].M_pf == 0.7f && dm.cells[1].M_pf == 1.25f,
           "a map-only resume must preserve stored values for unlisted cells");

    // Case 1: an explicit uniform plus a map rebases every cell from the
    // uniform, then applies the map overrides.
    SimParams p_reb = dm.params;
    p_reb.phase_field_mobility = 0.9;
    SimOverrides ov_reb;
    ov_reb.phase_field_mobility = true;
    PhaseFieldMobilityMap m2;
    m2.entries = {{1, kSoftM}};
    expect(resolve_phase_field_mobility(p_reb, ov_reb, m2, true, &dm) &&
               dm.cells[0].M_pf == 0.9f && dm.cells[1].M_pf == kSoftM,
           "an explicit uniform plus map must rebase then override");

    // Case 1 without a map: every cell takes the explicit uniform.
    p_reb.phase_field_mobility = 0.6;
    expect(resolve_phase_field_mobility(p_reb, ov_reb, none, false, &dm) &&
               dm.cells[0].M_pf == 0.6f && dm.cells[1].M_pf == 0.6f,
           "an explicit uniform alone must reassign every cell");

    // A map id absent from the checkpoint fails before start.
    PhaseFieldMobilityMap u;
    u.entries = {{3, 1.0f}};
    std::fprintf(stderr, "(expected unknown-id rejection follows)\n");
    expect(!resolve_phase_field_mobility(p_def, ov_map_only, u, true, &dm),
           "an unknown map id must fail the resume");

    // An old checkpoint plus a map-only resume overlays the all-0.5 load.
    CheckpointData d2;
    expect(checkpoint_read(plain, &d2) &&
               resolve_phase_field_mobility(p_def, ov_map_only, m, true, &d2) &&
               d2.cells[0].M_pf == 0.5f && d2.cells[1].M_pf == 1.25f,
           "an old checkpoint with a map must keep 0.5 for unlisted cells");

    // Duplicated global ids corrupt both keyed per-cell state and Philox
    // streams, so the checkpoint reader rejects them even without a map.
    CheckpointData dd;
    std::fprintf(stderr, "(expected duplicate-id rejection follows)\n");
    expect(!checkpoint_read(dup, &dd),
           "duplicated checkpoint global ids must fail every resume");
}

void test_mobility_sidecar_rejection() {
    const char* path = "mobility_tmp_invalid.bin";
    const float valid[kN] = {0.7f, kSoftM};
    const float invalid[] = {-0.1f, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()};
    for (float bad : invalid) {
        const float values[kN] = {0.5f, bad};
        expect(write_synthetic(path, values), "invalid MOBI fixture written");
        CheckpointData d;
        expect(!checkpoint_read(path, &d), "invalid MOBI value rejected");
    }
    for (int mode = 0; mode < 3; ++mode) {
        // Duplicate block; wrong count; truncated payload, respectively.
        expect(write_synthetic(path, mode == 0 ? valid : nullptr),
               "MOBI structural fixture written");
        std::FILE* f = std::fopen(path, "ab");
        expect(f != nullptr, "MOBI structural fixture reopened");
        if (!f) continue;
        const ckpt::SidecarBlockHeader sh{ckpt::MAGIC_MOBI,
                                          mode == 1 ? kN - 1 : kN};
        const int count = mode == 2 ? kN - 1 : kN;
        expect(put(f, &sh, sizeof(sh)) && put(f, valid, count * sizeof(float)),
               "MOBI structural payload written");
        std::fclose(f);
        CheckpointData d;
        expect(!checkpoint_read(path, &d), "malformed MOBI block rejected");
    }
    // Reusing a reader destination cannot leak a previous file's presence bit.
    CheckpointData reused;
    expect(write_synthetic(path, valid) && checkpoint_read(path, &reused) &&
               reused.had_mobi, "reader observes MOBI");
    expect(write_synthetic(path, nullptr) && checkpoint_read(path, &reused) &&
               !reused.had_mobi && reused.cells[0].M_pf == 0.5f,
           "reader resets absent-MOBI state");
}

// Real writer -> reader round trip. Checkpoint I/O performs only D2H copies,
// so any CUDA device suffices; the update kernels are never launched.
void test_writer_round_trip() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
        std::printf("writer_round_trip: SKIPPED (no CUDA device)\n");
        return;
    }
    const std::vector<float> tile = blob_tile();
    std::vector<float> pool((size_t)kN * kTileArea);
    for (int i = 0; i < kN; ++i)
        std::memcpy(pool.data() + (size_t)i * kTileArea, tile.data(),
                    (size_t)kTileArea * sizeof(float));
    float* d_phi = nullptr;
    if (cudaMalloc(&d_phi, pool.size() * sizeof(float)) != cudaSuccess) {
        expect(false, "writer GPU allocation succeeded");
        return;
    }
    if (cudaMemcpy(d_phi, pool.data(), pool.size() * sizeof(float),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        expect(false, "writer GPU upload succeeded");
        cudaFree(d_phi);
        return;
    }

    SimParams p;
    p.Nx = p.Ny = 320;
    p.num_cells = kN;
    p.target_radius = 20.0;
    std::vector<CellState> cell((size_t)kN);
    std::memset(cell.data(), 0, cell.size() * sizeof(CellState));
    std::vector<uint8_t> cls((size_t)kN, (uint8_t)kClassRound);
    const float mobi[kN] = {0.5f, kSoftM};
    for (int i = 0; i < kN; ++i) {
        cell[(size_t)i].global_id = kIds[i];
        cell[(size_t)i].gx0 = 10 + 60 * i;
        cell[(size_t)i].gy0 = 20;
        cell[(size_t)i].cls = (uint8_t)kClassRound;
        cell[(size_t)i].gamma = i ? 0.35f : 1.0f;
        cell[(size_t)i].v_A = 0.01f;
        cell[(size_t)i].R_tgt = 20.0f;
        cell[(size_t)i].theta = 0.5f * (float)i;
        cell[(size_t)i].V = 51.0 * 51.0 * 0.25;
        cell[(size_t)i].Cx = cell[(size_t)i].Cy = 31.0 * cell[(size_t)i].V;
        cell[(size_t)i].bb_lo_x = cell[(size_t)i].bb_lo_y = 6;
        cell[(size_t)i].bb_hi_x = cell[(size_t)i].bb_hi_y = 56;
        cell[(size_t)i].phi_max = 0.5f;
        cell[(size_t)i].M_pf = mobi[i];
    }
    CheckpointWriteView v;
    v.p = &p;
    v.step = 5;
    v.t = 0.05;
    v.N = kN;
    v.L = 320;
    v.cell = cell.data();
    v.cls = cls.data();
    v.d_phi = d_phi;
    v.trajectory_samples = 100;
    const std::string path = "mobility_tmp_roundtrip.bin";
    expect(checkpoint_write(v, {path}), "checkpoint_write must succeed");
    cudaFree(d_phi);

    CheckpointData d;
    expect(checkpoint_read(path, &d) && d.had_mobi && d.n == kN &&
               d.cells[0].M_pf == mobi[0] && d.cells[1].M_pf == mobi[1] &&
               d.cells[0].global_id == kIds[0] &&
               d.cells[1].global_id == kIds[1],
           "heterogeneous per-cell mobilities must survive the round trip");
    expect(d.cells[1].gamma == 0.35f && d.cells[1].v_A == 0.01f &&
               d.cells[1].R_tgt == 20.0f && d.cells[1].theta == 0.5f,
           "existing sidecars must survive alongside MOBI");
    bool phi_exact = d.phi.size() == pool.size();
    for (size_t i = 0; phi_exact && i < pool.size(); ++i)
        phi_exact = d.phi[i] == pool[i];
    expect(phi_exact, "the phase field must round-trip exactly");
    std::printf("writer_round_trip: RAN on a real CUDA device\n");
}

void cleanup() {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(".", ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("mobility_tmp_", 0) == 0) fs::remove(e.path(), ec);
    }
}

}  // namespace

int main() {
    const auto previous = std::filesystem::current_path();
    const auto stamp = std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count();
    std::filesystem::path temporary;
    for (int attempt = 0; attempt < 100; ++attempt) {
        const auto candidate = std::filesystem::temp_directory_path() /
            ("pf2d-mobility-" + std::to_string(stamp) + "-" + std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error)) {
            temporary = candidate;
            break;
        }
    }
    if (temporary.empty()) return 1;
    std::filesystem::current_path(temporary);
    test_map_parser();
    test_apply();
    test_reader_and_precedence();
    test_mobility_sidecar_rejection();
    test_writer_round_trip();
    cleanup();
    std::filesystem::current_path(previous);
    std::filesystem::remove(temporary);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("MOBILITY_CHECKPOINT_CPU_TEST_PASS");
    return 0;
}
