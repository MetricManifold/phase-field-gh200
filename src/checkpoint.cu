// Binary checkpoint I/O for the sole supported 2D schema. Record layouts are
// defined in common/checkpoint_format.h. A file records one tile pitch and one
// fixed-size phase-field tile per cell.
// Run-and-tumble randomness is counter-based in (polarity seed, cell id, step),
// so this implementation writes no generator-state sidecar.

#include "../include/checkpoint.cuh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace pf {
namespace {

static_assert(ckpt::CELL_TILE_PITCH == kTilePitch,
              "checkpoint and solver tile pitches must match");

// Reject truncated reads instead of accepting partial checkpoint state.
bool read_exact(std::FILE* f, void* dst, size_t bytes, const char* what) {
    if (bytes == 0) return true;
    if (std::fread(dst, 1, bytes, f) != bytes) {
        std::fprintf(stderr, "[ckpt] short read of %s (%zu B)\n", what, bytes);
        return false;
    }
    return true;
}

int wrapi_h(std::int64_t v, int L) {
    v %= L;
    return static_cast<int>(v < 0 ? v + L : v);
}

double wrapd_h(double v, double L) {
    double m = std::fmod(v, L);
    if (m < 0.0) m += L;
    return m;
}

// The parameter record preserves the global model and output contract. The
// realized per-cell gamma, speed, radius, and polarity are restored from
// sidecars.
void params_from_record(const ckpt::CheckpointParamsRecord& f, int num_cells,
                        SimParams* p) {
    p->Nx = f.Nx;
    p->Ny = f.Ny;
    p->dx = f.dx;
    p->dy = f.dy;
    p->dt = f.dt;
    p->t_end = f.t_end;
    p->rho = f.rho;
    p->lambda = f.lambda;
    p->gamma_normal = f.gamma_normal;
    p->gamma_cancer = f.gamma_soft;
    p->cancer_fraction = f.soft_fraction;
    p->kappa = f.kappa;
    p->target_radius = f.target_radius;
    p->mu = f.mu;
    p->v_A = f.v_A;
    p->v_A_sigma = f.v_A_sigma;
    p->xi = f.xi;
    p->tau = f.tau;
    p->num_cells = num_cells;
    p->seed = f.seed;
    // Writers store the resolved stream, so the continuation never depends on
    // the fallback relationship between the two command-line seed options.
    p->polarity_seed = f.polarity_seed;
    p->initialization_hash = f.initialization_hash;
    p->print_interval = f.print_interval;
    p->full_moment_every = f.full_moment_every;
    p->verify_every = f.verify_every;
}

void params_to_record(const SimParams& p, int trajectory_samples,
                      long long trajectory_interval, int save_interval,
                      ckpt::CheckpointParamsRecord* f) {
    std::memset(f, 0, sizeof(*f));
    f->Nx = p.Nx;
    f->Ny = p.Ny;
    f->dx = p.dx;
    f->dy = p.dy;
    f->dt = p.dt;
    f->t_end = p.t_end;
    f->rho = p.rho;
    f->lambda = p.lambda;
    f->gamma_normal = p.gamma_normal;
    f->gamma_soft = p.gamma_cancer;
    f->soft_fraction = p.cancer_fraction;
    f->kappa = p.kappa;
    f->target_radius = p.target_radius;
    f->mu = p.mu;
    f->v_A = p.v_A;
    f->v_A_sigma = p.v_A_sigma;
    f->xi = p.xi;
    f->tau = p.tau;
    f->seed = p.seed;
    f->polarity_seed = p.polarity_stream();
    f->initialization_hash = p.initialization_hash;
    f->print_interval = p.print_interval;
    f->full_moment_every = p.full_moment_every;
    f->verify_every = p.verify_every;
    f->save_interval = save_interval;
    f->trajectory_samples = trajectory_samples;
    f->reserved = 0;
    f->trajectory_interval = trajectory_interval;
}

// Measure both the numerical support (phi > kSupportEps) and full nonzero
// extent in checkpoint-tile coordinates.
bool tile_bboxes(const float* tile, int T, int sup_lo[kCkptDims],
                 int sup_hi[kCkptDims], int nz_lo[kCkptDims],
                 int nz_hi[kCkptDims]) {
    for (int d = 0; d < kCkptDims; ++d) {
        sup_lo[d] = nz_lo[d] = T;
        sup_hi[d] = nz_hi[d] = -1;
    }
    for (int y = 0; y < T; ++y) {
        const float* row = tile + (size_t)y * T;
        for (int x = 0; x < T; ++x) {
            const float v = row[x];
            if (v == 0.0f) continue;
            if (x < nz_lo[0]) nz_lo[0] = x;
            if (x > nz_hi[0]) nz_hi[0] = x;
            if (y < nz_lo[1]) nz_lo[1] = y;
            if (y > nz_hi[1]) nz_hi[1] = y;
            if (v <= kSupportEps) continue;
            if (x < sup_lo[0]) sup_lo[0] = x;
            if (x > sup_hi[0]) sup_hi[0] = x;
            if (y < sup_lo[1]) sup_lo[1] = y;
            if (y > sup_hi[1]) sup_hi[1] = y;
        }
    }
    return sup_hi[0] >= 0;
}

// Read the four required per-cell arrays. Current checkpoints contain each
// array exactly once; unknown or duplicate payloads are rejected.
bool read_sidecars(std::FILE* f, int n, CheckpointData* out) {
    std::vector<float> buf;
    bool had_vA = false;
    bool had_gamma = false;
    bool had_radius = false;
    bool had_polr = false;
    for (;;) {
        ckpt::SidecarBlockHeader sh{};
        const size_t got = std::fread(&sh, 1, sizeof(sh), f);
        if (got == 0) break;                       // clean EOF
        if (got != sizeof(sh)) {
            std::fprintf(stderr, "[ckpt] truncated sidecar header\n");
            return false;
        }
        if (sh.magic != ckpt::MAGIC_VA_A && sh.magic != ckpt::MAGIC_GAMA &&
            sh.magic != ckpt::MAGIC_RADI && sh.magic != ckpt::MAGIC_POLR) {
            std::fprintf(stderr,
                "[ckpt] unsupported sidecar 0x%08X in current checkpoint\n",
                sh.magic);
            return false;
        }
        if (sh.count != n) {
            std::fprintf(stderr,
                "[ckpt] sidecar 0x%08X has %d entries; expected %d\n",
                sh.magic, sh.count, n);
            return false;
        }
        buf.assign((size_t)sh.count, 0.0f);
        if (!read_exact(f, buf.data(), (size_t)sh.count * sizeof(float),
                        "sidecar payload"))
            return false;
        for (int i = 0; i < n; ++i) {
            const float value = buf[static_cast<std::size_t>(i)];
            const bool valid = std::isfinite(value) &&
                (sh.magic == ckpt::MAGIC_POLR ||
                 (sh.magic == ckpt::MAGIC_VA_A ? value >= 0.0f
                                                : value > 0.0f));
            if (!valid) {
                std::fprintf(stderr,
                    "[ckpt] sidecar 0x%08X has an invalid value at cell %d\n",
                    sh.magic, i);
                return false;
            }
        }
        switch (sh.magic) {
            case ckpt::MAGIC_VA_A:
                if (had_vA) {
                    std::fprintf(stderr, "[ckpt] duplicate VA_A sidecar\n");
                    return false;
                }
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].v_A = buf[(size_t)i];
                had_vA = true;
                break;
            case ckpt::MAGIC_GAMA:
                if (had_gamma) {
                    std::fprintf(stderr, "[ckpt] duplicate GAMA sidecar\n");
                    return false;
                }
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].gamma = buf[(size_t)i];
                had_gamma = true;
                break;
            case ckpt::MAGIC_RADI:
                if (had_radius) {
                    std::fprintf(stderr, "[ckpt] duplicate RADI sidecar\n");
                    return false;
                }
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].R_tgt = buf[(size_t)i];
                had_radius = true;
                break;
            default:   // MAGIC_POLR
                if (had_polr) {
                    std::fprintf(stderr, "[ckpt] duplicate POLR sidecar\n");
                    return false;
                }
                for (int i = 0; i < n; ++i) out->cells[(size_t)i].theta = buf[(size_t)i];
                had_polr = true;
                break;
        }
    }
    if (!had_vA || !had_gamma || !had_radius || !had_polr) {
        std::fprintf(stderr,
            "[ckpt] current checkpoint is missing required sidecars:%s%s%s%s\n",
            had_polr ? "" : " POLR", had_gamma ? "" : " GAMA",
            had_vA ? "" : " VA_A", had_radius ? "" : " RADI");
        return false;
    }
    return true;
}

// Write one gathered payload to every requested path. Temporary files are
// renamed only after all writes and closes succeed.
class FanOutWriter {
public:
    bool open(const std::vector<std::string>& paths) {
        final_ = paths;
        for (const std::string& p : paths) {
            const std::string t = p + ".tmp";
            std::FILE* f = std::fopen(t.c_str(), "wb");
            if (!f) {
                std::fprintf(stderr, "[ckpt] cannot open %s: %s\n", t.c_str(),
                             std::strerror(errno));
                discard();
                return false;
            }
            tmp_.push_back(t);
            fps_.push_back(f);
        }
        return !fps_.empty();
    }

    bool write(const void* src, size_t bytes, const char* what) {
        if (bytes == 0) return true;
        for (size_t i = 0; i < fps_.size(); ++i) {
            if (std::fwrite(src, 1, bytes, fps_[i]) != bytes) {
                std::fprintf(stderr, "[ckpt] short write of %s to %s: %s\n",
                             what, tmp_[i].c_str(), std::strerror(errno));
                return false;
            }
        }
        return true;
    }

    // A failed close may report delayed filesystem errors; never rename then.
    bool commit() {
        bool ok = true;
        for (size_t i = 0; i < fps_.size(); ++i) {
            if (std::fclose(fps_[i]) != 0) {
                std::fprintf(stderr, "[ckpt] failed to close %s: %s\n",
                             tmp_[i].c_str(), std::strerror(errno));
                ok = false;
            }
            fps_[i] = nullptr;
        }
        fps_.clear();
        if (!ok) { unlink_tmps(); return false; }
        for (size_t i = 0; i < tmp_.size(); ++i) {
            std::error_code ec;
            std::filesystem::rename(tmp_[i], final_[i], ec);
            if (ec) {
                std::fprintf(stderr, "[ckpt] rename %s -> %s failed: %s\n",
                             tmp_[i].c_str(), final_[i].c_str(),
                             ec.message().c_str());
                ok = false;
            }
        }
        return ok;
    }

    void discard() {
        for (std::FILE* f : fps_) if (f) std::fclose(f);
        fps_.clear();
        unlink_tmps();
    }

    ~FanOutWriter() { for (std::FILE* f : fps_) if (f) std::fclose(f); }

private:
    void unlink_tmps() {
        std::error_code ec;
        for (const std::string& t : tmp_) std::filesystem::remove(t, ec);
    }
    std::vector<std::FILE*>  fps_;
    std::vector<std::string> tmp_, final_;
};

// Stream contiguous cell tiles through a bounded host staging buffer.
constexpr int kStageCells = 64;

}  // namespace

void SimOverrides::apply(SimParams& p, const SimParams& cli) const {
    if (t_end)           p.t_end = cli.t_end;
    if (dt)              p.dt = cli.dt;
    if (v_A)             p.v_A = cli.v_A;
    if (v_A_sigma)       p.v_A_sigma = cli.v_A_sigma;
    if (tau)             p.tau = cli.tau;
    if (gamma)           p.gamma_normal = cli.gamma_normal;
    if (gamma_cancer)    p.gamma_cancer = cli.gamma_cancer;
    if (cancer_fraction) p.cancer_fraction = cli.cancer_fraction;
    if (kappa)           p.kappa = cli.kappa;
    if (mu)              p.mu = cli.mu;
    if (xi)              p.xi = cli.xi;
    if (lambda)          p.lambda = cli.lambda;
    if (target_radius)   p.target_radius = cli.target_radius;
    if (seed)            p.seed = cli.seed;
    if (polarity_seed)   p.polarity_seed = cli.polarity_seed;
    if (print_interval)  p.print_interval = cli.print_interval;
    if (full_moment)     p.full_moment_every = cli.full_moment_every;
    if (verify_every)    p.verify_every = cli.verify_every;
}

bool checkpoint_read(const std::string& path, CheckpointData* out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[ckpt] cannot open %s: %s\n", path.c_str(),
                     std::strerror(errno));
        return false;
    }
    struct Closer {
        std::FILE* f;
        ~Closer() { if (f) std::fclose(f); }
    } closer{f};

    ckpt::FixedPrefix pre{};
    if (!read_exact(f, &pre, sizeof(pre), "fixed prefix")) return false;
    if (pre.magic != ckpt::MAGIC) {
        std::fprintf(stderr, "[ckpt] %s is not a cell checkpoint "
                     "(magic 0x%08X, expected 0x%08X)\n",
                     path.c_str(), pre.magic, ckpt::MAGIC);
        return false;
    }
    if (pre.version != ckpt::CHECKPOINT_FORMAT) {
        std::fprintf(stderr,
            "[ckpt] %s uses unsupported 2D checkpoint schema %u; expected %u\n",
            path.c_str(), pre.version, ckpt::CHECKPOINT_FORMAT);
        return false;
    }
    if (pre.step < 0 || pre.reserved != 0 || pre.save_interval < 0 ||
        pre.trajectory_samples < 0 || pre.bools[0] != 0 ||
        pre.bools[1] != 0 || pre.bools[2] != 0 || pre.bools[3] != 0) {
        std::fprintf(stderr,
            "[ckpt] fixed checkpoint metadata is invalid or unsupported\n");
        return false;
    }
    if (pre.sp_sz != sizeof(ckpt::CheckpointParamsRecord)) {
        std::fprintf(stderr,
            "[ckpt] schema %u parameter record is %u B; expected %zu B. "
            "The record layout is incompatible.\n",
            pre.version, pre.sp_sz, sizeof(ckpt::CheckpointParamsRecord));
        return false;
    }
    if (pre.num_cells_local <= 0 || pre.num_cells_local > 4000000) {
        std::fprintf(stderr, "[ckpt] implausible cell count %d\n",
                     pre.num_cells_local);
        return false;
    }

    ckpt::CheckpointParamsRecord sp{};
    if (!read_exact(f, &sp, sizeof(sp), "parameter record")) return false;
    if (sp.reserved != 0 || sp.save_interval != pre.save_interval ||
        sp.trajectory_samples != pre.trajectory_samples ||
        sp.trajectory_samples <= 0 || sp.trajectory_interval < 0) {
        std::fprintf(stderr,
            "[ckpt] checkpoint parameter metadata is internally inconsistent\n");
        return false;
    }

    int32_t tile_pitch = 0;
    if (!read_exact(f, &tile_pitch, sizeof(tile_pitch), "tile pitch")) return false;
    if (tile_pitch != ckpt::CELL_TILE_PITCH) {
        std::fprintf(stderr,
            "[ckpt] tile pitch %d does not match schema %u pitch %d\n",
            tile_pitch, pre.version, ckpt::CELL_TILE_PITCH);
        return false;
    }

    ckpt::RankTrailer tr{};
    if (!read_exact(f, &tr, sizeof(tr), "rank trailer")) return false;
    if (tr.num_ranks != 1 || tr.rank_id != 0 ||
        tr.num_cells_global != pre.num_cells_local) {
        std::fprintf(stderr,
            "[ckpt] %s is not a complete single-GPU checkpoint "
            "(ranks=%d, rank=%d, global cells=%d, local cells=%d)\n",
            path.c_str(), tr.num_ranks, tr.rank_id, tr.num_cells_global,
            pre.num_cells_local);
        return false;
    }
    out->num_ranks = tr.num_ranks;
    out->rank_id = tr.rank_id;
    out->n_global = tr.num_cells_global;

    const int n = pre.num_cells_local;
    out->n = n;
    out->step = pre.step;
    out->t = pre.cur_time;
    out->file_tile_pitch = tile_pitch;
    out->trajectory_samples = sp.trajectory_samples;
    out->trajectory_interval = sp.trajectory_interval;
    out->params = SimParams{};
    params_from_record(sp, n, &out->params);

    const double expected_time = static_cast<double>(pre.step) * out->params.dt;
    const double time_tolerance = 1.0e-12 *
        std::max(1.0, std::fabs(expected_time));
    if (!std::isfinite(pre.cur_time) || !std::isfinite(expected_time) ||
        std::fabs(pre.cur_time - expected_time) > time_tolerance) {
        std::fprintf(stderr,
            "[ckpt] checkpoint time does not match step multiplied by dt\n");
        return false;
    }
    if (!validate(out->params)) {
        std::fprintf(stderr, "[ckpt] checkpoint parameters are invalid\n");
        return false;
    }

    const int L = out->params.Nx;

    // Seed fallback values before the required sidecars are decoded.
    out->cells.assign((size_t)n, CkptCell{});
    for (int i = 0; i < n; ++i) {
        CkptCell& c = out->cells[(size_t)i];
        c.gamma = (float)out->params.gamma_normal;
        c.v_A   = (float)out->params.v_A;
        c.R_tgt = (float)out->params.target_radius;
    }
    out->phi.assign((size_t)n * (size_t)kTileArea, 0.0f);

    std::vector<float> src((size_t)tile_pitch * (size_t)tile_pitch);
    int cls_hist[kNumClasses] = {};
    int max_ext[kCkptDims] = {0, 0};
    for (int i = 0; i < n; ++i) {
        ckpt::CellRecordHeader rec{};
        if (!read_exact(f, &rec, sizeof(rec), "cell record header")) return false;
        if (!read_exact(f, src.data(), src.size() * sizeof(float), "phi tile"))
            return false;
        if (rec.cell_id != i) {
            std::fprintf(stderr,
                "[ckpt] cell slot %d carries global id %d; current checkpoints "
                "require canonical IDs 0..N-1\n",
                i, rec.cell_id);
            return false;
        }
        if (rec.shape_class < 0 || rec.shape_class >= kNumClasses ||
            rec.promote_ctr >= static_cast<uint32_t>(kDemoteDwell) ||
            rec.reserved0 != 0 || rec.reserved1 != 0 ||
            rec.origin_x < 0 || rec.origin_x >= L ||
            rec.origin_y < 0 || rec.origin_y >= L ||
            !std::isfinite(rec.cx) || !std::isfinite(rec.cy) ||
            !std::isfinite(rec.vx) || !std::isfinite(rec.vy) ||
            !std::isfinite(rec.volume_moment) ||
            !(rec.volume_moment > 0.0) ||
            !std::isfinite(rec.moment_x) || !std::isfinite(rec.moment_y) ||
            !std::isfinite(rec.perimeter) || rec.perimeter < 0.0 ||
            !std::isfinite(rec.phi_max) || rec.phi_max < 0.0f) {
            std::fprintf(stderr,
                "[ckpt] cell %d has invalid state metadata\n",
                rec.cell_id);
            return false;
        }
        const ShapeClass sc = class_of(rec.shape_class);
        if (rec.support_lo_x < 0 ||
            rec.support_hi_x < rec.support_lo_x ||
            rec.support_hi_x >= sc.wx || rec.support_lo_y < 0 ||
            rec.support_hi_y < rec.support_lo_y ||
            rec.support_hi_y >= sc.wy) {
            std::fprintf(stderr,
                "[ckpt] cell %d support bounds do not fit shape class %d\n",
                rec.cell_id, rec.shape_class);
            return false;
        }
        float observed_phi_max = 0.0f;
        for (float value : src) {
            if (!std::isfinite(value)) {
                std::fprintf(stderr,
                    "[ckpt] cell %d phase field contains a non-finite value\n",
                    rec.cell_id);
                return false;
            }
            observed_phi_max = std::max(observed_phi_max, std::fabs(value));
        }
        int support_lo[kCkptDims], support_hi[kCkptDims];
        int nonzero_lo[kCkptDims], nonzero_hi[kCkptDims];
        if (!tile_bboxes(src.data(), tile_pitch, support_lo, support_hi,
                         nonzero_lo, nonzero_hi)) {
            std::fprintf(stderr,
                "[ckpt] cell %d has no phase-field support above %.1e\n",
                rec.cell_id, static_cast<double>(kSupportEps));
            return false;
        }
        const int expected_lo[kCkptDims] = {
            sc.tx0 + rec.support_lo_x, sc.ty0 + rec.support_lo_y};
        const int expected_hi[kCkptDims] = {
            sc.tx0 + rec.support_hi_x, sc.ty0 + rec.support_hi_y};
        if (support_lo[0] != expected_lo[0] ||
            support_hi[0] != expected_hi[0] ||
            support_lo[1] != expected_lo[1] ||
            support_hi[1] != expected_hi[1] ||
            nonzero_lo[0] < sc.tx0 || nonzero_hi[0] >= sc.tx0 + sc.wx ||
            nonzero_lo[1] < sc.ty0 || nonzero_hi[1] >= sc.ty0 + sc.wy ||
            observed_phi_max != rec.phi_max) {
            std::fprintf(stderr,
                "[ckpt] cell %d phase field disagrees with its stored class "
                "or moments\n",
                rec.cell_id);
            return false;
        }
        const float expected_cx = static_cast<float>(wrapd_h(
            static_cast<double>(rec.origin_x) + sc.tx0 +
                rec.moment_x / rec.volume_moment,
            static_cast<double>(L)));
        const float expected_cy = static_cast<float>(wrapd_h(
            static_cast<double>(rec.origin_y) + sc.ty0 +
                rec.moment_y / rec.volume_moment,
            static_cast<double>(L)));
        if (rec.cx != expected_cx || rec.cy != expected_cy) {
            std::fprintf(stderr,
                "[ckpt] cell %d derived centroid is internally inconsistent\n",
                rec.cell_id);
            return false;
        }

        std::memcpy(out->phi.data() + (size_t)i * kTileArea, src.data(),
                    (size_t)kTileArea * sizeof(float));
        max_ext[0] = std::max(max_ext[0],
                              rec.support_hi_x - rec.support_lo_x + 1);
        max_ext[1] = std::max(max_ext[1],
                              rec.support_hi_y - rec.support_lo_y + 1);

        CkptCell& c = out->cells[(size_t)i];
        c.global_id = rec.cell_id;
        c.cls = static_cast<uint8_t>(rec.shape_class);
        c.vx = rec.vx;
        c.vy = rec.vy;
        c.volume_moment = rec.volume_moment;
        c.moment_x = rec.moment_x;
        c.moment_y = rec.moment_y;
        c.perimeter = rec.perimeter;
        c.support_lo_x = rec.support_lo_x;
        c.support_hi_x = rec.support_hi_x;
        c.support_lo_y = rec.support_lo_y;
        c.support_hi_y = rec.support_hi_y;
        c.promote_ctr = rec.promote_ctr;
        c.phi_max = rec.phi_max;
        c.origin[0] = wrapi_h(static_cast<std::int64_t>(rec.origin_x) +
                                  sc.tx0,
                              L);
        c.origin[1] = wrapi_h(static_cast<std::int64_t>(rec.origin_y) +
                                  sc.ty0,
                              L);

        ++cls_hist[rec.shape_class];
    }

    if (!read_sidecars(f, n, out)) return false;

    std::printf("[ckpt] loaded %s: schema %u, step %d, t %.4f, %d cells, "
                "L=%d, tile %d\n"
                "       classes round/wide/tall/square/large/fallback "
                "%d/%d/%d/%d/%d/%d; stored fields:%s%s%s%s\n",
                path.c_str(), pre.version, pre.step, pre.cur_time, n, L,
                tile_pitch, cls_hist[0], cls_hist[1], cls_hist[2],
                cls_hist[3], cls_hist[4], cls_hist[5],
                " polarity", " gamma", " v_A", " radius");
    std::printf("       maximum support (phi > %.0e): %d x %d px; "
                "round guarded/fallback physical limits %d/%d px\n",
                (double)kSupportEps, max_ext[0], max_ext[1],
                kClasses[kClassRound].wx - kPromoteSlack,
                kClasses[kClassFallback].wx);
    return true;
}

void resolve_per_cell_scalars(const SimParams& p, const SimOverrides& ov,
                              CheckpointData* d) {
    const int n = d->n;

    // Reassign gamma only when explicitly requested; otherwise retain it.
    if (ov.gamma_policy_changed()) {
        const int n_cancer = (int)std::llround(p.cancer_fraction * (double)n);
        for (int i = 0; i < n; ++i) {
            CkptCell& c = d->cells[(size_t)i];
            c.gamma = (float)((c.global_id < n_cancer) ? p.gamma_cancer
                                                       : p.gamma_normal);
        }
        std::printf("[ckpt] per-cell gamma reassigned: "
                    "%d of %d cells at gamma_cancer = %.6g, the rest at "
                    "%.6g (stored values overridden)\n",
                    n_cancer, n, p.gamma_cancer, p.gamma_normal);
    }

    bool redo_vA = ov.v_A_policy_changed();
    // Replace an all-zero equilibration sidecar when active speed is nonzero.
    if (!redo_vA && p.v_A > 0.0) {
        constexpr double kZeroSidecarSumEpsilon = 1e-12;
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += std::fabs((double)d->cells[(size_t)i].v_A);
        if (sum <= kZeroSidecarSumEpsilon) {
            std::printf("[ckpt] stored per-cell v_A values are zero; "
                        "reassigning for v_A = %.6g.\n", p.v_A);
            redo_vA = true;
        }
    }
    if (redo_vA) {
        for (int i = 0; i < n; ++i) {
            CkptCell& c = d->cells[(size_t)i];
            c.v_A = (float)ic_v_A(c.global_id, p.seed, p.v_A, p.v_A_sigma);
        }
        std::printf("[ckpt] per-cell v_A reassigned: median %.6g, "
                    "lognormal sigma %.6g\n", p.v_A, p.v_A_sigma);
    }

    if (ov.target_radius) {
        for (int i = 0; i < n; ++i)
            d->cells[(size_t)i].R_tgt = (float)p.target_radius;
    }
}

bool checkpoint_write(const CheckpointWriteView& v,
                      const std::vector<std::string>& paths) {
    if (paths.empty()) return true;
    if (!v.p || !v.cell || !v.cls || !v.d_phi || v.N <= 0 || v.L <= 0) {
        std::fprintf(stderr, "[ckpt] incomplete write view\n");
        return false;
    }
    if (v.trajectory_samples <= 0 || v.trajectory_interval < 0 ||
        v.save_interval < 0) {
        std::fprintf(stderr, "[ckpt] invalid output schedule metadata\n");
        return false;
    }
    if (v.p->num_cells != v.N || v.p->Nx != v.L || v.p->Ny != v.L ||
        !validate(*v.p)) {
        std::fprintf(stderr,
                     "[ckpt] write view and simulation parameters disagree\n");
        return false;
    }
    // The current schema stores the step as int32; reject overflow.
    if (v.step < 0 || v.step > 2147483647LL) {
        std::fprintf(stderr,
            "[ckpt] step %lld exceeds the checkpoint int32 field; checkpoint "
            "not written.\n",
            v.step);
        return false;
    }
    FanOutWriter w;
    if (!w.open(paths)) return false;

    ckpt::FixedPrefix pre{};
    pre.magic = ckpt::MAGIC;
    pre.version = ckpt::CHECKPOINT_FORMAT;
    pre.step = (int32_t)v.step;
    pre.cur_time = v.t;
    pre.num_cells_local = v.N;
    pre.save_interval = v.save_interval;
    pre.reserved = 0;
    pre.trajectory_samples = v.trajectory_samples;
    pre.sp_sz = (uint32_t)sizeof(ckpt::CheckpointParamsRecord);

    ckpt::CheckpointParamsRecord sp{};
    params_to_record(*v.p, v.trajectory_samples, v.trajectory_interval,
                     v.save_interval, &sp);

    const int32_t tile_pitch = ckpt::CELL_TILE_PITCH;
    ckpt::RankTrailer tr{};
    tr.num_ranks = 1;
    tr.rank_id = 0;
    tr.num_cells_global = v.N;

    if (!w.write(&pre, sizeof(pre), "fixed prefix") ||
        !w.write(&sp, sizeof(sp), "parameter record") ||
        !w.write(&tile_pitch, sizeof(tile_pitch), "tile pitch") ||
        !w.write(&tr, sizeof(tr), "rank trailer")) {
        w.discard();
        return false;
    }

    // Stream per-cell records through bounded host memory.
    std::vector<float> stage((size_t)kStageCells * (size_t)kTileArea);
    for (int base = 0; base < v.N; base += kStageCells) {
        const int cnt = (v.N - base) < kStageCells ? (v.N - base) : kStageCells;
        const size_t words = (size_t)cnt * (size_t)kTileArea;
        const cudaError_t e = cudaMemcpy(stage.data(),
                                         v.d_phi + (size_t)base * kTileArea,
                                         words * sizeof(float),
                                         cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "[ckpt] D2H of cells %d..%d failed: %s\n",
                         base, base + cnt - 1, cudaGetErrorString(e));
            w.discard();
            return false;
        }
        for (int k = 0; k < cnt; ++k) {
            const int i = base + k;
            const CellState& c = v.cell[(size_t)i];
            const int cls = (int)c.cls;
            if (c.global_id != i) {
                std::fprintf(stderr,
                    "[ckpt] cell slot %d carries global id %d; refusing a "
                    "noncanonical checkpoint\n",
                    i, c.global_id);
                w.discard();
                return false;
            }
            if (!std::isfinite(c.gamma) || !(c.gamma > 0.0f) ||
                !std::isfinite(c.v_A) || c.v_A < 0.0f ||
                !std::isfinite(c.R_tgt) || !(c.R_tgt > 0.0f) ||
                !std::isfinite(c.theta) || !std::isfinite(c.vx) ||
                !std::isfinite(c.vy) || !std::isfinite(c.V) ||
                !std::isfinite(c.Cx) || !std::isfinite(c.Cy) ||
                !std::isfinite(c.perim) || c.perim < 0.0 ||
                !std::isfinite(c.phi_max) || c.phi_max < 0.0f ||
                !(c.V > 0.0) || c.gx0 < 0 || c.gx0 >= v.L ||
                c.gy0 < 0 || c.gy0 >= v.L ||
                c.promote_ctr >= static_cast<uint32_t>(kDemoteDwell)) {
                std::fprintf(stderr,
                    "[ckpt] cell %d has invalid state; checkpoint not written\n",
                    i);
                w.discard();
                return false;
            }
            if (cls < 0 || cls >= kNumClasses || (int)v.cls[(size_t)i] != cls) {
                std::fprintf(stderr,
                    "[ckpt] cell %d carries shape class %d (cell_cls says %d); "
                    "that is memory corruption, not a shape. Refusing to "
                    "write.\n", i, cls, (int)v.cls[(size_t)i]);
                w.discard();
                return false;
            }
            const ShapeClass sc = class_of(cls);
            if (c.cls_written[v.step & 1LL] != c.cls ||
                c.bb_lo_x < 0 || c.bb_hi_x < c.bb_lo_x ||
                c.bb_hi_x >= sc.wx || c.bb_lo_y < 0 ||
                c.bb_hi_y < c.bb_lo_y || c.bb_hi_y >= sc.wy) {
                std::fprintf(stderr,
                    "[ckpt] cell %d class layout or support bounds are invalid\n",
                    i);
                w.discard();
                return false;
            }

            ckpt::CellRecordHeader rec{};
            rec.cell_id = c.global_id;
            // Convert the active-window origin to the full-tile origin and use
            // a non-negative periodic coordinate.
            rec.origin_x = wrapi_h(c.gx0 - sc.tx0, v.L);
            rec.origin_y = wrapi_h(c.gy0 - sc.ty0, v.L);
            rec.shape_class = cls;
            rec.promote_ctr = c.promote_ctr;
            const double invV = 1.0 / c.V;
            rec.cx = (float)wrapd_h((double)c.gx0 + c.Cx * invV, (double)v.L);
            rec.cy = (float)wrapd_h((double)c.gy0 + c.Cy * invV, (double)v.L);
            rec.vx = c.vx;
            rec.vy = c.vy;
            rec.volume_moment = c.V;
            rec.moment_x = c.Cx;
            rec.moment_y = c.Cy;
            rec.perimeter = c.perim;
            rec.support_lo_x = c.bb_lo_x;
            rec.support_hi_x = c.bb_hi_x;
            rec.support_lo_y = c.bb_lo_y;
            rec.support_hi_y = c.bb_hi_y;
            rec.phi_max = c.phi_max;

            if (!w.write(&rec, sizeof(rec), "cell record") ||
                !w.write(stage.data() + (size_t)k * kTileArea,
                         (size_t)kTileArea * sizeof(float), "phi tile")) {
                w.discard();
                return false;
            }
        }
    }

    // Sidecars use the checkpoint-format order.
    std::vector<float> col((size_t)v.N);
    auto emit = [&](uint32_t magic, float CellState::*field) -> bool {
        for (int i = 0; i < v.N; ++i) col[(size_t)i] = v.cell[(size_t)i].*field;
        ckpt::SidecarBlockHeader sh{magic, v.N};
        return w.write(&sh, sizeof(sh), "sidecar header") &&
               w.write(col.data(), col.size() * sizeof(float), "sidecar payload");
    };
    if (!emit(ckpt::MAGIC_POLR, &CellState::theta) ||
        !emit(ckpt::MAGIC_GAMA, &CellState::gamma) ||
        !emit(ckpt::MAGIC_VA_A, &CellState::v_A) ||
        !emit(ckpt::MAGIC_RADI, &CellState::R_tgt)) {
        w.discard();
        return false;
    }
    // Counter-based Philox requires no generator-state sidecar.

    if (!w.commit()) return false;

    std::printf("[ckpt] saved step %lld  t %.4f  %d cells  tile pitch %d ->",
                v.step, v.t, v.N, (int)tile_pitch);
    for (const std::string& p : paths) std::printf(" %s", p.c_str());
    std::printf("\n");
    std::fflush(stdout);
    return true;
}

}  // namespace pf
