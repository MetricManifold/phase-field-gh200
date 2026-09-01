// Host orchestration for initialization, ordered CUDA updates, checkpointing,
// and diagnostics. A six-step graph covers the two phi buffers and three
// aggregate-field buffers.

#include "../include/sim.cuh"
#include "../include/palmieri_initializer.hpp"
#include "../include/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace pf {

namespace {

bool checked_product(std::size_t a, std::size_t b, std::size_t* result) {
    if (!result || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a))
        return false;
    *result = a * b;
    return true;
}

long long next_interval_boundary(long long current, long long interval) {
    if (current < 0 || interval <= 0)
        return std::numeric_limits<long long>::max();
    const long long remainder = current % interval;
    const long long delta = remainder == 0 ? interval : interval - remainder;
    return delta <= std::numeric_limits<long long>::max() - current
        ? current + delta : std::numeric_limits<long long>::max();
}

}  // namespace

#define CU_CHECK(expr)                                                        \
    do {                                                                      \
        const cudaError_t _e = (expr);                                        \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[cuda] %s:%d %s -> %s\n", __FILE__,         \
                         __LINE__, #expr, cudaGetErrorString(_e));            \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define CU_WARN(expr)                                                         \
    do {                                                                      \
        const cudaError_t _e = (expr);                                        \
        if (_e != cudaSuccess)                                                \
            std::fprintf(stderr, "[cuda-warn] %s:%d %s -> %s\n", __FILE__,    \
                         __LINE__, #expr, cudaGetErrorString(_e));            \
    } while (0)

Sim::~Sim() {
    if (traj_fp_) (void)close_trajectory();
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_)      cudaGraphDestroy(graph_);
    if (stream_)     cudaStreamDestroy(stream_);
    cudaFree(d_phi_[0]);
    cudaFree(d_phi_[1]);
    cudaFree(d_S_);
    cudaFree(d_cell_);
    cudaFree(d_cls_);
    cudaFree(d_perm_);
    cudaFree(d_cursor_);
    cudaFree(d_step_);
    cudaFree(d_flags_);
    cudaFree(d_vchk_);
    cudaFree(d_ochk_);
    cudaFree(d_smax_);
    if (h_traj_) cudaFreeHost(h_traj_);
}

// Initialize centres from a table checked for IDs, bounds, and separation, or
// use the default grid-and-jitter layout.
bool Sim::seed_positions(std::vector<float>& cx, std::vector<float>& cy,
                         std::vector<float>& gam, std::vector<float>& va,
                         std::vector<int32_t>& gid)
{
    const int N = p_.num_cells;
    cx.resize(N); cy.resize(N); gam.resize(N); va.resize(N); gid.resize(N);

    if (!opt_.initial_centres_path.empty()) {
        PalmieriCentresCsvDiagnostics diag{};
        std::string error;
        if (!palmieri_read_centres_csv(opt_.initial_centres_path, N,
                                        (double)side_, p_.target_radius,
                                        &cx, &cy, &diag, &error)) {
            std::fprintf(stderr, "[fatal] invalid --initial-centres '%s': %s\n",
                         opt_.initial_centres_path.c_str(), error.c_str());
            return false;
        }
        std::printf("  initializer      %s\n", kPalmieriInitializerMethod);
        std::printf("  initial centres  %s  (%zu rows, min distance %.9g, "
                    "table FNV-1a %016llx)\n",
                    opt_.initial_centres_path.c_str(), diag.accepted_count,
                    diag.minimum_periodic_distance,
                    (unsigned long long)diag.table_fnv1a64);
        p_.initialization_hash = diag.table_fnv1a64;
    } else {
        const int nx = (int)std::ceil(std::sqrt((double)N));
        const double sp = (double)side_ / (double)nx;
        for (int i = 0; i < N; ++i) {
            const int gxi = i % nx;
            const int gyi = i / nx;
            // Shared RNG helpers reproduce these values after checkpoint load.
            const Philox4 r = philox4x32_10(
                (uint32_t)i, kIcDomainJitter, 0u, 0u,
                (uint32_t)(p_.seed & 0xFFFFFFFFull),
                (uint32_t)(p_.seed >> 32));
            const double jx =
                (philox_uniform53(r.v[0], r.v[1]) - 0.5) * 0.10 * sp;
            const double jy =
                (philox_uniform53(r.v[2], r.v[3]) - 0.5) * 0.10 * sp;
            double x = ((double)gxi + 0.5 + ((gyi & 1) ? 0.5 : 0.0))
                       * sp + jx;
            double y = ((double)gyi + 0.5) * sp + jy;
            x -= std::floor(x / (double)side_) * (double)side_;
            y -= std::floor(y / (double)side_) * (double)side_;
            cx[(std::size_t)i] = (float)x;
            cy[(std::size_t)i] = (float)y;
        }
        p_.initialization_hash = palmieri_centre_table_fnv1a64(cx, cy);
        std::printf("  initializer      grid+jitter (default), table FNV-1a "
                    "%016llx\n",
                    (unsigned long long)p_.initialization_hash);
    }

    const int n_cancer = (int)std::llround(p_.cancer_fraction * (double)N);
    for (int i = 0; i < N; ++i) {
        gam[(std::size_t)i] =
            (float)((i < n_cancer) ? p_.gamma_cancer : p_.gamma_normal);
        // Per-cell v_A disorder: lognormal, median p_.v_A.
        va[(std::size_t)i] =
            (float)ic_v_A(i, p_.seed, p_.v_A, p_.v_A_sigma);
        gid[(std::size_t)i] = i;
    }
    return true;
}

// Allocate and initialize resources shared by fresh and resumed runs.
bool Sim::alloc_device(const SimParams& p, const RunOptions& opt, int device) {
    p_ = p;
    opt_ = opt;
    device_ = device;

    CU_CHECK(cudaSetDevice(device_));

    cudaDeviceProp prop{};
    CU_CHECK(cudaGetDeviceProperties(&prop, device_));
    grid_ = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 132;
    l2_persist_max_ = (size_t)prop.persistingL2CacheMaxSize;
    l2_window_max_  = (size_t)prop.accessPolicyMaxWindowSize;

    side_  = p_.Nx;
    pitch_ = s_pitch_for(side_);
    const int N = p_.num_cells;
    size_t pool_words = 0, pool_bytes = 0, s_buffer_bytes = 0;
    size_t all_s_bytes = 0;
    if (pitch_ <= 0 ||
        !checked_product(static_cast<size_t>(pitch_),
                         static_cast<size_t>(side_), &s_buf_words_) ||
        !checked_product(s_buf_words_, sizeof(uint32_t), &s_buffer_bytes) ||
        !checked_product(s_buffer_bytes, 3, &all_s_bytes) ||
        !checked_product(static_cast<size_t>(N),
                         static_cast<size_t>(kTileArea), &pool_words) ||
        !checked_product(pool_words, sizeof(float), &pool_bytes)) {
        std::fprintf(stderr,
            "[fatal] requested 2D domain or cell pool exceeds host size limits\n");
        return false;
    }

    std::printf("--- device ---\n");
    std::printf("  %s  cc %d.%d  %d SMs  %.1f GiB  L2 %.1f MB "
                "(persist max %.1f MB, window max %.1f MB)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                (double)prop.totalGlobalMem / 1073741824.0,
                (double)prop.l2CacheSize / 1048576.0,
                (double)l2_persist_max_ / 1048576.0,
                (double)l2_window_max_ / 1048576.0);
    std::printf("  phi pool  2 x %.2f MB    S  3 x %.2f MB    cells %.2f MB\n",
                (double)pool_words * 4.0 / 1048576.0,
                (double)s_buf_words_ * 4.0 / 1048576.0,
                (double)N * sizeof(CellState) / 1048576.0);

    if (prop.major < 9)
        std::fprintf(stderr, "[warn] built for sm_90; this device is sm_%d%d\n",
                     prop.major, prop.minor);

    CU_CHECK(cudaMalloc(&d_phi_[0], pool_bytes));
    CU_CHECK(cudaMalloc(&d_phi_[1], pool_bytes));
    CU_CHECK(cudaMalloc(&d_S_, all_s_bytes));
    CU_CHECK(cudaMalloc(&d_cell_, (size_t)N * sizeof(CellState)));
    CU_CHECK(cudaMalloc(&d_cls_, (size_t)N));
    CU_CHECK(cudaMalloc(&d_perm_, (size_t)N * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&d_cursor_, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMalloc(&d_step_, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMalloc(&d_flags_, FLAG_COUNT * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&d_vchk_, (size_t)N * sizeof(double)));
    CU_CHECK(cudaMalloc(&d_ochk_, (size_t)N * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_smax_, sizeof(uint32_t)));

    // The GPU writes trajectory records directly into mapped pinned memory.
    CU_CHECK(cudaHostAlloc((void**)&h_traj_, (size_t)N * sizeof(TrajPackedCell),
                           cudaHostAllocMapped));
    CU_CHECK(cudaHostGetDevicePointer((void**)&d_traj_, h_traj_, 0));

    CU_CHECK(cudaStreamCreate(&stream_));

    CU_CHECK(cudaMemset(d_cursor_, 0, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMemset(d_step_, 0, 2 * sizeof(unsigned long long)));
    CU_CHECK(cudaMemset(d_flags_, 0, FLAG_COUNT * sizeof(uint32_t)));
    CU_CHECK(cudaMemset(d_S_, 0, all_s_bytes));
    return true;
}

// Configure kernels and performance features after state initialization.
bool Sim::configure_and_capture() {
    const int N = p_.num_cells;
    const long long total = p_.total_steps();
    trajectory_every_ = opt_.traj_interval > 0
        ? opt_.traj_interval
        : std::max<long long>(1, total / std::max(1, opt_.traj_samples));

    // Configure dynamic shared memory before any step launch; the
    // MaxDynamicSharedMemorySize attribute is per-kernel and a missing opt-in
    // is a launch failure, not a slowdown.
    configure_k_step_smem();
    CU_CHECK(cudaGetLastError());

    cudaFuncAttributes fa{};
    CU_CHECK(cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(k_step)));
    std::printf("  k_step: %d regs, %zu B local frame/thread, %d B static smem, "
                "%d B dynamic smem requested\n",
                fa.numRegs, (size_t)fa.localSizeBytes,
                (int)fa.sharedSizeBytes, kSmemBytes);
    // localSizeBytes includes the ABI stack frame and any spills; only the
    // ptxas report distinguishes them.
    if (fa.localSizeBytes > kLocalBytesBudget)
        std::fprintf(stderr,
            "[warn] k_step uses %zu B/thread of local memory, over the %zu B "
            "budget. Check the ptxas spill-store report for the breakdown.\n",
            (size_t)fa.localSizeBytes, kLocalBytesBudget);

    cudaFuncAttributes fallback_fa{};
    CU_CHECK(cudaFuncGetAttributes(
        &fallback_fa, reinterpret_cast<const void*>(k_step_fallback)));
    std::printf("  k_step_fallback: %d regs, %zu B local frame/thread, "
                "%d B static smem, %d B dynamic smem requested\n",
                fallback_fa.numRegs, (size_t)fallback_fa.localSizeBytes,
                (int)fallback_fa.sharedSizeBytes, kScalarBytes);

    print_path_report();

    // Reserve persisting L2 only when both hot S buffers fit; otherwise the
    // reservation would evict phi without retaining the full working set.
    if (opt_.l2_persist && l2_persist_max_ > 0) {
        const size_t budget = (size_t)(0.85 * (double)l2_persist_max_);
        const size_t need   = 2 * s_buf_words_ * sizeof(uint32_t);
        if (need <= budget) {
            CU_WARN(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, need));
            std::printf("  L2 persisting carve-out %.2f MB (S buffer %.2f MB each)\n",
                        (double)need / 1048576.0,
                        (double)s_buf_words_ * 4.0 / 1048576.0);
        } else {
            opt_.l2_persist = false;
            std::printf("  L2 persisting carve-out DISABLED: 2 x S = %.2f MB exceeds "
                        "the %.2f MB budget\n"
                        "      (reserving it would evict phi without holding S)\n",
                        (double)need / 1048576.0, (double)budget / 1048576.0);
        }
    }

    if (opt_.morton) {
        int M = 1;
        while (M < N) M <<= 1;
        const size_t sm = (size_t)M * sizeof(unsigned long long);
        if (sm > (size_t)kSmemPerBlockOptinSm90) {
            std::fprintf(stderr,
                "[warn] Morton sort needs %.1f KB of shared memory for N=%d; "
                "over the per-block limit. Falling back to identity order.\n",
                (double)sm / 1024.0, N);
            opt_.morton = false;
        } else {
            configure_morton_smem((int)sm);
            CU_CHECK(cudaGetLastError());
        }
    }

    if (opt_.use_graph) {
        if (!build_graph())
            std::fprintf(stderr, "[warn] graph capture failed; using per-step "
                                 "launches.\n");
    }
    return true;
}

bool Sim::init(const SimParams& p, const RunOptions& opt, int device) {
    // Validate the host initial condition before opening a GPU context.
    p_ = p;
    opt_ = opt;
    side_ = p.Nx;
    std::vector<float> cx, cy, gam, va;
    std::vector<int32_t> gid;
    if (!seed_positions(cx, cy, gam, va, gid)) return false;

    const SimParams initialized_params = p_;
    if (!alloc_device(initialized_params, opt, device)) return false;
    const int N = p_.num_cells;


    std::vector<CellState> h_cell((size_t)N);
    std::memset(h_cell.data(), 0, h_cell.size() * sizeof(CellState));
    std::vector<uint8_t> h_cls((size_t)N, (uint8_t)kClassRound);
    std::vector<uint32_t> h_perm((size_t)N);
    for (int i = 0; i < N; ++i) {
        CellState& c = h_cell[(size_t)i];
        c.global_id = gid[i];
        c.gamma = gam[i];
        c.v_A   = va[i];
        c.R_tgt = (float)p_.target_radius;
        c.theta = ic_theta(gid[i], p_.polarity_stream());
        c.cls = (uint8_t)kClassRound;
        c.cls_written[0] = (uint8_t)kClassRound;
        c.cls_written[1] = (uint8_t)kClassRound;
        h_perm[(size_t)i] = (uint32_t)i;
    }
    CU_CHECK(cudaMemcpy(d_cell_, h_cell.data(), h_cell.size() * sizeof(CellState),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_cls_, h_cls.data(), h_cls.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_perm_, h_perm.data(), h_perm.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    float *d_cx = nullptr, *d_cy = nullptr;
    CU_CHECK(cudaMalloc(&d_cx, (size_t)N * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_cy, (size_t)N * sizeof(float)));
    CU_CHECK(cudaMemcpy(d_cx, cx.data(), (size_t)N * sizeof(float),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_cy, cy.data(), (size_t)N * sizeof(float),
                        cudaMemcpyHostToDevice));

    k_init_tiles<<<N, 256>>>(d_phi_[0], d_phi_[1], d_cell_, d_cls_, N, side_,
                             d_cx, d_cy,
                             (float)init_radius(p_.target_radius, p_.lambda),
                             (float)interface_k(p_.lambda));
    CU_CHECK(cudaGetLastError());
    k_init_moments<<<N, kBlockThreads>>>(d_cell_, d_cls_, d_phi_[0], N);
    CU_CHECK(cudaGetLastError());
    k_scatter_all<<<N, 256>>>(d_phi_[0], d_cell_, d_cls_, d_S_, N, side_,
                              pitch_, d_flags_);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());
    CU_WARN(cudaFree(d_cx));
    CU_WARN(cudaFree(d_cy));

    return configure_and_capture();
}

// Resume invariants: load phi into step%2, scatter S into step%3, and restore
// the step used as the Philox counter. Per-step launches advance an arbitrary
// restart point to the next six-step graph boundary.
bool Sim::init_from_checkpoint(const SimParams& p, const CheckpointData& d,
                               const RunOptions& opt, int device) {
    if ((int)d.cells.size() != d.n || d.phi.size() != (size_t)d.n * kTileArea) {
        std::fprintf(stderr, "[ckpt] payload is internally inconsistent "
                     "(%zu cells, %zu phi words, n=%d)\n",
                     d.cells.size(), d.phi.size(), d.n);
        return false;
    }
    if (p.num_cells != d.n) {
        std::fprintf(stderr, "[ckpt] params say %d cells, the file has %d\n",
                     p.num_cells, d.n);
        return false;
    }
    if (d.step < 0) {
        std::fprintf(stderr, "[ckpt] negative step %lld\n", d.step);
        return false;
    }
    if (p.total_steps() < d.step) {
        std::fprintf(stderr,
            "[ckpt] requested t_end ends at step %lld, before checkpoint "
            "step %lld\n",
            p.total_steps(), d.step);
        return false;
    }
    if (!alloc_device(p, opt, device)) return false;
    const int N = p_.num_cells;

    steps_done_ = d.step;
    const int pin = (int)(steps_done_ % 2);
    const int rot = (int)(steps_done_ % 3);

    std::vector<CellState> h_cell((size_t)N);
    std::memset(h_cell.data(), 0, h_cell.size() * sizeof(CellState));
    std::vector<uint8_t>  h_cls((size_t)N);
    std::vector<uint32_t> h_perm((size_t)N);
    for (int i = 0; i < N; ++i) {
        const CkptCell& s = d.cells[(size_t)i];
        CellState& c = h_cell[(size_t)i];
        c.global_id = s.global_id;
        c.gx0 = s.origin[0];
        c.gy0 = s.origin[1];
        c.gamma = s.gamma;
        c.v_A   = s.v_A;
        c.R_tgt = s.R_tgt;
        c.theta = s.theta;
        c.vx = s.vx;
        c.vy = s.vy;
        c.V = s.volume_moment;
        c.Cx = s.moment_x;
        c.Cy = s.moment_y;
        c.perim = s.perimeter;
        c.bb_lo_x = s.support_lo_x;
        c.bb_hi_x = s.support_hi_x;
        c.bb_lo_y = s.support_lo_y;
        c.bb_hi_y = s.support_hi_y;
        c.promote_ctr = s.promote_ctr;
        c.phi_max = s.phi_max;
        c.cls = s.cls;
        // Both phi buffers are initialized with the current class layout.
        c.cls_written[0] = s.cls;
        c.cls_written[1] = s.cls;
        h_cls[(size_t)i]  = s.cls;
        h_perm[(size_t)i] = (uint32_t)i;
    }
    CU_CHECK(cudaMemcpy(d_cell_, h_cell.data(), h_cell.size() * sizeof(CellState),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_cls_, h_cls.data(), h_cls.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_perm_, h_perm.data(), h_perm.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

    const size_t pool_bytes = (size_t)N * kTileArea * sizeof(float);
    CU_CHECK(cudaMemcpy(d_phi_[pin], d.phi.data(), pool_bytes,
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_phi_[1 - pin], 0, pool_bytes));

    const unsigned long long h_step[2] = {
        (unsigned long long)(pin == 0 ? steps_done_ : steps_done_ + 1),
        (unsigned long long)(pin == 1 ? steps_done_ : steps_done_ + 1)};
    CU_CHECK(cudaMemcpy(d_step_, h_step, sizeof(h_step), cudaMemcpyHostToDevice));

    // The checkpoint restores the moments, support, adaptive class, and
    // demotion counter consumed by the next update. Recomputing them here
    // would change a floating-point reduction and break restart equivalence.
    k_scatter_all<<<N, 256>>>(d_phi_[pin], d_cell_, d_cls_,
                              d_S_ + (size_t)rot * s_buf_words_, N, side_,
                              pitch_, d_flags_);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaDeviceSynchronize());

    // Time is derived as step*dt; warn when an overridden dt changes it.
    if (std::fabs(d.t - time()) > 1e-6 * std::max(1.0, std::fabs(d.t)))
        std::fprintf(stderr,
            "[ckpt] warning: the file records t = %.6f at step %lld, but "
            "step * dt = %.6f with the dt now in force. Time is derived from "
            "the step count here, so the run continues at %.6f.\n",
            d.t, steps_done_, time(), time());

    std::printf("  resumed at step %lld (t = %.4f), phi parity %d, "
                "S rotation slot %d\n", steps_done_, time(), pin, rot);

    return configure_and_capture();
}

// Serialize the current phi buffer; the writer stages D2H transfers in bounded
// chunks.
std::vector<std::string> Sim::checkpoint_paths(bool rolling, bool tagged) const {
    std::vector<std::string> paths;
    if (opt_.ckpt_dir.empty()) return paths;
    char buf[1100];
    if (rolling) {
        std::snprintf(buf, sizeof(buf), "%s/checkpoint.bin",
                      opt_.ckpt_dir.c_str());
        paths.emplace_back(buf);
    }
    if (tagged) {
        std::snprintf(buf, sizeof(buf), "%s/checkpoint_%08lld.bin",
                      opt_.ckpt_dir.c_str(), steps_done_);
        paths.emplace_back(buf);
    }
    return paths;
}

bool Sim::save_checkpoint(const std::vector<std::string>& paths) {
    if (paths.empty()) return true;
    const int N = p_.num_cells;
    const int pin = (int)(steps_done_ % 2);   // the buffer holding phi^step

    std::vector<CellState> h((size_t)N);
    std::vector<uint8_t>   hc((size_t)N);
    CU_CHECK(cudaMemcpy(h.data(), d_cell_, h.size() * sizeof(CellState),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(hc.data(), d_cls_, hc.size(), cudaMemcpyDeviceToHost));

    CheckpointWriteView v;
    v.p    = &p_;
    v.step = steps_done_;
    v.t    = time();
    v.N    = N;
    v.L    = side_;
    v.cell = h.data();
    v.cls  = hc.data();
    v.d_phi = d_phi_[pin];
    v.trajectory_samples = opt_.traj_samples;
    v.trajectory_interval = trajectory_every_;
    v.save_interval = static_cast<int>(opt_.save_interval);
    return checkpoint_write(v, paths);
}

// Report achieved occupancy and its register/shared-memory limits.
void Sim::print_path_report() const {
    auto report = [&](const char* name, const void* fn, int threads,
                      int dyn_smem, int target_ctas, int reg_budget) {
        KernelStats s{};
        if (!query_kernel_stats(fn, threads, dyn_smem, device_, &s)) {
            std::fprintf(stderr, "[warn] occupancy query failed for %s\n", name);
            return;
        }
        std::printf("  %-12s %3d thr  %3d regs (budget %d)  %6d B smem "
                     "(%d static + %d dyn)  %zu B local frame/thread\n",
                    name, threads, s.regs, reg_budget,
                    s.static_smem + s.dynamic_smem, s.static_smem,
                    s.dynamic_smem, s.local_bytes);
        std::printf("  %-12s ACHIEVED %d CTAs/SM  (%d warps/SM, %.1f%% "
                    "occupancy);  register ceiling %d CTAs/SM;  target %d\n",
                    "", s.ctas_per_sm, s.warps_per_sm, 100.0 * s.occupancy,
                    s.reg_limited_ctas, target_ctas);
        if (s.local_bytes > kLocalBytesBudget)
            std::fprintf(stderr,
                "[warn] %s uses %zu B/thread of local memory; verify ptxas "
                "reports zero spills.\n", name, s.local_bytes);
        if (s.ctas_per_sm < target_ctas)
            std::fprintf(stderr,
                "[warn] %s reached %d CTAs/SM instead of %d (%s); performance "
                "may differ from the expected occupancy.\n",
                name, s.ctas_per_sm, target_ctas,
                s.reg_limited_ctas < target_ctas ? "register-limited"
                                                 : "shared-memory-limited");
    };

    std::printf("  step kernels: shared-class update then fallback filter "
                "(two ordered launches per step)\n");
    report("k_step", reinterpret_cast<const void*>(k_step),
           kBlockThreads, kSmemBytes, 1,
           kRegsPerSmSm90 / kBlockThreads);   // __launch_bounds__(768, 1)
    report("fallback", reinterpret_cast<const void*>(k_step_fallback),
           kBlockThreads, kScalarBytes, 1,
           kRegsPerSmSm90 / kBlockThreads);
}

// Per-slot argument baking. slot in [0, 6): phi parity = slot % 2, S rotation
// slot = slot % 3, cursor and step-counter slots follow the phi parity.
StepArgs Sim::args_for_slot(int slot) const {
    const int pin  = slot % 2;
    const int pout = 1 - pin;
    const int rs   = slot % 3;

    StepArgs A{};
    A.phi_in  = d_phi_[pin];
    A.phi_out = d_phi_[pout];
    A.S_rd = d_S_ + (size_t)rs * s_buf_words_;
    A.S_sc = d_S_ + (size_t)((rs + 1) % 3) * s_buf_words_;
    A.S_cl = d_S_ + (size_t)((rs + 2) % 3) * s_buf_words_;
    A.cell = d_cell_;
    A.cell_cls = d_cls_;
    A.perm = d_perm_;
    A.cursor_use   = d_cursor_ + pin;
    A.cursor_clear = d_cursor_ + pout;
    A.step_rd = d_step_ + pin;
    A.step_wr = d_step_ + pout;
    A.flags = d_flags_;

    A.N = p_.num_cells;
    A.L = side_;
    A.P = pitch_;
    A.parity_out = pout;

    A.dt         = (float)p_.dt;
    A.A0         = p_.area0();
    A.vol_scale  = p_.volume();
    A.bulk_scale = (float)p_.bulk();
    A.rep_coeff  = (float)p_.interaction();
    A.mot_coeff  = (float)p_.motility();

    A.seed = p_.seed;
    A.polarity_seed = p_.polarity_stream();
    A.p_tumble = p_.p_tumble();
    A.full_moment_every = p_.full_moment_every;
    A.clear_ahead_words = (unsigned long long)s_buf_words_;
    return A;
}

// Do not pin the write-only clear-ahead buffer in L2. Pin the read buffer and,
// when contiguous and within budget, the scatter buffer.
void Sim::l2_window_for_slot(int slot, const void** base, size_t* bytes,
                             float* hit) const {
    *base = nullptr; *bytes = 0; *hit = 0.0f;
    if (!opt_.l2_persist || l2_persist_max_ == 0) return;

    const size_t buf = s_buf_words_ * sizeof(uint32_t);
    const int rs = slot % 3;
    const int ss = (rs + 1) % 3;

    size_t nb = buf;
    int    b0 = rs;
    if (ss == rs + 1) { nb = 2 * buf; b0 = rs; }      // contiguous read+scatter

    const size_t cap = std::min((size_t)(0.85 * (double)l2_persist_max_),
                                l2_window_max_);
    if (cap == 0) return;
    if (nb > cap) { nb = buf; }                        // read buffer only
    if (nb > cap) { nb = cap; }                        // still too big: clip

    *base  = (const void*)(d_S_ + (size_t)b0 * s_buf_words_);
    *bytes = nb;
    const size_t resident = std::min(nb, cap);
    *hit = (float)((double)resident / (double)nb);
    return;
}

void Sim::launch_one(int slot) {
    if (opt_.morton && (slot % kMortonEvery) == 0) {
        int M = 1;
        while (M < p_.num_cells) M <<= 1;
        k_morton_sort<<<1, 1024, (size_t)M * sizeof(unsigned long long),
                        stream_>>>(d_cell_, d_perm_, p_.num_cells, M, side_);
    }
    const void* base = nullptr; size_t nb = 0; float hit = 0.0f;
    l2_window_for_slot(slot, &base, &nb, &hit);
    launch_step(args_for_slot(slot), grid_, stream_, base, nb, hit);
}

bool Sim::build_graph() {
    CU_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    for (int s = 0; s < kGraphBody; ++s) launch_one(s);
    CU_CHECK(cudaStreamEndCapture(stream_, &graph_));
    CU_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_, graph_, 0));
    graph_ready_ = true;
    std::printf("  CUDA graph: %d-step body captured, %d kernel nodes (lcm of 2 "
                "phi parities and 3 S rotation slots)%s\n", kGraphBody,
                2 * kGraphBody,
                opt_.morton ? " + Morton sort at slot 0" : "");
    return true;
}

void Sim::print_line() {
    std::vector<CellState> h((size_t)p_.num_cells);
    if (cudaMemcpy(h.data(), d_cell_, h.size() * sizeof(CellState),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return;
    double vsum = 0.0, vmin = 1e300, vmax = -1e300, spd = 0.0, pmax = 0.0;
    long long shifts = 0, tumbles = 0;
    int cls_count[kNumClasses] = {};
    int cls_bad = 0;
    int fallback_seen = 0;
    unsigned long long no_margin_steps = 0;
    for (const CellState& c : h) {
        vsum += c.V;
        vmin = std::min(vmin, c.V);
        vmax = std::max(vmax, c.V);
        spd += std::sqrt((double)c.vx * c.vx + (double)c.vy * c.vy);
        pmax = std::max(pmax, (double)c.phi_max);
        shifts += c.shift_ctr;
        tumbles += c.tumble_ctr;
        if (c.cls < kNumClasses) cls_count[c.cls]++; else cls_bad++;
        fallback_seen += c.reserved[0] != 0u;
        no_margin_steps += c.reserved[1];
    }
    const double n = (double)p_.num_cells;
    std::printf("step %8lld  t %10.3f  <V>/A0 %.5f  V range [%.1f %.1f]  "
                "<|v|> %.4e  max|phi| %.5f  "
                "cls(r/w/t/sq/large/fallback) %d/%d/%d/%d/%d/%d bad=%d  "
                "shifts %lld  tumbles %lld\n",
                steps_done_, time(), vsum / n / p_.area0(), vmin, vmax,
                spd / n, pmax, cls_count[0], cls_count[1], cls_count[2],
                cls_count[3], cls_count[4], cls_count[5], cls_bad,
                shifts, tumbles);
    if (fallback_seen && !fallback_reported_) {
        std::printf("[geometry] fixed-tile fallback used by %d cell(s): "
                    "%dx%d interior in the %dx%d tile\n",
                    fallback_seen, kClasses[kClassFallback].wx,
                    kClasses[kClassFallback].wy, kTilePitch, kTilePitch);
        fallback_reported_ = true;
    }
    if (no_margin_steps && !fallback_no_margin_reported_) {
        std::printf("[geometry] fallback margin or boundary reached; output is "
                    "retained, but boundary dynamics may be clipped and require "
                    "review (cell-steps %llu)\n",
                    no_margin_steps);
        fallback_no_margin_reported_ = true;
    }
    std::fflush(stdout);
}

// Poll fatal integrity flags independently of output cadence. A failed readback
// is fatal because run validity can no longer be established.
bool Sim::fatal_flag_set() {
    uint32_t f[FLAG_COUNT] = {0};
    const cudaError_t e =
        cudaMemcpy(f, d_flags_, sizeof(f), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(stderr,
            "\n*** STOPPING AT STEP %lld: integrity-flag readback failed: %s ***\n"
            "    Run validity can no longer be established.\n",
            steps_done_, cudaGetErrorString(e));
        return true;
    }

    bool fatal = false;
    for (int i = 0; i < FLAG_COUNT; ++i)
        fatal = fatal || (f[i] != 0u && flag_is_fatal(i));
    if (!fatal) return false;

    std::fprintf(stderr,
        "\n*** STOPPING AT STEP %lld: FATAL INTEGRITY CHECK ***\n",
        steps_done_);
    for (int i = 0; i < FLAG_COUNT; ++i)
        if (f[i] != 0u && flag_is_fatal(i))
            std::fprintf(stderr, "    %-18s %u\n", flag_name(i), f[i]);
    if (f[FLAG_CLASS_EXHAUSTED] != 0u)
        std::fprintf(stderr,
            "    A cell outgrew every available shape class; its field can no "
            "longer be represented without truncation.\n");
    if (f[FLAG_CLASS_UNSUPPORTED] != 0u)
        std::fprintf(stderr, "    A CellState carried a shape class "
                             "outside the supported range.\n");
    std::fprintf(stderr,
        "    The trajectory is invalid from the first flagged step. A final "
        "diagnostic checkpoint will be attempted.\n");
    return true;
}

void Sim::report_flags() const {
    uint32_t f[FLAG_COUNT] = {0};
    const cudaError_t e =
        cudaMemcpy(f, d_flags_, sizeof(f), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[checks] flag readback failed: %s\n",
                     cudaGetErrorString(e));
        return;
    }
    // support_clip is advisory because it reports support touching a window
    // edge, not necessarily lost field mass. class_exhausted is fatal because
    // no available window contains the support.
    bool fatal = false, advisory = false;
    for (int i = 0; i < FLAG_COUNT; ++i) {
        if (!f[i]) continue;
        if (i == FLAG_SUPPORT_CLIP) advisory = true;
        else if (flag_is_fatal(i)) fatal = true;
    }
    if (!fatal && !advisory) { std::printf("integrity checks: all clear\n"); return; }

    if (fatal) {
        std::printf("fatal integrity flags set; trajectory is invalid from "
                    "the first flagged step\n");
        for (int i = 0; i < FLAG_COUNT; ++i)
            if (f[i] && flag_is_fatal(i))
                std::printf("  %-18s %u\n", flag_name(i), f[i]);
    } else {
        std::printf("integrity checks: no fatal flags\n");
    }
    if (advisory) {
        const double frac = (double)f[FLAG_SUPPORT_CLIP]
                          / ((double)p_.num_cells * (double)(steps_done_ ? steps_done_ : 1));
        std::printf("  advisory: %-18s %u  (%.3f%% of cell-steps)\n"
                    "    phi support touched the active-window edge; persistent "
                    "events may require a larger shape class.\n",
                    flag_name(FLAG_SUPPORT_CLIP), f[FLAG_SUPPORT_CLIP], 100.0 * frac);
    }
#if !PF_SUPPORT_CLIP_ENABLED
    std::printf("  advisory: support_clip instrumentation is disabled\n");
#endif
}

bool Sim::verify(double* max_rel_V, float* max_outside, uint32_t* max_S) {
    const int N = p_.num_cells;
    const int pin = (int)(steps_done_ % 2);
    CU_CHECK(cudaMemset(d_smax_, 0, sizeof(uint32_t)));
    k_verify_cells<<<N, kBlockThreads, 0, stream_>>>(d_phi_[pin], d_cell_,
                                                     d_cls_, N, d_vchk_, d_ochk_);
    k_verify_S<<<256, 256, 0, stream_>>>(d_S_ + (size_t)(steps_done_ % 3)
                                             * s_buf_words_,
                                         s_buf_words_, d_smax_);
    CU_CHECK(cudaStreamSynchronize(stream_));

    std::vector<double> v((size_t)N);
    std::vector<float>  o((size_t)N);
    std::vector<CellState> h((size_t)N);
    CU_CHECK(cudaMemcpy(v.data(), d_vchk_, v.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(o.data(), d_ochk_, o.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(h.data(), d_cell_, h.size() * sizeof(CellState),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(max_S, d_smax_, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    double mr = 0.0;
    float mo = 0.0f;
    for (int i = 0; i < N; ++i) {
        const double den = std::max(1e-12, std::fabs(h[(size_t)i].V));
        mr = std::max(mr, std::fabs(v[(size_t)i] - h[(size_t)i].V) / den);
        mo = std::max(mo, o[(size_t)i]);
    }
    *max_rel_V = mr;
    *max_outside = mo;
    return true;
}

bool Sim::run() {
    const long long total = p_.total_steps();
    const long long pi = p_.print_interval > 0 ? p_.print_interval : total;
    bool run_failed = false;

    // Use thresholds because graph replay advances in blocks and may skip exact
    // multiples of a cadence.
    long long next_verify = p_.verify_every > 0
        ? next_interval_boundary(steps_done_, p_.verify_every) : total + 1;

    // Sampling is aligned to absolute integration-step boundaries, so a
    // checkpoint continuation does not shift the trajectory cadence.
    // Checkpoints retain the resolved cadence through legs with no output.
    const long long traj_every = !opt_.out_path.empty()
        ? trajectory_every_ : total + 1;
    long long next_traj = next_interval_boundary(steps_done_, traj_every);

    // Fold checkpoint thresholds into next_stop so graph replay cannot skip them.
    const bool ckpt_on = !opt_.ckpt_dir.empty();
    if (ckpt_on &&
        (opt_.ckpt_interval > 0 || opt_.save_interval > 0 ||
         opt_.final_checkpoint) &&
        total > std::numeric_limits<std::int32_t>::max()) {
        std::fprintf(stderr,
            "[fatal] the current checkpoint schema supports at most %d steps\n",
            std::numeric_limits<std::int32_t>::max());
        return false;
    }
    const long long ckpt_every = ckpt_on ? opt_.ckpt_interval : 0;
    const long long save_every = ckpt_on ? opt_.save_interval : 0;
    long long next_ckpt = ckpt_every > 0 ? steps_done_ + ckpt_every : total + 1;
    long long next_save = save_every > 0 ? steps_done_ + save_every : total + 1;
    long long next_fatal_poll = steps_done_ + kFatalCheckPollEvery;

    if (!open_trajectory(opt_.out_path)) return false;

    while (steps_done_ < total) {
        const long long next_stop =
            std::min({total, ((steps_done_ / pi) + 1) * pi, next_traj,
                      next_ckpt, next_save, next_fatal_poll});

        if (graph_ready_ && (steps_done_ % kGraphBody) == 0 &&
            steps_done_ + kGraphBody <= next_stop) {
            CU_WARN(cudaGraphLaunch(graph_exec_, stream_));
            steps_done_ += kGraphBody;
        } else {
            launch_one((int)(steps_done_ % kGraphBody));
            steps_done_ += 1;
        }

        const bool do_print = (steps_done_ % pi == 0) || (steps_done_ >= total);
        const bool do_traj  = !opt_.out_path.empty() && steps_done_ >= next_traj;
        const bool do_ckpt  = ckpt_every > 0 && steps_done_ >= next_ckpt;
        const bool do_save  = save_every > 0 && steps_done_ >= next_save;
        const bool do_fatal_poll = steps_done_ >= next_fatal_poll;

        if (do_print || do_traj || do_ckpt || do_save || do_fatal_poll) {
            CU_WARN(cudaStreamSynchronize(stream_));
            if (fatal_flag_set()) {
                run_failed = true;
                break;
            }
            if (do_fatal_poll)
                next_fatal_poll = steps_done_ + kFatalCheckPollEvery;
            if (do_print) print_line();
            if (do_traj) {
                const int N = p_.num_cells;
                k_pack_traj<<<(N + 127) / 128, 128, 0, stream_>>>(d_cell_, d_cls_,
                                                                  d_traj_, N, side_);
                CU_WARN(cudaStreamSynchronize(stream_));
                if (!append_trajectory_frame(steps_done_)) {
                    run_failed = true;
                    break;
                }
                next_traj = next_interval_boundary(steps_done_, traj_every);
            }
            // One gather feeds both files when both fall due on the same step.
            if (do_ckpt || do_save) {
                if (!save_checkpoint(checkpoint_paths(do_ckpt, do_save))) {
                    std::fprintf(stderr,
                        "[ckpt] checkpoint failed at step %lld; stopping to "
                        "avoid an unresumable allocation.\n",
                        steps_done_);
                    run_failed = true;
                    break;
                }
                if (do_ckpt) next_ckpt = steps_done_ + ckpt_every;
                if (do_save) next_save = steps_done_ + save_every;
            }
        }

        // Cooperative shutdown retains all trajectory frames already streamed.
        if (s_terminate) {
            CU_WARN(cudaStreamSynchronize(stream_));
            std::printf("\n[signal] termination requested at step %lld (t = %.3f); "
                        "%lld trajectory frames already written\n",
                        steps_done_, time(), traj_frames_);
            std::fflush(stdout);
            break;
        }
        if (opt_.strict && (p_.verify_every > 0) && steps_done_ >= next_verify) {
            double mr = 0.0; float mo = 0.0f; uint32_t ms = 0;
            if (verify(&mr, &mo, &ms))
                std::printf("  [verify] step %lld  max rel V error %.3e   "
                            "max|phi| outside window %.3e   max S %.6f\n",
                            steps_done_, mr, (double)mo, (double)ms * kQInvD);
            next_verify = steps_done_ + p_.verify_every;
        }
    }
    CU_WARN(cudaStreamSynchronize(stream_));
    if (!run_failed && fatal_flag_set())
        run_failed = true;
    if (traj_fp_)
        std::printf("trajectory -> %lld frames x %d cells (streamed)\n",
                    traj_frames_, p_.num_cells);
    if (!close_trajectory())
        run_failed = true;
    // Preserve the last accepted rolling checkpoint when a fatal state occurs.
    if (ckpt_on && (opt_.final_checkpoint || run_failed)) {
        std::vector<std::string> final_paths;
        if (run_failed)
            final_paths.emplace_back(opt_.ckpt_dir + "/checkpoint_failed.bin");
        else
            final_paths = checkpoint_paths(true, false);
        if (!save_checkpoint(final_paths)) {
            std::fprintf(stderr, "[ckpt] FINAL CHECKPOINT FAILED. The run "
                         "cannot be resumed from where it stopped.\n");
            run_failed = true;
        }
    } else if (run_failed && !ckpt_on) {
        std::fprintf(stderr,
            "[ckpt] fatal run has no checkpoint directory; no diagnostic "
            "checkpoint can be written.\n");
    }
    report_flags();
    return !run_failed;
}

bool Sim::bench(int steps, double* ms_per_step) {
    // Warm-up: fill caches, resolve the first graph upload, settle clocks.
    const int warm = std::min(steps, 200);
    for (int i = 0; i < warm; ++i) { launch_one((int)(steps_done_ % kGraphBody)); ++steps_done_; }
    CU_CHECK(cudaStreamSynchronize(stream_));

    cudaEvent_t e0, e1;
    CU_CHECK(cudaEventCreate(&e0));
    CU_CHECK(cudaEventCreate(&e1));

    // Align to a graph-body boundary so the timed region is pure graph replay.
    while ((steps_done_ % kGraphBody) != 0) {
        launch_one((int)(steps_done_ % kGraphBody));
        ++steps_done_;
    }
    CU_CHECK(cudaStreamSynchronize(stream_));

    const long long timed_from = steps_done_;
    CU_CHECK(cudaEventRecord(e0, stream_));
    while (steps_done_ - timed_from < steps) {
        if (graph_ready_) {
            CU_CHECK(cudaGraphLaunch(graph_exec_, stream_));
            steps_done_ += kGraphBody;
        } else {
            launch_one((int)(steps_done_ % kGraphBody));
            steps_done_ += 1;
        }
    }
    CU_CHECK(cudaEventRecord(e1, stream_));
    CU_CHECK(cudaEventSynchronize(e1));

    float ms = 0.0f;
    CU_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const long long done = steps_done_ - timed_from;
    *ms_per_step = (double)ms / (double)done;
    CU_WARN(cudaEventDestroy(e0));
    CU_WARN(cudaEventDestroy(e1));

    std::printf("bench: %lld steps in %.3f ms -> %.6f ms/step "
                "(%.2f us/step, %d cells, L=%d, %s, %s)\n",
                done, (double)ms, *ms_per_step, *ms_per_step * 1000.0,
                p_.num_cells, side_, "shared+fallback",
                graph_ready_ ? "graph" : "per-step launch");
    report_flags();
    return true;
}

volatile std::sig_atomic_t Sim::s_terminate = 0;

bool Sim::open_trajectory(const std::string& path) {
    if (path.empty()) return true;
    TrajectoryAppendInfo2D info{};
    if (!open_trajectory_2d(path, p_, side_, steps_done_, trajectory_every_,
                            &info, &traj_fp_))
        return false;
    traj_frames_ = info.frames;
    return true;
}

bool Sim::append_trajectory_frame(long long step_at) {
    if (!traj_fp_) return true;
    const int N = p_.num_cells;
    const double tgt_r = p_.target_radius;
    const double Lw = (double)side_;
    auto wrap_d = [Lw](double v) {
        double m = std::fmod(v, Lw);
        if (m < 0.0) m += Lw;
        return m;
    };
    for (int i = 0; i < N; ++i) {
        const TrajPackedCell& c = h_traj_[i];
        const double th = (double)c.theta;
        const double l_n = (double)c.perim / (2.0 * kPi * tgt_r);
        if (std::fprintf(
                traj_fp_,
                "%.17g %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                (double)step_at * p_.dt, c.global_id,
                wrap_d((double)c.cx), wrap_d((double)c.cy),
                (double)c.vx, (double)c.vy,
                std::cos(th), std::sin(th), th,
                (double)c.v_A, l_n, (double)c.volume) < 0) {
            std::fprintf(stderr,
                         "[error] trajectory write failed at step %lld, cell %d\n",
                         step_at, i);
            return false;
        }
    }
    // Flush each frame so an interrupted run retains sampled data.
    if (std::fflush(traj_fp_) != 0 || std::ferror(traj_fp_)) {
        std::fprintf(stderr, "[error] trajectory flush failed at step %lld\n",
                     step_at);
        return false;
    }
    ++traj_frames_;
    return true;
}

bool Sim::close_trajectory() {
    if (!traj_fp_) return true;
    bool ok = std::fflush(traj_fp_) == 0 && !std::ferror(traj_fp_);
    if (std::fclose(traj_fp_) != 0) ok = false;
    traj_fp_ = nullptr;
    if (!ok)
        std::fprintf(stderr, "[error] failed to finalize trajectory output\n");
    return ok;
}

}  // namespace pf
