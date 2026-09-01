#pragma once
// CUDA kernel interfaces and state shared by host and device code. Each update
// assigns one thread block to a cell and evaluates its interaction velocity
// from the current phase field rather than a one-step-lagged value.

#include "params.cuh"
#include "philox.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 11070)
#define PF_GRID_CONSTANT __grid_constant__
#else
#define PF_GRID_CONSTANT
#endif

namespace pf {

// Per-cell state. V, Cx, Cy, and the support bounds describe the phase field
// read by the next update. A 64-byte alignment preserves the 192-byte record
// size; 128-byte alignment would pad each record to 256 bytes.
struct alignas(64) CellState {
    int32_t  gx0, gy0;            // global coords of rect pixel (0,0), in [0,L)
    uint8_t  cls;                 // shape class of the field to be read
    uint8_t  cls_written[2];      // class last written into phi[parity]
    uint8_t  pad0;
    int32_t  global_id;           // RNG key; stable across any reordering
    float    gamma, v_A, R_tgt;   // stiffness, active speed, and target radius
    float    theta;               // polarity angle; p_hat = (cos,sin) => |p|==1
    double   V;                   // sum(phi^2) of the field about to be read
    double   Cx, Cy;              // sum(phi^2 x), sum(phi^2 y), rect coords
    double   perim;               // full-moment steps only
    int32_t  bb_lo_x, bb_hi_x, bb_lo_y, bb_hi_y;   // support bbox, rect coords
    float    vx, vy;              // current cell velocity
    double   Ix, Iy;              // phase-field interaction integral
    uint32_t promote_ctr;         // demote-hysteresis dwell counter
    uint32_t shift_ctr;           // number of recentring shifts applied
    uint32_t tumble_ctr;          // number of tumbles
    float    phi_max;             // max |phi| over the stored frame
    // reserved[0] records whether fallback storage was used; reserved[1]
    // counts fallback cell-steps without a support margin. Both diagnostics
    // are cumulative within a run and are not checkpointed.
    uint32_t reserved[18];
};
static_assert(sizeof(CellState) == 192, "CellState must be exactly 192 B");
static_assert(alignof(CellState) == 64, "CellState must be 64 B aligned");

// Compact trajectory record written to mapped host-pinned memory on GH200.
struct TrajPackedCell {
    int32_t global_id;
    int32_t cls;
    float   cx, cy;          // global centre of mass (periodic-unwrapped rect)
    float   vx, vy;
    float   theta;
    float   volume;
    float   perim;
    float   gamma, v_A;
    float   phi_max;
};

// Immutable arguments for one update. Step counters remain device-resident
// because CUDA graph replays reuse captured argument values.
struct StepArgs {
    // Buffers.
    const float*    phi_in;
    float*          phi_out;
    const uint32_t* S_rd;          // read this step
    uint32_t*       S_sc;          // scatter into, for next step
    uint32_t*       S_cl;          // clear-ahead target
    CellState*      cell;
    uint8_t*        cell_cls;      // shape class per cell, hot 1 B copy
    const uint32_t* perm;          // Morton visit order (identity if disabled)
    unsigned long long* cursor_use;
    unsigned long long* cursor_clear;
    const unsigned long long* step_rd;
    unsigned long long* step_wr;
    uint32_t*       flags;

    // Geometry.
    int N;
    int L;                         // domain side
    int P;                         // S row pitch, uint32
    int parity_out;                // which phi buffer is being written (0/1)

    // PDE coefficients with mobility M = 1/2 included.
    float  dt;
    // Double precision avoids quantizing the small difference A0 - V.
    double A0;
    double vol_scale;              // 2*mu/A0        -> volC = vol_scale*(A0-V)
    float  bulk_scale;              // 30/lambda^2    -> dwC  = bulk_scale*gamma
    float  rep_coeff;               // 60*kappa/lambda^2
    float  mot_coeff;               // rep_coeff/xi

    // Random stream and output cadence.
    unsigned long long seed;
    // Independent stream used for initial polarity and all tumble events.
    unsigned long long polarity_seed;
    double p_tumble;               // -expm1(-dt/tau), computed in double, host
    int    full_moment_every;
    unsigned long long clear_ahead_words;   // number of S words to clear; 0 skips
};

// The counter primitive is shared with 3D. This solver defines its own counter
// domains below, so existing 2D trajectories retain their random streams.
using Philox4 = pf_common::Philox4;
using pf_common::philox4x32_10;
using pf_common::philox_uniform53;

// Disjoint counter domains for placement jitter, polarity, and motility
// disorder. Stable global-id keys reproduce a draw after reordering or resume.
constexpr uint32_t kIcDomainJitter   = 1u;
constexpr uint32_t kIcDomainPolarity = 2u;
constexpr uint32_t kIcDomainVA       = 3u;

// Host-side initial-condition draws; stream is SimParams::polarity_stream().
inline float ic_theta(int32_t gid, unsigned long long stream) {
    const Philox4 r = philox4x32_10((uint32_t)gid, kIcDomainPolarity, 0u, 0u,
                                    (uint32_t)(stream & 0xFFFFFFFFull),
                                    (uint32_t)(stream >> 32));
    return (float)(2.0 * kPi * philox_uniform53(r.v[0], r.v[1]));
}

// Lognormal motility with median v_A; z is standard normal from Box-Muller.
inline double ic_v_A(int32_t gid, unsigned long long seed,
                     double v_A, double sigma) {
    if (!(sigma > 0.0)) return v_A;
    const Philox4 r = philox4x32_10((uint32_t)gid, kIcDomainVA, 0u, 0u,
                                    (uint32_t)(seed & 0xFFFFFFFFull),
                                    (uint32_t)(seed >> 32));
    const double u1 = std::max(1e-300, philox_uniform53(r.v[0], r.v[1]));
    const double u2 = philox_uniform53(r.v[2], r.v[3]);
    const double z = std::sqrt(-2.0 * std::log(u1))
                   * std::cos(2.0 * kPi * u2);
    return v_A * std::exp(sigma * z);
}

// Kernel definitions are in kernels.cu. Dynamic shared memory limits k_step to
// one resident 768-thread block per SM.
__global__ __launch_bounds__(kBlockThreads, 1)
void k_step(PF_GRID_CONSTANT const StepArgs A);

__global__ __launch_bounds__(kBlockThreads, 1)
void k_step_fallback(PF_GRID_CONSTANT const StepArgs A);

__global__ void k_init_tiles(float* phi_a, float* phi_b, CellState* cell,
                             const uint8_t* cls, int N, int L,
                             const float* seed_cx, const float* seed_cy,
                             float radius_eff, float kappa_iface);

__global__ __launch_bounds__(kBlockThreads, 1)
void k_init_moments(CellState* cell, const uint8_t* cls,
                    const float* phi, int N);

__global__ void k_scatter_all(const float* phi, const CellState* cell,
                              const uint8_t* cls, uint32_t* S,
                              int N, int L, int P, uint32_t* flags);

__global__ void k_zero_u32(uint32_t* p, size_t n);

// Recompute volume and measure max|phi| outside each active window.
__global__ void k_verify_cells(const float* phi, const CellState* cell,
                               const uint8_t* cls, int N,
                               double* out_V, float* out_outside_max);
__global__ void k_verify_S(const uint32_t* S, size_t n, uint32_t* out_max);

__global__ void k_pack_traj(const CellState* cell, const uint8_t* cls,
                            TrajPackedCell* out, int N, int L);

// One-block bitonic sort by Morton-ordered centre of mass. M is a power of two
// at least N and requires M*8 bytes of dynamic shared memory.
__global__ void k_morton_sort(const CellState* cell, uint32_t* perm,
                              int N, int M, int L);

// Host-side launchers.
void configure_k_step_smem();                // cudaFuncSetAttribute opt-in
void configure_morton_smem(int smem_bytes);
int  k_step_grid(int device);                 // = numSMs

void launch_step(const StepArgs& A, int grid, cudaStream_t stream,
                 const void* l2_base, size_t l2_bytes, float l2_hit_ratio);


// Resource and occupancy values from the CUDA runtime. Configure the dynamic
// shared-memory opt-in before querying occupancy.
struct KernelStats {
    int    regs         = 0;
    // CUDA reports the ABI stack frame and any spills together here. The ptxas
    // build report is authoritative for spill loads and stores.
    size_t local_bytes  = 0;
    int    static_smem  = 0;
    int    dynamic_smem = 0;
    int    ctas_per_sm  = 0;      // measured, from the occupancy calculator
    int    reg_limited_ctas = 0;  // 65536 / (regs * threads), the reg ceiling
    int    warps_per_sm  = 0;
    double occupancy     = 0.0;   // resident threads / max resident threads
};

bool query_kernel_stats(const void* fn, int block_threads, int dynamic_smem,
                        int device, KernelStats* out);

}  // namespace pf
