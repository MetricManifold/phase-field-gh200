// CUDA kernels for the phase-field update. k_step handles shared-memory shape
// classes; k_step_fallback immediately follows it for cells using the fixed
// tile interior through global memory.

#include "../include/kernels.cuh"

#include <cuda_pipeline.h>

namespace pf {

__device__ __forceinline__ int wrapi(int v, int L) {
    v %= L;
    return v < 0 ? v + L : v;
}

// The aggregate field and each cell's self-contribution use the same Q5.27
// encoding, so the integer subtraction is exact and non-negative. Report
// invariant violations instead of clamping them.
__device__ __forceinline__ float s_other(uint32_t qS, float phi_self,
                                         uint32_t* flags) {
    const uint32_t qc = q_of(phi_self);
    if (qS >= qc) return (float)(qS - qc) * kQInvF;
    PF_FATAL_ADD(flags, FLAG_S_NEGATIVE);
    return 0.0f;
}

// __restrict__ is a promise that, for as long as the pointer is in scope, the
// output tile has no aliases. The non-staged path writes and rereads that tile,
// so only the staged specialization may make this promise.
template <bool STAGED> struct TileOutPtr;
template <> struct TileOutPtr<true>  { typedef float* __restrict__ type; };
template <> struct TileOutPtr<false> { typedef float* type; };

__device__ __forceinline__ uint32_t part1by1(uint32_t x) {
    x &= 0x0000FFFFu;
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}
__device__ __forceinline__ uint32_t morton2d(uint32_t x, uint32_t y) {
    return part1by1(x) | (part1by1(y) << 1);
}

// Advance one cell by one step. The source shape class is a template parameter,
// fixing the pixel map and reduction order. The destination class is runtime
// state because a cell may change class after its support is measured.
//
// Most shared-memory classes stage both phi and aggregate S. Class 4 stages
// only phi, reads S globally, and temporarily stores phi_next globally before
// the moment/scatter pass.
template <int CLS>
__device__ void process_cell(int n, const StepArgs& A, char* smem,
                             unsigned long long step)
{
    constexpr int WX  = kClasses[CLS].wx;
    constexpr int WY  = kClasses[CLS].wy;
    constexpr int TX0 = kClasses[CLS].tx0;
    constexpr int TY0 = kClasses[CLS].ty0;
    constexpr int PX  = phi_pitch(WX);
    constexpr int NS  = WY / kStripRows;
    constexpr int RB  = (WY + kWarpsPerBlock - 1) / kWarpsPerBlock;
    constexpr int kRing = 2 * (WX + 2) + 2 * WY;

    // The CTA's shared-memory request is kSmemBytes = max over classes, so this
    // instantiation must not need more than the class table says it does.
    static_assert(class_smem_of(CLS) <= kSmemRaw,
                  "this shape class needs more shared memory than the launch "
                  "requests");
    static_assert(RB * kWarpsPerBlock >= WY,
                  "thread-block row bands must cover the whole window");
    static_assert(WY % kStripRows == 0,
                  "asynchronous load strips must tile the window exactly");

    double*   red_s = reinterpret_cast<double*>(smem);
    int*      bci   = reinterpret_cast<int*>(smem + kRedBytes);
    float*    bcf   = reinterpret_cast<float*>(smem + kRedBytes);
    float*    phi_s = reinterpret_cast<float*>(smem + kScalarBytes);
    // A null pointer makes an accidental use in a non-staged specialization
    // fail instead of aliasing adjacent shared memory.
    [[maybe_unused]] uint32_t* const S_s =
        kStagesS<CLS> ? reinterpret_cast<uint32_t*>(smem + kScalarBytes
                                                    + phi_bytes(WX, WY))
                      : nullptr;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Initialize per-cell scalars and a one-pixel zero stencil ring. Pixels
    // outside the active window are exactly zero by the tile invariant.
    if (tid == 0) {
        // A reference avoids copying unused CellState members and increasing
        // register or local-memory pressure. State remains read-only here.
        const CellState& cs = A.cell[n];

        // Run durations follow P(t_r)=(1/tau) exp(-t_r/tau).
        // A.p_tumble is -expm1(-dt/tau) computed in double on the host.
        // Recomputing p_hat from the angle every step keeps |p_hat| == 1 by
        // construction: no renormalisation drift, no RNG state to checkpoint.
        const Philox4 r = philox4x32_10(
            (uint32_t)(step & 0xFFFFFFFFull), (uint32_t)(step >> 32),
            (uint32_t)cs.global_id, 0xA5A5A5A5u,
            (uint32_t)(A.polarity_seed & 0xFFFFFFFFull),
            (uint32_t)(A.polarity_seed >> 32));
        float theta = cs.theta;
        int tumbled = 0;
        if (philox_uniform53(r.v[0], r.v[1]) < A.p_tumble) {
            theta = (float)(2.0 * kPi * philox_uniform53(r.v[2], r.v[3]));
            tumbled = 1;
        }
        float ph_sin, ph_cos;
        sincosf(theta, &ph_sin, &ph_cos);

        const float gam  = cs.gamma;
        const float dwC  = A.bulk_scale * gam;
        const float volC = (float)(A.vol_scale * (A.A0 - cs.V));

        // Select the smallest class containing the support plus promotion
        // margin. Demotion requires kDemoteDwell consecutive checks with the
        // larger demotion margin. Growth beyond shared classes enters the
        // fallback; no path clips support to force a fit.
        const int ex = cs.bb_hi_x - cs.bb_lo_x + 1;
        const int ey = cs.bb_hi_y - cs.bb_lo_y + 1;
        int dcls = CLS;
        unsigned pctr = cs.promote_ctr;
        if (ex > 0) {                    // ex <= 0 <=> the support bbox is empty
            if (ex + kPromoteSlack > WX || ey + kPromoteSlack > WY) {
                const int grow = class_containing(ex, ey, kPromoteSlack);
                if (grow >= 0) dcls = grow;
                // No class retains the ordinary promotion margin.
                else PF_FATAL_ADD(A.flags, FLAG_CLASS_EXHAUSTED);
                pctr = 0u;
            } else {
                const int small = class_containing(ex, ey, kDemoteSlack);
                if (small >= 0 && small != CLS
                    && class_of(small).wx * class_of(small).wy < WX * WY) {
                    if (++pctr >= (unsigned)kDemoteDwell) { dcls = small; pctr = 0u; }
                } else {
                    pctr = 0u;
                }
            }
        }
        const ShapeClass dc = class_of(dcls);
        const int dwx = dc.wx,  dwy = dc.wy;
        const int dtx0 = dc.tx0, dty0 = dc.ty0;

        // Recentring changes the source index during the output pass, avoiding
        // another global-memory transfer. Class changes permit a larger shift.
        int sx = 0, sy = 0;
        if (cs.V > 0.0) {
            const double cxr = cs.Cx / cs.V;
            const double cyr = cs.Cy / cs.V;
            sx = __double2int_rn(cxr - 0.5 * (double)(dwx - 1));
            sy = __double2int_rn(cyr - 0.5 * (double)(dwy - 1));
            if (dcls == CLS) {
                sx = max(-kMaxShiftPerStep, min(kMaxShiftPerStep, sx));
                sy = max(-kMaxShiftPerStep, min(kMaxShiftPerStep, sy));
            } else {
                sx = max(-WX, min(WX, sx));
                sy = max(-WY, min(WY, sy));
            }
        } else {
            PF_FATAL_OR(A.flags, FLAG_V_NONPOS);
        }

        // Perimeter is measured from phi^{step+1}, so its cadence uses step+1.
        const int fm = (A.full_moment_every > 0 &&
                        ((step + 1ull) % (unsigned long long)A.full_moment_every) == 0ull);

        bci[0]  = sx;             bci[1]  = sy;
        bci[2]  = cs.gx0;         bci[3]  = cs.gy0;
        bci[4]  = wrapi(cs.gx0 + sx, A.L);
        bci[5]  = wrapi(cs.gy0 + sy, A.L);
        bci[6]  = dcls;
        bci[7]  = (dcls != (int)cs.cls_written[A.parity_out]);
        bcf[8]  = gam;            bcf[9]  = dwC;
        bcf[10] = volC;           bcf[11] = cs.v_A;
        bcf[12] = ph_cos;         bcf[13] = ph_sin;
        bcf[16] = theta;
        bci[17] = fm;
        bci[18] = (int)pctr;
        bci[19] = tumbled;
        bci[20] = dwx;            bci[21] = dwy;
        bci[22] = dtx0;           bci[23] = dty0;
    }

    for (int i = tid; i < kRing; i += kBlockThreads) {
        if (i < WX + 2) {
            phi_s[(kPhiPadLeft - 1) + i] = 0.0f;                        // row -1
        } else if (i < 2 * (WX + 2)) {
            const int k = i - (WX + 2);
            phi_s[(WY + 1) * PX + (kPhiPadLeft - 1) + k] = 0.0f;        // row WY
        } else if (i < 2 * (WX + 2) + WY) {
            const int k = i - 2 * (WX + 2);
            phi_s[(k + 1) * PX + (kPhiPadLeft - 1)] = 0.0f;             // col -1
        } else {
            const int k = i - 2 * (WX + 2) - WY;
            phi_s[(k + 1) * PX + (kPhiPadLeft + WX)] = 0.0f;            // col WX
        }
    }
    __syncthreads();

    const int gx0 = bci[2];
    const int gy0 = bci[3];
    // Resolve periodic x-wrap once per window into two contiguous segments;
    // the y-wrap is one add/select on the row index. No per-pixel modulo.
    const int split = min(WX, A.L - gx0);
    const float* __restrict__ tile_in = A.phi_in + (size_t)n * kTileArea;

    // Load phi and aggregate S in strips while accumulating interaction force.
    auto issue_strip = [&](int s) {
        constexpr int kChunksPerRow = WX / 4;                 // 16 B chunks
        constexpr int kPhiUnits     = kStripRows * kChunksPerRow;
        for (int u = tid; u < kPhiUnits; u += kBlockThreads) {
            const int cch = u % kChunksPerRow;
            const int y   = s * kStripRows + u / kChunksPerRow;
            __pipeline_memcpy_async(
                &phi_s[(y + 1) * PX + kPhiPadLeft + cch * 4],
                &tile_in[(size_t)(TY0 + y) * kTilePitch + TX0 + cch * 4],
                16);
        }
        if constexpr (kStagesS<CLS>) {
            // S rows are only 4-byte aligned, so use coalesced 4-byte copies.
            constexpr int kSUnits = kStripRows * WX;
            for (int u = tid; u < kSUnits; u += kBlockThreads) {
                const int i = u % WX;
                const int y = s * kStripRows + u / WX;
                int gy = gy0 + y;
                if (gy >= A.L) gy -= A.L;
                const int gx = (i < split) ? (gx0 + i) : (i - split);
                __pipeline_memcpy_async(&S_s[y * WX + i],
                                        &A.S_rd[(size_t)gy * A.P + gx], 4);
            }
        }
        // The non-staged class pipelines phi alone.
    };

    int committed = 0;
    for (int s = 0; s < kPipeStages && s < NS; ++s) {
        issue_strip(s);
        __pipeline_commit();
        ++committed;
    }

    double aIx = 0.0, aIy = 0.0;
    for (int s = 0; s < NS; ++s) {
        // Strip s's stencil needs row 16(s+1), i.e. strip s+1's first row.
        // With the prologue depth held at 3, exactly one group may remain
        // pending until the last two strips -- both literals, because
        // cp.async.wait_group takes an immediate.
        if (s <= NS - 3) __pipeline_wait_prior(1);
        else             __pipeline_wait_prior(0);
        __syncthreads();

        // A fixed single accumulator preserves reduction order and register use.
        const int ybase = s * kStripRows;
        for (int idx = tid; idx < kStripRows * WX; idx += kBlockThreads) {
            const int i = idx % WX;
            const int y = ybase + idx / WX;
            const float* p = &phi_s[(y + 1) * PX + kPhiPadLeft + i];
            const float c  = p[0];
            const float pE = p[1],  pW = p[-1];
            const float pN = p[PX], pS = p[-PX];
            // S is pointwise, so class 4 reads the same quantized word
            // directly from global memory using the staged path's coordinate map.
            uint32_t qS;
            if constexpr (kStagesS<CLS>) {
                qS = S_s[y * WX + i];
            } else {
                int ggy = gy0 + y;
                if (ggy >= A.L) ggy -= A.L;
                const int ggx = (i < split) ? (gx0 + i) : (i - split);
                qS = A.S_rd[(size_t)ggy * A.P + ggx];
            }
            const float So = s_other(qS, c, A.flags);
            const float gx = 0.5f * (pE - pW);
            const float gy = 0.5f * (pN - pS);
            aIx += (double)(c * gx * So);
            aIy += (double)(c * gy * So);
        }

        if (committed < NS) { issue_strip(committed); __pipeline_commit(); ++committed; }
    }

    // Reduce in a fixed order: warp butterfly followed by ascending warp index.
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aIx += __shfl_down_sync(0xFFFFFFFFu, aIx, d);
        aIy += __shfl_down_sync(0xFFFFFFFFu, aIy, d);
    }
    if (lane == 0) {
        red_s[warp * kRedSlots + 0] = aIx;
        red_s[warp * kRedSlots + 1] = aIy;
    }
    __syncthreads();
    if (tid == 0) {
        double sIx = 0.0, sIy = 0.0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sIx += red_s[w * kRedSlots + 0];
            sIy += red_s[w * kRedSlots + 1];
        }
        // v_n = v_A p_hat + motility * integral(phi grad(phi) S_other dA).
        // The interaction term is repulsive: for cell n left of cell m the
        // overlap sits on n's right, where grad_x(phi_n) < 0, so Ix < 0 and
        // n accelerates away from m.
        const double vxd = (double)bcf[11] * (double)bcf[12]
                         + (double)A.mot_coeff * sIx;
        const double vyd = (double)bcf[11] * (double)bcf[13]
                         + (double)A.mot_coeff * sIy;
        bcf[14] = (float)vxd;
        bcf[15] = (float)vyd;
        CellState* cs = &A.cell[n];
        cs->vx = (float)vxd;  cs->vy = (float)vyd;
        cs->Ix = sIx;         cs->Iy = sIy;
    }
    __syncthreads();

    // The non-staged path writes output here, so class-change zeroing must occur
    // first. The branch is block-uniform because bci[7] is broadcast.
    if constexpr (!kStagesS<CLS>) {
        if (bci[7]) {
            float* tz = A.phi_out + (size_t)n * kTileArea;
            for (int i = tid; i < kTileArea; i += kBlockThreads) tz[i] = 0.0f;
            __syncthreads();
        }
    }

    // Advance the PDE using a three-row rolling stencil. Staged classes reuse
    // S_s for phi_next; class 4 reads S and writes phi_next globally.
    {
        const float gam  = bcf[8];
        const float dwC  = bcf[9];
        const float volC = bcf[10];
        const float vxf  = bcf[14];
        const float vyf  = bcf[15];
        const float repC = A.rep_coeff;
        const float dtf  = A.dt;
        const int   y0   = warp * RB;

        // These values are used only by the non-staged specialization; its
        // output map must match the moment/scatter pass below.
        [[maybe_unused]] const int dsx   = kStagesS<CLS> ? 0 : bci[0];
        [[maybe_unused]] const int dsy   = kStagesS<CLS> ? 0 : bci[1];
        [[maybe_unused]] const int ddwx  = kStagesS<CLS> ? 0 : bci[20];
        [[maybe_unused]] const int ddwy  = kStagesS<CLS> ? 0 : bci[21];
        [[maybe_unused]] const int ddtx0 = kStagesS<CLS> ? 0 : bci[22];
        [[maybe_unused]] const int ddty0 = kStagesS<CLS> ? 0 : bci[23];
        [[maybe_unused]] float* const tout =
            kStagesS<CLS> ? nullptr : (A.phi_out + (size_t)n * kTileArea);

        if (y0 < WY) {
            const int y1 = min(y0 + RB, WY);
            for (int xb = 0; xb < WX; xb += 32) {
                const int x = lane + xb;
                if (x >= WX) break;
                const float* p = &phi_s[y0 * PX + kPhiPadLeft + x];  // row y0-1
                float sW = p[-1], sC = p[0], sE = p[1];
                p += PX;                                             // row y0
                float cW = p[-1], cC = p[0], cE = p[1];
                for (int y = y0; y < y1; ++y) {
                    p += PX;                                         // row y+1
                    const float nW = p[-1], nC = p[0], nE = p[1];
                    const float lap = ((float)kLapEdgeW * (nC + sC + cE + cW)
                                     + (float)kLapDiagW * (nE + nW + sE + sW)
                                     + (float)kLapCentreW * cC)
                                    * (float)(1.0 / kLapDenom);
                    const float gx = 0.5f * (cE - cW);
                    const float gy = 0.5f * (nC - sC);
                    uint32_t qS;
                    if constexpr (kStagesS<CLS>) {
                        qS = S_s[y * WX + x];
                    } else {
                        int ggy = gy0 + y;
                        if (ggy >= A.L) ggy -= A.L;
                        const int ggx = (x < split) ? (gx0 + x) : (x - split);
                        qS = A.S_rd[(size_t)ggy * A.P + ggx];
                    }
                    const float So = s_other(qS, cC, A.flags);
                    const float rhs = gam * lap
                                    - dwC * (cC * (1.0f - cC) * (1.0f - 2.0f * cC))
                                    + volC * cC
                                    - repC * cC * So
                                    - (vxf * gx + vyf * gy);
                    const float pnew = cC + dtf * rhs;
                    if constexpr (kStagesS<CLS>) {
                        S_s[y * WX + x] = __float_as_uint(pnew);
                    } else {
                        // Use the shifted output map and keep the value cacheable
                        // because the moment/scatter pass rereads it.
                        const int a = x - dsx;
                        const int b = y - dsy;
                        if ((unsigned)a < (unsigned)ddwx &&
                            (unsigned)b < (unsigned)ddwy)
                            tout[(size_t)(ddty0 + b) * kTilePitch + ddtx0 + a]
                                = pnew;
                    }
                    sW = cW; sC = cC; sE = cE;
                    cW = nW; cC = nC; cE = nE;
                }
            }
        }

        // Zero pixels not covered by the shifted source. This is disjoint from
        // the stores above, so every destination pixel is written once.
        if constexpr (!kStagesS<CLS>) {
            for (int b = warp; b < ddwy; b += kWarpsPerBlock) {
                const int j = b + dsy;
                const bool jin = ((unsigned)j < (unsigned)WY);
                for (int aa = lane; aa < ddwx; aa += 32) {
                    const int i = aa + dsx;
                    const bool iin = ((unsigned)i < (unsigned)WX);
                    if (!(jin && iin))
                        tout[(size_t)(ddty0 + b) * kTilePitch + ddtx0 + aa]
                            = 0.0f;
                }
            }
        }
    }
    // Make the non-staged global stores visible to this block's reads below.
    __syncthreads();

    // Store the shifted field, scatter S, and accumulate moments and support.
    const int sx = bci[0], sy = bci[1];
    const int gx0n = bci[4], gy0n = bci[5];
    const int dcls = bci[6];
    const int fm   = bci[17];
    const int dwx = bci[20], dwy = bci[21], dtx0 = bci[22], dty0 = bci[23];

    // Only the staged specialization can promise that tile_out is unaliased.
    typename TileOutPtr<kStagesS<CLS>>::type tile_out =
        A.phi_out + (size_t)n * kTileArea;
    if constexpr (kStagesS<CLS>) {
        if (bci[7]) {
            // A class change clears residue outside the new active window.
            for (int i = tid; i < kTileArea; i += kBlockThreads) tile_out[i] = 0.0f;
            __syncthreads();
        }
    }

    double aV = 0.0, aCx = 0.0, aCy = 0.0, aPer = 0.0;
    int blox = dwx, bhix = -1, bloy = dwy, bhiy = -1, pmaxb = 0;

    for (int b = warp; b < dwy; b += kWarpsPerBlock) {
        // Map each destination pixel to its shifted source when staged.
        [[maybe_unused]] const int j = b + sy;
        [[maybe_unused]] const bool jin = ((unsigned)j < (unsigned)WY);
        int gy = gy0n + b;
        if (gy >= A.L) gy -= A.L;
        for (int a = lane; a < dwx; a += 32) {
            [[maybe_unused]] const int i = a + sx;
            [[maybe_unused]] const bool iin = ((unsigned)i < (unsigned)WX);
            // Staged classes store phi_next here; class 4 rereads the
            // complete destination written above.
            float pn;
            if constexpr (kStagesS<CLS>) {
                pn = (jin && iin) ? __uint_as_float(S_s[j * WX + i]) : 0.0f;
                __stcs(&tile_out[(size_t)(dty0 + b) * kTilePitch + dtx0 + a], pn);
            } else {
                pn = tile_out[(size_t)(dty0 + b) * kTilePitch + dtx0 + a];
            }

            if (!isfinite(pn)) PF_FATAL_OR(A.flags, FLAG_NONFINITE);
            if (pn * pn > kQClampPhiSq) PF_FATAL_OR(A.flags, FLAG_Q_CLAMP);
            const int pb = (int)__float_as_uint(fabsf(pn));
            pmaxb = max(pmaxb, pb);

            const uint32_t q = q_of(pn);
            if (q) {                       // adding 0 is a no-op: bit-exact skip
                int gx = gx0n + a;
                if (gx >= A.L) gx -= A.L;
                const uint32_t old = atomicAdd(&A.S_sc[(size_t)gy * A.P + gx], q);
                if (old > 0xFFFFFFFFu - q) PF_FATAL_OR(A.flags, FLAG_S_OVERFLOW);
            }

            // V(phi^{n+1}) is accumulated over the frame actually stored, so
            // it is exactly consistent with the field the next step reads.
            const double d = (double)pn * (double)pn;
            aV  += d;
            aCx += d * (double)a;
            aCy += d * (double)b;

            if (pn > kSupportEps) {
                blox = min(blox, a); bhix = max(bhix, a);
                bloy = min(bloy, b); bhiy = max(bhiy, b);
            }

            if (fm) {
                float e = 0.0f, w = 0.0f, nn = 0.0f, ss = 0.0f;
                if constexpr (kStagesS<CLS>) {
                    if (jin) {
                        if ((unsigned)(i + 1) < (unsigned)WX) e = __uint_as_float(S_s[j * WX + i + 1]);
                        if ((unsigned)(i - 1) < (unsigned)WX) w = __uint_as_float(S_s[j * WX + i - 1]);
                    }
                    if (iin) {
                        if ((unsigned)(j + 1) < (unsigned)WY) nn = __uint_as_float(S_s[(j + 1) * WX + i]);
                        if ((unsigned)(j - 1) < (unsigned)WY) ss = __uint_as_float(S_s[(j - 1) * WX + i]);
                    }
                } else {
                    // Perimeter uses gradients of the stored destination with
                    // zeros outside its window. For class 4 this can
                    // differ at the sub-threshold border; phi, moments, support,
                    // and scatter are unaffected.
                    const size_t row = (size_t)(dty0 + b) * kTilePitch + dtx0;
                    if (a + 1 < dwx) e = tile_out[row + a + 1];
                    if (a - 1 >= 0)  w = tile_out[row + a - 1];
                    if (b + 1 < dwy)
                        nn = tile_out[(size_t)(dty0 + b + 1) * kTilePitch + dtx0 + a];
                    if (b - 1 >= 0)
                        ss = tile_out[(size_t)(dty0 + b - 1) * kTilePitch + dtx0 + a];
                }
                const float pgx = 0.5f * (e - w), pgy = 0.5f * (nn - ss);
                aPer += (double)sqrtf(pgx * pgx + pgy * pgy);
            }
        }
    }

    // Use the fixed fp64 reduction order; integer bounds are order-independent.
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aV   += __shfl_down_sync(0xFFFFFFFFu, aV,   d);
        aCx  += __shfl_down_sync(0xFFFFFFFFu, aCx,  d);
        aCy  += __shfl_down_sync(0xFFFFFFFFu, aCy,  d);
        aPer += __shfl_down_sync(0xFFFFFFFFu, aPer, d);
        blox  = min(blox, __shfl_down_sync(0xFFFFFFFFu, blox, d));
        bhix  = max(bhix, __shfl_down_sync(0xFFFFFFFFu, bhix, d));
        bloy  = min(bloy, __shfl_down_sync(0xFFFFFFFFu, bloy, d));
        bhiy  = max(bhiy, __shfl_down_sync(0xFFFFFFFFu, bhiy, d));
        pmaxb = max(pmaxb, __shfl_down_sync(0xFFFFFFFFu, pmaxb, d));
    }
    if (lane == 0) {
        double* rw = red_s + warp * kRedSlots;
        rw[0] = aV; rw[1] = aCx; rw[2] = aCy; rw[3] = aPer;
        int* iw = reinterpret_cast<int*>(rw + 4);
        iw[0] = blox; iw[1] = bhix; iw[2] = bloy; iw[3] = bhiy; iw[4] = pmaxb;
    }
    __syncthreads();
    if (tid == 0) {
        double sV = 0.0, sCx = 0.0, sCy = 0.0, sPer = 0.0;
        int Blox = dwx, Bhix = -1, Bloy = dwy, Bhiy = -1, Pmax = 0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            const double* rw = red_s + w * kRedSlots;
            sV += rw[0]; sCx += rw[1]; sCy += rw[2]; sPer += rw[3];
            const int* iw = reinterpret_cast<const int*>(rw + 4);
            Blox = min(Blox, iw[0]); Bhix = max(Bhix, iw[1]);
            Bloy = min(Bloy, iw[2]); Bhiy = max(Bhiy, iw[3]);
            Pmax = max(Pmax, iw[4]);
        }
        CellState* cs = &A.cell[n];
        cs->gx0 = gx0n;  cs->gy0 = gy0n;
        cs->cls = (uint8_t)dcls;
        cs->cls_written[A.parity_out] = (uint8_t)dcls;
        cs->theta = bcf[16];
        cs->V = sV;  cs->Cx = sCx;  cs->Cy = sCy;
        if (fm) cs->perim = sPer;
        cs->bb_lo_x = Blox;  cs->bb_hi_x = Bhix;
        cs->bb_lo_y = Bloy;  cs->bb_hi_y = Bhiy;
        cs->promote_ctr = (uint32_t)bci[18];
        cs->shift_ctr  += (sx | sy) ? 1u : 0u;
        cs->tumble_ctr += (uint32_t)bci[19];
        cs->phi_max = __uint_as_float((uint32_t)Pmax);
        A.cell_cls[n] = (uint8_t)dcls;
        if (Bhix >= 0 && (Blox == 0 || Bhix == dwx - 1 ||
                          Bloy == 0 || Bhiy == dwy - 1))
            PF_ADVISORY_ADD(A.flags, FLAG_SUPPORT_CLIP);
    }
    __syncthreads();   // smem is reused by the next cell in this CTA
}

// Rare whole-tile fallback. The active 286x286 window reads phi and S from
// global memory; tile rows and columns 0 and 287 remain the stencil zero ring.
__device__ __noinline__ void process_cell_fallback(
    int n, const StepArgs& A, char* smem, unsigned long long step)
{
    constexpr int CLS = kClassFallback;
    constexpr int WX  = kClasses[CLS].wx;
    constexpr int WY  = kClasses[CLS].wy;
    constexpr int TX0 = kClasses[CLS].tx0;
    constexpr int TY0 = kClasses[CLS].ty0;
    constexpr int RB  = (WY + kWarpsPerBlock - 1) / kWarpsPerBlock;
    static_assert(WX == kTilePitch - 2 && WY == kTilePitch - 2);
    static_assert(TX0 == 1 && TY0 == 1);

    double* red_s = reinterpret_cast<double*>(smem);
    int*    bci   = reinterpret_cast<int*>(smem + kRedBytes);
    float*  bcf   = reinterpret_cast<float*>(smem + kRedBytes);
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid == 0) {
        const CellState& cs = A.cell[n];
        const Philox4 r = philox4x32_10(
            (uint32_t)(step & 0xFFFFFFFFull), (uint32_t)(step >> 32),
            (uint32_t)cs.global_id, 0xA5A5A5A5u,
            (uint32_t)(A.polarity_seed & 0xFFFFFFFFull),
            (uint32_t)(A.polarity_seed >> 32));
        float theta = cs.theta;
        int tumbled = 0;
        if (philox_uniform53(r.v[0], r.v[1]) < A.p_tumble) {
            theta = (float)(2.0 * kPi * philox_uniform53(r.v[2], r.v[3]));
            tumbled = 1;
        }
        float ph_sin, ph_cos;
        sincosf(theta, &ph_sin, &ph_cos);

        const float gam  = cs.gamma;
        const float dwC  = A.bulk_scale * gam;
        const float volC = (float)(A.vol_scale * (A.A0 - cs.V));

        const int ex = cs.bb_hi_x - cs.bb_lo_x + 1;
        const int ey = cs.bb_hi_y - cs.bb_lo_y + 1;
        int dcls = CLS;
        int no_margin = 0;
        unsigned pctr = cs.promote_ctr;
        if (ex > 0) {
            if (ex + kPromoteSlack > WX || ey + kPromoteSlack > WY) {
                const int grow = class_containing(ex, ey, kPromoteSlack);
                if (grow >= 0) {
                    dcls = grow;
                } else if (ex <= WX && ey <= WY) {
                    // Keep a representable field when it consumes the ordinary
                    // promotion margin, and retain that fact for host warning.
                    no_margin = 1;
                } else {
                    PF_FATAL_ADD(A.flags, FLAG_CLASS_EXHAUSTED);
                }
                pctr = 0u;
            } else {
                const int small = class_containing(ex, ey, kDemoteSlack);
                if (small >= 0 && small != CLS
                    && class_of(small).wx * class_of(small).wy < WX * WY) {
                    if (++pctr >= (unsigned)kDemoteDwell) {
                        dcls = small;
                        pctr = 0u;
                    }
                } else {
                    pctr = 0u;
                }
            }
        }
        const ShapeClass dc = class_of(dcls);
        const int dwx = dc.wx, dwy = dc.wy;

        int sx = 0, sy = 0;
        if (cs.V > 0.0) {
            sx = __double2int_rn(cs.Cx / cs.V - 0.5 * (double)(dwx - 1));
            sy = __double2int_rn(cs.Cy / cs.V - 0.5 * (double)(dwy - 1));
            if (dcls == CLS) {
                sx = max(-kMaxShiftPerStep, min(kMaxShiftPerStep, sx));
                sy = max(-kMaxShiftPerStep, min(kMaxShiftPerStep, sy));
            } else {
                sx = max(-WX, min(WX, sx));
                sy = max(-WY, min(WY, sy));
            }
        } else {
            PF_FATAL_OR(A.flags, FLAG_V_NONPOS);
        }

        const int fm = (A.full_moment_every > 0 &&
                        ((step + 1ull) %
                         (unsigned long long)A.full_moment_every) == 0ull);
        bci[0] = sx; bci[1] = sy;
        bci[2] = cs.gx0; bci[3] = cs.gy0;
        bci[4] = wrapi(cs.gx0 + sx, A.L);
        bci[5] = wrapi(cs.gy0 + sy, A.L);
        bci[6] = dcls;
        bci[7] = (dcls != (int)cs.cls_written[A.parity_out]);
        bcf[8] = gam; bcf[9] = dwC; bcf[10] = volC; bcf[11] = cs.v_A;
        bcf[12] = ph_cos; bcf[13] = ph_sin; bcf[16] = theta;
        bci[17] = fm; bci[18] = (int)pctr; bci[19] = tumbled;
        bci[20] = dwx; bci[21] = dwy; bci[22] = dc.tx0; bci[23] = dc.ty0;
        bci[24] = no_margin;
    }
    __syncthreads();

    const int gx0 = bci[2], gy0 = bci[3];
    const int split = min(WX, A.L - gx0);
    const float* tile_in = A.phi_in + (size_t)n * kTileArea;

    double aIx = 0.0, aIy = 0.0;
    for (int y = warp; y < WY; y += kWarpsPerBlock) {
        int gy = gy0 + y;
        if (gy >= A.L) gy -= A.L;
        for (int x = lane; x < WX; x += 32) {
            const float* p = &tile_in[(size_t)(TY0 + y) * kTilePitch + TX0 + x];
            const float c = p[0];
            const float gx = 0.5f * (p[1] - p[-1]);
            const float gy_phi = 0.5f * (p[kTilePitch] - p[-kTilePitch]);
            const int gx_global = (x < split) ? gx0 + x : x - split;
            const float So = s_other(A.S_rd[(size_t)gy * A.P + gx_global],
                                     c, A.flags);
            aIx += (double)(c * gx * So);
            aIy += (double)(c * gy_phi * So);
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aIx += __shfl_down_sync(0xFFFFFFFFu, aIx, d);
        aIy += __shfl_down_sync(0xFFFFFFFFu, aIy, d);
    }
    if (lane == 0) {
        red_s[warp * kRedSlots] = aIx;
        red_s[warp * kRedSlots + 1] = aIy;
    }
    __syncthreads();
    if (tid == 0) {
        double sIx = 0.0, sIy = 0.0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sIx += red_s[w * kRedSlots];
            sIy += red_s[w * kRedSlots + 1];
        }
        const double vxd = (double)bcf[11] * (double)bcf[12]
                         + (double)A.mot_coeff * sIx;
        const double vyd = (double)bcf[11] * (double)bcf[13]
                         + (double)A.mot_coeff * sIy;
        bcf[14] = (float)vxd; bcf[15] = (float)vyd;
        CellState* cs = &A.cell[n];
        cs->vx = (float)vxd; cs->vy = (float)vyd;
        cs->Ix = sIx; cs->Iy = sIy;
    }
    __syncthreads();

    float* tile_out = A.phi_out + (size_t)n * kTileArea;
    if (bci[7]) {
        for (int i = tid; i < kTileArea; i += kBlockThreads)
            tile_out[i] = 0.0f;
        __syncthreads();
    }

    const float gam = bcf[8], dwC = bcf[9], volC = bcf[10];
    const float vxf = bcf[14], vyf = bcf[15];
    const int sx = bci[0], sy = bci[1];
    const int dwx = bci[20], dwy = bci[21];
    const int dtx0 = bci[22], dty0 = bci[23];
    const int y0 = warp * RB;
    if (y0 < WY) {
        const int y1 = min(y0 + RB, WY);
        for (int xb = 0; xb < WX; xb += 32) {
            const int x = lane + xb;
            if (x >= WX) break;
            const float* p = &tile_in[(size_t)(TY0 + y0 - 1) * kTilePitch
                                      + TX0 + x];
            float sW = p[-1], sC = p[0], sE = p[1];
            p += kTilePitch;
            float cW = p[-1], cC = p[0], cE = p[1];
            const int gx_global = (x < split) ? gx0 + x : x - split;
            for (int y = y0; y < y1; ++y) {
                p += kTilePitch;
                const float nW = p[-1], nC = p[0], nE = p[1];
                const float lap = ((float)kLapEdgeW * (nC + sC + cE + cW)
                                 + (float)kLapDiagW * (nE + nW + sE + sW)
                                 + (float)kLapCentreW * cC)
                                * (float)(1.0 / kLapDenom);
                const float gx = 0.5f * (cE - cW);
                const float gy_phi = 0.5f * (nC - sC);
                int gy = gy0 + y;
                if (gy >= A.L) gy -= A.L;
                const float So = s_other(
                    A.S_rd[(size_t)gy * A.P + gx_global], cC, A.flags);
                const float rhs = gam * lap
                                - dwC * (cC * (1.0f - cC) * (1.0f - 2.0f * cC))
                                + volC * cC
                                - A.rep_coeff * cC * So
                                - (vxf * gx + vyf * gy_phi);
                const float pnew = cC + A.dt * rhs;
                const int a = x - sx, b = y - sy;
                if ((unsigned)a < (unsigned)dwx &&
                    (unsigned)b < (unsigned)dwy)
                    tile_out[(size_t)(dty0 + b) * kTilePitch + dtx0 + a] = pnew;
                sW = cW; sC = cC; sE = cE;
                cW = nW; cC = nC; cE = nE;
            }
        }
    }
    for (int b = warp; b < dwy; b += kWarpsPerBlock) {
        const int j = b + sy;
        const bool jin = (unsigned)j < (unsigned)WY;
        for (int a = lane; a < dwx; a += 32) {
            const int i = a + sx;
            if (!(jin && (unsigned)i < (unsigned)WX))
                tile_out[(size_t)(dty0 + b) * kTilePitch + dtx0 + a] = 0.0f;
        }
    }
    __syncthreads();

    const int gx0n = bci[4], gy0n = bci[5];
    const int dcls = bci[6], fm = bci[17];
    double aV = 0.0, aCx = 0.0, aCy = 0.0, aPer = 0.0;
    int blox = dwx, bhix = -1, bloy = dwy, bhiy = -1, pmaxb = 0;
    for (int b = warp; b < dwy; b += kWarpsPerBlock) {
        int gy = gy0n + b;
        if (gy >= A.L) gy -= A.L;
        const size_t row = (size_t)(dty0 + b) * kTilePitch + dtx0;
        for (int a = lane; a < dwx; a += 32) {
            const float pn = tile_out[row + a];
            if (!isfinite(pn)) PF_FATAL_OR(A.flags, FLAG_NONFINITE);
            if (pn * pn > kQClampPhiSq) PF_FATAL_OR(A.flags, FLAG_Q_CLAMP);
            pmaxb = max(pmaxb, (int)__float_as_uint(fabsf(pn)));
            const uint32_t q = q_of(pn);
            if (q) {
                int gx = gx0n + a;
                if (gx >= A.L) gx -= A.L;
                const uint32_t old = atomicAdd(&A.S_sc[(size_t)gy * A.P + gx], q);
                if (old > 0xFFFFFFFFu - q)
                    PF_FATAL_OR(A.flags, FLAG_S_OVERFLOW);
            }
            const double d = (double)pn * (double)pn;
            aV += d; aCx += d * (double)a; aCy += d * (double)b;
            if (pn > kSupportEps) {
                blox = min(blox, a); bhix = max(bhix, a);
                bloy = min(bloy, b); bhiy = max(bhiy, b);
            }
            if (fm) {
                const float e = a + 1 < dwx ? tile_out[row + a + 1] : 0.0f;
                const float w = a > 0 ? tile_out[row + a - 1] : 0.0f;
                const float nn = b + 1 < dwy
                    ? tile_out[row + kTilePitch + a] : 0.0f;
                const float ss = b > 0
                    ? tile_out[row - kTilePitch + a] : 0.0f;
                const float pgx = 0.5f * (e - w), pgy = 0.5f * (nn - ss);
                aPer += (double)sqrtf(pgx * pgx + pgy * pgy);
            }
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aV += __shfl_down_sync(0xFFFFFFFFu, aV, d);
        aCx += __shfl_down_sync(0xFFFFFFFFu, aCx, d);
        aCy += __shfl_down_sync(0xFFFFFFFFu, aCy, d);
        aPer += __shfl_down_sync(0xFFFFFFFFu, aPer, d);
        blox = min(blox, __shfl_down_sync(0xFFFFFFFFu, blox, d));
        bhix = max(bhix, __shfl_down_sync(0xFFFFFFFFu, bhix, d));
        bloy = min(bloy, __shfl_down_sync(0xFFFFFFFFu, bloy, d));
        bhiy = max(bhiy, __shfl_down_sync(0xFFFFFFFFu, bhiy, d));
        pmaxb = max(pmaxb, __shfl_down_sync(0xFFFFFFFFu, pmaxb, d));
    }
    if (lane == 0) {
        double* rw = red_s + warp * kRedSlots;
        rw[0] = aV; rw[1] = aCx; rw[2] = aCy; rw[3] = aPer;
        int* iw = reinterpret_cast<int*>(rw + 4);
        iw[0] = blox; iw[1] = bhix; iw[2] = bloy; iw[3] = bhiy;
        iw[4] = pmaxb;
    }
    __syncthreads();
    if (tid == 0) {
        double sV = 0.0, sCx = 0.0, sCy = 0.0, sPer = 0.0;
        int Blox = dwx, Bhix = -1, Bloy = dwy, Bhiy = -1, Pmax = 0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            const double* rw = red_s + w * kRedSlots;
            sV += rw[0]; sCx += rw[1]; sCy += rw[2]; sPer += rw[3];
            const int* iw = reinterpret_cast<const int*>(rw + 4);
            Blox = min(Blox, iw[0]); Bhix = max(Bhix, iw[1]);
            Bloy = min(Bloy, iw[2]); Bhiy = max(Bhiy, iw[3]);
            Pmax = max(Pmax, iw[4]);
        }
        CellState* cs = &A.cell[n];
        cs->gx0 = gx0n; cs->gy0 = gy0n;
        cs->cls = (uint8_t)dcls;
        cs->cls_written[A.parity_out] = (uint8_t)dcls;
        cs->theta = bcf[16]; cs->V = sV; cs->Cx = sCx; cs->Cy = sCy;
        if (fm) cs->perim = sPer;
        cs->bb_lo_x = Blox; cs->bb_hi_x = Bhix;
        cs->bb_lo_y = Bloy; cs->bb_hi_y = Bhiy;
        cs->promote_ctr = (uint32_t)bci[18];
        cs->shift_ctr += (sx | sy) ? 1u : 0u;
        cs->tumble_ctr += (uint32_t)bci[19];
        cs->phi_max = __uint_as_float((uint32_t)Pmax);
        cs->reserved[0] = 1u;
        A.cell_cls[n] = (uint8_t)dcls;
        const bool edge = Bhix >= 0 &&
            (Blox == 0 || Bhix == dwx - 1 ||
             Bloy == 0 || Bhiy == dwy - 1);
        if (bci[24] || (dcls == kClassFallback && edge))
            ++cs->reserved[1];
        if (edge)
            PF_ADVISORY_ADD(A.flags, FLAG_SUPPORT_CLIP);
    }
    __syncthreads();
}

__global__ __launch_bounds__(kBlockThreads, 1)
void k_step(PF_GRID_CONSTANT const StepArgs A)
{
    // uint4 provides the 16-byte alignment required by cp.async destinations.
    extern __shared__ uint4 smem_raw[];
    char* smem = reinterpret_cast<char*>(smem_raw);
    int* ctrl = reinterpret_cast<int*>(smem + kRedBytes);

    const int tid = (int)threadIdx.x;

    // Clear S[(step+2)%3] before cell work so stores overlap load latency. The
    // kernel does not otherwise access this slot; streaming stores avoid
    // evicting the persisting read buffer.
    if (A.clear_ahead_words > 0ull) {
        uint4* dst = reinterpret_cast<uint4*>(A.S_cl);
        const size_t nvec   = (size_t)(A.clear_ahead_words >> 2);
        const size_t stride = (size_t)gridDim.x * (size_t)kBlockThreads;
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        for (size_t i = (size_t)blockIdx.x * kBlockThreads + tid; i < nvec; i += stride)
            __stcs(&dst[i], z);
    }

    // Cursor and step-counter slots alternate with phi parity. The kernel
    // boundary orders their reset and publication for the next launch.
    if (blockIdx.x == 0 && tid == 0) {
        *A.cursor_clear = 0ull;
        *A.step_wr = *A.step_rd + 1ull;
    }
    const unsigned long long step = *A.step_rd;

    for (;;) {
        if (tid == 0) {
            const unsigned long long k = atomicAdd(A.cursor_use, 1ull);
            if (k < (unsigned long long)A.N) {
                const int nn = (int)A.perm[(size_t)k];
                ctrl[kBcastCtrlN]   = nn;
                ctrl[kBcastCtrlCls] = (int)A.cell_cls[nn];
            } else {
                ctrl[kBcastCtrlN]   = -1;
                ctrl[kBcastCtrlCls] = -1;
            }
        }
        __syncthreads();
        const int cls = ctrl[kBcastCtrlCls];
        if (cls < 0) break;
        const int n = ctrl[kBcastCtrlN];
        // Explicit dispatch prevents invalid classes from selecting mismatched
        // window geometry.
        static_assert(kNumClasses == 6,
                      "k_step's dispatch switch must enumerate every class");
        switch (cls) {
            case kClassRound: process_cell<kClassRound>(n, A, smem, step); break;
            case kClassWide:  process_cell<kClassWide >(n, A, smem, step); break;
            case kClassTall:  process_cell<kClassTall >(n, A, smem, step); break;
            case kClassBig:   process_cell<kClassBig  >(n, A, smem, step); break;
            case kClassLarge: process_cell<kClassLarge>(n, A, smem, step); break;
            // The ordered fallback kernel advances cells whose input parity
            // was written in the global-memory class.
            case kClassFallback: __syncthreads(); break;
            default:
                if (tid == 0) atomicAdd(&A.flags[FLAG_CLASS_UNSUPPORTED], 1u);
                // Match process_cell's final barrier before reusing shared state.
                __syncthreads();
                break;
        }
    }
}

// cls_written for the input parity is a stable source-class snapshot. A cell
// promoted by k_step therefore cannot be advanced twice in the same step.
__global__ __launch_bounds__(kBlockThreads, 1)
void k_step_fallback(PF_GRID_CONSTANT const StepArgs A)
{
    extern __shared__ uint4 smem_raw[];
    char* smem = reinterpret_cast<char*>(smem_raw);
    const int input_parity = A.parity_out ^ 1;
    const unsigned long long step = *A.step_rd;
    for (int n = (int)blockIdx.x; n < A.N; n += (int)gridDim.x) {
        const bool active =
            A.cell[n].cls_written[input_parity] == (uint8_t)kClassFallback;
        if (active)
            process_cell_fallback(n, A, smem, step);
        else
            __syncthreads();
    }
}

__global__ void k_zero_u32(uint32_t* p, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = 0u;
}

__global__ void k_init_tiles(float* phi_a, float* phi_b, CellState* cell,
                             const uint8_t* clsv, int N, int L,
                             const float* seed_cx, const float* seed_cy,
                             float radius_eff, float kappa_iface)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;

    float* ta = phi_a + (size_t)n * kTileArea;
    float* tb = phi_b + (size_t)n * kTileArea;
    for (int i = (int)threadIdx.x; i < kTileArea; i += (int)blockDim.x) {
        ta[i] = 0.0f;
        tb[i] = 0.0f;
    }
    __syncthreads();

    const float cx = seed_cx[n], cy = seed_cy[n];
    const int gx0 = wrapi((int)lrintf(cx - 0.5f * (float)(wx - 1)), L);
    const int gy0 = wrapi((int)lrintf(cy - 0.5f * (float)(wy - 1)), L);

    for (int p = (int)threadIdx.x; p < wx * wy; p += (int)blockDim.x) {
        const int a = p % wx, b = p / wx;
        float dx = (float)(gx0 + a) - cx;
        float dy = (float)(gy0 + b) - cy;
        if (dx >  0.5f * L) dx -= (float)L;
        if (dx < -0.5f * L) dx += (float)L;
        if (dy >  0.5f * L) dy -= (float)L;
        if (dy < -0.5f * L) dy += (float)L;
        const float r = sqrtf(dx * dx + dy * dy);
        ta[(size_t)(ty0 + b) * kTilePitch + tx0 + a] =
            0.5f * (1.0f - tanhf(kappa_iface * (r - radius_eff)));
    }

    if (threadIdx.x == 0) {
        CellState* cs = &cell[n];
        cs->gx0 = gx0;  cs->gy0 = gy0;
        cs->cls = (uint8_t)cls;
        cs->cls_written[0] = (uint8_t)cls;
        cs->cls_written[1] = (uint8_t)cls;
    }
}

// Initialize moments and support before the first update using the same
// reduction order as the update kernel.
__global__ __launch_bounds__(kBlockThreads, 1)
void k_init_moments(CellState* cell, const uint8_t* clsv,
                    const float* phi, int N)
{
    __shared__ double red_s[kWarpsPerBlock * kRedSlots];
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;
    const float* t = phi + (size_t)n * kTileArea;

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;

    double aV = 0.0, aCx = 0.0, aCy = 0.0;
    int blox = wx, bhix = -1, bloy = wy, bhiy = -1, pmaxb = 0;
    for (int b = warp; b < wy; b += kWarpsPerBlock) {
        for (int a = lane; a < wx; a += 32) {
            const float pn = t[(size_t)(ty0 + b) * kTilePitch + tx0 + a];
            const double d = (double)pn * (double)pn;
            aV += d; aCx += d * (double)a; aCy += d * (double)b;
            pmaxb = max(pmaxb, (int)__float_as_uint(fabsf(pn)));
            if (pn > kSupportEps) {
                blox = min(blox, a); bhix = max(bhix, a);
                bloy = min(bloy, b); bhiy = max(bhiy, b);
            }
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        aV  += __shfl_down_sync(0xFFFFFFFFu, aV,  d);
        aCx += __shfl_down_sync(0xFFFFFFFFu, aCx, d);
        aCy += __shfl_down_sync(0xFFFFFFFFu, aCy, d);
        blox = min(blox, __shfl_down_sync(0xFFFFFFFFu, blox, d));
        bhix = max(bhix, __shfl_down_sync(0xFFFFFFFFu, bhix, d));
        bloy = min(bloy, __shfl_down_sync(0xFFFFFFFFu, bloy, d));
        bhiy = max(bhiy, __shfl_down_sync(0xFFFFFFFFu, bhiy, d));
        pmaxb = max(pmaxb, __shfl_down_sync(0xFFFFFFFFu, pmaxb, d));
    }
    if (lane == 0) {
        double* rw = red_s + warp * kRedSlots;
        rw[0] = aV; rw[1] = aCx; rw[2] = aCy;
        int* iw = reinterpret_cast<int*>(rw + 4);
        iw[0] = blox; iw[1] = bhix; iw[2] = bloy; iw[3] = bhiy; iw[4] = pmaxb;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double sV = 0.0, sCx = 0.0, sCy = 0.0;
        int Blox = wx, Bhix = -1, Bloy = wy, Bhiy = -1, Pmax = 0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            const double* rw = red_s + w * kRedSlots;
            sV += rw[0]; sCx += rw[1]; sCy += rw[2];
            const int* iw = reinterpret_cast<const int*>(rw + 4);
            Blox = min(Blox, iw[0]); Bhix = max(Bhix, iw[1]);
            Bloy = min(Bloy, iw[2]); Bhiy = max(Bhiy, iw[3]);
            Pmax = max(Pmax, iw[4]);
        }
        CellState* cs = &cell[n];
        cs->V = sV; cs->Cx = sCx; cs->Cy = sCy;
        cs->bb_lo_x = Blox; cs->bb_hi_x = Bhix;
        cs->bb_lo_y = Bloy; cs->bb_hi_y = Bhiy;
        cs->phi_max = __uint_as_float((uint32_t)Pmax);
        cs->promote_ctr = 0u;
        cs->Ix = 0.0; cs->Iy = 0.0;
    }
}

__global__ void k_scatter_all(const float* phi, const CellState* cell,
                              const uint8_t* clsv, uint32_t* S,
                              int N, int L, int P, uint32_t* flags)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;
    const CellState cs = cell[n];
    const float* t = phi + (size_t)n * kTileArea;
    for (int p = (int)threadIdx.x; p < wx * wy; p += (int)blockDim.x) {
        const int a = p % wx, b = p / wx;
        const float pn = t[(size_t)(ty0 + b) * kTilePitch + tx0 + a];
        const uint32_t q = q_of(pn);
        if (!q) continue;
        const int gx = wrapi(cs.gx0 + a, L);
        const int gy = wrapi(cs.gy0 + b, L);
        const uint32_t old = atomicAdd(&S[(size_t)gy * P + gx], q);
        if (old > 0xFFFFFFFFu - q) atomicOr(&flags[FLAG_S_OVERFLOW], 1u);
        if (pn * pn > kQClampPhiSq) atomicOr(&flags[FLAG_Q_CLAMP], 1u);
    }
}

// Diagnostic kernels.
__global__ void k_verify_cells(const float* phi, const CellState* cell,
                               const uint8_t* clsv, int N,
                               double* out_V, float* out_outside_max)
{
    __shared__ double vred[kWarpsPerBlock];
    __shared__ float  ored[kWarpsPerBlock];
    const int n = blockIdx.x;
    if (n >= N) return;
    const int cls = (int)clsv[n];
    const ShapeClass sc = class_of(cls);
    const int wx = sc.wx,  wy = sc.wy;
    const int tx0 = sc.tx0, ty0 = sc.ty0;
    const float* t = phi + (size_t)n * kTileArea;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int nwarp = (int)blockDim.x >> 5;

    double v = 0.0;
    float  omax = 0.0f;
    for (int i = (int)threadIdx.x; i < kTileArea; i += (int)blockDim.x) {
        const int x = i % kTilePitch, y = i / kTilePitch;
        const float p = t[i];
        const bool inside = (x >= tx0 && x < tx0 + wx && y >= ty0 && y < ty0 + wy);
        if (inside) v += (double)p * (double)p;
        else        omax = fmaxf(omax, fabsf(p));
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        v    += __shfl_down_sync(0xFFFFFFFFu, v, d);
        omax  = fmaxf(omax, __shfl_down_sync(0xFFFFFFFFu, omax, d));
    }
    if (lane == 0) { vred[warp] = v; ored[warp] = omax; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double sv = 0.0; float so = 0.0f;
        for (int w = 0; w < nwarp; ++w) { sv += vred[w]; so = fmaxf(so, ored[w]); }
        out_V[n] = sv;
        out_outside_max[n] = so;
    }
}

__global__ void k_verify_S(const uint32_t* S, size_t n, uint32_t* out_max) {
    uint32_t m = 0u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        m = max(m, S[i]);
    atomicMax(out_max, m);
}

__global__ void k_pack_traj(const CellState* cell, const uint8_t* clsv,
                            TrajPackedCell* out, int N, int L)
{
    const int n = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (n >= N) return;
    const CellState cs = cell[n];
    const int cls = (int)clsv[n];
    const double V = cs.V > 0.0 ? cs.V : 1.0;
    const double cx = (double)cs.gx0 + cs.Cx / V;
    const double cy = (double)cs.gy0 + cs.Cy / V;
    TrajPackedCell t;
    t.global_id = cs.global_id;
    t.cls   = cls;
    t.cx    = (float)(cx - floor(cx / (double)L) * (double)L);
    t.cy    = (float)(cy - floor(cy / (double)L) * (double)L);
    t.vx    = cs.vx;   t.vy = cs.vy;
    t.theta = cs.theta;
    t.volume = (float)cs.V;
    t.perim  = (float)cs.perim;
    t.gamma  = cs.gamma;
    t.v_A    = cs.v_A;
    t.phi_max = cs.phi_max;
    out[n] = t;                    // mapped pinned host memory
}

__global__ void k_morton_sort(const CellState* cell, uint32_t* perm,
                              int N, int M, int L)
{
    extern __shared__ unsigned long long ks[];
    for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x) {
        unsigned long long k = 0xFFFFFFFFFFFFFFFFull;
        if (i < N) {
            const CellState cs = cell[i];
            const double V = cs.V > 0.0 ? cs.V : 1.0;
            const int cx = wrapi(cs.gx0 + (int)(cs.Cx / V), L);
            const int cy = wrapi(cs.gy0 + (int)(cs.Cy / V), L);
            k = ((unsigned long long)morton2d((uint32_t)cx, (uint32_t)cy) << 32)
              | (unsigned long long)(uint32_t)i;
        }
        ks[i] = k;
    }
    __syncthreads();
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x) {
                const int q = i ^ j;
                if (q > i) {
                    const bool up = ((i & k) == 0);
                    if ((ks[i] > ks[q]) == up) {
                        const unsigned long long tmp = ks[i];
                        ks[i] = ks[q];
                        ks[q] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = (int)threadIdx.x; i < N; i += (int)blockDim.x)
        perm[i] = (uint32_t)(ks[i] & 0xFFFFFFFFull);
}

void configure_k_step_smem() {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(k_step),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    cudaFuncSetAttribute(reinterpret_cast<const void*>(k_step),
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

void configure_morton_smem(int smem_bytes) {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(k_morton_sort),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
}

int k_step_grid(int device) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms > 0 ? sms : 1;
}

void launch_step(const StepArgs& A, int grid, cudaStream_t stream,
                 const void* l2_base, size_t l2_bytes, float l2_hit_ratio)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid, 1, 1);
    cfg.blockDim = dim3((unsigned)kBlockThreads, 1, 1);
    cfg.dynamicSmemBytes = (size_t)kSmemBytes;
    cfg.stream = stream;

    cudaLaunchAttribute attr[1];
    int nattr = 0;
    if (l2_base != nullptr && l2_bytes > 0) {
        attr[0].id = cudaLaunchAttributeAccessPolicyWindow;
        attr[0].val.accessPolicyWindow.base_ptr = const_cast<void*>(l2_base);
        attr[0].val.accessPolicyWindow.num_bytes = l2_bytes;
        attr[0].val.accessPolicyWindow.hitRatio = l2_hit_ratio;
        attr[0].val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr[0].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        nattr = 1;
    }
    cfg.attrs = nattr ? attr : nullptr;
    cfg.numAttrs = (unsigned)nattr;

    cudaLaunchKernelEx(&cfg, k_step, A);
    cfg.dynamicSmemBytes = (size_t)kScalarBytes;
    cudaLaunchKernelEx(&cfg, k_step_fallback, A);
}

// Query occupancy after configuring dynamic shared memory. Report the driver's
// result and an independent register-limit estimate.
bool query_kernel_stats(const void* fn, int block_threads, int dynamic_smem,
                        int device, KernelStats* out)
{
    if (out == nullptr) return false;
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, fn) != cudaSuccess) return false;

    int blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, fn, block_threads, (size_t)dynamic_smem) != cudaSuccess)
        return false;

    int max_threads = 0;
    if (cudaDeviceGetAttribute(&max_threads,
                               cudaDevAttrMaxThreadsPerMultiProcessor,
                               device) != cudaSuccess || max_threads <= 0)
        max_threads = kMaxThreadsPerSmSm90;

    out->regs         = fa.numRegs;
    out->local_bytes  = (size_t)fa.localSizeBytes;
    out->static_smem  = (int)fa.sharedSizeBytes;
    out->dynamic_smem = dynamic_smem;
    out->ctas_per_sm  = blocks;
    out->reg_limited_ctas =
        (fa.numRegs > 0 && block_threads > 0)
            ? kRegsPerSmSm90 / (fa.numRegs * block_threads) : 0;
    out->warps_per_sm = blocks * block_threads / 32;
    out->occupancy    = (double)(blocks * block_threads) / (double)max_threads;
    return true;
}

}  // namespace pf
