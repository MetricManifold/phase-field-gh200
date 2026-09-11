#include "../include/boundary.cuh"
#include "../common/boundary_format.h"
#include "output_file.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#ifdef PF_BOUNDARY_ZSTD
#include <zstd.h>
#endif

namespace pf {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
bool cuda_ok(cudaError_t e, const char* operation) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "[boundary] %s: %s\n", operation, cudaGetErrorString(e));
    return false;
}
#define BCU(call) do { if (!cuda_ok((call), #call)) return false; } while (0)

__device__ bool crossed(const float* phi, int q, boundary::Square* out) {
    const int x = q % (kTilePitch - 1), y = q / (kTilePitch - 1);
    const int j = y * kTilePitch + x;
    out->xy = (uint32_t)x | ((uint32_t)y << 16);
    out->phi[0] = phi[j];
    out->phi[1] = phi[j + 1];
    out->phi[2] = phi[j + kTilePitch + 1];
    out->phi[3] = phi[j + kTilePitch];
    const bool first = out->phi[0] >= 0.5f;
    return (out->phi[1] >= 0.5f) != first ||
           (out->phi[2] >= 0.5f) != first ||
           (out->phi[3] >= 0.5f) != first;
}

__global__ void count_squares(const float* phi, const CellState* cells,
                              boundary::Cell* meta, int side) {
    const int i = blockIdx.x, t = threadIdx.x;
    const float* tile = phi + (size_t)i * kTileArea;
    unsigned count = 0, bad = 0;
    for (int q = t; q < (kTilePitch-1)*(kTilePitch-1); q += blockDim.x) {
        boundary::Square s;
        count += crossed(tile, q, &s);
        for (int k = 0; k < 4; ++k) bad |= !isfinite(s.phi[k]);
    }
    __shared__ unsigned counts[256], invalid[256];
    counts[t] = count; invalid[t] = bad;
    __syncthreads();
    for (int stride = blockDim.x/2; stride; stride >>= 1) {
        if (t < stride) {
            counts[t] += counts[t+stride];
            invalid[t] |= invalid[t+stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        const CellState c = cells[i];
        const ShapeClass sc = class_of(c.cls);
        boundary::Cell m{};
        m.id = c.global_id;
        m.origin_x = ((c.gx0 - sc.tx0) % side + side) % side;
        m.origin_y = ((c.gy0 - sc.ty0) % side + side) % side;
        const double x = (double)c.gx0 + c.Cx/c.V;
        const double y = (double)c.gy0 + c.Cy/c.V;
        m.cx = (float)(x - floor(x/side)*side);
        m.cy = (float)(y - floor(y/side)*side);
        m.gamma = c.gamma; m.mobility = (float)kPhaseFieldMobility;
        m.active_speed = c.v_A; m.radius = c.R_tgt;
        m.squares = invalid[0] ? UINT32_MAX : counts[0];
        meta[i] = m;
    }
}

__global__ void gather_squares(const float* phi, const uint64_t* offsets,
                               boundary::Square* result) {
    const int i = blockIdx.x, t = threadIdx.x;
    const float* tile = phi + (size_t)i * kTileArea;
    __shared__ unsigned count;
    if (t == 0) count = 0;
    __syncthreads();
    for (int base = 0; base < (kTilePitch-1)*(kTilePitch-1); base += blockDim.x) {
        const int q = base+t;
        boundary::Square s;
        bool hit = false;
        if (q < (kTilePitch-1)*(kTilePitch-1)) hit = crossed(tile, q, &s);
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        unsigned begin = 0;
        if ((t & 31) == 0) begin = atomicAdd(&count, __popc(mask));
        begin = __shfl_sync(0xffffffffu, begin, 0);
        if (hit) {
            const unsigned rank = __popc(mask & ((1u << (t & 31))-1u));
            result[offsets[i] + begin + rank] = s;
        }
    }
}
uint64_t checksum(const unsigned char* p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
} // namespace

struct BoundaryOutput::Impl {
    struct Slot {
        unsigned char* data = nullptr;
        size_t capacity = 0, bytes = 0;
        uint64_t step = 0;
        double time = 0;
    };
    std::array<Slot, 3> slots;
    std::deque<int> free{0,1,2}, pending;
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    bool stopping = false, failed = false, capture_failed = false;
    bool compress = false;
    std::FILE* fp = nullptr;
    int n = 0, side = 0;
    boundary::Cell* d_meta = nullptr;
    uint64_t* d_offsets = nullptr;
    boundary::Square* d_squares = nullptr;
    size_t capacity = 0;
    std::vector<boundary::Cell> meta;
    std::vector<uint64_t> offsets;
    uint64_t frames = 0, raw_bytes = 0, stored_bytes = 0, last_step = 0;
    double last_time = 0, capture_seconds = 0, queue_seconds = 0, writer_seconds = 0;

    ~Impl() {
        if (worker.joinable()) finish();
        if (fp) std::fclose(fp);
        cudaFree(d_meta); cudaFree(d_offsets); cudaFree(d_squares);
        for (auto& s : slots) if (s.data) cudaFreeHost(s.data);
    }
    bool write(const void* p, size_t bytes) {
        if (std::fwrite(p, 1, bytes, fp) == bytes) return true;
        std::perror("[boundary] write");
        return false;
    }
    void write_frames() {
        std::vector<unsigned char> encoded;
#ifdef PF_BOUNDARY_ZSTD
        ZSTD_CCtx* context = compress ? ZSTD_createCCtx() : nullptr;
        if (compress && !context) {
            std::lock_guard<std::mutex> lock(mutex);
            failed = true; changed.notify_all(); return;
        }
#endif
        for (;;) {
            int index;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&]{ return stopping || !pending.empty(); });
                if (pending.empty()) break;
                index = pending.front(); pending.pop_front();
            }
            const auto start = Clock::now();
            Slot& s = slots[index];
            const void* payload = s.data;
            size_t bytes = s.bytes;
            bool ok = true;
#ifdef PF_BOUNDARY_ZSTD
            if (compress) {
                encoded.resize(ZSTD_compressBound(s.bytes));
                bytes = ZSTD_compressCCtx(context, encoded.data(), encoded.size(),
                                          s.data, s.bytes, 1);
                ok = !ZSTD_isError(bytes);
                if (!ok) std::fprintf(stderr, "[boundary] zstd: %s\n", ZSTD_getErrorName(bytes));
                payload = encoded.data();
            }
#endif
            boundary::FrameHeader h{};
            std::memcpy(h.magic, "PFBFRM1", 8);
            h.step = s.step; h.time = s.time;
            h.raw_bytes = s.bytes; h.stored_bytes = bytes;
            h.checksum = checksum(s.data, s.bytes);
            ok = ok && write(&h, sizeof(h)) && write(payload, bytes);
            writer_seconds += seconds(start);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!ok) { failed = true; changed.notify_all(); break; }
                ++frames; raw_bytes += s.bytes; stored_bytes += bytes + sizeof(h);
                last_step = s.step; last_time = s.time;
                free.push_back(index);
            }
            changed.notify_all();
        }
#ifdef PF_BOUNDARY_ZSTD
        if (context) ZSTD_freeCCtx(context);
#endif
    }
    bool finish() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        changed.notify_all();
        if (worker.joinable()) worker.join();
        bool ok = !failed && !capture_failed;
        if (fp) {
            if (ok) {
                boundary::FrameHeader end{};
                std::memcpy(end.magic, "PFBEND1", 8);
                end.step = last_step; end.time = last_time; end.checksum = frames;
                ok = write(&end, sizeof(end));
            }
            if (std::fclose(fp) != 0) { std::perror("[boundary] close"); ok = false; }
            fp = nullptr;
            std::printf("[boundary] frames=%llu raw_bytes=%llu stored_bytes=%llu "
                        "capture_s=%.6f queue_wait_s=%.6f writer_s=%.6f status=%s\n",
                        (unsigned long long)frames, (unsigned long long)raw_bytes,
                        (unsigned long long)stored_bytes, capture_seconds,
                        queue_seconds, writer_seconds, ok ? "complete" : "FAILED");
        }
        failed = !ok;
        return ok;
    }
    bool capture(const float* phi, const CellState* cells, cudaStream_t stream,
                 long long step, double time) {
        const auto start = Clock::now();
        count_squares<<<n,256,0,stream>>>(phi, cells, d_meta, side);
        BCU(cudaGetLastError());
        BCU(cudaMemcpyAsync(meta.data(), d_meta, n*sizeof(boundary::Cell),
                            cudaMemcpyDeviceToHost, stream));
        BCU(cudaStreamSynchronize(stream));
        uint64_t total = 0;
        for (int i = 0; i < n; ++i) {
            if (!meta[i].squares || meta[i].squares == UINT32_MAX) {
                std::fprintf(stderr, "[boundary] cell %d has missing/non-finite boundary\n", meta[i].id);
                return false;
            }
            offsets[i] = total; total += meta[i].squares;
        }
        const size_t prefix = n*sizeof(boundary::Cell);
        const uint64_t bytes64 = prefix + total*sizeof(boundary::Square);
        if (bytes64 > (1ull<<31)) {
            std::fprintf(stderr, "[boundary] frame exceeds 2 GiB safety limit\n");
            return false;
        }
        const size_t bytes = (size_t)bytes64;
        if (total > capacity) {
            BCU(cudaFree(d_squares)); d_squares = nullptr;
            capacity = (size_t)(total + total/8 + 256);
            BCU(cudaMalloc((void**)&d_squares, capacity*sizeof(boundary::Square)));
        }
        BCU(cudaMemcpyAsync(d_offsets, offsets.data(), n*sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream));
        gather_squares<<<n,256,0,stream>>>(phi, d_offsets, d_squares);
        BCU(cudaGetLastError());
        const auto wait_start = Clock::now();
        int index;
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&]{ return failed || !free.empty(); });
            if (failed) return false;
            index = free.front(); free.pop_front();
        }
        queue_seconds += seconds(wait_start);
        Slot& s = slots[index];
        if (s.capacity < bytes) {
            if (s.data) { BCU(cudaFreeHost(s.data)); s.data = nullptr; }
            s.capacity = bytes + bytes/8 + 4096;
            BCU(cudaHostAlloc((void**)&s.data, s.capacity, cudaHostAllocDefault));
        }
        std::memcpy(s.data, meta.data(), prefix);
        BCU(cudaMemcpyAsync(s.data + prefix, d_squares,
                            total*sizeof(boundary::Square), cudaMemcpyDeviceToHost, stream));
        BCU(cudaStreamSynchronize(stream));
        s.bytes = bytes; s.step = step; s.time = time;
        { std::lock_guard<std::mutex> lock(mutex); pending.push_back(index); }
        changed.notify_all();
        capture_seconds += seconds(start);
        return true;
    }
};

BoundaryOutput::BoundaryOutput() = default;
BoundaryOutput::~BoundaryOutput() { if (impl_) impl_->finish(); }
bool BoundaryOutput::open(const std::string& path, const SimParams& p, int side,
                          long long interval, bool compress) {
    if (impl_) return false;
    if (path.empty() || p.num_cells <= 0 || side <= 0 || interval <= 0 ||
        !std::isfinite(p.dt) || p.dt <= 0 || !std::isfinite(p.tau) || p.tau <= 0) {
        std::fprintf(stderr, "[boundary] invalid output geometry or cadence\n");
        return false;
    }
#ifndef PF_BOUNDARY_ZSTD
    if (compress) {
        std::fprintf(stderr, "[boundary] this build has no zstd; select --boundary-compression none\n");
        return false;
    }
#endif
    impl_ = std::make_unique<Impl>();
    auto& b = *impl_;
    b.n = p.num_cells; b.side = side; b.compress = compress;
    b.meta.resize(b.n); b.offsets.resize(b.n);
    BCU(cudaMalloc((void**)&b.d_meta, b.n*sizeof(boundary::Cell)));
    BCU(cudaMalloc((void**)&b.d_offsets, b.n*sizeof(uint64_t)));
    b.fp = open_new_binary_file(path);
    if (!b.fp) { std::perror("[boundary] cannot create new output file"); return false; }
    boundary::FileHeader h{};
    std::memcpy(h.magic, "PFBND01", 8);
    h.version = 1; h.header_bytes = sizeof(h); h.cells = b.n; h.side = side;
    h.tile_pitch = kTilePitch; h.square_bytes = sizeof(boundary::Square);
    h.dt = p.dt; h.tau = p.tau; h.level = 0.5; h.interval = interval;
    h.codec = compress ? 1 : 0;
    if (!b.write(&h, sizeof(h))) { b.failed = true; return false; }
    b.worker = std::thread([&b]{ b.write_frames(); });
    return true;
}
bool BoundaryOutput::capture(const float* phi, const CellState* cells,
                             cudaStream_t stream, long long step, double time) {
    if (!impl_ || !impl_->capture(phi, cells, stream, step, time)) {
        if (impl_) impl_->capture_failed = true;
        return false;
    }
    return true;
}
bool BoundaryOutput::close() { return !impl_ || impl_->finish(); }
} // namespace pf
