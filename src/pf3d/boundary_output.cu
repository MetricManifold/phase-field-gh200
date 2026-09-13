#include "pf3d/boundary_output.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif
#ifdef PF_BOUNDARY_ZSTD
#include <zstd.h>
#endif

namespace pf3d {
namespace {

constexpr int kProjectionThreads = 256;
constexpr int kProjectionWarps = kProjectionThreads / 32;

bool report_error(const char* message) {
    std::fprintf(stderr, "[boundary3d] %s\n", message);
    return false;
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[boundary3d] %s: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

#define BOUNDARY_CUDA(call) \
    do { if (!cuda_ok((call), #call)) return false; } while (false)

bool checked_add_size(std::size_t a, std::size_t b, std::size_t* out) {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool valid_edge(int edge) {
    return edge > 0 && edge <= 65534 && edge % kBrickAlignment == 0;
}

bool positive_finite(double value) {
    return std::isfinite(value) && value > 0.0;
}

bool valid_header_params(const SimParams3D& p, boundary::Projection projection,
                         std::uint64_t interval) {
    const bool known_geometry = p.periodic_xyz() || p.bounded_z();
    const bool known_projection = projection == boundary::Projection::Maximum ||
        (projection == boundary::Projection::Basal && p.substrate_slab());
    return p.num_cells > 0 && p.Nx > 0 && p.Ny > 0 && p.Nz > 0 &&
        known_geometry && known_projection && interval > 0 &&
        positive_finite(p.dx) && positive_finite(p.dy) && positive_finite(p.dz) &&
        positive_finite(p.dt) && positive_finite(p.tau);
}

bool valid_cell(const CellState3D& cell, int base_edge) {
    if (cell.storage_edge > 65534u) return false;
    const int edge = cell_support_edge(cell, base_edge);
    const auto low = std::numeric_limits<std::int64_t>::min();
    const auto high = std::numeric_limits<std::int64_t>::max();
    return valid_edge(edge) && edge >= base_edge && cell.global_id >= 0 &&
        cell.origin_x > low && cell.origin_y > low &&
        cell.origin_x <= high - edge && cell.origin_y <= high - edge &&
        cell.origin_z <= high - (edge - 1) &&
        positive_finite(cell.gamma) && positive_finite(cell.R_tgt) &&
        std::isfinite(cell.v_A) && cell.v_A >= 0.0f;
}

// No world-coordinate addition is needed on the device. Out-of-domain planes
// are not inspected, and an empty physical intersection projects to zero.
void projection_z_range(const CellState3D& cell, int edge, int nz,
                        bool periodic, boundary::Projection projection,
                        int* begin, int* end) {
    *begin = 0;
    *end = edge;
    if (projection == boundary::Projection::Basal) {
        if (cell.origin_z > 0 || cell.origin_z <= -std::int64_t(edge)) {
            *end = 0;
        } else {
            *begin = static_cast<int>(-cell.origin_z);
            *end = *begin + 1;
        }
        return;
    }
    if (periodic) return;
    if (cell.origin_z >= nz || cell.origin_z <= -std::int64_t(edge)) {
        *end = 0;
        return;
    }
    if (cell.origin_z < 0) *begin = static_cast<int>(-cell.origin_z);
    *end = static_cast<int>(std::min<std::int64_t>(edge,
        std::int64_t(nz) - cell.origin_z));
}

__global__ void project_cell(const float* base_phi,
                             const float* const* promoted_phi,
                             std::size_t base_voxels, int slot, int edge,
                             bool promoted, int z_begin, int z_end,
                             CellFieldStorage3D storage, std::int64_t origin_z,
                             float* plane, std::uint32_t* invalid) {
    const std::size_t plane_edge = static_cast<std::size_t>(edge) + 2;
    const std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (q >= plane_edge * plane_edge) return;
    const std::size_t x = q % plane_edge;
    const std::size_t y = q / plane_edge;
    plane[q] = 0.0f;
    if (x == 0 || y == 0 || x + 1 == plane_edge || y + 1 == plane_edge)
        return;
    const float* source = promoted
        ? (promoted_phi ? promoted_phi[slot] : nullptr)
        : (base_phi ? base_phi + static_cast<std::size_t>(slot) * base_voxels
                    : nullptr);
    if (!source) {
        atomicOr(invalid, 1u);
        return;
    }
    if (z_begin >= z_end) return;
    const std::size_t source_plane = static_cast<std::size_t>(edge) * edge;
    std::size_t at = storage.index(static_cast<int>(x - 1),
        static_cast<int>(y - 1), z_begin, edge, origin_z);
    float value = source[at];
    if (!isfinite(value)) atomicOr(invalid, 1u);
    for (int z = z_begin + 1; z < z_end; ++z) {
        at += source_plane;
        const float sample = source[at];
        if (!isfinite(sample)) atomicOr(invalid, 1u);
        if (sample > value) value = sample;
    }
    plane[q] = value;
}

__device__ bool crossing_square(const float* plane, std::uint32_t q,
                                int plane_edge, boundary::Square* square) {
    const std::uint32_t x = q % (plane_edge - 1);
    const std::uint32_t y = q / (plane_edge - 1);
    const std::size_t at = static_cast<std::size_t>(y) * plane_edge + x;
    square->xy = x | (y << 16);
    square->phi[0] = plane[at];
    square->phi[1] = plane[at + 1];
    square->phi[2] = plane[at + plane_edge + 1];
    square->phi[3] = plane[at + plane_edge];
    const bool inside = square->phi[0] >= 0.5f;
    return (square->phi[1] >= 0.5f) != inside ||
        (square->phi[2] >= 0.5f) != inside ||
        (square->phi[3] >= 0.5f) != inside;
}

__global__ void count_squares(const float* plane, int plane_edge,
                              std::uint32_t* count) {
    const std::size_t squares = static_cast<std::size_t>(plane_edge - 1) *
        (plane_edge - 1);
    std::uint32_t local_count = 0;
    for (std::size_t q = threadIdx.x; q < squares; q += blockDim.x) {
        boundary::Square square;
        local_count += crossing_square(plane, static_cast<std::uint32_t>(q),
                                       plane_edge, &square);
    }
    __shared__ std::uint32_t counts[kProjectionThreads];
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *count = counts[0];
}

__global__ void gather_squares(const float* plane, int plane_edge,
                               boundary::Square* output) {
    const std::size_t squares = static_cast<std::size_t>(plane_edge - 1) *
        (plane_edge - 1);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x / 32u;
    __shared__ std::uint32_t offsets[kProjectionWarps];
    __shared__ std::uint32_t total;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    // A tiny ordered warp prefix makes records deterministic in local y/x
    // order without a per-pixel host buffer or a full-size scan workspace.
    for (std::size_t base = 0; base < squares; base += blockDim.x) {
        const std::size_t q = base + threadIdx.x;
        boundary::Square square;
        const bool hit = q < squares && crossing_square(
            plane, static_cast<std::uint32_t>(q), plane_edge, &square);
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) offsets[warp] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < kProjectionWarps; ++w) {
                const std::uint32_t count = offsets[w];
                offsets[w] = total;
                total += count;
            }
        }
        __syncthreads();
        if (hit) output[offsets[warp] + __popc(mask & ((1u << lane) - 1u))] = square;
        __syncthreads();
    }
}

std::uint64_t checksum(const unsigned char* bytes, std::size_t size) {
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::FILE* create_exclusive(const std::string& path) {
#ifdef _WIN32
    const int descriptor = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL |
        _O_BINARY, _S_IREAD | _S_IWRITE);
    if (descriptor < 0) return nullptr;
    std::FILE* file = _fdopen(descriptor, "wb");
    if (!file) _close(descriptor);
#else
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (descriptor < 0) return nullptr;
    std::FILE* file = fdopen(descriptor, "wb");
    if (!file) ::close(descriptor);
#endif
    return file;
}

template <typename T>
bool free_device(T*& pointer) {
    if (!pointer) return true;
    BOUNDARY_CUDA(cudaFree(pointer));
    pointer = nullptr;
    return true;
}

template <typename T>
bool reserve_device(T*& pointer, std::size_t& capacity, std::size_t count) {
    if (count <= capacity) return true;
    std::size_t bytes = 0;
    if (!checked_mul_size(count, sizeof(T), &bytes))
        return report_error("device workspace size overflow");
    if (!free_device(pointer)) return false;
    capacity = 0;
    BOUNDARY_CUDA(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes));
    capacity = count;
    return true;
}

} // namespace

struct BoundaryOutput3D::Impl {
    std::FILE* file = nullptr;
    SimParams3D params{};
    boundary::Projection projection = boundary::Projection::Maximum;
    bool compress = false;
    bool failed = false;
    bool closed = false;
    std::uint64_t frames = 0;
    std::uint64_t last_step = 0;
    double last_time = 0.0;
    float* device_plane = nullptr;
    boundary::Square* device_squares = nullptr;
    std::uint32_t* device_status = nullptr;
    std::size_t plane_capacity = 0;
    std::size_t square_capacity = 0;
    std::size_t status_capacity = 0;
    std::vector<unsigned char> payload;
    std::vector<unsigned char> encoded;

    ~Impl() {
        // Destruction is not evidence that the simulation completed.
        finish(false);
    }

    bool write(const void* bytes, std::size_t size) {
        if (std::fwrite(bytes, 1, size, file) == size) return true;
        std::perror("[boundary3d] write");
        failed = true;
        return false;
    }

    bool flush() {
        if (std::fflush(file) == 0) return true;
        std::perror("[boundary3d] flush");
        failed = true;
        return false;
    }

    bool finish(bool complete) {
        if (closed) return !failed;
        bool ok = free_device(device_plane);
        ok = free_device(device_squares) && ok;
        ok = free_device(device_status) && ok;
        if (!ok) failed = true;
        if (file) {
            if (complete && !failed) {
                boundary::FrameHeader end{};
                std::memcpy(end.magic, "P3BEND1", 8);
                end.step = last_step;
                end.time = last_time;
                write(&end, sizeof(end));
            }
            if (std::fclose(file) != 0) {
                std::perror("[boundary3d] close");
                failed = true;
            }
            file = nullptr;
        }
        closed = true;
        return !failed;
    }

    bool append_cell(const float* base_phi, const float* const* promoted_phi,
                     const CellState3D& state, int slot, int base_edge,
                     std::size_t base_voxels, cudaStream_t stream,
                     CellFieldStorage3D storage) {
        const int edge = cell_support_edge(state, base_edge);
        const int plane_edge = edge + 2;
        std::size_t plane_values = 0;
        std::size_t field_bytes = 0;
        if (!storage.checked_bytes(edge, &field_bytes) ||
            !checked_mul_size(plane_edge, plane_edge, &plane_values) ||
            !reserve_device(device_plane, plane_capacity, plane_values) ||
            !reserve_device(device_status, status_capacity, 2)) return false;
        int z_begin = 0, z_end = 0;
        projection_z_range(state, edge, params.Nz, params.periodic_xyz(),
                           projection, &z_begin, &z_end);
        BOUNDARY_CUDA(cudaMemsetAsync(device_status, 0,
            2 * sizeof(std::uint32_t), stream));
        const unsigned blocks = static_cast<unsigned>(
            (plane_values + kProjectionThreads - 1) / kProjectionThreads);
        project_cell<<<blocks, kProjectionThreads, 0, stream>>>(
            base_phi, promoted_phi, base_voxels, slot, edge,
            cell_is_promoted(state, base_edge), z_begin, z_end,
            storage, state.origin_z,
            device_plane, device_status);
        BOUNDARY_CUDA(cudaGetLastError());
        count_squares<<<1, kProjectionThreads, 0, stream>>>(
            device_plane, plane_edge, device_status + 1);
        BOUNDARY_CUDA(cudaGetLastError());
        std::uint32_t status[2]{};
        BOUNDARY_CUDA(cudaMemcpyAsync(status, device_status, sizeof(status),
                                      cudaMemcpyDeviceToHost, stream));
        BOUNDARY_CUDA(cudaStreamSynchronize(stream));
        if (status[0]) return report_error("missing field pointer or nonfinite projected sample");
        const std::uint32_t count = status[1];
        std::size_t square_bytes = 0, next_bytes = 0;
        if (!checked_mul_size(count, sizeof(boundary::Square), &square_bytes) ||
            !checked_add_size(payload.size(), square_bytes, &next_bytes) ||
            next_bytes > payload.max_size())
            return report_error("compact frame size overflow");
        const std::size_t square_offset = payload.size();
        payload.resize(next_bytes);
        if (count > 0) {
            if (!reserve_device(device_squares, square_capacity, count)) return false;
            gather_squares<<<1, kProjectionThreads, 0, stream>>>(
                device_plane, plane_edge, device_squares);
            BOUNDARY_CUDA(cudaGetLastError());
            BOUNDARY_CUDA(cudaMemcpyAsync(payload.data() + square_offset,
                device_squares, square_bytes, cudaMemcpyDeviceToHost, stream));
            BOUNDARY_CUDA(cudaStreamSynchronize(stream));
        }
        boundary::Cell record{};
        record.id = state.global_id;
        record.origin_x = state.origin_x - 1;
        record.origin_y = state.origin_y - 1;
        record.origin_z = state.origin_z;
        record.brick_edge = edge;
        record.plane_edge = plane_edge;
        record.squares = count;
        record.gamma = state.gamma;
        record.active_speed = state.v_A;
        record.radius = state.R_tgt;
        std::memcpy(payload.data() + static_cast<std::size_t>(slot) * sizeof(record),
                    &record, sizeof(record));
        return true;
    }

    bool write_frame(std::uint64_t step, double time) {
        const void* stored = payload.data();
        std::size_t stored_bytes = payload.size();
#ifdef PF_BOUNDARY_ZSTD
        if (compress) {
            const std::size_t bound = ZSTD_compressBound(payload.size());
            if (ZSTD_isError(bound) || bound > encoded.max_size())
                return report_error("zstd compression bound overflow");
            encoded.resize(bound);
            stored_bytes = ZSTD_compress(encoded.data(), encoded.size(),
                payload.data(), payload.size(), 1);
            if (ZSTD_isError(stored_bytes)) {
                std::fprintf(stderr, "[boundary3d] zstd: %s\n",
                             ZSTD_getErrorName(stored_bytes));
                return false;
            }
            stored = encoded.data();
        }
#endif
        boundary::FrameHeader header{};
        std::memcpy(header.magic, "P3BFRM1", 8);
        header.step = step;
        header.time = time;
        header.raw_bytes = payload.size();
        header.stored_bytes = stored_bytes;
        header.checksum = checksum(payload.data(), payload.size());
        if (!write(&header, sizeof(header)) || !write(stored, stored_bytes) || !flush())
            return false;
        ++frames;
        last_step = step;
        last_time = time;
        return true;
    }

    bool capture(const float* base_phi, const float* const* promoted_phi,
                 const CellState3D* cells, int base_edge, cudaStream_t stream,
                 std::uint64_t step, double time, CellFieldStorage3D storage) {
        if (failed || closed || !file) return false;
        if (!cells || !valid_edge(base_edge) || !std::isfinite(time) || time < 0.0 ||
            (frames > 0 && (step <= last_step || time < last_time)))
            return report_error("invalid capture pointers, brick edge, or frame order");
        if (storage.z_cap != 0 &&
            (!params.bounded_z() || storage.z_cap != params.Nz))
            return report_error("compact storage must match the bounded domain height");
        std::size_t base_voxels = 0, field_bytes = 0, metadata_bytes = 0;
        std::size_t base_bytes = 0;
        if (!storage.checked_words(base_edge, &base_voxels) ||
            !storage.checked_bytes(base_edge, &base_bytes) ||
            !checked_mul_size(params.num_cells, base_bytes, &field_bytes) ||
            !checked_mul_size(params.num_cells, sizeof(boundary::Cell), &metadata_bytes) ||
            metadata_bytes > payload.max_size())
            return report_error("capture size overflow");
        payload.resize(metadata_bytes);
        std::unordered_set<std::int64_t> ids;
        for (int slot = 0; slot < params.num_cells; ++slot) {
            CellState3D state{};
            BOUNDARY_CUDA(cudaMemcpyAsync(&state, cells + slot, sizeof(state),
                                          cudaMemcpyDeviceToHost, stream));
            BOUNDARY_CUDA(cudaStreamSynchronize(stream));
            if (!valid_cell(state, base_edge) || !ids.insert(state.global_id).second)
                return report_error("invalid or duplicate cell metadata");
            if (!append_cell(base_phi, promoted_phi, state, slot, base_edge,
                             base_voxels, stream, storage)) return false;
        }
        return write_frame(step, time);
    }
};

BoundaryOutput3D::BoundaryOutput3D() = default;
BoundaryOutput3D::~BoundaryOutput3D() = default;

bool BoundaryOutput3D::open(const std::string& path, const SimParams3D& params,
                           boundary::Projection projection,
                           std::uint64_t interval, bool compress) {
    if (impl_) return report_error("output object has already been opened");
    const std::uint16_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1)
        return report_error("boundary output requires a little-endian host");
    if (path.empty() || !valid_header_params(params, projection, interval))
        return report_error("invalid output geometry, projection, or cadence");
#ifndef PF_BOUNDARY_ZSTD
    if (compress) return report_error("zstd unavailable; select --boundary-compression none");
#endif
    try {
        impl_ = std::make_unique<Impl>();
        Impl& output = *impl_;
        output.params = params;
        output.projection = projection;
        output.compress = compress;
        output.file = create_exclusive(path);
        if (!output.file) {
            output.failed = true;
            std::perror("[boundary3d] cannot create new output file");
            return false;
        }
        boundary::FileHeader header{};
        std::memcpy(header.magic, "PFB3D1", 7);
        header.version = 1;
        header.header_bytes = sizeof(header);
        header.cells = params.num_cells;
        header.projection = static_cast<std::uint32_t>(projection);
        header.boundary_flags = params.boundary_flags;
        header.codec = compress ? 1 : 0;
        header.nx = params.Nx;
        header.ny = params.Ny;
        header.nz = params.Nz;
        header.dx = params.dx;
        header.dy = params.dy;
        header.dz = params.dz;
        header.dt = params.dt;
        header.tau = params.tau;
        header.level = 0.5;
        header.interval = interval;
        header.square_bytes = sizeof(boundary::Square);
        header.cell_bytes = sizeof(boundary::Cell);
        return output.write(&header, sizeof(header)) && output.flush();
    } catch (const std::exception& error) {
        if (impl_) impl_->failed = true;
        std::fprintf(stderr, "[boundary3d] open: %s\n", error.what());
        return false;
    }
}

bool BoundaryOutput3D::capture(const float* base_phi,
                              const float* const* promoted_phi,
                              const CellState3D* cells, int base_edge,
                              cudaStream_t stream, std::uint64_t step,
                              double time, CellFieldStorage3D storage) {
    if (!impl_) return report_error("capture requires an open output");
    try {
        if (impl_->capture(base_phi, promoted_phi, cells, base_edge, stream,
                           step, time, storage))
            return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[boundary3d] capture: %s\n", error.what());
    }
    impl_->failed = true;
    return false;
}

bool BoundaryOutput3D::close(bool complete) {
    return !impl_ || impl_->finish(complete);
}

#undef BOUNDARY_CUDA

} // namespace pf3d
