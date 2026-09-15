#include "velocity_output.cuh"
#include "velocity_moments.cuh"
#include "velocity_format.h"
#include "output_file.hpp"
#include <cmath>
#include <cstring>
#include <limits>

namespace pf {
namespace {
bool cuda_ok(cudaError_t result, const char* operation) {
    if (result == cudaSuccess)
        return true;
    std::fprintf(stderr, "[velocity] %s: %s\n", operation, cudaGetErrorString(result));
    return false;
}
#define VCU(call)                                                                              \
    do {                                                                                       \
        if (!cuda_ok((call), #call))                                                           \
            return false;                                                                      \
    } while (0)
} // namespace

VelocityOutput::~VelocityOutput() {
    close();
    cudaFree(moments_);
    cudaFree(endpoint_);
}

bool VelocityOutput::configure(const VelocityOptions& options, int cells, double dt) {
    if (enabled())
        return false;
    options_ = options;
    dt_ = dt;
    if (options.path.empty())
        return true;
    if (cells <= 0 || !std::isfinite(dt) || dt <= 0.0 || options.spatial_stride < 1 ||
        options.spatial_stride > 1000 || options.dense_start < 0) {
        std::fprintf(stderr, "[velocity] invalid geometry or sampling cadence\n");
        return false;
    }
    sampling_.configure(true, options.reference, options.spatial_stride);
    cells_.resize(cells);
    integrals_.resize(cells);
    endpoints_.resize(cells);
    rows_.resize(std::size_t(cells) * velocity::kColumns);
    const auto bytes = std::size_t(cells) * sizeof(velocity::MomentAccum);
    VCU(cudaMalloc(&moments_, bytes));
    VCU(cudaMalloc(&endpoint_, bytes));
    VCU(cudaMemset(moments_, 0, bytes));
    return true;
}

bool VelocityOutput::write(const void* data, std::size_t bytes) {
    if (std::fwrite(data, 1, bytes, file_) == bytes)
        return true;
    std::perror("[velocity] write");
    failed_ = true;
    return false;
}

bool VelocityOutput::open(const StepArgs& state, long long step, cudaStream_t stream) {
    if (!enabled())
        return true;
    if (file_ || failed_ || step < 0 ||
        options_.dense_start > std::numeric_limits<long long>::max() - step)
        return false;
    file_ = open_new_binary_file(options_.path);
    if (!file_) {
        std::perror("[velocity] cannot create new output file");
        failed_ = true;
        return false;
    }
    start_step_ = step;
    const long long dense = reference() ? 0 : options_.dense_start;
    sampling_.start_run(step, dense);
    velocity::FileHeader header{};
    std::memcpy(header.magic, "PFVMOM4", sizeof(header.magic));
    header.cells = static_cast<std::uint32_t>(cells_.size());
    header.columns = velocity::kColumns;
    header.dt = dt_;
    header.start_step = step;
    header.side = state.L;
    header.spatial_stride = reference() ? 1 : options_.spatial_stride;
    header.quadrature = velocity::kDiscreteLinearQuadrature;
    header.dense_start = static_cast<std::uint64_t>(dense);
    return write(&header, sizeof(header)) && capture(state, step, stream);
}

bool VelocityOutput::readback(const StepArgs& state, cudaStream_t stream) {
    const auto bytes = integrals_.size() * sizeof(velocity::MomentAccum);
    VCU(cudaMemsetAsync(endpoint_, 0, bytes, stream));
    launch_velocity_moments(state, endpoint_, dt_, stream);
    VCU(cudaGetLastError());
    VCU(cudaMemcpyAsync(cells_.data(), state.cell, cells_.size() * sizeof(CellState),
                        cudaMemcpyDeviceToHost, stream));
    VCU(cudaMemcpyAsync(integrals_.data(), moments_, bytes, cudaMemcpyDeviceToHost, stream));
    VCU(cudaMemcpyAsync(endpoints_.data(), endpoint_, bytes, cudaMemcpyDeviceToHost, stream));
    VCU(cudaStreamSynchronize(stream));
    return true;
}

bool VelocityOutput::pack_frame(long long step, int side) {
    using namespace velocity;
    for (std::size_t i = 0; i < cells_.size(); ++i) {
        const auto& cell = cells_[i];
        auto& moments = integrals_[i];
        if (!reference())
            finish_snapshot(moments, cell.advection, endpoints_[i], dt_);
        if (moments.count != static_cast<unsigned long long>(step - start_step_)) {
            std::fprintf(stderr, "[velocity] missed observation at step %lld, cell %d\n", step,
                         cell.global_id);
            return false;
        }
        double* row = rows_.data() + i * kColumns;
        row[0] = cell.global_id;
        row[1] = std::fmod(double(cell.gx0) + cell.Cx / cell.V, double(side));
        row[2] = std::fmod(double(cell.gy0) + cell.Cy / cell.V, double(side));
        row[3] = cell.theta;
        row[4] = cell.gamma;
        row[5] = cell.M_pf;
        row[6] = cell.v_A;
        row[7] = cell.V;
        row[8] = double(moments.count);
        for (int j = 0; j < kIntegralValues; ++j)
            row[kIntegralColumn + j] = moments.q[j];
        row[kSpatialCountColumn] = double(reference() ? moments.count : moments.spatial_count);
        for (int j = 0; j < kSpatialValues; ++j)
            row[kEndpointColumn + j] = endpoints_[i].q[kInterfacial + j] / dt_;
        for (std::uint32_t j = 0; j < kColumns; ++j) {
            if (!std::isfinite(row[j])) {
                std::fprintf(stderr, "[velocity] non-finite column %u at step %lld, cell %d\n",
                             j, step, cell.global_id);
                return false;
            }
        }
    }
    return true;
}

bool VelocityOutput::capture(const StepArgs& state, long long step, cudaStream_t stream) {
    if (!enabled())
        return true;
    if (!file_ || failed_ || step <= last_frame_ || state.N != static_cast<int>(cells_.size()))
        return false;
    if (!readback(state, stream) || !pack_frame(step, state.L)) {
        failed_ = true;
        return false;
    }
    const std::int64_t disk_step = step;
    if (!write(&disk_step, sizeof(disk_step)) ||
        !write(rows_.data(), rows_.size() * sizeof(double)))
        return false;
    if (std::fflush(file_) != 0 || std::ferror(file_)) {
        std::perror("[velocity] flush");
        failed_ = true;
        return false;
    }
    last_frame_ = step;
    return true;
}

bool VelocityOutput::close() {
    if (!file_)
        return !failed_;
    bool ok = std::fflush(file_) == 0 && !std::ferror(file_);
    if (std::fclose(file_) != 0)
        ok = false;
    file_ = nullptr;
    if (!ok) {
        std::perror("[velocity] close");
        failed_ = true;
    }
    return !failed_;
}
#undef VCU
} // namespace pf
