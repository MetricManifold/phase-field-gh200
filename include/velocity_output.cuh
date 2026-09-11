#pragma once
#include "kernels.cuh"
#include "velocity_sampling.hpp"
#include <cstdio>
#include <string>
#include <vector>

namespace pf {
struct VelocityOptions {
    std::string path;
    int spatial_stride = 100;
    long long dense_start = 10000;
    bool reference = false;
};

// Owns recorder allocations, sampling policy and the checked output stream.
// Sim only schedules captures; the numerical kernels own per-step updates.
class VelocityOutput {
  public:
    VelocityOutput() = default;
    ~VelocityOutput();
    VelocityOutput(const VelocityOutput&) = delete;
    VelocityOutput& operator=(const VelocityOutput&) = delete;

    bool configure(const VelocityOptions& options, int cells, double dt);
    bool open(const StepArgs& state, long long step, cudaStream_t stream);
    bool capture(const StepArgs& state, long long step, cudaStream_t stream);
    bool close();
    bool enabled() const { return moments_ != nullptr; }
    bool reference() const { return options_.reference; }
    velocity::MomentAccum* accumulators() const { return moments_; }
    long long last_frame() const { return last_frame_; }
    VelocitySampling& sampling() { return sampling_; }
    const VelocitySampling& sampling() const { return sampling_; }

  private:
    bool readback(const StepArgs& state, cudaStream_t stream);
    bool pack_frame(long long step, int side);
    bool write(const void* data, std::size_t bytes);

    VelocityOptions options_;
    VelocitySampling sampling_;
    double dt_ = 0;
    velocity::MomentAccum* moments_ = nullptr;
    velocity::MomentAccum* endpoint_ = nullptr;
    std::vector<CellState> cells_;
    std::vector<velocity::MomentAccum> integrals_, endpoints_;
    std::vector<double> rows_;
    std::FILE* file_ = nullptr;
    long long start_step_ = 0;
    long long last_frame_ = -1;
    bool failed_ = false;
};
} // namespace pf
