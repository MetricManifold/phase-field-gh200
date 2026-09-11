#pragma once
#include "kernels.cuh"

namespace pf {
// Independent dense oracle and read-only output endpoint pass. The update
// kernels use fused_moments.cuh; this pass deliberately shares no reductions.
void launch_velocity_moments(const StepArgs& state, velocity::MomentAccum* out,
                             double physical_dt, cudaStream_t stream);
} // namespace pf
