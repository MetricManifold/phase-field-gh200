#pragma once
#include <numeric>

namespace pf {
// Host-only scheduling policy. CUDA graph capture must depend on graph slots,
// never on transient startup state; normal launches additionally prime the
// observer and cover the dense startup interval.
class VelocitySampling {
  public:
    static constexpr int kPhysicsPeriod = 6;
    void configure(bool enabled, bool reference, int stride) {
        enabled_ = enabled;
        reference_ = reference;
        stride_ = stride;
        period_ = enabled && !reference ? std::lcm(kPhysicsPeriod, stride) : kPhysicsPeriod;
        // Split long periods at spatial samples. All phases together contain
        // the same nodes, but frequent output no longer breaks a long replay.
        graph_steps_ = enabled && !reference && stride >= kPhysicsPeriod ? stride : period_;
    }
    void start_run(long long step, long long dense_steps) { dense_until_ = step + dense_steps; }
    int graph_period() const { return period_; }
    int graph_steps() const { return graph_steps_; }
    int graph_count() const { return period_ / graph_steps_; }
    int graph_index(long long step) const { return int(step % period_) / graph_steps_; }
    bool can_replay(long long step) const {
        return !enabled_ || reference_ || (primed_ && step >= dense_until_);
    }
    bool spatial_sample(int slot, long long step, bool capture) const {
        return slot % stride_ == 0 || (!capture && (!primed_ || step < dense_until_));
    }
    void advanced() { primed_ = true; }

  private:
    bool enabled_ = false;
    bool reference_ = false;
    bool primed_ = false;
    int stride_ = 1;
    int period_ = kPhysicsPeriod;
    int graph_steps_ = kPhysicsPeriod;
    long long dense_until_ = 0;
};
} // namespace pf
