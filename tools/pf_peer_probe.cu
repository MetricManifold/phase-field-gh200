// Standalone two-GPU diagnostic for the communication proposed by the
// substrate-slab planner. It measures the peer link and a local checked merge;
// it does not run the simulator or read scientific output.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pf3d/substrate_planner.hpp"

namespace {

constexpr std::uint32_t kPatternA = 0xA5A5A5A5u;
constexpr std::uint32_t kPatternB = 0x5A5A5A5Au;
constexpr int kPitchWords = 928;
constexpr int kDomainRows = 916;
constexpr int kDomainPlanes = 288;
constexpr std::uint32_t kBaseEdge = 152;

struct Options {
    int device_a = 0;
    int device_b = 1;
    int iterations = 10;
    int warmup = 3;
    bool direct_bands = true;
    bool checked_add = true;
};

struct Timing {
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double effective_bandwidth_gib_s = 0.0;
    double median_ab_ms = 0.0;
    double median_ba_ms = 0.0;
    bool has_direction_medians = false;
    bool host_wall_clock = false;
    bool verified = false;
    bool ok = false;
    std::string failure;
};

struct Result {
    const char* label = nullptr;
    std::size_t bytes = 0;
    Timing timing{};
};

struct DeviceResources {
    int device = -1;
    void* send = nullptr;
    void* receive = nullptr;
    std::uint32_t* flag = nullptr;
    cudaStream_t stream = nullptr;
};

bool record(cudaError_t error, const char* action, std::string* failure) {
    if (error == cudaSuccess) return true;
    if (failure != nullptr && failure->empty()) {
        *failure = std::string(action) + ": " + cudaGetErrorString(error);
    }
    return false;
}

void release(DeviceResources* resources) {
    if (resources == nullptr || resources->device < 0) return;
    cudaSetDevice(resources->device);
    if (resources->stream != nullptr) cudaStreamDestroy(resources->stream);
    if (resources->flag != nullptr) cudaFree(resources->flag);
    if (resources->send != nullptr) cudaFree(resources->send);
    if (resources->receive != nullptr) cudaFree(resources->receive);
    *resources = {};
}

bool allocate(DeviceResources* resources, int device, std::size_t bytes,
              std::string* failure) {
    if (resources == nullptr) return false;
    resources->device = device;
    return record(cudaSetDevice(device), "cudaSetDevice", failure) &&
           record(cudaMalloc(&resources->send, bytes),
                  "cudaMalloc(send)", failure) &&
           record(cudaMalloc(&resources->receive, bytes),
                  "cudaMalloc(receive)", failure) &&
           record(cudaMalloc(&resources->flag, sizeof(std::uint32_t)),
                  "cudaMalloc(flag)", failure) &&
           record(cudaStreamCreateWithFlags(&resources->stream,
                                             cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags", failure);
}

bool fill(DeviceResources* resources, int send_byte, std::size_t bytes,
          std::string* failure) {
    return record(cudaSetDevice(resources->device), "cudaSetDevice(fill)",
                  failure) &&
           record(cudaMemsetAsync(resources->send, send_byte, bytes,
                                  resources->stream),
                  "cudaMemsetAsync(send)", failure) &&
           record(cudaMemsetAsync(resources->receive, 0, bytes,
                                  resources->stream),
                  "cudaMemsetAsync(receive)", failure) &&
           record(cudaStreamSynchronize(resources->stream),
                  "cudaStreamSynchronize(fill)", failure);
}

bool clear_receive(DeviceResources* resources, std::size_t bytes,
                   std::string* failure) {
    return record(cudaSetDevice(resources->device),
                  "cudaSetDevice(clear receive)", failure) &&
           record(cudaMemsetAsync(resources->receive, 0, bytes,
                                  resources->stream),
                  "cudaMemsetAsync(clear receive)", failure) &&
           record(cudaStreamSynchronize(resources->stream),
                  "cudaStreamSynchronize(clear receive)", failure);
}

bool enable_peer_access(int device, int peer, std::string* failure) {
    if (!record(cudaSetDevice(device), "cudaSetDevice(peer)", failure))
        return false;
    const cudaError_t error = cudaDeviceEnablePeerAccess(peer, 0);
    if (error == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return true;
    }
    return record(error, "cudaDeviceEnablePeerAccess", failure);
}

double median(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    return n % 2 == 0 ? 0.5 * (samples[n / 2 - 1] + samples[n / 2])
                      : samples[n / 2];
}

double nearest_rank_p95(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(samples.size())));
    return samples[rank == 0 ? 0 : rank - 1];
}

Timing summarize(const std::vector<double>& samples, std::size_t bytes) {
    Timing timing{};
    if (samples.empty()) {
        timing.failure = "no timing samples";
        return timing;
    }
    timing.median_ms = median(samples);
    timing.p95_ms = nearest_rank_p95(samples);
    if (timing.median_ms <= 0.0) {
        timing.failure = "non-positive median duration";
        return timing;
    }
    const double gib = static_cast<double>(bytes) /
                       (1024.0 * 1024.0 * 1024.0);
    timing.effective_bandwidth_gib_s = gib / (timing.median_ms / 1000.0);
    timing.ok = true;
    return timing;
}

template <class Launch, class Verify>
Timing time_single_device(int device, cudaStream_t stream, int warmup,
                          int iterations, std::size_t bytes, Launch launch,
                          Verify verify) {
    Timing timing{};
    if (!record(cudaSetDevice(device), "cudaSetDevice(timing)",
                &timing.failure)) {
        return timing;
    }
    for (int i = 0; i < warmup; ++i) {
        if (!record(launch(), "warmup launch", &timing.failure)) return timing;
    }
    if (!record(cudaStreamSynchronize(stream), "warmup synchronize",
                &timing.failure)) {
        return timing;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!record(cudaEventCreate(&start), "cudaEventCreate(start)",
                &timing.failure) ||
        !record(cudaEventCreate(&stop), "cudaEventCreate(stop)",
                &timing.failure)) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        return timing;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        float milliseconds = 0.0f;
        if (!record(cudaEventRecord(start, stream), "cudaEventRecord(start)",
                    &timing.failure) ||
            !record(launch(), "timed launch", &timing.failure) ||
            !record(cudaEventRecord(stop, stream), "cudaEventRecord(stop)",
                    &timing.failure) ||
            !record(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)",
                    &timing.failure) ||
            !record(cudaEventElapsedTime(&milliseconds, start, stop),
                    "cudaEventElapsedTime", &timing.failure)) {
            break;
        }
        samples.push_back(milliseconds);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (samples.size() != static_cast<std::size_t>(iterations)) return timing;
    timing = summarize(samples, bytes);
    if (!timing.ok) return timing;
    timing.verified = verify(&timing.failure);
    timing.ok = timing.verified;
    return timing;
}

__global__ void check_uniform_u32(const std::uint32_t* values,
                                  std::size_t count,
                                  std::uint32_t expected,
                                  std::uint32_t* mismatch) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && values[index] != expected) atomicOr(mismatch, 1u);
}

bool verify_uniform(DeviceResources* resources, const void* values,
                    std::size_t bytes, std::uint32_t expected,
                    std::string* failure) {
    if (bytes % sizeof(std::uint32_t) != 0) {
        if (failure != nullptr) *failure = "payload is not uint32-aligned";
        return false;
    }
    if (!record(cudaSetDevice(resources->device), "cudaSetDevice(verify)",
                failure) ||
        !record(cudaMemsetAsync(resources->flag, 0, sizeof(std::uint32_t),
                                resources->stream),
                "cudaMemsetAsync(verify flag)", failure)) {
        return false;
    }
    const std::size_t count = bytes / sizeof(std::uint32_t);
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    check_uniform_u32<<<grid, block, 0, resources->stream>>>(
        static_cast<const std::uint32_t*>(values), count, expected,
        resources->flag);
    if (!record(cudaGetLastError(), "check_uniform_u32 launch", failure))
        return false;
    std::uint32_t mismatch = 0;
    if (!record(cudaMemcpyAsync(&mismatch, resources->flag,
                                sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                resources->stream),
                "cudaMemcpyAsync(verify flag)", failure) ||
        !record(cudaStreamSynchronize(resources->stream),
                "cudaStreamSynchronize(verify)", failure)) {
        return false;
    }
    if (mismatch != 0) {
        if (failure != nullptr) *failure = "payload verification failed";
        return false;
    }
    return true;
}

Timing time_flat_copy(DeviceResources* source, DeviceResources* destination,
                      std::size_t bytes, std::uint32_t expected,
                      int warmup, int iterations) {
    Timing failure_timing{};
    if (!clear_receive(destination, bytes, &failure_timing.failure))
        return failure_timing;
    auto launch = [&]() {
        return cudaMemcpyPeerAsync(destination->receive, destination->device,
                                   source->send, source->device, bytes,
                                   source->stream);
    };
    auto verify = [&](std::string* failure) {
        return verify_uniform(destination, destination->receive, bytes,
                              expected, failure);
    };
    return time_single_device(source->device, source->stream, warmup,
                              iterations, bytes, launch, verify);
}

struct BandCase {
    const char* label;
    std::array<std::array<int, 2>, 3> ranges;
    int range_count;
    int z_lo;
    int z_hi;
    std::size_t bytes;
};

enum ProbeBandIndex : std::size_t {
    kBaseInitialBand,
    kBaseElevatedBand,
    kPromoted224Band,
    kPromoted280Band,
    kBaseFullZBand,
    kProbeBandCount
};

pf3d::TwoRankSubstrateGeometry probe_geometry() {
    return {kDomainRows, kDomainRows, kDomainPlanes, kPitchWords,
            pf3d::kBoundarySubstrateSlab3D};
}

bool make_band_case(const char* label, const pf3d::SeamExchangePlan& plan,
                    BandCase* band, std::string* failure) {
    if (band == nullptr || plan.status != pf3d::SeamPlanStatus::Ok ||
        plan.bands.rows.count < 1 || plan.bands.rows.count > 3 ||
        pf3d::empty(plan.z) || plan.payload_bytes == 0) {
        if (failure != nullptr) *failure = "invalid planner benchmark case";
        return false;
    }
    *band = {};
    band->label = label;
    band->range_count = plan.bands.rows.count;
    for (int i = 0; i < band->range_count; ++i) {
        band->ranges[i] = {plan.bands.rows.pieces[i].lo,
                           plan.bands.rows.pieces[i].hi};
    }
    band->z_lo = plan.z.lo;
    band->z_hi = plan.z.hi;
    band->bytes = plan.payload_bytes;
    return true;
}

bool build_band_cases(std::vector<BandCase>* bands, std::string* failure) {
    // These are representative states for one fixed scenario, not universal
    // bounds on the communication required by every substrate run.
    if (bands == nullptr) return false;
    bands->clear();
    bands->reserve(kProbeBandCount);
    auto add_cell_case = [&](const char* label, std::int64_t origin_y,
                             std::int64_t origin_z,
                             std::uint32_t storage_edge) {
        pf3d::AllocatedBrick cell{};
        cell.origin_y = origin_y;
        cell.origin_z = origin_z;
        cell.storage_edge = storage_edge;
        const pf3d::SeamExchangePlan plan = pf3d::build_seam_exchange_plan(
            probe_geometry(), kBaseEdge, 0, &cell, 1,
            sizeof(std::uint32_t));
        BandCase band{};
        if (!make_band_case(label, plan, &band, failure)) return false;
        bands->push_back(band);
        return true;
    };

    // The elevated case exercises a pitched copy whose z interval starts above 0.
    if (!add_cell_case("base_initial", 300, -76, 0) ||
        !add_cell_case("base_elevated", 300, 37, 0) ||
        !add_cell_case("promoted_e224", 300, -112, 224) ||
        !add_cell_case("promoted_e280", 300, -140, 280)) {
        return false;
    }

    BandCase full_z = (*bands)[kBaseInitialBand];
    // Conservative comparison point when dynamic z cropping is unavailable.
    full_z.label = "base_full_z_fallback";
    full_z.z_lo = 0;
    full_z.z_hi = kDomainPlanes;
    int full_z_rows = 0;
    for (int i = 0; i < full_z.range_count; ++i)
        full_z_rows += full_z.ranges[i][1] - full_z.ranges[i][0];
    if (!pf3d::pitched_payload_bytes(
            full_z_rows, kPitchWords, kDomainPlanes,
            sizeof(std::uint32_t), &full_z.bytes)) {
        if (failure != nullptr) *failure = "full-z payload size overflow";
        return false;
    }
    bands->push_back(full_z);
    return bands->size() == kProbeBandCount;
}

bool valid_band_case(const BandCase& band) {
    if (band.range_count < 1 || band.range_count > 3 ||
        band.z_lo < 0 || band.z_hi <= band.z_lo ||
        band.z_hi > kDomainPlanes) {
        return false;
    }
    std::size_t rows = 0;
    int previous_hi = 0;
    for (int i = 0; i < band.range_count; ++i) {
        const int lo = band.ranges[i][0];
        const int hi = band.ranges[i][1];
        if (lo < previous_hi || hi <= lo || hi > kDomainRows) return false;
        rows += static_cast<std::size_t>(hi - lo);
        previous_hi = hi;
    }
    const std::size_t expected =
        rows * kPitchWords *
        static_cast<std::size_t>(band.z_hi - band.z_lo) *
        sizeof(std::uint32_t);
    return expected == band.bytes;
}

cudaError_t copy_bands(DeviceResources* source,
                       DeviceResources* destination, const BandCase& band);
bool verify_bands(DeviceResources* destination, const BandCase& band,
                  std::uint32_t expected, std::string* failure);

class StartGate {
  public:
    using Clock = std::chrono::steady_clock;

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++ready_;
        condition_.notify_all();
        condition_.wait(lock, [&]() { return released_; });
    }

    Clock::time_point release_when_ready() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&]() { return ready_ == 2; });
        const Clock::time_point released = Clock::now();
        released_ = true;
        condition_.notify_all();
        return released;
    }

  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    int ready_ = 0;
    bool released_ = false;
};

struct DuplexSide {
    float milliseconds = 0.0f;
    std::string failure;
};

void duplex_side(StartGate* gate, DeviceResources* source,
                 DeviceResources* destination, std::size_t bytes,
                 const BandCase* band,
                 cudaEvent_t start, cudaEvent_t stop, DuplexSide* result) {
    const bool timed = start != nullptr && stop != nullptr;
    const bool device_ok = record(cudaSetDevice(source->device),
                                  "cudaSetDevice(duplex)", &result->failure);
    gate->arrive_and_wait();
    if (!device_ok) return;
    if (timed &&
        !record(cudaEventRecord(start, source->stream),
                "cudaEventRecord(duplex start)", &result->failure)) {
        return;
    }
    const cudaError_t copy_error = band == nullptr
        ? cudaMemcpyPeerAsync(destination->receive, destination->device,
                              source->send, source->device, bytes,
                              source->stream)
        : copy_bands(source, destination, *band);
    if (!record(copy_error, band == nullptr ? "cudaMemcpyPeerAsync(duplex)"
                                            : "copy_bands(duplex)",
                &result->failure)) {
        return;
    }
    if (timed &&
        !record(cudaEventRecord(stop, source->stream),
                "cudaEventRecord(duplex stop)", &result->failure)) {
        return;
    }
    if (!record(cudaStreamSynchronize(source->stream),
                "cudaStreamSynchronize(duplex)", &result->failure)) {
        return;
    }
    if (timed) {
        record(cudaEventElapsedTime(&result->milliseconds, start, stop),
               "cudaEventElapsedTime(duplex)", &result->failure);
    }
}

bool run_duplex_once(DeviceResources* a, DeviceResources* b,
                     std::size_t bytes, const BandCase* band,
                     cudaEvent_t start_a,
                     cudaEvent_t stop_a, cudaEvent_t start_b,
                     cudaEvent_t stop_b, DuplexSide* ab, DuplexSide* ba,
                     double* wall_milliseconds) {
    StartGate gate;
    std::thread thread_ab(duplex_side, &gate, a, b, bytes, band,
                          start_a, stop_a, ab);
    std::thread thread_ba(duplex_side, &gate, b, a, bytes, band,
                          start_b, stop_b, ba);
    // CUDA events from different devices are not a shared clock. The common
    // host interval is the aggregate bidirectional duration.
    const auto released = gate.release_when_ready();
    thread_ab.join();
    thread_ba.join();
    if (wall_milliseconds != nullptr) {
        *wall_milliseconds = std::chrono::duration<double, std::milli>(
            StartGate::Clock::now() - released).count();
    }
    return ab->failure.empty() && ba->failure.empty();
}

Timing time_bidirectional(DeviceResources* a, DeviceResources* b,
                          std::size_t bytes, int warmup, int iterations,
                          const BandCase* band = nullptr,
                          std::size_t allocation_bytes = 0) {
    Timing timing{};
    const std::size_t clear_bytes = band == nullptr ? bytes : allocation_bytes;
    if (!clear_receive(a, clear_bytes, &timing.failure) ||
        !clear_receive(b, clear_bytes, &timing.failure)) {
        return timing;
    }
    for (int i = 0; i < warmup; ++i) {
        DuplexSide ab{};
        DuplexSide ba{};
        if (!run_duplex_once(a, b, bytes, band, nullptr, nullptr, nullptr,
                             nullptr, &ab, &ba, nullptr)) {
            timing.failure = !ab.failure.empty() ? "a_to_b: " + ab.failure
                                                 : "b_to_a: " + ba.failure;
            return timing;
        }
    }

    cudaEvent_t start_a = nullptr;
    cudaEvent_t stop_a = nullptr;
    cudaEvent_t start_b = nullptr;
    cudaEvent_t stop_b = nullptr;
    auto create_pair = [&](DeviceResources* resources, cudaEvent_t* start,
                           cudaEvent_t* stop) {
        return record(cudaSetDevice(resources->device),
                      "cudaSetDevice(events)", &timing.failure) &&
               record(cudaEventCreate(start), "cudaEventCreate(start)",
                      &timing.failure) &&
               record(cudaEventCreate(stop), "cudaEventCreate(stop)",
                      &timing.failure);
    };
    if (!create_pair(a, &start_a, &stop_a) ||
        !create_pair(b, &start_b, &stop_b)) {
        if (start_a) { cudaSetDevice(a->device); cudaEventDestroy(start_a); }
        if (stop_a) { cudaSetDevice(a->device); cudaEventDestroy(stop_a); }
        if (start_b) { cudaSetDevice(b->device); cudaEventDestroy(start_b); }
        if (stop_b) { cudaSetDevice(b->device); cudaEventDestroy(stop_b); }
        return timing;
    }

    std::vector<double> durations;
    std::vector<double> durations_ab;
    std::vector<double> durations_ba;
    durations.reserve(static_cast<std::size_t>(iterations));
    durations_ab.reserve(static_cast<std::size_t>(iterations));
    durations_ba.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        DuplexSide ab{};
        DuplexSide ba{};
        double wall_milliseconds = 0.0;
        if (!run_duplex_once(a, b, bytes, band, start_a, stop_a, start_b,
                             stop_b, &ab, &ba, &wall_milliseconds)) {
            timing.failure = !ab.failure.empty() ? "a_to_b: " + ab.failure
                                                 : "b_to_a: " + ba.failure;
            break;
        }
        durations_ab.push_back(ab.milliseconds);
        durations_ba.push_back(ba.milliseconds);
        durations.push_back(wall_milliseconds);
    }
    cudaSetDevice(a->device);
    cudaEventDestroy(start_a);
    cudaEventDestroy(stop_a);
    cudaSetDevice(b->device);
    cudaEventDestroy(start_b);
    cudaEventDestroy(stop_b);
    if (durations.size() != static_cast<std::size_t>(iterations)) return timing;

    timing = summarize(durations, 2 * bytes);
    timing.host_wall_clock = true;
    timing.has_direction_medians = true;
    timing.median_ab_ms = median(durations_ab);
    timing.median_ba_ms = median(durations_ba);
    if (!timing.ok) return timing;
    std::string verify_failure;
    const bool verified_ab = band == nullptr
        ? verify_uniform(b, b->receive, bytes, kPatternA, &verify_failure)
        : verify_bands(b, *band, kPatternA, &verify_failure);
    const bool verified_ba = verified_ab && (band == nullptr
        ? verify_uniform(a, a->receive, bytes, kPatternB, &verify_failure)
        : verify_bands(a, *band, kPatternB, &verify_failure));
    timing.verified = verified_ab && verified_ba;
    timing.ok = timing.verified;
    if (!timing.ok) timing.failure = verify_failure;
    return timing;
}

cudaError_t copy_bands(DeviceResources* source, DeviceResources* destination,
                       const BandCase& band) {
    // Complete x rows remain contiguous; each canonical y interval is one
    // direct pitched peer copy over the planner's active z interval.
    for (int i = 0; i < band.range_count; ++i) {
        const int lo = band.ranges[i][0];
        const int rows = band.ranges[i][1] - lo;
        cudaMemcpy3DPeerParms parameters{};
        parameters.srcPtr = make_cudaPitchedPtr(
            source->send, kPitchWords * sizeof(std::uint32_t),
            kPitchWords * sizeof(std::uint32_t), kDomainRows);
        parameters.srcPos = make_cudaPos(0, lo, band.z_lo);
        parameters.srcDevice = source->device;
        parameters.dstPtr = make_cudaPitchedPtr(
            destination->receive, kPitchWords * sizeof(std::uint32_t),
            kPitchWords * sizeof(std::uint32_t), kDomainRows);
        parameters.dstPos = make_cudaPos(0, lo, band.z_lo);
        parameters.dstDevice = destination->device;
        parameters.extent = make_cudaExtent(
            kPitchWords * sizeof(std::uint32_t), rows,
            band.z_hi - band.z_lo);
        const cudaError_t error =
            cudaMemcpy3DPeerAsync(&parameters, source->stream);
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}

__global__ void check_band_layout_u32(
    const std::uint32_t* values, int pitch_words, int domain_rows,
    int domain_planes, int2 range0, int2 range1, int2 range2,
    int range_count, int copied_z_lo, int copied_z_hi,
    std::uint32_t copied_value,
    std::uint32_t* mismatch) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t plane_words =
        static_cast<std::size_t>(domain_rows) * pitch_words;
    const std::size_t count = plane_words * domain_planes;
    if (index >= count) return;
    const int z = static_cast<int>(index / plane_words);
    const int y = static_cast<int>((index % plane_words) / pitch_words);
    const bool in_range =
        (range_count > 0 && y >= range0.x && y < range0.y) ||
        (range_count > 1 && y >= range1.x && y < range1.y) ||
        (range_count > 2 && y >= range2.x && y < range2.y);
    const std::uint32_t expected =
        z >= copied_z_lo && z < copied_z_hi && in_range ? copied_value : 0u;
    if (values[index] != expected) atomicOr(mismatch, 1u);
}

bool verify_bands(DeviceResources* destination, const BandCase& band,
                  std::uint32_t expected, std::string* failure) {
    if (!record(cudaSetDevice(destination->device),
                "cudaSetDevice(verify bands)", failure) ||
        !record(cudaMemsetAsync(destination->flag, 0, sizeof(std::uint32_t),
                                destination->stream),
                "cudaMemsetAsync(band flag)", failure)) {
        return false;
    }
    const std::size_t count = static_cast<std::size_t>(kPitchWords) *
                              kDomainRows * kDomainPlanes;
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    const int2 range0 = make_int2(band.ranges[0][0], band.ranges[0][1]);
    const int2 range1 = make_int2(band.ranges[1][0], band.ranges[1][1]);
    const int2 range2 = make_int2(band.ranges[2][0], band.ranges[2][1]);
    check_band_layout_u32<<<grid, block, 0, destination->stream>>>(
        static_cast<const std::uint32_t*>(destination->receive), kPitchWords,
        kDomainRows, kDomainPlanes, range0, range1, range2, band.range_count,
        band.z_lo, band.z_hi, expected, destination->flag);
    if (!record(cudaGetLastError(), "check_band_layout_u32 launch", failure))
        return false;
    std::uint32_t mismatch = 0;
    if (!record(cudaMemcpyAsync(&mismatch, destination->flag,
                                sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                destination->stream),
                "cudaMemcpyAsync(band flag)", failure) ||
        !record(cudaStreamSynchronize(destination->stream),
                "cudaStreamSynchronize(verify bands)", failure)) {
        return false;
    }
    if (mismatch != 0) {
        if (failure != nullptr) *failure = "band verification failed";
        return false;
    }
    return true;
}

Timing time_direct_band(DeviceResources* source,
                        DeviceResources* destination, const BandCase& band,
                        std::size_t allocation_bytes, int warmup,
                        int iterations) {
    Timing timing{};
    if (!clear_receive(destination, allocation_bytes, &timing.failure)) {
        return timing;
    }
    auto launch = [&]() { return copy_bands(source, destination, band); };
    auto verify = [&](std::string* failure) {
        return verify_bands(destination, band, kPatternA, failure);
    };
    return time_single_device(source->device, source->stream, warmup,
                              iterations, band.bytes, launch, verify);
}

__global__ void add_checked_u32(const std::uint32_t* source,
                                std::uint32_t* destination,
                                std::size_t count,
                                std::uint32_t* overflow) {
    // Q5.27 interaction fields add as uint32 words; wrap is a fatal condition.
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t previous = destination[index];
    const std::uint32_t sum = previous + source[index];
    if (sum < previous) atomicOr(overflow, 1u);
    destination[index] = sum;
}

bool checked_add_self_test(DeviceResources* resources, std::string* failure) {
    if (!record(cudaSetDevice(resources->device), "cudaSetDevice(add test)",
                failure) ||
        !record(cudaMemsetAsync(resources->send, 0xFF,
                                sizeof(std::uint32_t), resources->stream),
                "cudaMemsetAsync(add source)", failure) ||
        !record(cudaMemsetAsync(resources->receive, 0x01,
                                sizeof(std::uint32_t), resources->stream),
                "cudaMemsetAsync(add destination)", failure) ||
        !record(cudaMemsetAsync(resources->flag, 0, sizeof(std::uint32_t),
                                resources->stream),
                "cudaMemsetAsync(add flag)", failure)) {
        return false;
    }
    add_checked_u32<<<1, 1, 0, resources->stream>>>(
        static_cast<const std::uint32_t*>(resources->send),
        static_cast<std::uint32_t*>(resources->receive), 1, resources->flag);
    if (!record(cudaGetLastError(), "add_checked_u32 self-test launch", failure))
        return false;
    std::uint32_t overflow = 0;
    if (!record(cudaMemcpyAsync(&overflow, resources->flag,
                                sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                resources->stream),
                "cudaMemcpyAsync(add flag)", failure) ||
        !record(cudaStreamSynchronize(resources->stream),
                "cudaStreamSynchronize(add test)", failure)) {
        return false;
    }
    if (overflow != 1) {
        if (failure != nullptr) *failure = "checked add missed an overflow";
        return false;
    }
    return true;
}

Timing time_checked_add(DeviceResources* resources, std::size_t bytes,
                        int warmup, int iterations) {
    Timing timing{};
    if (!checked_add_self_test(resources, &timing.failure) ||
        !fill(resources, 0x01, bytes, &timing.failure)) {
        return timing;
    }
    const std::size_t words = bytes / sizeof(std::uint32_t);
    const int block = 256;
    const int grid = static_cast<int>((words + block - 1) / block);
    auto prepare = [&]() {
        return record(cudaMemsetAsync(resources->receive, 0, bytes,
                                      resources->stream),
                      "cudaMemsetAsync(add destination)", &timing.failure) &&
               record(cudaMemsetAsync(resources->flag, 0,
                                      sizeof(std::uint32_t),
                                      resources->stream),
                      "cudaMemsetAsync(add flag)", &timing.failure);
    };
    auto launch = [&]() {
        add_checked_u32<<<grid, block, 0, resources->stream>>>(
            static_cast<const std::uint32_t*>(resources->send),
            static_cast<std::uint32_t*>(resources->receive), words,
            resources->flag);
        return cudaGetLastError();
    };

    if (!record(cudaSetDevice(resources->device), "cudaSetDevice(add timing)",
                &timing.failure)) {
        return timing;
    }
    for (int i = 0; i < warmup; ++i) {
        if (!prepare() || !record(launch(), "checked add warmup",
                                  &timing.failure) ||
            !record(cudaStreamSynchronize(resources->stream),
                    "checked add warmup synchronize", &timing.failure)) {
            return timing;
        }
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!record(cudaEventCreate(&start), "cudaEventCreate(add start)",
                &timing.failure) ||
        !record(cudaEventCreate(&stop), "cudaEventCreate(add stop)",
                &timing.failure)) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        return timing;
    }
    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        float milliseconds = 0.0f;
        if (!prepare() ||
            !record(cudaEventRecord(start, resources->stream),
                    "cudaEventRecord(add start)", &timing.failure) ||
            !record(launch(), "checked add launch", &timing.failure) ||
            !record(cudaEventRecord(stop, resources->stream),
                    "cudaEventRecord(add stop)", &timing.failure) ||
            !record(cudaEventSynchronize(stop),
                    "cudaEventSynchronize(add stop)", &timing.failure) ||
            !record(cudaEventElapsedTime(&milliseconds, start, stop),
                    "cudaEventElapsedTime(add)", &timing.failure)) {
            break;
        }
        samples.push_back(milliseconds);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (samples.size() != static_cast<std::size_t>(iterations)) return timing;
    timing = summarize(samples, bytes);
    if (!timing.ok) return timing;
    std::uint32_t overflow = 0;
    if (!record(cudaMemcpy(&overflow, resources->flag, sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(add flag)", &timing.failure) ||
        overflow != 0) {
        if (timing.failure.empty()) timing.failure = "unexpected add overflow";
        timing.ok = false;
        return timing;
    }
    timing.verified = verify_uniform(resources, resources->receive, bytes,
                                     0x01010101u, &timing.failure);
    timing.ok = timing.verified;
    return timing;
}

bool parse_integer(const char* text, int* value) {
    if (text == nullptr || value == nullptr) return false;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

void print_usage(const char* program) {
    std::fprintf(stderr,
        "usage: %s [--device-a ID] [--device-b ID] [--iterations N] "
        "[--warmup N] [--no-direct-bands] [--no-checked-add]\n",
        program);
}

bool parse_options(int argc, char** argv, Options* options,
                   std::string* failure, bool* help) {
    *help = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto take_integer = [&](int* destination) {
            if (index + 1 >= argc ||
                !parse_integer(argv[index + 1], destination)) {
                *failure = argument + " expects an integer";
                return false;
            }
            ++index;
            return true;
        };
        if (argument == "--device-a") {
            if (!take_integer(&options->device_a)) return false;
        } else if (argument == "--device-b") {
            if (!take_integer(&options->device_b)) return false;
        } else if (argument == "--iterations") {
            if (!take_integer(&options->iterations)) return false;
        } else if (argument == "--warmup") {
            if (!take_integer(&options->warmup)) return false;
        } else if (argument == "--no-direct-bands") {
            options->direct_bands = false;
        } else if (argument == "--no-checked-add") {
            options->checked_add = false;
        } else if (argument == "--help" || argument == "-h") {
            *help = true;
            return true;
        } else {
            *failure = "unknown argument: " + argument;
            return false;
        }
    }
    if (options->iterations < 1 || options->iterations > 1024 ||
        options->warmup < 0 || options->warmup > 1024) {
        *failure = "iterations must be in [1,1024] and warmup in [0,1024]";
        return false;
    }
    return true;
}

void print_json_string(const char* key, const std::string& value) {
    std::printf("\"%s\":\"", key);
    for (unsigned char character : value) {
        switch (character) {
            case '"': std::fputs("\\\"", stdout); break;
            case '\\': std::fputs("\\\\", stdout); break;
            case '\b': std::fputs("\\b", stdout); break;
            case '\f': std::fputs("\\f", stdout); break;
            case '\n': std::fputs("\\n", stdout); break;
            case '\r': std::fputs("\\r", stdout); break;
            case '\t': std::fputs("\\t", stdout); break;
            default:
                if (character < 0x20)
                    std::printf("\\u%04x", static_cast<unsigned>(character));
                else
                    std::fputc(character, stdout);
        }
    }
    std::fputc('"', stdout);
}

int fail_json(int exit_code, const std::string& failure) {
    std::printf("{\"tool\":\"pf_peer_probe\",\"ok\":false,");
    print_json_string("failure", failure);
    std::printf("}\n");
    return exit_code;
}

void print_result(const Result& result) {
    std::printf("{");
    print_json_string("label", result.label);
    std::printf(",\"bytes\":%zu,\"ok\":%s,\"verified\":%s",
                result.bytes, result.timing.ok ? "true" : "false",
                result.timing.verified ? "true" : "false");
    std::printf(",\"timing_clock\":\"%s\"",
                result.timing.host_wall_clock ? "host_wall" : "cuda_event");
    if (result.timing.ok) {
        std::printf(",\"median_ms\":%.6f,\"p95_ms\":%.6f,"
                    "\"effective_payload_gib_s\":%.6f",
                    result.timing.median_ms, result.timing.p95_ms,
                    result.timing.effective_bandwidth_gib_s);
        if (result.timing.has_direction_medians) {
            std::printf(",\"median_a_to_b_ms\":%.6f,"
                        "\"median_b_to_a_ms\":%.6f",
                        result.timing.median_ab_ms,
                        result.timing.median_ba_ms);
        }
    } else {
        std::printf(",");
        print_json_string("failure", result.timing.failure);
    }
    std::printf("}");
}

void print_results(const char* key, const std::vector<Result>& results) {
    std::printf(",\"%s\":[", key);
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i != 0) std::printf(",");
        print_result(results[i]);
    }
    std::printf("]");
}

std::string device_uuid(const cudaDeviceProp& properties) {
    char text[33]{};
    for (int i = 0; i < 16; ++i) {
        std::snprintf(text + 2 * i, sizeof(text) - 2 * i, "%02x",
                      static_cast<unsigned>(static_cast<unsigned char>(
                          properties.uuid.bytes[i])));
    }
    return text;
}

int run(int argc, char** argv) {
    Options options{};
    std::string failure;
    bool help = false;
    if (!parse_options(argc, argv, &options, &failure, &help)) {
        print_usage(argv[0]);
        return fail_json(2, failure);
    }
    if (help) {
        print_usage(argv[0]);
        return 0;
    }

    int device_count = 0;
    if (!record(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount",
                &failure)) {
        return fail_json(3, failure);
    }
    if (device_count < 2)
        return fail_json(3, "fewer than two CUDA devices are visible");
    if (options.device_a == options.device_b || options.device_a < 0 ||
        options.device_b < 0 || options.device_a >= device_count ||
        options.device_b >= device_count) {
        return fail_json(2, "device IDs must select two distinct visible GPUs");
    }

    int can_ab = 0;
    int can_ba = 0;
    if (!record(cudaDeviceCanAccessPeer(&can_ab, options.device_a,
                                        options.device_b),
                "cudaDeviceCanAccessPeer(a,b)", &failure) ||
        !record(cudaDeviceCanAccessPeer(&can_ba, options.device_b,
                                        options.device_a),
                "cudaDeviceCanAccessPeer(b,a)", &failure)) {
        return fail_json(4, failure);
    }
    if (can_ab == 0 || can_ba == 0)
        return fail_json(4, "bidirectional CUDA peer access is unavailable");
    if (!enable_peer_access(options.device_a, options.device_b, &failure) ||
        !enable_peer_access(options.device_b, options.device_a, &failure)) {
        return fail_json(4, failure);
    }

    cudaDeviceProp properties_a{};
    cudaDeviceProp properties_b{};
    if (!record(cudaGetDeviceProperties(&properties_a, options.device_a),
                "cudaGetDeviceProperties(a)", &failure) ||
        !record(cudaGetDeviceProperties(&properties_b, options.device_b),
                "cudaGetDeviceProperties(b)", &failure)) {
        return fail_json(4, failure);
    }

    std::vector<BandCase> band_cases;
    if (!build_band_cases(&band_cases, &failure)) return fail_json(5, failure);
    for (const BandCase& band : band_cases) {
        if (!valid_band_case(band))
            return fail_json(5, "internal y-band geometry is inconsistent");
    }

    std::size_t full_s_bytes = 0;
    if (!pf3d::pitched_payload_bytes(kDomainRows, kPitchWords, kDomainPlanes,
                                     sizeof(std::uint32_t), &full_s_bytes)) {
        return fail_json(5, "full interaction-field size overflow");
    }

    struct Payload { const char* label; std::size_t bytes; };
    const std::array<Payload, 7> payloads{{
        {"small_3_mib", 3ULL * 1024ULL * 1024ULL},
        {"base_initial", band_cases[kBaseInitialBand].bytes},
        {"evolved_base_100_mib", 100ULL * 1024ULL * 1024ULL},
        {"promoted_e224", band_cases[kPromoted224Band].bytes},
        {"promoted_e280", band_cases[kPromoted280Band].bytes},
        {"base_full_z_fallback", band_cases[kBaseFullZBand].bytes},
        {"full_s", full_s_bytes}
    }};
    std::size_t allocation_bytes = 0;
    for (const Payload& payload : payloads) {
        allocation_bytes = std::max(allocation_bytes, payload.bytes);
    }
    for (const BandCase& band : band_cases) {
        allocation_bytes = std::max(allocation_bytes, band.bytes);
    }

    DeviceResources a{};
    DeviceResources b{};
    if (!allocate(&a, options.device_a, allocation_bytes, &failure) ||
        !allocate(&b, options.device_b, allocation_bytes, &failure) ||
        !fill(&a, 0xA5, allocation_bytes, &failure) ||
        !fill(&b, 0x5A, allocation_bytes, &failure)) {
        release(&a);
        release(&b);
        return fail_json(5, failure);
    }

    std::vector<Result> unidirectional_ab;
    std::vector<Result> unidirectional_ba;
    std::vector<Result> bidirectional;
    for (const Payload& payload : payloads) {
        unidirectional_ab.push_back({
            payload.label, payload.bytes,
            time_flat_copy(&a, &b, payload.bytes, kPatternA,
                           options.warmup, options.iterations)});
        unidirectional_ba.push_back({
            payload.label, payload.bytes,
            time_flat_copy(&b, &a, payload.bytes, kPatternB,
                           options.warmup, options.iterations)});
        bidirectional.push_back({
            payload.label, 2 * payload.bytes,
            time_bidirectional(&a, &b, payload.bytes,
                               options.warmup, options.iterations)});
    }

    std::vector<Result> direct_bands;
    std::vector<Result> bidirectional_bands;
    if (options.direct_bands) {
        for (const BandCase& band : band_cases) {
            direct_bands.push_back({
                band.label, band.bytes,
                time_direct_band(&a, &b, band, allocation_bytes,
                                 options.warmup, options.iterations)});
            bidirectional_bands.push_back({
                band.label, 2 * band.bytes,
                time_bidirectional(&a, &b, band.bytes, options.warmup,
                                   options.iterations, &band,
                                   allocation_bytes)});
        }
    }

    Result add_result{};
    if (options.checked_add) {
        add_result = {"base_initial", payloads[1].bytes,
                      time_checked_add(&b, payloads[1].bytes,
                                       options.warmup, options.iterations)};
    }

    bool all_ok = true;
    auto include = [&](const std::vector<Result>& results) {
        for (const Result& result : results) all_ok = all_ok && result.timing.ok;
    };
    include(unidirectional_ab);
    include(unidirectional_ba);
    include(bidirectional);
    include(direct_bands);
    include(bidirectional_bands);
    if (options.checked_add) all_ok = all_ok && add_result.timing.ok;

    std::printf("{\"tool\":\"pf_peer_probe\",\"ok\":%s,"
                "\"iterations\":%d,\"warmup\":%d,"
                "\"scenario\":{\"cells\":100,\"radius\":49,"
                "\"area_fraction\":0.9,\"base_edge\":%u,"
                "\"nx\":%d,\"ny\":%d,\"nz\":%d,"
                "\"pitch_words\":%d},\"devices\":[",
                all_ok ? "true" : "false", options.iterations,
                options.warmup, kBaseEdge, kDomainRows, kDomainRows,
                kDomainPlanes, kPitchWords);
    std::printf("{\"id\":%d,", options.device_a);
    print_json_string("name", properties_a.name);
    std::printf(",\"cc\":\"%d.%d\",", properties_a.major,
                properties_a.minor);
    print_json_string("uuid", device_uuid(properties_a));
    std::printf("},{\"id\":%d,", options.device_b);
    print_json_string("name", properties_b.name);
    std::printf(",\"cc\":\"%d.%d\",", properties_b.major,
                properties_b.minor);
    print_json_string("uuid", device_uuid(properties_b));
    std::printf("}],\"peer_access\":{\"a_to_b\":true,\"b_to_a\":true}");
    print_results("unidirectional_a_to_b", unidirectional_ab);
    print_results("unidirectional_b_to_a", unidirectional_ba);
    print_results("bidirectional", bidirectional);
    if (options.direct_bands) {
        print_results("direct_y_bands_a_to_b", direct_bands);
        print_results("direct_y_bands_bidirectional", bidirectional_bands);
    }
    if (options.checked_add) {
        std::printf(",\"local_checked_add\":");
        print_result(add_result);
    }
    std::printf("}\n");

    release(&a);
    release(&b);
    return all_ok ? 0 : 6;
}

}  // namespace

int main(int argc, char** argv) {
    return run(argc, argv);
}
