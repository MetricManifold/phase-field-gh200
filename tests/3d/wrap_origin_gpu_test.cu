#include "../../src/pf3d/device_common.cuh"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

struct Case {
    std::int64_t origin;
    int period;
    int mapped = -1;
};

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void probe(Case* cases, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count)
        cases[i].mapped = pf3d::wrap_origin(cases[i].origin, cases[i].period);
}

} // namespace

int main() {
    int devices = 0;
    const auto available = cudaGetDeviceCount(&devices);
    if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver ||
        (available == cudaSuccess && devices == 0)) {
        std::puts("[SKIP] periodic-origin probe requires a CUDA device");
        return 77;
    }
    check(available, "query device");

    constexpr auto low = std::numeric_limits<std::int64_t>::min();
    constexpr auto high = std::numeric_limits<std::int64_t>::max();
    std::vector<Case> cases;
    std::mt19937_64 random(4171);
    for (const int period : {1, 2, 64, 140, 733, 65537,
                             std::numeric_limits<int>::max()}) {
        const std::int64_t n = period;
        for (const std::int64_t origin : {
                 low, low + n, -n * n, -2 * n - 1, -2 * n,
                 -n - 1, -n, -n + 1, std::int64_t{-1}, std::int64_t{0},
                 std::int64_t{1}, n - 1, n, n + 1, 2 * n - 1, 2 * n,
                 2 * n + 1, n * n, high - n, high})
            cases.push_back({origin, period});
        for (int i = 0; i < 1024; ++i) {
            const auto value = static_cast<std::int64_t>(
                random() & static_cast<std::uint64_t>(high));
            cases.push_back({value, period});
            cases.push_back({-value, period});
        }
        for (std::int64_t origin = -256; origin <= 256; ++origin)
            cases.push_back({origin, period});
    }

    Case* device_cases = nullptr;
    const std::size_t bytes = cases.size() * sizeof(Case);
    check(cudaMalloc(reinterpret_cast<void**>(&device_cases), bytes), "allocate cases");
    check(cudaMemcpy(device_cases, cases.data(), bytes, cudaMemcpyHostToDevice),
          "upload cases");
    probe<<<static_cast<unsigned>((cases.size() + 127) / 128), 128>>>(
        device_cases, static_cast<int>(cases.size()));
    check(cudaGetLastError(), "launch probe");
    check(cudaMemcpy(cases.data(), device_cases, bytes, cudaMemcpyDeviceToHost),
          "download cases");
    check(cudaFree(device_cases), "free cases");

    for (const auto& sample : cases) {
        auto expected = sample.origin % static_cast<std::int64_t>(sample.period);
        if (expected < 0) expected += sample.period;
        if (sample.mapped != expected) {
            std::fprintf(stderr, "origin %lld, period %d: got %d, expected %lld\n",
                         static_cast<long long>(sample.origin), sample.period,
                         sample.mapped, static_cast<long long>(expected));
            return 1;
        }
    }
    std::printf("[PASS] %zu periodic origins match host modulo exactly\n", cases.size());
    return 0;
}
