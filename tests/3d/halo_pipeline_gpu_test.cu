// Compile the production loader and scan together so their device helpers
// remain private. This target does not link another copy of pf3d_core.
#include "../../src/pf3d/kernels.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int edge = 40, tiles_x = 2, tiles_y = 3, tile_index = 8;
constexpr unsigned poison_bits = 0x7fc12345u;
constexpr std::size_t source_words = edge * edge * edge;

void check(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "line %d: %s: %s\n", line, expression,
                     cudaGetErrorString(status));
        std::exit(1);
    }
}
#define CHECK(expression) check((expression), #expression, __LINE__)

__global__ void probe_scan(const float* source, int delayed_warp, int* result) {
    extern __shared__ __align__(16) float raw[];
    for (int q = threadIdx.x; q < 2 * pf3d::kHaloBufferWords; q += blockDim.x)
        raw[q] = __uint_as_float(poison_bits);
    __syncthreads();
    // Other warps may load/wait/scan before this warp has issued any copies.
    if (static_cast<int>(threadIdx.x) / 32 == delayed_warp) {
        const unsigned long long start = clock64();
        while (clock64() - start < 100000ull) {}
    }
    float* const halo0 = raw + pf3d::kHaloLeadingPadding;
    float* const halo1 = halo0 + pf3d::kHaloBufferWords;
    if (threadIdx.x == 0 &&
        ((reinterpret_cast<std::uintptr_t>(halo0 + 1)
        | reinterpret_cast<std::uintptr_t>(halo1 + 1)) & 15u))
        atomicAdd(result + 3, 1);
    const auto fetch_z = [](int z, int* source_z) {
        *source_z = z;
        return z >= 0 && z < edge;
    };
    pf3d::prefetch_halo(source, edge, tile_index, tiles_x, tiles_y,
                        fetch_z, 0, halo0);
    pf3d::prefetch_halo(source, edge, tile_index, tiles_x, tiles_y,
                        fetch_z, 0, halo1);
    __pipeline_wait_prior(1);
    const bool first = pf3d::halo_any_nonzero(halo0);
    __syncthreads();
    __pipeline_wait_prior(0);
    const bool second = pf3d::halo_any_nonzero(halo1);
    if (threadIdx.x == 0) {
        result[0] = first;
        result[1] = second;
    }
    // The final scan collective publishes all producers before this check.
    for (int buffer = 0; buffer < 2; ++buffer) {
        const float* const halo = buffer == 0 ? halo0 : halo1;
        for (int q = threadIdx.x; q < pf3d::kHaloVoxels; q += blockDim.x) {
            const int hx = q % pf3d::kHaloX;
            const int hy = (q / pf3d::kHaloX) % pf3d::kHaloY;
            const int hz = q / (pf3d::kHaloX * pf3d::kHaloY);
            // tile (0,1,1) has loader origin (-1,15,7).
            const int x = hx - 1, y = hy + 15, z = hz + 7;
            const float expected = x >= 0 && x < edge
                ? source[(z * edge + y) * edge + x] : 0.0f;
            if (__float_as_uint(halo[pf3d::halo_index(hx, hy, hz)])
                != __float_as_uint(expected)) atomicAdd(result + 2, 1);
        }
    }
    for (int q = threadIdx.x; q < 2 * pf3d::kHaloBufferWords; q += blockDim.x) {
        const int offset = q % pf3d::kHaloBufferWords - pf3d::kHaloLeadingPadding;
        const bool logical = offset >= 0
            && offset / pf3d::kHaloRowStride < pf3d::kHaloY * pf3d::kHaloZ
            && offset % pf3d::kHaloRowStride < pf3d::kHaloX;
        if (!logical && __float_as_uint(raw[q]) != poison_bits)
            atomicAdd(result + 3, 1);
    }
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess || device.major < 9) return 77;
    CHECK(cudaSetDevice(0));
    CHECK(cudaFuncSetAttribute(probe_scan, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(pf3d::kHaloPipelineBytes)));
    float* device_source = nullptr;
    int* device_result = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&device_source), source_words * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&device_result), 4 * sizeof(int)));
    std::vector<float> source(source_words, 0.0f);
    constexpr unsigned patterns[] = {0u, 0x80000000u, 1u, 0x80000001u,
        0x7fc00001u, 0xffc00001u, 0x7f800000u, 0xff800000u, 0x3f800000u};
    // First is interior; others are x-, y-, and z-ghost-only nonzeros.
    constexpr int positions[][3] = {{7, 18, 10}, {32, 18, 10},
                                    {7, 15, 10}, {7, 18, 7}};
    constexpr int delayed_warps[] = {0, 3, 7};
    int cases = 0;
    for (unsigned bits : patterns) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        const int expected = value != 0.0f;
        for (const auto& position : positions) {
            const int index = (position[2] * edge + position[1]) * edge + position[0];
            source[index] = value;
            CHECK(cudaMemcpy(device_source, source.data(), source_words * sizeof(float),
                             cudaMemcpyHostToDevice));
            CHECK(cudaMemset(device_result, 0, 4 * sizeof(int)));
            const int delayed = delayed_warps[cases % 3];
            probe_scan<<<1, pf3d::kThreads3D, pf3d::kHaloPipelineBytes>>>(
                device_source, delayed, device_result);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            int result[4]{};
            CHECK(cudaMemcpy(result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
            if (result[0] != expected || result[1] != expected || result[2] || result[3]) {
                std::fprintf(stderr, "FAIL bits=%08x xyz=%d,%d,%d delayed_warp=%d "
                    "expected=%d got=%d,%d field_errors=%d layout_errors=%d\n",
                    bits, position[0], position[1], position[2], delayed, expected,
                    result[0], result[1], result[2], result[3]);
                CHECK(cudaFree(device_result));
                CHECK(cudaFree(device_source));
                return 1;
            }
            source[index] = 0.0f;
            ++cases;
        }
    }
    CHECK(cudaFree(device_result));
    CHECK(cudaFree(device_source));
    std::printf("PASS: %d one-CTA cases; both buffers, wait_prior(1/0), delayed "
                "producer warps, exact logical values and poisoned padding.\n", cases);
    return 0;
}
