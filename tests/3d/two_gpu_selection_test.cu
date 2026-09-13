#include "pf3d/kernels.cuh"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(status));
}

template <typename T>
class Buffer {
public:
    explicit Buffer(std::size_t count) : count_(count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
              "cudaMalloc");
    }
    ~Buffer() {
        const cudaError_t status = cudaFree(data_);
        if (status != cudaSuccess)
            std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(status));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* get() const { return data_; }
    void upload(const std::vector<T>& values) {
        require(values.size() == count_, "upload size");
        check(cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                         cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> values(count_);
        check(cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                         cudaMemcpyDeviceToHost), "download");
        return values;
    }
    void zero() {
        check(cudaMemset(data_, 0, count_ * sizeof(T)), "cudaMemset");
    }
private:
    T* data_ = nullptr;
    std::size_t count_;
};

constexpr int kN = 4;
constexpr int kB = 24;
constexpr int kShards = 4;
constexpr std::size_t kWords = kB * kB * kB;
constexpr float kSentinel = -17.0f;

bool selected(int n) { return n == 1 || n == 3; }

template <typename T>
bool same(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

struct Fixture {
    pf3d::SLayout3D layout{64, 64, 48, 64,
                          pf3d::kBoundarySubstrateSlab3D};
    Buffer<float> phi{kN * kWords}, out{kN * kWords};
    Buffer<pf3d::CellState3D> cells{kN};
    Buffer<std::uint32_t> aggregate{layout.words()};
    Buffer<std::uint32_t> flags{pf3d::FLAG3D_COUNT};
    Buffer<std::uint64_t> step{1};
    Buffer<unsigned long long> cursor{1};
    Buffer<pf3d::MomentPartial3D> partials{kN * kShards};
    Buffer<int> ids{2};
    std::vector<pf3d::CellState3D> initial;
    std::vector<float> source;

    Fixture() {
        check(pf3d::configure_tile_kernel_shared_memory(), "shared memory");
        ids.upload({3, 1});
        step.upload({17});
        flags.zero();
        std::vector<pf3d::CellState3D> states(kN);
        for (int n = 0; n < kN; ++n) {
            states[n].global_id = 31 + n;
            states[n].gamma = 0.3f;
            states[n].v_A = 0.01f;
            states[n].R_tgt = 4.0f;
            states[n].polarity_x = 1.0f;
        }
        cells.upload(states);
        Buffer<pf3d::Vec3> centres{kN};
        centres.upload({{16, 16, 0}, {23, 16, 0},
                        {16, 23, 0}, {23, 23, 0}});
        pf3d::InitArgs3D init{};
        init.phi_even = phi.get();
        init.cells = cells.get();
        init.centres = centres.get();
        init.N = kN;
        init.B = kB;
        init.layout = layout;
        init.lambda = 1.0f;
        init.seed_radius = 4.0f;
        pf3d::launch_initialize_spheres(init);
        check(cudaDeviceSynchronize(), "initialize");
        initial = cells.download();
        source = phi.download();
        scatter({});
    }

    pf3d::CellSelection3D owners() const { return {ids.get(), 2}; }
    pf3d::CellSelection3D empty() const { return {ids.get(), 0}; }

    void healthy() const {
        for (const std::uint32_t flag : flags.download())
            require(flag == 0u, "unexpected integrity flag");
    }

    void scatter(pf3d::CellSelection3D selection, bool clear = true) {
        if (clear) aggregate.zero();
        pf3d::ScatterArgs3D args{phi.get(), cells.get(), aggregate.get(),
                               flags.get(), layout, kN, kB, selection};
        pf3d::launch_scatter_current(args);
        check(cudaDeviceSynchronize(), "scatter");
    }

    pf3d::MeasureArgs3D measurement(pf3d::CellSelection3D selection,
                                    int shards) const {
        pf3d::MeasureArgs3D args{};
        args.phi = phi.get();
        args.S = aggregate.get();
        args.cells = cells.get();
        args.global_flags = flags.get();
        args.step = step.get();
        args.layout = layout;
        args.N = kN;
        args.B = kB;
        args.partials = partials.get();
        args.shards = shards;
        args.polarity_stream = 29;
        args.p_tumble = 1.0;  // Every selected cell must consume one event.
        args.motility_coeff = 0.01f;
        args.max_shift = 0;
        args.support_margin = 2;
        args.apply_tumble = true;
        args.compute_surface = true;
        args.selection = selection;
        return args;
    }

    pf3d::UpdateArgs3D update(pf3d::CellSelection3D selection) const {
        pf3d::UpdateArgs3D args{};
        args.phi_in = phi.get();
        args.phi_out = out.get();
        args.S_in = aggregate.get();
        args.cells = cells.get();
        args.global_flags = flags.get();
        args.work_cursor = cursor.get();
        args.layout = layout;
        args.N = kN;
        args.B = kB;
        args.dt = 0.001f;
        args.V0 = 100.0;
        args.volume_scale = 0.01;
        args.bulk_scale = 1.0f;
        args.interaction_coeff = 0.1f;
        args.selection = selection;
        return args;
    }
};

void check_cells(const std::vector<pf3d::CellState3D>& actual,
                 const std::vector<pf3d::CellState3D>& owned,
                 const std::vector<pf3d::CellState3D>& unowned) {
    for (int n = 0; n < kN; ++n)
        require(same(actual[n], selected(n) ? owned[n] : unowned[n]),
                "selected cell state or nonowner sentinel differs");
}

void check_fields(const std::vector<float>& actual,
                  const std::vector<float>& expected, bool no_owners = false) {
    for (int n = 0; n < kN; ++n) {
        if (selected(n) && !no_owners) {
            require(std::memcmp(actual.data() + n * kWords,
                                expected.data() + n * kWords,
                                kWords * sizeof(float)) == 0,
                    "selected field differs bitwise");
        } else {
            for (std::size_t q = n * kWords; q < (n + 1) * kWords; ++q)
                require(actual[q] == kSentinel, "nonowner field overwritten");
        }
    }
}

std::vector<pf3d::CellState3D> test_measurement(Fixture& f) {
    std::vector<pf3d::CellState3D> reference;
    for (const int shards : {1, kShards}) {
        f.cells.upload(f.initial);
        require(pf3d::launch_measure_cells(f.measurement({}, shards)),
                "full measurement launch");
        reference = f.cells.download();
        f.healthy();
        for (const auto& cell : reference)
            require(cell.V > 0.0 && cell.tumble_ctr == 1,
                    "measurement and tumble fixture must be active");

        f.cells.upload(f.initial);
        require(pf3d::launch_measure_cells(f.measurement(f.owners(), shards)),
                "selected measurement launch");
        check_cells(f.cells.download(), reference, f.initial);
        f.healthy();

        f.cells.upload(f.initial);
        require(pf3d::launch_measure_cells(f.measurement(f.empty(), shards)),
                "empty measurement launch");
        const auto unchanged = f.cells.download();
        for (int n = 0; n < kN; ++n)
            require(same(unchanged[n], f.initial[n]),
                    "empty selection changed a cell");
    }
    return reference;
}

void test_scatter(Fixture& f) {
    f.cells.upload(f.initial);
    f.scatter(f.owners());
    const auto selected_sum = f.aggregate.download();
    f.scatter({f.ids.get(), 1});
    f.scatter({f.ids.get() + 1, 1}, false);
    require(f.aggregate.download() == selected_sum,
            "selection scatter differs from individual-cell sum");
    f.scatter(f.empty(), false);
    require(f.aggregate.download() == selected_sum,
            "empty scatter changed the aggregate");
    f.scatter({});
    require(f.aggregate.download() != selected_sum,
            "scatter fixture must distinguish all and owned cells");
    f.healthy();
}

void test_updates(Fixture& f,
                  const std::vector<pf3d::CellState3D>& measured) {
    std::vector<float> fast_reference;
    for (const bool strict : {false, true}) {
        auto launch = [&](pf3d::CellSelection3D selection) {
            const auto args = f.update(selection);
            return strict
                ? pf3d::launch_update_tiled_sharded(args, f.partials.get(),
                                                    kShards)
                : pf3d::detail::launch_update_tiled_sharded_fast(args, kShards);
        };
        f.cells.upload(measured);
        require(launch({}), "full update launch");
        const auto reference = f.out.download();
        const auto reference_cells = f.cells.download();
        f.healthy();
        if (strict)
            require(std::memcmp(reference.data(), fast_reference.data(),
                                reference.size() * sizeof(float)) == 0,
                    "strict and fast field updates differ");
        else
            fast_reference = reference;

        f.cells.upload(measured);
        f.out.upload(std::vector<float>(kN * kWords, kSentinel));
        require(launch(f.owners()), "selected update launch");
        check_fields(f.out.download(), reference);
        check_cells(f.cells.download(), reference_cells, measured);
        f.healthy();

        f.cells.upload(measured);
        f.out.upload(std::vector<float>(kN * kWords, kSentinel));
        require(launch(f.empty()), "empty update launch");
        check_fields(f.out.download(), reference, true);
        const auto unchanged = f.cells.download();
        for (int n = 0; n < kN; ++n)
            require(same(unchanged[n], measured[n]),
                    "empty update changed cell state");
    }
}

void test_repair(Fixture& f,
                 const std::vector<pf3d::CellState3D>& measured) {
    f.cells.upload(measured);
    f.out.upload(std::vector<float>(kN * kWords, kSentinel));
    std::vector<std::uint32_t> flags(pf3d::FLAG3D_COUNT);
    flags[pf3d::FLAG3D_NONFINITE] = 1;
    f.flags.upload(flags);
    pf3d::launch_repair_after_fatal(f.update(f.owners()));
    check_fields(f.out.download(), f.source);
    check_cells(f.cells.download(), measured, measured);
    require(f.flags.download() == flags, "repair cleared the fatal flag");
    f.flags.zero();
}

}  // namespace

int main() {
    try {
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice ||
            (status == cudaSuccess && devices == 0)) return 77;
        check(status, "cudaGetDeviceCount");
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaFuncAttributes attributes{};
        const cudaError_t image = cudaFuncGetAttributes(
            &attributes, pf3d::k_scatter_current);
        if (image == cudaErrorNoKernelImageForDevice ||
            image == cudaErrorInvalidDeviceFunction) return 77;
        check(image, "kernel image");
        Fixture fixture;
        test_scatter(fixture);
        const auto measured = test_measurement(fixture);
        test_updates(fixture, measured);
        test_repair(fixture, measured);
        std::puts("owner selection: scatter, motion, fast/strict update, repair, "
                  "nonowner preservation, and empty selection passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "owner selection: %s\n", error.what());
        return 1;
    }
}
