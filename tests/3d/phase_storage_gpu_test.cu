#include "pf3d/kernels.cuh"

#include <algorithm>
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
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<class T> class Buffer {
public:
    explicit Buffer(std::size_t count) : count_(count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "allocate");
        zero();
    }
    ~Buffer() {
        const auto status = cudaFree(data_);
        if (status != cudaSuccess) std::fprintf(stderr, "free: %s\n", cudaGetErrorString(status));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* get() const { return data_; }
    void zero() { check(cudaMemset(data_, 0, count_ * sizeof(T)), "clear"); }
    void upload(const std::vector<T>& values) {
        require(values.size() == count_, "upload count");
        check(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> values(count_);
        check(cudaMemcpy(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return values;
    }
private:
    T* data_ = nullptr;
    std::size_t count_;
};

template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

enum class Update { fast, measured, sharded_fast, sharded_measured, inplace_fast, inplace_measured };

struct Result {
    std::vector<float> fields;
    std::vector<pf3d::CellState3D> cells;
    std::vector<pf3d::VerifyCell3D> verified;
    std::vector<std::uint32_t> aggregate, flags, crops;
};

Result run(int B, pf3d::CellFieldStorage3D storage, Update path,
           bool promoted, bool fail_update, int origin_z) {
    constexpr int N = 2, nz = 16, shards = 4;
    const int base = promoted ? 8 : B;
    const pf3d::SLayout3D layout{64, 64, nz, 64, pf3d::kBoundaryHardWallChannel3D};
    const std::size_t words = storage.words(B);
    Buffer<float> phi(N * words), output(N * words), scratch(2 * words), wall(nz);
    Buffer<pf3d::CellState3D> cells(N);
    Buffer<pf3d::Vec3> centres(N);
    Buffer<std::uint32_t> aggregate(layout.words()), next_aggregate(layout.words());
    Buffer<std::uint32_t> flags(pf3d::FLAG3D_COUNT), crops(N);
    Buffer<std::uint64_t> step(1);
    Buffer<unsigned long long> cursor(1);
    Buffer<pf3d::MomentPartial3D> partials(N * shards);
    Buffer<pf3d::VerifyCell3D> verified(N);
    Buffer<int> ids(N), candidates(N);
    Buffer<float*> promoted_phi(N), promoted_out(N);
    std::vector<pf3d::CellState3D> state(N);
    for (int n = 0; n < N; ++n) {
        state[n].global_id = 31 + n;
        state[n].gamma = n ? 0.35f : 1.0f;
        state[n].v_A = 0.01f;
        state[n].R_tgt = 1.0f;
        state[n].polarity_x = 1.0f;
    }
    cells.upload(state);
    const double z = origin_z + 0.5 * (B - 1);
    centres.upload({{20, 20, z}, {22, 21, z}});
    wall.upload(std::vector<float>(nz, 0.1f));
    ids.upload({0, 1});
    step.upload({17});
    promoted_phi.upload({phi.get(), phi.get() + words});
    promoted_out.upload({output.get(), output.get() + words});
    pf3d::InitArgs3D init{};
    init.phi_even = phi.get(); init.phi_odd = output.get();
    init.cells = cells.get(); init.centres = centres.get();
    init.N = N; init.B = B; init.layout = layout;
    init.lambda = 0.2f; init.seed_radius = 1.0f; init.storage = storage;
    pf3d::launch_initialize_spheres(init);
    check(cudaDeviceSynchronize(), "initialize");
    if (promoted) {
        state = cells.download();
        for (auto& cell : state) cell.storage_edge = B;
        cells.upload(state);
    }
    pf3d::ScatterArgs3D scatter{};
    scatter.phi = phi.get(); scatter.cells = cells.get(); scatter.S = aggregate.get();
    scatter.global_flags = flags.get(); scatter.layout = layout;
    scatter.N = N; scatter.B = base; scatter.storage = storage;
    if (promoted)
        pf3d::launch_scatter_promoted(scatter, promoted_phi.get(), ids.get(), N, B);
    else pf3d::launch_scatter_current(scatter);

    pf3d::MeasureArgs3D measure{};
    measure.phi = phi.get(); measure.S = aggregate.get(); measure.wall = {wall.get(), nz};
    measure.cells = cells.get(); measure.global_flags = flags.get(); measure.step = step.get();
    measure.layout = layout; measure.N = N; measure.B = base;
    measure.partials = partials.get(); measure.shards = shards;
    measure.polarity_stream = 29; measure.p_tumble = 1.0;
    measure.motility_coeff = 0.01f; measure.support_margin = 1;
    measure.apply_tumble = true; measure.compute_surface = true; measure.storage = storage;
    if (promoted) {
        require(pf3d::detail::launch_measure_promoted_shards(measure,
            promoted_phi.get(), ids.get(), N, B, partials.get()), "promoted measurement");
        pf3d::launch_apply_cell_motion(measure);
    } else require(pf3d::launch_measure_cells(measure), "measurement");
    state = cells.download();
    for (auto& cell : state) {
        require(cell.tumble_ctr == 1, "active measurement consumes one tumble");
        cell.pending_shift_x = 1;
        cell.pending_shift_y = -1;
        cell.pending_shift_z = 1;
    }
    cells.upload(state);
    if (fail_update) {
        auto failures = flags.download();
        failures[pf3d::FLAG3D_NONFINITE] = 1;
        flags.upload(failures);
    }
    pf3d::UpdateArgs3D update{};
    update.phi_in = phi.get(); update.phi_out = output.get();
    update.S_in = aggregate.get(); update.S_out = next_aggregate.get();
    update.wall = {wall.get(), nz}; update.cells = cells.get();
    update.global_flags = flags.get(); update.work_cursor = cursor.get();
    update.layout = layout; update.N = N; update.B = base; update.dt = 0.001f;
    update.V0 = 8.0; update.volume_scale = 0.01; update.bulk_scale = 1.0f;
    update.interaction_coeff = 0.1f; update.storage = storage;
    const bool inplace = path == Update::inplace_fast || path == Update::inplace_measured;
    require(!promoted || !inplace, "promoted storage uses two fields");
    bool launched = false;
    if (promoted) {
        update.compute_surface = path == Update::measured;
        if (path == Update::sharded_fast)
            launched = pf3d::detail::launch_update_promoted_sharded_fast(update,
                promoted_phi.get(), promoted_out.get(), ids.get(), N, B, shards);
        else launched = pf3d::launch_update_promoted(update, promoted_phi.get(),
            promoted_out.get(), ids.get(), N, B, path != Update::fast);
    } else switch (path) {
        case Update::fast: launched = pf3d::detail::launch_update_tiled_fast(update, 2); break;
        case Update::measured:
            update.compute_surface = true;
            launched = pf3d::launch_update_tiled(update, 2); break;
        case Update::sharded_fast:
            launched = pf3d::detail::launch_update_tiled_sharded_fast(update, shards); break;
        case Update::sharded_measured:
            launched = pf3d::launch_update_tiled_sharded(update, partials.get(), shards); break;
        case Update::inplace_fast:
            update.phi_out = nullptr;
            launched = pf3d::detail::launch_update_inplace_fast(update, scratch.get(), 2); break;
        case Update::inplace_measured:
            update.phi_out = nullptr;
            update.compute_surface = true;
            launched = pf3d::launch_update_inplace(update, scratch.get(), 2); break;
    }
    require(launched, "update launch");
    if (fail_update) {
        pf3d::launch_repair_after_fatal(update);
        if (promoted) pf3d::launch_repair_promoted_after_fatal(update,
            promoted_phi.get(), promoted_out.get(), ids.get(), N, B);
    }
    pf3d::launch_finalize_origins(cells.get(), N, flags.get());
    const float* result = inplace ? phi.get() : output.get();
    float* const* result_promoted = promoted ? promoted_out.get() : nullptr;
    pf3d::launch_measure_wall_diagnostics(result, result_promoted, cells.get(), N,
        base, layout, wall.get(), 10, 3, 0, storage);
    if (promoted)
        pf3d::launch_verify_promoted(result_promoted, cells.get(), verified.get(),
            ids.get(), N, N, B, layout, 0, storage);
    else pf3d::launch_verify_cells(result, cells.get(), verified.get(), N, B, layout, 0, storage);
    if (promoted && B >= 32) {
        candidates.upload({24, 24});
        require(pf3d::launch_check_promoted_crops(result_promoted, cells.get(),
            candidates.get(), crops.get(), N, base, 0, storage), "crop check");
    }
    Result report;
    report.cells = cells.download(); report.verified = verified.download();
    report.aggregate = next_aggregate.download(); report.flags = flags.download();
    report.crops = crops.download();
    const auto packed = inplace ? phi.download() : output.download();
    const std::size_t cube_words = static_cast<std::size_t>(B) * B * B;
    report.fields.assign(N * cube_words, 0.0f);
    for (int n = 0; n < N; ++n) {
        pf3d::CellFieldPlaneRange3D range{};
        require(storage.retained_planes(B, report.cells[n].origin_z, &range), "unpack");
        std::copy_n(packed.begin() + n * words + range.stored_begin * B * B,
            static_cast<std::size_t>(range.count) * B * B,
            report.fields.begin() + n * cube_words + range.logical_begin * B * B);
    }
    if (!fail_update) for (int f = 0; f < pf3d::FLAG3D_COUNT; ++f)
        require(!pf3d::flag3d_is_fatal(static_cast<pf3d::Flag3D>(f)) || report.flags[f] == 0,
                "unexpected fatal integrity flag");
    return report;
}

void compare(int B, Update path, bool promoted, bool failed, int origin_z) {
    const auto cube = run(B, {}, path, promoted, failed, origin_z);
    const auto capped = run(B, {16}, path, promoted, failed, origin_z);
    require(same(cube.fields, capped.fields), "field bytes differ");
    require(same(cube.cells, capped.cells), "cell state/tumbles/moments differ");
    require(same(cube.verified, capped.verified), "verification reductions differ");
    require(same(cube.aggregate, capped.aggregate), "aggregate differs");
    require(same(cube.flags, capped.flags), "integrity flags differ");
    require(same(cube.crops, capped.crops), "crop decisions differ");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess || device.major < 9) return 77;
    try {
        check(pf3d::configure_tile_kernel_shared_memory(), "shared memory");
        for (int B : {8, 16, 24, 32}) {
            const int origin = (16 - B) / 2;
            for (auto path : {Update::fast, Update::measured, Update::sharded_fast,
                              Update::sharded_measured, Update::inplace_fast,
                              Update::inplace_measured}) {
                compare(B, path, false, false, origin);
                compare(B, path, false, true, origin);
            }
        }
        for (int B : {24, 32})
            for (auto path : {Update::fast, Update::measured, Update::sharded_fast})
                for (bool failed : {false, true})
                    compare(B, path, true, failed, (16 - B) / 2);
        compare(24, Update::sharded_fast, false, false, 1);
        std::puts("PASS compact GPU storage: exact field/state/tumble/aggregate/diagnostic parity; "
                  "thresholds, recentering, promoted cells, in-place and fatal recovery");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "phase_storage_gpu: %s\n", error.what());
        return 1;
    }
}
