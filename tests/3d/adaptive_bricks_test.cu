#include "pf3d/sim.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kCells = 2;
constexpr int kBase = 32;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::size_t words(int edge) {
    return static_cast<std::size_t>(edge) * edge * edge;
}

template <typename T>
bool identical(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

bool identical(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "allocate test buffer");
    }
    ~DeviceBuffer() {
        const cudaError_t status = cudaFree(data);
        if (status != cudaSuccess)
            std::fprintf(stderr, "release test buffer: %s\n", cudaGetErrorString(status));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct CheckpointFile {
    std::filesystem::path directory;
    std::filesystem::path file;

    CheckpointFile() {
        std::random_device entropy;
        for (int attempt = 0; attempt < 16; ++attempt) {
            char name[80]{};
            std::snprintf(name, sizeof(name), "pf3d_adaptive_%08x%08x",
                          static_cast<unsigned>(entropy()), static_cast<unsigned>(entropy()));
            const auto candidate = std::filesystem::temp_directory_path() / name;
            if (std::filesystem::create_directory(candidate)) {
                directory = candidate;
                file = directory / "mixed.pf3d";
                return;
            }
        }
        throw std::runtime_error("cannot create owned checkpoint-test directory");
    }

    ~CheckpointFile() {
        std::error_code error;
        std::filesystem::remove(file, error);
        if (error) std::fprintf(stderr, "remove test checkpoint: %s\n", error.message().c_str());
        error.clear();
        // Only remove this verified-owned directory when empty; never recurse.
        std::filesystem::remove(directory, error);
        if (error) std::fprintf(stderr, "remove test directory: %s\n", error.message().c_str());
    }
};

pf3d::SimParams3D parameters(bool channel = false) {
    pf3d::SimParams3D p{};
    p.num_cells = kCells;
    p.Nx = p.Ny = p.Nz = 96;
    p.target_radius = 3.0;
    p.lambda = 1.0;
    p.dt = 0.001;
    p.t_end = 0.01;
    if (channel) {
        p.boundary_flags = pf3d::kBoundaryHardWallChannel3D;
        p.Nz = 40;
        p.channel_height = 16;
        p.channel_padding = 12;
        p.wall_width = 1.0;
        p.wall_kappa = 10.0;
    }
    return p;
}

pf3d::RunOptions3D options() {
    pf3d::RunOptions3D opt{};
    opt.storage_mode = pf3d::StorageMode3D::Throughput;
    opt.brick_edge = kBase;
    opt.measure_shards = 4;
    opt.measure_shards_supplied = true;
    opt.promoted_measure_shards = 4;
    opt.promoted_measure_shards_supplied = true;
    opt.final_checkpoint = false;
    return opt;
}

}  // namespace

namespace pf3d {

struct Sim3DTestAccess {
    static void initialize(Sim3D& sim, bool channel = false) {
        require(sim.allocate(parameters(channel), options(), kBase, false, std::nullopt), "allocate fixture");
        std::vector<CellState3D> cells(kCells);
        std::vector<float> phi(kCells * words(kBase), 0.0f);
        constexpr int centre = kBase / 2 - 1;
        for (int n = 0; n < kCells; ++n) {
            auto& cell = cells[n];
            cell.global_id = n;
            cell.origin_x = cell.origin_y = cell.origin_z = 12 + 40 * n;
            if (channel) cell.origin_z = n == 0 ? -2 : 12;
            cell.polarity_x = n == 0 ? 1.0f : 0.0f;
            cell.polarity_y = n == 1 ? 1.0f : 0.0f;
            cell.v_A = 0.01f;
            cell.velocity_x = cell.v_A * cell.polarity_x;
            cell.velocity_y = cell.v_A * cell.polarity_y;
            cell.gamma = n == 0 ? 0.35f : 1.0f;
            cell.R_tgt = 3.0f;
            cell.phi_max = 1.0f;
            cell.V = 1.0;
            cell.Cx = cell.Cy = cell.Cz = centre;
            cell.bb_lo_x = cell.bb_hi_x = centre;
            cell.bb_lo_y = cell.bb_hi_y = centre;
            cell.bb_lo_z = cell.bb_hi_z = centre;
            cell.shift_ctr = 7u + n;
            cell.tumble_ctr = 19u + n;
            phi[n * words(kBase) + (centre * kBase + centre) * kBase + centre] = 1.0f;
            if (channel) {
                const auto add = [&](int x, int y, int z, float value) {
                    phi[n * words(kBase) + (z * kBase + y) * kBase + x] = value;
                    const double square = static_cast<double>(value) * value;
                    cell.V += square;
                    cell.Cx += x * square; cell.Cy += y * square; cell.Cz += z * square;
                    cell.bb_lo_x = std::min(cell.bb_lo_x, x); cell.bb_hi_x = std::max(cell.bb_hi_x, x);
                    cell.bb_lo_y = std::min(cell.bb_lo_y, y); cell.bb_hi_y = std::max(cell.bb_hi_y, y);
                    cell.bb_lo_z = std::min(cell.bb_lo_z, z); cell.bb_hi_z = std::max(cell.bb_hi_z, z);
                };
                add(13, 14, 17, 0.5f);
                add(18, 16, 12, 0.25f);
            }
        }
        check(cudaMemcpy(sim.d_cells_, cells.data(), cells.size() * sizeof(CellState3D),
                         cudaMemcpyHostToDevice), "initialize cell state");
        for (float* buffer : sim.d_phi_)
            check(cudaMemcpy(buffer, phi.data(), phi.size() * sizeof(float),
                             cudaMemcpyHostToDevice), "initialize both base fields");
        require(sim.reconstruct_current_S(), "initialize fixture aggregate");
        sim.volume_current_ = sim.surface_current_ = true;
        check(cudaStreamSynchronize(sim.stream_), "finish fixture initialization");
    }

    static std::vector<CellState3D> states(Sim3D& sim) {
        check(cudaStreamSynchronize(sim.stream_), "wait for state");
        std::vector<CellState3D> result(kCells);
        check(cudaMemcpy(result.data(), sim.d_cells_, result.size() * sizeof(CellState3D),
                         cudaMemcpyDeviceToHost), "read cell state");
        return result;
    }

    static float* field_pointer(Sim3D& sim, int id, bool alternate = false) {
        const auto cells = states(sim);
        if (cell_is_promoted(cells[id], sim.B_)) {
            const int buffer = sim.current_promoted_phi_index_ ^ static_cast<int>(alternate);
            return sim.h_promoted_phi_[buffer][id];
        }
        const int buffer = sim.current_phi_index_ ^ static_cast<int>(alternate);
        return sim.d_phi_[buffer] + static_cast<std::size_t>(id) * sim.brick_words_;
    }

    static std::vector<float> field(Sim3D& sim, int id, bool alternate = false) {
        const auto cells = states(sim);
        const int edge = cell_support_edge(cells[id], sim.B_);
        std::vector<float> stored(sim.field_storage_.words(edge));
        check(cudaMemcpy(stored.data(), field_pointer(sim, id, alternate),
                         stored.size() * sizeof(float), cudaMemcpyDeviceToHost), "read phase allocation");
        if (!sim.field_storage_.compact(edge)) return stored;
        // Decode independently of the device addressing helper: compact z is
        // world z, and every plane outside this logical brick must stay zero.
        std::vector<float> result(words(edge), 0.0f);
        const std::size_t plane = static_cast<std::size_t>(edge) * edge;
        for (int world = 0; world < sim.params_.Nz; ++world) {
            const auto z = static_cast<std::int64_t>(world) - cells[id].origin_z;
            const float* source = stored.data() + static_cast<std::size_t>(world) * plane;
            if (z >= 0 && z < edge)
                std::copy_n(source, plane, result.data() + static_cast<std::size_t>(z) * plane);
            else
                require(std::all_of(source, source + plane, [](float v) { return v == 0.0f; }),
                        "unused compact world plane is not zero");
        }
        return result;
    }

    static void upload_field(Sim3D& sim, int id, const std::vector<float>& values) {
        const auto cells = states(sim);
        const int edge = cell_support_edge(cells[id], sim.B_);
        require(field(sim, id).size() == values.size(), "field upload extent");
        std::vector<float> packed;
        const float* data = values.data();
        std::size_t count = values.size();
        if (sim.field_storage_.compact(edge)) {
            packed.assign(sim.field_storage_.words(edge), 0.0f);
            const std::size_t plane = static_cast<std::size_t>(edge) * edge;
            for (int z = 0; z < edge; ++z) {
                const auto world = cells[id].origin_z + z;
                const float* source = values.data() + static_cast<std::size_t>(z) * plane;
                if (world >= 0 && world < sim.params_.Nz)
                    std::copy_n(source, plane, packed.data() + static_cast<std::size_t>(world) * plane);
                else
                    require(std::all_of(source, source + plane, [](float v) { return v == 0.0f; }),
                            "fixture would discard a nonzero out-of-domain plane");
            }
            data = packed.data(); count = packed.size();
        }
        for (bool alternate : {false, true})
            check(cudaMemcpy(field_pointer(sim, id, alternate), data,
                             count * sizeof(float), cudaMemcpyHostToDevice), "stage crop fixture");
    }

    static bool resize(Sim3D& sim, int id, int edge) {
        auto cells = states(sim);
        return sim.resize_cells({{id, edge}}, &cells);
    }

    static std::uint32_t crop_rejection(Sim3D& sim, int id, int edge) {
        std::vector<int> candidates(kCells, 0);
        candidates[id] = edge;
        DeviceBuffer<int> d_candidates(kCells);
        DeviceBuffer<std::uint32_t> d_rejected(kCells);
        check(cudaMemcpy(d_candidates.data, candidates.data(), kCells * sizeof(int),
                         cudaMemcpyHostToDevice), "stage crop edges");
        check(cudaMemset(d_rejected.data, 0, kCells * sizeof(std::uint32_t)), "clear crop rejection");
        require(launch_check_promoted_crops(sim.d_promoted_phi_[sim.current_promoted_phi_index_],
                    sim.d_cells_, d_candidates.data, d_rejected.data, kCells, sim.B_, sim.stream_,
                    sim.field_storage_),
                "launch crop safety probe");
        check(cudaStreamSynchronize(sim.stream_), "finish crop safety probe");
        std::vector<std::uint32_t> rejected(kCells);
        check(cudaMemcpy(rejected.data(), d_rejected.data, kCells * sizeof(std::uint32_t),
                         cudaMemcpyDeviceToHost), "read crop rejection");
        require(rejected[1 - id] == 0u, "crop probe touched skipped cell");
        return rejected[id];
    }

    static bool compact(Sim3D& sim) { return sim.compact_promoted_fields(); }
    static std::size_t memory(Sim3D& sim) { return sim.required_device_bytes_; }
    static bool base_reclaimed(Sim3D& sim, int id) {
        return !sim.h_promoted_phi_[0][id] && !sim.h_promoted_phi_[1][id] &&
            std::find(sim.h_promoted_ids_.begin(), sim.h_promoted_ids_.end(), id)
                == sim.h_promoted_ids_.end();
    }

    static bool write(Sim3D& sim, const std::string& path) {
        CheckpointWriteView3D view{};
        view.params = &sim.params_;
        view.step = sim.steps_done_;
        view.time = sim.time();
        view.brick_edge = sim.B_;
        view.base_measure_shards = sim.measurement_shards_;
        view.promoted_measure_reduction = {sim.options_.promoted_measure_shards,
                                           sim.options_.promoted_measure_auto_wave_ctas};
        view.d_cells = sim.d_cells_;
        view.d_phi = sim.current_phi();
        view.h_promoted_phi = sim.h_promoted_phi_[sim.current_promoted_phi_index_].data();
        view.stream = sim.stream_;
        view.storage = sim.field_storage_;
        return checkpoint_write_3d(path, view);
    }
};

}  // namespace pf3d

namespace {

using Access = pf3d::Sim3DTestAccess;

void unchanged_identity(const pf3d::CellState3D& a, const pf3d::CellState3D& b) {
    require(a.global_id == b.global_id && identical(a.gamma, b.gamma) &&
            identical(a.R_tgt, b.R_tgt) && identical(a.v_A, b.v_A) &&
            identical(a.polarity_x, b.polarity_x) && identical(a.polarity_y, b.polarity_y) &&
            identical(a.polarity_z, b.polarity_z) && identical(a.velocity_x, b.velocity_x) &&
            identical(a.velocity_y, b.velocity_y) && identical(a.velocity_z, b.velocity_z) &&
            a.shift_ctr == b.shift_ctr && a.tumble_ctr == b.tumble_ctr && identical(a.V, b.V),
            "resize changed cell parameters, motion, volume, or event history");
}

void translated_exactly(const pf3d::CellState3D& before, const std::vector<float>& old_field,
                        const pf3d::CellState3D& after, const std::vector<float>& new_field) {
    const int old_edge = pf3d::cell_support_edge(before, kBase);
    const int new_edge = pf3d::cell_support_edge(after, kBase);
    const int offset = (new_edge - old_edge) / 2;
    unchanged_identity(before, after);
    require(after.origin_x == before.origin_x - offset &&
            after.origin_y == before.origin_y - offset && after.origin_z == before.origin_z - offset,
            "resize did not translate the unwrapped origin");
    require(after.Cx == before.Cx + offset * before.V &&
            after.Cy == before.Cy + offset * before.V && after.Cz == before.Cz + offset * before.V,
            "resize did not translate local moments");
    std::vector<float> expected(words(new_edge), 0.0f);
    for (int z = 0; z < old_edge; ++z)
        for (int y = 0; y < old_edge; ++y)
            for (int x = 0; x < old_edge; ++x) {
                const float value = old_field[(z * old_edge + y) * old_edge + x];
                const int xx = x + offset, yy = y + offset, zz = z + offset;
                if (xx >= 0 && xx < new_edge && yy >= 0 && yy < new_edge && zz >= 0 && zz < new_edge)
                    expected[(zz * new_edge + yy) * new_edge + xx] = value;
                else
                    require(value == 0.0f, "test attempted to discard nonzero support");
            }
    require(identical(expected, new_field), "phase field changed in physical coordinates");
}

void test_adaptive_bricks() {
    pf3d::Sim3D sim;
    Access::initialize(sim);
    const auto initial = Access::states(sim);
    const auto initial0 = Access::field(sim, 0), initial1 = Access::field(sim, 1);
    const std::size_t base_memory = Access::memory(sim);
    require(Access::resize(sim, 0, 48), "grow first cell to 48");
    auto cells = Access::states(sim);
    translated_exactly(initial[0], initial0, cells[0], Access::field(sim, 0));
    require(identical(Access::field(sim, 0), Access::field(sim, 0, true)), "growth alternate field differs");
    require(identical(initial[1], cells[1]) && identical(initial1, Access::field(sim, 1)),
            "growing cell 0 modified cell 1");
    require(Access::resize(sim, 1, 64), "grow second cell independently to 64");
    cells = Access::states(sim);
    require(cells[0].storage_edge == 48u && cells[1].storage_edge == 64u, "mixed promoted sizes lost");
    translated_exactly(initial[1], initial1, cells[1], Access::field(sim, 1));

    CheckpointFile checkpoint;
    require(Access::write(sim, checkpoint.file.string()), "write mixed-edge production checkpoint");
    pf3d::CheckpointMeta3D metadata{};
    require(pf3d::checkpoint_probe_3d(checkpoint.file.string(), &metadata), "probe mixed-edge checkpoint");
    require(metadata.storage_edges == std::vector<int>({48, 64}), "checkpoint edge table changed");
    {
        pf3d::Sim3D restored;
        require(restored.init_checkpoint(metadata, checkpoint.file.string(), options()), "restore mixed-edge checkpoint");
        const auto restored_cells = Access::states(restored);
        for (int id = 0; id < kCells; ++id) {
            unchanged_identity(cells[id], restored_cells[id]);
            require(identical(Access::field(sim, id), Access::field(restored, id)), "restored field differs");
            require(identical(Access::field(restored, id), Access::field(restored, id, true)),
                    "restored alternate cube has wrong edge or data");
        }
    }

    const auto zero_tail = Access::field(sim, 0);
    for (float tail : {1.0f, 1.0e-8f, std::numeric_limits<float>::denorm_min()}) {
        auto contaminated = zero_tail;
        contaminated[0] = tail;
        Access::upload_field(sim, 0, contaminated);
        const auto before = Access::states(sim);
        require(Access::crop_rejection(sim, 0, kBase) != 0u, "crop accepted a nonzero discarded tail");
        require(!Access::resize(sim, 0, kBase), "unsafe resize did not reject the transaction");
        require(identical(before[0], Access::states(sim)[0]) &&
                identical(contaminated, Access::field(sim, 0)), "rejected crop changed accepted state");
    }
    for (std::size_t index : {std::size_t{0}, (std::size_t{23} * 48 + 23) * 48 + 23}) {
        auto nonfinite = zero_tail;
        nonfinite[index] = std::numeric_limits<float>::quiet_NaN();
        Access::upload_field(sim, 0, nonfinite);
        require(Access::crop_rejection(sim, 0, kBase) != 0u, "crop failed to reject a nonfinite voxel");
    }
    auto guard_tail = zero_tail;
    guard_tail[(23 * 48 + 23) * 48 + 8] = 1.0e-8f;
    Access::upload_field(sim, 0, guard_tail);
    require(Access::crop_rejection(sim, 0, kBase) != 0u,
            "crop accepted nonzero support in the retained zero guard");
    Access::upload_field(sim, 0, zero_tail);
    require(Access::crop_rejection(sim, 0, kBase) == 0u, "exact-zero crop was rejected");
    const auto before_shrink = Access::states(sim);
    require(Access::resize(sim, 0, kBase), "shrink first cell to base storage");
    cells = Access::states(sim);
    translated_exactly(before_shrink[0], zero_tail, cells[0], Access::field(sim, 0));
    require(Access::base_reclaimed(sim, 0), "demotion retained obsolete promoted allocations");
    require(identical(before_shrink[1], cells[1]), "demotion altered another cell");
    const auto before_compaction = cells[1];
    const auto field_before_compaction = Access::field(sim, 1);
    require(Access::compact(sim), "compact remaining zero-tail promoted cell");
    cells = Access::states(sim);
    require(cells[1].storage_edge == 48u, "compaction did not retain its sixteen-voxel face margin");
    translated_exactly(before_compaction, field_before_compaction, cells[1], Access::field(sim, 1));
    require(Access::resize(sim, 1, kBase), "explicit safe demotion of second cell");
    cells = Access::states(sim);
    for (int id = 0; id < kCells; ++id) {
        require(pf3d::cell_support_edge(cells[id], kBase) == kBase && Access::base_reclaimed(sim, id),
                "automatic compaction did not reclaim promoted storage");
        unchanged_identity(initial[id], cells[id]);
        require(identical(id == 0 ? initial0 : initial1, Access::field(sim, id)),
                "compaction changed the base field");
    }
    require(Access::memory(sim) == base_memory, "promoted memory accounting was not reclaimed");
}

void test_channel_resize() {
    pf3d::Sim3D sim;
    Access::initialize(sim, true);
    const auto initial = Access::states(sim);
    const std::vector<float> original[kCells] = {Access::field(sim, 0), Access::field(sim, 1)};
    const std::size_t base_memory = Access::memory(sim);
    require(initial[0].origin_z < 0 && initial[1].origin_z + kBase > 40,
            "channel fixture must cross both outer allocation faces");

    const auto check_memory = [&] {
        std::size_t expected = base_memory;
        for (const auto& cell : Access::states(sim)) {
            const int edge = pf3d::cell_support_edge(cell, kBase);
            if (edge > kBase)
                expected += 2 * sizeof(float) * static_cast<std::size_t>(edge) * edge * 40;
        }
        require(Access::memory(sim) == expected, "channel allocation accounting is not B*B*Nz");
    };
    const auto resize = [&](int id, int edge) {
        const auto before = Access::states(sim);
        const auto old_field = Access::field(sim, id);
        const auto other_field = Access::field(sim, 1 - id);
        require(Access::resize(sim, id, edge), "resize channel fixture");
        const auto after = Access::states(sim);
        translated_exactly(before[id], old_field, after[id], Access::field(sim, id));
        require(identical(before[1 - id], after[1 - id]) &&
                identical(other_field, Access::field(sim, 1 - id)),
                "channel resizing modified an unselected cell");
        for (int cell = 0; cell < kCells; ++cell)
            require(identical(Access::field(sim, cell), Access::field(sim, cell, true)),
                    "channel alternate buffer differs after resizing");
        check_memory();
    };

    resize(0, 48); // Cubic B32 crosses the Nz40 storage threshold.
    resize(1, 64);
    {
        CheckpointFile checkpoint;
        require(Access::write(sim, checkpoint.file.string()), "write bounded mixed-edge checkpoint");
        pf3d::CheckpointMeta3D metadata{};
        require(pf3d::checkpoint_probe_3d(checkpoint.file.string(), &metadata),
                "probe bounded mixed-edge checkpoint");
        require(metadata.params.Nz == 40 && metadata.storage_edges == std::vector<int>({48, 64}),
                "bounded checkpoint lost geometry or storage classes");
        pf3d::Sim3D restored;
        require(restored.init_checkpoint(metadata, checkpoint.file.string(), options()),
                "restore bounded mixed-edge checkpoint");
        const auto before = Access::states(sim), after = Access::states(restored);
        for (int id = 0; id < kCells; ++id) {
            unchanged_identity(before[id], after[id]);
            for (bool alternate : {false, true})
                require(identical(Access::field(sim, id), Access::field(restored, id, alternate)),
                        "bounded checkpoint did not restore both physical fields");
        }
        require(Access::memory(restored) == Access::memory(sim),
                "checkpoint reconstruction changed bounded allocation accounting");
    }

    resize(0, 64); // Compact-to-compact growth also changes the logical origin.
    {
        const auto before = Access::states(sim);
        const auto clean = Access::field(sim, 0);
        auto contaminated = clean;
        const auto z = static_cast<std::size_t>(20 - before[0].origin_z);
        contaminated[(z * 64 + 32) * 64] = 1.0e-8f;
        Access::upload_field(sim, 0, contaminated);
        require(Access::crop_rejection(sim, 0, kBase) != 0u &&
                !Access::resize(sim, 0, kBase), "channel crop discarded a nonzero stored tail");
        require(identical(before[0], Access::states(sim)[0]), "failed channel crop changed metadata");
        for (bool alternate : {false, true})
            require(identical(contaminated, Access::field(sim, 0, alternate)),
                    "failed channel crop changed a field buffer");
        check_memory();
        Access::upload_field(sim, 0, clean);
    }
    resize(0, 48); // Compact-to-compact crop, then compact-to-cubic demotion.
    resize(0, kBase);
    resize(1, kBase);
    const auto final = Access::states(sim);
    for (int id = 0; id < kCells; ++id) {
        require(Access::base_reclaimed(sim, id), "channel demotion retained promoted storage");
        unchanged_identity(initial[id], final[id]);
        require(final[id].origin_z == initial[id].origin_z &&
                identical(original[id], Access::field(sim, id)),
                "channel resize round trip changed the physical field");
    }
    require(Access::memory(sim) == base_memory, "channel demotion failed to reclaim its allocation");
}

}  // namespace

int main() {
    try {
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && devices == 0)) return 77;
        check(status, "cudaGetDeviceCount");
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaFuncAttributes attributes{};
        const cudaError_t image = cudaFuncGetAttributes(&attributes, pf3d::k_scatter_current);
        if (image == cudaErrorNoKernelImageForDevice || image == cudaErrorInvalidDeviceFunction) return 77;
        check(image, "kernel image");
        test_adaptive_bricks();
        test_channel_resize();
        std::puts("adaptive bricks: independent growth, mixed checkpoint/restart, exact crop rejection, "
                  "demotion, compaction, bounded-height resize, and metadata preservation passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "adaptive bricks: %s\n", error.what());
        return 1;
    }
}
