// Exercise the production checkpoint codec using CUDA copies only. No solver
// kernel is launched, so a device older than the solver's target can run this.
#include "pf3d/checkpoint.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr int kCells = 3;
constexpr int kBaseEdge = 24;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template<class T> struct DeviceBuffer {
    T* pointer = nullptr;
    std::size_t count;
    explicit DeviceBuffer(std::size_t size) : count(size) {
        checked(cudaMalloc(reinterpret_cast<void**>(&pointer), size * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (cudaFree(pointer) != cudaSuccess) std::terminate();
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void put(const std::vector<T>& data) {
        require(data.size() == count, "device upload extent");
        checked(cudaMemcpy(pointer, data.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void equals(const std::vector<T>& expected) const {
        require(expected.size() == count, "device comparison extent");
        std::vector<T> actual(count);
        checked(cudaMemcpy(actual.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(), expected.data(), count * sizeof(T)) == 0,
                "device payload differs after load");
    }
};

struct Files {
    fs::path directory;
    std::vector<fs::path> owned;
    Files() {
        std::random_device random;
        for (int attempt = 0; attempt < 32; ++attempt) {
            const auto candidate = fs::temp_directory_path() /
                ("pf3d-storage-io-" + std::to_string(random()) + "-" +
                 std::to_string(random()));
            if (fs::create_directory(candidate)) { directory = candidate; return; }
        }
        throw std::runtime_error("cannot reserve fixture directory");
    }
    ~Files() {
        std::error_code error;
        for (const auto& path : owned) fs::remove(path, error);
        // An unexpected child is preserved: directory removal is nonrecursive.
        fs::remove(directory, error);
    }
    std::string path(const char* name) {
        owned.push_back(directory / name);
        return owned.back().string();
    }
};

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    require(static_cast<bool>(file), "open fixture for comparison");
    std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
    require(!file.bad(), "read fixture");
    return bytes;
}

void write_file(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    file.close();
    require(static_cast<bool>(file), "write tampered fixture");
}

void restamp(std::vector<char>& bytes) {
    const auto offset = offsetof(ckpt3d::FileHeader3D, file_crc64);
    std::memset(bytes.data() + offset, 0, sizeof(std::uint64_t));
    const auto crc = ckpt3d::crc64_ecma(bytes.data(), bytes.size());
    std::memcpy(bytes.data() + offset, &crc, sizeof(crc));
}

std::size_t cube(int edge) { return std::size_t(edge) * edge * edge; }

void exercise(int nz, int promoted_edge) {
    Files files;
    const auto cubic_path = files.path("cubic.pf3d");
    const auto compact_path = files.path("compact.pf3d");
    const auto rewritten_path = files.path("rewritten.pf3d");
    const auto corrupt_path = files.path("corrupt.pf3d");
    pf3d::SimParams3D params{};
    params.num_cells = kCells;
    params.Nx = params.Ny = 512;
    params.Nz = nz;
    params.target_radius = 5.0;
    params.lambda = 1.0;
    params.dt = 0.001;
    params.boundary_flags = pf3d::kBoundaryHardWallChannel3D;
    params.channel_height = nz - 6;
    params.channel_padding = 3;
    params.wall_kappa = 1.0;
    params.wall_width = 1.0;
    const char* error = nullptr;
    require(pf3d::validate(params, &error), error ? error : "fixture geometry");
    const pf3d::CellFieldStorage3D storage{nz};

    std::vector<pf3d::CellState3D> cells(kCells);
    std::vector<float> cubic_base(kCells * cube(kBaseEdge), 0.0f);
    std::vector<float> compact_base(kCells * storage.words(kBaseEdge), 0.0f);
    std::vector<float> cubic_promoted(cube(promoted_edge), 0.0f);
    std::vector<float> compact_promoted(storage.words(promoted_edge), 0.0f);
    for (int slot = 0; slot < kCells; ++slot) {
        const int edge = slot == 1 ? promoted_edge : kBaseEdge;
        auto& cell = cells[slot];
        cell.global_id = slot;
        cell.origin_x = 40 * slot;
        cell.origin_z = slot == 0 ? -4 : slot == 1 ?
            (promoted_edge > 64 ? -214 : 3) : nz - 8;
        cell.storage_edge = slot == 1 ? promoted_edge : 0;
        cell.polarity_z = 1.0f;
        cell.gamma = 0.35f + 0.25f * slot;
        cell.v_A = 0.01f;
        cell.R_tgt = 5.0f;
        cell.phi_max = 1.0f;
        cell.V = 50.0 + slot;
        cell.bb_lo_x = cell.bb_lo_y = 0;
        cell.bb_hi_x = cell.bb_hi_y = edge - 1;
        cell.bb_lo_z = static_cast<int>(std::max<std::int64_t>(0, -cell.origin_z));
        cell.bb_hi_z = static_cast<int>(std::min<std::int64_t>(edge, nz - cell.origin_z)) - 1;
        float* cubic = slot == 1 ? cubic_promoted.data() :
            cubic_base.data() + std::size_t(slot) * cube(kBaseEdge);
        float* compact = slot == 1 ? compact_promoted.data() :
            compact_base.data() + std::size_t(slot) * storage.words(kBaseEdge);
        for (int z = cell.bb_lo_z; z <= cell.bb_hi_z; ++z) {
            for (int y = 0; y < edge; ++y) {
                for (int x = 0; x < edge; ++x) {
                    const float value = float(1 + (x + 3 * y + 7 * z + slot) % 31) / 32.0f;
                    cubic[pf3d::CellFieldStorage3D{}.index(x, y, z, edge, cell.origin_z)] = value;
                    compact[storage.index(x, y, z, edge, cell.origin_z)] = value;
                }
            }
        }
    }

    DeviceBuffer<pf3d::CellState3D> d_cells(kCells);
    DeviceBuffer<float> d_cubic_base(cubic_base.size()), d_cubic_promoted(cubic_promoted.size());
    DeviceBuffer<float> d_compact_base(compact_base.size()), d_compact_promoted(compact_promoted.size());
    d_cells.put(cells);
    d_cubic_base.put(cubic_base); d_cubic_promoted.put(cubic_promoted);
    d_compact_base.put(compact_base); d_compact_promoted.put(compact_promoted);
    float* cubic_table[kCells]{nullptr, d_cubic_promoted.pointer, nullptr};
    float* compact_table[kCells]{nullptr, d_compact_promoted.pointer, nullptr};
    pf3d::CheckpointWriteView3D write{};
    write.params = &params;
    write.brick_edge = kBaseEdge;
    write.step = 11; write.time = write.step * params.dt;
    write.base_measure_shards = 4;
    write.d_cells = d_cells.pointer;
    write.d_phi = d_cubic_base.pointer;
    write.h_promoted_phi = cubic_table;
    require(pf3d::checkpoint_write_3d(cubic_path, write), "default cubic write");
    write.storage = storage;
    write.d_phi = d_compact_base.pointer;
    write.h_promoted_phi = compact_table;
    require(pf3d::checkpoint_write_3d(compact_path, write), "compact write");
    const auto canonical = read_file(cubic_path);
    require(canonical == read_file(compact_path), "compact and cubic files must be byte-identical");

    pf3d::CheckpointMeta3D meta{};
    require(pf3d::checkpoint_probe_3d(cubic_path, &meta), "probe cubic fixture");
    pf3d::CheckpointLoadView3D load{};
    load.d_cells = d_cells.pointer;
    load.d_phi = d_compact_base.pointer;
    load.h_promoted_phi = compact_table;
    load.storage = storage;
    // Every allocated plane is initialized, even when only part intersects
    // this logical brick. Base slots reserved for promoted cells are unused.
    checked(cudaMemset(d_compact_base.pointer, 0x3f, compact_base.size() * sizeof(float)));
    checked(cudaMemset(d_compact_promoted.pointer, 0x3f, compact_promoted.size() * sizeof(float)));
    require(pf3d::checkpoint_load_3d(cubic_path, meta, load), "load cubic into compact storage");
    checked(cudaMemset(d_compact_base.pointer + storage.words(kBaseEdge), 0,
                       storage.words(kBaseEdge) * sizeof(float)));
    d_compact_base.equals(compact_base); d_compact_promoted.equals(compact_promoted);
    require(pf3d::checkpoint_write_3d(rewritten_path, write), "rewrite loaded compact checkpoint");
    require(canonical == read_file(rewritten_path), "load and rewrite preserve cubic ABI and metadata");

    load.storage = {};
    load.d_phi = d_cubic_base.pointer;
    load.h_promoted_phi = cubic_table;
    require(pf3d::checkpoint_load_3d(compact_path, meta, load), "default cubic reader accepts compact writer");
    d_cubic_base.equals(cubic_base); d_cubic_promoted.equals(cubic_promoted);
    load.storage = storage; load.d_phi = d_compact_base.pointer; load.h_promoted_phi = compact_table;

    auto changed = canonical;
    const std::size_t first_phase = sizeof(ckpt3d::FileHeader3D) +
        sizeof(ckpt3d::ParamsRecord3D) + sizeof(ckpt3d::CellRecord3D);
    // A retained value is tampered without restamping to isolate CRC checking.
    const std::size_t retained = first_phase +
        4 * std::size_t(kBaseEdge) * kBaseEdge * sizeof(float);
    changed[retained] ^= 1;
    write_file(corrupt_path, changed);
    pf3d::CheckpointMeta3D corrupt_meta{};
    require(pf3d::checkpoint_probe_3d(corrupt_path, &corrupt_meta), "probe does not hash payload");
    require(!pf3d::checkpoint_load_3d(corrupt_path, corrupt_meta, load), "loader rejects payload CRC corruption");
    if (storage.compact(kBaseEdge)) {
        for (float value : {0.25f, std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
            changed = canonical;
            std::memcpy(changed.data() + first_phase, &value, sizeof(value));
            restamp(changed);
            write_file(corrupt_path, changed);
            require(pf3d::checkpoint_probe_3d(corrupt_path, &corrupt_meta), "probe restamped discarded plane");
            require(!pf3d::checkpoint_load_3d(corrupt_path, corrupt_meta, load),
                    "compact load rejects nonzero/nonfinite discarded plane despite valid CRC");
        }
    }
    load.storage.z_cap = nz - 1;
    require(!pf3d::checkpoint_load_3d(cubic_path, meta, load), "load rejects mismatched storage height");
    write.storage.z_cap = nz - 1;
    require(!pf3d::checkpoint_write_3d(compact_path, write), "write rejects mismatched storage height");
    require(canonical == read_file(compact_path), "failed write preserves existing checkpoint");
    std::printf("phase_storage_io: Nz=%d base=%d promoted=%d passed\n", nz, kBaseEdge, promoted_edge);
}
} // namespace

int main() {
    try {
        int devices = 0;
        checked(cudaGetDeviceCount(&devices));
        if (devices == 0) return 77;
        for (int nz : {16, 23, 24, 25, 32}) exercise(nz, 32);
        // The retained planes straddle the codec's 64 MiB staging boundary.
        exercise(16, 280);
        std::puts("phase_storage_io: all cases passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "phase_storage_io: %s\n", error.what());
        return 1;
    }
}
