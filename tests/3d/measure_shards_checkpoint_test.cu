// Checkpoint contract coverage for the base-measurement shard count. Uses
// the production writer, prober, and loader with device staging buffers but
// launches no kernels, so any CUDA device suffices. Kernel-path coverage is a
// separate GH200 validation step.

#include "pf3d/checkpoint.cuh"
#include "pf3d/sim.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[FAIL] %s\n", message);
        ++failures;
    }
}

pf3d::SimParams3D small_params() {
    pf3d::SimParams3D p{};
    p.num_cells = 2;
    p.Nx = p.Ny = p.Nz = 304;
    p.boundary_flags = pf3d::kBoundaryPeriodicXYZ3D;
    return p;
}

bool read_file(const std::string& path, std::vector<char>* bytes) {
    if (!bytes) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    bytes->assign(std::istreambuf_iterator<char>(in),
                  std::istreambuf_iterator<char>());
    return !in.bad();
}

bool write_file(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    return static_cast<bool>(out);
}

// Re-checksum a byte-patched file: the stored CRC is CRC-64/ECMA over the
// whole file with file_crc64 treated as zero.
bool restamp_crc(std::vector<char>* bytes) {
    if (!bytes || bytes->size() < sizeof(ckpt3d::FileHeader3D)) return false;
    auto& b = *bytes;
    const std::size_t crc_offset = offsetof(ckpt3d::FileHeader3D, file_crc64);
    std::memset(b.data() + crc_offset, 0, sizeof(std::uint64_t));
    const std::uint64_t crc = ckpt3d::crc64_ecma(b.data(), b.size());
    std::memcpy(b.data() + crc_offset, &crc, sizeof(crc));
    return true;
}

bool set_stored_shards(std::vector<char>* bytes, std::uint64_t value,
                       bool fix_crc) {
    if (!bytes || bytes->size() < sizeof(ckpt3d::FileHeader3D)) return false;
    const std::size_t offset =
        offsetof(ckpt3d::FileHeader3D, base_measure_shards);
    std::memcpy(bytes->data() + offset, &value, sizeof(value));
    return !fix_crc || restamp_crc(bytes);
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[FAIL] %s: %s\n", what,
                 cudaGetErrorString(status));
    ++failures;
    return false;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    bool allocate(std::size_t count, const char* what) {
        return cuda_ok(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count * sizeof(T)), what);
    }
    bool release(const char* what) {
        if (!data_) return true;
        T* data = data_;
        data_ = nullptr;
        return cuda_ok(cudaFree(data), what);
    }
    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

struct FixtureFiles {
    std::filesystem::path directory;
    std::filesystem::path checkpoint;

    ~FixtureFiles() { remove(); }

    bool remove() {
        bool ok = true;
        std::error_code ec;
        if (!checkpoint.empty()) {
            const bool exists = std::filesystem::exists(checkpoint, ec);
            if (ec) {
                ok = false;
            } else if (exists) {
                ok = std::filesystem::remove(checkpoint, ec) && !ec && ok;
            }
        }
        ec.clear();
        if (!directory.empty()) {
            const bool exists = std::filesystem::exists(directory, ec);
            if (ec) {
                ok = false;
            } else if (exists) {
                ok = std::filesystem::remove(directory, ec) && !ec && ok;
            }
        }
        checkpoint.clear();
        directory.clear();
        return ok;
    }
};

// A uniquely owned directory: never a predictable shared path, and its
// creation (not reuse) is verified.
std::filesystem::path make_unique_directory() {
    std::random_device entropy;
    for (int attempt = 0; attempt < 16; ++attempt) {
        char name[64]{};
        std::snprintf(name, sizeof(name), "pf3d_ms_ckpt_%08x%08x",
                      entropy(), entropy());
        const std::filesystem::path candidate =
            std::filesystem::temp_directory_path() / name;
        std::error_code ec;
        if (std::filesystem::create_directory(candidate, ec) && !ec)
            return candidate;
    }
    return {};
}

}  // namespace

int main() {
    int devices = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&devices);
    if (device_status != cudaSuccess) {
        cuda_ok(device_status, "cudaGetDeviceCount");
        return 1;
    }
    if (devices == 0) {
        std::fprintf(stderr,
                     "measure_shards_checkpoint: no CUDA device; skipping\n");
        return 77;  // ctest SKIP_RETURN_CODE
    }

    const pf3d::SimParams3D params = small_params();
    const int B = 144;
    const std::size_t voxels = static_cast<std::size_t>(B) * B * B;
    const std::size_t phi_bytes =
        static_cast<std::size_t>(params.num_cells) * voxels * sizeof(float);

    DeviceBuffer<pf3d::CellState3D> d_cells;
    DeviceBuffer<float> d_phi;
    if (!d_cells.allocate(static_cast<std::size_t>(params.num_cells),
                          "cudaMalloc(cells)") ||
        !d_phi.allocate(phi_bytes / sizeof(float), "cudaMalloc(phi)"))
        return 1;
    std::vector<pf3d::CellState3D> cells(
        static_cast<std::size_t>(params.num_cells));
    for (int n = 0; n < params.num_cells; ++n) {
        pf3d::CellState3D& c = cells[static_cast<std::size_t>(n)];
        std::memset(&c, 0, sizeof(c));
        c.global_id = n;
        c.origin_x = 10 + 160 * n;
        c.origin_y = 12;
        c.origin_z = 14;
        c.bb_lo_x = c.bb_lo_y = c.bb_lo_z = 60;
        c.bb_hi_x = c.bb_hi_y = c.bb_hi_z = 80;
        c.V = 1000.0;
        c.polarity_x = 1.0f;
        c.gamma = 1.0f;
        c.v_A = 0.01f;
        c.R_tgt = 49.0f;
        c.phi_max = 1.0f;
    }
    if (!cuda_ok(cudaMemcpy(d_cells.get(), cells.data(),
                            cells.size() * sizeof(pf3d::CellState3D),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(cells)") ||
        !cuda_ok(cudaMemset(d_phi.get(), 0, phi_bytes), "cudaMemset(phi)"))
        return 1;

    FixtureFiles fixture;
    fixture.directory = make_unique_directory();
    if (fixture.directory.empty()) {
        std::fprintf(stderr, "[FAIL] cannot create a unique directory\n");
        return 1;
    }
    fixture.checkpoint = fixture.directory / "fixture.pf3d";
    const std::string path = fixture.checkpoint.string();

    pf3d::CheckpointWriteView3D view{};
    view.params = &params;
    view.step = 7;
    view.time = 0.07;
    view.brick_edge = B;
    view.trajectory_interval = 17;
    view.base_measure_shards = 11;
    view.d_cells = d_cells.get();
    view.d_phi = d_phi.get();

    // The writer must refuse a view without a resolved count.
    pf3d::CheckpointWriteView3D unresolved = view;
    unresolved.base_measure_shards = 0;
    expect(!pf3d::checkpoint_write_3d(path, unresolved),
           "writer refuses base_measure_shards=0");
    pf3d::CheckpointWriteView3D inconsistent_time = view;
    inconsistent_time.time += params.dt;
    expect(!pf3d::checkpoint_write_3d(path, inconsistent_time),
           "writer refuses time inconsistent with step and dt");

    if (!pf3d::checkpoint_write_3d(path, view)) {
        expect(false, "production writer stores the fixture");
        return 1;
    }

    pf3d::CheckpointMeta3D meta{};
    expect(pf3d::checkpoint_probe_3d(path, &meta) &&
               meta.base_measure_shards == 11 &&
               meta.trajectory_interval == 17,
           "probe returns the stored shard count and trajectory cadence");

    // Programmatic continuations reject invalid metadata before allocation.
    for (int value : {-2, -1, 65}) {
        pf3d::CheckpointMeta3D invalid = meta;
        invalid.base_measure_shards = value;
        pf3d::Sim3D simulation;
        char message[96]{};
        std::snprintf(message, sizeof(message),
                      "init_checkpoint rejects stored value %d", value);
        expect(!simulation.init_checkpoint(invalid, path,
                                           pf3d::RunOptions3D{}),
               message);
    }

    // Full production load, which verifies the file checksum.
    pf3d::CheckpointLoadView3D load{};
    load.d_cells = d_cells.get();
    load.d_phi = d_phi.get();
    expect(pf3d::checkpoint_load_3d(path, meta, load),
           "production loader accepts the intact file");

    std::vector<char> original;
    if (!read_file(path, &original) ||
        original.size() <= sizeof(ckpt3d::FileHeader3D)) {
        expect(false, "fixture can be read and has a payload");
        return 1;
    }

    // Checksum protection through the production loader: the probe (which
    // does not verify the whole-file CRC) must accept the tampered header,
    // and the loader must then independently reject it.
    {
        std::vector<char> tampered = original;
        if (!set_stored_shards(&tampered, 12, false) ||
            !write_file(path, tampered)) {
            expect(false, "tampered fixture can be written");
            return 1;
        }
        pf3d::CheckpointMeta3D m{};
        expect(pf3d::checkpoint_probe_3d(path, &m) &&
                   m.base_measure_shards == 12,
               "probe accepts the tampered header before CRC verification");
        expect(!pf3d::checkpoint_load_3d(path, m, load),
               "loader rejects a count edit that breaks the checksum");
    }

    {
        std::vector<char> inconsistent = original;
        const std::size_t offset = offsetof(ckpt3d::FileHeader3D, time);
        double time = 0.0;
        std::memcpy(&time, inconsistent.data() + offset, sizeof(time));
        time += params.dt;
        std::memcpy(inconsistent.data() + offset, &time, sizeof(time));
        if (!restamp_crc(&inconsistent) || !write_file(path, inconsistent)) {
            expect(false, "inconsistent-time fixture can be written");
            return 1;
        }
        pf3d::CheckpointMeta3D m{};
        expect(!pf3d::checkpoint_probe_3d(path, &m),
               "probe rejects time inconsistent with step and dt");
    }

    // Standard-policy and boundary values through the production reader.
    const std::uint64_t accepted[] = {0, 1, 11, 64};
    for (std::uint64_t value : accepted) {
        std::vector<char> variant = original;
        if (!set_stored_shards(&variant, value, true) ||
            !write_file(path, variant)) {
            expect(false, "restamped fixture can be written");
            return 1;
        }
        pf3d::CheckpointMeta3D m{};
        char message[96]{};
        std::snprintf(message, sizeof(message),
                      "probe accepts stored value %llu",
                      static_cast<unsigned long long>(value));
        expect(pf3d::checkpoint_probe_3d(path, &m) &&
                   m.base_measure_shards == static_cast<int>(value),
               message);
        std::snprintf(message, sizeof(message),
                      "loader accepts restamped stored value %llu",
                      static_cast<unsigned long long>(value));
        expect(pf3d::checkpoint_load_3d(path, m, load), message);
    }
    {
        std::vector<char> variant = original;
        if (!set_stored_shards(&variant, 65, true) ||
            !write_file(path, variant)) {
            expect(false, "invalid-count fixture can be written");
            return 1;
        }
        pf3d::CheckpointMeta3D m{};
        expect(!pf3d::checkpoint_probe_3d(path, &m),
               "probe rejects stored value 65");
    }

    expect(fixture.remove(), "fixture and unique directory removed");
    d_phi.release("cudaFree(phi)");
    d_cells.release("cudaFree(cells)");
    if (failures == 0)
        std::printf("measure_shards_checkpoint: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
