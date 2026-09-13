#include "pf3d/boundary_output.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
namespace wire = pf3d::boundary;
constexpr int B = 8, N = 4, G = 16;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(cudaError_t result, const char* call) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}
#define CUDA(call) checked((call), #call)

template<class T> struct Device {
    T* p = nullptr;
    std::size_t count;
    explicit Device(std::size_t n) : count(n) {
        CUDA(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    }
    ~Device() {
        const cudaError_t status = cudaFree(p);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(status));
            std::abort();
        }
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& values) {
        require(values.size() == count, "upload size");
        CUDA(cudaMemcpy(p, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> values(count);
        CUDA(cudaMemcpy(values.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
    void unchanged(const std::vector<T>& expected) const {
        const auto actual = get();
        require(actual.size() == expected.size() &&
                std::memcmp(actual.data(), expected.data(), count * sizeof(T)) == 0,
                "capture changed input bytes (including cell padding/RNG/polarity)");
    }
};

struct TempDir {
    fs::path path;
    TempDir() {
        std::random_device random;
        for (int attempt = 0; attempt < 32; ++attempt) {
            const fs::path candidate = fs::temp_directory_path() /
                ("pf3d-boundary-test-" + std::to_string(random()) + "-" +
                 std::to_string(random()));
            if (fs::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("cannot create unique test directory");
    }
    ~TempDir() {
        // Only immediate children of the directory we exclusively created;
        // deliberately no recursive removal and no shared/predictable path.
        std::error_code error;
        for (fs::directory_iterator it(path, error), end; !error && it != end;
             it.increment(error)) fs::remove(it->path(), error);
        fs::remove(path, error);
    }
};

std::size_t cube(int edge) { return std::size_t(edge) * edge * edge; }
std::size_t xyz(int edge, int x, int y, int z) {
    return x + std::size_t(edge) * (y + std::size_t(edge) * z);
}
pf3d::SimParams3D parameters(std::uint32_t flags) {
    pf3d::SimParams3D p;
    p.Nx = p.Ny = p.Nz = 32;
    p.num_cells = N;
    p.boundary_flags = flags;
    p.dx = 0.75; p.dy = 1.25; p.dz = 1.5; p.dt = 0.03125; p.tau = 17.5;
    return p;
}

struct Fixture {
    std::vector<pf3d::CellState3D> cells{N};
    std::array<std::vector<float>, N> phi;
    std::vector<float> base = std::vector<float>(N * cube(B), 0.2123f);
    Fixture() {
        std::memset(cells.data(), 0x5a, cells.size() * sizeof(cells[0]));
        for (int i = 0; i < N; ++i) {
            auto& c = cells[i];
            c.global_id = (std::int64_t(1) << 40) + 91 + i * 17;
            c.origin_x = -(std::int64_t(1) << 34) - 3 - i;
            c.origin_y = (std::int64_t(1) << 35) + 31 + i;
            c.origin_z = i < 2 ? -2 : (i == 2 ? 3 : 30);
            c.storage_edge = i == 1 ? G : 0;
            c.gamma = 0.8123f + i; c.v_A = 0.012345f + i; c.R_tgt = 2.75f + i;
            c.polarity_x = -0.25f; c.polarity_y = 0.75f; c.polarity_z = 0.125f;
            c.tumble_ctr = 0xf1234567u + i;
            const int edge = i == 1 ? G : B;
            phi[i].assign(cube(edge), 0.0f);
            auto set = [&](int x, int y, int z, float value) {
                phi[i][xyz(edge, x, y, z)] = value;
            };
            if (i < 2) {
                // Basal support reaches both actual brick faces. Padding must
                // close those contours; an overhang adds a separate footprint.
                set(0, 0, 2, 0.812345f); set(1, 0, 2, 0.5f);
                set(1, 1, 2, 0.49999997f); set(0, 1, 2, 0.50000006f);
                set(0, 2, 2, -0.03125f); set(2, 0, 2, -0.0f);
                set(edge - 1, edge - 1, 2, 0.73125f);
                set(edge - 2, 1, 4, 0.93751f);
                set(3, edge - 2, 0, 0.875123f); // below bounded domain
            } else if (i == 2) set(3, 4, 0, 0.67891f); // off substrate
            else {
                set(1, 1, 0, 0.3f);
                set(5, 6, 5, 0.98761f); // above bounded domain
            }
            if (i != 1)
                std::copy(phi[i].begin(), phi[i].end(), base.begin() + i * cube(B));
        }
    }
    void grow() {
        std::vector<float> larger(cube(G), 0.0f);
        for (int z = 0; z < B; ++z) for (int y = 0; y < B; ++y)
            for (int x = 0; x < B; ++x)
                larger[xyz(G, x + 4, y + 4, z + 4)] = phi[0][xyz(B, x, y, z)];
        larger[xyz(G, G - 1, 0, 6)] = 0.912345f;
        phi[0] = std::move(larger);
        cells[0].storage_edge = G;
        cells[0].origin_x -= 4; cells[0].origin_y -= 4; cells[0].origin_z -= 4;
    }
};

// Independent dense CPU projection and marching-square classification. No
// extractor helpers are reused, and all four float32 samples are compared.
std::vector<wire::Square> reference(const Fixture& f, int i,
        const pf3d::SimParams3D& p, wire::Projection projection) {
    const auto& c = f.cells[i];
    const int edge = c.storage_edge ? int(c.storage_edge) : B, plane = edge + 2;
    std::vector<float> image(std::size_t(plane) * plane, 0.0f);
    for (int y = 0; y < edge; ++y) for (int x = 0; x < edge; ++x) {
        float value = 0.0f;
        bool sampled = false;
        for (int z = 0; z < edge; ++z) {
            const auto world_z = c.origin_z + z;
            if (projection == wire::Projection::Basal && world_z != 0) continue;
            if (p.boundary_flags != pf3d::kBoundaryPeriodicXYZ3D &&
                (world_z < 0 || world_z >= p.Nz)) continue;
            const float sample = f.phi[i][xyz(edge, x, y, z)];
            if (!sampled || sample > value) value = sample;
            sampled = true;
        }
        image[x + 1 + std::size_t(plane) * (y + 1)] = value;
    }
    std::vector<wire::Square> squares;
    for (int y = 0; y < plane - 1; ++y) for (int x = 0; x < plane - 1; ++x) {
        const auto at = x + std::size_t(plane) * y;
        wire::Square s{std::uint32_t(x) | (std::uint32_t(y) << 16),
            {image[at], image[at + 1], image[at + plane + 1], image[at + plane]}};
        const auto extremes = std::minmax_element(s.phi, s.phi + 4);
        if (*extremes.first < 0.5f && *extremes.second >= 0.5f) squares.push_back(s);
    }
    return squares;
}

struct Expected {
    std::uint64_t step;
    double time;
    std::vector<pf3d::CellState3D> cells;
    std::array<std::vector<wire::Square>, N> squares;
    Expected(const Fixture& f, const pf3d::SimParams3D& p,
             wire::Projection projection, std::uint64_t s)
        : step(s), time(s * p.dt), cells(f.cells) {
        for (int i = 0; i < N; ++i) squares[i] = reference(f, i, p, projection);
    }
};

std::vector<char> contents(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "open captured file");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
template<class T> T take(const std::vector<char>& bytes, std::size_t& offset) {
    require(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset,
            "truncated capture");
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}
void verify_file(const fs::path& path, const pf3d::SimParams3D& p,
                 wire::Projection projection, const std::vector<Expected>& frames,
                 bool complete) {
    const auto bytes = contents(path);
    std::size_t offset = 0;
    const auto h = take<wire::FileHeader>(bytes, offset);
    require(std::memcmp(h.magic, "PFB3D1\0", 8) == 0 && h.version == 1 &&
        h.header_bytes == sizeof(h) && h.cells == N && h.codec == 0 &&
        h.projection == std::uint32_t(projection) && h.boundary_flags == p.boundary_flags &&
        h.nx == p.Nx && h.ny == p.Ny && h.nz == p.Nz && h.dx == p.dx &&
        h.dy == p.dy && h.dz == p.dz && h.dt == p.dt && h.tau == p.tau &&
        h.level == 0.5 && h.interval == 7 && h.square_bytes == sizeof(wire::Square) &&
        h.cell_bytes == sizeof(wire::Cell) && h.reserved == 0, "file header metadata");
    for (const auto& expected : frames) {
        const auto frame = take<wire::FrameHeader>(bytes, offset);
        require(std::memcmp(frame.magic, "P3BFRM1", 8) == 0 &&
            frame.step == expected.step && frame.time == expected.time &&
            frame.raw_bytes == frame.stored_bytes && frame.raw_bytes <= bytes.size() - offset,
            "frame metadata/raw codec");
        const auto end = offset + std::size_t(frame.raw_bytes);
        std::uint64_t hash = 14695981039346656037ull;
        for (auto at = offset; at < end; ++at)
            hash = (hash ^ static_cast<unsigned char>(bytes[at])) * 1099511628211ull;
        require(hash == frame.checksum, "frame FNV-1a checksum");
        std::array<wire::Cell, N> records;
        for (auto& record : records) record = take<wire::Cell>(bytes, offset);
        for (int i = 0; i < N; ++i) {
            const auto& c = records[i];
            const auto& source = expected.cells[i];
            const auto edge = source.storage_edge ? source.storage_edge : B;
            require(c.id == source.global_id && c.origin_x == source.origin_x - 1 &&
                c.origin_y == source.origin_y - 1 && c.origin_z == source.origin_z &&
                c.brick_edge == edge && c.plane_edge == edge + 2 &&
                c.squares == expected.squares[i].size() && c.gamma == source.gamma &&
                c.active_speed == source.v_A && c.radius == source.R_tgt &&
                c.reserved == 0 && c.reserved_float == 0.0f, "cell metadata/stable ID");
            std::vector<wire::Square> actual;
            for (std::uint32_t j = 0; j < c.squares; ++j)
                actual.push_back(take<wire::Square>(bytes, offset));
            std::sort(actual.begin(), actual.end(), [](const auto& a, const auto& b) {
                return a.xy < b.xy;
            });
            for (std::size_t j = 0; j < actual.size(); ++j)
                require(std::memcmp(&actual[j], &expected.squares[i][j], sizeof(wire::Square)) == 0,
                        "CPU/GPU crossing coordinate or original float bits differ");
        }
        require(offset == end, "frame payload length");
    }
    if (complete) {
        const auto end = take<wire::FrameHeader>(bytes, offset);
        require(std::memcmp(end.magic, "P3BEND1", 8) == 0 &&
            end.step == frames.back().step && end.time == frames.back().time &&
            end.raw_bytes == 0 && end.stored_bytes == 0 && end.checksum == 0,
            "clean terminator metadata");
    }
    require(offset == bytes.size(), complete ? "bytes after terminator" :
            "failed capture emitted a false clean terminator or partial frame");
}

void run_case(const fs::path& path, std::uint32_t flags,
              wire::Projection projection, bool fail_nonfinite = false) {
    const auto p = parameters(flags);
    Fixture f;
    Device<float> base(f.base.size()), promoted(f.phi[1].size()), grown(cube(G));
    Device<pf3d::CellState3D> cells(N);
    Device<const float*> table(N);
    std::vector<const float*> pointers{nullptr, promoted.p, nullptr, nullptr};
    std::vector<float> grow_image(cube(G), 0.12345f);
    base.put(f.base); promoted.put(f.phi[1]); grown.put(grow_image);
    cells.put(f.cells); table.put(pointers);
    pf3d::BoundaryOutput3D output;
    require(output.open(path.string(), p, projection, 7, false), "open raw output");
    std::vector<Expected> expected;
    auto capture = [&](std::uint64_t step, bool success) {
        require(output.capture(base.p, table.p, cells.p, B, nullptr, step, step * p.dt)
                == success, "capture result");
        CUDA(cudaDeviceSynchronize());
        base.unchanged(f.base); promoted.unchanged(f.phi[1]); grown.unchanged(grow_image);
        cells.unchanged(f.cells); table.unchanged(pointers);
        if (success) expected.emplace_back(f, p, projection, step);
    };
    capture(7, true);
    require(!expected[0].squares[0].empty(), "fixture has face-touching basal support");
    require(expected[0].squares[2].empty() == (projection == wire::Projection::Basal),
            "off-substrate fixture distinguishes projections");
    if (projection == wire::Projection::Maximum)
        require(expected[0].squares[3].empty() == (flags != pf3d::kBoundaryPeriodicXYZ3D),
                "upper clipping fixture distinguishes periodic and bounded z");
    if (fail_nonfinite) {
        f.base[xyz(B, 0, 0, 2)] = std::numeric_limits<float>::quiet_NaN();
        base.put(f.base);
        capture(14, false);
        require(!output.close(true), "failed output cannot close as successful");
    } else {
        f.grow(); grow_image = f.phi[0]; pointers[0] = grown.p;
        grown.put(grow_image); cells.put(f.cells); table.put(pointers);
        capture(14, true);
        require(output.close(), "close complete output");
    }
    verify_file(path, p, projection, expected, !fail_nonfinite);
}
} // namespace

int main() {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) {
            std::puts("SKIP: no usable CUDA device"); return 77;
        }
        checked(status, "cudaGetDeviceCount");
        CUDA(cudaSetDevice(0));
        TempDir temp;
        run_case(temp.path / "basal.raw", pf3d::kBoundarySubstrateSlab3D, wire::Projection::Basal);
        run_case(temp.path / "maximum.raw", pf3d::kBoundarySubstrateSlab3D, wire::Projection::Maximum);
        run_case(temp.path / "periodic.raw", pf3d::kBoundaryPeriodicXYZ3D, wire::Projection::Maximum);
        run_case(temp.path / "nonfinite.raw", pf3d::kBoundarySubstrateSlab3D, wire::Projection::Maximum, true);
        auto p = parameters(pf3d::kBoundarySubstrateSlab3D);
        auto rejected = [&](const fs::path& path, wire::Projection projection) {
            pf3d::BoundaryOutput3D invalid;
            return !invalid.open(path.string(), p, projection, 7, false);
        };
        require(rejected(temp.path / "absent" / "file.raw", wire::Projection::Basal),
            "missing parent path must fail");
        const auto before = contents(temp.path / "basal.raw");
        require(rejected(temp.path / "basal.raw", wire::Projection::Basal),
            "existing file must be refused");
        require(contents(temp.path / "basal.raw") == before, "existing file changed");
        require(rejected(temp.path / "invalid.raw", static_cast<wire::Projection>(99)),
            "invalid projection must fail");
        p.boundary_flags = pf3d::kBoundaryPeriodicXYZ3D;
        require(rejected(temp.path / "invalid.raw", wire::Projection::Basal),
            "periodic basal projection must fail");
        require(!fs::exists(temp.path / "invalid.raw"), "invalid projection created output");
        std::puts("boundary_output_gpu: all checks passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "boundary_output_gpu: %s\n", error.what());
        return 1;
    }
}
