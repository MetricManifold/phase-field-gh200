#include "boundary.cuh"
#include "boundary_format.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef PF_BOUNDARY_ZSTD
#include <zstd.h>
#endif

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct DeviceState {
    float* phi = nullptr;
    pf::CellState* cells = nullptr;
    ~DeviceState() { cudaFree(phi); cudaFree(cells); }
};
uint64_t checksum(const std::vector<unsigned char>& bytes) {
    uint64_t h = 14695981039346656037ull;
    for (auto v : bytes) { h ^= v; h *= 1099511628211ull; }
    return h;
}
template<class T> T read(std::ifstream& in) {
    T value{};
    require(bool(in.read(reinterpret_cast<char*>(&value), sizeof(value))), "short record");
    return value;
}

void verify(const std::filesystem::path& path, const pf::SimParams& p,
            const std::vector<pf::CellState>& cells,
            const std::vector<std::vector<pf::boundary::Square>>& expected,
            bool compressed) {
    std::ifstream in(path, std::ios::binary);
    const auto file = read<pf::boundary::FileHeader>(in);
    require(!std::memcmp(file.magic, "PFBND01", 8) && file.version == 1 &&
            file.header_bytes == 72 && file.cells == cells.size() &&
            file.side == static_cast<unsigned>(p.Nx) && file.tile_pitch == pf::kTilePitch &&
            file.square_bytes == 20 && file.dt == p.dt && file.tau == p.tau &&
            file.interval == 1000 && file.level == 0.5 && file.codec == unsigned(compressed),
            "file header");
    for (uint64_t step : {0ull, 1000ull, 1503ull}) {
        const auto h = read<pf::boundary::FrameHeader>(in);
        require(!std::memcmp(h.magic, "PFBFRM1", 8) && h.step == step &&
                h.time == step*p.dt && h.raw_bytes < (1ull<<28) &&
                h.stored_bytes < (1ull<<28), "frame header");
        std::vector<unsigned char> stored(h.stored_bytes), payload(h.raw_bytes);
        require(bool(in.read(reinterpret_cast<char*>(stored.data()), stored.size())), "payload");
        if (compressed) {
#ifdef PF_BOUNDARY_ZSTD
            require(ZSTD_decompress(payload.data(), payload.size(), stored.data(), stored.size()) ==
                    payload.size(), "decompression");
#else
            require(false, "unexpected compressed fixture");
#endif
        } else {
            require(stored.size() == payload.size(), "raw size");
            payload = std::move(stored);
        }
        require(checksum(payload) == h.checksum, "checksum");
        size_t offset = cells.size()*sizeof(pf::boundary::Cell);
        for (size_t i = 0; i < cells.size(); ++i) {
            pf::boundary::Cell m{};
            std::memcpy(&m, payload.data()+i*sizeof(m), sizeof(m));
            const auto& c = cells[i];
            const auto sc = pf::class_of(c.cls);
            const int ox = ((c.gx0-sc.tx0)%p.Nx+p.Nx)%p.Nx;
            const int oy = ((c.gy0-sc.ty0)%p.Ny+p.Ny)%p.Ny;
            require(m.id == c.global_id && m.origin_x == ox && m.origin_y == oy &&
                    m.gamma == c.gamma && m.mobility == 0.5f &&
                    m.active_speed == c.v_A && m.radius == c.R_tgt &&
                    m.squares == expected[i].size(), "cell metadata and periodic tile origin");
            require(m.cx >= 0 && m.cx < p.Nx && m.cy >= 0 && m.cy < p.Ny,
                    "periodic centroid");
            const size_t bytes = m.squares*sizeof(pf::boundary::Square);
            require(offset+bytes <= payload.size(), "square bounds");
            std::vector<pf::boundary::Square> actual(m.squares);
            std::memcpy(actual.data(), payload.data()+offset, bytes);
            std::sort(actual.begin(), actual.end(), [](const auto& a, const auto& b) {
                return a.xy < b.xy;
            });
            require(!std::memcmp(actual.data(), expected[i].data(), bytes),
                    "all crossing squares preserve coordinates and original float bits");
            offset += bytes;
        }
        require(offset == payload.size(), "no trailing payload");
    }
    const auto end = read<pf::boundary::FrameHeader>(in);
    require(!std::memcmp(end.magic, "PFBEND1", 8) && end.checksum == 3 &&
            end.step == 1503 && end.time == 1503*p.dt && in.peek() == EOF,
            "completion record");
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const auto directory = std::filesystem::temp_directory_path() /
            ("pf-boundary-"+std::to_string(stamp));
        require(std::filesystem::create_directory(directory), "new fixture directory");
        pf::SimParams p{};
        p.num_cells = pf::kNumClasses; p.Nx = p.Ny = 550;
        std::vector<pf::CellState> cells(p.num_cells);
        std::vector<float> phi(size_t(p.num_cells)*pf::kTileArea, 0.125f);
        std::vector<std::vector<pf::boundary::Square>> expected(p.num_cells);
        for (int i = 0; i < p.num_cells; ++i) {
            auto& c = cells[i];
            c.global_id = i; c.cls = static_cast<uint8_t>(i);
            c.gx0 = i%2 ? 549 : 0; c.gy0 = i%3 ? 549 : 0;
            c.V = 1; c.Cx = 2; c.Cy = 3;
            c.gamma = i%2 ? 0.35f : 1.f; c.R_tgt = 49; c.v_A = 0.01f;
            float* tile = phi.data()+size_t(i)*pf::kTileArea;
            for (int y = 64; y < 210; ++y)
                for (int x = 64; x < 210; ++x)
                    tile[y*pf::kTilePitch+x] = float((x*y+3*i)%17)/16.f;
            for (int y = 0; y < pf::kTilePitch-1; ++y) {
                for (int x = 0; x < pf::kTilePitch-1; ++x) {
                    const int j = y*pf::kTilePitch+x;
                    pf::boundary::Square s{uint32_t(x)|(uint32_t(y)<<16),
                        {tile[j], tile[j+1], tile[j+pf::kTilePitch+1], tile[j+pf::kTilePitch]}};
                    bool below = false, above = false;
                    for (float v : s.phi) { below |= v < 0.5f; above |= v >= 0.5f; }
                    if (below && above) expected[i].push_back(s);
                }
            }
        }
        DeviceState d;
        require(cudaMalloc(&d.phi, phi.size()*sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d.cells, cells.size()*sizeof(pf::CellState)) == cudaSuccess, "device allocation");
        require(cudaMemcpy(d.phi, phi.data(), phi.size()*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(d.cells, cells.data(), cells.size()*sizeof(pf::CellState), cudaMemcpyHostToDevice) == cudaSuccess,
                "fixture upload");
        std::vector<bool> codecs{false};
#ifdef PF_BOUNDARY_ZSTD
        codecs.push_back(true);
#else
        pf::BoundaryOutput unavailable;
        const auto unavailable_path = directory/"unavailable.pfb";
        require(!unavailable.open(unavailable_path.string(), p, p.Nx, 1000, true) &&
                !std::filesystem::exists(unavailable_path), "missing compression support fails before file creation");
#endif
        for (bool compressed : codecs) {
            const auto path = directory/(compressed ? "zstd.pfb" : "raw.pfb");
            pf::BoundaryOutput out;
            require(out.open(path.string(), p, p.Nx, 1000, compressed), "open");
            for (long long step : {0ll, 1000ll, 1503ll})
                require(out.capture(d.phi, d.cells, nullptr, step, step*p.dt), "capture");
            require(out.close(), "close");
            verify(path, p, cells, expected, compressed);
            pf::BoundaryOutput duplicate;
            require(!duplicate.open(path.string(), p, p.Nx, 1000, compressed), "refuse overwrite");
            verify(path, p, cells, expected, compressed);
            require(std::filesystem::remove(path), "remove known fixture");
        }
        std::vector<float> after(phi.size());
        std::vector<pf::CellState> cells_after(cells.size());
        require(cudaMemcpy(after.data(), d.phi, after.size()*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(cells_after.data(), d.cells, cells_after.size()*sizeof(pf::CellState), cudaMemcpyDeviceToHost) == cudaSuccess,
                "state readback");
        require(!std::memcmp(after.data(), phi.data(), phi.size()*sizeof(float)) &&
                !std::memcmp(cells_after.data(), cells.data(), cells.size()*sizeof(pf::CellState)),
                "export does not change fields or cell state");
        const auto broken = directory/"nonfinite.pfb";
        pf::BoundaryOutput invalid;
        require(invalid.open(broken.string(), p, p.Nx, 1000, false), "open invalid fixture");
        const float nan = std::numeric_limits<float>::quiet_NaN();
        require(cudaMemcpy(d.phi, &nan, sizeof(nan), cudaMemcpyHostToDevice) == cudaSuccess, "inject NaN");
        require(!invalid.capture(d.phi, d.cells, nullptr, 0, 0) && !invalid.close(), "reject nonfinite field");
        require(std::filesystem::file_size(broken) == sizeof(pf::boundary::FileHeader), "failed capture has no completion record");
        require(std::filesystem::remove(broken) && std::filesystem::is_empty(directory) &&
                std::filesystem::remove(directory), "remove empty fixture directory");
        std::puts("boundary output: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "boundary output: FAIL: %s\n", e.what());
        return 1;
    }
}
