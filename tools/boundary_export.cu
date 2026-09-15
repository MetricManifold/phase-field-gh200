// Exercise boundary output from saved states without launching the time-step kernel.
#include "boundary.cuh"
#include "checkpoint.cuh"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4 || (argc == 4 && std::strcmp(argv[3], "zstd"))) {
        std::fprintf(stderr, "usage: boundary_export CHECKPOINT_OR_DIRECTORY NEW_OUTPUT [zstd]\n");
        return 2;
    }
    std::vector<std::filesystem::path> paths;
    const std::filesystem::path input(argv[1]);
    if (std::filesystem::is_directory(input)) {
        for (const auto& e : std::filesystem::directory_iterator(input)) {
            const auto name = e.path().filename().string();
            if (name.rfind("checkpoint_", 0) == 0 && e.path().extension() == ".bin") paths.push_back(e.path());
        }
        std::sort(paths.begin(),paths.end());
    } else paths.push_back(input);
    if (paths.empty()) return 2;
    pf::BoundaryOutput output;
    float* phi = nullptr; pf::CellState* cells = nullptr;
    int n = 0;
    int first_side = 0;
    double first_dt = 0, first_tau = 0;
    long long previous_step = -1;
    for (size_t k = 0; k < paths.size(); ++k) {
        pf::CheckpointData ck;
        if (!pf::checkpoint_read(paths[k].string(), &ck)) return 3;
        const int side = ck.params.Nx;
        if (k == 0) {
            n = ck.n;
            first_side = side; first_dt = ck.params.dt; first_tau = ck.params.tau;
            long long interval = 1;
            if (paths.size() > 1) {
                std::ifstream next(paths[1], std::ios::binary);
                int32_t next_step = 0;
                next.seekg(8); next.read(reinterpret_cast<char*>(&next_step), sizeof(next_step));
                if (!next || next_step <= ck.step) return 4;
                interval = next_step - ck.step;
            }
            if (!output.open(argv[2], ck.params, side, interval, argc == 4)) return 4;
            if (cudaMalloc((void**)&phi, ck.phi.size()*sizeof(float)) != cudaSuccess ||
                cudaMalloc((void**)&cells, n*sizeof(pf::CellState)) != cudaSuccess) return 5;
        }
        if (ck.n != n || side != first_side || ck.params.dt != first_dt ||
            ck.params.tau != first_tau || ck.step <= previous_step) {
            std::fprintf(stderr, "inconsistent geometry, time units or checkpoint ordering\n");
            return 6;
        }
        previous_step = ck.step;
        std::vector<pf::CellState> state(n);
        for (int i = 0; i < n; ++i) {
            const auto& c = ck.cells[i]; auto& s = state[i];
            s.global_id = c.global_id; s.gx0 = c.origin[0]; s.gy0 = c.origin[1];
            s.cls = c.cls; s.gamma = c.gamma; s.v_A = c.v_A; s.R_tgt = c.R_tgt;
            s.M_pf = c.M_pf;
            s.V = c.volume_moment; s.Cx = c.moment_x; s.Cy = c.moment_y;
        }
        if (cudaMemcpy(phi, ck.phi.data(), ck.phi.size()*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(cells, state.data(), n*sizeof(pf::CellState), cudaMemcpyHostToDevice) != cudaSuccess) return 7;
        if (!output.capture(phi,cells,nullptr,ck.step,ck.t)) return 8;
    }
    if (!output.close()) return 9;
    cudaFree(phi); cudaFree(cells);
    return 0;
}
