#include "../include/pf3d/palmieri_initializer.hpp"
#include "../include/pf3d/params.cuh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
        "generate: %s --N <int> (--side <L> | --rho <f> --cell-radius <R>) "
        "--radius <min separation> --seed <u64> --out <initial_centres.csv> "
        "[--force]\n"
        "validate: %s --N <int> (--side <L> | --rho <f> --cell-radius <R>) "
        "--radius <min separation> --validate <initial_centres.csv>\n"
        "--rho/--cell-radius derive the cubic side exactly as cell_gh200_3d "
        "does for a fresh run.\n", argv0, argv0);
}

bool parse_i(const char* text, int* value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed < 1 ||
        parsed > std::numeric_limits<int>::max()) return false;
    *value = (int)parsed;
    return true;
}

bool parse_u64(const char* text, std::uint64_t* value) {
    if (text[0] == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0') return false;
    *value = (std::uint64_t)parsed;
    return true;
}

bool parse_d(const char* text, double* value) {
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno || end == text || *end != '\0' || !std::isfinite(parsed) ||
        !(parsed > 0.0)) return false;
    *value = parsed;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    double side = 0.0, radius = 0.0, rho = 0.0, cell_radius = 0.0;
    std::uint64_t seed = 0;
    bool have_seed = false, force = false;
    std::string output, validation_input;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        auto need = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[fatal] %s requires a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(flag, "-h") || !std::strcmp(flag, "--help")) {
            usage(argv[0]);
            return 0;
        } else if (!std::strcmp(flag, "--N")) {
            if (!parse_i(need(), &n)) { std::fprintf(stderr, "[fatal] invalid --N\n"); return 2; }
        } else if (!std::strcmp(flag, "--side")) {
            if (!parse_d(need(), &side)) { std::fprintf(stderr, "[fatal] invalid --side\n"); return 2; }
        } else if (!std::strcmp(flag, "--rho")) {
            if (!parse_d(need(), &rho)) { std::fprintf(stderr, "[fatal] invalid --rho\n"); return 2; }
        } else if (!std::strcmp(flag, "--cell-radius")) {
            if (!parse_d(need(), &cell_radius)) { std::fprintf(stderr, "[fatal] invalid --cell-radius\n"); return 2; }
        } else if (!std::strcmp(flag, "--radius")) {
            if (!parse_d(need(), &radius)) { std::fprintf(stderr, "[fatal] invalid --radius\n"); return 2; }
        } else if (!std::strcmp(flag, "--seed")) {
            if (!parse_u64(need(), &seed)) { std::fprintf(stderr, "[fatal] invalid --seed\n"); return 2; }
            have_seed = true;
        } else if (!std::strcmp(flag, "--out")) {
            output = need();
        } else if (!std::strcmp(flag, "--validate")) {
            validation_input = need();
        } else if (!std::strcmp(flag, "--force")) {
            force = true;
        } else {
            std::fprintf(stderr, "[fatal] unknown option %s\n", flag);
            return 2;
        }
    }
    const bool derive_side = rho > 0.0 || cell_radius > 0.0;
    if (derive_side) {
        if (side > 0.0 || !(rho > 0.0) || !(cell_radius > 0.0)) {
            usage(argv[0]);
            return 2;
        }
        const int derived = pf3d::domain_side_for(n, cell_radius, rho);
        if (derived <= 0) {
            std::fprintf(stderr,
                "[fatal] could not derive a finite cubic domain from "
                "--rho/--cell-radius\n");
            return 2;
        }
        side = (double)derived;
    }
    if (n < 1 || !(side > 0.0) || !(radius > 0.0) ||
        (output.empty() == validation_input.empty()) ||
        (!validation_input.empty() && (have_seed || force)) ||
        (!output.empty() && !have_seed)) {
        usage(argv[0]);
        return 2;
    }
    if (!validation_input.empty()) {
        std::vector<float> x, y, z;
        pf3d::PalmieriCentres3DCsvDiagnostics checked{};
        std::string error;
        if (!pf3d::palmieri3d_read_centres_csv(validation_input, n, side,
                                               radius, &x, &y, &z, &checked,
                                               &error)) {
            std::fprintf(stderr, "[fatal] %s\n", error.c_str());
            return 3;
        }
        std::printf("method=%s\n", pf3d::kPalmieriInitializer3DMethod);
        std::printf("mode=validate\n");
        std::printf("N=%d L=%.17g R=%.17g\n", n, side, radius);
        std::printf("accepted=%zu minimum_periodic_distance=%.17g\n",
                    checked.accepted_count, checked.minimum_periodic_distance);
        std::printf("table_fnv1a64=%016llx\n",
                    (unsigned long long)checked.table_fnv1a64);
        std::printf("initial_centres_csv=%s\n", validation_input.c_str());
        return 0;
    }
    if (!force) {
        std::ifstream existing(output, std::ios::binary);
        if (existing.good()) {
            std::fprintf(stderr,
                "[fatal] output exists; pass --force to replace it: %s\n",
                output.c_str());
            return 2;
        }
    }

    std::vector<float> x, y, z;
    pf3d::PalmieriInit3DDiagnostics generated{};
    try {
        pf3d::palmieri3d_sequential_centres(n, side, radius, seed,
                                            &x, &y, &z, &generated);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fatal] %s\n", e.what());
        return 3;
    }
    pf3d::PalmieriCentres3DCsvDiagnostics before{};
    std::string error;
    if (!pf3d::palmieri3d_validate_centres(x, y, z, n, side, radius, &before,
                                           &error)) {
        std::fprintf(stderr,
                     "[fatal] generated table failed validation: %s\n",
                     error.c_str());
        return 3;
    }
    if (!pf3d::palmieri3d_write_centres_csv(output, x, y, z, &error)) {
        std::fprintf(stderr, "[fatal] %s\n", error.c_str());
        return 4;
    }
    std::vector<float> round_x, round_y, round_z;
    pf3d::PalmieriCentres3DCsvDiagnostics after{};
    if (!pf3d::palmieri3d_read_centres_csv(output, n, side, radius, &round_x,
                                           &round_y, &round_z, &after,
                                           &error) ||
        round_x != x || round_y != y || round_z != z ||
        before.table_fnv1a64 != after.table_fnv1a64) {
        std::fprintf(stderr,
                     "[fatal] written CSV did not round-trip exactly: %s\n",
                     error.c_str());
        return 4;
    }
    std::printf("method=%s\n", pf3d::kPalmieriInitializer3DMethod);
    std::printf("N=%d L=%.17g R=%.17g seed=%llu\n", n, side, radius,
                (unsigned long long)seed);
    std::printf("accepted=%zu candidates=%llu rejected=%llu "
                "minimum_periodic_distance=%.17g\n",
                after.accepted_count,
                (unsigned long long)generated.candidates_drawn,
                (unsigned long long)generated.candidates_rejected,
                after.minimum_periodic_distance);
    std::printf("table_fnv1a64=%016llx\n",
                (unsigned long long)after.table_fnv1a64);
    std::printf("initial_centres_csv=%s\n", output.c_str());
    return 0;
}
