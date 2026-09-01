// Command-line interface for the three-dimensional phase-field simulator.

#include "pf3d/checkpoint.cuh"
#include "pf3d/measure_shards.hpp"
#include "pf3d/params.cuh"
#include "pf3d/sim.cuh"

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>

namespace {

using pf3d::RunOptions3D;
using pf3d::SimParams3D;
using pf3d::StorageMode3D;

void usage(const char* program) {
    std::printf(
        "Three-dimensional active phase-field cell simulator\n"
        "\n"
        "usage: %s [options]\n"
        "\n"
        "model and initial condition (fresh runs)\n"
        "  --geometry <periodic|slab|channel> boundary geometry             (periodic)\n"
        "  --slab-height <int>       substrate-slab Nz; default is 2*automatic B\n"
        "  --channel-height <int>    hard-wall separation in voxels; default ceil(2R)\n"
        "  --wall-kappa <f>          channel wall repulsion; default --kappa\n"
        "  --wall-width <f>          channel wall profile width; default --lambda\n"
        "  --wall-padding <int>      solid voxels outside each channel wall;\n"
        "                            minimum/default ceil(3*wall_width)\n"
        "  --N <int>                 number of cells                         (288)\n"
        "  --radius <f>              sphere or slab cap/footprint radius R    (49)\n"
        "  --rho <f>                 target periodic/channel rho_V or slab rho_A (0.90)\n"
        "  --lambda <f>              interface-width parameter               (7)\n"
        "  --kappa <f>               cell-cell repulsion strength            (10)\n"
        "  --mu <f>                  volume-constraint strength              (1)\n"
        "  --xi <f>                  friction                                (1500)\n"
        "  --tau <f>                 mean run time                            (1e4)\n"
        "  --v-A <f>                 active speed                            (1e-2)\n"
        "  --v-A-sigma <f>           lognormal spread in per-cell active speed (0)\n"
        "  --gamma <f>               normal-cell deformability coefficient   (1)\n"
        "  --gamma-cancer <f>        soft-cell deformability coefficient      (0.35)\n"
        "  --cancer-fraction <f>     fraction assigned the soft coefficient  (0)\n"
        "  --seed <u64>              placement and cell-property seed        (1234)\n"
        "  --polarity-seed <u64>     polarity/tumble stream; 0 follows --seed (0)\n"
        "  --initial-centres <csv>   periodic/multilayer channel: id,x,y,z;\n"
        "                            slab/one-layer channel: id,x,y\n"
        "\n"
        "integration and storage\n"
        "  --dt <f>                  explicit-Euler time step                (0.01)\n"
        "  --t-end <f>               target simulation time                  (100)\n"
        "  --aging-time <f>          initial passive duration                (0)\n"
        "  --full-moment <int>       stored full-moment cadence              (100)\n"
        "  --verify-every <int>      strict-check cadence                     (4096)\n"
        "  --memory-mode <name>      auto, throughput, balanced, or compact  (auto)\n"
        "  --memory-fraction <f>     usable fraction of currently free HBM   (0.95)\n"
        "  --brick-edge <int>        aligned brick edge; 0 selects minimum   (0)\n"
        "  --scratch-slots <int>     in-place scratch bricks; 0 selects auto (0)\n"
        "  --device <int>            CUDA device                             (0)\n"
        "  --bench <int>             time this many steps, then exit\n"
        "  --bench-phases            with --bench: also report per-phase\n"
        "                            CUDA-event timings from a second window\n"
        "  --promoted-shards <int>   fast promoted-update CTAs per cell;\n"
        "                            0 selects the occupancy-derived count (0)\n"
        "  --fast-base-shards <int>  fast base-update CTAs per cell override;\n"
        "                            0 keeps the standard policy         (0)\n"
        "  --measure-shards <int>    base measurement CTAs per cell; 0 keeps\n"
        "                            the standard at-most-four wave-fitting\n"
        "                            policy, -1 raises its cap to 64; 1..64 pins a\n"
        "                            count. The resolved count is stored in\n"
        "                            checkpoints and trajectory headers, and\n"
        "                            resumes must match the stored grouping\n"
        "                            or omit the option                  (0)\n"
        "  --promoted-measure-shards <int>  promoted-measurement CTAs per\n"
        "                            cell; 0 keeps the bitwise one-CTA fold,\n"
        "                            -1 selects the occupancy-derived count;\n"
        "                            sharded folds change reduction grouping\n"
        "                            and are not bitwise vs the 1-CTA fold (0)\n"
        "  --strict                  enable scheduled invariant verification\n"
        "  --print-interval <int>    status cadence in steps; 0 disables     (100)\n"
        "\n"
        "trajectory output\n"
        "  --out <path>              plain-text 3-D trajectory\n"
        "  --trajectory-samples <n>  approximately evenly spaced frames     (100)\n"
        "  --trajectory-interval <n> exact frame cadence; overrides samples\n"
        "\n"
        "checkpointing (current PF3D format)\n"
        "  -c, --checkpoint <path>   resume the complete stored state\n"
        "  --checkpoint-interval <n> rolling checkpoint cadence in steps\n"
        "  --save-interval <n>       tagged-checkpoint cadence in steps\n"
        "  --checkpoint-dir <path>   checkpoint output directory\n"
        "  --no-final-checkpoint     omit the rolling checkpoint at exit\n"
        "\n"
        "On resume, stored model parameters and verification cadence remain\n"
        "fixed. An explicit --t-end may extend the run; device/memory choices,\n"
        "output/checkpoint schedules, and --strict may be changed. An explicit\n"
        "brick edge must match the checkpoint. Measurement groupings are\n"
        "restored; an explicitly conflicting shard policy is rejected.\n"
        "For a slab, --rho targets rho_A when deriving integer Lx; the realized\n"
        "rho_A=N*pi*R^2/Lx^2 is reported. Nz is independent of rho_A,\n"
        "and V0 is the neutral 90-degree hemispherical cap 2*pi*R^3/3.\n"
        "For a channel, --rho likewise targets rho_V; the realized value is\n"
        "N*(4*pi*R^3/3)/(Lx^2*H), where H is the\n"
        "accessible wall separation. Smooth static repulsive wall fields are\n"
        "resolved outside the slit; cells retain full 3-D motion and polarity.\n"
        "\n"
        "  -h, --help                show this message\n",
        program);
}

bool parse_double(const char* flag, const char* text, double minimum,
                  double maximum, double* value) {
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
        std::fprintf(stderr,
                     "[3d] %s expects a finite number in [%.9g, %.9g], got '%s'\n",
                     flag, minimum, maximum, text);
        return false;
    }
    *value = parsed;
    return true;
}

bool parse_integer(const char* flag, const char* text, long long minimum,
                   long long maximum, long long* value) {
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        std::fprintf(stderr,
                     "[3d] %s expects an integer in [%lld, %lld], got '%s'\n",
                     flag, minimum, maximum, text);
        return false;
    }
    *value = parsed;
    return true;
}

bool parse_u64(const char* flag, const char* text,
               unsigned long long* value) {
    if (text[0] == '-') {
        std::fprintf(stderr, "[3d] %s expects an unsigned integer, got '%s'\n",
                     flag, text);
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "[3d] %s expects an unsigned integer, got '%s'\n",
                     flag, text);
        return false;
    }
    *value = parsed;
    return true;
}

std::filesystem::path default_checkpoint_directory(
    const std::string& trajectory, const std::string& checkpoint) {
    std::filesystem::path base;
    if (!trajectory.empty()) base = std::filesystem::path(trajectory).parent_path();
    if (base.empty() && !checkpoint.empty())
        base = std::filesystem::path(checkpoint).parent_path();
    return base.empty() ? std::filesystem::path(".") : base;
}

void print_configuration(const SimParams3D& p, const RunOptions3D& options,
                         int brick_edge, bool resumed) {
    const pf3d::BrickSizing3D sizing = pf3d::brick_sizing(
        p.target_radius, p.lambda);
    std::printf(
        "--- 3D model ---\n"
        "  %s  geometry=%s  N=%d  domain=%dx%dx%d  %s target/realized=%.9g/%.9g\n"
        "  R=%.9g  lambda=%.9g  V0=%.9g  B=%d%s\n"
        "  dt=%.9g  t_end=%.9g  steps=%lld  aging=%.9g (%lld steps)\n"
        "  kappa=%.9g  mu=%.9g  xi=%.9g\n"
        "  tau=%.9g  v_A=%.9g  sigma=%.9g  gamma=(%.9g, %.9g)  soft=%.9g\n"
        "  seed=%llu  polarity_seed=%llu  storage=%s  memory_fraction=%.3f\n",
        resumed ? "resume" : "fresh",
        p.substrate_slab() ? "slab"
            : p.hard_wall_channel() ? "channel" : "periodic",
        p.num_cells, p.Nx, p.Ny, p.Nz,
        p.substrate_slab() ? "rho_A" : "rho_V", p.rho, p.realized_rho(),
        p.target_radius, p.lambda, p.volume0(), brick_edge,
        resumed ? " (checkpoint)"
                : options.brick_edge > 0 ? " (explicit)" : " (model-derived)",
        p.dt,
        p.t_end, p.total_steps(), p.aging_time, p.aging_steps(),
        p.kappa, p.mu, p.xi, p.tau, p.v_A,
        p.v_A_sigma, p.gamma_normal, p.gamma_cancer, p.cancer_fraction,
        static_cast<unsigned long long>(p.seed),
        static_cast<unsigned long long>(p.polarity_stream()),
        pf3d::storage_mode_name(options.storage_mode), options.memory_fraction);
    std::printf(
        "  support estimate %.4f voxels + %.4f safety -> minimum aligned B=%d\n",
        sizing.physical_support_extent, sizing.safety_margin, sizing.edge);
    if (p.resolved_wall_channel())
        std::printf(
            "  channel H=%d, padding=%d per wall, kappa_w=%.9g, lambda_w=%.9g\n",
            p.channel_height, p.channel_padding, p.wall_kappa, p.wall_width);
}

struct ParsedCommand {
    SimParams3D params{};
    RunOptions3D options{};
    std::string checkpoint_input;
    bool checkpoint_directory_supplied = false;
    bool checkpoint_schedule_supplied = false;
    bool fresh_parameter_supplied = false;
    bool initial_centres_supplied = false;
    bool t_end_supplied = false;
    bool print_interval_supplied = false;
    bool trajectory_samples_supplied = false;
    bool trajectory_interval_supplied = false;
    bool slab_height_supplied = false;
    int slab_height = 0;
    bool channel_height_supplied = false;
    int channel_height = 0;
    bool wall_kappa_supplied = false;
    bool wall_width_supplied = false;
    bool wall_padding_supplied = false;
    int wall_padding = 0;
};

struct PreparedRun {
    SimParams3D effective{};
    pf3d::CheckpointMeta3D checkpoint{};
    int effective_brick = 0;
    bool resumed = false;
};

enum class CommandStatus {
    Ready,
    Help,
    UsageError,
    RuntimeError
};

enum class OptionStatus {
    Unrecognized,
    Accepted,
    Error
};

const char* value_after(int argc, char** argv, int* index,
                        const char* argument) {
    if (*index + 1 >= argc) {
        std::fprintf(stderr, "[3d] %s requires a value\n", argument);
        return nullptr;
    }
    return argv[++*index];
}

struct DoubleParameterOption {
    const char* name;
    double minimum;
    double maximum;
    double SimParams3D::*member;
};

struct IntegerParameterOption {
    const char* name;
    long long minimum;
    long long maximum;
    int SimParams3D::*member;
};

OptionStatus parse_fresh_parameter_option(int argc, char** argv, int* index,
                                          const char* argument,
                                          ParsedCommand* command) {
    SimParams3D& params = command->params;
    const char* value = nullptr;
    long long integer = 0;

    if (std::strcmp(argument, "--geometry") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value) return OptionStatus::Error;
        if (std::strcmp(value, "periodic") == 0) {
            params.boundary_flags = pf3d::kBoundaryPeriodicXYZ3D;
        } else if (std::strcmp(value, "slab") == 0) {
            params.boundary_flags = pf3d::kBoundarySubstrateSlab3D;
        } else if (std::strcmp(value, "channel") == 0) {
            params.boundary_flags = pf3d::kBoundaryHardWallChannel3D;
        } else {
            std::fprintf(stderr,
                "[3d] --geometry expects periodic, slab, or channel, got '%s'\n",
                value);
            return OptionStatus::Error;
        }
        command->fresh_parameter_supplied = true;
        return OptionStatus::Accepted;
    }

    if (std::strcmp(argument, "--slab-height") == 0 ||
        std::strcmp(argument, "--channel-height") == 0 ||
        std::strcmp(argument, "--wall-padding") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(argument, value, 1,
                                     std::numeric_limits<int>::max(),
                                     &integer)) {
            return OptionStatus::Error;
        }
        const int parsed = static_cast<int>(integer);
        if (std::strcmp(argument, "--slab-height") == 0) {
            command->slab_height = parsed;
            command->slab_height_supplied = true;
        } else if (std::strcmp(argument, "--channel-height") == 0) {
            command->channel_height = parsed;
            command->channel_height_supplied = true;
        } else {
            command->wall_padding = parsed;
            command->wall_padding_supplied = true;
        }
        command->fresh_parameter_supplied = true;
        return OptionStatus::Accepted;
    }

    if (std::strcmp(argument, "--wall-kappa") == 0 ||
        std::strcmp(argument, "--wall-width") == 0) {
        value = value_after(argc, argv, index, argument);
        double* target = std::strcmp(argument, "--wall-kappa") == 0
            ? &params.wall_kappa : &params.wall_width;
        const double minimum = std::strcmp(argument, "--wall-kappa") == 0
            ? 1.0e-12 : 1.0e-6;
        const double maximum = std::strcmp(argument, "--wall-kappa") == 0
            ? 1.0e12 : 1.0e6;
        if (!value || !parse_double(argument, value, minimum, maximum, target))
            return OptionStatus::Error;
        if (std::strcmp(argument, "--wall-kappa") == 0)
            command->wall_kappa_supplied = true;
        else
            command->wall_width_supplied = true;
        command->fresh_parameter_supplied = true;
        return OptionStatus::Accepted;
    }

    static const DoubleParameterOption double_options[] = {
        {"--radius", 1.0e-6, 1.0e6, &SimParams3D::target_radius},
        {"--rho", 1.0e-9, 0.999999999, &SimParams3D::rho},
        {"--lambda", 1.0e-6, 1.0e6, &SimParams3D::lambda},
        {"--kappa", 0.0, 1.0e12, &SimParams3D::kappa},
        {"--mu", 0.0, 1.0e12, &SimParams3D::mu},
        {"--xi", 1.0e-12, 1.0e15, &SimParams3D::xi},
        {"--tau", 1.0e-12, 1.0e15, &SimParams3D::tau},
        {"--v-A", 0.0, 1.0e9, &SimParams3D::v_A},
        {"--v-A-sigma", 0.0, 10.0, &SimParams3D::v_A_sigma},
        {"--gamma", 1.0e-12, 1.0e9, &SimParams3D::gamma_normal},
        {"--gamma-cancer", 1.0e-12, 1.0e9, &SimParams3D::gamma_cancer},
        {"--cancer-fraction", 0.0, 1.0, &SimParams3D::cancer_fraction},
        {"--dt", 1.0e-12, 1.0e6, &SimParams3D::dt},
        {"--aging-time", 0.0, 1.0e15, &SimParams3D::aging_time},
    };
    for (const DoubleParameterOption& option : double_options) {
        if (std::strcmp(argument, option.name) != 0) continue;
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_double(argument, value, option.minimum,
                                    option.maximum,
                                    &(params.*option.member))) {
            return OptionStatus::Error;
        }
        command->fresh_parameter_supplied = true;
        return OptionStatus::Accepted;
    }

    static const IntegerParameterOption integer_options[] = {
        {"--N", 1, 4000000, &SimParams3D::num_cells},
        {"--full-moment", 0, 1000000000, &SimParams3D::full_moment_every},
        {"--verify-every", 0, 1000000000, &SimParams3D::verify_every},
    };
    for (const IntegerParameterOption& option : integer_options) {
        if (std::strcmp(argument, option.name) != 0) continue;
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(argument, value, option.minimum,
                                     option.maximum, &integer)) {
            return OptionStatus::Error;
        }
        params.*option.member = static_cast<int>(integer);
        command->fresh_parameter_supplied = true;
        return OptionStatus::Accepted;
    }

    if (std::strcmp(argument, "--seed") == 0 ||
        std::strcmp(argument, "--polarity-seed") == 0) {
        value = value_after(argc, argv, index, argument);
        unsigned long long* target = std::strcmp(argument, "--seed") == 0
            ? &params.seed : &params.polarity_seed;
        if (!value || !parse_u64(argument, value, target))
            return OptionStatus::Error;
        command->fresh_parameter_supplied = true;
        return OptionStatus::Accepted;
    }

    if (std::strcmp(argument, "--initial-centres") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || value[0] == '\0') return OptionStatus::Error;
        command->options.initial_centres_path = value;
        command->initial_centres_supplied = true;
        return OptionStatus::Accepted;
    }

    return OptionStatus::Unrecognized;
}

struct IntegerRunOption {
    const char* name;
    long long minimum;
    long long maximum;
    int RunOptions3D::*member;
};

OptionStatus parse_execution_option(int argc, char** argv, int* index,
                                    const char* argument,
                                    ParsedCommand* command) {
    RunOptions3D& options = command->options;
    const char* value = nullptr;
    long long integer = 0;

    if (std::strcmp(argument, "--t-end") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_double(argument, value, 0.0, 1.0e15,
                                    &command->params.t_end)) {
            return OptionStatus::Error;
        }
        command->t_end_supplied = true;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--memory-mode") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !pf3d::parse_storage_mode(value, &options.storage_mode)) {
            std::fprintf(stderr,
                "[3d] --memory-mode expects auto, throughput, balanced, or compact\n");
            return OptionStatus::Error;
        }
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--memory-fraction") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_double(argument, value, 0.50, 0.99,
                                    &options.memory_fraction)) {
            return OptionStatus::Error;
        }
        return OptionStatus::Accepted;
    }

    static const IntegerRunOption integer_options[] = {
        {"--brick-edge", 0, 4096, &RunOptions3D::brick_edge},
        {"--scratch-slots", 0, 4000000, &RunOptions3D::scratch_slots},
        {"--device", 0, 63, &RunOptions3D::device},
        {"--bench", 1, 100000000, &RunOptions3D::bench_steps},
        {"--promoted-shards", 0, 1024, &RunOptions3D::promoted_shards},
        {"--fast-base-shards", 0, 1024, &RunOptions3D::fast_base_shards},
        {"--print-interval", 0, 1000000000, &RunOptions3D::print_interval},
    };
    for (const IntegerRunOption& option : integer_options) {
        if (std::strcmp(argument, option.name) != 0) continue;
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(argument, value, option.minimum,
                                     option.maximum, &integer)) {
            return OptionStatus::Error;
        }
        options.*option.member = static_cast<int>(integer);
        if (std::strcmp(argument, "--print-interval") == 0)
            command->print_interval_supplied = true;
        return OptionStatus::Accepted;
    }

    if (std::strcmp(argument, "--promoted-measure-shards") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(argument, value, -1,
                                     pf3d::kMaximumPromotedMeasureShards,
                                     &integer)) {
            return OptionStatus::Error;
        }
        options.promoted_measure_shards = static_cast<int>(integer);
        options.promoted_measure_shards_supplied = true;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--measure-shards") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(argument, value,
                                     pf3d::kBaseMeasureOneWave,
                                     pf3d::kMaximumBaseMeasureShards,
                                     &integer)) {
            return OptionStatus::Error;
        }
        options.measure_shards = static_cast<int>(integer);
        options.measure_shards_supplied = true;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--bench-phases") == 0) {
        options.bench_phases = true;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--strict") == 0) {
        options.strict = true;
        return OptionStatus::Accepted;
    }
    return OptionStatus::Unrecognized;
}

OptionStatus parse_trajectory_option(int argc, char** argv, int* index,
                                     const char* argument,
                                     ParsedCommand* command) {
    RunOptions3D& options = command->options;
    const char* value = nullptr;
    long long integer = 0;

    if (std::strcmp(argument, "--out") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || value[0] == '\0') return OptionStatus::Error;
        options.trajectory_path = value;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--trajectory-samples") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(argument, value, 1, 1000000000,
                                     &integer)) {
            return OptionStatus::Error;
        }
        options.trajectory_samples = static_cast<int>(integer);
        command->trajectory_samples_supplied = true;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--trajectory-interval") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(
                argument, value, 1,
                std::numeric_limits<std::uint32_t>::max(), &integer)) {
            return OptionStatus::Error;
        }
        options.trajectory_interval = integer;
        command->trajectory_interval_supplied = true;
        return OptionStatus::Accepted;
    }
    return OptionStatus::Unrecognized;
}

OptionStatus parse_checkpoint_option(int argc, char** argv, int* index,
                                     const char* argument,
                                     ParsedCommand* command) {
    RunOptions3D& options = command->options;
    const char* value = nullptr;
    long long integer = 0;

    if (std::strcmp(argument, "-c") == 0 ||
        std::strcmp(argument, "--checkpoint") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || value[0] == '\0') return OptionStatus::Error;
        command->checkpoint_input = value;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--checkpoint-interval") == 0 ||
        std::strcmp(argument, "--save-interval") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || !parse_integer(argument, value, 1, 1000000000000LL,
                                     &integer)) {
            return OptionStatus::Error;
        }
        if (std::strcmp(argument, "--checkpoint-interval") == 0)
            options.checkpoint_interval = integer;
        else
            options.save_interval = integer;
        command->checkpoint_schedule_supplied = true;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--checkpoint-dir") == 0) {
        value = value_after(argc, argv, index, argument);
        if (!value || value[0] == '\0') return OptionStatus::Error;
        options.checkpoint_dir = value;
        command->checkpoint_directory_supplied = true;
        return OptionStatus::Accepted;
    }
    if (std::strcmp(argument, "--no-final-checkpoint") == 0) {
        options.final_checkpoint = false;
        return OptionStatus::Accepted;
    }
    return OptionStatus::Unrecognized;
}

CommandStatus parse_command_line(int argc, char** argv,
                                 ParsedCommand* command) {
    using OptionParser = OptionStatus (*)(int, char**, int*, const char*,
                                           ParsedCommand*);
    static const OptionParser parsers[] = {
        parse_fresh_parameter_option,
        parse_execution_option,
        parse_trajectory_option,
        parse_checkpoint_option,
    };

    for (int i = 1; i < argc; ++i) {
        const char* argument = argv[i];
        if (std::strcmp(argument, "-h") == 0 ||
            std::strcmp(argument, "--help") == 0) {
            usage(argv[0]);
            return CommandStatus::Help;
        }
        bool accepted = false;
        for (OptionParser parser : parsers) {
            const OptionStatus status = parser(argc, argv, &i, argument, command);
            if (status == OptionStatus::Error)
                return CommandStatus::UsageError;
            if (status == OptionStatus::Accepted) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            std::fprintf(stderr, "[3d] unknown option '%s' (try --help)\n",
                         argument);
            return CommandStatus::UsageError;
        }
    }
    return CommandStatus::Ready;
}

bool resolve_fresh_geometry(ParsedCommand* command) {
    SimParams3D& params = command->params;
    if (command->slab_height_supplied && !params.substrate_slab()) {
        std::fprintf(stderr, "[3d] --slab-height requires --geometry slab\n");
        return false;
    }
    if (command->channel_height_supplied && !params.hard_wall_channel()) {
        std::fprintf(stderr,
                     "[3d] --channel-height requires --geometry channel\n");
        return false;
    }
    if ((command->wall_kappa_supplied || command->wall_width_supplied) &&
        !params.hard_wall_channel()) {
        std::fprintf(stderr,
                     "[3d] --wall-kappa/--wall-width require --geometry channel\n");
        return false;
    }
    if (command->wall_padding_supplied && !params.hard_wall_channel()) {
        std::fprintf(stderr,
                     "[3d] --wall-padding requires --geometry channel\n");
        return false;
    }
    if (command->slab_height_supplied && command->channel_height_supplied) {
        std::fprintf(stderr,
                     "[3d] slab and channel heights are mutually exclusive\n");
        return false;
    }

    int bounded_height = 0;
    if (params.hard_wall_channel()) {
        if (!command->wall_kappa_supplied) params.wall_kappa = params.kappa;
        if (!command->wall_width_supplied) params.wall_width = params.lambda;
        const double diameter = std::ceil(2.0 * params.target_radius);
        if (!std::isfinite(diameter) || diameter < 1.0 ||
            diameter > std::numeric_limits<int>::max()) {
            std::fprintf(stderr, "[3d] could not derive a channel height\n");
            return false;
        }
        bounded_height = command->channel_height_supplied
            ? command->channel_height : static_cast<int>(diameter);
        if (bounded_height < static_cast<int>(diameter)) {
            std::fprintf(stderr,
                "[3d] channel height %d is smaller than the supported "
                "one-diameter minimum %d\n",
                bounded_height, static_cast<int>(diameter));
            return false;
        }
    }

    const int side = params.substrate_slab()
        ? pf3d::slab_domain_side_for(
              params.num_cells, params.target_radius, params.rho)
        : params.hard_wall_channel()
            ? pf3d::channel_domain_side_for(
                  params.num_cells, params.target_radius, params.rho,
                  bounded_height)
            : pf3d::domain_side_for(
                  params.num_cells, params.target_radius, params.rho);
    if (side <= 0) {
        std::fprintf(stderr, "[3d] could not derive a finite %s domain\n",
            params.substrate_slab() ? "slab lateral"
                : params.hard_wall_channel() ? "channel lateral" : "cubic");
        return false;
    }

    if (params.substrate_slab()) {
        const int automatic_brick = params.brick_edge();
        if (automatic_brick <= 0 ||
            (!command->slab_height_supplied &&
             automatic_brick > std::numeric_limits<int>::max() / 2)) {
            std::fprintf(stderr, "[3d] could not derive a finite slab height\n");
            return false;
        }
        const int height = command->slab_height_supplied
            ? command->slab_height : 2 * automatic_brick;
        if (height <= automatic_brick) {
            std::fprintf(stderr,
                "[3d] slab height %d must exceed automatic brick edge %d\n",
                height, automatic_brick);
            return false;
        }
        params.Nx = params.Ny = side;
        params.Nz = height;
        return true;
    }

    if (params.hard_wall_channel()) {
        const int minimum_padding =
            pf3d::channel_wall_padding(params.wall_width);
        const int padding = command->wall_padding_supplied
            ? command->wall_padding : minimum_padding;
        const std::int64_t allocated_height =
            static_cast<std::int64_t>(bounded_height) +
            2 * static_cast<std::int64_t>(padding);
        if (minimum_padding <= 0 || allocated_height <= 0 ||
            allocated_height > std::numeric_limits<int>::max()) {
            std::fprintf(stderr,
                         "[3d] could not derive finite channel wall padding\n");
            return false;
        }
        if (padding < minimum_padding) {
            std::fprintf(stderr,
                "[3d] --wall-padding %d is smaller than the resolved-wall "
                "minimum %d for wall width %.9g\n",
                padding, minimum_padding, params.wall_width);
            return false;
        }
        params.Nx = params.Ny = side;
        params.channel_height = bounded_height;
        params.channel_padding = padding;
        params.Nz = static_cast<int>(allocated_height);
        return true;
    }

    params.Nx = params.Ny = params.Nz = side;
    return true;
}

CommandStatus resolve_resume_contract(ParsedCommand* command,
                                      PreparedRun* prepared) {
    if (!pf3d::checkpoint_probe_3d(command->checkpoint_input,
                                   &prepared->checkpoint)) {
        return CommandStatus::RuntimeError;
    }

    pf3d::PromotedMeasureReduction3D reduction{};
    const pf3d::ReductionResumeResult reduction_result =
        pf3d::resolve_promoted_measure_resume(
            prepared->checkpoint.promoted_measure_reduction,
            command->options.promoted_measure_shards_supplied,
            command->options.promoted_measure_shards, &reduction);
    if (reduction_result != pf3d::ReductionResumeResult::Ok) {
        if (reduction_result == pf3d::ReductionResumeResult::Mismatch) {
            std::fprintf(stderr,
                "[3d] checkpoint requires --promoted-measure-shards %d; "
                "requested %d would change the numerical reduction grouping\n",
                prepared->checkpoint.promoted_measure_reduction.policy,
                command->options.promoted_measure_shards);
        } else {
            std::fprintf(stderr,
                "[3d] checkpoint has an invalid promoted-measurement "
                "reduction contract\n");
        }
        return CommandStatus::UsageError;
    }

    command->options.promoted_measure_shards = reduction.policy;
    command->options.promoted_measure_auto_wave_ctas = reduction.auto_wave_ctas;
    if (!command->print_interval_supplied)
        command->options.print_interval = prepared->checkpoint.print_interval;
    if (!command->trajectory_samples_supplied &&
        !command->trajectory_interval_supplied &&
        prepared->checkpoint.trajectory_interval > 0) {
        command->options.trajectory_interval =
            prepared->checkpoint.trajectory_interval;
    }
    prepared->effective = prepared->checkpoint.params;
    if (command->t_end_supplied)
        prepared->effective.t_end = command->params.t_end;
    if (!command->checkpoint_directory_supplied &&
        command->checkpoint_schedule_supplied) {
        command->options.checkpoint_dir = default_checkpoint_directory(
            command->options.trajectory_path,
            command->checkpoint_input).string();
    }
    return CommandStatus::Ready;
}

CommandStatus prepare_run(ParsedCommand* command, PreparedRun* prepared) {
    prepared->resumed = !command->checkpoint_input.empty();
    if (prepared->resumed &&
        (command->fresh_parameter_supplied ||
         command->initial_centres_supplied)) {
        std::fprintf(stderr,
            "[3d] checkpoint continuation preserves its model and initialized "
            "state; only --t-end and run/output/storage options may change\n");
        return CommandStatus::UsageError;
    }

    if (prepared->resumed) {
        const CommandStatus status = resolve_resume_contract(command, prepared);
        if (status != CommandStatus::Ready) return status;
    } else {
        if (!resolve_fresh_geometry(command)) return CommandStatus::UsageError;
        prepared->effective = command->params;
        if (!command->checkpoint_directory_supplied &&
            command->checkpoint_schedule_supplied) {
            command->options.checkpoint_dir = default_checkpoint_directory(
                command->options.trajectory_path,
                command->checkpoint_input).string();
        }
    }

    const char* validation_error = nullptr;
    if (!pf3d::validate(prepared->effective, &validation_error)) {
        std::fprintf(stderr, "[3d] invalid parameters: %s\n",
                     validation_error ? validation_error : "unknown reason");
        return CommandStatus::UsageError;
    }
    if (prepared->resumed &&
        prepared->effective.total_steps() <
            static_cast<long long>(prepared->checkpoint.step)) {
        std::fprintf(stderr,
                     "[3d] --t-end precedes checkpoint step %llu (t=%.9g)\n",
                     static_cast<unsigned long long>(prepared->checkpoint.step),
                     prepared->checkpoint.time);
        return CommandStatus::UsageError;
    }

    prepared->effective_brick = prepared->resumed
        ? prepared->checkpoint.brick_edge
        : command->options.brick_edge > 0
            ? command->options.brick_edge : prepared->effective.brick_edge();
    if (prepared->effective.substrate_slab() &&
        prepared->effective.Nz <= prepared->effective_brick) {
        std::fprintf(stderr,
            "[3d] slab height %d must exceed selected brick edge %d\n",
            prepared->effective.Nz, prepared->effective_brick);
        return CommandStatus::UsageError;
    }
    return CommandStatus::Ready;
}

int launch_simulation(const ParsedCommand& command,
                      const PreparedRun& prepared) {
    print_configuration(prepared.effective, command.options,
                        prepared.effective_brick, prepared.resumed);

    pf3d::Sim3D simulation;
    const bool initialized = prepared.resumed
        ? simulation.init_checkpoint(
              prepared.checkpoint, command.checkpoint_input, command.options,
              command.t_end_supplied ? command.params.t_end : -1.0)
        : simulation.init_fresh(prepared.effective, command.options);
    if (!initialized) return 1;

    if (command.options.bench_steps > 0) {
        double milliseconds = 0.0;
        return simulation.bench(command.options.bench_steps, &milliseconds)
            ? 0 : 1;
    }

    std::signal(SIGINT, pf3d::Sim3D::request_termination);
    std::signal(SIGTERM, pf3d::Sim3D::request_termination);
    return simulation.run() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    ParsedCommand command;
    const CommandStatus parse_status = parse_command_line(argc, argv, &command);
    if (parse_status == CommandStatus::Help) return 0;
    if (parse_status != CommandStatus::Ready) return 2;

    PreparedRun prepared;
    const CommandStatus preparation_status = prepare_run(&command, &prepared);
    if (preparation_status == CommandStatus::RuntimeError) return 1;
    if (preparation_status != CommandStatus::Ready) return 2;
    return launch_simulation(command, prepared);
}
