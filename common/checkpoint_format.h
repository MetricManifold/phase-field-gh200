// Binary checkpoint schema shared by the simulator and conversion tools.
// Multi-byte fields are little-endian, and the current implementation assumes
// a little-endian host. Schema changes require a new checkpoint format version.
#pragma once
#include <cstdint>
#include <cstddef>

namespace ckpt {

constexpr uint32_t MAGIC = 0x43454C4C;  // 'CELL'
constexpr uint32_t CHECKPOINT_FORMAT = 1;  // sole supported on-disk schema
constexpr int32_t CELL_TILE_PITCH = 288;

// Sidecar block magics (appear after per-cell records, in any order).
constexpr uint32_t MAGIC_VA_A = 0x56415F41;  // 'VA_A' — per-cell v_A
constexpr uint32_t MAGIC_GAMA = 0x47414D41;  // 'GAMA' — per-cell gamma
constexpr uint32_t MAGIC_RADI = 0x52414449;  // 'RADI' — per-cell target radius
constexpr uint32_t MAGIC_POLR = 0x504F4C52;  // 'POLR' — per-cell polarity theta

// Packed on-disk records. Their sizes and field offsets are part of the schema.
// The checkpoint stores:
//   FixedPrefix (44 B)
//   parameter record (prefix.sp_sz bytes; see CheckpointParamsRecord below)
//   int32 tile pitch
//   RankTrailer (12 B)
//   num_cells_local x { CellRecordHeader (96 B), tile_pitch^2 float32 }
//   zero or more { SidecarBlockHeader (8 B), payload } blocks
#pragma pack(push, 1)

// File bytes 0..43.
struct FixedPrefix {
    uint32_t magic;               // = MAGIC
    uint32_t version;             // checkpoint format identifier
    int32_t  step;
    double   cur_time;
    int32_t  num_cells_local;
    int32_t  save_interval;
    int32_t  reserved;
    int32_t  trajectory_samples;
    uint8_t  bools[4];
    uint32_t sp_sz;
};

// Fixed schema record retained by the file layout. This implementation writes
// and accepts only num_ranks=1 and rank_id=0.
struct RankTrailer {
    int32_t num_ranks;
    int32_t rank_id;
    int32_t num_cells_global;
};

// Followed by CELL_TILE_PITCH^2 float32 phase-field values in row-major order.
// origin is the global coordinate of tile pixel (0,0), modulo the domain.
// cx/cy are derived global centres for external tools. The exact moments,
// support bounds, class, and demotion counter are the state consumed by the
// next update and make restart independent of a host-side remeasurement.
struct CellRecordHeader {
    int32_t cell_id;              // global id
    int32_t origin_x;
    int32_t origin_y;
    int32_t shape_class;
    uint32_t promote_ctr;
    uint32_t reserved0;           // written as zero
    float   cx;
    float   cy;
    float   vx;
    float   vy;
    double  volume_moment;        // sum(phi^2), before dA
    double  moment_x;             // sum(phi^2*x), active-window coordinates
    double  moment_y;
    double  perimeter;            // cached full-moment interface measure
    int32_t support_lo_x;
    int32_t support_hi_x;
    int32_t support_lo_y;
    int32_t support_hi_y;
    float   phi_max;
    uint32_t reserved1;           // written as zero
};

// Followed by count values; magic determines the value type.
struct SidecarBlockHeader {
    uint32_t magic;
    int32_t  count;
};

// The stable 192-byte parameter record is separate from the solver's runtime
// parameters so changes to host-side types cannot alter disk ordering.
struct CheckpointParamsRecord {
    int32_t  Nx, Ny;                 //   0,   4
    double   dx, dy;                 //   8,  16
    double   dt;                     //  24
    double   t_end;                  //  32
    double   rho;                     //  40   requested packing fraction
    double   lambda;                  //  48
    double   gamma_normal;            //  56   GAMA stores realized per-cell values
    double   gamma_soft;              //  64
    double   soft_fraction;           //  72
    double   kappa;                   //  80
    double   target_radius;           //  88
    double   mu;                      //  96
    double   v_A;                     // 104
    double   v_A_sigma;               // 112
    double   xi;                      // 120
    double   tau;                     // 128
    uint64_t seed;                    // 136
    uint64_t polarity_seed;           // 144   resolved tumble stream
    uint64_t initialization_hash;     // 152   accepted centre-table fingerprint
    int32_t  print_interval;          // 160
    int32_t  full_moment_every;       // 164
    int32_t  verify_every;            // 168
    int32_t  save_interval;           // 172
    int32_t  trajectory_samples;      // 176
    int32_t  reserved;                // 180   written as zero
    int64_t  trajectory_interval;     // 184   resolved absolute-step cadence
};

#pragma pack(pop)

static_assert(sizeof(FixedPrefix)       == 44, "FixedPrefix layout drift");
static_assert(sizeof(RankTrailer)       == 12, "RankTrailer layout drift");
static_assert(sizeof(CellRecordHeader)  == 96, "CellRecordHeader layout drift");
static_assert(sizeof(SidecarBlockHeader) == 8, "SidecarBlockHeader layout drift");
static_assert(sizeof(CheckpointParamsRecord) == 192,
              "checkpoint parameter layout drift");

// External parsers depend on these byte offsets.
static_assert(offsetof(CheckpointParamsRecord, rho)           ==  40, "rho offset");
static_assert(offsetof(CheckpointParamsRecord, lambda)        ==  48, "lambda offset");
static_assert(offsetof(CheckpointParamsRecord, gamma_normal)  ==  56, "gamma offset");
static_assert(offsetof(CheckpointParamsRecord, target_radius) ==  88, "radius offset");
static_assert(offsetof(CheckpointParamsRecord, v_A)           == 104, "v_A offset");
static_assert(offsetof(CheckpointParamsRecord, tau)           == 128, "tau offset");
static_assert(offsetof(CheckpointParamsRecord, seed)          == 136, "seed offset");
static_assert(offsetof(CheckpointParamsRecord, polarity_seed) == 144, "polarity offset");
static_assert(offsetof(CheckpointParamsRecord, initialization_hash) == 152,
              "initialization hash offset");
static_assert(offsetof(CheckpointParamsRecord, print_interval) == 160,
              "print interval offset");
static_assert(offsetof(CheckpointParamsRecord, trajectory_interval) == 184,
              "trajectory interval offset");
static_assert(offsetof(CellRecordHeader, volume_moment) == 40,
              "cell volume-moment offset");
static_assert(offsetof(CellRecordHeader, perimeter) == 64,
              "cell perimeter offset");
static_assert(offsetof(CellRecordHeader, phi_max) == 88,
              "cell phi-max offset");

}  // namespace ckpt
