#include "checkpoint_format_3d.h"

#include <cstdint>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;

    ckpt3d::FileHeader3D header{};
    header.magic = ckpt3d::kMagic;
    header.version = ckpt3d::kCheckpointFormat;
    header.header_bytes = sizeof(header);
    header.endian_marker = ckpt3d::kEndianMarker;
    header.dimensions = ckpt3d::kDimensions;
    header.scalar_format = ckpt3d::kScalarFloat32;
    header.flags = ckpt3d::kRequiredHeaderFlags;
    header.step = 10;
    header.time = 0.1;
    header.num_cells = 1;
    header.params_bytes = sizeof(ckpt3d::ParamsRecord3D);
    header.cell_record_bytes = sizeof(ckpt3d::CellRecord3D);
    header.brick_edge = 8;
    header.phase_order = ckpt3d::kPhaseOrderXFastest;
    header.phase_values_per_cell = ckpt3d::phase_values_per_cell(8);
    header.phase_bytes_per_cell =
        header.phase_values_per_cell * sizeof(float);
    header.file_crc64 = UINT64_C(0x1122334455667788);
    header.base_measure_shards = 11;

    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    return output ? 0 : 1;
}
