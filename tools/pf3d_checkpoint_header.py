"""Fixed header contract shared by PF3D Python tools."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass


FILE_HEADER = struct.Struct("<IHHIIIIqdQIIIIQQQQ")
MAGIC = 0x44334650
CHECKPOINT_FORMAT = 1
HEADER_BYTES = 96
PARAMS_BYTES = 288
CELL_RECORD_BYTES = 256
ENDIAN_MARKER = 0x01020304
DIMENSIONS = 3
SCALAR_FLOAT32 = 1
PHASE_ORDER_X_FASTEST = 1
REQUIRED_HEADER_FLAGS = 3
MAX_BRICK_EDGE = 4096
STORAGE_EDGE_ALIGNMENT = 8
MAX_INT32 = 2_147_483_647
MAX_BASE_MEASURE_SHARDS = 64


@dataclass(frozen=True)
class FileHeader:
    step: int
    time: float
    num_cells: int
    brick_edge: int
    phase_values: int
    phase_bytes: int
    stored_crc64: int
    base_measure_shards: int


def parse_file_header(raw: bytes) -> FileHeader:
    if len(raw) != FILE_HEADER.size:
        raise ValueError("truncated PF3D header")
    (
        magic,
        format_version,
        header_bytes,
        endian_marker,
        dimensions,
        scalar_format,
        flags,
        step,
        time,
        num_cells,
        params_bytes,
        cell_record_bytes,
        brick_edge,
        phase_order,
        phase_values,
        phase_bytes,
        stored_crc64,
        base_measure_shards,
    ) = FILE_HEADER.unpack(raw)
    if (
        magic != MAGIC
        or format_version != CHECKPOINT_FORMAT
        or header_bytes != HEADER_BYTES
        or endian_marker != ENDIAN_MARKER
        or dimensions != DIMENSIONS
        or scalar_format != SCALAR_FLOAT32
        or flags != REQUIRED_HEADER_FLAGS
        or params_bytes != PARAMS_BYTES
        or cell_record_bytes != CELL_RECORD_BYTES
        or phase_order != PHASE_ORDER_X_FASTEST
        or base_measure_shards > MAX_BASE_MEASURE_SHARDS
    ):
        raise ValueError("unsupported or malformed PF3D checkpoint header")
    if (
        step < 0
        or not math.isfinite(time)
        or time < 0.0
        or num_cells <= 0
        or num_cells > MAX_INT32
        or brick_edge < 3
        or brick_edge > MAX_BRICK_EDGE
        or brick_edge % STORAGE_EDGE_ALIGNMENT != 0
    ):
        raise ValueError("invalid PF3D header values")
    if phase_values != brick_edge**3 or phase_bytes != 4 * phase_values:
        raise ValueError("header phase-field sizes do not match the base edge")
    return FileHeader(
        step,
        time,
        num_cells,
        brick_edge,
        phase_values,
        phase_bytes,
        stored_crc64,
        base_measure_shards,
    )


def canonical_crc_header(raw: bytes) -> bytes:
    if len(raw) != FILE_HEADER.size:
        raise ValueError("truncated PF3D header")
    canonical = bytearray(raw)
    canonical[80:88] = b"\0" * 8
    return bytes(canonical)
