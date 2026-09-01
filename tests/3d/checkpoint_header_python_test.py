#!/usr/bin/env python3
"""Regression checks for the PF3D Python header reader."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from pf3d_checkpoint_header import (  # noqa: E402
    CELL_RECORD_BYTES,
    CHECKPOINT_FORMAT,
    DIMENSIONS,
    ENDIAN_MARKER,
    FILE_HEADER,
    HEADER_BYTES,
    MAGIC,
    PARAMS_BYTES,
    PHASE_ORDER_X_FASTEST,
    REQUIRED_HEADER_FLAGS,
    SCALAR_FLOAT32,
    canonical_crc_header,
    parse_file_header,
)


def packed_header(shards: int) -> bytes:
    edge = 8
    values = edge**3
    return FILE_HEADER.pack(
        MAGIC,
        CHECKPOINT_FORMAT,
        HEADER_BYTES,
        ENDIAN_MARKER,
        DIMENSIONS,
        SCALAR_FLOAT32,
        REQUIRED_HEADER_FLAGS,
        10,
        0.1,
        1,
        PARAMS_BYTES,
        CELL_RECORD_BYTES,
        edge,
        PHASE_ORDER_X_FASTEST,
        values,
        4 * values,
        0x1122334455667788,
        shards,
    )


def check_python_contract() -> None:
    for shards in (0, 1, 64):
        raw = packed_header(shards)
        header = parse_file_header(raw)
        if header.base_measure_shards != shards:
            raise AssertionError("base-measurement shard count was not preserved")
        canonical = canonical_crc_header(raw)
        if canonical[80:88] != bytes(8) or canonical[88:96] != raw[88:96]:
            raise AssertionError("CRC canonicalization altered the shard count")
    try:
        parse_file_header(packed_header(65))
    except ValueError:
        pass
    else:
        raise AssertionError("shard count 65 was accepted")


def check_cpp_layout(writer: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pf3d-header-") as directory:
        fixture = Path(directory) / "header.bin"
        subprocess.run([str(writer), str(fixture)], check=True)
        raw = fixture.read_bytes()
    header = parse_file_header(raw)
    if header.base_measure_shards != 11:
        raise AssertionError("Python reader disagrees with the C++ header layout")
    if header.stored_crc64 != 0x1122334455667788:
        raise AssertionError("Python reader disagrees with the C++ CRC field offset")
    canonical = canonical_crc_header(raw)
    if canonical[80:88] != bytes(8) or canonical[88:96] != raw[88:96]:
        raise AssertionError("Python CRC canonicalization disagrees with C++ layout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-writer", type=Path, required=True)
    args = parser.parse_args()
    check_python_contract()
    check_cpp_layout(args.fixture_writer)


if __name__ == "__main__":
    main()
