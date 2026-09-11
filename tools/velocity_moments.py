#!/usr/bin/env python3
"""Stream and validate PFVMOM4 records using only the Python standard library."""
from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys
from typing import BinaryIO, Iterator

HEADER = struct.Struct("<8sIIdqdIIQ")
STEP = struct.Struct("<q")
COLUMNS = 26
CHANNELS = ("A_x", "A_y", "I_x", "I_y", "G_x", "G_y",
            "K_x", "K_y", "area_x", "area_y", "L_x", "L_y")


@dataclass(frozen=True)
class Header:
    cells: int
    dt: float
    start_step: int
    side: float
    spatial_stride: int
    dense_start: int


@dataclass(frozen=True)
class Frame:
    step: int
    rows: tuple[tuple[float, ...], ...]


def read_header(stream: BinaryIO) -> Header:
    raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError("truncated PFVMOM4 header")
    magic, cells, columns, dt, start, side, stride, quadrature, dense = HEADER.unpack(raw)
    if magic != b"PFVMOM4\0" or columns != COLUMNS or quadrature != 2:
        raise ValueError("unsupported velocity format or quadrature")
    if not 0 < cells <= 4_000_000 or not 1 <= stride <= 1000:
        raise ValueError("invalid cell count or spatial stride")
    if start < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid start step or timestep")
    if not math.isfinite(side) or side <= 0:
        raise ValueError("invalid periodic domain")
    return Header(cells, dt, start, side, stride, dense)


def spatial_samples(header: Header, count: int) -> int:
    """Count pre-step samples, including forced priming on an unaligned restart."""
    dense = min(count, header.dense_start)
    first = header.start_step + dense
    first += (-first) % header.spatial_stride
    last = header.start_step + count - 1
    aligned = max(0, (last - first) // header.spatial_stride + 1)
    priming = int(count > 0 and dense == 0 and
                  header.start_step % header.spatial_stride != 0)
    return dense + aligned + priming


def frames(stream: BinaryIO, header: Header) -> Iterator[Frame]:
    previous = header.start_step - 1
    cell_ids: tuple[int, ...] | None = None
    metadata: tuple[tuple[float, ...], ...] | None = None
    while raw_step := stream.read(STEP.size):
        if len(raw_step) != STEP.size:
            raise ValueError("truncated frame step")
        step, = STEP.unpack(raw_step)
        if step <= previous or (cell_ids is None and step != header.start_step):
            raise ValueError("frame steps are not strictly increasing from the origin")
        data = stream.read(header.cells * COLUMNS * 8)
        if len(data) != header.cells * COLUMNS * 8:
            raise ValueError("truncated velocity frame")
        values = array("d")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        rows = tuple(tuple(values[i:i+COLUMNS])
                     for i in range(0, len(values), COLUMNS))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("non-finite velocity data")
        if any(row[0] != int(row[0]) or row[0] < 0 for row in rows):
            raise ValueError("invalid cell ID")
        ids = tuple(int(row[0]) for row in rows)
        current_metadata = tuple(tuple(row[4:7]) for row in rows)
        if cell_ids is None:
            if len(set(ids)) != header.cells:
                raise ValueError("duplicate cell ID")
            cell_ids = ids
            metadata = current_metadata
        if ids != cell_ids or current_metadata != metadata:
            raise ValueError("cell identity or material parameters changed within segment")
        count = step - header.start_step
        expected_samples = spatial_samples(header, count)
        for row in rows:
            if row[8] != count or row[21] != expected_samples:
                raise ValueError("incorrect integration or spatial sample count")
            if row[7] <= 0 or row[4] <= 0 or row[5] <= 0 or row[6] < 0:
                raise ValueError("invalid cell geometry or material parameters")
            if not 0 <= row[1] < header.side or not 0 <= row[2] < header.side:
                raise ValueError("centroid outside periodic domain")
            if count == 0 and any(row[9:21]):
                raise ValueError("nonzero cumulative integral at segment origin")
        previous = step
        yield Frame(step, rows)
    if cell_ids is None:
        raise ValueError("missing initial velocity frame")


def summarize(path: Path, expected_final_step: int | None = None) -> dict:
    with path.open("rb") as stream:
        header = read_header(stream)
        count = 0
        last = header.start_step
        for frame in frames(stream, header):
            last = frame.step
            count += 1
    if expected_final_step is not None and last != expected_final_step:
        raise ValueError(f"final step {last} differs from expected {expected_final_step}")
    return {"format": "PFVMOM4", "cells": header.cells, "frames": count,
            "start_step": header.start_step, "last_step": last, "dt": header.dt,
            "spatial_stride": header.spatial_stride, "dense_start": header.dense_start}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--expected-final-step", type=int)
    args = parser.parse_args()
    try:
        result = summarize(args.file, args.expected_final_step)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{args.file}: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
