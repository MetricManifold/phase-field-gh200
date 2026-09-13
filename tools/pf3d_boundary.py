#!/usr/bin/env python3
"""Bounded, frame-streaming reader for PF3D projected boundary observations.

The binary contract is in common/boundary_format_3d.h. Reading to exhaustion
checks the terminal marker; stopping an iterator early does not certify a file
as complete. No phase-field reconstruction or contact/T1 inference is done.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import struct
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


FILE_HEADER = struct.Struct("<8s6I3q6dQ2IQ")
FRAME_HEADER = struct.Struct("<8sQd3Q")
CELL_RECORD = struct.Struct("<4q4I4f")
SQUARE_RECORD = struct.Struct("<I4f")
FILE_MAGIC = b"PFB3D1\0\0"
FRAME_MAGIC = b"P3BFRM1\0"
END_MAGIC = b"P3BEND1\0"
VERSION = 1
PROJECTIONS = {1: "basal", 2: "maximum"}
GEOMETRIES = {7: "periodic", 3: "slab", 11: "channel"}
CODECS = {0: "none", 1: "zstd"}
MAX_I64 = (1 << 63) - 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


class BoundaryFormatError(ValueError):
    """Malformed, incomplete, unsupported, or over-limit boundary stream."""


@dataclass(frozen=True)
class ReaderLimits:
    """Explicit memory/work bounds; raise these only for trusted larger files.

    Square objects are decoded lazily, but a frame's encoded and decoded bytes,
    cell metadata and a per-cell duplicate-coordinate set can coexist in memory.
    """

    max_frame_bytes: int = 64 * 1024 * 1024
    max_stored_bytes: int = 64 * 1024 * 1024
    max_cells: int = 100_000
    max_squares: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("max_frame_bytes", "max_stored_bytes", "max_cells", "max_squares"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class FileHeader:
    cells: int
    projection: int
    boundary_flags: int
    codec: int
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    dt: float
    tau: float
    level: float
    interval: int

    @property
    def projection_name(self) -> str:
        return PROJECTIONS[self.projection]

    @property
    def geometry(self) -> str:
        return GEOMETRIES[self.boundary_flags]

    @property
    def codec_name(self) -> str:
        return CODECS[self.codec]


@dataclass(frozen=True)
class Square:
    x: int
    y: int
    # Lower-left, lower-right, upper-right, upper-left, in the padded plane.
    phi: tuple[float, float, float, float]


class Squares(Sequence[Square]):
    """Read-only lazy square view; retaining a view retains its frame payload."""

    __slots__ = ("_data",)

    def __init__(self, data: memoryview) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data) // SQUARE_RECORD.size

    def __getitem__(self, index: int | slice) -> Square | tuple[Square, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("square index out of range")
        xy, *phi = SQUARE_RECORD.unpack_from(self._data, index * SQUARE_RECORD.size)
        return Square(xy & 0xFFFF, xy >> 16, tuple(phi))

    def __iter__(self) -> Iterator[Square]:
        for xy, p0, p1, p2, p3 in SQUARE_RECORD.iter_unpack(self._data):
            yield Square(xy & 0xFFFF, xy >> 16, (p0, p1, p2, p3))


@dataclass(frozen=True)
class Cell:
    id: int
    origin_x: int
    origin_y: int
    origin_z: int
    brick_edge: int
    plane_edge: int
    gamma: float
    active_speed: float
    radius: float
    squares: Squares


@dataclass(frozen=True)
class Frame:
    step: int
    time: float
    cells: tuple[Cell, ...]

    @property
    def square_count(self) -> int:
        return sum(len(cell.squares) for cell in self.cells)


def fnv1a64(data: bytes | memoryview) -> int:
    """FNV-1a of the complete decoded payload, including all cell records."""
    value = FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def _read_exact(stream: BinaryIO, size: int, description: str) -> bytes:
    # Fixed-size chunks also work on streams that return short reads. The caller
    # must validate a payload size against ReaderLimits before reaching here.
    result = bytearray()
    while len(result) < size:
        chunk = stream.read(min(size - len(result), 64 * 1024))
        if not chunk:
            raise BoundaryFormatError(f"truncated {description}")
        result.extend(chunk)
    return bytes(result)


def parse_file_header(raw: bytes, limits: ReaderLimits | None = None) -> FileHeader:
    limits = limits or ReaderLimits()
    if len(raw) != FILE_HEADER.size:
        raise BoundaryFormatError("truncated file header")
    (
        magic, version, header_bytes, cells, projection, flags, codec,
        nx, ny, nz, dx, dy, dz, dt, tau, level, interval,
        square_bytes, cell_bytes, reserved,
    ) = FILE_HEADER.unpack(raw)
    if (
        magic != FILE_MAGIC or version != VERSION or header_bytes != FILE_HEADER.size
        or square_bytes != SQUARE_RECORD.size or cell_bytes != CELL_RECORD.size
        or reserved != 0
    ):
        raise BoundaryFormatError("unsupported or malformed file header")
    if projection not in PROJECTIONS or flags not in GEOMETRIES or codec not in CODECS:
        raise BoundaryFormatError("unsupported projection, geometry, or codec")
    if projection == 1 and flags != 3:
        raise BoundaryFormatError("basal projection requires slab geometry")
    if not 0 < cells <= limits.max_cells:
        raise BoundaryFormatError("cell count is zero or exceeds reader limit")
    if cells * CELL_RECORD.size > limits.max_frame_bytes:
        raise BoundaryFormatError("cell records exceed frame byte limit")
    if any(not 0 < dimension <= 2_147_483_647 for dimension in (nx, ny, nz)):
        raise BoundaryFormatError("invalid domain dimensions")
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy, dz, dt, tau)):
        raise BoundaryFormatError("grid spacing, dt, and tau must be finite and positive")
    if any(not math.isfinite(n * spacing) for n, spacing in ((nx, dx), (ny, dy), (nz, dz))):
        raise BoundaryFormatError("physical domain extent is not finite")
    if level != 0.5 or interval == 0:
        raise BoundaryFormatError("level must be 0.5 and interval must be positive")
    return FileHeader(cells, projection, flags, codec, nx, ny, nz,
                      dx, dy, dz, dt, tau, level, interval)


def _decode_zstd(stored: bytes, raw_bytes: int, limits: ReaderLimits) -> bytes:
    try:
        zstd = importlib.import_module("zstandard")
    except ImportError as exc:
        raise BoundaryFormatError(
            "zstd frame requires the optional Python package 'zstandard'"
        ) from exc
    try:
        # Check advertised output and window BEFORE invoking the allocator. The
        # one-shot API otherwise ignores max_output_size when content size is
        # present in a zstd frame header.
        parameters = zstd.get_frame_parameters(stored)
        if parameters.content_size not in (zstd.CONTENTSIZE_UNKNOWN, raw_bytes):
            raise BoundaryFormatError("zstd content size disagrees with frame header")
        if parameters.window_size > limits.max_frame_bytes:
            raise BoundaryFormatError("zstd window exceeds reader frame limit")
        decoder = zstd.ZstdDecompressor(
            max_window_size=max(1024, (limits.max_frame_bytes + 1023) // 1024)
        )
        raw = decoder.decompress(stored, max_output_size=raw_bytes, allow_extra_data=False)
    except zstd.ZstdError as exc:
        raise BoundaryFormatError(f"invalid zstd payload: {exc}") from exc
    if len(raw) != raw_bytes:
        raise BoundaryFormatError("decoded payload size disagrees with frame header")
    return raw


def _decode_cells(raw: bytes, header: FileHeader, limits: ReaderLimits) -> tuple[Cell, ...]:
    view = memoryview(raw)
    records_end = header.cells * CELL_RECORD.size
    offset = records_end
    records = []
    ids: set[int] = set()
    total_squares = 0
    for index in range(header.cells):
        record = CELL_RECORD.unpack_from(view, index * CELL_RECORD.size)
        (cell_id, ox, oy, oz, edge, plane, count, reserved,
         gamma, speed, radius, reserved_float) = record
        if cell_id < 0 or cell_id in ids:
            raise BoundaryFormatError("cell IDs must be unique and nonnegative")
        ids.add(cell_id)
        if edge < 8 or edge > 65534 or edge % 8 or plane != edge + 2:
            raise BoundaryFormatError(f"cell {cell_id}: invalid brick or padded plane edge")
        if ox + plane - 1 > MAX_I64 or oy + plane - 1 > MAX_I64 or oz + edge - 1 > MAX_I64:
            raise BoundaryFormatError(f"cell {cell_id}: origin extent overflows int64")
        if reserved != 0 or reserved_float != 0:
            raise BoundaryFormatError(f"cell {cell_id}: nonzero reserved metadata")
        if (not all(math.isfinite(value) for value in (gamma, speed, radius))
                or gamma <= 0 or speed < 0 or radius <= 0):
            raise BoundaryFormatError(f"cell {cell_id}: invalid physical metadata")
        total_squares += count
        if count > (plane - 1) ** 2 or total_squares > limits.max_squares:
            raise BoundaryFormatError("square count exceeds plane capacity or reader limit")
        records.append(record)
    if records_end + total_squares * SQUARE_RECORD.size != len(raw):
        raise BoundaryFormatError("cell square counts do not exhaust decoded payload")

    result = []
    for record in records:
        cell_id, ox, oy, oz, edge, plane, count, _, gamma, speed, radius, _ = record
        end = offset + count * SQUARE_RECORD.size
        square_data = view[offset:end]
        seen: set[int] = set()
        for xy, p0, p1, p2, p3 in SQUARE_RECORD.iter_unpack(square_data):
            x, y = xy & 0xFFFF, xy >> 16
            if x >= plane - 1 or y >= plane - 1:
                raise BoundaryFormatError(f"cell {cell_id}: square coordinate outside padded plane")
            if xy in seen:
                raise BoundaryFormatError(f"cell {cell_id}: duplicate square coordinate")
            seen.add(xy)
            phi = (p0, p1, p2, p3)
            if not all(math.isfinite(value) for value in phi):
                raise BoundaryFormatError(f"cell {cell_id}: nonfinite square corner")
            if not (any(value >= header.level for value in phi)
                    and any(value < header.level for value in phi)):
                raise BoundaryFormatError(f"cell {cell_id}: square does not cross the contour level")
            if ((x == 0 and (p0 != 0 or p3 != 0))
                    or (y == 0 and (p0 != 0 or p1 != 0))
                    or (x == plane - 2 and (p1 != 0 or p2 != 0))
                    or (y == plane - 2 and (p2 != 0 or p3 != 0))):
                raise BoundaryFormatError(f"cell {cell_id}: nonzero corner on zero-padded border")
        result.append(Cell(cell_id, ox, oy, oz, edge, plane, gamma, speed, radius,
                           Squares(square_data)))
        offset = end
    return tuple(result)


class BoundaryReader(Iterator[Frame]):
    """Read one fully validated frame at a time from a path or binary stream.

    Use as a context manager when stopping early. ``complete`` becomes true
    only after the end marker and physical EOF have both been checked. External
    streams remain caller-owned. Yielded frames stay usable after closing.
    """

    def __init__(self, source: str | Path | BinaryIO, *, limits: ReaderLimits | None = None):
        self.limits = limits or ReaderLimits()
        self.complete = False
        self._closed = False
        self._last_step: int | None = None
        self._last_time = 0.0
        self._owns_stream = isinstance(source, (str, Path))
        self._stream = open(source, "rb") if self._owns_stream else source
        try:
            self.header = parse_file_header(
                _read_exact(self._stream, FILE_HEADER.size, "file header"), self.limits
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> BoundaryReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed and self._owns_stream:
            self._stream.close()
        self._closed = True

    def __iter__(self) -> BoundaryReader:
        return self

    def __next__(self) -> Frame:
        if self._closed:
            raise StopIteration
        try:
            return self._next_frame()
        except BaseException:
            self.close()
            raise

    def _next_frame(self) -> Frame:
        first = self._stream.read(1)
        if not first:
            raise BoundaryFormatError("missing end marker (incomplete boundary stream)")
        packed = first + _read_exact(self._stream, FRAME_HEADER.size - 1, "frame header")
        magic, step, time, raw_bytes, stored_bytes, checksum = FRAME_HEADER.unpack(packed)
        if magic not in (FRAME_MAGIC, END_MAGIC):
            raise BoundaryFormatError("invalid frame or end-marker magic")
        if not math.isfinite(time) or time < 0:
            raise BoundaryFormatError("frame time must be finite and nonnegative")
        if magic == END_MAGIC:
            if raw_bytes or stored_bytes or checksum:
                raise BoundaryFormatError("end marker has nonzero payload fields")
            if step != (self._last_step or 0) or time != self._last_time:
                raise BoundaryFormatError("end marker does not match the last frame")
            if self._stream.read(1):
                raise BoundaryFormatError("trailing bytes after end marker")
            self.complete = True
            self.close()
            raise StopIteration
        if self._last_step is not None and (step <= self._last_step or time < self._last_time):
            raise BoundaryFormatError("frame steps must increase and times must not decrease")
        minimum = self.header.cells * CELL_RECORD.size
        if raw_bytes < minimum or (raw_bytes - minimum) % SQUARE_RECORD.size:
            raise BoundaryFormatError("invalid raw payload size")
        if raw_bytes > self.limits.max_frame_bytes or stored_bytes > self.limits.max_stored_bytes:
            raise BoundaryFormatError("frame byte count exceeds reader limit")
        if (raw_bytes - minimum) // SQUARE_RECORD.size > self.limits.max_squares:
            raise BoundaryFormatError("frame square count exceeds reader limit")
        if stored_bytes == 0 or (self.header.codec == 0 and stored_bytes != raw_bytes):
            raise BoundaryFormatError("invalid stored payload size")
        stored = _read_exact(self._stream, stored_bytes, "frame payload")
        raw = stored if self.header.codec == 0 else _decode_zstd(stored, raw_bytes, self.limits)
        if fnv1a64(raw) != checksum:
            raise BoundaryFormatError("decoded payload checksum mismatch")
        cells = _decode_cells(raw, self.header, self.limits)
        self._last_step, self._last_time = step, time
        return Frame(step, time, cells)


def iter_frames(source: str | Path | BinaryIO, *, limits: ReaderLimits | None = None) -> Iterator[Frame]:
    """Yield validated frames; exhaust the iterator to verify successful EOF."""
    with BoundaryReader(source, limits=limits) as reader:
        yield from reader


def inspect_file(path: str | Path, *, limits: ReaderLimits | None = None) -> dict:
    """Validate the entire file and return a small summary, not retained frames."""
    with BoundaryReader(path, limits=limits) as reader:
        first_step = last_step = None
        frames = squares = empty_cells = 0
        for frame in reader:
            if first_step is None:
                first_step = frame.step
            last_step = frame.step
            frames += 1
            squares += frame.square_count
            empty_cells += sum(not cell.squares for cell in frame.cells)
        header = reader.header
        return {
            "path": str(path), "complete": reader.complete,
            "projection": header.projection_name, "geometry": header.geometry,
            "codec": header.codec_name, "cells": header.cells,
            "grid": [header.nx, header.ny, header.nz],
            "spacing": [header.dx, header.dy, header.dz],
            "dt": header.dt, "tau": header.tau, "level": header.level,
            "interval": header.interval, "frames": frames,
            "first_step": first_step, "last_step": last_step,
            "squares": squares, "empty_cell_observations": empty_cells,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="boundary stream to inspect and fully validate")
    parser.add_argument("--json", action="store_true", help="print a machine-readable summary")
    parser.add_argument("--max-frame-mib", type=int, default=64,
                        help="decoded and stored frame byte limit in MiB (default: 64)")
    parser.add_argument("--max-cells", type=int, default=100_000)
    parser.add_argument("--max-squares", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    try:
        limits = ReaderLimits(args.max_frame_mib * 1024 * 1024,
                              args.max_frame_mib * 1024 * 1024,
                              args.max_cells, args.max_squares)
        summary = inspect_file(args.path, limits=limits)
    except (OSError, ValueError) as exc:
        print(f"pf3d_boundary: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"{summary['path']}: complete {summary['projection']} boundary stream")
        print(f"{summary['geometry']}, grid {tuple(summary['grid'])}, "
              f"{summary['cells']} cells, codec {summary['codec']}")
        print(f"{summary['frames']} frames, steps {summary['first_step']}..{summary['last_step']}, "
              f"interval {summary['interval']}, {summary['squares']} crossing squares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
