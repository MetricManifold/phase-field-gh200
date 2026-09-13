#!/usr/bin/env python3
"""CPU fixtures for the 3D boundary reader; no CUDA or third-party dependency.

Zstd fixtures run additionally when the optional zstandard package is present.
All files created by these tests live in owned TemporaryDirectory instances.
"""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import pf3d_boundary as boundary  # noqa: E402


HEADER_FIELDS = {
    "magic": b"PFB3D1\0\0", "version": 1, "header_bytes": 128,
    "cells": 2, "projection": 1, "boundary_flags": 3, "codec": 0,
    "nx": 64, "ny": 64, "nz": 32,
    "dx": 1.0, "dy": 2.0, "dz": 0.5, "dt": 0.01, "tau": 2.0,
    "level": 0.5, "interval": 10, "square_bytes": 20, "cell_bytes": 64,
    "reserved": 0,
}
CELL_FIELDS = {
    "id": 42, "origin_x": -2, "origin_y": 63, "origin_z": -1,
    "brick_edge": 8, "plane_edge": 10, "squares": 2, "reserved": 0,
    "gamma": 0.25, "active_speed": 0.0, "radius": 4.0, "reserved_float": 0.0,
}


def header(**changes) -> bytes:
    values = HEADER_FIELDS | changes
    return struct.pack("<8s6I3q6dQ2IQ", *values.values())


def cell(**changes) -> bytes:
    values = CELL_FIELDS | changes
    return struct.pack("<4q4I4f", *values.values())


def square(x=1, y=1, phi=(0.25, 0.5, 0.75, 0.125)) -> bytes:
    return struct.pack("<I4f", x | (y << 16), *phi)


def payload(*, first=None, second=None, squares=None) -> bytes:
    # Cell records precede EVERY square; an interleaved reader cannot pass.
    return (
        (cell() if first is None else first)
        + (cell(id=7, brick_edge=16, plane_edge=18, squares=0) if second is None else second)
        + (square() + square(x=2, phi=(0.5, 0.25, 0.125, 0.75))
           if squares is None else squares)
    )


def frame(raw=None, *, step=10, time=0.1, stored=None, **changes) -> bytes:
    raw = payload() if raw is None else raw
    stored = raw if stored is None else stored
    fields = {"magic": b"P3BFRM1\0", "step": step, "time": time,
              "raw_bytes": len(raw), "stored_bytes": len(stored),
              "checksum": boundary.fnv1a64(raw)} | changes
    return struct.pack("<8sQd3Q", *fields.values()) + stored


def end(*, step=10, time=0.1, raw_bytes=0, stored_bytes=0, checksum=0,
        magic=b"P3BEND1\0") -> bytes:
    return struct.pack("<8sQd3Q", magic, step, time, raw_bytes, stored_bytes, checksum)


def stream(*, file_header=None, frames=None, terminal=None) -> bytes:
    return ((header() if file_header is None else file_header)
            + (frame() if frames is None else frames)
            + (end() if terminal is None else terminal))


def read(blob: bytes, **kwargs) -> list[boundary.Frame]:
    return list(boundary.iter_frames(io.BytesIO(blob), **kwargs))


class BoundaryFormatTests(unittest.TestCase):
    def assert_invalid(self, blob: bytes, pattern=None, **kwargs) -> None:
        with self.assertRaisesRegex(boundary.BoundaryFormatError, pattern or "."):
            read(blob, **kwargs)

    def test_fixed_layout_and_known_checksum(self):
        self.assertEqual(boundary.FILE_HEADER.size, 128)
        self.assertEqual(boundary.FRAME_HEADER.size, 48)
        self.assertEqual(boundary.CELL_RECORD.size, 64)
        self.assertEqual(boundary.SQUARE_RECORD.size, 20)
        raw = header()
        self.assertEqual(raw[:8], b"PFB3D1\0\0")
        self.assertEqual(struct.unpack_from("<q", raw, 32)[0], 64)
        self.assertEqual(struct.unpack_from("<d", raw, 96)[0], 0.5)
        self.assertEqual(struct.unpack_from("<Q", raw, 104)[0], 10)
        self.assertEqual(struct.unpack_from("<II", raw, 112), (20, 64))
        self.assertEqual(boundary.fnv1a64(b""), 14695981039346656037)
        self.assertEqual(boundary.fnv1a64(b"hello"), 0xA430D84680AABD0B)

    def test_raw_frames_cell_identity_padding_and_lazy_squares(self):
        blob = stream(frames=frame() + frame(step=20, time=0.2),
                      terminal=end(step=20, time=0.2))
        external = io.BytesIO(blob)
        with boundary.BoundaryReader(external) as reader:
            self.assertFalse(reader.complete)
            self.assertEqual(reader.header.projection_name, "basal")
            self.assertEqual(reader.header.geometry, "slab")
            self.assertEqual(reader.header.codec_name, "none")
            first = next(reader)
            self.assertEqual(first.step, 10)
            self.assertEqual(first.time, 0.1)
            self.assertEqual(first.square_count, 2)
            self.assertEqual([c.id for c in first.cells], [42, 7])
            observed = first.cells[0]
            self.assertEqual((observed.origin_x, observed.origin_y, observed.origin_z), (-2, 63, -1))
            self.assertEqual(observed.plane_edge, observed.brick_edge + 2)
            self.assertEqual(observed.squares[0].phi, (0.25, 0.5, 0.75, 0.125))
            self.assertEqual((observed.squares[-1].x, observed.squares[-1].y), (2, 1))
            self.assertEqual(len(observed.squares[:1]), 1)
            self.assertEqual(len(first.cells[1].squares), 0)
            self.assertEqual(first.cells[1].brick_edge, 16)
            with self.assertRaises(IndexError):
                _ = observed.squares[2]
            with self.assertRaises(IndexError):
                _ = observed.squares[-3]
            self.assertEqual([f.step for f in reader], [20])
            self.assertTrue(reader.complete)
        self.assertFalse(external.closed)
        self.assertEqual(len(list(observed.squares)), 2)

    def test_maximum_projection_all_geometries(self):
        for flags, name in ((7, "periodic"), (3, "slab"), (11, "channel")):
            with self.subTest(flags=flags):
                blob = stream(file_header=header(projection=2, boundary_flags=flags))
                with boundary.BoundaryReader(io.BytesIO(blob)) as reader:
                    self.assertEqual(reader.header.geometry, name)
                    self.assertEqual(reader.header.projection_name, "maximum")
                    self.assertEqual(len(list(reader)), 1)

    def test_empty_cells_empty_stream_and_initial_step_zero(self):
        raw = payload(first=cell(squares=0), squares=b"")
        frames = read(stream(frames=frame(raw, step=0, time=0), terminal=end(step=0, time=0)))
        self.assertEqual(frames[0].square_count, 0)
        self.assertEqual(read(stream(frames=b"", terminal=end(step=0, time=0))), [])

    def test_stop_early_does_not_claim_completion(self):
        with boundary.BoundaryReader(io.BytesIO(stream(terminal=b""))) as reader:
            next(reader)
        self.assertFalse(reader.complete)
        self.assertEqual(list(reader), [])

    def test_every_truncation_rejected(self):
        blob = stream()
        for cut in range(len(blob)):
            with self.subTest(cut=cut):
                self.assert_invalid(blob[:cut])

    def test_short_stream_reads(self):
        class ShortReads(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 3))

        self.assertEqual(len(list(boundary.iter_frames(ShortReads(stream())))), 1)

    def test_bad_header_metadata(self):
        changes = [
            {"magic": b"PFB3D1xx"}, {"version": 2}, {"header_bytes": 127},
            {"square_bytes": 24}, {"cell_bytes": 56}, {"reserved": 1},
            {"projection": 0}, {"projection": 3}, {"boundary_flags": 0},
            {"boundary_flags": 7}, {"boundary_flags": 11}, {"codec": 2},
            {"cells": 0}, {"cells": 100_001}, {"nx": 0}, {"ny": -1},
            {"nz": 2_147_483_648}, {"dx": 0}, {"dy": -1}, {"dz": float("nan")},
            {"dt": float("inf")}, {"dt": 0}, {"tau": -1}, {"tau": float("nan")},
            {"level": 0.25}, {"level": float("nan")}, {"interval": 0}, {"dx": 1e308},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assert_invalid(stream(file_header=header(**change)))

    def test_bad_cell_metadata(self):
        changes = [
            {"id": -1}, {"id": 7}, {"brick_edge": 0}, {"brick_edge": 7},
            {"brick_edge": 9}, {"brick_edge": 65536}, {"plane_edge": 8},
            {"reserved": 1}, {"reserved_float": 1}, {"reserved_float": float("nan")},
            {"gamma": 0}, {"gamma": float("inf")}, {"active_speed": -1},
            {"active_speed": float("nan")}, {"radius": 0}, {"radius": float("inf")},
            {"origin_x": (1 << 63) - 1}, {"origin_y": (1 << 63) - 1},
            {"origin_z": (1 << 63) - 1}, {"squares": 82}, {"squares": 1},
            {"squares": 3},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assert_invalid(stream(frames=frame(payload(first=cell(**change)))))

    def test_bad_square_records(self):
        bad = [
            square(x=9), square(y=9), square(x=65535), square(y=65535),
            square(phi=(0, float("nan"), 1, 0)),
            square(phi=(float("inf"), 0, 0, 0)),
            square(phi=(0, 0, 0, 0)), square(phi=(0.5, 0.5, 0.5, 0.5)),
            square(phi=(1, 1, 1, 1)), square(x=0), square(y=0),
            square(x=8), square(y=8),
        ]
        for invalid_square in bad:
            with self.subTest(square=invalid_square):
                raw = payload(first=cell(squares=1), squares=invalid_square)
                self.assert_invalid(stream(frames=frame(raw)))
        duplicate = payload(squares=square() + square())
        self.assert_invalid(stream(frames=frame(duplicate)), "duplicate square")

    def test_threshold_tie_negative_corners_and_zero_padding(self):
        for valid_square in (
            square(phi=(0.5, 0, 0, 0)), square(phi=(-0.01, 1.01, 0.25, 0.125)),
            square(x=0, phi=(0, 1, 0.75, 0)),
            square(y=0, phi=(0, 0, 0.75, 1)),
            square(x=8, phi=(1, 0, 0, 0.75)),
            square(y=8, phi=(1, 0.75, 0, 0)),
        ):
            with self.subTest(square=valid_square):
                raw = payload(first=cell(squares=1), squares=valid_square)
                self.assertEqual(read(stream(frames=frame(raw)))[0].square_count, 1)

    def test_checksum_includes_cell_metadata(self):
        damaged = bytearray(stream())
        damaged[128 + 48 + 48] ^= 1  # Gamma bits, not a square record.
        self.assert_invalid(bytes(damaged), "checksum")
        damaged = bytearray(stream())
        damaged[128 + 48 + 128 + 4] ^= 1
        self.assert_invalid(bytes(damaged), "checksum")

    def test_frame_metadata_sizes_and_monotonicity(self):
        changes = [
            {"magic": b"NOTFRAME"}, {"raw_bytes": 127}, {"raw_bytes": 169},
            {"raw_bytes": 1 << 40}, {"stored_bytes": 1 << 40},
            {"stored_bytes": 0}, {"stored_bytes": 167}, {"checksum": 0},
            {"time": float("nan")}, {"time": float("inf")}, {"time": -0.1},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assert_invalid(stream(frames=frame(**change)))
        for step, time in ((10, 0.2), (9, 0.2), (20, 0.09)):
            with self.subTest(step=step, time=time):
                self.assert_invalid(stream(frames=frame() + frame(step=step, time=time),
                                           terminal=end(step=step, time=time)), "increase")
        # Finite times may be equal; ordering is guaranteed by absolute steps.
        self.assertEqual(len(read(stream(frames=frame() + frame(step=11),
                                         terminal=end(step=11)))), 2)
        self.assert_invalid(stream(frames=frame(payload() + square())), "exhaust")

    def test_end_marker_is_mandatory_exact_and_terminal(self):
        for changes in ({"raw_bytes": 1}, {"stored_bytes": 1}, {"checksum": 1},
                        {"step": 11}, {"time": 0.2}, {"time": float("nan")},
                        {"magic": b"P3BEND1x"}):
            with self.subTest(changes=changes):
                self.assert_invalid(stream(terminal=end(**changes)))
        self.assert_invalid(stream(terminal=b""), "missing end marker")
        self.assert_invalid(stream() + b"\0", "trailing")
        self.assert_invalid(stream() + frame(step=20), "trailing")

    def test_limits_checked_before_large_reads(self):
        class GuardedReads(io.BytesIO):
            def read(self, size=-1):
                if not 0 <= size <= 65536:
                    raise AssertionError(f"unbounded read request: {size}")
                return super().read(size)

        # No payload is supplied; a malicious size must fail before any attempt
        # to reserve or request the advertised terabyte.
        malformed = header() + struct.pack("<8sQd3Q", b"P3BFRM1\0", 10, 0.1,
                                           128 + 20 * (1 << 40), 1 << 40, 0)
        source = GuardedReads(malformed)
        with self.assertRaisesRegex(boundary.BoundaryFormatError, "reader limit"):
            list(boundary.iter_frames(source))
        self.assertEqual(source.tell(), 176)
        for limits in (boundary.ReaderLimits(max_cells=1),
                       boundary.ReaderLimits(max_squares=1),
                       boundary.ReaderLimits(max_frame_bytes=160),
                       boundary.ReaderLimits(max_stored_bytes=160),
                       boundary.ReaderLimits(max_frame_bytes=100)):
            with self.subTest(limits=limits):
                self.assert_invalid(stream(), limits=limits)
        for value in (0, -1, 0.5, True):
            with self.assertRaises(ValueError):
                boundary.ReaderLimits(max_frame_bytes=value)

    def test_optional_zstandard_dependency_only_for_compressed_frames(self):
        with mock.patch.object(boundary.importlib, "import_module", side_effect=ImportError):
            self.assertEqual(len(read(stream())), 1)
            self.assert_invalid(stream(file_header=header(codec=1)), "optional Python package")

    def test_cli_inspects_paths_and_reports_damage(self):
        with tempfile.TemporaryDirectory(prefix="pf3d-boundary-reader-") as directory:
            fixture = Path(directory) / "projected boundaries.bin"
            fixture.write_bytes(stream())
            result = subprocess.run(
                [sys.executable, str(TOOLS / "pf3d_boundary.py"), str(fixture), "--json"],
                capture_output=True, text=True, check=True,
            )
            summary = json.loads(result.stdout)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["squares"], 2)
            self.assertEqual(summary["empty_cell_observations"], 1)
            self.assertEqual(summary["first_step"], 10)
            self.assertEqual(summary["last_step"], 10)
            fixture.write_bytes(stream(terminal=b""))
            result = subprocess.run(
                [sys.executable, str(TOOLS / "pf3d_boundary.py"), str(fixture)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("missing end marker", result.stderr)
            self.assertEqual(result.stdout, "")
        self.assertFalse(Path(directory).exists())


@unittest.skipUnless(importlib.util.find_spec("zstandard") is not None,
                     "optional zstandard package is unavailable")
class ZstdBoundaryFormatTests(unittest.TestCase):
    def setUp(self):
        import zstandard
        self.zstd = zstandard

    def test_compressed_round_trip_known_and_unknown_content_size(self):
        for content_size in (True, False):
            with self.subTest(content_size=content_size):
                raw = payload()
                stored = self.zstd.ZstdCompressor(write_content_size=content_size).compress(raw)
                observed = read(stream(file_header=header(codec=1), frames=frame(raw, stored=stored)))
                self.assertEqual(observed[0].square_count, 2)
                self.assertEqual(observed[0].cells[0].squares[0].phi, (0.25, 0.5, 0.75, 0.125))

    def test_bad_zstd_frames_and_trailing_compressed_data(self):
        raw = payload()
        valid = self.zstd.ZstdCompressor().compress(raw)
        bad = (b"not-zstd", valid[:-1], valid + b"trailing", valid + valid,
               self.zstd.ZstdCompressor().compress(raw + bytes(20)))
        for stored in bad:
            with self.subTest(stored=stored):
                with self.assertRaises(boundary.BoundaryFormatError):
                    read(stream(file_header=header(codec=1), frames=frame(raw, stored=stored)))

    def test_unknown_size_decompression_and_window_are_bounded(self):
        raw = payload()
        # The payload frame claims 168 bytes while the zstd stream really emits
        # more: max_output_size bounds allocation even without a content size.
        larger = self.zstd.ZstdCompressor(write_content_size=False).compress(raw + bytes(4096))
        with self.assertRaises(boundary.BoundaryFormatError):
            read(stream(file_header=header(codec=1), frames=frame(raw, stored=larger)))
        stored = self.zstd.ZstdCompressor(write_content_size=False).compress(raw)
        self.assertGreater(self.zstd.get_frame_parameters(stored).window_size, 200)
        with self.assertRaisesRegex(boundary.BoundaryFormatError, "window exceeds"):
            read(stream(file_header=header(codec=1), frames=frame(raw, stored=stored)),
                 limits=boundary.ReaderLimits(max_frame_bytes=200))

    def test_huge_advertised_content_size_rejected_before_decoder(self):
        # A single-segment zstd header with an eight-byte content size; no
        # block is needed to demonstrate pre-allocation rejection of 1 TiB.
        stored = b"\x28\xb5\x2f\xfd\xe0" + struct.pack("<Q", 1 << 40)
        self.assertEqual(self.zstd.get_frame_parameters(stored).content_size, 1 << 40)
        with mock.patch.object(self.zstd, "ZstdDecompressor") as decoder:
            with self.assertRaisesRegex(boundary.BoundaryFormatError, "content size"):
                read(stream(file_header=header(codec=1), frames=frame(stored=stored)))
            decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
