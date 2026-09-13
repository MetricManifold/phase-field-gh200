#!/usr/bin/env python3
"""Compare the two-device substrate runner with the shared single-device solver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pf3d_checkpoint_header import (  # noqa: E402
    CELL_RECORD_BYTES, FILE_HEADER, PARAMS_BYTES, FileHeader, canonical_crc_header,
    parse_file_header,
)


DT = 0.001
POLYNOMIAL = 0x42F0E1EBA9EA3693
MASK64 = (1 << 64) - 1


def crc_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        value = byte << 56
        for _ in range(8):
            value = ((value << 1) ^ (POLYNOMIAL if value >> 63 else 0)) & MASK64
        table.append(value)
    return tuple(table)


CRC_TABLE = crc_table()


def checksum(raw: bytes) -> int:
    crc = 0
    for byte in raw:
        crc = CRC_TABLE[((crc >> 56) ^ byte) & 255] ^ ((crc << 8) & MASK64)
    return crc


@dataclass
class Snapshot:
    raw: bytes
    header: FileHeader
    params: bytes
    cells: list[tuple[bytes, bytes]]


def snapshot(path: Path, expected_step: int | None = None) -> Snapshot:
    raw = path.read_bytes()
    header = parse_file_header(raw[:FILE_HEADER.size])
    if expected_step is not None and header.step != expected_step:
        raise ValueError(f"{path}: expected step {expected_step}, got {header.step}")
    if checksum(canonical_crc_header(raw[:FILE_HEADER.size])
                + raw[FILE_HEADER.size:]) != header.stored_crc64:
        raise ValueError(f"{path}: checksum mismatch")
    pos = FILE_HEADER.size
    params = raw[pos:pos + PARAMS_BYTES]
    pos += PARAMS_BYTES
    if len(params) != PARAMS_BYTES:
        raise ValueError(f"{path}: truncated parameters")
    nx, ny, nz = struct.unpack_from("<qqq", params)
    if struct.unpack_from("<I", params, 52)[0] != 3 or nx != ny:
        raise ValueError(f"{path}: expected substrate geometry")
    cells = []
    ids = set()
    for _ in range(header.num_cells):
        record = raw[pos:pos + CELL_RECORD_BYTES]
        pos += CELL_RECORD_BYTES
        if len(record) != CELL_RECORD_BYTES:
            raise ValueError(f"{path}: truncated cell record")
        gid = struct.unpack_from("<q", record)[0]
        edge = struct.unpack_from("<I", record, 184)[0]
        if gid in ids or not header.brick_edge <= edge < min(nx, ny, nz) or edge % 8:
            raise ValueError(f"{path}: invalid cell identity/storage")
        ids.add(gid)
        phase = raw[pos:pos + 4 * edge**3]
        pos += 4 * edge**3
        if len(phase) != 4 * edge**3:
            raise ValueError(f"{path}: truncated phase cube")
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", phase)):
            raise ValueError(f"{path}: nonfinite phase field")
        doubles = struct.unpack_from("<8d", record, 72)
        if not all(math.isfinite(v) for v in doubles) or doubles[0] <= 0:
            raise ValueError(f"{path}: invalid cell moments")
        if struct.unpack_from("<I", record, 144)[0] != 0:
            raise ValueError(f"{path}: cell carries an integrity flag")
        cells.append((record, phase))
    if pos != len(raw):
        raise ValueError(f"{path}: trailing or missing checkpoint bytes")
    return Snapshot(raw, header, params, cells)


def compare_bytes(reference: Path, candidate: Path) -> None:
    a, b = reference.read_bytes(), candidate.read_bytes()
    if a != b:
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                     min(len(a), len(b)))
        raise ValueError(f"byte mismatch at {first}: {reference} ({len(a)}) vs "
                         f"{candidate} ({len(b)})")


def fresh_arguments(centres: Path, rho: float = 0.3,
                    verify_every: int = 10) -> list[str]:
    return [
        "--geometry", "slab", "--N", "4", "--radius", "5",
        "--lambda", "1", "--rho", str(rho), "--slab-height", "80",
        "--brick-edge", "24", "--dt", str(DT), "--aging-time", "0",
        "--tau", "0.02", "--v-A", "0.1", "--gamma-cancer", "0.35",
        "--cancer-fraction", "0.5", "--seed", "20260909",
        "--polarity-seed", "20260910", "--initial-centres", str(centres),
        "--full-moment", "10", "--verify-every", str(verify_every),
    ]


def run(executable: Path, output: Path, steps: int, *, centres: Path,
        resume: Path | None = None, strict: bool = False,
        two_gpu: bool = False, rho: float = 0.3,
        trajectory_interval: int = 1, verify_every: int = 10,
        measure_shards: int = 4) -> Path:
    output.mkdir()  # Every launch owns a new directory; previous data survives.
    arguments = (["--checkpoint", str(resume)] if resume
                 else fresh_arguments(centres, rho, verify_every))
    arguments += [
        "--t-end", format(steps * DT, ".12g"), "--memory-mode", "throughput",
        "--measure-shards", str(measure_shards), "--promoted-measure-shards", "4",
        "--print-interval", "0",
        "--trajectory-interval", str(trajectory_interval),
        "--out", str(output / "trajectory.txt"),
        "--checkpoint-dir", str(output / "checkpoints"),
    ]
    if strict:
        arguments.append("--strict")
    completed = subprocess.run([str(executable), *arguments], cwd=output,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=90)
    (output / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{output.name}: exit {completed.returncode}; see run.log")
    if two_gpu and "[two-gpu] devices " not in completed.stdout:
        raise ValueError(f"{output.name}: missing two-device runner banner")
    path = output / "checkpoints" / "checkpoint.pf3d"
    if snapshot(path, steps).header.base_measure_shards != measure_shards:
        raise ValueError(f"{output.name}: checkpoint did not retain measurement grouping")
    trajectory = output / "trajectory.txt"
    if not trajectory.is_file() or trajectory.stat().st_size == 0:
        raise ValueError(f"{output.name}: trajectory missing")
    return path


def promote_fixture(source: Path, destination: Path) -> None:
    """Pad two cubes with exact zeros while preserving their world coordinates."""
    state = snapshot(source)
    records = []
    for n, (old_record, old_phase) in enumerate(state.cells):
        if n not in (0, 2):
            records.append(old_record + old_phase)
            continue
        old_edge = struct.unpack_from("<I", old_record, 184)[0]
        edge = old_edge + 8
        offset = 4
        record = bytearray(old_record)
        origin = struct.unpack_from("<3q", record, 8)
        struct.pack_into("<3q", record, 8, *(value - offset for value in origin))
        volume, *moments = struct.unpack_from("<4d", record, 72)
        struct.pack_into("<3d", record, 80,
                         *(value + offset * volume for value in moments))
        bounds = struct.unpack_from("<6i", record, 148)
        struct.pack_into("<6i", record, 148, *(value + offset for value in bounds))
        struct.pack_into("<I", record, 184, edge)
        phase = bytearray(4 * edge**3)
        for z in range(old_edge):
            for y in range(old_edge):
                src = 4 * old_edge * (y + old_edge * z)
                dst = 4 * (offset + edge * (y + offset + edge * (z + offset)))
                phase[dst:dst + 4 * old_edge] = old_phase[src:src + 4 * old_edge]
        records.append(bytes(record) + phase)
    raw = bytearray(canonical_crc_header(state.raw[:FILE_HEADER.size])
                    + state.params + b"".join(records))
    struct.pack_into("<Q", raw, 80, checksum(raw))
    with destination.open("xb") as stream:
        stream.write(raw)
    promoted = snapshot(destination, state.header.step)
    if sum(struct.unpack_from("<I", cell[0], 184)[0] > promoted.header.brick_edge
           for cell in promoted.cells) != 2:
        raise ValueError("promoted fixture did not contain exactly two enlarged cubes")


def migration_fixture(source: Path, destination: Path) -> None:
    """Place intact cubes before both y ownership cuts with upward polarity."""
    state = snapshot(source)
    ny = struct.unpack_from("<q", state.params, 8)[0]
    params = bytearray(state.params)
    struct.pack_into("<d", params, 128, 10.0)
    struct.pack_into("<d", params, 152, 1.0e9)
    struct.pack_into("<Q", params, 224, 0)  # Synthetic placement has no source-table hash.
    records = []
    for n, (old_record, phase) in enumerate(state.cells):
        record = bytearray(old_record)
        edge = struct.unpack_from("<I", record, 184)[0]
        if n in (0, 2):
            midpoint = ny // 2 - 1 if n == 0 else ny - 1
            struct.pack_into("<q", record, 16, midpoint - edge // 2)
        speed = 10.0 if n in (0, 2) else 0.0
        struct.pack_into("<6f", record, 32, 0.0, 1.0, 0.0, 0.0, speed, 0.0)
        struct.pack_into("<f", record, 60, speed)
        records.append(record + phase)
    raw = bytearray(canonical_crc_header(state.raw[:FILE_HEADER.size])
                    + params + b"".join(records))
    struct.pack_into("<Q", raw, 80, checksum(raw))
    with destination.open("xb") as stream:
        stream.write(raw)
    snapshot(destination, state.header.step)


def require_migration(before: Path, after: Path) -> None:
    start, end = snapshot(before), snapshot(after)
    ny = struct.unpack_from("<q", start.params, 8)[0]
    for n in (0, 2):
        a, b = start.cells[n][0], end.cells[n][0]
        old_edge = struct.unpack_from("<I", a, 184)[0]
        new_edge = struct.unpack_from("<I", b, 184)[0]
        old_midpoint = struct.unpack_from("<q", a, 16)[0] + old_edge // 2
        new_midpoint = struct.unpack_from("<q", b, 16)[0] + new_edge // 2
        cut = ny // 2 if n == 0 else ny
        if not old_midpoint < cut <= new_midpoint:
            raise ValueError(f"cell {n} did not cross ownership cut {cut}")
        if (old_midpoint % ny < ny // 2) == (new_midpoint % ny < ny // 2):
            raise ValueError(f"cell {n} did not change owner")
        if struct.unpack_from("<I", a, 140) != struct.unpack_from("<I", b, 140):
            raise ValueError("migration fixture unexpectedly tumbled")


def require_partial_seam(path: Path) -> None:
    state = snapshot(path)
    ny = struct.unpack_from("<q", state.params, 8)[0]
    edges = [struct.unpack_from("<I", record, 184)[0]
             for record, _ in state.cells]
    half_width = (max(edges) + 1) // 2
    seam_rows = {(cut + offset) % ny for cut in (0, ny // 2)
                 for offset in range(-half_width, half_width)}
    owners = {(struct.unpack_from("<q", record, 16)[0] + edge // 2) % ny
              >= ny // 2 for (record, _), edge in zip(state.cells, edges)}
    if ny != 80 or len(seam_rows) >= ny or owners != {False, True}:
        raise ValueError("partial-seam fixture must retain two owners and unexchanged rows")


def partial_seam(reference: Path, executable: Path, root: Path,
                 measure_shards: int = 4) -> None:
    # Each pair overlaps across an ownership cut; the exchanged rows do not
    # cover the whole domain, unlike the compact four-cell smoke geometry.
    centres = root / "partial-seam-centres.csv"
    centres.write_text("global_id,x,y\n0,10,37\n1,10,43\n2,50,77\n3,50,3\n",
                       encoding="ascii")
    for steps in (1, 10):
        paths = []
        for name, binary in [("reference", reference), ("two-gpu", executable)]:
            path = run(binary, root / f"partial-seam-{steps}-{name}", steps,
                       centres=centres, rho=0.05, two_gpu=name == "two-gpu",
                       measure_shards=measure_shards)
            require_partial_seam(path)
            paths.append(path)
        compare_bytes(*paths)
        compare_bytes(paths[0].parents[1] / "trajectory.txt",
                      paths[1].parents[1] / "trajectory.txt")
        print(f"PASS partial seam {steps}: checkpoint and trajectory byte-identical",
              flush=True)


def require_mixed_tile_passes(origins_y: list[int]) -> None:
    # Ny=96, B=24: exchanged rows are [0,12), [36,60), [84,96).
    # The fast kernel uses 16-row tiles; its final tile has only eight rows.
    seam_rows = set(range(12)) | set(range(36, 60)) | set(range(84, 96))
    by_owner: list[set[bool]] = [set(), set()]
    for origin in origins_y:
        owner = int((origin + 12) % 96 >= 48)
        for y0 in (0, 16):
            boundary = any((origin + y) % 96 in seam_rows
                           for y in range(y0, min(y0 + 16, 24)))
            by_owner[owner].add(boundary)
    if any(passes != {False, True} for passes in by_owner):
        raise ValueError("boundary/interior fixture must exercise both tile passes "
                         "on each owner")


def boundary_interior(reference: Path, executable: Path, root: Path,
                      measure_shards: int = 4) -> None:
    # Keep an interacting pair across y=48. The 24-row interior gaps leave
    # room for tile classification to survive recentering. All cells share the
    # hemisphere's live z span, so these row classifications also apply in 3D.
    positions = ((10, 45), (10, 51), (50, 24), (50, 72))
    require_mixed_tile_passes([round(y - 11.5) for _, y in positions])
    centres = root / "boundary-interior-centres.csv"
    centres.write_text("global_id,x,y\n" + "".join(
        f"{gid},{x},{y}\n" for gid, (x, y) in enumerate(positions)),
        encoding="ascii")

    whole = []
    starts = []
    resumed = []
    for name, binary in (("reference", reference), ("two-gpu", executable)):
        options = dict(centres=centres, rho=0.0342, two_gpu=name == "two-gpu",
                       trajectory_interval=5, measure_shards=measure_shards)
        complete = run(binary, root / f"boundary-interior-whole-{name}", 10,
                       **options)
        start = run(binary, root / f"boundary-interior-split-5-{name}", 5,
                    **options)
        end = run(binary, root / f"boundary-interior-split-10-{name}", 10,
                  resume=start, **options)
        for path in (complete, start, end):
            state = snapshot(path)
            if struct.unpack_from("<q", state.params, 8)[0] != 96:
                raise ValueError("boundary/interior fixture must have Ny=96")
            if state.header.brick_edge != 24 or any(
                   struct.unpack_from("<I", record, 184)[0] != 24
                   for record, _ in state.cells):
                raise ValueError("boundary/interior fixture must retain B=24")
            require_mixed_tile_passes([
                struct.unpack_from("<q", record, 16)[0]
                for record, _ in state.cells])
        compare_bytes(complete, end)
        whole.append(complete)
        starts.append(start)
        resumed.append(end)
    for paths in (whole, starts, resumed):
        compare_bytes(*paths)
        compare_bytes(paths[0].parents[1] / "trajectory.txt",
                      paths[1].parents[1] / "trajectory.txt")
    print("PASS boundary/interior tiles: 10-step and 5+5 restart checkpoints "
          "and trajectories byte-identical", flush=True)


def require_trajectory_steps(path: Path, steps: tuple[int, ...]) -> None:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    expected = [(step, gid) for step in steps for gid in range(4)]
    if len(rows) != len(expected):
        raise ValueError(f"{path}: expected {len(expected)} trajectory rows")
    for values, (step, gid) in zip(rows, expected):
        if (len(values) != 17 or int(values[1]) != gid or
                not math.isclose(float(values[0]), step * DT,
                                 rel_tol=0.0, abs_tol=1.0e-12)):
            raise ValueError(f"{path}: unexpected trajectory frame/cell sequence")


def sparse_outputs(reference: Path, executable: Path, root: Path,
                   centres: Path, measure_shards: int = 4) -> None:
    # No output at the routine polls on steps 16, 32, and 48. At strict step
    # 64, verification consumes moments already cached by the measured update.
    for strict, steps in ((False, 65), (True, 64)):
        label = f"sparse-{'strict' if strict else 'fast'}"
        paths = []
        for name, binary in (("reference", reference), ("two-gpu", executable)):
            path = run(binary, root / f"{label}-{name}", steps,
                       centres=centres, strict=strict, two_gpu=name == "two-gpu",
                       trajectory_interval=64, verify_every=64,
                       measure_shards=measure_shards)
            require_trajectory_steps(path.parents[1] / "trajectory.txt",
                                     (0, 64) if strict else (0, 64, 65))
            paths.append(path)
        compare_bytes(*paths)
        compare_bytes(paths[0].parents[1] / "trajectory.txt",
                      paths[1].parents[1] / "trajectory.txt")
        print(f"PASS {label}: checkpoint and trajectory byte-identical after "
              "routine polls", flush=True)


def empty_owner(reference: Path, executable: Path, root: Path,
                centres: Path, measure_shards: int = 4) -> None:
    # Widen the existing fixture to Ny=80 so all four cells belong to rank 0.
    # Rank 0 stages no incoming payload; both staging events must still complete.
    paths = []
    for name, binary in (("reference", reference), ("two-gpu", executable)):
        path = run(binary, root / f"empty-owner-{name}", 10, centres=centres,
                   rho=4 * math.pi * 25 / 80**2, two_gpu=name == "two-gpu",
                   measure_shards=measure_shards)
        if struct.unpack_from("<q", snapshot(path).params, 8)[0] != 80:
            raise ValueError("empty-owner fixture must have Ny=80")
        paths.append(path)
    log = (paths[1].parents[1] / "run.log").read_text(encoding="utf-8")
    match = re.search(r"^\[two-gpu\] initial owners 4/0; seam bytes (\d+)/0$",
                      log, re.MULTILINE)
    if match is None or int(match[1]) <= 0:
        raise ValueError("empty-owner fixture did not establish a zero-payload peer")
    compare_bytes(*paths)
    compare_bytes(paths[0].parents[1] / "trajectory.txt",
                  paths[1].parents[1] / "trajectory.txt")
    print("PASS empty owner: checkpoint and trajectory byte-identical with "
          "zero peer payload", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measure-shards", type=int, default=4,
                        help="base measurement CTAs per cell, pinned for every leg (1..64; default 4)")
    parser.add_argument("--case", choices=("all", "partial-seam", "boundary-interior",
                                          "sparse-output", "empty-owner"),
                        default="all")
    args = parser.parse_args()
    if not 1 <= args.measure_shards <= 64:
        parser.error("--measure-shards must lie in 1..64")
    reference = args.reference.resolve(strict=True)
    executable = args.executable.resolve(strict=True)
    root = args.output_dir.resolve()
    root.mkdir()  # Refuse reuse, including partial output from an earlier failure.
    if args.case == "partial-seam":
        partial_seam(reference, executable, root, args.measure_shards)
        print("TWO_GPU_SMOKE_PASS", flush=True)
        return 0
    if args.case == "boundary-interior":
        boundary_interior(reference, executable, root, args.measure_shards)
        print("TWO_GPU_SMOKE_PASS", flush=True)
        return 0
    centres = root / "centres.csv"
    centres.write_text("global_id,x,y\n0,2,4\n1,18,4\n2,2,20\n3,18,20\n",
                       encoding="ascii")
    if args.case == "sparse-output":
        sparse_outputs(reference, executable, root, centres, args.measure_shards)
        print("TWO_GPU_SMOKE_PASS", flush=True)
        return 0
    if args.case == "empty-owner":
        empty_owner(reference, executable, root, centres, args.measure_shards)
        print("TWO_GPU_SMOKE_PASS", flush=True)
        return 0

    whole = {}
    for strict, steps in [(False, 1), (False, 10), (False, 100), (True, 100)]:
        label = f"{'strict' if strict else 'fast'}-{steps}"
        paths = []
        for name, binary in [("reference", reference), ("two-gpu", executable)]:
            paths.append(run(binary, root / f"{label}-{name}", steps,
                             centres=centres, strict=strict, two_gpu=name == "two-gpu",
                             measure_shards=args.measure_shards))
        compare_bytes(*paths)
        compare_bytes(paths[0].parents[1] / "trajectory.txt",
                      paths[1].parents[1] / "trajectory.txt")
        whole[strict] = paths
        print(f"PASS {label}: checkpoint and trajectory byte-identical", flush=True)

    final = snapshot(whole[False][0], 100)
    if sum(struct.unpack_from("<I", cell[0], 140)[0] for cell in final.cells) == 0:
        raise ValueError("active fixture produced no tumble events")

    for strict in (False, True):
        label = "strict" if strict else "fast"
        resumed = []
        for index, (name, binary) in enumerate(
                [("reference", reference), ("two-gpu", executable)]):
            start = run(binary, root / f"split-{label}-{name}-50", 50,
                        centres=centres, strict=strict, two_gpu=name == "two-gpu",
                        measure_shards=args.measure_shards)
            end = run(binary, root / f"split-{label}-{name}-100", 100,
                      centres=centres, resume=start, strict=strict,
                      two_gpu=name == "two-gpu", measure_shards=args.measure_shards)
            resumed.append(end)
            compare_bytes(whole[strict][index], end)
        compare_bytes(*resumed)
        compare_bytes(resumed[0].parents[1] / "trajectory.txt",
                      resumed[1].parents[1] / "trajectory.txt")
        print(f"PASS {label} 50+50 restart: whole state preserved", flush=True)

    promoted = root / "promoted-fixture.pf3d"
    promote_fixture(root / "fast-10-reference" / "checkpoints" / "checkpoint.pf3d",
                    promoted)
    paths = []
    for name, binary in [("reference", reference), ("two-gpu", executable)]:
        paths.append(run(binary, root / f"promoted-{name}", 20, centres=centres,
                         resume=promoted, two_gpu=name == "two-gpu",
                         measure_shards=args.measure_shards))
    compare_bytes(*paths)
    compare_bytes(paths[0].parents[1] / "trajectory.txt",
                  paths[1].parents[1] / "trajectory.txt")
    print("PASS mixed promoted/base continuation: checkpoint and trajectory byte-identical")
    migration = root / "migration-fixture.pf3d"
    migration_fixture(root / "fast-10-reference" / "checkpoints" / "checkpoint.pf3d",
                      migration)
    paths = []
    for name, binary in [("reference", reference), ("two-gpu", executable)]:
        paths.append(run(binary, root / f"migration-{name}", 210, centres=centres,
                         resume=migration, two_gpu=name == "two-gpu",
                         measure_shards=args.measure_shards))
        require_migration(migration, paths[-1])
    compare_bytes(*paths)
    compare_bytes(paths[0].parents[1] / "trajectory.txt",
                  paths[1].parents[1] / "trajectory.txt")
    print("PASS migration across middle and periodic y cuts: state byte-identical")
    partial_seam(reference, executable, root, args.measure_shards)
    boundary_interior(reference, executable, root, args.measure_shards)
    sparse_outputs(reference, executable, root, centres, args.measure_shards)
    empty_owner(reference, executable, root, centres, args.measure_shards)
    print("TWO_GPU_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"TWO_GPU_SMOKE_FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
