#!/usr/bin/env python3
"""Compare one- and two-GPU hard-wall dynamics, restarts, and observations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from run_two_gpu_smoke import (
    CELL_RECORD_BYTES, DT, FILE_HEADER, PARAMS_BYTES, Snapshot,
    canonical_crc_header, checksum, compare_bytes, parse_file_header,
    require_mixed_tile_passes,
)
from pf3d_boundary import BoundaryReader


POSITIONS = ((48, 48), (48, 42), (50, 24), (50, 72))
# Choosing an unrounded side inside (95, 96) avoids a ceil-roundoff ambiguity.
RHO = 4 * (4 * math.pi * 5**3 / 3) / (10 * 95.5**2)


def validate_centres() -> None:
    """Check the public initializer contract before spending any GPU time."""
    side = math.ceil(math.sqrt(4 * (4 * math.pi * 5**3 / 3) / (10 * RHO)))
    if side != 96 or POSITIONS[0] != (side / 2, side / 2):
        raise ValueError("cell zero must be exactly at the channel centre")
    for i, (x, y) in enumerate(POSITIONS):
        if not (0 <= x < side and 0 <= y < side):
            raise ValueError("initial centre lies outside the lateral domain")
        for px, py in POSITIONS[:i]:
            dx, dy = abs(x - px), abs(y - py)
            if min(dx, side - dx)**2 + min(dy, side - dy)**2 < 5**2:
                raise ValueError("initial centres violate the radius-five separation")
    # CSV z=5 is physical height. The solver converts it to lattice z=7.5
    # using z + padding - 0.5, so the symmetric B24 origin is -4.
    require_mixed_tile_passes([round(y - 11.5) for _, y in POSITIONS])


def snapshot(path: Path, expected_step: int | None = None) -> Snapshot:
    raw = path.read_bytes()
    header = parse_file_header(raw[:FILE_HEADER.size])
    if expected_step is not None and header.step != expected_step:
        raise ValueError(f"{path}: expected step {expected_step}, got {header.step}")
    if checksum(canonical_crc_header(raw[:FILE_HEADER.size]) + raw[FILE_HEADER.size:]) != header.stored_crc64:
        raise ValueError(f"{path}: checksum mismatch")
    pos = FILE_HEADER.size + PARAMS_BYTES
    params = raw[FILE_HEADER.size:pos]
    if len(params) != PARAMS_BYTES or struct.unpack_from("<3q", params) != (96, 96, 16):
        raise ValueError(f"{path}: expected a 96 x 96 x 16 channel")
    if (struct.unpack_from("<I", params, 52)[0] != 11
            or struct.unpack_from("<2q", params, 256) != (10, 3)
            or struct.unpack_from("<2d", params, 272) != (10.0, 1.0)):
        raise ValueError(f"{path}: incorrect hard-wall parameters")
    cells = []
    ids = set()
    for _ in range(header.num_cells):
        record = raw[pos:pos + CELL_RECORD_BYTES]
        pos += CELL_RECORD_BYTES
        if len(record) != CELL_RECORD_BYTES:
            raise ValueError(f"{path}: truncated cell record")
        gid = struct.unpack_from("<q", record)[0]
        edge = struct.unpack_from("<I", record, 184)[0]
        if gid in ids or not header.brick_edge <= edge < 96 or edge % 8:
            raise ValueError(f"{path}: invalid identity or lateral storage extent")
        ids.add(gid)
        phase = raw[pos:pos + 4 * edge**3]
        pos += 4 * edge**3
        if len(phase) != 4 * edge**3 or not all(
                math.isfinite(value[0]) for value in struct.iter_unpack("<f", phase)):
            raise ValueError(f"{path}: truncated or nonfinite field")
        moments = struct.unpack_from("<8d", record, 72)
        polarity = struct.unpack_from("<3f", record, 32)
        if (not all(math.isfinite(value) for value in moments) or moments[0] <= 0
                or not math.isclose(sum(v * v for v in polarity), 1.0, abs_tol=1e-5)
                or struct.unpack_from("<I", record, 144)[0]):
            raise ValueError(f"{path}: invalid moments, polarity, or integrity flag")
        cells.append((record, phase))
    if pos != len(raw) or ids != set(range(4)) or header.brick_edge != 24:
        raise ValueError(f"{path}: incorrect fixture population or payload length")
    return Snapshot(raw, header, params, cells)


def boundary_frames(path: Path) -> dict[int, tuple]:
    """Read to the completion marker, retaining exact small-frame values."""
    frames = {}
    with BoundaryReader(path) as reader:
        h = reader.header
        if (h.geometry != "channel" or h.projection_name != "maximum" or h.codec != 0
                or h.cells != 4 or (h.nx, h.ny, h.nz) != (96, 96, 16)
                or (h.dt, h.tau, h.level, h.interval) != (DT, 0.02, 0.5, 10)):
            raise ValueError(f"{path}: incorrect boundary observation geometry")
        for frame in reader:
            frames[frame.step] = (frame.time, tuple(
                (cell.id, cell.origin_x, cell.origin_y, cell.origin_z,
                 cell.brick_edge, cell.plane_edge, cell.gamma, cell.active_speed,
                 cell.radius, tuple((s.x, s.y, struct.pack("<4f", *s.phi)) for s in cell.squares))
                for cell in frame.cells))
            if [cell.id for cell in frame.cells] != list(range(4)) or not math.isclose(
                    frame.time, frame.step * DT, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{path}: incorrect boundary identity or time")
        if not reader.complete or not frames or not any(
                cell[-1] for _, cells in frames.values() for cell in cells):
            raise ValueError(f"{path}: empty or incomplete boundary output")
    return frames


def run(binary: Path, directory: Path, end: int, centres: Path, *,
        resume: Path | None = None, strict: bool = False, two_gpu: bool = False,
        memory_mode: str = "throughput") -> Path:
    if memory_mode not in ("throughput", "balanced", "compact"):
        raise ValueError(f"unsupported smoke-test memory mode: {memory_mode}")
    if two_gpu and memory_mode != "throughput":
        raise ValueError("two-device execution requires throughput storage")
    directory.mkdir()
    start = snapshot(resume).header.step if resume else 0
    arguments = ["--checkpoint", str(resume)] if resume else [
        "--geometry", "channel", "--N", "4", "--radius", "5", "--lambda", "1",
        "--channel-height", "10", "--wall-padding", "3", "--wall-kappa", "10",
        "--wall-width", "1", "--rho", str(RHO), "--brick-edge", "24",
        "--dt", str(DT), "--aging-time", "0", "--tau", "0.02", "--v-A", "0.1",
        "--gamma-cancer", "0.35", "--cancer-fraction", "0.5",
        "--seed", "20260909", "--polarity-seed", "20260910",
        "--initial-centres", str(centres), "--full-moment", "10", "--verify-every", "10",
        "--measure-shards", "4", "--promoted-measure-shards", "4",
    ]
    arguments += [
        "--t-end", format(end * DT, ".12g"), "--memory-mode", memory_mode,
        "--print-interval", "0", "--trajectory-interval", "10",
        "--out", str(directory / "trajectory.txt"),
        "--boundary-out", str(directory / "boundary.pfb3d"),
        "--boundary-interval", "10", "--boundary-projection", "maximum",
        "--checkpoint-dir", str(directory / "checkpoints"),
    ]
    if strict:
        arguments.append("--strict")
    log_path = directory / "run.log"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run([str(binary), *arguments], cwd=directory,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=45)
    if completed.returncode:
        raise RuntimeError(f"{directory.name}: exit {completed.returncode}; see run.log")
    log_text = log_path.read_text(encoding="utf-8")
    if f"  mode {memory_mode}:" not in log_text:
        raise ValueError(f"{directory.name}: requested storage mode was not selected")
    if two_gpu and "[two-gpu] devices " not in log_text:
        raise ValueError(f"{directory.name}: missing two-device banner")
    path = directory / "checkpoints" / "checkpoint.pf3d"
    state = snapshot(path, end)
    if state.header.base_measure_shards != 4:
        raise ValueError("checkpoint changed measurement grouping")
    frames = boundary_frames(directory / "boundary.pfb3d")
    expected = {start, end} | set(range(((start // 10) + 1) * 10, end + 1, 10))
    if set(frames) != expected:
        raise ValueError(f"{directory.name}: missing or extra boundary steps")
    rows = [line.split() for line in (directory / "trajectory.txt").read_text(
        encoding="utf-8").splitlines() if line and not line.startswith("#")]
    sequence = [(step, gid) for step in sorted(expected) for gid in range(4)]
    if len(rows) != len(sequence) or any(
            len(row) != 21 or int(row[1]) != gid
            or not math.isclose(float(row[0]), step * DT, rel_tol=0, abs_tol=1e-12)
            or not all(math.isfinite(float(value)) for value in row)
            for row, (step, gid) in zip(rows, sequence)):
        raise ValueError("incorrect trajectory fields, identities, or output steps")
    return path


def compare_outputs(left: Path, right: Path) -> None:
    compare_bytes(left, right)
    for name in ("trajectory.txt", "boundary.pfb3d"):
        compare_bytes(left.parents[1] / name, right.parents[1] / name)


def require_output_subset(whole: Path, segment: Path) -> None:
    complete = boundary_frames(whole.parents[1] / "boundary.pfb3d")
    part = boundary_frames(segment.parents[1] / "boundary.pfb3d")
    for step, frame in part.items():
        if complete.get(step) != frame:
            raise ValueError(f"restart changed boundary frame {step}")
    def data_rows(path: Path) -> set[str]:
        return {line for line in path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")}
    if not data_rows(segment.parents[1] / "trajectory.txt") <= data_rows(whole.parents[1] / "trajectory.txt"):
        raise ValueError("restart changed trajectory rows")


def write_state(source: Snapshot, records: list[bytes], destination: Path) -> None:
    raw = bytearray(canonical_crc_header(source.raw[:FILE_HEADER.size]) + source.params + b"".join(records))
    struct.pack_into("<Q", raw, 80, checksum(raw))
    with destination.open("xb") as stream:
        stream.write(raw)


def promote_fixture(source: Path, destination: Path) -> None:
    """Enlarge one cell on each owner, without moving any world-space voxel."""
    state = snapshot(source)
    records = []
    for old_record, old_phase in state.cells:
        if struct.unpack_from("<I", old_record, 184)[0] != 24:
            raise ValueError("padding fixture requires uniformly B24 source cells")
        if struct.unpack_from("<q", old_record)[0] not in (0, 1):
            records.append(old_record + old_phase)
            continue
        record = bytearray(old_record)
        origin = struct.unpack_from("<3q", record, 8)
        struct.pack_into("<3q", record, 8, *(v - 4 for v in origin))
        volume, *moments = struct.unpack_from("<4d", record, 72)
        struct.pack_into("<3d", record, 80, *(v + 4 * volume for v in moments))
        bounds = struct.unpack_from("<6i", record, 148)
        struct.pack_into("<6i", record, 148, *(v + 4 for v in bounds))
        struct.pack_into("<I", record, 184, 32)
        phase = bytearray(4 * 32**3)
        for z in range(24):
            for y in range(24):
                src = 4 * 24 * (y + 24 * z)
                dst = 4 * (4 + 32 * (y + 4 + 32 * (z + 4)))
                phase[dst:dst + 4 * 24] = old_phase[src:src + 4 * 24]
        records.append(bytes(record) + phase)
    write_state(state, records, destination)
    promoted = snapshot(destination)
    if [struct.unpack_from("<I", r, 184)[0] for r, _ in promoted.cells] != [32, 32, 24, 24]:
        raise ValueError("mixed-storage fixture did not preserve the intended classes")
    owners = {(struct.unpack_from("<q", record, 16)[0] + 16) % 96 >= 48
              for record, _ in promoted.cells[:2]}
    if owners != {False, True}:
        raise ValueError("promoted fixture must place one enlarged cell on each owner")


def self_test() -> None:
    """Exercise the channel reader and zero-padding fixture without CUDA."""
    validate_centres()
    root = Path(tempfile.mkdtemp(prefix="pf3d-channel-fixture-"))
    paths = [root / name for name in ("source.pf3d", "promoted.pf3d", "tampered.pf3d")]
    try:
        params = bytearray(PARAMS_BYTES)
        struct.pack_into("<3q", params, 0, 96, 96, 16)
        struct.pack_into("<I", params, 52, 11)
        struct.pack_into("<2q", params, 256, 10, 3)
        struct.pack_into("<2d", params, 272, 10, 1)
        header = FILE_HEADER.pack(0x44334650, 1, 96, 0x01020304, 3, 1, 3,
                                  10, 10 * DT, 4, PARAMS_BYTES, CELL_RECORD_BYTES,
                                  24, 1, 24**3, 4 * 24**3, 0, 4)
        records = []
        for gid, (x, y) in enumerate(POSITIONS):
            record = bytearray(CELL_RECORD_BYTES)
            struct.pack_into("<4q", record, 0, gid, round(x - 11.5), round(y - 11.5), -4)
            struct.pack_into("<3f", record, 32, 0, 0, 1)
            struct.pack_into("<4d", record, 72, 1, 12, 12, 12)
            struct.pack_into("<6i", record, 148, 12, 12, 12, 12, 12, 12)
            struct.pack_into("<I", record, 184, 24)
            phase = bytearray(4 * 24**3)
            struct.pack_into("<f", phase, 4 * (12 + 24 * (12 + 24 * 12)), 1)
            records.append(bytes(record) + phase)
        source = Snapshot(header, parse_file_header(header), bytes(params), [])
        write_state(source, records, paths[0])
        promote_fixture(paths[0], paths[1])
        def world_voxels(state: Snapshot) -> dict:
            voxels = {}
            for record, phase in state.cells:
                gid, ox, oy, oz = struct.unpack_from("<4q", record)
                edge = struct.unpack_from("<I", record, 184)[0]
                for index, (value,) in enumerate(struct.iter_unpack("<I", phase)):
                    if value:
                        voxels[gid, ox + index % edge, oy + index // edge % edge,
                               oz + index // (edge * edge)] = value
            return voxels
        if world_voxels(snapshot(paths[0])) != world_voxels(snapshot(paths[1])):
            raise ValueError("zero padding changed world-space fields")
        require_mixed_tile_passes([round(y - 11.5) for _, y in POSITIONS])
        damaged = bytearray(paths[0].read_bytes())
        damaged[-1] ^= 1
        paths[2].write_bytes(damaged)
        try:
            snapshot(paths[2])
        except ValueError as error:
            if "checksum mismatch" not in str(error):
                raise
        else:
            raise ValueError("tampered checkpoint was accepted")
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        root.rmdir()
    print("CHANNEL_FIXTURE_CPU_PASS", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true", help="CPU-only fixture checks")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.reference, args.executable, args.output_dir)):
        parser.error("--reference, --executable and --output-dir are required")
    binaries = (args.reference.resolve(strict=True), args.executable.resolve(strict=True))
    validate_centres()
    root = args.output_dir.resolve()
    root.mkdir()
    centres = root / "centres.csv"
    with centres.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("global_id,x,y,z\n" + "".join(
            f"{gid},{x},{y},5\n" for gid, (x, y) in enumerate(POSITIONS)))
    whole = {}
    for strict, steps in ((False, 1), (False, 10), (False, 100), (True, 100)):
        label = f"{'strict' if strict else 'fast'}-{steps}"
        paths = [run(binary, root / f"{label}-gpu{i + 1}", steps, centres,
                     strict=strict, two_gpu=bool(i)) for i, binary in enumerate(binaries)]
        compare_outputs(*paths)
        whole[strict, steps] = paths
        print(f"PASS {label}: checkpoint, trajectory, and boundaries identical", flush=True)
    state = snapshot(whole[False, 100][0])
    if sum(struct.unpack_from("<I", record, 140)[0] for record, _ in state.cells) == 0:
        raise ValueError("active fixture produced no tumble events")
    if not any(abs(struct.unpack_from("<f", record, 40)[0]) > 1e-6 for record, _ in state.cells):
        raise ValueError("channel fixture did not retain three-dimensional polarity")
    if sum(struct.unpack_from("<f", record, 56)[0] < 0.5 for record, _ in state.cells) != 2:
        raise ValueError("fixture must contain exactly two soft cells")
    initial = snapshot(whole[False, 1][0])
    require_mixed_tile_passes([struct.unpack_from("<q", r, 16)[0] for r, _ in initial.cells])

    for strict in (False, True):
        label = "strict" if strict else "fast"
        halves = []
        for i, binary in enumerate(binaries):
            start = run(binary, root / f"{label}-gpu{i + 1}-50", 50, centres,
                        strict=strict, two_gpu=bool(i))
            end = run(binary, root / f"{label}-gpu{i + 1}-resume", 100, centres,
                      resume=start, strict=strict, two_gpu=bool(i))
            compare_bytes(whole[strict, 100][i], end)
            require_output_subset(whole[strict, 100][i], end)
            halves.append(start)
        cross_two = run(binaries[1], root / f"{label}-cross-two", 80, centres,
                        resume=halves[0], strict=strict, two_gpu=True)
        cross_one = run(binaries[0], root / f"{label}-cross-one", 100, centres,
                        resume=cross_two, strict=strict)
        compare_bytes(whole[strict, 100][0], cross_one)
        for path in (cross_two, cross_one):
            require_output_subset(whole[strict, 100][0], path)
        print(f"PASS {label}: 50+50 and one/two/one restart continuity", flush=True)

    promoted = root / "mixed-storage.pf3d"
    promote_fixture(whole[False, 10][0], promoted)
    for strict in (False, True):
        label = f"promoted-{'strict' if strict else 'fast'}"
        paths = [run(binary, root / f"{label}-gpu{i + 1}", 30, centres,
                     resume=promoted, strict=strict, two_gpu=bool(i))
                 for i, binary in enumerate(binaries)]
        compare_outputs(*paths)
        part = run(binaries[1], root / f"{label}-split", 20, centres,
                   resume=promoted, strict=strict, two_gpu=True)
        end = run(binaries[1], root / f"{label}-resume", 30, centres,
                  resume=part, strict=strict, two_gpu=True)
        compare_bytes(paths[1], end)
        require_output_subset(paths[1], end)
        print(f"PASS {label}: mixed B24/B32 > Nz16 wall-coupled states identical", flush=True)

    # In-place field and aggregate paths must obey the same observation and
    # restart contracts as throughput storage, for each evaluation policy.
    for memory_mode in ("balanced", "compact"):
        for strict in (False, True):
            label = f"{memory_mode}-{'strict' if strict else 'fast'}"
            reference = whole[strict, 100][0]
            end = run(binaries[0], root / f"{label}-100", 100, centres,
                      strict=strict, memory_mode=memory_mode)
            compare_outputs(reference, end)
            start = run(binaries[0], root / f"{label}-50", 50, centres,
                        strict=strict, memory_mode=memory_mode)
            resumed = run(binaries[0], root / f"{label}-resume", 100, centres,
                          resume=start, strict=strict, memory_mode=memory_mode)
            compare_bytes(end, resumed)
            for segment in (start, resumed):
                require_output_subset(reference, segment)
            print(f"PASS {label}: throughput parity and 50+50 restart continuity", flush=True)
    print("TWO_GPU_CHANNEL_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"TWO_GPU_CHANNEL_SMOKE_FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
