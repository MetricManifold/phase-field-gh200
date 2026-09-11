"""H100 regression: velocity and boundary recorders together and on restart."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from velocity_moments import frames, read_header

BOUNDARY_HEADER = struct.Struct("<8s6I3dQ2I")
BOUNDARY_FRAME = struct.Struct("<8sQdQQQ")
BOUNDARY_CELL = struct.Struct("<iii6fI")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_velocity(path: Path):
    with path.open("rb") as stream:
        header = read_header(stream)
        return header, list(frames(stream, header))


def read_boundary(path: Path) -> dict:
    """Independently check raw frame checksums and normalize square order."""
    result = {}
    with path.open("rb") as stream:
        header = BOUNDARY_HEADER.unpack(stream.read(BOUNDARY_HEADER.size))
        assert header[0] == b"PFBND01\0" and header[1] == 1 and header[-2] == 0
        cells, dt = header[3], header[7]
        while True:
            raw = stream.read(BOUNDARY_FRAME.size)
            assert len(raw) == BOUNDARY_FRAME.size, "missing boundary terminator"
            magic, step, time, raw_bytes, stored_bytes, checksum = BOUNDARY_FRAME.unpack(raw)
            assert time == step*dt
            if magic == b"PFBEND1\0":
                assert checksum == len(result) and step == max(result)
                assert not stream.read(1)
                return result
            assert magic == b"PFBFRM1\0" and raw_bytes == stored_bytes
            payload = stream.read(stored_bytes)
            assert len(payload) == stored_bytes
            fingerprint = 14695981039346656037
            for byte in payload:
                fingerprint = ((fingerprint ^ byte)*1099511628211) & ((1 << 64)-1)
            assert fingerprint == checksum
            metadata = list(BOUNDARY_CELL.iter_unpack(payload[:cells*BOUNDARY_CELL.size]))
            offset = cells*BOUNDARY_CELL.size
            normalized = []
            for cell in metadata:
                end = offset+cell[-1]*20
                squares = tuple(sorted(payload[i:i+20] for i in range(offset, end, 20)))
                normalized.append((cell, squares))
                offset = end
            assert offset == len(payload) and step not in result
            result[step] = tuple(normalized)


def run(exe: Path, directory: Path, options: list, *, velocity=False,
        boundary=False, extra=()) -> None:
    directory.mkdir()
    command = [str(exe), *map(str, options), "--out", str(directory/"trajectory.txt"),
               "--checkpoint-dir", str(directory)]
    if velocity:
        command += ["--velocity-moments", str(directory/"velocity.bin")]
    if boundary:
        command += ["--boundary-out", str(directory/"boundaries.pfb"),
                    "--boundary-interval", "1000"]
    command += list(extra)
    result = subprocess.run(command, capture_output=True, text=True)
    (directory/"run.log").write_text(result.stdout+result.stderr)
    assert result.returncode == 0, (command, result.stdout[-2000:], result.stderr[-2000:])


def assert_same_physics(directories: list[Path]) -> None:
    for name in ("checkpoint.bin", "trajectory.txt"):
        hashes = {str(path): sha(path/name) for path in directories}
        assert len(set(hashes.values())) == 1, (name, hashes)


def check_join(velocity, boundary) -> int:
    by_step = {frame.step: frame for frame in velocity}
    joined = 0
    for step in sorted(by_step.keys() & boundary.keys()):
        rows = {int(row[0]): row for row in by_step[step].rows}
        for cell, _squares in boundary[step]:
            row = rows[cell[0]]
            assert abs(cell[3]-row[1]) < 1e-4 and abs(cell[4]-row[2]) < 1e-4
            assert tuple(cell[5:8]) == tuple(row[4:7])
            joined += 1
    assert joined
    return joined


def validate(exe: Path, baseline: Path | None, root: Path) -> dict:
    common = ["--N", 36, "--radius", 49, "--rho", 0.9, "--dt", 0.01,
              "--v-A", 0.01, "--gamma", 1, "--gamma-cancer", 0.35,
              "--cancer-fraction", 1/3, "--seed", 20260911,
              "--polarity-seed", 3110911, "--t-end", 120.03,
              "--trajectory-interval", 1000, "--print-interval", 1000000,
              "--full-moment", 1000]
    modes = {
        "off": (False, False, []),
        "velocity": (True, False, []),
        "dense": (True, False, ["--velocity-spatial-stride", "1"]),
        "reference": (True, False, ["--velocity-moments-reference"]),
        "boundary": (False, True, []),
        "both": (True, True, []),
        "no_graph": (True, True, ["--no-graph"]),
        "morton": (True, True, ["--morton"]),
    }
    outputs = []
    for name, (velocity, boundary, extra) in modes.items():
        run(exe, root/name, common, velocity=velocity, boundary=boundary, extra=extra)
        outputs.append(root/name)
    if baseline:
        run(baseline, root/"baseline", common)
        outputs.append(root/"baseline")
    assert_same_physics(outputs)
    for mode in ("both", "no_graph", "morton"):
        assert sha(root/mode/"velocity.bin") == sha(root/"velocity/velocity.bin")
    sparse_header, sparse = read_velocity(root/"velocity/velocity.bin")
    _, dense = read_velocity(root/"dense/velocity.bin")
    _, reference = read_velocity(root/"reference/velocity.bin")
    errors = [0.0]*12
    for sampled, exact, oracle in zip(sparse, dense, reference):
        assert sampled.step == exact.step == oracle.step
        for a, b, c in zip(sampled.rows, exact.rows, oracle.rows):
            assert a[:13] == b[:13], "physics metadata or exact A/I changed"
            if sampled.step <= sparse_header.dense_start:
                assert a == b, "dense startup changed"
            for j in range(12):
                errors[j] = max(errors[j], abs(b[9+j]-c[9+j]))
    assert max(errors) < 1e-8, errors
    boundaries = read_boundary(root/"boundary/boundaries.pfb")
    both = read_boundary(root/"both/boundaries.pfb")
    assert both == boundaries, "velocity recording changed boundary data"
    joined = check_join(sparse, both)

    # Neither output cadence divides this restart offset. Cross a deliberately
    # short dense startup and finish inside a spatial interval.
    restart = ["-c", root/"off/checkpoint.bin", "--t-end", 123.19,
               "--trajectory-interval", 37, "--print-interval", 1000000]
    run(exe, root/"restart_off", restart)
    extra = ["--velocity-dense-start", "17", "--velocity-spatial-stride", "25",
             "--boundary-interval", "37"]
    run(exe, root/"restart_both", restart, velocity=True, boundary=True, extra=extra)
    run(exe, root/"restart_no_graph", restart, velocity=True, boundary=True,
        extra=extra+["--no-graph"])
    assert_same_physics([root/name for name in
                         ("restart_off", "restart_both", "restart_no_graph")])
    assert sha(root/"restart_both/velocity.bin") == sha(root/"restart_no_graph/velocity.bin")
    restart_header, resumed = read_velocity(root/"restart_both/velocity.bin")
    assert restart_header.start_step == 12003 and resumed[-1].step == 12319
    resumed_boundaries = read_boundary(root/"restart_both/boundaries.pfb")
    assert min(resumed_boundaries) == 12003 and max(resumed_boundaries) == 12319
    joined += check_join(resumed, resumed_boundaries)

    # A pre-existing recorder file must survive a failed attempt to reuse it.
    before = sha(root/"both/velocity.bin")
    result = subprocess.run(
        [str(exe), "-c", str(root/"off/checkpoint.bin"), "--t-end", "120.04",
         "--out", str(root/"unused.txt"), "--no-final-checkpoint",
         "--velocity-moments", str(root/"both/velocity.bin")],
        capture_output=True, text=True)
    assert result.returncode != 0 and sha(root/"both/velocity.bin") == before
    return {"status": "PASS", "modes": list(modes), "physics_identical": True,
            "boundary_payloads_identical": True, "velocity_files_identical": True,
            "joined_cell_frames": joined, "dense_reference_max_errors": errors,
            "unaligned_restart_and_partial_tail": True, "exclusive_output": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.mkdir()
        result = validate(args.executable.resolve(), args.baseline, args.output)
        (args.output/"RESULT.json").write_text(json.dumps(result, indent=2)+"\n")
    else:
        with tempfile.TemporaryDirectory(prefix="pf-observations-") as directory:
            result = validate(args.executable.resolve(), args.baseline, Path(directory))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
