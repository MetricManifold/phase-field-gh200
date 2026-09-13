#!/usr/bin/env python3
"""Tiny boundary-observer integration smoke; run on supported GH200/SM90 GPUs.

No build, scheduler, or remote access is performed. Failed runs retain their
unique directory and logs; successful runs clean up unless --keep-output is set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

from run_two_gpu_smoke import DT, Snapshot, compare_bytes, fresh_arguments, snapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pf3d_boundary import BoundaryReader, Frame  # noqa: E402


FINAL, SPLIT, INTERVAL = 41, 19, 7
TAU = 0.002


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checkpoint(output: Path, step: int | None = None) -> Path:
    name = "checkpoint.pf3d" if step is None else f"checkpoint_{step:012d}.pf3d"
    return output / "checkpoints" / name


def launch(executable: Path, output: Path, centres: Path, steps: int,
           timeout: int, *, boundary: Path | None = None,
           resume: Path | None = None, reject: bool = False,
           two_gpu: bool = False, projection: str = "basal") -> None:
    output.mkdir()  # Never reuse a launch directory, even after a failed run.
    arguments = ["--checkpoint", str(resume)] if resume else fresh_arguments(centres)
    if resume is None:
        arguments[arguments.index("--tau") + 1] = str(TAU)
    arguments += [
        "--t-end", format(steps * DT, ".12g"), "--memory-mode", "throughput",
        "--measure-shards", "4", "--promoted-measure-shards", "4",
        "--print-interval", "0", "--trajectory-interval", "17",
        "--out", str(output / "trajectory.txt"),
        "--save-interval", str(SPLIT), "--checkpoint-dir", str(output / "checkpoints"),
    ]
    if boundary is not None:
        arguments += ["--boundary-out", str(boundary), "--boundary-interval", str(INTERVAL),
                      "--boundary-projection", projection, "--boundary-compression", "none"]
    command = [str(executable), *arguments]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    with (output / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if reject:
        require(result.returncode != 0, f"{output}: existing boundary path was accepted")
        return
    require(result.returncode == 0, f"{output}: exit {result.returncode}; see run.log")
    require(snapshot(checkpoint(output), steps).header.base_measure_shards == 4,
            f"{output}: measurement grouping changed")
    if two_gpu:
        log = (output / "run.log").read_text(encoding="utf-8")
        owners = re.search(r"\[two-gpu\] initial owners (\d+)/(\d+)", log)
        require("[two-gpu] devices " in log and owners is not None and
                all(int(count) > 0 for count in owners.groups()) and
                sum(map(int, owners.groups())) == 4,
                f"{output}: fixture did not exercise both GPU owners")


def metadata(frame: Frame, state: Snapshot) -> None:
    for cell, (record, _) in zip(frame.cells, state.cells):
        gid, ox, oy, oz = struct.unpack_from("<4q", record)
        edge = struct.unpack_from("<I", record, 184)[0]
        gamma, speed, radius = struct.unpack_from("<3f", record, 56)
        require((cell.id, cell.origin_x, cell.origin_y, cell.origin_z,
                 cell.brick_edge, cell.plane_edge, cell.gamma, cell.active_speed, cell.radius)
                == (gid, ox - 1, oy - 1, oz, edge, edge + 2, gamma, speed, radius),
                f"step {frame.step}: boundary metadata differs from checkpoint")


def read_boundary(path: Path, start: int, final: int, state: Snapshot,
                  projection: str = "basal") -> dict[int, Frame]:
    expected = sorted({start, final, *range((start // INTERVAL + 1) * INTERVAL,
                                         final + 1, INTERVAL)})
    with BoundaryReader(path) as reader:
        frames = list(reader)  # Exhaustion validates every FNV checksum and clean EOF.
        h = reader.header
        require(reader.complete and h.projection_name == projection and h.codec == 0 and
                h.boundary_flags == 3 and h.cells == 4 and h.interval == INTERVAL and
                h.dt == DT and h.tau == TAU and h.level == 0.5 and
                (h.nx, h.ny, h.nz) == struct.unpack_from("<3q", state.params),
                f"{path}: header or completion mismatch")
    require([frame.step for frame in frames] == expected,
            f"{path}: expected absolute boundary steps {expected}, got {[f.step for f in frames]}")
    for frame in frames:
        require([cell.id for cell in frame.cells] == [0, 1, 2, 3] and frame.square_count > 0 and
                math.isclose(frame.time, frame.step * DT, rel_tol=0, abs_tol=1e-12),
                f"{path}: missing/duplicate global IDs, empty contour, or incorrect time")
    metadata(frames[-1], state)
    return {frame.step: frame for frame in frames}


def frame_signature(frame: Frame) -> tuple:
    # Public reader API; pack corners again to retain binary32 bit comparisons.
    return (frame.step, frame.time, tuple(
        (cell.id, cell.origin_x, cell.origin_y, cell.origin_z, cell.brick_edge,
         cell.plane_edge, cell.gamma, cell.active_speed, cell.radius,
         tuple((s.x, s.y, struct.pack("<4f", *s.phi)) for s in cell.squares))
        for cell in frame.cells))


def counters(state: Snapshot) -> list[int]:
    return [struct.unpack_from("<I", record, 140)[0] for record, _ in state.cells]


def compare_pair(a: Path, b: Path) -> None:
    names = sorted(path.name for path in (a / "checkpoints").glob("*.pf3d"))
    require(names == sorted(path.name for path in (b / "checkpoints").glob("*.pf3d")),
            "enabled/disabled checkpoint schedules differ")
    for name in names:
        snapshot(a / "checkpoints" / name)
        snapshot(b / "checkpoints" / name)
        compare_bytes(a / "checkpoints" / name, b / "checkpoints" / name)
    compare_bytes(a / "trajectory.txt", b / "trajectory.txt")


def exercise(executable: Path, root: Path, centres: Path, timeout: int) -> Path:
    outputs: dict[str, Path] = {}
    for label, steps, resume in (("initial", 0, None), ("whole", FINAL, None),
                                 ("resumed", FINAL, "whole")):
        for enabled in (False, True):
            key = f"{label}-{'on' if enabled else 'off'}"
            output = outputs[key] = root / key
            source = checkpoint(outputs["whole-on"], SPLIT) if resume else None
            launch(executable, output, centres, steps, timeout, resume=source,
                   boundary=output / "boundary.raw" if enabled else None)
        compare_pair(outputs[f"{label}-off"], outputs[f"{label}-on"])
        print(f"PASS {label}: complete checkpoints and trajectories observer-invariant", flush=True)

    initial = snapshot(checkpoint(outputs["initial-on"]), 0)
    whole = snapshot(checkpoint(outputs["whole-on"]), FINAL)
    middle = snapshot(checkpoint(outputs["whole-on"], SPLIT), SPLIT)
    resumed = snapshot(checkpoint(outputs["resumed-on"]), FINAL)
    for state in (initial, middle, whole, resumed):
        require(struct.unpack_from("<QQ", state.params, 168) == (20260909, 20260910) and
                struct.unpack_from("<Q", state.params, 224)[0] != 0,
                "initialization/polarity seeds or centre identity hash changed")
    require(counters(initial) == [0] * 4, "initial boundary capture consumed a tumble")
    for record, _ in initial.cells:
        polarity = struct.unpack_from("<3f", record, 32)
        require(math.isclose(sum(v * v for v in polarity), 1.0, abs_tol=2e-5) and
                polarity[2] == 0 and struct.unpack_from("<f", record, 60)[0] > 0,
                "fixture lacks active planar polarity")
    require(sum(counters(middle)) > 0 and sum(counters(resumed)) > sum(counters(middle)) and
            all(a <= b for a, b in zip(counters(middle), counters(resumed))),
            "checkpoint counters show no tumble before/after restart or a counter reset")
    compare_bytes(checkpoint(outputs["whole-on"]), checkpoint(outputs["resumed-on"]))
    print(f"PASS restart RNG continuity: recorded tumbles {sum(counters(middle))} -> "
          f"{sum(counters(resumed))}; full final state byte-identical", flush=True)

    read_boundary(outputs["initial-on"] / "boundary.raw", 0, 0, initial)
    frames = read_boundary(outputs["whole-on"] / "boundary.raw", 0, FINAL, whole)
    continued = read_boundary(outputs["resumed-on"] / "boundary.raw", SPLIT, FINAL, resumed)
    metadata(continued[SPLIT], middle)
    for step in frames.keys() & continued.keys():
        require(frame_signature(frames[step]) == frame_signature(continued[step]),
                f"resumed boundary frame {step} differs from uninterrupted output")

    maximum = root / "maximum-on"
    launch(executable, maximum, centres, FINAL, timeout, projection="maximum",
           boundary=maximum / "boundary.raw")
    compare_pair(outputs["whole-off"], maximum)
    read_boundary(maximum / "boundary.raw", 0, FINAL, whole, projection="maximum")
    print("PASS maximum projection: readable frames and byte-identical observer-free state", flush=True)

    existing = outputs["whole-on"] / "boundary.raw"
    before = existing.read_bytes()
    launch(executable, root / "refused-existing", centres, FINAL, timeout,
           resume=checkpoint(outputs["whole-on"], SPLIT), boundary=existing, reject=True)
    require(existing.read_bytes() == before, "refused boundary path was overwritten")
    print("PASS initial/interval/final frames, absolute restart cadence, and no overwrite", flush=True)
    return outputs["whole-on"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--two-gpu-executable", type=Path)
    parser.add_argument("--output-parent", type=Path,
                        help="existing parent for a unique temporary test directory")
    parser.add_argument("--keep-output", action="store_true",
                        help="also retain successful artifacts for independent inspection")
    parser.add_argument("--timeout", type=int, default=90, help="seconds per tiny simulator launch")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    executable = args.executable.resolve(strict=True)
    peer = args.two_gpu_executable.resolve(strict=True) if args.two_gpu_executable else None
    parent = (args.output_parent or Path(tempfile.gettempdir())).resolve(strict=True)
    root = Path(tempfile.mkdtemp(prefix="pf3d-boundary-smoke-", dir=parent)).resolve()
    print(f"Boundary smoke workspace: {root}", flush=True)
    try:
        centres = root / "centres.csv"
        centres.write_text("global_id,x,y\n0,2,4\n1,18,4\n2,2,20\n3,18,20\n", encoding="ascii")
        reference = exercise(executable, root, centres, args.timeout)
        if peer is not None:
            off, on = root / "two-gpu-off", root / "two-gpu-on"
            for output in (off, on):
                launch(peer, output, centres, FINAL, args.timeout, two_gpu=True,
                       boundary=output / "boundary.raw" if output == on else None)
            compare_pair(off, on)
            compare_bytes(checkpoint(reference), checkpoint(on))
            read_boundary(on / "boundary.raw", 0, FINAL, snapshot(checkpoint(on), FINAL))
            compare_bytes(reference / "boundary.raw", on / "boundary.raw")
            print("PASS two-GPU gather: both owners, all IDs, invariant state and identical boundaries",
                  flush=True)
    except BaseException:
        print(f"Boundary smoke failed; preserved outputs and logs: {root}", file=sys.stderr)
        raise
    if args.keep_output:
        print(f"BOUNDARY_SMOKE_PASS (outputs retained: {root})", flush=True)
        return 0
    # Validate the exact exclusively-created target before recursive cleanup.
    require(not root.is_symlink() and root.resolve() == root and root.parent == parent and
            root.name.startswith("pf3d-boundary-smoke-"), "unsafe temporary cleanup target")
    shutil.rmtree(root)
    print("BOUNDARY_SMOKE_PASS (temporary outputs removed)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"BOUNDARY_SMOKE_FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
