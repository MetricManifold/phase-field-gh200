#!/usr/bin/env python3
"""Tiny mixed-storage test of explicit checkpoint measurement regrouping.

Uses only the standard library and the public two-GPU smoke's R=5/N=4
fixture. Two exact-zero-padded cubes exercise the promoted measurement
kernels; two unchanged cubes retain the base path. Different reduction
policies are compared with bounded numerical tolerances, never a universal
field-bitwise requirement. Event timing and polarity remain exact contracts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from run_two_gpu_smoke import DT, Snapshot, fresh_arguments, promote_fixture, snapshot


@dataclass
class Run:
    state: Snapshot
    checkpoint: Path
    trajectory: Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def policy(state: Snapshot) -> tuple[int, int]:
    return struct.unpack_from("<qQ", state.params, 240)


def launch(executable: Path, output: Path, steps: int, *, centres: Path,
           resume: Path | None = None, shards: int | None = None,
           allow: bool = False, failure: str | None = None,
           two_gpu: bool = False) -> Run | None:
    output.mkdir()  # Refuse reuse; retain all diagnostics on failure.
    arguments = (["--checkpoint", str(resume)] if resume
                 else fresh_arguments(centres))
    arguments += [
        "--t-end", format(steps * DT, ".12g"),
        "--memory-mode", "throughput", "--measure-shards", "4",
        "--print-interval", "0", "--trajectory-interval", "1",
        "--out", str(output / "trajectory.txt"),
        "--checkpoint-dir", str(output / "checkpoints"),
    ]
    if shards is not None:
        arguments += ["--promoted-measure-shards", str(shards)]
    if allow:
        arguments.append("--allow-promoted-measure-regroup")
    completed = subprocess.run([str(executable), *arguments], cwd=output,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=60)
    (output / "run.log").write_text(completed.stdout, encoding="utf-8")
    checkpoint = output / "checkpoints" / "checkpoint.pf3d"
    if failure is not None:
        require(completed.returncode != 0 and failure in completed.stdout,
                f"{output.name}: expected failure containing {failure!r}; see run.log")
        require(not checkpoint.exists(),
                f"{output.name}: rejected request wrote a checkpoint")
        return None
    require(completed.returncode == 0,
            f"{output.name}: exit {completed.returncode}; see run.log")
    if two_gpu:
        require("[two-gpu] devices " in completed.stdout,
                f"{output.name}: missing two-device runner banner")
    if allow:
        require("floating-point reduction grouping changes" in completed.stdout,
                f"{output.name}: missing regroup notice")
    state = snapshot(checkpoint, steps)
    require(state.header.base_measure_shards == 4,
            f"{output.name}: regroup changed base measurement shards")
    expected_policy = shards if shards is not None else policy(snapshot(resume))[0]
    require(policy(state) == (expected_policy, 0),
            f"{output.name}: promoted policy was not persisted")
    trajectory = output / "trajectory.txt"
    metadata = trajectory.read_text(encoding="utf-8").splitlines()[0]
    require(f" promoted_measure_policy={expected_policy} " in metadata and
            " promoted_measure_auto_wave_ctas=0 " in metadata and
            " base_measure_shards=4 " in metadata,
            f"{output.name}: trajectory metadata does not identify new grouping")
    require(sum(struct.unpack_from("<I", record, 184)[0] > state.header.brick_edge
                for record, _ in state.cells) == (2 if resume else 0),
            f"{output.name}: expected a genuinely mixed promoted/base fixture")
    return Run(state, checkpoint, trajectory)


def near_values(a: bytes, b: bytes, fmt: str, *, atol: float,
                rtol: float, label: str) -> None:
    require(len(a) == len(b), f"{label}: size mismatch")
    for index, (left, right) in enumerate(zip(struct.iter_unpack(fmt, a),
                                             struct.iter_unpack(fmt, b))):
        x, y = left[0], right[0]
        require(math.isfinite(x) and math.isfinite(y) and
                abs(x - y) <= atol + rtol * max(abs(x), abs(y)),
                f"{label}[{index}]: {x!r} vs {y!r}")


def compare_state(a: Snapshot, b: Snapshot, *, same_step_restore: bool = False,
                  exact_policy: bool = False) -> None:
    require(a.header.step == b.header.step and a.header.time == b.header.time and
            a.header.num_cells == b.header.num_cells and
            a.header.brick_edge == b.header.brick_edge and
            a.header.base_measure_shards == b.header.base_measure_shards,
            "checkpoint time/identity/storage/base grouping changed")
    require(a.params[:240] == b.params[:240] and a.params[256:] == b.params[256:],
            "regroup changed physical, RNG, initialization, or output parameters")
    if exact_policy:
        require(a.params == b.params, "same-policy continuation changed parameters")
    require(len(a.cells) == len(b.cells), "cell count mismatch")
    for (left, left_phi), (right, right_phi) in zip(a.cells, b.cells):
        gid = struct.unpack_from("<q", left)[0]
        # Exact identity/origin/polarity, heterogeneity, counters, bounds,
        # pending shifts, storage and reserved state. Motion and measurements
        # are derived from reductions and are checked numerically below.
        require(left[:44] == right[:44] and left[56:68] == right[56:68] and
                left[136:] == right[136:],
                f"cell {gid}: discrete state, polarity, or heterogeneity changed")
        if same_step_restore or exact_policy:
            require(left_phi == right_phi,
                    f"cell {gid}: phase field changed across same-step/same-policy restore")
        else:
            near_values(left_phi, right_phi, "<f", atol=2e-6, rtol=2e-6,
                        label=f"cell {gid} phase after 41 active steps")
        if exact_policy:
            require(left == right, f"cell {gid}: same-policy whole/split state mismatch")
        else:
            near_values(left[44:56], right[44:56], "<f", atol=2e-6, rtol=2e-5,
                        label=f"cell {gid} derived velocity")
            near_values(left[68:72], right[68:72], "<f", atol=2e-6, rtol=2e-6,
                        label=f"cell {gid} phi maximum")
            # Double-precision accumulation changes ordering; input fields
            # remain float32. Same-step checks use reduction-roundoff bounds.
            near_values(left[72:136], right[72:136], "<d",
                        atol=1e-8 if same_step_restore else 2e-5,
                        rtol=1e-11 if same_step_restore else 2e-5,
                        label=f"cell {gid} derived moments")


def polarity_frames(path: Path) -> dict[tuple[int, int], tuple[str, ...]]:
    frames = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        values = line.split()
        require(len(values) == 17, f"{path}: unexpected trajectory columns")
        frame_step = round(float(values[0]) / DT)
        key = (frame_step, int(values[1]))
        require(key not in frames, f"{path}: repeated trajectory identity/time")
        frames[key] = tuple(values[8:11])
    return frames


def events(frames: dict, initial: Snapshot, final: Snapshot) -> list[tuple[int, int, int]]:
    """Recover (absolute RNG step, global ID, tumble ordinal) from every frame.

    A completed frame at step s records the polarity consumed at RNG step
    s-1. Check final counters so no event can silently disappear from this
    polarity-transition observation.
    """
    expected_keys = {(step, struct.unpack_from("<q", record)[0])
                     for step in range(initial.header.step, final.header.step + 1)
                     for record, _ in initial.cells}
    require(set(frames) == expected_keys, "trajectory has missing or extra frames")
    result = []
    for (start, _), (end, _) in zip(initial.cells, final.cells):
        gid = struct.unpack_from("<q", start)[0]
        count = struct.unpack_from("<I", start, 140)[0]
        for step in range(initial.header.step + 1, final.header.step + 1):
            if frames[(step, gid)] != frames[(step - 1, gid)]:
                count += 1
                result.append((step - 1, gid, count))
        require(count == struct.unpack_from("<I", end, 140)[0],
                f"cell {gid}: observed event count disagrees with checkpoint counter")
    require(result, "active continuation produced no tumble events")
    return sorted(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True,
                        help="single-GPU phase_field_3d executable")
    parser.add_argument("--two-gpu-executable", type=Path,
                        help="optionally check two-GPU regroup and gathered checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    two_gpu = args.two_gpu_executable.resolve(strict=True) if args.two_gpu_executable else None
    parent = args.output_dir.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    print(f"Retained smoke outputs: {root}", flush=True)
    centres = root / "centres.csv"
    centres.write_text("global_id,x,y\n0,2,4\n1,18,4\n2,2,20\n3,18,20\n",
                       encoding="ascii")

    def run(name: str, steps: int, **kwargs) -> Run | None:
        return launch(executable, root / name, steps, centres=centres, **kwargs)

    fresh = run("fresh-4", 0, shards=4)
    padded = root / "mixed-zero-padded.pf3d"
    promote_fixture(fresh.checkpoint, padded)
    # A native save after full remeasurement removes any synthetic-moment
    # differences before the actual regroup comparison.
    old = run("saved-mixed-4", 0, resume=padded, shards=4)
    run("guard-mismatch", 0, resume=old.checkpoint, shards=64,
        failure="would change the numerical reduction grouping")
    run("guard-missing-policy", 0, resume=old.checkpoint, allow=True,
        failure="requires --checkpoint and explicit --promoted-measure-shards")
    run("guard-fresh", 0, shards=64, allow=True,
        failure="requires --checkpoint and explicit --promoted-measure-shards")
    regrouped = run("same-step-64", 0, resume=old.checkpoint, shards=64, allow=True)
    compare_state(old.state, regrouped.state, same_step_restore=True)
    print("PASS same-step 4 -> 64: exact phi, polarity, identity and parameters; "
          "derived reductions checked within roundoff", flush=True)

    old_final = run("active-4", 41, resume=old.checkpoint)
    whole = run("whole-64", 41, resume=regrouped.checkpoint)
    first = run("split-64-first", 19, resume=regrouped.checkpoint)
    second = run("split-64-second", 41, resume=first.checkpoint)
    run("guard-stored-64", 41, resume=whole.checkpoint, shards=4,
        failure="would change the numerical reduction grouping")
    compare_state(old_final.state, whole.state)
    compare_state(whole.state, second.state, exact_policy=True)
    old_frames = polarity_frames(old_final.trajectory)
    whole_frames = polarity_frames(whole.trajectory)
    first_frames = polarity_frames(first.trajectory)
    second_frames = polarity_frames(second.trajectory)
    require(all(first_frames[key] == second_frames[key]
                for key in first_frames.keys() & second_frames.keys()),
            "split boundary changed polarity")
    split_frames = first_frames | second_frames
    require(old_frames == whole_frames == split_frames,
            "4/64 or whole/split per-step polarity mismatch")
    old_events = events(old_frames, old.state, old_final.state)
    require(old_events == events(whole_frames, regrouped.state, whole.state) ==
            events(split_frames, regrouped.state, second.state),
            "4/64 or whole/split event tuple mismatch")
    print(f"PASS 41 active steps: {len(old_events)} exact event tuples and all "
          "per-step polarities; 64 whole/split state identical; omitted policy "
          "restores 64 and unauthorized override fails", flush=True)

    if two_gpu:
        gathered = launch(two_gpu, root / "two-gpu-same-step-64", 0,
                          centres=centres, resume=old.checkpoint,
                          shards=64, allow=True, two_gpu=True)
        compare_state(regrouped.state, gathered.state, exact_policy=True)
        continued = launch(two_gpu, root / "two-gpu-omitted-64", 41,
                           centres=centres, resume=gathered.checkpoint, two_gpu=True)
        compare_state(whole.state, continued.state, exact_policy=True)
        require(polarity_frames(continued.trajectory) == whole_frames,
                "two-GPU gathered continuation changed polarity events")
        print("PASS two-GPU regroup/gather and omitted-policy restore retain 64", flush=True)
    print("PROMOTED_REGROUP_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"PROMOTED_REGROUP_SMOKE_FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
