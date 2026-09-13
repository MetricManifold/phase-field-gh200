#!/usr/bin/env python3
"""Exercise automatic two-device support growth using small synthetic fields."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
import sys

from run_two_gpu_channel_smoke import (
    POSITIONS, compare_bytes, compare_outputs, require_output_subset, run,
    snapshot, validate_centres, write_state,
)


def growth_fixture(source: Path, destination: Path, ids: tuple[int, ...]) -> None:
    """Seed safe symmetric tails that diffuse across the next support margin.

    These deliberately synthetic fields test recovery, not initialization
    physics. Their support fits at restore; ordinary updates must cause growth.
    """
    state = snapshot(source)
    records = []
    owners = {}
    for record, old_phase in state.cells:
        gid, _, origin_y, origin_z = struct.unpack_from("<4q", record)
        edge = struct.unpack_from("<I", record, 184)[0]
        owners[gid] = ((origin_y + edge // 2) % 96) >= 48
        phase = bytearray(old_phase)
        if gid in ids:
            if edge != 24:
                raise ValueError("growth fixture requires B24 input")
            for z in (11, 12):
                if not 0 <= origin_z + z < 16:
                    raise ValueError("synthetic tail is outside the channel")
                for y in (11, 12):
                    for x in (4, 19):
                        struct.pack_into("<f", phase, 4 * (x + edge * (y + edge * z)), 0.25)
        records.append(record + phase)
    if owners[0] is not True or owners[1] is not False:
        raise ValueError("fixture no longer places gids 0 and 1 on opposite owners")
    write_state(state, records, destination)
    snapshot(destination, state.header.step)


def require_runtime_growth(path: Path, ids: tuple[int, ...], *, two_gpu: bool,
                           incremental: bool = False) -> None:
    text = (path.parents[1] / "run.log").read_text(encoding="utf-8")
    first_resize = text.find("[3d] resized ")
    if first_resize < 0:
        raise ValueError("fixture did not exercise automatic support growth")
    if two_gpu:
        start = text.find("[two-gpu] initial owners ")
        if start < 0 or first_resize < start:
            raise ValueError("growth occurred during restore, not distributed execution")
    if incremental and "[two-gpu] synchronized peer growth:" not in text:
        raise ValueError("candidate never exercised incremental peer growth")
    state = snapshot(path)
    grown = {struct.unpack_from("<q", record)[0] for record, _ in state.cells
             if struct.unpack_from("<I", record, 184)[0] > 24}
    if grown != set(ids):
        raise ValueError(f"unexpected resized cells: {sorted(grown)}, expected {ids}")
    if not any(struct.unpack_from("<I", record, 140)[0] for record, _ in state.cells):
        raise ValueError("active recovery fixture contains no tumble history")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="unchanged single-device binary")
    parser.add_argument("--reference-two", type=Path, required=True, help="unchanged two-device binary")
    parser.add_argument("--executable", type=Path, required=True, help="candidate two-device binary")
    parser.add_argument("--candidate-one", type=Path, help="also compare the candidate single-device binary")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    binaries = [("reference-one", args.reference.resolve(strict=True), False),
                ("reference-two", args.reference_two.resolve(strict=True), True),
                ("candidate-two", args.executable.resolve(strict=True), True)]
    if args.candidate_one:
        binaries.append(("candidate-one", args.candidate_one.resolve(strict=True), False))
    root = args.output_dir.resolve()
    root.mkdir()
    validate_centres()
    centres = root / "centres.csv"
    with centres.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("global_id,x,y,z\n" + "".join(
            f"{gid},{x},{y},5\n" for gid, (x, y) in enumerate(POSITIONS)))
    source = run(binaries[0][1], root / "accepted-source", 10, centres)
    for label, ids, strict in (("owner-zero", (1,), False),
                               ("owner-one", (0,), False),
                               ("both-strict", (0, 1), True)):
        fixture = root / f"{label}.pf3d"
        growth_fixture(source, fixture, ids)
        paths = {}
        for name, binary, two_gpu in binaries:
            path = run(binary, root / f"{label}-{name}", 40, centres,
                       resume=fixture, strict=strict, two_gpu=two_gpu)
            require_runtime_growth(path, ids, two_gpu=two_gpu,
                                   incremental=name == "candidate-two")
            if paths:
                compare_outputs(paths["reference-one"], path)
            paths[name] = path
        split = run(binaries[2][1], root / f"{label}-split", 20, centres,
                    resume=fixture, strict=strict, two_gpu=True)
        require_runtime_growth(split, ids, two_gpu=True, incremental=True)
        end = run(binaries[2][1], root / f"{label}-resumed", 40, centres,
                  resume=split, strict=strict, two_gpu=True)
        compare_bytes(paths["candidate-two"], end)
        for part in (split, end):
            require_output_subset(paths["candidate-two"], part)
        print(f"PASS {label}: runtime growth, full-state/output parity, restart", flush=True)
    print("TWO_GPU_GROWTH_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"TWO_GPU_GROWTH_SMOKE_FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
