#!/usr/bin/env python3
"""Run the public smoke configuration in a disposable directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile


def read_arguments(path: Path) -> list[str]:
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens.extend(shlex.split(line, comments=True))
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="phase-field-gh200-smoke-") as tmp:
        example = subprocess.run(
            [
                sys.executable,
                str(root / "examples" / "run_example.py"),
                "--executable",
                str(args.executable.resolve()),
                "--output-dir",
                tmp,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(example.stdout, end="")
        marker = "GH200_SMOKE_PASS output="
        output_lines = [line for line in example.stdout.splitlines()
                        if line.startswith(marker)]
        if len(output_lines) != 1:
            raise SystemExit("example runner did not report its output path")
        output = Path(output_lines[0][len(marker):])
        if output.parent != Path(tmp):
            raise SystemExit("example runner wrote outside its disposable parent")
        checkpoint = output / "checkpoints" / "checkpoint.bin"
        resumed = output / "resumed"
        subprocess.run(
            [
                str(args.executable.resolve()),
                "--checkpoint", str(checkpoint),
                "--t-end", "0.03",
                "--out", str(resumed / "trajectory.txt"),
                "--checkpoint-dir", str(resumed / "checkpoints"),
                "--checkpoint-interval", "1",
            ],
            check=True,
        )
        resumed_checkpoint = resumed / "checkpoints" / "checkpoint.bin"
        if not resumed_checkpoint.is_file() or resumed_checkpoint.stat().st_size == 0:
            raise SystemExit("current-format checkpoint did not resume")

        # Compare a three-step run with a 1+1+1 continuation. The middle leg
        # deliberately writes no trajectory, exercising cadence preservation.
        executable = args.executable.resolve()
        base = read_arguments(root / "examples" / "smoke.args")
        direct = output / "restart-direct"
        split = output / "restart-split"
        direct.mkdir()
        split.mkdir()
        common = ["--tau", "0.001", "--full-moment", "1", "--strict"]
        subprocess.run(
            [str(executable), *base, *common,
             "--t-end", "0.03",
             "--out", str(direct / "trajectory.txt"),
             "--checkpoint-dir", str(direct / "checkpoints")],
            cwd=direct, check=True,
        )
        subprocess.run(
            [str(executable), *base, *common,
             "--t-end", "0.01",
             "--out", str(split / "trajectory.txt"),
             "--checkpoint-dir", str(split / "checkpoints")],
            cwd=split, check=True,
        )
        split_checkpoint = split / "checkpoints" / "checkpoint.bin"
        subprocess.run(
            [str(executable), "--checkpoint", str(split_checkpoint),
             "--t-end", "0.02", "--strict",
             "--checkpoint-dir", str(split / "checkpoints")],
            cwd=split, check=True,
        )
        subprocess.run(
            [str(executable), "--checkpoint", str(split_checkpoint),
             "--t-end", "0.03", "--strict",
             "--out", str(split / "trajectory.txt"),
             "--checkpoint-dir", str(split / "checkpoints")],
            cwd=split, check=True,
        )
        direct_checkpoint = direct / "checkpoints" / "checkpoint.bin"
        if direct_checkpoint.read_bytes() != split_checkpoint.read_bytes():
            raise SystemExit(
                "three-step state differs after split checkpoint continuation"
            )

        unsupported = output / "unsupported-checkpoint.bin"
        shutil.copyfile(checkpoint, unsupported)
        with unsupported.open("r+b") as stream:
            stream.seek(4)
            current_format = struct.unpack("<I", stream.read(4))[0]
            stream.seek(4)
            stream.write(struct.pack("<I", current_format + 1))
        rejected = subprocess.run(
            [
                str(args.executable.resolve()),
                "--checkpoint", str(unsupported),
                "--t-end", "0.03",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if rejected.returncode == 0 or "unsupported 2D checkpoint schema" not in rejected.stdout:
            raise SystemExit(
                "unsupported checkpoint schema was not rejected cleanly:\n" +
                rejected.stdout
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
