#!/usr/bin/env python3
"""Run the small 3-D example and verify one checkpoint continuation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile


CURRENT_CHECKPOINT_FORMAT = 1


def read_arguments(path: Path) -> list[str]:
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens.extend(shlex.split(line, comments=True))
    return tokens


def require_output(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty output: {path}")


def require_trajectory(path: Path) -> None:
    require_output(path)
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if (
        "schema=1 dim=3 geometry=periodic-xyz" not in first_line
        or "promoted_measure_reduction=one-cta" not in first_line
        or "promoted_measure_policy=0" not in first_line
        or "promoted_measure_auto_wave_ctas=0" not in first_line
    ):
        raise SystemExit(f"unexpected trajectory header in {path}")


def require_rejected(
    executable: Path, arguments: list[str], cwd: Path, expected: str
) -> None:
    completed = subprocess.run(
        [str(executable), *arguments],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 2 or expected not in completed.stderr:
        raise SystemExit(
            f"expected rejection containing {expected!r}; "
            f"exit={completed.returncode}, stderr={completed.stderr!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("smoke.args"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example-3d-output"),
        help="parent directory; each invocation creates a fresh child",
    )
    args = parser.parse_args()

    executable = args.executable.resolve()
    config = args.config.resolve()
    output_parent = args.output_dir.resolve()
    if not executable.is_file():
        raise SystemExit(f"executable not found: {executable}")
    if not config.is_file():
        raise SystemExit(f"configuration not found: {config}")
    output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=output_parent))

    subprocess.run(
        [str(executable), *read_arguments(config)], cwd=output, check=True
    )

    trajectory = output / "trajectory.txt"
    checkpoint = output / "checkpoints" / "checkpoint.pf3d"
    require_trajectory(trajectory)
    require_output(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    if checkpoint_bytes[:4] != b"PF3D":
        raise SystemExit(f"unexpected checkpoint magic in {checkpoint}")
    if (
        int.from_bytes(checkpoint_bytes[4:6], "little")
        != CURRENT_CHECKPOINT_FORMAT
    ):
        raise SystemExit(f"unexpected checkpoint format in {checkpoint}")

    reordered_trajectory = output / "reordered-trajectory.txt"
    reordered_lines = trajectory.read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    if len(reordered_lines) < 6:
        raise SystemExit("trajectory is too short for ordering validation")
    reordered_lines[2], reordered_lines[3] = (
        reordered_lines[3], reordered_lines[2]
    )
    reordered_trajectory.write_text("".join(reordered_lines), encoding="utf-8")
    reordered_resume = subprocess.run(
        [str(executable), "--checkpoint", str(checkpoint),
         "--t-end", "0.03", "--out", str(reordered_trajectory),
         "--trajectory-interval", "1", "--no-final-checkpoint"],
        cwd=output, text=True, capture_output=True, check=False,
    )
    if (reordered_resume.returncode != 1 or
            "canonical ID order" not in reordered_resume.stderr):
        raise SystemExit(
            "trajectory with a reordered first frame was not rejected: "
            f"exit={reordered_resume.returncode}, "
            f"stderr={reordered_resume.stderr!r}"
        )

    unsupported_checkpoint = output / "unsupported-format.pf3d"
    unsupported_bytes = bytearray(checkpoint_bytes)
    unsupported_bytes[4:6] = (CURRENT_CHECKPOINT_FORMAT - 1).to_bytes(
        2, "little"
    )
    unsupported_checkpoint.write_bytes(unsupported_bytes)
    unsupported_resume = subprocess.run(
        [str(executable), "--checkpoint", str(unsupported_checkpoint)],
        cwd=output,
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        unsupported_resume.returncode != 1
        or "unsupported PF3D checkpoint format" not in unsupported_resume.stderr
    ):
        raise SystemExit(
            "checkpoint with a non-current format was not rejected: "
            f"exit={unsupported_resume.returncode}, "
            f"stderr={unsupported_resume.stderr!r}"
        )

    before_resume_size = trajectory.stat().st_size
    subprocess.run(
        [
            str(executable),
            "--checkpoint",
            str(checkpoint),
            "--t-end",
            "0.03",
            "--out",
            str(trajectory),
            "--trajectory-interval",
            "1",
            "--strict",
            "--print-interval",
            "0",
            "--no-final-checkpoint",
        ],
        cwd=output,
        check=True,
    )
    require_trajectory(trajectory)
    if trajectory.stat().st_size <= before_resume_size:
        raise SystemExit("checkpoint continuation did not append a trajectory frame")
    payload_rows = [
        line
        for line in trajectory.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(payload_rows) != 16:
        raise SystemExit(f"expected four complete four-cell frames, got {len(payload_rows)} rows")

    # Compare a three-step run with a 1+1+1 continuation. The middle leg
    # deliberately writes no trajectory, exercising cadence preservation.
    base = read_arguments(config)
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
    split_checkpoint = split / "checkpoints" / "checkpoint.pf3d"
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
    direct_checkpoint = direct / "checkpoints" / "checkpoint.pf3d"
    if direct_checkpoint.read_bytes() != split_checkpoint.read_bytes():
        raise SystemExit(
            "three-step state differs after split checkpoint continuation"
        )

    require_rejected(
        executable,
        ["--geometry", "periodic", "--wall-padding", "8"],
        output,
        "--wall-padding requires --geometry channel",
    )
    require_rejected(
        executable,
        [
            "--geometry",
            "channel",
            "--channel-height",
            "8",
            "--wall-padding",
            "5",
            "--N",
            "1",
            "--radius",
            "4",
            "--rho",
            "0.01",
            "--lambda",
            "2",
        ],
        output,
        "smaller than the resolved-wall minimum 6",
    )

    channel_output = output / "channel-padding"
    channel_output.mkdir()
    channel_checkpoint_dir = channel_output / "checkpoints"
    channel_arguments = [
        "--geometry",
        "channel",
        "--channel-height",
        "8",
        "--wall-padding",
        "8",
        "--N",
        "1",
        "--radius",
        "4",
        "--rho",
        "0.01",
        "--lambda",
        "2",
        "--kappa",
        "10",
        "--dt",
        "0.001",
        "--t-end",
        "0.001",
        "--print-interval",
        "0",
        "--checkpoint-dir",
        str(channel_checkpoint_dir),
    ]
    channel_run = subprocess.run(
        [str(executable), *channel_arguments],
        cwd=channel_output,
        text=True,
        capture_output=True,
        check=True,
    )
    if "channel H=8, padding=8" not in channel_run.stdout:
        raise SystemExit("explicit channel padding was not realized")
    channel_checkpoint = channel_checkpoint_dir / "checkpoint.pf3d"
    require_output(channel_checkpoint)

    channel_resume = subprocess.run(
        [
            str(executable),
            "--checkpoint",
            str(channel_checkpoint),
            "--t-end",
            "0.002",
            "--print-interval",
            "0",
            "--no-final-checkpoint",
        ],
        cwd=channel_output,
        text=True,
        capture_output=True,
        check=True,
    )
    if "channel H=8, padding=8" not in channel_resume.stdout:
        raise SystemExit("checkpoint continuation did not restore channel padding")
    require_rejected(
        executable,
        ["--checkpoint", str(channel_checkpoint), "--wall-padding", "8"],
        channel_output,
        "checkpoint continuation preserves its model",
    )

    print(f"GH200_3D_SMOKE_PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
