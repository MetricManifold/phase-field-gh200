#!/usr/bin/env python3
"""Exercise the public 2D-to-slab exporter against current file contracts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile


def run_export(exporter: Path, source_flag: str, source: Path,
               output: Path) -> dict[str, object]:
    command = [sys.executable, str(exporter), source_flag, str(source),
               "--output", str(output)]
    if source_flag == "--trajectory":
        command.append("--accept-trajectory-precision")
    subprocess.run(command, check=True, capture_output=True, text=True)
    with output.with_suffix(output.suffix + ".json").open(
            encoding="utf-8") as stream:
        return json.load(stream)


def assert_centres(path: Path) -> None:
    with path.open(encoding="utf-8") as stream:
        content = [line for line in stream if not line.startswith("#")]
    rows = list(csv.DictReader(content))
    assert [int(row["global_id"]) for row in rows] == [0, 1]
    assert [float(row["x"]) for row in rows] == [10.25, 11.25]
    assert [float(row["y"]) for row in rows] == [20.5, 21.5]


def expect_trajectory_rejected(exporter: Path, source: Path,
                               output: Path, expected: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(exporter), "--trajectory", str(source),
         "--output", str(output), "--accept-trajectory-precision"],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert expected in completed.stderr


def expect_checkpoint_rejected(exporter: Path, source: Path,
                               output: Path, expected: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(exporter), "--checkpoint", str(source),
         "--output", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert expected in completed.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-writer", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pf2d-exporter-") as temporary:
        root = Path(temporary)
        checkpoint = root / "fixture.bin"
        subprocess.run([str(args.fixture_writer), str(checkpoint)], check=True)
        checkpoint_csv = root / "checkpoint.csv"
        checkpoint_meta = run_export(
            args.exporter, "--checkpoint", checkpoint, checkpoint_csv
        )
        assert_centres(checkpoint_csv)
        assert checkpoint_meta["source"]["step"] == 10
        parameters = checkpoint_meta["source"]["parameters"]
        assert parameters["seed"] == 0x12345678ABCDEF01
        assert parameters["polarity_seed"] == 0xFEDCBA9876543210
        assert parameters["soft_fraction"] == 0.5
        assert parameters["trajectory_interval"] == 5

        corrupt_centroid = root / "corrupt-centroid.bin"
        corrupt_bytes = bytearray(checkpoint.read_bytes())
        first_cell = 44 + 192 + 4 + 12
        struct.pack_into("<f", corrupt_bytes, first_cell + 24, 12.5)
        corrupt_centroid.write_bytes(corrupt_bytes)
        expect_checkpoint_rejected(
            args.exporter, corrupt_centroid, root / "corrupt-centroid.csv",
            "derived centroid is inconsistent",
        )

        trajectory_text = (
            "# Trajectory data\n"
            "# Format: time cell_id x y vx vy px py theta v_A_i L_n volume\n"
            "# trajectory_schema=1 dim=2 model=run_tumble N=2 Lx=400 Ly=400 "
            "dx=1 dy=1 dt=0.01 rho_target=0.9 rho_realized=0.1 "
            "lambda=7 R=49 kappa=10 mu=1 xi=1500 tau=10000 v_A=0 "
            "v_A_sigma=0 gamma_normal=1 gamma_soft=0.35 soft_fraction=0.5 "
            "soft_assignment=lowest_global_ids seed=1311768467750121217 "
            "polarity_seed=18364758544493064720 "
            "initialization_hash=13579bdf2468ace0 full_moment=37 "
            "perim_offset=1 trajectory_interval=5\n"
            "0.10000000000000001 0 10.250000 20.500000 0 0 1 0 0 0 1 1\n"
            "0.10000000000000001 1 11.250000 21.500000 0 0 1 0 0 0 1 1\n"
        )
        trajectory = root / "trajectory.txt"
        trajectory.write_text(trajectory_text, encoding="utf-8")
        trajectory_csv = root / "trajectory.csv"
        trajectory_meta = run_export(
            args.exporter, "--trajectory", trajectory, trajectory_csv
        )
        assert_centres(trajectory_csv)
        assert trajectory_meta["source"]["step"] == 10
        assert trajectory_meta["source"]["parameters"]["seed"] == (
            0x12345678ABCDEF01
        )
        assert trajectory_meta["source"]["parameters"][
            "trajectory_interval"
        ] == 5

        wrong_model = root / "wrong-model.txt"
        wrong_model.write_text(
            trajectory_text.replace("model=run_tumble", "model=abp"),
            encoding="utf-8",
        )
        expect_trajectory_rejected(
            args.exporter, wrong_model, root / "wrong-model.csv",
            "does not use run-and-tumble polarity",
        )

        wrong_spacing = root / "wrong-spacing.txt"
        wrong_spacing.write_text(
            trajectory_text.replace("dx=1 dy=1", "dx=0.5 dy=1"),
            encoding="utf-8",
        )
        expect_trajectory_rejected(
            args.exporter, wrong_spacing, root / "wrong-spacing.csv",
            "PF3D uses unit spacing",
        )

    print("2D exporter contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
