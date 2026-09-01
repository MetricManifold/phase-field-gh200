#!/usr/bin/env python3
"""Run a whitespace-delimited simulator argument file in an output directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile


def read_arguments(path: Path) -> list[str]:
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens.extend(shlex.split(line, comments=True))
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("smoke.args"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("example-output"),
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

    command = [str(executable), *read_arguments(config)]
    subprocess.run(command, cwd=output, check=True)

    required = (
        output / "trajectory.txt",
        output / "checkpoints" / "checkpoint.bin",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("missing or empty outputs: " + ", ".join(missing))
    print(f"GH200_SMOKE_PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
