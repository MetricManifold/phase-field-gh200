#!/usr/bin/env python3
"""Export relaxed 2D centroids for fresh PF3D bounded-z starts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import BinaryIO


CHECKPOINT_MAGIC = 0x43454C4C
CHECKPOINT_FORMAT = 1
PREFIX = struct.Struct("<IIidiiii4sI")
# CheckpointParamsRecord: box, 16 model scalars, three provenance IDs,
# six output/diagnostic fields, and the resolved trajectory cadence.
PARAMETER_RECORD = struct.Struct("<ii16dQQQ6iq")
PARAMETER_BYTES = 192
RANK_TRAILER = struct.Struct("<iii")
CELL_TILE_PITCH = 288
CELL_RECORD = struct.Struct("<iiiiIIffffddddiiiifI")
SIDECAR_HEADER = struct.Struct("<Ii")
FLOAT32 = struct.Struct("<f")
DEMOTION_DWELL = 200

assert PARAMETER_RECORD.size == PARAMETER_BYTES

MAGIC_VA_A = 0x56415F41
MAGIC_GAMA = 0x47414D41
MAGIC_RADI = 0x52414449
MAGIC_POLR = 0x504F4C52
FLOAT_SIDECARS = {
    MAGIC_VA_A, MAGIC_GAMA, MAGIC_RADI, MAGIC_POLR,
}

TRAJECTORY_FIELDS = (
    "time", "cell_id", "x", "y", "vx", "vy", "px", "py", "theta",
    "v_A_i", "L_n", "volume",
)


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointPrefix:
    magic: int
    version: int
    step: int
    current_time: float
    local_n: int
    save_interval: int
    reserved: int
    trajectory_samples: int
    bools: bytes
    parameter_size: int


@dataclass(frozen=True)
class CheckpointParameters:
    nx: int
    ny: int
    dx: float
    dy: float
    dt: float
    t_end: float
    rho_target: float
    interface_width: float
    gamma: float
    gamma_soft: float
    soft_fraction: float
    kappa: float
    radius: float
    mu: float
    v_a_uniform: float
    v_a_sigma: float
    xi: float
    tau: float
    seed: int
    polarity_seed: int
    initialization_hash: int
    print_interval: int
    full_moment_every: int
    verify_every: int
    save_interval: int
    trajectory_samples: int
    reserved: int
    trajectory_interval: int

    def manifest_values(self, local_n: int) -> dict[str, object]:
        return {
            "N": local_n,
            "Lx": self.nx,
            "Ly": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "dt": self.dt,
            "t_end": self.t_end,
            "rho_target": self.rho_target,
            "lambda": self.interface_width,
            "gamma": self.gamma,
            "gamma_soft": self.gamma_soft,
            "soft_fraction": self.soft_fraction,
            "kappa": self.kappa,
            "radius": self.radius,
            "mu": self.mu,
            "v_A_uniform": self.v_a_uniform,
            "v_A_sigma": self.v_a_sigma,
            "xi": self.xi,
            "tau": self.tau,
            "save_interval": self.save_interval,
            "trajectory_samples": self.trajectory_samples,
            "seed": self.seed,
            "polarity_seed": self.polarity_seed,
            "initialization_hash": self.initialization_hash,
            "print_interval": self.print_interval,
            "full_moment_every": self.full_moment_every,
            "verify_every": self.verify_every,
            "trajectory_interval": self.trajectory_interval,
        }


@dataclass(frozen=True)
class RankLayout:
    num_ranks: int
    rank_id: int
    global_n: int


@dataclass(frozen=True)
class CellRecordHeader:
    global_id: int
    origin_x: int
    origin_y: int
    shape_class: int
    promote_counter: int
    reserved0: int
    cx: float
    cy: float
    vx: float
    vy: float
    volume: float
    moment_x: float
    moment_y: float
    perimeter: float
    support_lo_x: int
    support_hi_x: int
    support_lo_y: int
    support_hi_y: int
    phi_max: float
    reserved1: int


@dataclass(frozen=True)
class ShapeClass:
    width: int
    height: int
    tile_x: int
    tile_y: int


@dataclass(frozen=True)
class FloatSidecar:
    values: list[float]
    payload_sha256: str


SHAPE_CLASSES = (
    ShapeClass(144, 144, 64, 64),
    ShapeClass(176, 144, 32, 64),
    ShapeClass(144, 176, 64, 32),
    ShapeClass(160, 160, 32, 32),
    ShapeClass(224, 224, 32, 32),
    ShapeClass(286, 286, 1, 1),
)


@dataclass(frozen=True)
class Centre:
    global_id: int
    x: float
    y: float


@dataclass
class SourceState:
    kind: str
    centres: list[Centre]
    time: float
    step: int | None
    side: int
    parameters: dict[str, object]
    coordinate_precision: str
    parameter_precision: str
    passive: bool
    passivity_exact: bool
    passive_evidence: str
    notes: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ExportError(f"truncated 2D checkpoint while reading {label}")
    return payload


def binary32(value: float) -> float:
    """Round like the checkpoint writer's double-to-float conversion."""
    try:
        return FLOAT32.unpack(FLOAT32.pack(value))[0]
    except OverflowError as error:
        raise ExportError("derived checkpoint centroid exceeds binary32") from error


def wrapped(value: float, side: int) -> float:
    result = math.fmod(value, float(side))
    return result + side if result < 0.0 else result


def read_checkpoint_prefix(stream: BinaryIO) -> CheckpointPrefix:
    prefix = CheckpointPrefix(*PREFIX.unpack(
        read_exact(stream, PREFIX.size, "fixed prefix")
    ))
    if prefix.magic != CHECKPOINT_MAGIC:
        raise ExportError("input is not a CELL 2D checkpoint")
    if prefix.version != CHECKPOINT_FORMAT:
        raise ExportError(
            "unsupported 2D checkpoint schema: "
            f"found {prefix.version}, expected {CHECKPOINT_FORMAT}"
        )
    if prefix.local_n <= 0:
        raise ExportError(f"invalid checkpoint cell count {prefix.local_n}")
    if (prefix.step < 0 or not math.isfinite(prefix.current_time) or
            prefix.save_interval < 0 or prefix.trajectory_samples <= 0 or
            prefix.reserved != 0 or prefix.bools != b"\0\0\0\0"):
        raise ExportError("fixed checkpoint metadata is invalid")
    if prefix.parameter_size != PARAMETER_BYTES:
        raise ExportError(
            "unsupported checkpoint parameter record size "
            f"{prefix.parameter_size}"
        )
    return prefix


def read_checkpoint_parameters(
    stream: BinaryIO, prefix: CheckpointPrefix,
) -> CheckpointParameters:
    parameters = CheckpointParameters(*PARAMETER_RECORD.unpack(
        read_exact(stream, prefix.parameter_size, "parameter record")
    ))
    if (parameters.reserved != 0 or
            parameters.save_interval != prefix.save_interval or
            parameters.trajectory_samples != prefix.trajectory_samples or
            parameters.trajectory_interval < 0):
        raise ExportError("checkpoint output metadata is inconsistent")
    if parameters.nx <= 0 or parameters.nx != parameters.ny:
        raise ExportError(
            "2D checkpoint box must be square; found "
            f"{parameters.nx} x {parameters.ny}"
        )
    if parameters.dx != 1.0 or parameters.dy != 1.0:
        raise ExportError(
            "PF3D uses unit spacing; source checkpoint has "
            f"dx={parameters.dx}, dy={parameters.dy}"
        )
    scalar_parameters = (
        parameters.dx, parameters.dy, parameters.dt, parameters.t_end,
        parameters.rho_target, parameters.interface_width, parameters.gamma,
        parameters.gamma_soft, parameters.soft_fraction, parameters.kappa,
        parameters.radius, parameters.mu, parameters.v_a_uniform,
        parameters.v_a_sigma, parameters.xi, parameters.tau,
    )
    if not all(math.isfinite(value) for value in scalar_parameters):
        raise ExportError("checkpoint contains a non-finite parameter")
    if not parameters.dt > 0.0:
        raise ExportError("checkpoint dt must be positive")
    if not parameters.tau > 0.0:
        raise ExportError("checkpoint tau must be positive")
    if not parameters.radius > 0.0:
        raise ExportError("checkpoint target radius must be positive")
    if not parameters.interface_width > 0.0:
        raise ExportError("checkpoint interface width must be positive")
    if not parameters.xi > 0.0:
        raise ExportError("checkpoint friction must be positive")
    if (parameters.t_end < 0.0 or
            not 0.0 < parameters.rho_target < 1.0 or
            not 0.0 <= parameters.soft_fraction <= 1.0 or
            parameters.gamma <= 0.0 or parameters.gamma_soft <= 0.0):
        raise ExportError("checkpoint model parameters are outside their ranges")
    if any(value < 0.0 for value in (
            parameters.kappa, parameters.mu, parameters.v_a_uniform,
            parameters.v_a_sigma)):
        raise ExportError("checkpoint model coefficients must be non-negative")
    if any(value < 0 for value in (
            parameters.print_interval, parameters.full_moment_every,
            parameters.verify_every)):
        raise ExportError("checkpoint measurement cadences must be non-negative")
    expected_time = prefix.step * parameters.dt
    time_tolerance = 1.0e-12 * max(1.0, abs(expected_time))
    if abs(prefix.current_time - expected_time) > time_tolerance:
        raise ExportError("checkpoint time does not match step multiplied by dt")
    return parameters


def read_checkpoint_layout(
    stream: BinaryIO, prefix: CheckpointPrefix, file_size: int,
) -> tuple[int, int]:
    tile_pitch = struct.unpack("<i", read_exact(stream, 4, "tile pitch"))[0]
    if tile_pitch != CELL_TILE_PITCH:
        raise ExportError(
            f"checkpoint tile pitch {tile_pitch} does not match current "
            f"schema pitch {CELL_TILE_PITCH}"
        )
    ranks = RankLayout(*RANK_TRAILER.unpack(
        read_exact(stream, RANK_TRAILER.size, "rank trailer")
    ))
    if (ranks.num_ranks != 1 or ranks.rank_id != 0 or
            ranks.global_n != prefix.local_n):
        raise ExportError(
            "one file cannot preserve a distributed checkpoint's complete "
            "global-ID/centroid table"
        )
    tile_bytes = tile_pitch * tile_pitch * 4
    minimum_bytes = (
        stream.tell() + prefix.local_n * (CELL_RECORD.size + tile_bytes)
    )
    if minimum_bytes > file_size:
        raise ExportError("checkpoint cell records exceed the file size")
    return tile_pitch, tile_bytes


def cell_centre(
    record: CellRecordHeader, index: int, domain_side: int,
    seen: set[int],
) -> Centre:
    if record.global_id in seen:
        raise ExportError(
            f"duplicate global ID {record.global_id} in checkpoint"
        )
    if record.global_id != index:
        raise ExportError(
            "current checkpoints require canonical global IDs 0..N-1"
        )
    if (not 0 <= record.shape_class < len(SHAPE_CLASSES) or
            not 0 <= record.promote_counter < DEMOTION_DWELL):
        raise ExportError(
            f"invalid adaptive state for global ID {record.global_id}"
        )
    if record.reserved0 != 0 or record.reserved1 != 0:
        raise ExportError(
            f"nonzero reserved cell metadata for global ID {record.global_id}"
        )
    shape = SHAPE_CLASSES[record.shape_class]
    valid_scalars = all(math.isfinite(value) for value in (
        record.cx, record.cy, record.vx, record.vy, record.volume,
        record.moment_x, record.moment_y, record.perimeter, record.phi_max,
    ))
    valid_bounds = (
        0 <= record.origin_x < domain_side and
        0 <= record.origin_y < domain_side and
        0 <= record.support_lo_x <= record.support_hi_x < shape.width and
        0 <= record.support_lo_y <= record.support_hi_y < shape.height
    )
    if (not valid_scalars or record.volume <= 0.0 or
            record.perimeter < 0.0 or record.phi_max < 0.0 or
            not valid_bounds):
        raise ExportError(
            f"invalid cell state metadata for global ID {record.global_id}"
        )
    expected_cx = binary32(wrapped(
        record.origin_x + shape.tile_x + record.moment_x / record.volume,
        domain_side,
    ))
    expected_cy = binary32(wrapped(
        record.origin_y + shape.tile_y + record.moment_y / record.volume,
        domain_side,
    ))
    if record.cx != expected_cx or record.cy != expected_cy:
        raise ExportError(
            f"derived centroid is inconsistent for global ID {record.global_id}"
        )
    seen.add(record.global_id)
    return Centre(record.global_id, record.cx, record.cy)


def read_checkpoint_centres(
    stream: BinaryIO, prefix: CheckpointPrefix,
    parameters: CheckpointParameters, tile_bytes: int,
) -> list[Centre]:
    centres: list[Centre] = []
    seen: set[int] = set()
    for index in range(prefix.local_n):
        record = CellRecordHeader(*CELL_RECORD.unpack(
            read_exact(stream, CELL_RECORD.size, f"cell {index} record")
        ))
        centres.append(cell_centre(record, index, parameters.nx, seen))
        stream.seek(tile_bytes, 1)
    return centres


def read_float_sidecars(
    stream: BinaryIO, file_size: int, local_n: int,
) -> dict[int, FloatSidecar]:
    sidecars: dict[int, FloatSidecar] = {}
    while stream.tell() < file_size:
        remaining = file_size - stream.tell()
        if remaining < SIDECAR_HEADER.size:
            raise ExportError("truncated checkpoint sidecar header")
        sidecar_magic, count = SIDECAR_HEADER.unpack(
            read_exact(stream, SIDECAR_HEADER.size, "sidecar header")
        )
        if count < 0:
            raise ExportError("negative checkpoint sidecar count")
        if sidecar_magic in sidecars:
            raise ExportError(
                f"duplicate checkpoint sidecar 0x{sidecar_magic:08x}"
            )
        if sidecar_magic not in FLOAT_SIDECARS:
            raise ExportError(
                f"unrecognized checkpoint sidecar 0x{sidecar_magic:08x}; "
                "refusing to ignore an unknown payload"
            )
        if count != local_n:
            raise ExportError(
                f"per-cell sidecar 0x{sidecar_magic:08x} has wrong length"
            )
        payload = read_exact(stream, count * 4, "float sidecar payload")
        values = list(struct.unpack(f"<{count}f", payload))
        if not all(math.isfinite(value) for value in values):
            raise ExportError(
                f"per-cell sidecar 0x{sidecar_magic:08x} contains non-finite data"
            )
        sidecars[sidecar_magic] = FloatSidecar(
            values=values,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
    return sidecars


def add_sidecar_metadata(
    parameters: dict[str, object], sidecars: dict[int, FloatSidecar],
) -> bool:
    missing = FLOAT_SIDECARS - sidecars.keys()
    if missing:
        formatted = ", ".join(f"0x{magic:08x}" for magic in sorted(missing))
        raise ExportError(
            f"current 2D checkpoint is missing required sidecars: {formatted}"
        )
    per_cell_v_a = sidecars[MAGIC_VA_A].values
    parameters["per_cell_v_A_min"] = min(per_cell_v_a)
    parameters["per_cell_v_A_max"] = max(per_cell_v_a)
    summaries: dict[str, object] = {}
    parameters["per_cell_sidecars"] = summaries
    for magic, name in (
        (MAGIC_POLR, "polarity"),
        (MAGIC_GAMA, "gamma"),
        (MAGIC_VA_A, "v_A"),
        (MAGIC_RADI, "radius"),
    ):
        sidecar = sidecars[magic]
        summaries[name] = {
            "count": len(sidecar.values),
            "minimum": min(sidecar.values),
            "maximum": max(sidecar.values),
            "payload_sha256": sidecar.payload_sha256,
        }
    expected_radius = as_float32(float(parameters["radius"]))
    if any(value != expected_radius for value in sidecars[MAGIC_RADI].values):
        raise ExportError(
            "source has cell-specific target radii, but the slab centre "
            "interface transfers positions only"
        )
    return all(value == 0.0 for value in per_cell_v_a)


def parse_checkpoint(path: Path) -> SourceState:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = read_checkpoint_prefix(stream)
        checkpoint_parameters = read_checkpoint_parameters(stream, prefix)
        parameters = checkpoint_parameters.manifest_values(prefix.local_n)
        tile_pitch, tile_bytes = read_checkpoint_layout(
            stream, prefix, file_size
        )
        centres = read_checkpoint_centres(
            stream, prefix, checkpoint_parameters, tile_bytes
        )
        sidecars = read_float_sidecars(stream, file_size, prefix.local_n)

    passive = add_sidecar_metadata(parameters, sidecars)
    parameters["tile_pitch"] = tile_pitch
    return SourceState(
        kind="checkpoint-current",
        centres=centres,
        time=prefix.current_time,
        step=prefix.step,
        side=checkpoint_parameters.nx,
        parameters=parameters,
        coordinate_precision=(
            "binary32 derived centroids stored in CellRecordHeader; "
            "exported with a binary32 round-trip decimal"
        ),
        parameter_precision="native checkpoint binary fields",
        passive=passive,
        passivity_exact=True,
        passive_evidence="per-cell VA_A sidecar at the checkpoint endpoint",
        notes=[],
    )


HEADER_VALUE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")


def finish_trajectory_frame(
    frame: list[tuple[float, Centre, float]], expected_n: int,
    previous_time: float | None,
    expected_ids: tuple[int, ...] | None,
) -> tuple[float, list[Centre], bool, tuple[int, ...]]:
    if len(frame) != expected_n:
        raise ExportError(
            f"trajectory frame at t={frame[0][0] if frame else 'unknown'} "
            f"has {len(frame)} rows; expected {expected_n}"
        )
    time = frame[0][0]
    if any(row[0] != time for row in frame):
        raise ExportError("trajectory frame contains mixed times")
    if previous_time is not None and not time > previous_time:
        raise ExportError("trajectory frame times are not strictly increasing")
    centres = [row[1] for row in frame]
    ids = tuple(centre.global_id for centre in centres)
    if len(set(ids)) != expected_n:
        raise ExportError(f"trajectory frame at t={time} has duplicate cell IDs")
    if expected_ids is not None and ids != expected_ids:
        raise ExportError(
            f"trajectory cell identity/order changed at t={time}; "
            "the final frame cannot be matched to the original population"
        )
    return time, centres, all(row[2] == 0.0 for row in frame), ids


def parse_trajectory(path: Path) -> SourceState:
    metadata: dict[str, str] = {}
    format_seen = False
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("#"):
                    break
                if line.startswith("# Format:"):
                    if format_seen:
                        raise ExportError("trajectory repeats its Format header")
                    declared = tuple(line.split(":", 1)[1].strip().split())
                    if declared != TRAJECTORY_FIELDS:
                        raise ExportError("unexpected 2D trajectory column schema")
                    format_seen = True
                for key, value in HEADER_VALUE.findall(line):
                    if key in metadata:
                        raise ExportError(
                            f"trajectory metadata repeats {key} on line "
                            f"{line_number}"
                        )
                    metadata[key] = value
    except UnicodeDecodeError as error:
        raise ExportError(f"trajectory is not valid UTF-8: {error}") from error
    if not format_seen:
        raise ExportError("trajectory is missing its exact Format header")
    for required in (
        "trajectory_schema", "dim", "model", "N", "Lx", "Ly", "dx",
        "dy", "dt", "rho_target", "rho_realized", "lambda", "R",
        "kappa", "mu", "xi", "tau", "v_A", "v_A_sigma",
        "gamma_normal", "gamma_soft", "soft_fraction", "soft_assignment",
        "seed", "polarity_seed", "initialization_hash", "full_moment",
        "perim_offset", "trajectory_interval",
    ):
        if required not in metadata:
            raise ExportError(f"trajectory metadata is missing {required}")
    if metadata["dim"] != "2":
        raise ExportError("input trajectory is not two-dimensional")
    if metadata["trajectory_schema"] != "1":
        raise ExportError("unsupported 2D trajectory schema")
    if metadata["model"] != "run_tumble":
        raise ExportError("trajectory does not use run-and-tumble polarity")
    if metadata["soft_assignment"] != "lowest_global_ids":
        raise ExportError("trajectory has an unsupported soft-cell assignment")
    expected_n = int(metadata["N"])
    lx = int(metadata["Lx"])
    ly = int(metadata["Ly"])
    if expected_n <= 0 or lx <= 0 or lx != ly:
        raise ExportError("trajectory N and square box dimensions are invalid")
    dx = float(metadata["dx"])
    dy = float(metadata["dy"])
    if dx != 1.0 or dy != 1.0:
        raise ExportError(
            f"PF3D uses unit spacing; source trajectory has dx={dx}, dy={dy}"
        )

    current: list[tuple[float, Centre, float]] = []
    current_time: float | None = None
    previous_time: float | None = None
    final_time = 0.0
    final_centres: list[Centre] = []
    passive = True
    expected_ids: tuple[int, ...] | None = None
    data_rows = 0
    payload_started = False
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if payload_started and (
                        line.startswith("# Format:") or HEADER_VALUE.search(line)
                    ):
                        raise ExportError(
                            "trajectory contains mutable schema/metadata after "
                            f"data began (line {line_number})"
                        )
                    continue
                payload_started = True
                fields = line.split()
                if len(fields) != len(TRAJECTORY_FIELDS):
                    raise ExportError(
                        f"trajectory row {line_number} has {len(fields)} fields; "
                        f"expected {len(TRAJECTORY_FIELDS)}"
                    )
                try:
                    time = float(fields[0])
                    global_id = int(fields[1])
                    x = float(fields[2])
                    y = float(fields[3])
                    cell_v_a = float(fields[9])
                except ValueError as error:
                    raise ExportError(
                        f"malformed trajectory value on row {line_number}: {error}"
                    ) from error
                if not all(math.isfinite(value)
                           for value in (time, x, y, cell_v_a)):
                    raise ExportError("trajectory contains a non-finite value")
                if not -(1 << 63) <= global_id < (1 << 63):
                    raise ExportError(
                        "trajectory cell ID is outside signed 64-bit range"
                    )
                if current_time is None:
                    current_time = time
                if time != current_time:
                    final_time, final_centres, frame_passive, frame_ids = (
                        finish_trajectory_frame(
                            current, expected_n, previous_time, expected_ids
                        )
                    )
                    if expected_ids is None:
                        expected_ids = frame_ids
                    previous_time = final_time
                    passive = passive and frame_passive
                    current = []
                    current_time = time
                current.append((time, Centre(global_id, x, y), cell_v_a))
                data_rows += 1
    except UnicodeDecodeError as error:
        raise ExportError(f"trajectory is not valid UTF-8: {error}") from error
    if data_rows == 0:
        raise ExportError("trajectory contains no frames")
    final_time, final_centres, frame_passive, frame_ids = finish_trajectory_frame(
        current, expected_n, previous_time, expected_ids
    )
    if expected_ids is None:
        expected_ids = frame_ids
    passive = passive and frame_passive

    dt = float(metadata["dt"])
    tau = float(metadata["tau"])
    if not dt > 0.0 or not tau > 0.0:
        raise ExportError("trajectory dt and tau must be positive")
    if int(metadata["trajectory_interval"]) <= 0:
        raise ExportError("trajectory sampling interval must be positive")
    step_value = final_time / dt
    if not math.isfinite(step_value) or not 0.0 <= step_value <= 9.0e18:
        raise ExportError("trajectory endpoint is outside the step range")
    final_step = round(step_value)
    reconstructed_time = final_step * dt
    tolerance = 1.0e-10 * max(1.0, abs(final_time))
    if abs(reconstructed_time - final_time) > tolerance:
        raise ExportError("trajectory endpoint time is not an integration step")
    parameters: dict[str, object] = {
        "N": expected_n,
        "Lx": lx,
        "Ly": ly,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "tau": tau,
        "lambda": float(metadata["lambda"]),
        "radius": float(metadata["R"]),
        "v_A_uniform": float(metadata["v_A"]),
        "trajectory_interval": int(metadata["trajectory_interval"]),
        "raw_header_values": dict(sorted(metadata.items())),
    }
    for source, destination in (
        ("gamma_normal", "gamma"),
        ("gamma_soft", "gamma_soft"),
        ("soft_fraction", "soft_fraction"),
        ("kappa", "kappa"),
        ("mu", "mu"),
        ("xi", "xi"),
    ):
        if source in metadata:
            parameters[destination] = float(metadata[source])
    for key in ("seed", "polarity_seed"):
        if key in metadata:
            parameters[key] = int(metadata[key], 10)
    if "initialization_hash" in metadata:
        parameters["initialization_hash"] = int(
            metadata["initialization_hash"], 16
        )
    return SourceState(
        kind="trajectory-text",
        centres=final_centres,
        time=final_time,
        step=final_step,
        side=lx,
        parameters=parameters,
        coordinate_precision=(
            "text x/y values printed with six digits after the decimal point; "
            "unrecorded centroid precision cannot be recovered"
        ),
        parameter_precision=(
            "current trajectory metadata uses round-trip numeric precision; "
            "payload time uses round-trip precision and cell fields use six "
            "digits after the decimal point"
        ),
        passive=passive,
        passivity_exact=False,
        passive_evidence=(
            "all stored per-cell v_A_i values equal zero at six-decimal text "
            "precision" if passive else
            "a nonzero per-cell v_A_i value is present in the text trajectory"
        ),
        notes=[
            "trajectory fallback uses the final complete frame",
            "checkpoint input is preferred because its centroids are stored as binary32",
            "the exact step can be reconstructed from current trajectory output",
        ],
    )


def as_float32(value: float) -> float:
    try:
        return FLOAT32.unpack(FLOAT32.pack(value))[0]
    except OverflowError as error:
        raise ExportError(f"coordinate is outside binary32 range: {value}") from error


def wrap_coordinate(value: float, side: int) -> tuple[float, bool]:
    wrapped = value % float(side)
    changed = wrapped != value
    result = as_float32(wrapped)
    # A value immediately below L can round to binary32 L. Periodicity makes
    # that point exactly zero, which is the canonical in-box representation.
    if result >= float(side):
        result = 0.0
        changed = True
    if result == 0.0:
        result = 0.0  # canonicalize negative zero
    if not math.isfinite(result) or not 0.0 <= result < float(side):
        raise ExportError(f"wrapped coordinate {result} is outside [0,{side})")
    return result, changed


def prepare_centres(state: SourceState) -> tuple[list[Centre], int]:
    prepared: list[Centre] = []
    ids: set[int] = set()
    wrapped_components = 0
    for centre in state.centres:
        if centre.global_id in ids:
            raise ExportError(f"duplicate global ID {centre.global_id}")
        if not -(1 << 63) <= centre.global_id < (1 << 63):
            raise ExportError(f"global ID {centre.global_id} is outside int64")
        x, changed_x = wrap_coordinate(centre.x, state.side)
        y, changed_y = wrap_coordinate(centre.y, state.side)
        wrapped_components += int(changed_x) + int(changed_y)
        ids.add(centre.global_id)
        prepared.append(Centre(centre.global_id, x, y))
    if not prepared:
        raise ExportError("source has no cells")
    expected_ids = set(range(len(prepared)))
    if ids != expected_ids:
        raise ExportError(
            "source global IDs must be exactly 0..N-1; this preserves the "
            "simulator's identity-based soft-cell selection on transfer"
        )
    prepared.sort(key=lambda centre: centre.global_id)
    return prepared, wrapped_components


def minimum_periodic_distance(centres: list[Centre], side: int) -> float | None:
    minimum = math.inf
    for i in range(1, len(centres)):
        for j in range(i):
            dx = centres[i].x - centres[j].x
            dy = centres[i].y - centres[j].y
            dx -= side * math.floor(dx / side + 0.5)
            dy -= side * math.floor(dy / side + 0.5)
            minimum = min(minimum, math.hypot(dx, dy))
    return minimum if math.isfinite(minimum) else None


def format_float32(value: float) -> str:
    return format(value, ".9g")


def table_bytes(centres: list[Centre], side: int) -> bytes:
    lines = ["global_id,x,y\n", f"# source_L={side}\n"]
    for centre in centres:
        lines.append(
            f"{centre.global_id},{format_float32(centre.x)},"
            f"{format_float32(centre.y)}\n"
        )
    return "".join(lines).encode("ascii")


def safe_slab_rho(num_cells: int, radius: float, side: int) -> tuple[float, float]:
    area = float(num_cells) * math.pi * radius * radius
    realized = area / (float(side) * float(side))
    if not 0.0 < realized < 1.0:
        raise ExportError(
            f"source box implies slab area fraction {realized}, outside (0,1)"
        )
    cli_maximum = 0.999999
    recommended = min(math.nextafter(realized, 1.0), cli_maximum)
    predicted = math.ceil(math.sqrt(area / recommended))
    if predicted != side:
        recommended = min(
            area / ((float(side) - 0.5) ** 2), cli_maximum
        )
        predicted = math.ceil(math.sqrt(area / recommended))
    if predicted != side or not 0.0 < recommended <= cli_maximum:
        raise ExportError("could not derive a robust slab rho for the source box")
    return realized, recommended


def safe_one_layer_channel_rho(
    num_cells: int, radius: float, side: int,
) -> tuple[int, float, float]:
    height = math.ceil(2.0 * radius)
    volume = float(num_cells) * 4.0 * math.pi * radius**3 / 3.0
    realized = volume / (float(side) * float(side) * float(height))
    if not 0.0 < realized < 1.0:
        raise ExportError(
            "source box implies one-layer channel volume fraction "
            f"{realized}, outside (0,1)"
        )
    cli_maximum = 0.999999
    recommended = min(math.nextafter(realized, 1.0), cli_maximum)
    predicted = math.ceil(math.sqrt(volume / (recommended * height)))
    if predicted != side:
        recommended = min(
            volume / (((float(side) - 0.5) ** 2) * height),
            cli_maximum,
        )
        predicted = math.ceil(math.sqrt(volume / (recommended * height)))
    if predicted != side or not 0.0 < recommended <= cli_maximum:
        raise ExportError(
            "could not derive a robust one-layer channel rho for the source box"
        )
    return height, realized, recommended


def write_temporary(destination: Path, payload: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=destination.name + ".tmp.",
        dir=destination.parent, delete=False,
    ) as stream:
        stream.write(payload)
        stream.flush()
        return Path(stream.name)


def paths_alias(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def replace_pair(
    csv_temp: Path, output: Path, metadata_temp: Path, metadata_path: Path,
) -> None:
    """Replace a CSV/manifest pair, restoring the previous pair on failure."""
    destinations = ((csv_temp, output), (metadata_temp, metadata_path))
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for _, destination in destinations:
            if destination.exists():
                handle, backup_name = tempfile.mkstemp(
                    prefix=destination.name + ".backup.",
                    dir=destination.parent,
                )
                os.close(handle)
                backup = Path(backup_name)
                backup.unlink()
                destination.replace(backup)
                backups.append((destination, backup))
        for temporary, destination in destinations:
            temporary.replace(destination)
            installed.append(destination)
    except BaseException:
        for destination in reversed(installed):
            if destination.exists():
                destination.unlink()
        for destination, backup in reversed(backups):
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        for _, backup in backups:
            if backup.exists():
                backup.unlink()


def export(args: argparse.Namespace) -> None:
    source = args.checkpoint if args.checkpoint is not None else args.trajectory
    source = source.resolve()
    if not source.is_file():
        raise ExportError(f"source file does not exist: {source}")
    output = args.output.resolve()
    metadata_path = (args.metadata.resolve() if args.metadata is not None
                     else Path(str(output) + ".json"))
    if paths_alias(output, metadata_path):
        raise ExportError("CSV and metadata paths must differ")
    if paths_alias(source, output) or paths_alias(source, metadata_path):
        raise ExportError("source, CSV, and metadata paths must all differ")
    if not args.force:
        existing = [path for path in (output, metadata_path) if path.exists()]
        if existing:
            raise ExportError("refusing to replace existing output: " +
                              ", ".join(str(path) for path in existing))

    source_hash = sha256_file(source)
    state = (parse_checkpoint(source) if args.checkpoint is not None
             else parse_trajectory(source))
    if sha256_file(source) != source_hash:
        raise ExportError("source changed while its centroids were being read")
    if (args.trajectory is not None and
            not args.accept_trajectory_precision):
        raise ExportError(
            "trajectory coordinates and passivity evidence are rounded text; "
            "use --accept-trajectory-precision to acknowledge the fallback"
        )
    if not state.passive and not args.allow_active_source:
        raise ExportError(
            "source is not demonstrably passive: nonzero per-cell active speed "
            "was recorded; use --allow-active-source only for a deliberate "
            "non-passive transfer"
        )
    centres, wrapped_components = prepare_centres(state)
    radius = float(state.parameters["radius"])
    realized_rho, recommended_rho = safe_slab_rho(
        len(centres), radius, state.side
    )
    channel_height, channel_realized_rho, channel_recommended_rho = (
        safe_one_layer_channel_rho(
            len(centres), radius, state.side
        )
    )
    csv_payload = table_bytes(centres, state.side)
    csv_hash = hashlib.sha256(csv_payload).hexdigest()
    tau = float(state.parameters["tau"])
    manifest = {
        "schema": "phase_field_gh200.relaxed_2d_slab_centres.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": source.name,
            "sha256": source_hash,
            "kind": state.kind,
            "time": state.time,
            "step": state.step,
            "time_over_tau": state.time / tau,
            "box": {"Lx": state.side, "Ly": state.side},
            "parameters": state.parameters,
            "coordinate_precision": state.coordinate_precision,
            "parameter_precision": state.parameter_precision,
            "passive": state.passive if state.passivity_exact else None,
            "no_activity_at_recorded_precision": state.passive,
            "passivity_exact": state.passivity_exact,
            "passive_evidence": state.passive_evidence,
            "notes": state.notes,
        },
        "centres": {
            "path": output.name,
            "sha256": csv_hash,
            "header": "global_id,x,y",
            "rows": len(centres),
            "global_ids_in_table_order": [c.global_id for c in centres],
            "coordinate_encoding": "IEEE-754 binary32 round-trip decimal",
            "wrapped_coordinate_components": wrapped_components,
            "minimum_periodic_centroid_distance": minimum_periodic_distance(
                centres, state.side
            ),
        },
        "recommended_slab_start": {
            "geometry": "slab",
            "N": len(centres),
            "radius": radius,
            "rho_argument_for_source_box": recommended_rho,
            "realized_nominal_footprint_fraction": realized_rho,
            "expected_Lx": state.side,
            "initial_centres": output.name,
            "requires_passive_slab_relaxation": True,
        },
        "recommended_one_layer_channel": {
            "geometry": "channel",
            "N": len(centres),
            "radius": radius,
            "channel_height": channel_height,
            "rho_argument_for_source_box": channel_recommended_rho,
            "realized_nominal_volume_fraction": channel_realized_rho,
            "expected_Lx": state.side,
            "initial_centres": output.name,
            "initial_z": "H/2",
            "requires_passive_3d_relaxation": True,
            "scope": (
                "Transfers global IDs and x/y centroids only; the channel "
                "seeds fresh full spheres at its midplane. This one-layer "
                "recipe must not be vertically cloned for multilayers."
            ),
        },
        "scope": (
            "Only centroid positions and global IDs are transferred. The slab "
            "solver seeds new neutral hemispheres; 2D fields, shapes, polarity, "
            "and velocities are not copied."
        ),
    }
    metadata_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    csv_temp = write_temporary(output, csv_payload)
    metadata_temp: Path | None = None
    try:
        metadata_temp = write_temporary(metadata_path, metadata_payload)
        replace_pair(csv_temp, output, metadata_temp, metadata_path)
    finally:
        if csv_temp.exists():
            csv_temp.unlink()
        if metadata_temp is not None and metadata_temp.exists():
            metadata_temp.unlink()

    print(f"centres={output}")
    print(f"metadata={metadata_path}")
    print(f"source_sha256={source_hash}")
    print(f"centres_sha256={csv_hash}")
    print(f"N={len(centres)} L={state.side} t/tau={state.time / tau:.9g}")
    print(f"slab_rho_for_L={recommended_rho:.17g}")
    print(
        f"one_layer_channel_H={channel_height} "
        f"rho_for_L={channel_recommended_rho:.17g}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export final 2D phase-field centroids to the exact global_id,x,y "
            "CSV accepted by PF3D substrate and one-layer-channel "
            "--initial-centres."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--trajectory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metadata", type=Path,
        help="provenance JSON path (default: OUTPUT.json)",
    )
    parser.add_argument(
        "--allow-active-source", action="store_true",
        help="allow nonzero recorded active speed and mark it in metadata",
    )
    parser.add_argument(
        "--accept-trajectory-precision", action="store_true",
        help=(
            "acknowledge that trajectory cell fields and zero-activity "
            "evidence are limited to their printed precision"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing CSV and metadata pair",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        export(parse_args(sys.argv[1:] if argv is None else argv))
    except (ExportError, OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
