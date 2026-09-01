#!/usr/bin/env python3
"""Render a current-format PF3D substrate checkpoint.

The renderer checks the mandatory complete-file CRC, layout, and payload
length before extracting phi=0.5 isosurfaces. The simulator remains the
authority for model and restart compatibility.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastcrc import crc64
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import find_contours, marching_cubes

from pf3d_checkpoint_header import (
    CELL_RECORD_BYTES,
    FILE_HEADER,
    MAX_BRICK_EDGE,
    MAX_INT32,
    PARAMS_BYTES,
    STORAGE_EDGE_ALIGNMENT,
    canonical_crc_header,
    parse_file_header,
)


SUBSTRATE_SLAB_FLAGS = 3


def valid_reduction_contract(policy: int, wave: int) -> bool:
    if policy == (1 << 64) - 1:
        return 0 < wave <= MAX_INT32
    return 0 <= policy <= 64 and wave == 0


@dataclass(frozen=True)
class Surface:
    global_id: int
    vertices: np.ndarray
    faces: np.ndarray
    centre: np.ndarray


@dataclass(frozen=True)
class Footprint:
    global_id: int
    polygons: list[np.ndarray]


@dataclass(frozen=True)
class Checkpoint:
    domain: tuple[int, int, int]
    radius: float
    area_fraction: float
    basal_coverage: float
    step: int
    time: float
    surfaces: list[Surface]
    footprints: list[Footprint]
    crc_verified: bool


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def read_checkpoint(
    path: Path, marching_step: int, *, verify_crc: bool = True
) -> Checkpoint:
    surfaces: list[Surface] = []
    footprints: list[Footprint] = []
    with path.open("rb") as stream:
        raw_header = stream.read(FILE_HEADER.size)
        header = parse_file_header(raw_header)
        step = header.step
        time = header.time
        num_cells = header.num_cells
        brick_edge = header.brick_edge
        stored_crc64 = header.stored_crc64

        computed_crc64 = (
            crc64.ecma_182(canonical_crc_header(raw_header))
            if verify_crc
            else 0
        )

        params = stream.read(PARAMS_BYTES)
        if len(params) != PARAMS_BYTES:
            raise ValueError("truncated PF3D parameter record")
        if verify_crc:
            computed_crc64 = crc64.ecma_182(params, computed_crc64)
        nx, ny, nz = struct.unpack_from("<qqq", params, 0)
        params_brick_edge = struct.unpack_from("<I", params, 48)[0]
        boundary_flags = struct.unpack_from("<I", params, 52)[0]
        radius = struct.unpack_from("<d", params, 112)[0]
        area_fraction = struct.unpack_from("<d", params, 160)[0]
        if boundary_flags != SUBSTRATE_SLAB_FLAGS:
            raise ValueError("checkpoint is not a periodic-XY substrate slab")
        if (
            nx <= 0
            or nx > MAX_INT32
            or nx != ny
            or nz <= brick_edge
            or nz > MAX_INT32
            or params_brick_edge != brick_edge
        ):
            raise ValueError("invalid substrate-slab domain")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("invalid target radius")
        if not math.isfinite(area_fraction) or not 0.0 < area_fraction < 1.0:
            raise ValueError("invalid slab area fraction")
        reduction_policy, reduction_wave = struct.unpack_from("<QQ", params, 240)
        if not valid_reduction_contract(reduction_policy, reduction_wave):
            raise ValueError("invalid promoted-measurement reduction contract")
        if any(params[256:288]):
            raise ValueError(
                "substrate-slab checkpoint has nonzero channel-wall parameters"
            )

        global_ids: set[int] = set()
        basal_occupancy = np.zeros((ny, nx), dtype=bool)
        for _ in range(num_cells):
            record = stream.read(CELL_RECORD_BYTES)
            if len(record) != CELL_RECORD_BYTES:
                raise ValueError("truncated PF3D cell record")
            if verify_crc:
                computed_crc64 = crc64.ecma_182(record, computed_crc64)
            global_id = struct.unpack_from("<q", record, 0)[0]
            origin_integer = struct.unpack_from("<qqq", record, 8)
            origin = np.asarray(origin_integer, dtype=float)
            volume, moment_x, moment_y, moment_z = struct.unpack_from(
                "<dddd", record, 72
            )
            edge = struct.unpack_from("<I", record, 184)[0]
            if global_id in global_ids:
                raise ValueError(f"duplicate cell ID {global_id}")
            global_ids.add(global_id)
            if (
                edge < brick_edge
                or edge > MAX_BRICK_EDGE
                or edge >= min(nx, ny, nz)
                or edge % STORAGE_EDGE_ALIGNMENT != 0
            ):
                raise ValueError(f"invalid storage edge {edge} for cell {global_id}")
            if struct.unpack_from("<I", record, 188)[0] != 0 or any(record[192:256]):
                raise ValueError(f"nonzero reserved data for cell {global_id}")
            if not math.isfinite(volume) or volume <= 0:
                raise ValueError(f"invalid volume for cell {global_id}")
            moments = np.asarray([moment_x, moment_y, moment_z], dtype=float)
            if not np.all(np.isfinite(moments)):
                raise ValueError(f"non-finite moments for cell {global_id}")
            centre = origin + moments / volume
            if not np.all(np.isfinite(centre)):
                raise ValueError(f"non-finite centre for cell {global_id}")

            count = edge**3
            raw_phi = stream.read(4 * count)
            if len(raw_phi) != 4 * count:
                raise ValueError(f"truncated phase field for cell {global_id}")
            if verify_crc:
                computed_crc64 = crc64.ecma_182(raw_phi, computed_crc64)
            phi = (
                np.frombuffer(raw_phi, dtype="<f4")
                .reshape((edge, edge, edge))
                .copy()
            )
            del raw_phi
            phi_min = float(np.min(phi))
            phi_max = float(np.max(phi))
            if not math.isfinite(phi_min) or not math.isfinite(phi_max):
                raise ValueError(f"non-finite phase field for cell {global_id}")
            if not phi_min <= 0.5 <= phi_max:
                raise ValueError(f"cell {global_id} has no phi=0.5 isosurface")

            basal_z = -origin_integer[2]
            footprint_polygons = []
            if 0 <= basal_z < edge:
                basal_plane = phi[basal_z]
                basal_y, basal_x = np.nonzero(basal_plane >= 0.5)
                basal_occupancy[
                    (origin_integer[1] + basal_y) % ny,
                    (origin_integer[0] + basal_x) % nx,
                ] = True
                for contour in find_contours(basal_plane, level=0.5):
                    if len(contour) >= 3:
                        footprint_polygons.append(
                            contour[:, [1, 0]] + origin[:2]
                        )
            footprints.append(Footprint(global_id, footprint_polygons))

            vertices_zyx, faces, _, _ = marching_cubes(
                phi,
                level=0.5,
                step_size=marching_step,
                allow_degenerate=False,
            )
            vertices = vertices_zyx[:, [2, 1, 0]] + origin
            vertices[:, 2] += 0.5
            centre[2] += 0.5
            surfaces.append(Surface(global_id, vertices, faces, centre))

        if stream.read(1):
            raise ValueError("unexpected data after the final phase field")
        if verify_crc and computed_crc64 != stored_crc64:
            raise ValueError(
                "PF3D CRC-64 mismatch: "
                f"stored {stored_crc64:016x}, computed {computed_crc64:016x}"
            )

    return Checkpoint(
        (int(nx), int(ny), int(nz)),
        radius,
        area_fraction,
        float(np.mean(basal_occupancy)),
        int(step),
        float(time),
        surfaces,
        footprints,
        verify_crc,
    )


def add_substrate(axis, x_limits, y_limits, *, alpha=0.055) -> None:
    x0, x1 = x_limits
    y0, y1 = y_limits
    plane = [[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]]
    axis.add_collection3d(
        Poly3DCollection(
            plane,
            facecolor="#b8bec4",
            edgecolor="#59636c",
            linewidth=0.8,
            alpha=alpha,
        )
    )


def add_surface(axis, surface: Surface, color, alpha=0.96) -> None:
    mesh = Poly3DCollection(
        surface.vertices[surface.faces], linewidth=0.0, alpha=alpha
    )
    mesh.set_facecolor(color)
    axis.add_collection3d(mesh)


def periodic_images(surface: Surface, nx: int, ny: int):
    """Yield the wrapped primary mesh and any edge-crossing images."""
    base_x = -math.floor(surface.centre[0] / nx) * nx
    base_y = -math.floor(surface.centre[1] / ny) * ny
    for neighbor_x in (-nx, 0, nx):
        for neighbor_y in (-ny, 0, ny):
            shift = np.asarray(
                [base_x + neighbor_x, base_y + neighbor_y, 0.0]
            )
            shifted = surface.vertices + shift
            if (
                shifted[:, 0].max() < 0
                or shifted[:, 0].min() > nx
                or shifted[:, 1].max() < 0
                or shifted[:, 1].min() > ny
            ):
                continue
            yield Surface(
                surface.global_id,
                shifted,
                surface.faces,
                surface.centre + shift,
            )


def primary_box_cutaway(surface: Surface, nx: int, ny: int) -> Surface | None:
    """Clip a periodic image to the primary box, leaving boundary cuts open."""
    vertices = surface.vertices
    inside = (
        (vertices[:, 0] >= 0.0)
        & (vertices[:, 0] <= nx)
        & (vertices[:, 1] >= 0.0)
        & (vertices[:, 1] <= ny)
    )
    faces = surface.faces[np.all(inside[surface.faces], axis=1)]
    if faces.size == 0:
        return None
    return Surface(surface.global_id, vertices, faces, surface.centre)


def plane_cut_polygon(
    surface: Surface, axis: int, value: float, nx: int, ny: int
) -> np.ndarray | None:
    """Return an ordered mesh/box intersection polygon for a cutaway cap."""
    triangles = surface.vertices[surface.faces]
    distance = triangles[:, :, axis] - value
    points = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        first_distance = distance[:, first]
        second_distance = distance[:, second]
        crossing = (first_distance * second_distance <= 0.0) & (
            first_distance != second_distance
        )
        if not np.any(crossing):
            continue
        fraction = first_distance[crossing] / (
            first_distance[crossing] - second_distance[crossing]
        )
        start = triangles[crossing, first]
        end = triangles[crossing, second]
        points.append(start + fraction[:, None] * (end - start))
    if not points:
        return None

    intersection = np.concatenate(points)
    tolerance = 1.0e-5
    in_box = (
        (intersection[:, 0] >= -tolerance)
        & (intersection[:, 0] <= nx + tolerance)
        & (intersection[:, 1] >= -tolerance)
        & (intersection[:, 1] <= ny + tolerance)
    )
    intersection = intersection[in_box]
    if len(intersection) < 3:
        return None
    intersection[:, axis] = value
    intersection = np.unique(np.round(intersection, decimals=5), axis=0)
    if len(intersection) < 3:
        return None

    projection_axes = (1, 2) if axis == 0 else (0, 2)
    projected = intersection[:, projection_axes]
    centre = projected.mean(axis=0)
    angle = np.arctan2(projected[:, 1] - centre[1], projected[:, 0] - centre[0])
    return intersection[np.argsort(angle)]


def add_boundary_caps(axis, surface: Surface, color, nx: int, ny: int) -> None:
    dark = tuple(np.asarray(color[:3]) * 0.48)
    polygons = []
    for coordinate, limit in ((0, nx), (1, ny)):
        minimum = float(surface.vertices[:, coordinate].min())
        maximum = float(surface.vertices[:, coordinate].max())
        for plane in (0.0, float(limit)):
            if minimum < plane < maximum:
                polygon = plane_cut_polygon(surface, coordinate, plane, nx, ny)
                if polygon is not None:
                    polygons.append(polygon)
    if polygons:
        axis.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolor=dark,
                edgecolor="#20252a",
                linewidth=0.65,
                alpha=1.0,
            )
        )


def clip_polygon_to_box(points: np.ndarray, nx: int, ny: int) -> np.ndarray:
    polygon = points
    for coordinate, boundary, keep_lower in (
        (0, 0.0, False),
        (0, float(nx), True),
        (1, 0.0, False),
        (1, float(ny), True),
    ):
        if len(polygon) == 0:
            break
        output = []
        previous = polygon[-1]
        previous_inside = (
            previous[coordinate] <= boundary
            if keep_lower
            else previous[coordinate] >= boundary
        )
        for current in polygon:
            current_inside = (
                current[coordinate] <= boundary
                if keep_lower
                else current[coordinate] >= boundary
            )
            if current_inside != previous_inside:
                fraction = (boundary - previous[coordinate]) / (
                    current[coordinate] - previous[coordinate]
                )
                output.append(previous + fraction * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
        polygon = np.asarray(output)
    return polygon


def periodic_footprint_polygons(points: np.ndarray, nx: int, ny: int):
    centre = points.mean(axis=0)
    base_x = -math.floor(centre[0] / nx) * nx
    base_y = -math.floor(centre[1] / ny) * ny
    for neighbor_x in (-nx, 0, nx):
        for neighbor_y in (-ny, 0, ny):
            shifted = points + np.asarray(
                [base_x + neighbor_x, base_y + neighbor_y]
            )
            if (
                shifted[:, 0].max() < 0
                or shifted[:, 0].min() > nx
                or shifted[:, 1].max() < 0
                or shifted[:, 1].min() > ny
            ):
                continue
            clipped = clip_polygon_to_box(shifted, nx, ny)
            if len(clipped) >= 3:
                yield clipped


def add_basal_footprints(
    axis, footprints: list[Footprint], colors, nx: int, ny: int
) -> None:
    polygons = []
    facecolors = []
    for index, footprint in enumerate(footprints):
        for unwrapped in footprint.polygons:
            for polygon in periodic_footprint_polygons(unwrapped, nx, ny):
                polygons.append(
                    np.column_stack(
                        (polygon, np.full(len(polygon), 0.2, dtype=float))
                    )
                )
                facecolors.append((*colors[index, :3], 0.48))
    if polygons:
        axis.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolors=facecolors,
                edgecolor="#20252a",
                linewidth=0.9,
            )
        )


def style_axis(axis) -> None:
    axis.set_proj_type("ortho")
    axis.grid(False)
    axis.xaxis.pane.set_alpha(0.0)
    axis.yaxis.pane.set_alpha(0.0)
    axis.zaxis.pane.set_alpha(0.0)
    axis.tick_params(axis="both", which="major", labelsize=7, pad=0.5)
    axis.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    axis.zaxis.set_major_locator(MaxNLocator(3, integer=True))


def label_panel(axis, label: str) -> None:
    axis.text2D(
        0.01,
        0.985,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )


def render(checkpoint: Checkpoint, output: Path, dpi: int) -> None:
    nx, ny, _ = checkpoint.domain
    surfaces = checkpoint.surfaces
    colors = plt.cm.turbo(np.linspace(0.04, 0.96, max(3, len(surfaces))))
    figure = plt.figure(figsize=(7.2, 2.55))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.18, 1.0, 1.08),
        left=0.015,
        right=0.995,
        bottom=0.04,
        top=0.94,
        wspace=0.02,
    )
    full = figure.add_subplot(grid[0, 0], projection="3d")
    top = figure.add_subplot(grid[0, 1], projection="3d")
    close = figure.add_subplot(grid[0, 2], projection="3d")
    for axis in (full, top, close):
        style_axis(axis)

    add_substrate(full, (0, nx), (0, ny))
    add_substrate(top, (0, nx), (0, ny), alpha=0.025)
    add_basal_footprints(full, checkpoint.footprints, colors, nx, ny)
    maximum_height = 0.0
    for index, surface in enumerate(surfaces):
        maximum_height = max(maximum_height, float(surface.vertices[:, 2].max()))
        for image in periodic_images(surface, nx, ny):
            cutaway = primary_box_cutaway(image, nx, ny)
            if cutaway is not None:
                add_surface(full, cutaway, colors[index])
                add_surface(top, cutaway, colors[index], alpha=1.0)
                add_boundary_caps(full, image, colors[index], nx, ny)
    z_max = max(1.0, 1.15 * maximum_height)
    full.set(
        xlim=(0, nx),
        ylim=(0, ny),
        zlim=(0, z_max),
        xlabel="x",
        ylabel="y",
        zlabel=r"$z$",
    )
    full.set_xlabel("x", fontsize=8, labelpad=-1)
    full.set_ylabel("y", fontsize=8, labelpad=-1)
    full.set_zlabel(r"$z$", fontsize=8, labelpad=0)
    full.set_box_aspect((nx, ny, z_max))
    full.view_init(elev=27, azim=-58)
    full.set_title("Layer and basal footprints", fontsize=8.5, pad=1.5)
    label_panel(full, "a")

    top.set(
        xlim=(0, nx),
        ylim=(0, ny),
        zlim=(0, z_max),
        xlabel="x",
        ylabel="y",
        zlabel="",
    )
    # Matplotlib's projected 3-D axis labels can fall outside a tightly saved
    # overhead panel. Fixed 2-D labels remain legible at journal column widths.
    top.set_xlabel("")
    top.set_ylabel("")
    top.text2D(
        0.5, -0.055, "x", transform=top.transAxes,
        ha="center", va="top", fontsize=8,
    )
    top.text2D(
        -0.065, 0.5, "y", transform=top.transAxes,
        ha="right", va="center", rotation=90, fontsize=8,
    )
    top.set_box_aspect((nx, ny, max(1.0, 0.08 * nx)))
    top.set_zticks([])
    top.view_init(elev=90, azim=-90)
    top.set_title("Overhead view", fontsize=8.5, pad=1.5)
    label_panel(top, "b")

    anchor = surfaces[0].centre.copy()
    anchor[:2] -= np.floor(anchor[:2] / np.asarray([nx, ny])) * np.asarray(
        [nx, ny]
    )
    distances = []
    for index, surface in enumerate(surfaces):
        delta = surface.centre[:2] - anchor[:2]
        delta -= np.rint(delta / np.asarray([nx, ny])) * np.asarray([nx, ny])
        distances.append((float(delta @ delta), index, delta))
    selected = sorted(distances)[: min(3, len(surfaces))]
    close_surfaces = []
    for _, index, delta in selected:
        surface = surfaces[index]
        shift_xy = anchor[:2] + delta - surface.centre[:2]
        shift = np.asarray([shift_xy[0], shift_xy[1], 0.0])
        shifted_surface = Surface(
            surface.global_id,
            surface.vertices + shift,
            surface.faces,
            surface.centre + shift,
        )
        close_surfaces.append((index, shifted_surface))

    close_vertices = np.concatenate(
        [surface.vertices for _, surface in close_surfaces]
    )
    close_centre = 0.5 * (
        close_vertices[:, :2].min(axis=0) + close_vertices[:, :2].max(axis=0)
    )
    close_span = close_vertices[:, :2].max(axis=0) - close_vertices[:, :2].min(
        axis=0
    )
    half_width = max(checkpoint.radius, 0.56 * float(close_span.max()))
    x_limits = (close_centre[0] - half_width, close_centre[0] + half_width)
    y_limits = (close_centre[1] - half_width, close_centre[1] + half_width)
    add_substrate(close, x_limits, y_limits, alpha=0.035)
    for index, shifted_surface in close_surfaces:
        add_surface(close, shifted_surface, colors[index], alpha=1.0)
    close.set(
        xlim=x_limits,
        ylim=y_limits,
        zlim=(0, z_max),
        xlabel="x",
        ylabel="y",
        zlabel=r"$z$",
    )
    close.set_xlabel("x", fontsize=8, labelpad=-1)
    close.set_ylabel("y", fontsize=8, labelpad=-1)
    close.set_zlabel(r"$z$", fontsize=8, labelpad=0)
    close.set_box_aspect((2 * half_width, 2 * half_width, z_max))
    close.view_init(elev=24, azim=-62)
    close.set_title("Three-cell detail", fontsize=8.5, pad=1.5)
    label_panel(close, "c")

    if not checkpoint.crc_verified:
        figure.text(
            0.995,
            0.005,
            "CRC not verified",
            ha="right",
            va="bottom",
            color="#8b0000",
            fontsize=6.5,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render phi=0.5 surfaces from a PF3D substrate checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--marching-step",
        type=_positive_int,
        default=2,
        help="marching-cubes stride; 1 is highest fidelity and 2 is the default",
    )
    parser.add_argument("--dpi", type=_positive_int, default=210)
    parser.add_argument(
        "--skip-crc",
        action="store_true",
        help="skip the mandatory checkpoint CRC check (display recovery only)",
    )
    args = parser.parse_args()
    if args.skip_crc:
        print(
            "WARNING: checkpoint CRC was not verified; output is for display "
            "recovery only.",
            file=sys.stderr,
        )
    checkpoint = read_checkpoint(
        args.checkpoint, args.marching_step, verify_crc=not args.skip_crc
    )
    render(checkpoint, args.output, args.dpi)


if __name__ == "__main__":
    main()
