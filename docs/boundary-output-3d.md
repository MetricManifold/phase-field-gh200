# Projected boundary output (3D)

This optional observer saves each cell's projected `phi = 0.5` contour as
crossing squares with original float32 corner samples. It does not change
fields, moments, RNG state, checkpoint contents, or the advancement schedule.
It is an observation stream, not a checkpoint, Voronoi construction, contact
graph, or T1 detector. Overlapping projections do not establish physical contact.

## Run and inspect

Add these options to an otherwise valid `cell_gh200_3d` simulation command:

```text
--boundary-out segment-001.pfb3d --boundary-interval 100
--boundary-projection basal --boundary-compression none
```

`--boundary-out` and a positive `--boundary-interval` are required together.
Projection defaults to `maximum`; compression defaults to `none`. `basal` is
available only for substrate slabs. Boundary output cannot be combined with
`--bench`. `zstd` requires a build with the optional zstd dependency; the raw
format has no compression dependency.

The observer writes the initial accepted state, each absolute step divisible
by the interval, and the final accepted state, without duplicating coincident
frames. For example, a resumed segment from step 115 through 225 at interval
100 records 115, 200, and 225. Each resumed segment needs a new output path;
existing files are never appended to or overwritten. The next segment's
initial frame can repeat the previous segment's final state. A successful close
writes an end marker, including a cooperative stop at an accepted state.
Failed runs and abrupt process termination leave an incomplete stream.

```powershell
python tools/pf3d_boundary.py segment-001.pfb3d
python tools/pf3d_boundary.py segment-001.pfb3d --json
python tests/3d/boundary_format_python_test.py
```

Inspection validates the whole stream, including checksums and its final
marker, before reporting success. Only compressed frames need the optional
Python `zstandard` package. CPU fixture tests run without it and exercise zstd
additionally when installed.

## Projection and coordinates

- `basal` samples physical grid plane `z = 0`, not the brick's first plane. A
  brick without that plane has an empty projection.
- `maximum` takes the maximum over every valid physical z sample in that
  cell's brick. Periodic volumes use all brick z planes; slab/channel
  projections exclude samples outside `0 <= z < Nz`.
- A brick of edge `B`, including an adaptive promoted brick, projects into a
  `(B + 2) x (B + 2)` sample plane with one zero sample around all four faces.
  This closes contours even when phase support reaches a brick's x/y edge.
  Stored x/y origins are the original unwrapped brick origins minus one;
  stored z origin is the original brick origin, unchanged.
- A square at local `(x, y)` has global lattice lower-left corner
  `(origin_x + x, origin_y + y)`. Its corners are lower-left, lower-right,
  upper-right, upper-left. Multiply x/y by `dx`/`dy` for physical coordinates.
  All supported geometries are x/y periodic: use `Nx` and `Ny` to wrap or select
  periodic images, preserving the unwrapped origins while stitching seam
  crossings. Do not treat a box seam as a new interface.

A square is emitted when some corners are `>= 0.5` and others are `< 0.5`.
Corner values are not clamped or quantized. Empty cells still have a metadata
record. Saddle/tie resolution, interpolation, periodic contour stitching, and
any definition of contact or T1 events belong to downstream analysis; no
implicit contact graph is stored.

## Binary contract, version 1

All records are little-endian, with no extra padding between records. Exact
C++ definitions are in `common/boundary_format_3d.h`. `u32`, `u64`, and `i64`
denote integers; `f32` and `f64` denote IEEE floating-point values.

| Record | Bytes | Fields in order |
| --- | ---: | --- |
| File header | 128 | magic[8]; version, header_bytes, cells, projection, boundary_flags, codec (6 u32); Nx, Ny, Nz (3 i64); dx, dy, dz, dt, tau, level (6 f64); interval (u64); square_bytes, cell_bytes (2 u32); reserved (u64) |
| Frame header | 48 | magic[8]; absolute step (u64); time (f64); raw_bytes, stored_bytes, checksum (3 u64) |
| Cell | 64 | id, origin_x, origin_y, origin_z (4 i64); brick_edge, plane_edge, squares, reserved (4 u32); gamma, active_speed, radius, reserved_float (4 f32) |
| Square | 20 | packed xy (u32); four corner samples (4 f32) |

File magic is `PFB3D1\0\0`; frame magic is `P3BFRM1\0`. Header version is 1,
header_bytes is 128, square_bytes is 20, cell_bytes is 64, and level is exactly
0.5. Projection values are 1 = basal and 2 = maximum; boundary flags are
7 = periodic volume, 3 = substrate slab, and 11 = hard-wall channel. Codec is
0 = none or 1 = zstd. Reserved fields are zero.

Each frame payload consists of **all cell records first**, followed by the
squares of cell 0, then cell 1, and so on in cell-record order. The packed xy
word contains x in its low 16 bits and y in its high 16 bits. The writer emits
each cell's squares in y/x order. IDs are nonnegative and unique within a frame;
use IDs, not record slots, for downstream identity. Metadata preserves each
cell's current brick edge and physical parameters.

`raw_bytes = 64 * cells + 20 * sum(cell.squares)`. For codec 0, stored bytes
equal raw bytes. Codec 1 stores one independent zstd frame per payload.
Checksum is FNV-1a-64 over the entire **decoded** payload, including metadata:
start with 14695981039346656037, XOR each byte, multiply by 1099511628211,
and reduce modulo 2^64. The checksum detects accidental damage; it is not
cryptographic authentication of an untrusted producer.

The terminal record is a frame header with magic `P3BEND1\0`, the last frame's
step/time, and zero raw_bytes, stored_bytes, and checksum, followed by physical
EOF. A structurally valid zero-frame stream uses terminal step/time 0/0.
Frame steps strictly increase; times are finite, nonnegative, and nondecreasing.

## Streaming Python API and limits

```python
import sys
sys.path.insert(0, "tools")
from pf3d_boundary import BoundaryReader, iter_frames

with BoundaryReader("segment-001.pfb3d") as reader:
    metadata = reader.header
    for frame in reader:
        for cell in frame.cells:
            for square in cell.squares:
                x = cell.origin_x + square.x
                y = cell.origin_y + square.y
                # square.phi holds the four samples for offline contour work.
    assert reader.complete

# When header metadata is not needed:
for frame in iter_frames("segment-001.pfb3d"):
    pass
```

Each frame is fully checked before it is yielded. The reader rejects malformed
metadata, counts, coordinates, duplicate IDs/squares, nonfinite samples, invalid
padding, checksum errors, truncation, reordered steps, missing end markers,
and trailing bytes. Exhaust the iterator to validate completion: an early stop
does not check later corruption or the end marker. Context management closes
path-owned files even on early exit; supplied binary streams remain caller-owned.

Defaults limit each decoded/stored payload to 64 MiB, cells to 100,000, and
total squares per frame to 1,000,000, checked before payload allocation.
Compressed content size and decompressor window are also bounded. The reader
retains one payload per current frame and lazily decodes square objects; keeping
old frames or square views retains their payloads. Duplicate-coordinate checking
and metadata require additional bounded memory. For trusted larger files, set
`ReaderLimits` explicitly or use CLI `--max-frame-mib`, `--max-cells`, and
`--max-squares`; no allocation is based on the global grid volume.
