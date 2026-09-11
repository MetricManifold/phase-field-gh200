# Two-dimensional boundary snapshots

Boundary output is optional and read-only with respect to the numerical state.
At each requested step the GPU gathers the original float32 values at all four
corners of every grid square crossed by the `phi=0.5` contour. This retains the
field data needed for contour interpolation, including ambiguous squares,
without storing the cell interiors or far-field tails. It is not a restart
format or a measurement of contact force.

For a new output directory, an example continuation is:

```sh
mkdir boundary-run
build/cell_gh200 -c initial.bin --t-end 510000 \
  --out boundary-run/trajectory.txt --trajectory-interval 1000 \
  --checkpoint-dir boundary-run \
  --boundary-out boundary-run/boundaries.pfb --boundary-interval 1000 \
  --boundary-compression zstd
```

The end time must exceed that of the supplied checkpoint. Boundary cadence is
explicit: with `dt=0.01` and `tau=10000`, 1000 steps correspond to `0.001 tau`.
An initial frame and, if necessary, a final partial-interval frame are included.
Finer sampling must be assessed against event-count and timing convergence;
this example does not establish a universally sufficient cadence.

The writer uses three reusable host buffers and a background writing thread.
If writing falls behind, the simulation waits; it never drops frames. CUDA,
compression and file errors produce a failure. Existing boundary files are
never overwritten. Each restart segment needs a new file.

## Format

All records are little-endian, with layouts in `common/boundary_format.h`.
The 72-byte file header records the version, cell count, periodic box side,
tile pitch, integration time step, persistence time, contour level, sampling
interval and compression codec (0: raw, 1: zstd).

Each frame has a 48-byte header: `PFBFRM1`, step, time, decoded size, stored
size and an FNV-1a-64 checksum of the decoded payload. The payload contains
one 40-byte metadata record per cell, then the boundary squares grouped by
cell. Each 20-byte square record contains its local grid coordinates and four
original field values. Square ordering within a cell is not significant.
Metadata contain stable IDs, tile origins, centroids, gamma, phase-field
mobility, active speed, radius and square count.
The public two-dimensional solver has fixed phase-field mobility `M=0.5`;
this is the value recorded for every cell. Exporting boundaries does not add
cell-specific mobility or change the model.

A `PFBEND1` record records the last time and total frame count. It marks a
cleanly closed output stream, not completion of a scientific run: a clean
signal-driven stop can also close a stream. A missing terminator or invalid
payload must be reported when reading a damaged/interrupted file.

Storage grows with the number of cells, run duration and sampling frequency.
Estimate the bytes per frame from a representative pilot and check the user
quota before a production run. Ordinary checkpoint output remains separate.

## Exporting saved fields

`boundary_export CHECKPOINT_OR_DIRECTORY NEW_OUTPUT [zstd]` exports existing
checkpoints without advancing the simulation. A directory selects the tagged
`checkpoint_*.bin` snapshots in filename order; the steps must increase and
the geometry and time units must agree. The full simulation kernel is not
launched, so this tool can also run on smaller CUDA GPUs.

Contact definitions, persistence criteria and T1 detection belong in the
offline analysis, not this exporter. The format retains the contour data
without committing to a particular contact threshold.

With `PF_ENABLE_GPU_TESTS=ON`, `ctest --test-dir build -C Release
-R '^pf2d_boundary_output$' --output-on-failure` tests extraction against an
independent CPU enumeration, exact field-value preservation, periodic tile
origins, output-on/off state identity, compression when available, and failure
handling. This test only launches the small export kernels.
