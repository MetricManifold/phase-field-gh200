# Cell-specific two-dimensional mobility

The 2D solver stores a physical Allen-Cahn mobility `M_i` for every cell.
The reference and default are `M0=0.5`. Its update is

```text
dphi_i/dt = (M_i/M0) * passive_rhs_i - v_i . grad(phi_i).
```

`passive_rhs_i` includes the interfacial Laplacian, double-well, area-constraint
and overlap-repulsion terms. Mobility never scales advection, the interaction
velocity, active speed, polarity or tumble stream. Default-mobility cells use
the original arithmetic expression without an extra multiply by one.
Changing `M_i` to compensate `gamma_i` also changes area and overlap relaxation;
it is not a surface-tension-only adjustment.

## Fresh starts

`--phase-field-mobility M` sets the uniform mobility, default `0.5`.
`--phase-field-mobility-map PATH` applies overrides keyed by stable global ID.
Unlisted cells keep the uniform value. For example, save this as `mobility.txt`:

```text
# global_id mobility
7 0.75
12 0.25
```

```bash
./build/cell_gh200 --N 100 --t-end 10 \
  --phase-field-mobility 0.5 --phase-field-mobility-map mobility.txt \
  --out trajectory.txt --checkpoint-dir checkpoints
```

Maps accept blank lines and whole-line `#` comments. Each other line must have
exactly one integer ID and one mobility value. Malformed rows, duplicate or
unknown IDs, empty maps, and values that are not finite and positive in
binary32 are rejected. IDs are not row numbers in an external centre table.

The explicit-Euler guard checks the resolved maximum
`max_i[(M_i/M0)*gamma_i]`; a map cannot bypass that check. Increased area and
overlap relaxation also produce timestep-margin warnings. The solver never
changes `dt` automatically; convergence remains the user's responsibility.

## Checkpoints and restart precedence

Current checkpoint format **1** retains its fixed tile and parameter layouts.
An optional `MOBI` sidecar contains one float32 mobility per cell. New writers
always emit it, including when every value is `0.5`. Current-format files
without `MOBI` load with all mobilities set to `0.5`.

After loading, restart options have exactly three interpretations:

1. An explicit uniform mobility rebases every cell; map entries then override.
2. A map alone overlays its listed IDs on stored values, leaving all other
   cells untouched. The parameter record does not store the writing run's
   uniform mobility, so it must not be used to rebase unlisted cells.
3. With neither option supplied, stored values are preserved, or the all-`0.5`
   default when the input has no `MOBI`.

Older public readers reject the unknown `MOBI` sidecar: new checkpoints are
not readable by those executables, even for an all-default run. This does not
reintroduce any older checkpoint format or change the separate 3D schema.

## Outputs and dimensional scope

Boundary metadata and velocity-component rows record each cell's actual
mobility. Passive velocity components include its `M_i/M0` multiplier;
advection components do not. Plain-text trajectory columns are unchanged and
do not record mobility: retain the input map and checkpoints with the run.

Mobility maps apply only to `cell_gh200`. Exporting relaxed 2D centres creates
positions for fresh 3D cells, not a transfer of mobility or phase-field state.
The exporter's provenance records checkpoint mobility metadata when available.
See [boundary output](boundary-output.md) and [velocity output](velocity-output.md).
