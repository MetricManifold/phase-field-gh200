# Phase-field cell simulator for NVIDIA GH200

This repository contains single-GPU CUDA implementations of the active
phase-field cell model. `cell_gh200` advances the two-dimensional formulation
on a periodic square lattice. The separate `cell_gh200_3d` executable extends
the same coefficient convention and run-and-tumble process to a periodic
three-dimensional volume by default, with substrate-slab and two-hard-wall
channel geometries selected explicitly.

The implementation is specialized for one independent simulation replica on
one NVIDIA GH200. It does not use MPI, NCCL, domain decomposition, or
CUDA-aware MPI.

## Executables and geometry modes

The repository builds two solver executables, not one executable per geometry:

| executable | numerical core | supported geometry |
| --- | --- | --- |
| `cell_gh200` | `pf_core` | periodic two-dimensional monolayer |
| `cell_gh200_3d` | `pf3d_core` | periodic XYZ volume (default), substrate slab, or hard-wall channel |

All 3D geometries therefore share the same simulation
state, adaptive storage, checkpoint/restart implementation, measurement code,
and general update kernels. Geometry is an explicit runtime and checkpoint
property. The specialized source
`src/pf3d/kernels_periodic_slab_walk.cu` is a rolling-plane optimization for
fully periodic XYZ only; bounded-z geometries use the boundary-aware tiled
kernel family in `src/pf3d/kernels.cu`.

The two numerical cores also use the same dimension-independent model-
coefficient and Philox primitives from `common/`. Their stencils, field
storage, random-counter domains, and update kernels remain dimension-specific.

## Model and numerical scheme

For cell `n`, the phase field is advanced as

```text
dphi_n/dt = gamma_n lap(phi_n)
          - (30 gamma_n/lambda^2) phi_n(1-phi_n)(1-2phi_n)
          + (2 mu/A0)(A0-V_n) phi_n
          - (60 kappa/lambda^2) phi_n sum_(m!=n) phi_m^2
          - v_n . grad(phi_n),

v_n = v_A p_n + (60 kappa/(xi lambda^2))
      integral(phi_n grad(phi_n) sum_(m!=n) phi_m^2 dA).
```

Here `V_n = integral(phi_n^2 dA)` and `A0 = pi R^2`. The code uses `M=1/2`,
`dx=dy=1`, a nine-point isotropic Laplacian, centered gradients, periodic
boundaries, and binary32 phase fields. Coefficients are defined once in
[`include/params.cuh`](include/params.cuh). In particular, the interaction and
motility coefficients satisfy `interaction/motility = xi` by construction.

All lengths are expressed in lattice units (`dx=dy=1`, and `dz=1` in 3D) and
time in the solver's nondimensional integration unit. Parameters are therefore
model-unit inputs; this repository does not imply a mapping to physical units.

The normal update assigns one CUDA thread block per cell and keeps each active
rectangular field in shared memory. An ordered second kernel handles only cells
that outgrow those classes, reading phi and the interaction field from global
memory without changing the equations or run-and-tumble stream. Every cell has
a fixed `288 x 288` tile; the fallback uses its `286 x 286` interior at `(1,1)`.
The interaction field is accumulated in Q5.27 fixed point.

Model reference: B. Palmieri, Y. Bresler, D. Wirtz, and M. Grant, “Multiple
scale model for cell migration in monolayers: elastic mismatch between cells
enhances motility,” *Scientific Reports* 5, 11745 (2015),
[doi:10.1038/srep11745](https://doi.org/10.1038/srep11745).

The 3D solver defaults to periodic XYZ, `V0=4*pi*R^3/3`, and directions uniform
on the unit sphere. `--geometry slab` instead selects a fixed neutral substrate,
periodic x/y, bounded z, hemispherical `V0=2*pi*R^3/3`, and planar translation.
`--geometry channel` places full-volume 3D cells between two resolved static
steric walls, with periodic x/y and unrestricted 3D polarity. The numerical
z boundaries are buried in solid padding beyond the physical wall surfaces.
It offers throughput, balanced, and compact storage modes and an independent
`PF3D` checkpoint format. Its equations, initialization, memory formulas, CLI,
validation, and limitations are documented in
[`docs/three-dimensional-solver.md`](docs/three-dimensional-solver.md).

The slab is a nonadhesive 90-degree contact model. Related substrate-resolved
phase-field treatments include [Monfared *et al.*, *eLife* 12:e82435
(2023)](https://doi.org/10.7554/eLife.82435), [Winkler, Aranson, and Ziebert,
*Communications Physics* 2, 82
(2019)](https://doi.org/10.1038/s42005-019-0185-x), and the 3D-to-2D derivation
of [Chiang *et al.*, *Physical Review E* 110, 044403
(2024)](https://doi.org/10.1103/PhysRevE.110.044403). Slab height must be shown
not to affect observables before the geometry is used for scientific results.

## Requirements

- A CUDA 12.x toolkit and NVIDIA GPU supporting the requested CUDA architecture;
- CMake 3.24 or newer and a C++17 compiler;
- Python 3.10 or newer for the tests and command-line tools. The optional
  pinned visualization environment requires Python 3.11 or newer.

The optimized default targets CUDA compute capability 9.0 (`sm_90`).

## Build

The optimized production build is the default:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --config Release --parallel 8 2>&1 | tee build.log
```

This builds both `cell_gh200` and `cell_gh200_3d`; pass `-DPF_BUILD_3D=OFF` to
omit the latter.

The commands in this README assume a source checkout. An optional
`cmake --install build --prefix <prefix>` places executables in `<prefix>/bin`
and a copy of this documentation, the examples, and visualization tools in
`<prefix>/share/doc/phase_field_gh200`.

Do not add `-use_fast_math`. The build explicitly preserves its arithmetic and
FMA-contraction policy and prints the ptxas register/spill report. Check
`build.log` for spill stores before using a new compiler build for production.
The runtime may report a 32-byte local ABI frame when ptxas reports zero spill
loads and stores; the frame itself is not a register spill.

For `cell_gh200` (2D), the production build excludes the high-frequency
`support_clip` instrumentation. Fail-closed checks for overflow, non-finite
fields, invalid volume, and unsupported geometry remain in the update; their
sticky atomics execute only when an invalid condition is detected, and the host
polls them every 10,000 steps. Fallback use and any fallback margin or boundary
contact produce an always-on, nonfatal geometry warning. A diagnostic build
adds the higher-frequency `support_clip` counter for every 2D shape class and
brick-edge instrumentation for 3D fields:

```bash
cmake -S . -B build-diagnostic \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPF_DIAGNOSTIC_ALARMS=ON
cmake --build build-diagnostic --config Release --parallel 8
```

For `cell_gh200_3d`, fatal overflow, non-finite-field, invalid-volume, and
geometry flags are always compiled in and polled at most 16 steps after they
are raised. A cell that approaches its brick margin is moved exactly into a
larger per-cell allocation before another physical step is accepted. If no
larger allocation fits in the configured HBM budget or every domain extent,
the run stops without clipping the field. Brick-edge contact is also counted by
the diagnostic build and by the opt-in `--strict` verifier. The strict verifier
checks volume and aggregate-field invariants at `--verify-every`; the 3D smoke
configuration enables it.

## Minimal run

[`examples/smoke.args`](examples/smoke.args) is a four-cell, two-step
configuration that exercises initialization, the coefficient self-test,
trajectory output, strict checks, and checkpoint writing:

```bash
python3 examples/run_example.py \
  --executable ./build/cell_gh200 \
  --output-dir ./example-output
```

Run `./build/cell_gh200 --help` for the complete command-line interface. A
typical research invocation is:

For these parameters the domain rule
`L=ceil(sqrt(N*pi*R^2/rho))` gives `L=1554`. The centre-table `--side` must
equal the side derived by the solver from the same `N`, `R`, and `rho`.

```bash
./build/palmieri_centres \
  --N 288 --side 1554 --radius 49 --seed 1234 \
  --out initial_centres.csv

./build/cell_gh200 \
  --N 288 --radius 49 --rho 0.90 --dt 0.01 --t-end 100 \
  --lambda 7 --kappa 10 --mu 1 --xi 1500 --tau 10000 --v-A 0.01 \
  --gamma 1.0 --seed 1234 --polarity-seed 1234 \
  --initial-centres initial_centres.csv \
  --out trajectory.txt --checkpoint-dir checkpoints \
  --checkpoint-interval 10000
```

The corresponding 3D smoke and restart example is:

```bash
python3 examples/3d/run_example.py \
  --executable ./build/cell_gh200_3d \
  --output-dir ./example-3d-output
```

The substrate slab is explicitly selected and can be restarted normally:

```bash
./build/cell_gh200_3d \
  --geometry slab --slab-height 288 \
  --N 4 --radius 49 --rho 0.90 --t-end 1 \
  --out trajectory-slab.txt --checkpoint-dir checkpoints-slab

./build/cell_gh200_3d \
  -c checkpoints-slab/checkpoint.pf3d --t-end 2 \
  --out trajectory-slab.txt --checkpoint-dir checkpoints-slab
```

Omitting `--slab-height` chooses twice the automatically derived brick edge.
The explicit height must exceed the selected base brick.

A two-hard-wall channel is selected separately:

```bash
./build/cell_gh200_3d \
  --geometry channel --channel-height 196 \
  --N 32 --radius 49 --rho 0.60 --t-end 1 \
  --brick-edge 224 \
  --out trajectory-channel.txt --checkpoint-dir checkpoints-channel
```

For a channel, `--rho` is the target used to derive integer `Lx`; the realized
three-dimensional volume fraction is `N*(4*pi*R^3/3)/(Lx^2*H)`, where `H` is
the accessible separation between the two wall mid-surfaces. The solver adds
`ceil(3*wall_width)` solid voxels beyond
each wall by default; `--wall-padding` may select a larger value for padding-
independence checks, but never a smaller one. Padding is not included in `rho`
and is stored in checkpoints. `--wall-kappa` defaults to `kappa`,
`--wall-width` defaults to `lambda`, and omitting
`--channel-height` selects the minimum fresh-start height `ceil(2R)`. Wall
separation, rather than a prescribed layer label, is the model input; the
realized number and structure of layers are observables. The example shows an
explicit larger base brick; adaptive growth remains available if that brick is
still too small.

Periodic three-dimensional fresh starts use the three-dimensional analogue of
the 2D `palmieri_centres` table workflow. Continuous proposals are accepted
sequentially in the periodic cube when they are at least one target radius
from every accepted centre; cell zero is pinned at the box centre.
`palmieri_centres_3d` calls the same placement routine and writes the
realization as a strict `global_id,x,y,z` table for explicit provenance or
paired branches. Soft-cell identities are sampled independently of placement.
With `--rho`, `--cell-radius` derives the same integer box as the solver, while
`--radius` is the minimum accepted centre separation. Both normally equal the
model radius:

```bash
./build/palmieri_centres_3d \
  --N 200 --rho 0.90 --cell-radius 49 --radius 49 --seed 1234 \
  --out initial_centres_3d.csv

./build/palmieri_centres_3d \
  --N 200 --rho 0.90 --cell-radius 49 --radius 49 \
  --validate initial_centres_3d.csv
```

Pass the resulting table to a matching fresh run with
`--initial-centres initial_centres_3d.csv`.

Fresh slab starts without `--initial-centres` use the strict 2D Palmieri
placement. Every slab seed centre is placed at solver coordinate `z=-0.5`, the
lattice face represented as physical substrate height zero. Here `R` is both
footprint and neutral-cap radius. `--rho` is the target used to derive integer
`L`; the realized area fraction is `rho_A=N*pi*R^2/L^2` and is reported.

A substrate start may instead reuse the realized centroids of a passively
relaxed 2D configuration. `tools/relaxed_2d_to_slab_centres.py` reads a final
current 2D checkpoint (preferred) or final complete trajectory frame,
preserves global IDs, wraps coordinates into the primary box, and writes the
same exact `global_id,x,y` interface with a `source_L` box contract plus a
provenance JSON. The slab then
seeds new hemispheres at those positions and must be relaxed again. See
[`docs/relaxed-2d-slab-initialization.md`](docs/relaxed-2d-slab-initialization.md).

Channel starts normally use `global_id,x,y,z`. The built-in initializer is
periodic in x/y, enforces nominal-radius wall clearance in z, and applies the
same one-radius minimum centre separation as the periodic initializer. It uses
the full-sphere target volume and three-dimensional polarity. A one-layer
channel at `H=ceil(2R)` also accepts the exporter's `global_id,x,y` table,
constructs fresh spheres at `z=H/2`, and requires a new passive 3D relaxation.
The manifest supplies the exact `rho_V` that reproduces the source integer
lateral box; a usual 2D `rho_A=0.9` table corresponds to about `rho_V=0.6`
when `H=2R`, not `rho_V=0.9`. This transfer is not used for multilayers.

The two-dimensional executable uses the corresponding `palmieri_centres`
table with header `global_id,x,y`. Its soft cohort is deterministic: the
lowest `round(cancer_fraction*N)` global IDs receive `gamma_cancer`. This keeps
paired runs aligned and is distinct from the 3D solver's seeded sampling
without replacement. Omitting `--initial-centres` selects the grid-and-jitter
initializer used by the smoke test; production studies should record and pass
an explicit table.

## Outputs and restart behavior

### Two-dimensional executable

- `--out` writes a streamed text trajectory with time, cell identity,
  centroids, velocity, polarity, normalized interface measure `L_n`, and
  volume. Existing files are extended only after exact metadata and complete-
  frame validation. Here `L_n=P/(2*pi*R)` uses the cached diffuse-interface
  measure `P`; it is refreshed every `--full-moment` steps and retained between
  those updates.
- `--checkpoint-dir` writes the current binary checkpoint format.
  `-c checkpoint.bin` resumes a run; the checkpoint step and global cell
  identities restore the counter-based Philox tumble stream.

The current checkpoint stores each complete fixed tile without repacking. On a
fatal run, `checkpoint_failed.bin` is written separately so the last accepted
rolling `checkpoint.bin` remains available.

Starting twice from the same checkpoint with the same parameters produces the
same run-and-tumble events. Checkpoints preserve polarity angle, velocity,
cell identity, simulation step, phase field, both full 64-bit random streams,
the initial-centre fingerprint, cached interface measure, exact moments and
support bounds, shape class, and the shape-class demotion counter. Thus the
next update consumes the same adaptive and floating-point state as an
uninterrupted run. Trajectory sampling is aligned to absolute step boundaries;
the resolved cadence is restored unless a new trajectory cadence is supplied
explicitly on resume. Changing that cadence requires a new `--out` path;
appending to a trajectory created with a different cadence is rejected.

### Three-dimensional executable

`cell_gh200_3d` writes geometry-specific text trajectories and independent
`.pf3d` checkpoints with a required complete-file CRC-64 checksum. Each
checkpoint contains every cell's actual storage edge, phase field, unwrapped
origin, velocity, polarity, identity, and the accepted step. Starting twice
from the same checkpoint with the same executable and parameters therefore
replays the same counter-based run-and-tumble events. The checkpoint stores the
resolved base-measurement count and promoted-measurement reduction policy so a
restart cannot silently change floating-point grouping. Explicit conflicting
`--measure-shards` or `--promoted-measure-shards` values are rejected. Channel
checkpoints also store the accessible height, solid padding, wall strength, and
wall width. The resolved trajectory cadence is restored unless a new cadence
is supplied explicitly on resume. Changing it requires a new `--out` path;
appending to a trajectory created with a different cadence is rejected. The
public reader accepts only this current PF3D format and cannot read or overwrite
a 2D checkpoint.

All three-dimensional trajectories use schema 1 with an explicit geometry
token. Periodic x/y/z coordinates remain unwrapped, while slab x/y coordinates
remain unwrapped and `height` is measured above the substrate. Slab surface
columns report the free-interface proxy and compactness normalized to one for
a sharp hemisphere. Resolved-channel output reports the total diffuse
interface measure, a sphere-normalized sphericity proxy, and
`W_i=int(phi_i^2 psi_w^2)dV` with `W_i/V_i`. The population wall-overlap proxy
is `P_w=sum_i W_i/sum_i V_i`. Two further columns report the phase-field
volume outside the physical slit and its fraction of `V_i`, which is the direct
penetration diagnostic. Geometry flags and dimensions are stored in the
checkpoint.

### Rendering a substrate checkpoint

The optional renderer extracts `phi=0.5` surfaces from a current PF3D slab
checkpoint and displays an oblique primary-box cutaway, a top view, and a
local cell-scale view. The oblique view outlines each basal contact patch on
the substrate. Periodic cells are clipped open at the lateral box faces, with
darker caps marking their cross-sections. The title distinguishes the input
`rho_A` from the measured basal `phi>=0.5` coverage. Exact versions of the
renderer's direct Python dependencies are listed separately from the simulator:

```bash
python3 -m venv .venv-viz
. .venv-viz/bin/activate
python -m pip install -r tools/requirements-visualization.txt
python tools/render_pf3d_slab.py \
  checkpoints-slab/checkpoint.pf3d substrate.png
```

`--marching-step 1` selects full-resolution surface extraction; the default
stride of two is intended for inspection. The renderer validates
the complete-file CRC, format, geometry, records, and payload length. It is
not a replacement for the simulator reader's model and restart-compatibility
checks. `--skip-crc` is available only for deliberate display recovery from a
known damaged file.

## Tests

The default build registers the CPU references, the 2D trajectory and exporter
contracts, 3D wall and measurement-sharding contracts, CLI validation, and the
Python checkpoint-header test:

```bash
ctest --test-dir build -C Release --output-on-failure
```

`PF_ENABLE_CHECKPOINT_IO_TESTS=ON` adds 2D/3D writer and reader contract tests,
including the 3D prober and CRC checks. They require any CUDA device for
staging but launch no GPU kernels:

```bash
cmake -S . -B build -DPF_ENABLE_CHECKPOINT_IO_TESTS=ON
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure \
  -R '^(pf2d_checkpoint_seed|pf3d_measure_shards_checkpoint)$'
```

Enable the two public CUDA smoke tests explicitly:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPF_ENABLE_GPU_TESTS=ON
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure \
  -R '^(gh200_smoke|gh200_3d_smoke)$'
```

## Known limitations

- The build defaults to `sm_90`; other GPU architectures and toolchains
  require independent verification.
- One process controls one GPU and one replica. Multi-GPU execution means
  independent processes; a single replica is not spatially decomposed.
- The square periodic lattice and unit spacing are fixed numerical constraints
  enforced at runtime, not general mesh options.
- The 3D extension uses a unit lattice and one GPU. Periodic XYZ remains the
  default; the alternatives are the fixed substrate slab and the resolved
  steric two-wall channel. They are numerically separate from the original
  two-dimensional monolayer model.
- In the 2D solver, at `phi > 1e-5`, ordinary promotion retains an eight-pixel margin, giving the
  fallback a guarded capacity of 278 pixels per axis. An active fallback may
  continue up to its 286-pixel physical interior with a nonfatal warning;
  wider detected support aborts rather than being repacked or clipped.
- The 3D solver promotes cells that exhaust the base brick into a common
  enlarged-storage tier. Their base slots remain allocated, and enlarging the
  tier temporarily requires both its old and new phase-field pools. Growth is
  bounded by the configured HBM budget and by the periodic domain extents. A
  channel brick may exceed the wall separation because out-of-domain z planes
  are not stored in the aggregate field. If either applicable bound is reached, the run
  restores the last accepted in-memory state and exits without clipping it;
  restart uses the last completed rolling checkpoint.
- The slab has neutral contact but no adhesion, and constrains translation and
  polarity to the substrate plane. Multilayer motion and extrusion are outside
  this geometry's scope. Its volume interaction integral changes the length
  dimension of friction relative to 2D, so using the same coefficients in both
  geometries does not imply quantitative equivalence.
- The channel uses steric wall exclusion only. It has no adhesion, wall
  friction, prescribed contact angle, polarity reflection, or wall torque.
- Cross-compiler and cross-GPU bitwise identity is not promised. A build used
  to continue a long simulation should be pinned by source, compiler, and
  executable hash.

## License and citation

The code is released under the GNU Lesser General Public License, version 3
or later (`LGPL-3.0-or-later`). See [`LICENSE.txt`](LICENSE.txt) for the LGPL
terms and [`COPYING`](COPYING) for the GNU GPL version 3 terms incorporated by
it.
Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Publication
venue and DOI are intentionally omitted until they exist.

No third-party source is vendored. CUDA, CMake, and Python remain under
their respective licenses. The counter-based generator implements the
Philox4x32-10 algorithm described by Salmon *et al.*, “Parallel random numbers:
as easy as 1, 2, 3,” SC '11, [doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405).
