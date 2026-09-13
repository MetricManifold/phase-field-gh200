# Phase-field cell simulator for NVIDIA GH200

The canonical repository is [SoftSimu/phase-field-gh200](https://github.com/SoftSimu/phase-field-gh200).
It contains both the two- and three-dimensional solvers.

This repository contains CUDA implementations of the active
phase-field cell model. `cell_gh200` advances the two-dimensional formulation
on a periodic square lattice. The separate `cell_gh200_3d` executable extends
the same coefficient convention and run-and-tumble process to a periodic
three-dimensional volume by default, with substrate-slab and two-hard-wall
channel geometries selected explicitly.

The default executables run one independent simulation replica on one NVIDIA
GH200. An optional two-device runner partitions whole cells between matching,
peer-connected GPUs. Neither path uses MPI, NCCL, or CUDA-aware MPI.

## Executables and geometry modes

The two default executables use dimension-specific numerical cores; a third,
optional executable adds two-device coordination to the 3D core:

| executable | numerical core | supported geometry |
| --- | --- | --- |
| `cell_gh200` | `pf_core` | periodic two-dimensional monolayer |
| `cell_gh200_3d` | `pf3d_core` | periodic XYZ volume (default), substrate slab, or hard-wall channel |
| `cell_gh200_3d_two_gpu` (optional) | `pf3d_core` | substrate slab or hard-wall channel on two peer-connected GPUs |

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

For a two-dimensional cell `n`, the phase field is advanced as

```text
dphi_n/dt = gamma_n lap(phi_n)
          - (30 gamma_n/lambda^2) phi_n(1-phi_n)(1-2phi_n)
          + (2 mu/A0)(A0-V_n) phi_n
          - (60 kappa/lambda^2) phi_n sum_(m!=n) phi_m^2
          - v_n . grad(phi_n),

v_n = v_A p_n + (60 kappa/(xi lambda^2))
      integral(phi_n grad(phi_n) sum_(m!=n) phi_m^2 dA).
```

Here `V_n = integral(phi_n^2 dA)` and `A0 = pi R^2`. The 2D code uses `M=1/2`,
`dx=dy=1`, a nine-point isotropic Laplacian, centered gradients, periodic
boundaries, and binary32 phase fields. Coefficients are defined once in
[`include/params.cuh`](include/params.cuh). In particular, the interaction and
motility coefficients satisfy `interaction/motility = xi` by construction.

All lengths are expressed in lattice units (`dx=dy=1`, and `dz=1` in 3D) and
time in the solver's nondimensional integration unit. Parameters are therefore
model-unit inputs; this repository does not imply a mapping to physical units.

In 2D, each CUDA thread block takes one cell at a time from a shared work queue. The
normal update keeps the active rectangular field in shared memory. Cells that
outgrow those classes use global-memory reads in the same queue, allowing their
longer updates to overlap work on other cells without changing the equations
or run-and-tumble stream. Every cell has a fixed `288 x 288` tile; the fallback
uses its `286 x 286` interior at `(1,1)`. The interaction field is accumulated
in Q5.27 fixed point.

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

Logical cell support remains a `B x B x B` brick. In bounded-z geometries,
each GPU phase-field allocation uses `B x B x min(B, Nz)` values; when `B > Nz`,
the stored planes are indexed by computational world z. This includes the
channel's solid padding, not just the physical wall separation. Throughput,
balanced, and compact modes share this height-matched layout; their difference
is the number of phase, aggregate, and scratch buffers. Exterior planes are
implicit zero, while the boundary conditions and reflected stencil ghosts
are unchanged. Checkpoint files retain the cubic phase-field layout.

The slab is a nonadhesive 90-degree contact model. Related substrate-resolved
phase-field treatments include [Monfared *et al.*, *eLife* 12:e82435
(2023)](https://doi.org/10.7554/eLife.82435), [Winkler, Aranson, and Ziebert,
*Communications Physics* 2, 82
(2019)](https://doi.org/10.1038/s42005-019-0185-x), and the 3D-to-2D derivation
of [Chiang *et al.*, *Physical Review E* 110, 044403
(2024)](https://doi.org/10.1103/PhysRevE.110.044403). Slab height must be shown
not to affect observables before the geometry is used for scientific results.

## Optional 2D observations

`--velocity-moments velocity.bin` records the active, interaction, interfacial,
overlap, area and discrete-advection contributions to centroid motion. Exact
per-step advection integrals are combined with spatial measurements every
100 steps after 10000 densely measured startup steps. The spatial integrals
use a discrete linear correction; sampling accuracy must be checked for the
intended dynamics. Velocity frames follow the trajectory cadence and include
the invocation's initial and final states. See
[the component definitions, format and reader](docs/velocity-output.md).

`--boundary-out boundaries.pfb --boundary-interval 1000` saves compact field
samples for offline contour and neighbour-exchange analysis. Output is disabled
by default; it does not detect T1 events inside the solver. The interval is in
integration steps, not trajectory frames. `--boundary-compression zstd` enables
lossless compression when CMake finds libzstd. Use a new output file for each
restart segment. See [the format and usage notes](docs/boundary-output.md).

Both recorders can run together. Their integer integration steps and stable
cell IDs join the outputs; each has its own explicit file format and sampling
policy. The original update specialization runs when velocity recording is
disabled, and neither recorder changes the model or checkpoint schema.

## Requirements

- CUDA 12 or newer and an NVIDIA GPU supporting the requested CUDA architecture;
- CMake 3.24 or newer and a C++17 compiler;
- Python 3.10 or newer for the tests and command-line tools. The optional
  pinned visualization environment requires Python 3.11 or newer.

The optimized default targets CUDA compute capability 9.0 (`sm_90`).
The measured GH200 configurations below identify their tested compiler and
toolkit versions; the minimum requirements are not a claim that every version
or GPU has been validated.
Windows builds with MSVC 19.44 and CUDA 12.5.82 also pass the host and
checkpoint-I/O tests. That compiler emits register spills in some 2D
specializations, so these build checks do not establish GH200 performance;
use the measured toolchain when reproducing the reported timings.

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

### Two-GPU communication probe

`pf_peer_probe` is an optional fixed-scenario feasibility tool, not a two-GPU
simulator. It validates peer access and measures flat transfers, direct
pitched y-band transfers, and the checked Q5.27 merge used by a possible
two-GPU substrate decomposition:

```bash
cmake -S . -B build-peer \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPF_BUILD_PEER_PROBE=ON
cmake --build build-peer --target pf_peer_probe --parallel 8
./build-peer/pf_peer_probe --iterations 10 --warmup 3 > peer-results.json
```

The JSON records both devices and the fixed `N=100`, `R=49`, `rho_A=0.9`,
`B=152`, `916 x 916 x 288` substrate scenario. Bidirectional bandwidth uses a
common host-wall interval; per-direction CUDA-event medians are reported
separately. These are component timings and do not establish an end-to-end
two-GPU solver speedup.

### Experimental two-device slab and channel solver

`cell_gh200_3d_two_gpu` is an opt-in prototype, not a production release.
It runs one substrate-slab or hard-wall-channel replica on two matching
peer-connected GPUs. The single-device executables and their defaults remain
unchanged.

```bash
cmake -S . -B build-two-gpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DPF_BUILD_TWO_GPU_SUBSTRATE=ON \
  -DPF_ENABLE_GPU_TESTS=ON
cmake --build build-two-gpu --parallel 8
ctest --test-dir build-two-gpu -R '^pf3d_(owner_selection|tile_partition_cpu)$' --output-on-failure
python tests/3d/run_two_gpu_smoke.py \
  --reference build-two-gpu/cell_gh200_3d \
  --executable build-two-gpu/cell_gh200_3d_two_gpu \
  --output-dir build-two-gpu/two-gpu-smoke
```

The executable accepts the existing 3D CLI plus `--peer-device` (default 1);
`--device` selects the primary GPU (default 0). This version supports only
`--geometry slab` or `--geometry channel` and throughput storage; `auto` selects
throughput. `--bench` measures a write-free window; timing a normal run also
includes its regular integrity polls and any requested output.

Whole cells are assigned by their allocated brick's midpoint in periodic y.
Each GPU updates only its owned cells using the shared kernels. The exchange
uses conservative allocated-brick bounds, not a phase-field cutoff, and joins
the quantized overlap contributions with an overflow check. Ownership follows
accepted recentering and adaptive support changes. Output and recovery gather
the accepted state onto the primary GPU and use the ordinary checkpoint and
trajectory routines. Reduction grouping remains fixed by the existing
checkpoint contract; decomposition does not reseed the random generator.

The channel uses the same two resolved steric-wall profiles, full cell volume,
and three-dimensional polarity as the single-device solver. Both update passes
include wall coupling; the peer exchange contains cell contributions only.
Each device constructs its own immutable wall profile from the stored parameters.
Channel exchange bounds include the padded computational height and do not wrap
in z. A cell brick may exceed this height; its exchanged z range is clipped to
the allocated domain. GPU phase-field storage omits the excess z planes, and
kernels skip tiles outside the bounded z domain.

The small channel comparison exercises both wall-coupled update passes,
mixed brick sizes, active tumbles, projected boundaries, and checkpoint restart:

```bash
python tests/3d/run_two_gpu_channel_smoke.py \
  --reference build-two-gpu/cell_gh200_3d \
  --executable build-two-gpu/cell_gh200_3d_two_gpu \
  --output-dir build-two-gpu/two-gpu-channel-smoke
```

The channel test passed on two GH200s: full checkpoints, trajectories, and
maximum-projection boundaries were identical at 1, 10, and 100 active steps,
including strict updates, split restarts, switching device count across restart,
and mixed B24/B32 storage in a 16-voxel-high domain. Resume tests omit measurement
overrides to exercise restoration of the saved grouping. These small tests do
not establish long-run production behavior.

Automatic growth is also tested after distributed stepping starts, with a cell
on either owner or cells on both owners growing together. The recovery path
retains the peer's base fields and shared workspaces, resizes only changed
bricks, then refreshes every future peer-owned field from the accepted state.
Shrinking, incompatible layouts, or insufficient temporary allocation headroom
use the full-rebuild fallback. The comparison includes active tumbles, strict
updates, exact checkpoints and observations, and split restart:

```bash
python tests/3d/run_two_gpu_growth_smoke.py \
  --reference build-two-gpu/cell_gh200_3d \
  --reference-two build-two-gpu/cell_gh200_3d_two_gpu \
  --candidate-one build-two-gpu/cell_gh200_3d \
  --executable build-two-gpu/cell_gh200_3d_two_gpu \
  --output-dir build-two-gpu/two-gpu-growth-smoke
```

This command compares the current one- and two-device executables. For a
cross-version regression, replace the two reference paths with separately
built, pinned reference executables.

Short write-free benchmarks on GH200 GPUs gave the following medians of two
windows, bracketed by runs of the preceding implementation. Hours/tau use
`dt=0.01` and `tau=10000`; they exclude initialization, growth and output costs.

| Fixture | 1 GPU ms/step | 2 GPUs ms/step | 1 GPU hours/tau | 2 GPUs hours/tau |
| --- | ---: | ---: | ---: | ---: |
| N=40 substrate, evolved | 3.460 | 2.939 | 0.961 | 0.816 |
| N=64 channel, evolved | 6.505 | 3.872 | 1.807 | 1.076 |
| N=200 channel, fresh | 18.685 | 10.907 | 5.190 | 3.030 |

The N=40 state is at 0.201 tau, with domain 580 x 580 x 288, R49,
39 B160 cells and one B192 cell. The N=64 state is at 0.71 tau, with
domain 733 x 733 x 140, R49, wall gap 98 and no promoted cells. Both use
64 base measurement shards; the N=40 promoted measurement count is also 64.
Their windows contain 100 accepted steps. The fresh N=200 channel has domain
1296 x 1296 x 140, R49, wall gap 98, B160 and 64 base measurement shards.
Its windows cover only the first 20 steps before support growth: they are
not measurements of a mature N=200 state or a long-run forecast.

Tile staging advances integer coordinates with row/plane carries, and
measurements reuse fixed boundary maps. These changes reduced runtime by
1.8–4.5% in the bracketed comparisons, without changing halo contents or
measurement grouping. The build used CUDA 13.1.115 and `sm_90`, without
fast-math or register spills; all timing guards passed without promotion or
recovery. Ten-step continuations of both saved states produced byte-identical
full checkpoints, trajectories and projected boundaries against the preceding
implementation on one and two devices. The N=40 two-device 5+5 restart also
matched. Height-matched storage saves 250 MiB of phase allocation per device
in the N=64 fixture; checkpoint sizes are unchanged.

This implementation replicates storage: it does not double cell capacity.
Fast updates first process whole tiles intersecting the outgoing aggregate
bands, then overlap peer copies with the remaining tiles. Classification uses
source voxel coordinates, before recentering, so late tiles cannot modify a
transmitted band. Both copies finish before either received aggregate is added.
Strict measured updates retain their complete walk and reduction grouping.
Bounded base updates automatically queue multiple occupancy waves of shorter
tasks to balance uneven live-tile and boundary/interior work. This does not regroup any
floating-point reduction; `--fast-base-shards` can override the task count.
Enlarged-cell fast updates use the same queued-wave scheduler, with an automatic
cap of 512 tasks per cell. `--promoted-shards 1..1024` pins their count. These
settings do not change promoted measurement grouping or strict-update reductions.
Metadata and integrity counters are still
coordinated on the host each step, using reusable pinned snapshots and two
joined download phases. Uploads precede their consumers in each device stream.
Recentering refreshes exchange bounds without reinstalling unchanged owner lists.
Healthy polls read control state without gathering fields; output, verification,
and recovery materialize the canonical
state and check integrity. It remains experimental, without a long-run
production gate.

`--bench-phases` reports the primary device's timeline. Its final phase includes
interior updates, the exchange join, and pre-commit integrity coordination;
fast-update phases cover only boundary tiles. The interstep gap includes
post-commit coordination. These are not separate timings of every peer's work.

On two GH200 GPUs, the supplied tests produced byte-identical single-/two-device
checkpoints and trajectories for active tumbles, strict updates, split restart,
mixed promoted/base storage, and migration across both y cuts with adaptive
growth. The tests also cover cropped exchange, genuine boundary/interior tile
work with split restart, routine polls between sparse outputs, and a partition
with no owned cells. The test system
uses four cells with `R=5`; this does not establish long-time accuracy or
cross-hardware reproducibility.

An earlier N=100 normal-run comparison, before the bounded-z measurement
refinement below, used identical `R=49`,
`rho_A=0.9`, `B=224`, `Nz=288`, `dt=0.01`, seed 20471, and four measurement
shards on both executables. The table uses the automatic two-device scheduling
policy and `--fast-base-shards 512` on the single-device executable:

| Cells | One GPU (ms/step) | Two GPUs (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| 100 | 22.978 | 12.838 | 1.79× |

With standard single-device scheduling, the corresponding time was about
24.69 ms/step; that default remains
unchanged. Explicit fast-update counts affect scheduling, not the pinned
measurement reductions or checkpoint format.

Each table value is the median of three runs with varied execution order. Host receipt
timestamps span steps 32 to 288, including the usual 16-step integrity polls;
trajectory and checkpoint writing are disabled. No promotions or recoveries
occurred. These short fresh/evolving-state windows are not mature-state or
long-run forecasts, and host-pipe jitter limits sub-percent comparisons.
Both devices maintained 1,980 MHz SM clocks under load. Matching GPU names and
SM counts alone does not establish equal throughput; clocks must also be checked.
A smaller N=24 case with the same conservative brick edge gained essentially
nothing from a second GPU. Do not extrapolate the N=100 gain to all populations
or geometries.
The tested build used CUDA 13.1.115, GNU 11.5 as CUDA host compiler, NVHPC 26.3
for C++ tools, and `sm_90`, without fast-math or reported register spills.

For N=250, precomputing the fixed bounded-z mapping avoids repeated world-coordinate
checks in the base measurement kernel. Its integer interval and reflected-ghost
rules are covered by a CPU oracle; the voxel order and floating-point reduction
are unchanged. The same-grouping GPU tests compare exact checkpoints and
trajectories, including restart, promotion and migration, with additional
periodic and channel checks.

With the same substrate geometry and timing protocol, three varied-order
repetitions at 64 base measurement shards gave:

| N=250 configuration | Median ms/step |
| --- | ---: |
| Previous two-device kernel, 64 measurement shards | 26.195 |
| Bounded-z refinement, two devices, 64 measurement shards | 22.823 |
| Bounded-z refinement, one device, 64 measurement shards, fast256 | 39.746 |

The kernel refinement reduces two-device runtime by 12.9% at a fixed grouping.
Together with scheduling, the time is 18.5% below the earlier four-shard
two-device baseline (27.997 ms/step, a single same-job window). The resulting
one-to-two-device speedup is 1.74x; it is not the N=100 result above.

The **fresh N=250 substrate benchmark** used
`--memory-mode throughput --measure-shards 64 --promoted-measure-shards 4`;
leave `--fast-base-shards` unset so the two-device scheduler adapts to ownership.
The benchmark had no promoted cells; four promoted measurement shards were
held fixed, not optimized for a promoted-heavy state. Standard measurement
defaults are unchanged. On continuation, omit a new measurement override and
restore the checkpoint's stored grouping. For performance-oriented new runs,
start with `--promoted-measure-shards -1` and benchmark an evolved state;
the fresh benchmark does not establish four shards as a suitable long-run policy.

Changing the measurement grouping changes floating-point summation and can
select a different integer recentering at a rounding tie. Therefore local
arrays need not match between groupings. The small four-versus-sixteen and
four-versus-sixty-four checks had bit-identical phase fields after alignment
to physical coordinates at steps 1, 10 and 100. This is not a cross-grouping
long-time trajectory guarantee. Each fixed grouping remains restart-exact
in the tested cases.

`B=224` is a conservative benchmark allocation, **not an established production
brick size for 3D**. Two-dimensional deformation limits do not establish it.
Use evolved normal and soft cells to assess full diffuse-field support and
the required margins before choosing campaign storage; retain adaptive growth
without clipping. The timings above must not be extrapolated to mature,
promoted-heavy, or differently confined systems without measurement.

The independently sized brick implementation was also measured with smaller
starting cubes, keeping N=250, the substrate geometry, physical parameters,
two GPUs, and 64/4 measurement settings fixed. Two varied-order repetitions
of the same 32-to-288-step timing window gave:

| Starting brick edge | Median ms/step | Allocated GiB per GPU |
| --- | ---: | ---: |
| 224 | 22.840 | 25.51 |
| 192 | 17.123 | 17.76 |
| 176 | 15.402 | 14.73 |
| 160 | 11.977 | 12.20 |

These gains come from using smaller cubes, not from a faster update equation.
The preceding implementation at B=224 measured 22.819 ms/step in the same
job. None of these short timing windows required promotion. A separate
2,000-step B=160 diagnostic with 25 soft cells among 250 completed with finite
trajectories and no resizing; it is only 0.002 tau, not production equilibration
or a long-time storage calibration. B=160 is a candidate for further diagnostics,
not a proven universally sufficient size.

Mixed-size growth, guarded cropping, memory recovery and checkpoint loading
passed GPU tests. Small active one-/two-device runs matched checkpoints and
trajectories across scheduled compaction, including split restarts in fast and
strict update modes. Enable `PF_ENABLE_GPU_TESTS` to build the retained
`pf3d_adaptive_bricks_gpu` test; the size-policy CPU test runs by default.

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
checkpoint contains every cell's logical brick edge, phase field, unwrapped
origin, velocity, polarity, identity, and the accepted step. Starting twice
from the same checkpoint with the same executable and parameters therefore
replays the same counter-based run-and-tumble events. The checkpoint stores the
resolved base-measurement count and promoted-measurement reduction policy so a
restart cannot silently change floating-point grouping. Explicit conflicting
`--measure-shards` or `--promoted-measure-shards` values are rejected by default. Channel
checkpoints also store the accessible height, solid padding, wall strength, and
wall width. The resolved trajectory cadence is restored unless a new cadence
is supplied explicitly on resume. Changing it requires a new `--out` path;
appending to a trajectory created with a different cadence is rejected. The
public reader accepts only this current PF3D format and cannot read or overwrite
a 2D checkpoint.

The on-disk phase payload remains `B^3` float32 values per cell, independent of
GPU storage mode. Writing expands omitted z planes as zero through a bounded
host buffer. Loading packs the retained planes and rejects nonzero or nonfinite
values outside the compact allocation; the complete-file checksum still
covers every serialized plane. Height-matched GPU storage therefore saves
device memory, not checkpoint disk space.

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

### Tuning measurement of enlarged cells

Base and enlarged cells have separate measurement policies. A small number of
enlarged cells can underutilize the GPU with a low fixed
`--promoted-measure-shards` count. `-1` selects an occupancy-derived count
(up to 64); a positive value pins the count. Benchmark an evolved state, since
a fresh run may have no enlarged cells at all.

To retune an existing checkpoint, supply both `--promoted-measure-shards`
and `--allow-promoted-measure-regroup`. The latter explicitly permits a change
in floating-point summation order. It preserves the loaded phase fields, cell
identities, model parameters, simulation step and tumble stream; derived
moments and velocities are recomputed. Subsequent floating-point trajectories
need not be identical to those using the previous grouping.

Keep the input checkpoint and write to a new trajectory/checkpoint directory.
New checkpoints store the selected policy, so later restarts can omit both
flags. Repeating an unchanged automatic policy preserves its stored occupancy
wave. Base-cell grouping is unaffected. Without the opt-in, incompatible
groupings remain errors; mismatched trajectory appends are always refused.

For example, retune a checkpoint at simulation time 2000 and continue to 2010
in a new output directory:

```sh
mkdir retuned
build/cell_gh200_3d -c checkpoint.pf3d --t-end 2010 \
  --memory-mode throughput --promoted-measure-shards 64 \
  --allow-promoted-measure-regroup \
  --out retuned/trajectory.txt --checkpoint-dir retuned/checkpoints
```

An evolved N=40 substrate state at `t=0.2*tau`, with 39 cells in B=160
cubes and one in B=192, was measured on GH200 with 64 base-measurement shards.
Each entry is the median of two 200-step write-free windows, tested in opposite
orders; all timing guards passed. At `dt=0.01` and `tau=10000`:

| Promoted measurement shards | One GPU, ms/step | Two GPUs, ms/step |
| --- | ---: | ---: |
| 4 | 8.929 | 7.802 |
| 64 | 4.530 | 3.479 |
| Automatic (`-1`) | 4.552 | 3.489 |

The fixed-64 result projects to 1.26 hours per tau on one GPU and 0.97 hours
on two, excluding output time. It is a measurement of this mixed-size state,
not a guarantee for other populations or later cell shapes. Using two GPUs
reduces elapsed time but uses about 1.54 times as many GPU-hours here.
On this checkpoint, the 4-to-64 change preserved the initial fields exactly;
after 1000 steps the two runs also had identical fields in world coordinates,
polarities, and tumble counters. This short comparison does not establish
long-time bitwise equivalence between reduction policies.

The retained smoke test checks unchanged initial fields, matching active
tumble events and polarities, checkpoint-policy restoration, and split restart:

```sh
python tests/3d/run_promoted_regroup_smoke.py \
  --executable build/cell_gh200_3d --output-dir regroup-smoke
```

Add `--two-gpu-executable build/cell_gh200_3d_two_gpu` to exercise the
two-device continuation and gathered state as well. The test requires suitable
GPU resources; it does not submit a scheduler job.

### Compact projected boundaries

Add `--boundary-out boundaries.pfb3d --boundary-interval 1000` to save
the actual `phi=0.5` top-down contours without writing full 3D fields.
`--boundary-projection maximum` (default) records each cell's silhouette;
`--boundary-projection basal` records its footprint at the substrate plane
and is available only with `--geometry slab`. Both retain cell IDs and
unwrapped origins for periodic reconstruction. This is an optional observer:
it does not change initialization, integration, or tumble events.

The stream includes the initial state, the requested absolute step cadence,
and the final state. Use a new file for each restart segment. Inspect it with
`python tools/pf3d_boundary.py boundaries.pfb3d`; the reader also exposes frames
for analysis. Projected boundaries are input for neighbor-exchange analysis,
not automatically a 3D contact graph or a T1-event classification.
See [the format and usage guide](docs/boundary-output-3d.md) for projection
semantics, optional lossless zstd compression, and tests.

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
- The default executables use one GPU per replica. The optional two-device
  slab/channel prototype above is not yet validated for production.
- The square periodic lattice and unit spacing are fixed numerical constraints
  enforced at runtime, not general mesh options.
- The 3D extension uses a unit lattice. Periodic XYZ remains the
  default; the alternatives are the fixed substrate slab and the resolved
  steric two-wall channel. They are numerically separate from the original
  two-dimensional monolayer model.
- In the 2D solver, at `phi > 1e-5`, ordinary promotion retains an eight-pixel margin, giving the
  fallback a guarded capacity of 278 pixels per axis. An active fallback may
  continue up to its 286-pixel physical interior with a nonfatal warning;
  wider detected support aborts rather than being repacked or clipped.
- The 3D solver enlarges each cell's cube independently. Growth adds roughly
  32 planes to the edge, rounded to a 16-plane size class (the final
  domain-limited class can use the underlying eight-plane alignment). Every
  1024 accepted steps it considers smaller cubes, retaining 16 planes around
  the measured support bounds. A GPU check additionally requires every
  discarded voxel and an eight-plane inner guard to be exactly zero, and
  rejects nonfinite fields. Cells whose tails do not fit remain unchanged.
  Resizing preserves the current world-coordinate field; changing its local
  cube can change floating-point moment reductions and is not guaranteed to
  reproduce a trajectory computed in a permanently larger cube.
  Each cell's edge is already stored in the checkpoint; the compaction schedule
  uses absolute steps and needs no additional checkpoint history.
  Base slots remain allocated. An existing checkpoint cannot shrink below
  its base edge, so a uniform B=224 checkpoint does not automatically gain
  smaller storage. Replacement allocation temporarily requires old and new
  buffers; optional compaction is postponed if that exceeds the budget. Growth is
  bounded by the configured HBM budget and by the periodic domain extents. A
  channel brick may exceed the wall separation; the solid padding is retained,
  while planes outside the computational z domain are omitted from GPU phase
  and aggregate storage. If either applicable bound is reached, the run
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

No third-party source is vendored. CUDA, CMake, Python, and optional zstd remain under
their respective licenses. The counter-based generator implements the
Philox4x32-10 algorithm described by Salmon *et al.*, “Parallel random numbers:
as easy as 1, 2, 3,” SC '11, [doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405).
