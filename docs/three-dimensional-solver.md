# Three-dimensional solver

`cell_gh200_3d` is a separate three-dimensional formulation of the active
phase-field model. It shares the coefficient convention and run-and-tumble
process of `cell_gh200`, but it does not reinterpret two-dimensional fields or
2D checkpoints. The two executables can therefore be built and used
independently. Fully periodic XYZ remains the default. The sessile-cap
substrate and the two-hard-wall channel are explicit alternatives selected with
`--geometry slab` and `--geometry channel`.

## Model

For cell \(n\),

```text
dphi_n/dt = gamma_n lap(phi_n)
          - (30 gamma_n/lambda^2) phi_n(1-phi_n)(1-2phi_n)
          + (2 mu/V0)(V0-V_n) phi_n
          - (60 kappa/lambda^2) phi_n sum_(m!=n) phi_m^2
          - v_n . grad(phi_n),

v_n = v_A,n p_n + (60 kappa/(xi lambda^2))
      integral(phi_n grad(phi_n) sum_(m!=n) phi_m^2 dV),

V_n = integral(phi_n^2 dV),       V0 = 4 pi R^3/3  (periodic XYZ).
```

The default domain is a periodic cube with unit lattice spacing. Length and
time are solver model units; no physical-unit mapping is implied. The explicit
Euler update uses centered gradients and the isotropic 27-point Laplacian with
face, edge, corner, and center weights `14`, `3`, `1`, and `-128`, divided by
`30`. The aggregate field `sum(phi_n^2)` is accumulated in Q5.27 fixed point,
and a cell's own contribution is subtracted as the same integer used in the
sum.

At every active step, a Bernoulli tumble with probability
`-expm1(-dt/tau)` reproduces exponential survival at the timestep resolution.
For periodic XYZ, a new direction is uniform on the unit sphere: `z=2u-1` and
the azimuth is `2*pi*v`. Counter-based Philox draws are a function of the
complete 64-bit cell identity, step, and stream, so checkpoint continuation
does not depend on a mutable generator state.

### Substrate slab

`--geometry slab` selects the fixed domain

\[
\Omega=\mathbb T^2_L\times[0,H].
\]

The x/y directions are periodic. At the impenetrable substrate, `z=0`, the
phase field has the neutral Neumann condition `dphi/dz=0`, corresponding to a
90-degree contact angle. The top is a far-field `phi=0` boundary, not a second
substrate; supported phase reaching its top plane is fatal. The implementation
uses cell-centred solver planes `z=0,1,...`, whose lower lattice face is
`z=-0.5`, and reports that face as physical height zero.

For this geometry, `R` is both the projected footprint radius and the neutral
hemispherical-cap radius:

\[
V_0=\frac{2\pi R^3}{3},\qquad
L=\left\lceil\sqrt{\frac{N\pi R^2}{\rho_{A,\mathrm{target}}}}\right\rceil,
\qquad
\rho_{A,\mathrm{realized}}=\frac{N\pi R^2}{L^2}.
\]

The height `H=Nz` is independent of `N` and `rho_A`. By default it is twice the
automatically derived base brick edge; `--slab-height H` may override it but
must remain larger than the selected base brick. Production observables must
be unchanged when `H` is increased, with no top-contact alarm, before a slab
run is used for science.

Slab polarity is `p=(cos(theta),sin(theta),0)`, and tumbles redraw only
`theta`. The complete translational velocity, including the interaction term,
is projected into x/y. Shapes and interfaces still relax in all three spatial
dimensions. This neutral-wetting substrate model has no adhesion; multilayer
motion and extrusion are outside its scope.

### Hard-wall channel

`--geometry channel` uses an accessible slit

\[
\Omega=\mathbb T^2_L\times[0,H]
\]

bounded by two identical static diffuse solid fields. In physical coordinates
relative to the lower wall mid-surface,

\[
\psi_w(z)=1-\frac14[1+\tanh(k_w z)]
                    [1-\tanh(k_w(z-H))],\qquad
k_w=\frac{\sqrt{7.5}}{\lambda_w}.
\]

Thus `psi_w` is approximately zero in the slit and one in the solid. The wall
overlap energy is

\[
F_w=\frac{60\kappa_w}{\lambda^2}
    \sum_i\int \phi_i^2\psi_w^2\,dV .
\]

With the model mobility `M=1/2`, it adds
`-(60*kappa_w/lambda^2)*phi_i*psi_w^2` to the Allen--Cahn right-hand side. The
same static overlap enters Palmieri passive translation as

\[
\frac{60\kappa_w}{\xi\lambda^2}
\int\phi_i\nabla\phi_i\,\psi_w^2\,dV .
\]

This is the reciprocal-pair normalization of the ordered cell--cell term. It
contains no adhesion, wall friction, contact-angle energy, wall torque, or
polarity reflection. Cells and run-and-tumble polarity remain fully
three-dimensional.

Numerically, each wall is followed by at least
`padding=ceil(3*lambda_w)` solid voxels. This is the default;
`--wall-padding` accepts a larger integer for a padding-independence check. The
outer allocation faces use the bounded-z reflected stencil, but they lie in
resolved solid rather than at the physical fluid boundary. The allocated
height is therefore `Nz=H+2*padding`; padding does not enter `rho_V`, is stored
in the checkpoint, and cannot be changed on restart. Trajectory heights
subtract the lower padding and are reported relative to the lower physical
wall.

The input `--channel-height H` is the wall separation in lattice units, and

\[
L=\left\lceil\sqrt{\frac{N(4\pi R^3/3)}{
\rho_{V,\mathrm{target}}H}}\right\rceil,
\qquad
\rho_{V,\mathrm{realized}}=\frac{N(4\pi R^3/3)}{L^2H}.
\]

The ceiling in either geometry generally makes the realized fraction slightly
smaller than its target. Both values are written to trajectory metadata.

If the height is omitted, `H=ceil(2R)` gives one nominal cell diameter and is
the minimum supported fresh-start height. Heights near `2R`, `4R`, ..., `12R`
accommodate nominally one through six layers, but layer occupancy and ordering
are measured outcomes rather than constraints. `--wall-kappa` defaults to the
cell repulsion `kappa`, while `--wall-width` defaults to `lambda`. The explicit
Euler gate requires `dt*(60*kappa_w/lambda^2)<1`; wall-strength convergence
must remain inside this bound or reduce `dt`.

Replacing the 2D interaction integral `dA` by `dV` adds one length dimension to
the friction parameter. The default `xi=1500` retains the solver's coefficient
convention, but its numerical value is not a dimension-independent physical
friction and does not imply quantitative equivalence between 2D and 3D.

### Initialization and aging

Fresh periodic runs use

```text
L = ceil(cuberoot(N V0/rho_V_target))
rho_V_realized = N V0/L^3
```

The built-in initializer is the three-dimensional analogue of the
accepted-centre placement used by the 2D solver. Cell zero is fixed at the
centre of the box. Further centres are drawn sequentially and uniformly in the
continuous periodic domain and accepted when they are at least one target
radius from all accepted centres. The random coordinates use the high 53 bits
of `mt19937_64`. An accepted coordinate is stored in binary32 before it is used
as an existing centre in later acceptance tests.

`palmieri_centres_3d` generates the same realization as a strict CSV with the
header `global_id,x,y,z`. The table contains exactly `N` rows ordered by global
identity, fixes cell zero at the box centre, and is revalidated for the run's
box and radius before use. Soft identities are sampled independently without
replacement, with exactly `round(cancer_fraction*N)` cells assigned the soft
coefficient.

A slab instead uses the 2D Palmieri placement and strict
`global_id,x,y` CSV. Every diffuse sphere is centred at solver `z=-0.5`, the
substrate face, so only its upper half occupies the fluid and the neutral seed
has target volume `2*pi*R^3/3`. The stored initialization fingerprint includes
the resulting fixed z coordinate.

A channel normally uses a strict `global_id,x,y,z` table. The built-in sequential
initializer uses periodic minimum-image distances in x/y and ordinary distance
in z. Centres retain one nominal target radius of clearance from each wall,
and cell zero is fixed at the channel centre. The calibrated diffuse seed is
slightly wider than that nominal radius and can initially contact a wall. The
same binary32 rounding, ordered table hash, and one-radius minimum centre
separation used by the periodic initializer are retained.

For a one-layer channel only (`H=ceil(2R)`), the solver also accepts the
exporter's `global_id,x,y` table. It transfers identities and lateral centroids,
constructs fresh full spheres at `z=H/2`, and then requires passive 3D
relaxation. The exporter manifest gives the exact `rho_V` argument that
reproduces the source integer `Lx`; at `H=2R`, the continuum conversion is
`rho_V=(2/3)rho_A`, so a standard `rho_A=0.9` source corresponds to about
`rho_V=0.6`. The reader rejects any table whose realized `Lx` differs. This
one-layer preparation must not be vertically cloned for multilayer channels,
which retain true three-dimensional initialization.

The periodic seed radius is calibrated numerically against `sum(phi^2)` for a
centered brick; the slab uses the upper half of the same wall-centred radial
profile. Subvoxel placement introduces a lattice-quadrature difference, so the
actual volume is measured before the first update rather than assumed to equal
the centered calibration value. A 64-bit FNV-1a fingerprint of each ordered
row's little-endian identity and binary32 coordinates is stored in checkpoints
and trajectory metadata and is included in continuation compatibility checks.

`--aging-time T` makes the first `round(T/dt)` updates passive. During this
interval self-propulsion is zero, while interaction-induced motion and the
phase-field relaxation remain active. Tumble events are not consumed. The
first motile update uses active-process step zero, so a passive checkpoint can
be extended with `--t-end` without changing the subsequent tumble stream.

## Execution strategy

Every accepted step first measures the complete current field. This pass
computes volume, centre, interaction-induced velocity, support bounds, and the
step's run-and-tumble decision. The update then advances the phase fields and
constructs the next aggregate field. These passes are ordered on one CUDA
stream, so an update never observes partially measured cell state.

The standard base-cell measurement and measured update use up to four CUDA
blocks per cell. For small systems, `--measure-shards -1` fills otherwise idle
capacity in one population-wide occupancy wave, up to 64 blocks per cell. If
one block per cell already spans multiple waves, it resolves to one. Values
1--64 pin an explicit count. Because the count determines the floating-point moment-reduction
grouping, it is stored in checkpoints and
must remain unchanged across a continuation. The fast base and enlarged-cell
updates may use up to 64 blocks per cell because they do not carry that
reduction layout. Enlarged-cell measurement remains one block per cell by
default; `--promoted-measure-shards -1` enables an occupancy-derived,
deterministic reduction with up to 64 blocks per cell. Aggregate-field updates
remain exact Q5.27 integer additions.

Both passes walk each brick through dynamic shared memory with asynchronous
staging. The measurement and the sharded update walk halo tiles held in a
double buffer, overlapping each tile's copy with the previous tile's stencil
work. The unsharded throughput update instead walks y-strip slabs with a
rolling four-plane ring and a register-resident stencil window, and writes
every destination voxel exactly once, so it needs no separate clear. In both
mechanisms the staged values, the per-voxel arithmetic, and the per-thread
accumulation order are identical to a synchronous walk: the overlap changes
throughput but no stored result. That y-strip slab-walk kernel is periodic-only;
both bounded-z geometries use the boundary-aware tiled update.

Normal updates do not recompute derived summaries for the field they have just
written, because the next step's mandatory measurement would immediately
replace them. Scheduled strict verification retains the measured updater.
Trajectory, checkpoint, and arbitrary verification requests materialize any
stale volume or surface measurement without consuming a tumble event. The
phase update, integrity flags, fatal repair, and checkpoint state are identical
between the measured and deferred-summary paths.

## Cell bricks and memory modes

Each cell begins with a `B x B x B` phase-field brick. The initialized profile
uses the effective radius `R_eff = R + 1/(2 k_lambda)`, with
`k_lambda = sqrt(7.5)/lambda`, to match the `integral(phi^2 dV)` volume
convention. The default edge contains this effective diameter, the diffuse tail
down to `|phi|=1e-5`, and eight voxels of total diameter slack, then rounds
upward to a multiple of eight. `--brick-edge` may select a larger aligned base
brick.

Before an update is accepted, the measurement pass checks every cell's support
against its current margin. A cell that needs more room is copied without
rounding or clipping into a centered, larger cube; its unwrapped origin is
adjusted so every phase-field value retains the same world coordinate. The
physical step and its run-and-tumble event are then retried. Large-cell storage
is allocated only on demand and grows again if necessary, up to the HBM budget
and the largest aligned cube smaller than the applicable periodic extent. In a
channel, a local cube may exceed `H`; its out-of-domain z planes remain zero and
do not enlarge the dense aggregate field. Base and enlarged
cells share the same aggregate field and equations.

Let `P=4 N B^3` be one phase-field pool and let `G` be one padded, dense
aggregate field. The solver offers three numerically equivalent layouts:

- `throughput`: two phase pools and two aggregate fields, approximately
  `2P+2G` bytes;
- `balanced`: one phase pool, two aggregate fields, and a bounded set of
  per-CTA scratch bricks, approximately `P+2G+4sB^3` bytes;
- `compact`: one phase pool, one aggregate field, and scratch,
  approximately `P+G+4sB^3` bytes.

In the one-phase modes a persistent CUDA block completes one cell in a private
scratch brick before copying it back. Other cells read only the immutable
aggregate field, so the update has no cross-cell phase-field race. Compact mode
reconstructs the aggregate field after every accepted update. `auto` first
tries throughput mode, then prefers balanced mode only while it retains at
least one update block per SM; near that capacity boundary it selects compact
mode when compact restores useful concurrency. By default, at most 95% of free
HBM after a fixed reserve is considered;
`--memory-fraction` may select 0.50--0.99 for a more conservative or
capacity-oriented run. `--scratch-slots` can lower scratch allocation when
memory is binding. Startup warns when capacity pressure reduces scratch
concurrency below the occupancy target or below one CTA per SM; an
allocation-only maximum is not necessarily a useful production size.
In `auto` mode the selector first reserves enough headroom for one pair of
first-tier enlarged cubes; it falls back to a base-only fit only when that
reserve would otherwise prevent the requested system from starting.

An enlarged cell uses two private phase cubes regardless of the base storage
mode. If `K` cells currently use the common enlarged edge `E`, their additional
storage is `8 K E^3` bytes. The original base slots remain allocated, so enough
headroom must be left for a transactional grow operation to allocate the new
cubes before releasing the old ones.

Checkpoint storage is separate from HBM capacity. With per-cell storage edges
`B_n`, a checkpoint occupies
`96 + 288 + sum_n(256 + 4 B_n^3)` bytes and is streamed through bounded host
staging. Estimate the required filesystem space from this formula before
enabling frequent checkpoints.

## Build and run

The normal build produces both executables:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPF_BUILD_3D=ON
cmake --build build --config Release --parallel 8
```

Run the small included example on an `sm_90`-compatible CUDA GPU such as GH200:

```bash
python3 examples/3d/run_example.py \
  --executable ./build/cell_gh200_3d \
  --output-dir ./example-3d-output
```

A representative invocation is:

```bash
./build/cell_gh200_3d \
  --N 32 --radius 49 --rho 0.60 --brick-edge 224 \
  --dt 0.01 --t-end 100 --lambda 7 \
  --aging-time 10 \
  --kappa 10 --mu 1 --xi 1500 --tau 10000 --v-A 0.01 \
  --gamma 1 --gamma-cancer 0.35 --cancer-fraction 0.10 \
  --seed 1234 --polarity-seed 1234 \
  --memory-mode auto --out trajectory-3d.txt \
  --checkpoint-dir checkpoints-3d --checkpoint-interval 10000
```

An opt-in slab invocation and continuation are:

```bash
./build/cell_gh200_3d \
  --geometry slab --slab-height 288 \
  --N 288 --radius 49 --rho 0.90 --brick-edge 152 \
  --dt 0.01 --t-end 100 --lambda 7 \
  --kappa 10 --mu 1 --xi 1500 --tau 10000 --v-A 0.01 \
  --out trajectory-slab.txt --checkpoint-dir checkpoints-slab \
  --checkpoint-interval 10000

./build/cell_gh200_3d \
  -c checkpoints-slab/checkpoint.pf3d --t-end 200 \
  --out trajectory-slab.txt --checkpoint-dir checkpoints-slab
```

A channel example is:

```bash
./build/cell_gh200_3d \
  --geometry channel --channel-height 196 \
  --N 32 --radius 49 --rho 0.60 \
  --brick-edge 224 \
  --dt 0.01 --t-end 100 --lambda 7 \
  --kappa 10 --mu 1 --xi 1500 --tau 10000 --v-A 0.01 \
  --out trajectory-channel.txt --checkpoint-dir checkpoints-channel \
  --checkpoint-interval 10000
```

The examples use explicit base bricks so that a short fresh-start demonstration
does not depend on an immediate adaptive promotion. Adaptive growth remains
available if a selected brick is still too small.

Here `t-end` includes the aging interval. `-c checkpoint.pf3d` restores a
complete successful-step state. An explicit `--t-end` may extend the stored
target; model parameters remain fixed. Supplying only `-c` never overwrites the
input checkpoint. PF3D stores explicit state followed by each cell's
actual `x`-fastest binary32 phase cube; its variable payload preserves enlarged
cells. Boundary flags and all three dimensions distinguish the periodic cube,
substrate slab, and resolved-wall channel. Channel checkpoints store `H`,
padding, `kappa_w`, and `lambda_w`. Geometry is restored from the checkpoint and
cannot be overridden on resume. The promoted-measurement reduction policy is
also restored. Automatic sharding records its originating occupancy wave, and
an explicit conflicting policy is rejected rather than changing reduction
grouping at continuation. The resolved trajectory interval is retained through
checkpoint-only legs and restored unless an explicit cadence is supplied on
resume. Changing the cadence requires a new trajectory path; append to a file
whose stored cadence differs is rejected. The reader accepts only the current
PF3D checkpoint format.

Three-dimensional trajectory schema 1 carries an explicit geometry token.
Periodic output retains unwrapped x/y/z coordinates, so crossing a boundary
does not introduce a box-length displacement jump. Slab output keeps x/y
unwrapped but names its z coordinate `height`: the centroid height above the
physical substrate. It reports the free-interface proxy
`A_free=integral(|grad(phi)| dV)` and

\[
C_{\rm hemi}=2\pi\left(\frac{3}{2\pi}\right)^{2/3}
\frac{V^{2/3}}{A_{\rm free}},
\]

which is one for a sharp hemisphere. The neutral substrate contact is not
included as free interface. Existing trajectory files are appended only when
their exact schema, model, initialization hash, cell ordering, and last time
are compatible with the initialized state. The trajectory interval is part of
that exact append contract.

Resolved-channel output reports height above the lower wall and retains full
three-dimensional velocity and polarity. `interface_measure` is
`int|grad(phi_i)|dV` over the complete allocated cell field and
`sphericity_proxy` is its sphere-normalized ratio. The last columns give the
diffuse wall-overlap proxy `W_i=int(phi_i^2 psi_w^2)dV` and `W_i/V_i`; compute
the population value as `P_w=sum_i W_i/sum_i V_i`. These are overlap measures,
not literal geometric penetration volumes. `outside_slit_volume` separately
integrates `phi_i^2` at voxel centres with physical `z<0` or `z>H`, and
`physical_penetration_fraction` divides it by `V_i`. These diagnostics run only
at trajectory cadence. Every trajectory schema records the promoted-measurement
reduction policy.

### Phase-field visualization

`tools/render_pf3d_slab.py` reads a current PF3D substrate checkpoint and renders
the `phi=0.5` surfaces in an oblique primary-box cutaway, a top view, and a
local three-cell view. The oblique view outlines each basal contact patch on
the substrate. Periodic cells are clipped open at the lateral box faces, with
darker caps marking their cross-sections. The title reports both the input
area fraction and the measured basal `phi>=0.5` coverage. Install the listed
exact versions of the optional direct dependencies and invoke it with:

```bash
python3 --version  # Python 3.11 or newer
python -m pip install -r tools/requirements-visualization.txt
python tools/render_pf3d_slab.py checkpoint.pf3d substrate.png
```

The default marching-cubes stride is two; use `--marching-step 1` when surface
fidelity matters more than rendering time. This is a display tool. Numerical
model and restart-compatibility validation remain responsibilities of the
simulator checkpoint reader; the renderer checks the same mandatory
complete-file CRC before parsing fields.

## Public tests and scientific validation

The default suite covers the CPU stencil and dense one-step references for all
three geometries, the 2D trajectory and exporter contracts, shard-policy and
wall contracts, CLI validation, and the Python checkpoint-header reader:

```bash
ctest --test-dir build -C Release --output-on-failure
```

Configure with `PF_ENABLE_CHECKPOINT_IO_TESTS=ON` to add production checkpoint
writer, prober, loader, and CRC contract tests. They stage through any CUDA
device but launch no kernels.

Enable the public GPU smokes to run the small 2D example and the 3D
checkpoint/continuation example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPF_ENABLE_GPU_TESTS=ON
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure \
  -R '^(gh200_smoke|gh200_3d_smoke)$'
```

For slab science, validation must additionally show neutral-wall contact, zero
normal polarity and translation, no substrate penetration or top contact, and
height-independent volume, footprint, free surface, compactness, and dynamics.
For channel science, validation must show symmetric lower/upper steric force,
bounded wall overlap, finite and stable full-sphere volume, unrestricted 3D
motion, checkpoint-equivalent continuation, and convergence with timestep,
wall strength/width, padding, and lateral box size.

## Limits

- One process advances one replica on one GPU; there is no spatial
  decomposition or multi-GPU replica.
- The mesh spacing, float32 phase field, and stencil are fixed numerical
  choices. Periodic XYZ, the fixed substrate slab, and the resolved steric-wall
  channel are the only geometries; they are not a general boundary-condition
  framework.
- Startup enforces explicit-Euler bounds for diffusion, bulk reaction,
  volume restoration, and nominal active-speed scales. Interaction and
  emergent advection also depend on the realized state; a new parameter regime
  still requires a convergence and stability study.
- Adaptive cell storage cannot exceed the configured HBM budget or the largest
  aligned cube smaller than its applicable periodic extent. Reaching either limit restores
  the last accepted in-memory state and stops the run before supported field
  values are clipped. Restart uses the last completed rolling checkpoint; the
  failed support check does not replace it.
- Enlarged cells share one common cubic edge. Growth of that tier therefore
  enlarges every cell already in it; strongly anisotropic or numerous enlarged
  cells can exhaust HBM before their support approaches the domain size.
- An unrelated fatal error discovered during a `balanced` or `compact`
  in-place update can leave some device fields already updated. That state is
  never written as a checkpoint; restart from the last rolling checkpoint.
- Bitwise identity across CUDA toolchains or GPU architectures is not claimed.
  Pin the source, compiler, executable hash, and hardware for a long
  checkpoint chain.

The original Palmieri model was formulated for a two-dimensional monolayer.
This executable is a three-dimensional numerical extension and should be
validated for the scientific question and parameter regime in which it is
used.

## Model references

- B. Palmieri, Y. Bresler, D. Wirtz, and M. Grant, “Multiple scale model for
  cell migration in monolayers: elastic mismatch between cells enhances
  motility,” *Scientific Reports* 5, 11745 (2015),
  [doi:10.1038/srep11745](https://doi.org/10.1038/srep11745).
- S. Monfared *et al.*, “Mechanical basis and topological routes to cell
  elimination,” *eLife* 12:e82435 (2023),
  [doi:10.7554/eLife.82435](https://doi.org/10.7554/eLife.82435).
- B. Winkler, I. S. Aranson, and F. Ziebert, “Confinement and substrate
  topography control cell migration in a 3D computational model,”
  *Communications Physics* 2, 82 (2019),
  [doi:10.1038/s42005-019-0185-x](https://doi.org/10.1038/s42005-019-0185-x).
- M. Chiang, A. Hopkins, B. Loewe, D. Marenduzzo, and M. C. Marchetti,
  “Multiphase field model of cells on a substrate: From three dimensional to
  two dimensional,” *Physical Review E* 110, 044403 (2024),
  [doi:10.1103/PhysRevE.110.044403](https://doi.org/10.1103/PhysRevE.110.044403).
