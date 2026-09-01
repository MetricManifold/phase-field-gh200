# Seeding a substrate slab from relaxed 2D centroids

This workflow transfers the realized lateral organization of a passive 2D
monolayer into a substrate-resolved run. It transfers only global cell IDs and
phase-field centroid coordinates. PF3D creates new neutral hemispherical phase
fields at those coordinates; it does not copy the 2D fields, cell shapes,
velocities, or polarities. A separate passive slab relaxation is therefore
required.

## 1. Relax the two-dimensional configuration

Run the 2D model with zero active speed for a preselected interval. For example,
the following endpoint is at `3 tau`:

```bash
./build/cell_gh200 \
  --N 24 --radius 49 --rho 0.90 \
  --dt 0.01 --tau 10000 --t-end 30000 --v-A 0 \
  --lambda 7 --kappa 10 --mu 1 --xi 1500 --gamma 1 \
  --seed 1234 --polarity-seed 1234 \
  --out passive-2d/trajectory.txt \
  --checkpoint-dir passive-2d/checkpoints
```

Two to four persistence times is a preparation choice, not an equilibrium
certificate. Check the intended stationarity observables before selecting the
endpoint.

## 2. Export the final centroids

The current checkpoint is preferred because every record contains the
phase-field centroid as binary32. The exporter writes enough decimal digits to
round-trip that exact binary32 value through the PF3D reader.

```bash
python3 tools/relaxed_2d_to_slab_centres.py \
  --checkpoint passive-2d/checkpoints/checkpoint.bin \
  --output relaxed-centres.csv
```

This creates `relaxed-centres.csv` with the exact header `global_id,x,y`, a
`# source_L=...` box-contract line, and `relaxed-centres.csv.json`. The JSON records the source and
table SHA-256 hashes, endpoint time and step, `t/tau`, box, available model
parameters, source coordinate precision, IDs, wrapping, and a `rho` argument
that reproduces the source integer box under PF3D's `ceil` sizing rule.
The source must contain the complete canonical ID population `0..N-1`; the
exporter preserves these identities and writes them in ID order.

A text trajectory is a supported fallback:

```bash
python3 tools/relaxed_2d_to_slab_centres.py \
  --trajectory passive-2d/trajectory.txt \
  --accept-trajectory-precision \
  --output relaxed-centres.csv
```

Current trajectories store metadata and time with round-trip precision, but
cell coordinates and other payload fields retain six digits after the decimal
point. Exact checkpoint centroids therefore cannot be recovered from text. The
explicit flag acknowledges that limitation, which is also recorded in the
JSON. A partial final frame, changing ID sequence, or mutable metadata after
data begins is rejected. Both inputs are rejected by default if their per-cell
active speed is nonzero. A checkpoint establishes zero activity exactly at its
endpoint; a trajectory establishes it only at the stored text precision.

## 3. Seed and relax the slab

Read the exact box-preserving `rho` argument from the manifest:

```bash
rho=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["recommended_slab_start"]["rho_argument_for_source_box"])' \
  relaxed-centres.csv.json)
```

Use the same `N` and radius recorded in the manifest. The startup report must
show the manifest's `expected_Lx`:

```bash
./build/cell_gh200_3d \
  --geometry slab --N 24 --radius 49 --rho "$rho" \
  --initial-centres relaxed-centres.csv \
  --slab-height 288 --brick-edge 168 \
  --dt 0.01 --tau 10000 --t-end 10000 --aging-time 10000 \
  --v-A 0.01 --lambda 7 --kappa 10 --mu 1 --xi 1500 --gamma 1 \
  --seed 1234 --polarity-seed 1234 \
  --out slab-relaxation/trajectory.txt \
  --checkpoint-dir slab-relaxation/checkpoints
```

For an external slab table, `--initial-centres` requires finite coordinates in
`[0,L)` and canonical global IDs `0..N-1` in order. It deliberately does not require
cell zero at the box centre or impose Palmieri's initial minimum separation:
both constraints can be lost during a valid confluent 2D relaxation. The
built-in fresh initializer remains the strict Palmieri sequential placement.
Close or overlapping hemisphere seeds can produce a large initial transient;
the slab relaxation and normal integrity checks remain mandatory.

## One-layer two-wall channel

The same CSV can seed fresh full spheres at the midplane of a one-layer
resolved-wall channel. Read the distinct channel recommendation; do not reuse
the slab `rho_A` argument:

```bash
read H rho_v < <(python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1]))["recommended_one_layer_channel"]; print(r["channel_height"],r["rho_argument_for_source_box"])' \
  relaxed-centres.csv.json)

./build/cell_gh200_3d \
  --geometry channel --channel-height "$H" \
  --N 24 --radius 49 --rho "$rho_v" \
  --initial-centres relaxed-centres.csv \
  --dt 0.01 --tau 10000 --t-end 10000 --aging-time 10000 \
  --v-A 0.01 --lambda 7 --kappa 10 --mu 1 --xi 1500 --gamma 1 \
  --out channel-relaxation/trajectory.txt \
  --checkpoint-dir channel-relaxation/checkpoints
```

The manifest value is chosen so the channel's `ceil` sizing rule reproduces
the source `Lx` exactly. For `H=2R`, the continuum relation is
`rho_V=(2/3)rho_A`; a 2D `rho_A=0.9` table therefore corresponds to about
`rho_V=0.6`, not `0.9`. PF3D checks the CSV's `source_L` against its derived
lateral box, transfers only
IDs/x/y, places fresh spheres at `z=H/2`, and then relaxes them in 3D. This
path is restricted to `H=ceil(2R)` and must not be copied into multiple z
layers; multilayer channels use genuine `global_id,x,y,z` initialization.
