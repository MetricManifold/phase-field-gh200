# Two-dimensional velocity components

The optional recorder measures the contributions to the phase-field centroid
velocity. It uses the actual field, interaction sum and imposed direction
consumed by the update. It does not change the PDE, timestep, random stream,
initialization or checkpoint format. The file records each cell's actual
phase-field mobility, whose default is M=0.5. Cell-specific mobility scales
the passive interfacial, overlap and area contributions by M_i/0.5; propulsion,
interaction advection and the advection correction are not scaled.

For example, in a new output directory:

~~~sh
mkdir measured-run
build/cell_gh200 --N 400 --t-end 1000 \
  --out measured-run/trajectory.txt --trajectory-interval 1000 \
  --checkpoint-dir measured-run \
  --velocity-moments measured-run/velocity.bin \
  --boundary-out measured-run/boundaries.pfb --boundary-interval 1000 \
  --boundary-compression zstd
python tools/velocity_moments.py measured-run/velocity.bin \
  --expected-final-step 100000
~~~

Boundary compression requires libzstd at build time. The timestep in this
example is the default dt=0.01. For a checkpoint continuation, supply -c,
an end time greater than its current time, and new recorder paths. The writer
refuses to overwrite an existing recorder file.

## Measurement policy

| Option | Meaning | Default |
| --- | --- | --- |
| --velocity-moments PATH | Enable recording into a new PFVMOM4 file | disabled |
| --velocity-spatial-stride S | Measure spatial terms every S physical steps after startup; 1 through 1000 | 100 |
| --velocity-dense-start D | Measure every step during the first D steps of this invocation | 10000 |
| --velocity-moments-reference | Independent dense observer used for validation | disabled |

Propulsion A, interaction advection I, and the rounding difference between
their double sum and the stored float velocity are integrated every physical
step. These scalar updates are fused into the physics kernel. They include
the newly drawn direction on a tumble step.

G and K are spatial moments of the interfacial and overlap-relaxation terms.
Their variational arithmetic is evaluated in double on the actual float field.
Sampling integrates the original pre-step discrete sum. For S steps between
two sampled rates, the contribution is

~~~text
dt * [S*g_start + (S-1)/2 * (g_end - g_start)].
~~~

The factor S-1 is required by the discrete pre-step sum. Output uses an
independent measurement at the current endpoint to finish partial intervals.
Dense startup covers rapid responses to changed parameters; it adds no
equilibration steps. An unaligned restart always primes the observer before
replaying a sparse graph.

The discrete-advection correction L uses the same-cell neighbor overlap
Bx=sum(phi(x,y)*phi(x+1,y))/V and its y counterpart. With B held between
spatial samples,

~~~text
L = (B-1)*delta(Q_A+Q_I) + B*delta(Q_rounding).
~~~

The velocity changes inside that interval remain exact. The centroid
contribution of area relaxation is analytically zero; the independent
reference observer measures its floating-point residual.

Sparse spatial integration is an approximation. Check stride convergence for
new parameters, particularly rapid direction changes or shape transients.
--velocity-spatial-stride 1 provides the dense fused comparison. Numerical
closure also contains finite-Euler and float error: a residual is not itself
an additional physical mechanism.

## Ownership and performance

The update kernels choose compile-time specializations for disabled,
advection-only, and spatial recording. Disabled recording performs no observer
arithmetic, allocations or extra measurement launches. Runtime advection data
occupy typed fields in otherwise unused CellState storage; the record remains
192 bytes and those fields are excluded from checkpoints.

The recorder owns its device arrays, reusable host buffers and output stream.
The simulation owns launch order and output scheduling. Measurement math and
the sampling policy are independently testable without CUDA. Spatial launch
variants share one launch-configuration path with the original kernels.

Long observation periods are split into correctly phased CUDA graphs at the
spatial cadence (three 100-step graphs at the default stride). The total
captured work is unchanged. Boundary saves can interrupt replay at these
shorter boundaries without forcing the rest of a 300-step period into separate
launches. Output cadences that are not multiples of a graph length still use
individual launches for their exact endpoints.

--bench STEPS --velocity-moments unused.bin measures steady observation cost
without opening output files or applying dense startup. It does not measure
file I/O. Compare complete runs with the same checkpoint, trajectory cadence
and boundary settings to measure total cost. Boundary output may dominate
storage and I/O at fine cadences even when velocity recording is inexpensive.

## PFVMOM4 layout

All numbers are little-endian. common/velocity_format.h fixes the layout
independently of runtime structures.

| Header offset (bytes) | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | PFVMOM4 plus NUL |
| 8 | uint32 | cell count |
| 12 | uint32 | columns per cell, 26 |
| 16 | float64 | physical dt |
| 24 | int64 | invocation's initial integration step |
| 32 | float64 | periodic domain side |
| 40 | uint32 | spatial sampling stride |
| 44 | uint32 | quadrature mode, 2 |
| 48 | uint64 | densely measured startup steps |

Each frame is an int64 integration step followed by N rows of 26 float64
values. Column numbers below are zero-based.

| Columns | Meaning |
| --- | --- |
| 0 | stable cell ID |
| 1–2 | wrapped phase-field centroid x/y |
| 3 | stored polarity angle |
| 4–7 | gamma, mobility, active speed, volume |
| 8 | integrated physical step count since the invocation began |
| 9–10 | cumulative A x/y |
| 11–12 | cumulative I x/y |
| 13–14 | cumulative G x/y |
| 15–16 | cumulative K x/y |
| 17–18 | cumulative area-relaxation contribution x/y |
| 19–20 | cumulative L x/y |
| 21 | number of spatial observations |
| 22–25 | instantaneous endpoint Gx, Gy, Kx, Ky |

G/K integrals already include the discrete linear correction. Do not apply it
again during analysis. The first frame has zero integrals but can have
nonzero endpoint rates. Each invocation has a new accumulator origin; combine
segments explicitly using their step metadata.

The standard-library reader in tools/velocity_moments.py streams one frame
at a time, checks metadata, identity, counts and finite values, and rejects
truncated records. PFVMOM4 has no final terminator: a file ending exactly on a
frame boundary does not prove that a scientific run finished. Supply
--expected-final-step or cross-check its last step against the successful
run/checkpoint record.

Boundary snapshots retain their existing PFBND01 format. Match the two streams
by step and stable cell ID. A stored polarity angle describes the output state;
the cumulative A channel already accounts for the direction actually used on
each completed update. A T1 detector should retain its own event-time brackets.
