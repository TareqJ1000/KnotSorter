# Knot Sorter

This project optimizes phase-only optical systems that direct distinct knotted
fields into assigned detector apertures. The current physical model supports
one, two, or three phase planes and can optimize the phase masks jointly with
the free-space distances and lens focal lengths.

## Physical propagation model

Every phase plane owns one optical stage:

```text
input -> phase 1 -> P(z1) -> lens(f1) -> P(z2) -> [next phase or detector]
```

The same stage is repeated for phase planes 2 and 3 when present. `P(z)` is the
Fresnel transfer-function propagator

```text
U(x,y;z) = F^-1{ F[U(x,y;0)] exp[-i*pi*lambda*z*(fx^2+fy^2)] },
```

and an ideal thin lens applies

```text
t_lens(x,y) = pupil(x,y) exp[-i*k*(x^2+y^2)/(2*f)].
```

The longitudinal carrier phase is omitted because it is spatially uniform and
does not change any intensity or later phase-only modulation. A finite circular
lens pupil is optional; pupil loss is retained in the detector efficiency.

The direct FFT sorter remains available for archived configurations: any YAML
without an `optical_train` block is interpreted as `legacy_fft`. It supports
any positive number of phase planes by alternating centered forward and inverse
transforms: `FFT`, `FFT -> IFFT`, `FFT -> IFFT -> FFT`, and so on. The physical
Fresnel/lens model remains limited to one, two, or three planes.

## Configuration

The ready-to-run configurations are:

- `configs/ga0.yaml`: one phase plane
- `configs/ga1.yaml`: two phase planes
- `configs/ga2.yaml`: three phase planes
- `configs/ga3.yaml`: legacy FFT, one phase plane
- `configs/ga4.yaml`: legacy FFT, two phase planes
- `configs/ga5.yaml`: legacy FFT, three phase planes
- `configs/ga_legacy_smoke.yaml`: tiny seven-plane legacy integration check
- `configs/ga_base.yaml`: documented two-plane template
- `configs/ga_chirality.yaml`: two-plane Fresnel sorter for mirrored trefoils
- `configs/ga_smoke.yaml`: tiny three-plane integration check, not a science run

The number of entries under `optical_train.stages` must equal
`num_phase_planes`. Distances and focal lengths are in centimetres. A scalar
fixes a parameter; a bounded mapping adds an optimizable normalized gene:

```yaml
num_phase_planes: 2
optical_train:
  model: fresnel_lens_train
  optimize_geometry: true
  lens_aperture_radius_mm: 1.0
  padding_factor: 2.0
  output_coordinate_mode: physical
  geometry_mutation_probability: 0.20
  geometry_mutation_scale: 0.05
  stages:
    - z_to_lens_cm: {initial: 6.0, min: 3.0, max: 6.5}
      focal_length_cm: {initial: 6.0, min: 5.5, max: 10.0}
      z_after_lens_cm: {initial: 6.0, min: 3.0, max: 6.5}
    - z_to_lens_cm: 6.0
      focal_length_cm: {initial: 6.0, min: 5.5, max: 10.0}
      z_after_lens_cm: 6.0
```

For an arbitrary legacy train, omit `optical_train` (or set its model to
`legacy_fft`) and choose any positive plane count:

```yaml
num_phase_planes: 7
multiPhase: false
multiPhaseLens: false
simulateLens: false
```

Such a configuration contains `num_phase_planes * dim**2` phase genes. These
planes alternate between the same two Fourier-conjugate numerical coordinate
systems; they do not represent independently spaced free-space planes.

Geometry genes are stored internally on `[0, 1]`, decoded to their physical
bounds for every candidate, mutated independently from the phase pixels, and
clipped to the allowed intervals. The initial layout is always seeded into the
first candidate.

For a knot alphabet, `mirror` may be a boolean applied to every input or a
boolean list aligned with `knotType` and `shapeParams`. It defaults to `false`
for backward compatibility. For example, the two trefoil enantiomers are:

```yaml
isKnot: true
knotType: [Trefoil, Trefoil]
shapeParams: [[0.9, 0.9, 1.0], [0.9, 0.9, 1.0]]
mirror: [false, true]
```

### Numerical-window padding

Set `optical_train.padding_factor` to any value at least 1. A value of `1`
disables padding; `2` doubles the sampled side length while retaining the same
pixel pitch. The incident field is zero-padded, while each optimized phase mask
is embedded in a unity-transmission background. The field remains on this
expanded grid throughout every Fresnel/lens stage, allowing diffracted light to
leave the original window without wrapping periodically into it. The result is
cropped back to the original detector window for fitness and efficiency
measurements.

Padding changes neither the phase-plate pixel count nor its pixel pitch. FFT
memory use grows approximately as the square of the padding factor, so the
production YAML files use `1.0` by default. `configs/ga_smoke.yaml` uses `2.0`
to exercise the padded path.

### Padded refinement stage

The optimizer can run a third stage after the native-grid warm-up and full
optimization. It seeds the final full-metric population into a padded-grid GA,
re-evaluates every candidate with the full fitness metric, and continues for a
user-selected number of generations:

```yaml
gen_start: 10000                 # stage 1: sorting/throughput warm-up
num_of_gens: 100000              # stage 2: native-grid full metric
refinement_generations: 2000     # stage 3: padded-grid full metric
refinement_padding_factor: 2     # 128 -> 256 propagation grid
refinement_saturate: null        # optional early stopping, e.g. 500
```

Set `refinement_generations: 0` to disable this stage. The device phase arrays
and chromosome remain at their native `dim x dim` resolution. For a legacy FFT
train, the padding factor must be a positive integer. Input/image-plane devices
retain a centred finite footprint, while Fourier-plane device pixels are
replicated to preserve their physical size on the finer conjugate-plane grid.
Detector masks are mapped by the same parity rule. For a Fresnel/lens train,
the refinement factor overrides `optical_train.padding_factor` only during the
third stage.

The best phase pickle and GA instance are updated throughout refinement. The
geometry metadata records `optimization_stage: padded refinement` and the
refinement padding factor for the final refined solution.

### Resuming from a best-phase checkpoint

Use `start_stage` with `seed_from` to skip completed stages. A bare
`seed_from` value resolves to `best_phases/<name>.pkl`; an explicit pickle path
is also accepted. To continue directly with the native-grid full metric:

```yaml
start_stage: full
seed_from: tref_cinque_legacy_2plane
gen_start: 0                    # ignored because warm-up is skipped
num_of_gens: 50000              # additional full-stage generations
refinement_generations: 0       # set >0 to refine after the full stage
ga_instance: tref_cinque_legacy_2plane_full_continued
```

To continue directly on the padded refinement metric:

```yaml
start_stage: padded_refinement
seed_from: tref_cinque_legacy_2plane
gen_start: 0                    # ignored
num_of_gens: 0                  # ignored
refinement_generations: 10000   # additional padded generations
refinement_padding_factor: 2
refinement_saturate: null
ga_instance: tref_cinque_legacy_2plane_refine_continued
```

`start_stage` defaults to `warm_up`, preserving the original
warm-up -> full -> optional refinement sequence. Supplying `seed_from` with
that default merely seeds the warm-up population.

The checkpoint candidate occupies the first row of a newly initialized GA
population; the remaining candidates provide fresh diversity. Consequently,
this continues from the saved best optical solution but does not restore the
old population, generation counter, or fitness history. Use a new
`ga_instance` unless overwriting the source checkpoint is intentional.

The best-phase files contain already-smoothed masks. The loader recovers phase
genes that reproduce those masks after `decode_solution`, avoiding an extra
Gaussian blur at resume time. For optimized Fresnel geometry, the matching
`<name>_geometry.yaml` sidecar is required and its values must lie inside the
new YAML bounds. The phase-plane count, grid dimension, propagation model,
input alphabet, detector geometry, wavelength, pixel pitch, and beam parameters
should otherwise match the checkpoint's original run.

### Fitness and detector throughput

The conditional assignment matrix still controls sorting contrast, key rate,
and determinant-based distinguishability. The `full`/`bread` objective also
multiplies those terms by a balanced absolute detector-throughput factor:

```text
eta_n = sum_m I_nm
eta_bal = (product_n eta_n)^(1/d)
F_full = C_bal R_d |det(p)|^gamma eta_bal^beta.
```

Configure the throughput term alongside `alpha` and `gamma`:

```yaml
throughput_metric: geometric_mean # geometric_mean, minimum, arithmetic_mean
throughput_exponent: 1.0          # beta; 0 restores conditional-only fitness
```

The geometric mean is recommended because a solution cannot obtain a high
throughput score by sacrificing one knot while transmitting the others. With
`throughput_exponent: 1.0`, a uniform factor-of-ten reduction in accepted
detector power produces a factor-of-ten reduction in the full fitness.

## Running

Validate the setup, sampling diagnostics, and a candidate propagation without
starting a long optimization:

```powershell
.\venv\Scripts\python.exe run_ga.py --ii 0 --validate-only
.\venv\Scripts\python.exe run_ga.py --ii 1 --validate-only
.\venv\Scripts\python.exe run_ga.py --ii 2 --validate-only
.\venv\Scripts\python.exe run_ga.py --ii 5 --validate-only
```

Launch a run by omitting `--validate-only`, or select any YAML explicitly:

```powershell
.\venv\Scripts\python.exe run_ga.py --ii 2
.\venv\Scripts\python.exe run_ga.py --config configs\ga_base.yaml
```

At each generation the best smoothed phase angles are saved to
`best_phases/<ga_instance>.pkl`. The corresponding decoded physical layout is
saved to `best_phases/<ga_instance>_geometry.yaml`. `PhaseAnalyzer.ipynb` loads
both files for the physical train when `experiment_name = None` and `index` is
set to 0, 1, or 2. Legacy configurations use indices 3, 4, and 5 and require
only their saved phase-mask pickle.

## Numerical sampling

`run_ga.py` prints the requested and effective computational grid, followed by
two dimensionless checks for the initial geometry:

- `TF ratio = lambda*|z|/(L*dx)` should not exceed 1 for the transfer-function
  chirp to meet the stated sampling condition.
- `lens ratio = (|f|/(2R))/(dx/lambda)` should be at least 1 for a finite lens.

For the 128 x 128, 20 micrometre, 780 nm defaults, the critical Fresnel
distance is about 6.56 cm. The supplied 3--6.5 cm bounds remain below it, and
the tightest supplied lens sampling ratio remains above one. A padding factor
of 2 raises the transfer-function critical distance to about 13.13 cm because
the computational side length doubles while the pixel pitch stays fixed.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe run_ga.py --config configs\ga_smoke.yaml
```

The tests cover Fresnel power conservation, centered padding and cropping, the
exact thin-lens phase, the front-to-back focal-plane Fourier limit, all three
physical stages, the exact historical one- and two-plane FFT formulas, the
three-plane legacy FFT extension, geometry encoding, detector placement,
sampling diagnostics, and balanced detector-throughput fitness.
