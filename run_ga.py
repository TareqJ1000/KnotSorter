# -*- coding: utf-8 -*-
"""Optimize one-, two-, or three-plane phase sorters for optical knots.

The preferred architecture explicitly propagates each field through

    phase plate -> free space -> thin lens -> free space

for every phase plane. The free-space distances and focal lengths may be fixed
or appended to the genetic chromosome as bounded optimization parameters. A
legacy FFT architecture remains available for existing masks and configurations.
"""

import argparse
import ast
import time
from pathlib import Path
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pygad
import scipy as sp
import yaml
from scipy.fft import fft2, fftshift, ifft2, ifftshift

from optical_functions import (
    LG,
    balanced_detector_throughput,
    build_fresnel_lens_kernels,
    cart2pol,
    fresnel_sampling_diagnostics,
    norm_field,
    oamModes,
    output_chan_circle,
    padded_grid_size,
    propFF,
    propTF,
    propagate_fresnel_lens_train,
    setKnotType,
    shannon_entropy,
)
from sorter_configuration import parse_optical_train_config


CM = 1e-2
MM = 1e-3
UM = 1e-6
NM = 1e-9


def _number(value):
    """Parse historical numeric strings without using eval()."""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _append_json_line(path, payload):
    """Append one compact JSON object as a line to ``path`` (NDJSON)."""
    import json
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload)+"\n")


_START_TIME = None


def _generation_cost():
    """Deterministic per-generation compute proxy (phase evaluations)."""
    import numpy as _np
    return int(
        num_of_phase_maps*len(list_of_OAMs)
        * (padded_grid_size(N, optical_train.padding_factor)**2)
        * sol_per_pop
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ii", type=int, default=None,
                    help="Load configs/ga<ii>.yaml (gaNone.yaml when omitted).")
parser.add_argument("--config", type=Path,
                    help="Load an explicit YAML file instead of configs/ga<ii>.yaml.")
parser.add_argument("--validate-only", action="store_true",
                    help="Build the optical system, print diagnostics, and exit.")
args = parser.parse_args()

if args.config is not None:
    config_path = args.config
else:
    config_name = "gaNone.yaml" if args.ii is None else f"ga{args.ii}.yaml"
    config_path = Path("configs")/config_name

with config_path.open("r", encoding="utf-8") as stream:
    cnfg = yaml.safe_load(stream)

# Backward-compatible defaults for archived configurations.
cnfg.setdefault("circle_radius", 1.5)
cnfg.setdefault("fitness_func", "secret_key")
cnfg.setdefault("alpha", 1.0)
cnfg.setdefault("gamma", 1.0)
cnfg.setdefault("throughput_metric", "geometric_mean")
cnfg.setdefault("throughput_exponent", 1.0)
cnfg.setdefault("secret_key_softplus", False)
cnfg.setdefault("keep_elitism", 1)
cnfg.setdefault("random_mutation_min_val", -1.0)
cnfg.setdefault("random_mutation_max_val", 1.0)
cnfg.setdefault("pixel_pitch_um", 20.0)
cnfg.setdefault("wavelength_nm", 780.0)
cnfg.setdefault("random_seed", None)

throughput_metric = str(cnfg["throughput_metric"])
throughput_exponent = float(cnfg["throughput_exponent"])
if throughput_metric not in {"geometric_mean", "minimum", "arithmetic_mean"}:
    raise ValueError(
        "throughput_metric must be 'geometric_mean', 'minimum', "
        "or 'arithmetic_mean'."
    )
if not np.isfinite(throughput_exponent) or throughput_exponent < 0:
    raise ValueError("throughput_exponent must be finite and non-negative.")

secret_key_softplus = bool(cnfg["secret_key_softplus"])


# ---------------------------------------------------------------------------
# Field, mode, and optical-system configuration
# ---------------------------------------------------------------------------

N = int(cnfg["dim"])
num_of_output_chans = int(cnfg["num_output_chans"])
output_chan_width = float(cnfg["output_chan_width"])*MM
channel_sep = float(cnfg["channel_sep"])
circle_radius = float(cnfg["circle_radius"])

LG_modes = cnfg["LG_modes"]
w0 = float(cnfg["w0"])*MM
isKnot = bool(cnfg["isKnot"])
knotType = cnfg["knotType"]
shapeParams = cnfg["shapeParams"]

num_phase_maps_near = int(cnfg.get("num_phase_maps_near", 0))
num_phase_maps_far = int(cnfg.get("num_phase_maps_far", 0))
num_of_phase_maps = int(
    cnfg.get("num_phase_planes", num_phase_maps_near+num_phase_maps_far)
)
if not 1 <= num_of_phase_maps <= 3:
    raise ValueError("The sorter supports one, two, or three phase planes.")

optical_train = parse_optical_train_config(cnfg, num_of_phase_maps)

# Legacy propagation settings.
simulateLens = bool(cnfg.get("simulateLens", False))
fourier_lens = float(cnfg.get("fourier_length", 10.0))*CM
multiPhaseLens = bool(cnfg.get("multiPhaseLens", False))
multiPhase = bool(cnfg.get("multiPhase", False))
z_o = float(cnfg.get("z_o", 30.0))*CM

rot_angle = float(_number(cnfg.get("rot_angle", "0")))
fixedRotation = bool(cnfg.get("fixedRotation", True))
randomRotation = bool(cnfg.get("randomRotation", False))

wavelength = float(cnfg["wavelength_nm"])*NM
k = 2*np.pi/wavelength
pixel_pitch = float(cnfg["pixel_pitch_um"])*UM
grid_side_length = pixel_pitch*N
h = pixel_pitch

X = pixel_pitch*(np.arange(N)-N//2)
Y = pixel_pitch*(np.arange(N)-N//2)
xx, yy = np.meshgrid(X, Y)
r, phi = cart2pol(xx, yy)

output_chans = output_chan_circle(
    X, Y, output_chan_width, grid_side_length, num_of_output_chans,
    circle_radius=circle_radius,
    coordinate_mode=optical_train.output_coordinate_mode,
)

list_of_OAMs = []
if isKnot:
    if len(knotType) != num_of_output_chans:
        raise ValueError("The knot alphabet and output-channel count must match.")
    for knot_name, parameters, channel in zip(knotType, shapeParams, output_chans):
        list_of_OAMs.append(oamModes(
            setKnotType(r, phi, w0, knot_name, parameters), channel
        ))
else:
    if len(LG_modes) != num_of_output_chans:
        raise ValueError("The LG alphabet and output-channel count must match.")
    for (ell, radial_index), channel in zip(LG_modes, output_chans):
        list_of_OAMs.append(oamModes(
            LG(r, phi, ell, radial_index, w0, h, 0, k), channel
        ))


def create_rotated_modes(rotation_angle):
    """Recreate the complete input alphabet on rotated coordinates."""
    xx_rot = np.cos(rotation_angle)*xx-np.sin(rotation_angle)*yy
    yy_rot = np.sin(rotation_angle)*xx+np.cos(rotation_angle)*yy
    r_rot, phi_rot = cart2pol(xx_rot, yy_rot)

    rotated = []
    if isKnot:
        for knot_name, parameters, channel in zip(
                knotType, shapeParams, output_chans):
            rotated.append(oamModes(
                setKnotType(r_rot, phi_rot, w0, knot_name, parameters), channel
            ))
    else:
        for (ell, radial_index), channel in zip(LG_modes, output_chans):
            rotated.append(oamModes(
                LG(r_rot, phi_rot, ell, radial_index, w0, h, 0, k), channel
            ))
    return rotated


# ---------------------------------------------------------------------------
# Genetic representation: phase pixels followed by normalized geometry genes
# ---------------------------------------------------------------------------

GFilterStrength = float(cnfg["gauss_filter_sigma"])
gaussian_filter_sigma_pixels = float(
    cnfg.get("gaussian_filter_sigma_pixels", grid_side_length*GFilterStrength)
)

phase_gene_count = num_of_phase_maps*N**2
num_genes = phase_gene_count+optical_train.num_geometry_genes


def decode_solution(solution):
    """Decode a chromosome into smoothed phase masks and physical geometry."""
    phase_angles = np.empty((num_of_phase_maps, N, N), dtype=float)
    phase_maps = np.empty((num_of_phase_maps, N, N), dtype=np.complex128)
    for index in range(num_of_phase_maps):
        start = index*N**2
        stop = (index+1)*N**2
        angle = np.reshape(solution[start:stop], (N, N))
        angle = sp.ndimage.gaussian_filter(
            angle, sigma=gaussian_filter_sigma_pixels
        )
        phase_angles[index] = angle
        phase_maps[index] = np.exp(1j*angle)

    geometry_genes = solution[phase_gene_count:]
    stages = optical_train.decode_geometry(geometry_genes)
    return phase_angles, phase_maps, stages


def propagate_legacy(field, phase_maps):
    """Preserve the original FFT/multi-plane behavior for old configurations."""
    field_mod_1 = field*phase_maps[0]

    if multiPhase:
        field_after = field_mod_1
        for phase_map in phase_maps[1:]:
            field_after = propTF(
                field_after, grid_side_length, wavelength, z_o
            )*phase_map
        return propTF(field_after, grid_side_length, wavelength, z_o)

    if multiPhaseLens:
        field_after = field_mod_1
        for index in range(1, num_phase_maps_near):
            field_after = propTF(
                field_after, grid_side_length, wavelength, z_o
            )*phase_maps[index]
        field_lens = fftshift(fft2(field_after))
    elif simulateLens:
        field_lens, _ = propFF(
            field_mod_1, grid_side_length, wavelength, fourier_lens
        )
    else:
        field_lens = fftshift(fft2(field_mod_1))

    if num_phase_maps_far == 0:
        return field_lens

    field_after_2 = field_lens*phase_maps[num_phase_maps_near]
    if multiPhaseLens:
        for index in range(num_phase_maps_near+1, num_of_phase_maps):
            field_after_2 = propTF(
                field_after_2, grid_side_length, wavelength, z_o
            )*phase_maps[index]
        return ifft2(ifftshift(field_after_2))
    if simulateLens:
        field_lens_2, _ = propFF(
            field_after_2, grid_side_length, wavelength, fourier_lens
        )
        return field_lens_2
    return ifft2(ifftshift(field_after_2))


def compute_sorting_performance(phase_maps, input_modes, stages, alpha=None):
    """Return conditional sorting metrics and absolute detector throughput."""
    if alpha is None:
        alpha = float(cnfg["alpha"])
    d = len(input_modes)
    if d < 2:
        raise ValueError("Sorting requires at least two input modes.")

    efficiency_matrix = np.zeros((d, num_of_output_chans), dtype=float)
    kernels = None
    if optical_train.model == "fresnel_lens_train":
        kernels = build_fresnel_lens_kernels(
            (N, N), grid_side_length, wavelength, stages, r,
            lens_radius=optical_train.lens_aperture_radius,
            padding_factor=optical_train.padding_factor,
        )

    for input_index, input_mode in enumerate(input_modes):
        field = norm_field(input_mode.oamBeam, h)
        input_power = np.sum(np.abs(field)**2)

        if optical_train.model == "fresnel_lens_train":
            final_field = propagate_fresnel_lens_train(
                field, phase_maps, grid_side_length, wavelength, stages, r,
                lens_radius=optical_train.lens_aperture_radius,
                kernels=kernels,
                padding_factor=optical_train.padding_factor,
            )
        else:
            final_field = propagate_legacy(field, phase_maps)
            # The unscaled FFT legacy path is normalized to preserve its former
            # interpretation. Fresnel propagation is already power preserving;
            # importantly, it retains real loss from a finite lens pupil.
            final_field = norm_field(final_field, h)

        final_intensity = np.abs(final_field)**2
        for output_index, channel in enumerate(output_chans):
            efficiency_matrix[input_index, output_index] = np.real(
                np.sum(final_intensity*channel)/input_power
            )

    accepted_power = efficiency_matrix.sum(axis=1, keepdims=True)
    assignment_matrix = np.divide(
        efficiency_matrix,
        accepted_power,
        out=np.zeros_like(efficiency_matrix),
        where=accepted_power > 0,
    )
    balanced_throughput, accepted_efficiencies = balanced_detector_throughput(
        efficiency_matrix, method=throughput_metric
    )

    correct = np.diag(assignment_matrix)
    incorrect_mean = (
        assignment_matrix.sum(axis=1)-correct
    )/(d-1)
    channel_contrast = correct-incorrect_mean
    balanced_contrast = (
        alpha*np.min(channel_contrast)
        +(1-alpha)*np.mean(channel_contrast)
    )

    symbol_error = 1-np.mean(correct)
    key_rate = np.log2(d)-2*shannon_entropy(symbol_error, d)
    if secret_key_softplus:
        # Smooth softplus(t)=log(1+exp(t)) gives sub-threshold designs
        # (key_rate <= 0, i.e. <=50% mean correct) a small but nonzero
        # gradient instead of the hard max(0,.) plateau at zero. This lets
        # the GA climb out of the flat sub-threshold regime.
        secret_key = float(np.log1p(np.exp(np.clip(key_rate, -50.0, 50.0))))
    else:
        secret_key = max(0.0, key_rate)
    return (
        balanced_contrast,
        efficiency_matrix,
        secret_key,
        assignment_matrix,
        balanced_throughput,
        accepted_efficiencies,
    )


def _rotation_angle_for_fitness():
    if randomRotation:
        return np.random.uniform(0, 2*np.pi)
    if fixedRotation:
        return rot_angle
    return 0.0


def fitness_func_sorting(ga_instance, solution, solution_idx):
    _, phase_maps, stages = decode_solution(solution)
    input_modes = create_rotated_modes(_rotation_angle_for_fitness())
    sorting_performance, _, _, _, throughput, _ = compute_sorting_performance(
        phase_maps, input_modes, stages
    )

    throughput_factor = throughput**throughput_exponent

    return float(np.real(sorting_performance*throughput_factor))


def fitness_func_secret_key(ga_instance, solution, solution_idx):
    _, phase_maps, stages = decode_solution(solution)
    input_modes = create_rotated_modes(_rotation_angle_for_fitness())
    sorting_performance, _, secret_key, _, _, _ = compute_sorting_performance(
        phase_maps, input_modes, stages
    )
    return float(np.real(sorting_performance*secret_key))


def fitness_func_full(ga_instance, solution, solution_idx):
    _, phase_maps, stages = decode_solution(solution)
    input_modes = create_rotated_modes(_rotation_angle_for_fitness())
    sorting_performance, _, secret_key, assignment_matrix, throughput, _ = (
        compute_sorting_performance(phase_maps, input_modes, stages)
    )
    distinguishability = abs(np.linalg.det(assignment_matrix))**float(cnfg["gamma"])
    throughput_factor = (
        1.0 if throughput_exponent == 0.0
        else throughput**throughput_exponent
    )
    return float(np.real(
        sorting_performance*secret_key*distinguishability*throughput_factor
    ))


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

num_generations = int(float(_number(cnfg["num_of_gens"])))
gen_start = int(float(_number(cnfg["gen_start"])))
num_parents_mating = int(cnfg["parents_mating"])
sol_per_pop = int(cnfg["sol_per_pop"])
parent_c = float(cnfg["parent_c"])
parent_k = float(cnfg["parent_k"])
crossover_type = "single_point"
crossover_probability = _number(cnfg.get("crossover_prob"))
mutation_type = cnfg["mutation_type"]
mutation_probability = _number(cnfg["mutation_prob"])
if isinstance(mutation_probability, list):
    mutation_probability = tuple(mutation_probability)
gen_saturate = int(cnfg["gen_saturate"])
keep_elitism = int(cnfg["keep_elitism"])

phase_mutation_min = float(cnfg["random_mutation_min_val"])*np.pi
phase_mutation_max = float(cnfg["random_mutation_max_val"])*np.pi
if phase_mutation_min > phase_mutation_max:
    raise ValueError("random_mutation_min_val must not exceed max_val.")

mutation_min = np.concatenate((
    np.full(phase_gene_count, phase_mutation_min),
    np.zeros(optical_train.num_geometry_genes),
))
mutation_max = np.concatenate((
    np.full(phase_gene_count, phase_mutation_max),
    np.zeros(optical_train.num_geometry_genes),
))

rng = np.random.default_rng(cnfg["random_seed"])
initial_population = np.empty((sol_per_pop, num_genes), dtype=float)
initial_population[:, :phase_gene_count] = rng.uniform(
    -np.pi, np.pi, size=(sol_per_pop, phase_gene_count)
)
if optical_train.num_geometry_genes:
    initial_population[:, phase_gene_count:] = rng.uniform(
        0.0, 1.0,
        size=(sol_per_pop, optical_train.num_geometry_genes),
    )
    # Always seed the YAML's initial physical layout as one candidate.
    initial_population[0, phase_gene_count:] = (
        optical_train.initial_normalized_geometry
    )


def mutate_geometry(ga_instance, offspring):
    """Mutate the small geometry block independently of the phase pixels."""
    if not optical_train.num_geometry_genes:
        return offspring

    geometry = offspring[:, phase_gene_count:]
    mutate = rng.random(geometry.shape) < optical_train.geometry_mutation_probability
    offsets = rng.uniform(
        -optical_train.geometry_mutation_scale,
        optical_train.geometry_mutation_scale,
        size=geometry.shape,
    )
    geometry += mutate*offsets
    np.clip(geometry, 0.0, 1.0, out=geometry)
    offspring[:, phase_gene_count:] = geometry
    return offspring


def exp_rank_selection(fitness, num_parents, ga_instance):
    """Exponential rank selection with the best individual at rank zero."""
    order = np.argsort(np.asarray(fitness))[::-1]
    ranks = np.arange(len(order), dtype=float)
    probabilities = parent_c*np.exp(-ranks/max(parent_k, 1e-12))
    probabilities /= probabilities.sum()
    selected_ranks = rng.choice(
        len(order), size=num_parents, replace=True, p=probabilities
    )
    selected_indices = order[selected_ranks]
    parents = ga_instance.population[selected_indices].copy()
    return parents, selected_indices


last_pop = None


def on_stop(ga_instance, last_population_fitness):
    global last_pop
    last_pop = ga_instance.population.copy()


def on_gen(ga_instance):
    solution, fitness, _ = ga_instance.best_solution()
    phase_angles, _, stages = decode_solution(solution)
    instance_name = cnfg["ga_instance"]

    Path("best_phases").mkdir(exist_ok=True)
    Path("genetic_instances").mkdir(exist_ok=True)
    plot_dir = Path("plots")/instance_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    with (Path("best_phases")/f"{instance_name}.pkl").open("wb") as file:
        pkl.dump(phase_angles, file)
    with (Path("best_phases")/f"{instance_name}_geometry.yaml").open(
            "w", encoding="utf-8") as file:
        yaml.safe_dump(
            optical_train.metadata(stages), file, sort_keys=False
        )
    ga_instance.save(filename=f"genetic_instances/{instance_name}")

    # Per-generation evolution-loss history (NDJSON). Record the best solution's
    # decomposed factors alongside the product fitness so curves stay
    # informative when the key-rate/throughput product gates to zero.
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    factors = {}
    try:
        best_phase_maps, best_stages = decode_solution(solution)[1:]
        best_input_modes = create_rotated_modes(_rotation_angle_for_fitness())
        fr = compute_sorting_performance(
            best_phase_maps, best_input_modes, best_stages
        )
        factors = {
            "contrast": float(np.real(fr[0])),
            "secret_key": float(np.real(fr[2])),
            "det_assignment": float(abs(np.linalg.det(fr[3]))),
            "throughput": float(np.real(fr[4])),
            "accepted_efficiencies": [float(x) for x in fr[5]],
        }
    except Exception:  # noqa: BLE001 - history capture must never kill the run
        factors = {}
    event = {
        "stage": _CURRENT_STAGE,
        "ga": _CURRENT_STAGE,
        "generation": ga_instance.generations_completed,
        "best_fitness": float(fitness),
        "wall_seconds": time.time()-_START_TIME,
        "cost_per_gen": _generation_cost(),
        "factors": factors,
    }
    _append_json_line(
        Path("best_phases")/f"{instance_name}_history.json", event
    )

    print(
        f"Generation {ga_instance.generations_completed}: best fitness {fitness:.8g}"
    )
    if ga_instance.generations_completed % 1000 == 0:
        plt.figure()
        plt.plot(ga_instance.best_solutions_fitness)
        plt.xlabel("Generation")
        plt.ylabel("Best fitness")
        plt.tight_layout()
        plt.savefig(
            plot_dir/f"fitness_{ga_instance.generations_completed}.jpg"
        )
        plt.close()


fitness_name = cnfg["fitness_func"]
if fitness_name == "secret_key":
    full_fitness = fitness_func_secret_key
elif fitness_name in {"bread", "full"}:
    full_fitness = fitness_func_full
else:
    raise ValueError("fitness_func must be 'secret_key', 'bread', or 'full'.")


def print_configuration():
    mode_names = knotType if isKnot else LG_modes
    print("\n"+"="*80)
    print("KNOT SORTER CONFIGURATION")
    print("="*80)
    print(f"Configuration: {config_path}")
    print(f"Inputs: {mode_names}")
    print(f"Grid: {N} x {N}, pitch={pixel_pitch/UM:.3f} um, wavelength={wavelength/NM:.1f} nm")
    print(f"Phase planes: {num_of_phase_maps}")
    print(f"Propagation model: {optical_train.model}")
    print(f"Genes: {num_genes} ({phase_gene_count} phase + {optical_train.num_geometry_genes} geometry)")

    initial_stages = optical_train.decode_geometry(
        optical_train.initial_normalized_geometry
        if optical_train.num_geometry_genes else None
    )
    if optical_train.model == "fresnel_lens_train":
        computational_size = padded_grid_size(
            N, optical_train.padding_factor
        )
        computational_length = pixel_pitch*computational_size
        lens_radius_text = (
            "unbounded" if optical_train.lens_aperture_radius is None
            else f"{optical_train.lens_aperture_radius/MM:.3f} mm"
        )
        print(f"Lens aperture radius: {lens_radius_text}")
        print(
            f"Padding: requested {optical_train.padding_factor:.3g}x, "
            f"effective {computational_size/N:.3g}x, "
            f"computational grid {computational_size} x {computational_size} "
            f"({computational_length/MM:.3f} mm side)"
        )
        critical = fresnel_sampling_diagnostics(
            grid_side_length, N, wavelength, propagation_distance=0,
            padding_factor=optical_train.padding_factor,
        )["critical_distance"]
        print(f"TF critical propagation distance: {critical/CM:.3f} cm")
        for index, stage in enumerate(initial_stages, start=1):
            before = fresnel_sampling_diagnostics(
                grid_side_length, N, wavelength,
                propagation_distance=stage["z_to_lens"],
                padding_factor=optical_train.padding_factor,
            )
            after = fresnel_sampling_diagnostics(
                grid_side_length, N, wavelength,
                propagation_distance=stage["z_after_lens"],
                padding_factor=optical_train.padding_factor,
            )
            message = (
                f"Stage {index}: z1={stage['z_to_lens']/CM:.3f} cm "
                f"(TF ratio {before['tf_ratio']:.3f}), "
                f"f={stage['focal_length']/CM:.3f} cm, "
                f"z2={stage['z_after_lens']/CM:.3f} cm "
                f"(TF ratio {after['tf_ratio']:.3f})"
            )
            if optical_train.lens_aperture_radius is not None:
                lens = fresnel_sampling_diagnostics(
                    grid_side_length, N, wavelength,
                    focal_length=stage["focal_length"],
                    lens_radius=optical_train.lens_aperture_radius,
                    padding_factor=optical_train.padding_factor,
                )
                message += f" (lens ratio {lens['lens_ratio']:.3f})"
            print(message)
    print(f"Fitness: {fitness_name}, alpha={cnfg['alpha']}, gamma={cnfg['gamma']}")
    print(
        f"Throughput factor: {throughput_metric}, "
        f"exponent={throughput_exponent:g}"
    )
    print("="*80+"\n")


print_configuration()

if args.validate_only:
    _, validation_maps, validation_stages = decode_solution(initial_population[0])
    metrics = compute_sorting_performance(
        validation_maps, list_of_OAMs, validation_stages
    )
    print("Validation candidate efficiency matrix:")
    print(metrics[1])
    print("Validation candidate assignment matrix:")
    print(metrics[3])
    print("Validation accepted detector efficiency per input:")
    print(metrics[5])
    print(f"Validation balanced throughput: {metrics[4]:.8g}")
    print(f"Validation secret_key: {metrics[2]:.8g} (softplus={secret_key_softplus})")
    raise SystemExit(0)


common_ga_arguments = dict(
    num_parents_mating=num_parents_mating,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=exp_rank_selection,
    crossover_type=crossover_type,
    crossover_probability=crossover_probability,
    mutation_type=mutation_type,
    mutation_probability=mutation_probability,
    random_mutation_min_val=mutation_min,
    random_mutation_max_val=mutation_max,
    on_mutation=mutate_geometry,
    on_generation=on_gen,
    keep_elitism=keep_elitism,
    random_seed=cnfg["random_seed"],
)

_CURRENT_STAGE = "sorting"
ga_instance_sorting = pygad.GA(
    num_generations=gen_start,
    fitness_func=fitness_func_sorting,
    initial_population=initial_population,
    on_stop=on_stop,
    **common_ga_arguments,
)
ga_instance_sorting.run()

_CURRENT_STAGE = "full"
ga_instance_full = pygad.GA(
    num_generations=num_generations,
    fitness_func=full_fitness,
    initial_population=last_pop,
    stop_criteria=f"saturate_{gen_saturate}",
    **common_ga_arguments,
)
ga_instance_full.run()
