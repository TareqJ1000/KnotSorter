# -*- coding: utf-8 -*-
"""Optimize phase-plane sorters for optical knots.

The preferred architecture explicitly propagates each field through

    phase plate -> free space -> thin lens -> free space

for every phase plane. The free-space distances and focal lengths may be fixed
or appended to the genetic chromosome as bounded optimization parameters. A
legacy FFT architecture remains available for existing masks and supports any
positive number of alternating Fourier-conjugate phase planes.
"""

import argparse
import ast
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
    map_legacy_plane_to_padded_grid,
    norm_field,
    oamModes,
    output_chan_circle,
    padded_grid_size,
    propFF,
    propTF,
    propagate_fresnel_lens_train,
    propagate_legacy_fft,
    propagate_legacy_fft_padded,
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
mirror = cnfg.get("mirror", [False]*len(knotType))
if isinstance(mirror, bool):
    mirror = [mirror]*len(knotType)
if not isinstance(mirror, list) or any(
        not isinstance(value, bool) for value in mirror):
    raise ValueError("mirror must be a boolean or a list of booleans.")

num_phase_maps_near = int(cnfg.get("num_phase_maps_near", 0))
num_phase_maps_far = int(cnfg.get("num_phase_maps_far", 0))
num_of_phase_maps = int(
    cnfg.get("num_phase_planes", num_phase_maps_near+num_phase_maps_far)
)
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
    if not (
        len(knotType) == len(shapeParams) == len(mirror)
        == num_of_output_chans
    ):
        raise ValueError(
            "knotType, shapeParams, mirror, and the output-channel count "
            "must have matching lengths."
        )
    for knot_name, parameters, is_mirrored, channel in zip(
            knotType, shapeParams, mirror, output_chans):
        list_of_OAMs.append(oamModes(
            setKnotType(
                r, phi, w0, knot_name, parameters, mirror=is_mirrored
            ),
            channel,
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
        for knot_name, parameters, is_mirrored, channel in zip(
                knotType, shapeParams, mirror, output_chans):
            rotated.append(oamModes(
                setKnotType(
                    r_rot, phi_rot, w0, knot_name, parameters,
                    mirror=is_mirrored,
                ),
                channel,
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

    # Preserve the historical one- and two-plane FFT formulas exactly, then
    # extend their alternating forward/inverse convention to a third plane.
    if not multiPhaseLens and not simulateLens:
        return propagate_legacy_fft(field, phase_maps)

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


_legacy_refinement_channel_cache = {}


def _legacy_refinement_channels(padding_factor):
    """Return physically mapped detector masks on a padded legacy grid."""
    padding_factor = int(padding_factor)
    if padding_factor not in _legacy_refinement_channel_cache:
        output_is_fourier_plane = bool(num_of_phase_maps % 2)
        _legacy_refinement_channel_cache[padding_factor] = np.asarray([
            map_legacy_plane_to_padded_grid(
                channel, padding_factor,
                fourier_plane=output_is_fourier_plane,
                fill_value=0.0,
            )
            for channel in output_chans
        ])
    return _legacy_refinement_channel_cache[padding_factor]


def compute_sorting_performance(phase_maps, input_modes, stages, alpha=None,
                                padding_factor=None):
    """Return conditional sorting metrics and absolute detector throughput."""
    if alpha is None:
        alpha = float(cnfg["alpha"])
    d = len(input_modes)
    if d < 2:
        raise ValueError("Sorting requires at least two input modes.")

    efficiency_matrix = np.zeros((d, num_of_output_chans), dtype=float)
    refinement_mode = padding_factor is not None
    evaluation_channels = output_chans
    kernels = None
    if optical_train.model == "fresnel_lens_train":
        effective_padding = (
            optical_train.padding_factor
            if padding_factor is None else float(padding_factor)
        )
        kernels = build_fresnel_lens_kernels(
            (N, N), grid_side_length, wavelength, stages, r,
            lens_radius=optical_train.lens_aperture_radius,
            padding_factor=effective_padding,
        )
    elif refinement_mode:
        evaluation_channels = _legacy_refinement_channels(padding_factor)

    for input_index, input_mode in enumerate(input_modes):
        field = norm_field(input_mode.oamBeam, h)
        input_power = np.sum(np.abs(field)**2)

        if optical_train.model == "fresnel_lens_train":
            final_field = propagate_fresnel_lens_train(
                field, phase_maps, grid_side_length, wavelength, stages, r,
                lens_radius=optical_train.lens_aperture_radius,
                kernels=kernels,
                padding_factor=effective_padding,
            )
        elif refinement_mode:
            final_field = propagate_legacy_fft_padded(
                field, phase_maps, padding_factor=padding_factor
            )
        else:
            final_field = propagate_legacy(field, phase_maps)
            # The unscaled FFT legacy path is normalized to preserve its former
            # interpretation. Fresnel propagation is already power preserving;
            # importantly, it retains real loss from a finite lens pupil.
            final_field = norm_field(final_field, h)

        final_intensity = np.abs(final_field)**2
        for output_index, channel in enumerate(evaluation_channels):
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
    secret_key = max(
        0.0,
        np.log2(d)-2*shannon_entropy(symbol_error, d),
    )
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

    throughput_factor = (
        1.0 if throughput_exponent == 0.0
        else throughput**throughput_exponent
    )

    return float(np.real(sorting_performance*throughput_factor))


def fitness_func_secret_key(ga_instance, solution, solution_idx):
    _, phase_maps, stages = decode_solution(solution)
    input_modes = create_rotated_modes(_rotation_angle_for_fitness())
    sorting_performance, _, secret_key, _, _, _ = compute_sorting_performance(
        phase_maps, input_modes, stages
    )
    return float(np.real(sorting_performance*secret_key))


def _full_metric_value(performance_metrics):
    sorting_performance, _, secret_key, assignment_matrix, throughput, _ = (
        performance_metrics
    )
    distinguishability = abs(np.linalg.det(assignment_matrix))**float(cnfg["gamma"])
    throughput_factor = (
        1.0 if throughput_exponent == 0.0
        else throughput**throughput_exponent
    )
    return float(np.real(
        sorting_performance*secret_key*distinguishability*throughput_factor
    ))


def fitness_func_full(ga_instance, solution, solution_idx):
    _, phase_maps, stages = decode_solution(solution)
    input_modes = create_rotated_modes(_rotation_angle_for_fitness())
    return _full_metric_value(
        compute_sorting_performance(phase_maps, input_modes, stages)
    )


def fitness_func_padded_refinement(ga_instance, solution, solution_idx):
    """Evaluate the full metric on the configured padded propagation grid."""
    _, phase_maps, stages = decode_solution(solution)
    input_modes = create_rotated_modes(_rotation_angle_for_fitness())
    return _full_metric_value(compute_sorting_performance(
        phase_maps, input_modes, stages,
        padding_factor=refinement_padding_factor,
    ))


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

num_generations = int(float(_number(cnfg["num_of_gens"])))
gen_start = int(float(_number(cnfg["gen_start"])))
refinement_generations = int(float(_number(cnfg.get(
    "refinement_generations", cnfg.get("refinement_epochs", 0)
))))
refinement_padding_factor = float(cnfg.get("refinement_padding_factor", 2.0))
raw_refinement_saturate = cnfg.get("refinement_saturate")
refinement_saturate = (
    None if raw_refinement_saturate is None
    else int(float(_number(raw_refinement_saturate)))
)

_START_STAGE_ALIASES = {
    "warm_up": "warm_up",
    "warm-up": "warm_up",
    "warmup": "warm_up",
    "full": "full",
    "padded_refinement": "padded_refinement",
    "padded-refinement": "padded_refinement",
    "padded refinement": "padded_refinement",
    "refinement": "padded_refinement",
}
raw_start_stage = str(cnfg.get("start_stage", "warm_up")).strip().lower()
try:
    start_stage = _START_STAGE_ALIASES[raw_start_stage]
except KeyError as error:
    raise ValueError(
        "start_stage must be 'warm_up', 'full', or 'padded_refinement'."
    ) from error
seed_from = cnfg.get("seed_from")
if isinstance(seed_from, str):
    seed_from = seed_from.strip() or None
if start_stage != "warm_up" and seed_from is None:
    raise ValueError(
        f"start_stage={start_stage!r} requires a seed_from checkpoint."
    )
if refinement_generations < 0:
    raise ValueError("refinement_generations cannot be negative.")
if not np.isfinite(refinement_padding_factor) or refinement_padding_factor < 1:
    raise ValueError("refinement_padding_factor must be finite and at least 1.")
if (optical_train.model == "legacy_fft"
        and int(refinement_padding_factor) != refinement_padding_factor):
    raise ValueError(
        "Legacy FFT refinement_padding_factor must be a positive integer."
    )
if (refinement_generations > 0
        and optical_train.model == "fresnel_lens_train"
        and refinement_padding_factor < optical_train.padding_factor):
    raise ValueError(
        "Fresnel refinement_padding_factor cannot be smaller than "
        "optical_train.padding_factor."
    )
if refinement_saturate is not None and refinement_saturate < 1:
    raise ValueError("refinement_saturate must be null or a positive integer.")
if start_stage == "padded_refinement" and refinement_generations < 1:
    raise ValueError(
        "start_stage='padded_refinement' requires "
        "refinement_generations to be positive."
    )
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


def _resolve_phase_checkpoint(checkpoint):
    """Resolve a checkpoint name or path to a best_phases pickle."""
    requested = Path(str(checkpoint))
    requested = (
        requested if requested.suffix.lower() == ".pkl"
        else requested.with_suffix(".pkl")
    )
    if requested.is_absolute() or requested.parent != Path("."):
        candidates = [requested]
    else:
        candidates = [Path("best_phases")/requested, requested]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find seed_from checkpoint {checkpoint!r}; searched {searched}."
    )


def _checkpoint_phase_angles(checkpoint_path):
    """Load decoded phase angles from a historical or structured checkpoint."""
    with checkpoint_path.open("rb") as file:
        checkpoint_data = pkl.load(file)

    if isinstance(checkpoint_data, dict):
        for key in ("phase_angles", "phases", "phase_maps"):
            if key in checkpoint_data:
                checkpoint_data = checkpoint_data[key]
                break
        else:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is a mapping but contains no "
                "phase_angles, phases, or phase_maps entry."
            )

    checkpoint_array = np.asarray(checkpoint_data)
    if np.iscomplexobj(checkpoint_array):
        checkpoint_array = np.angle(checkpoint_array)
    checkpoint_array = np.asarray(checkpoint_array, dtype=float)
    expected_shape = (num_of_phase_maps, N, N)
    if checkpoint_array.shape != expected_shape:
        if checkpoint_array.size == phase_gene_count:
            checkpoint_array = checkpoint_array.reshape(expected_shape)
        else:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has phase shape "
                f"{checkpoint_array.shape}; expected {expected_shape}."
            )
    if not np.all(np.isfinite(checkpoint_array)):
        raise ValueError(f"Checkpoint {checkpoint_path} contains non-finite phases.")
    return checkpoint_array


def _recover_phase_genes(phase_angles):
    """Invert the decode-time Gaussian filter so the seed is reproduced once."""
    if gaussian_filter_sigma_pixels <= 0:
        return phase_angles.reshape(-1).copy(), 0.0

    # scipy.ndimage.gaussian_filter is separable.  Building its exact 1-D
    # reflect-boundary matrix lets us recover a chromosome whose decoded masks
    # equal the saved (already-smoothed) best_phases masks, avoiding a second
    # Gaussian blur when the optimization resumes.
    filter_matrix = sp.ndimage.gaussian_filter1d(
        np.eye(N), sigma=gaussian_filter_sigma_pixels, axis=0
    )
    recovered = np.empty_like(phase_angles)
    maximum_error = 0.0
    for plane_index, target in enumerate(phase_angles):
        raw = sp.linalg.solve(filter_matrix, target, assume_a="gen")
        raw = sp.linalg.solve(filter_matrix, raw.T, assume_a="gen").T
        reconstructed = sp.ndimage.gaussian_filter(
            raw, sigma=gaussian_filter_sigma_pixels
        )
        error = float(np.max(np.abs(reconstructed-target)))
        tolerance = 1e-8*max(1.0, float(np.max(np.abs(target))))
        if error > tolerance:
            raise ValueError(
                f"Could not reconstruct phase plane {plane_index+1} from "
                f"checkpoint after Gaussian filtering (max error {error:.3g})."
            )
        recovered[plane_index] = raw
        maximum_error = max(maximum_error, error)
    return recovered.reshape(-1), maximum_error


def _checkpoint_geometry_genes(checkpoint_path):
    """Recover normalized geometry genes from the checkpoint YAML sidecar."""
    metadata_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_geometry.yaml"
    )
    if not metadata_path.is_file():
        if optical_train.num_geometry_genes:
            raise FileNotFoundError(
                f"Optimized geometry requires checkpoint sidecar {metadata_path}."
            )
        if optical_train.model == "fresnel_lens_train":
            print(
                "WARNING: checkpoint geometry sidecar is missing; using the "
                "fixed optical_train geometry from the current YAML."
            )
        return np.empty(0, dtype=float), None

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    checkpoint_model = metadata.get("model")
    if checkpoint_model is not None and checkpoint_model != optical_train.model:
        raise ValueError(
            f"Checkpoint model {checkpoint_model!r} does not match current "
            f"model {optical_train.model!r}."
        )
    checkpoint_plane_count = metadata.get("num_phase_planes")
    if (checkpoint_plane_count is not None
            and int(checkpoint_plane_count) != num_of_phase_maps):
        raise ValueError(
            f"Checkpoint has {checkpoint_plane_count} phase planes; current "
            f"configuration has {num_of_phase_maps}."
        )

    metadata_stages = metadata.get("stages", [])
    if optical_train.model == "fresnel_lens_train" and (
            len(metadata_stages) != num_of_phase_maps):
        raise ValueError(
            f"Checkpoint geometry contains {len(metadata_stages)} stages; "
            f"expected {num_of_phase_maps}."
        )

    yaml_key = {
        "z_to_lens": "z_to_lens_cm",
        "focal_length": "focal_length_cm",
        "z_after_lens": "z_after_lens_cm",
    }
    if not optical_train.num_geometry_genes:
        if optical_train.model == "fresnel_lens_train":
            current_stages = optical_train.decode_geometry()
            for stage_index, (current, saved) in enumerate(zip(
                    current_stages, metadata_stages)):
                for name, key in yaml_key.items():
                    saved_value = float(saved[key])*CM
                    if not np.isclose(
                            current[name], saved_value, rtol=1e-9, atol=1e-12):
                        raise ValueError(
                            f"Fixed geometry mismatch at stage {stage_index+1} "
                            f"{key}: YAML={current[name]/CM:g} cm, "
                            f"checkpoint={saved_value/CM:g} cm."
                        )
        return np.empty(0, dtype=float), metadata_path.resolve()

    geometry_genes = []
    for parameter in optical_train.geometry_parameters:
        key = yaml_key[parameter.name]
        try:
            physical_value = (
                float(metadata_stages[parameter.stage_index][key])*CM
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Checkpoint geometry is missing a numeric stage "
                f"{parameter.stage_index+1} {key}."
            ) from error
        tolerance = max(
            1e-12, (parameter.maximum-parameter.minimum)*1e-9
        )
        if not (
            parameter.minimum-tolerance
            <= physical_value
            <= parameter.maximum+tolerance
        ):
            raise ValueError(
                f"Checkpoint stage {parameter.stage_index+1} {key}="
                f"{physical_value/CM:g} cm lies outside the current bounds "
                f"[{parameter.minimum/CM:g}, {parameter.maximum/CM:g}] cm."
            )
        geometry_genes.append(
            (physical_value-parameter.minimum)
            /(parameter.maximum-parameter.minimum)
        )
    return np.clip(geometry_genes, 0.0, 1.0), metadata_path.resolve()


seed_checkpoint_path = None
seed_geometry_path = None
seed_reconstruction_error = None
if seed_from is not None:
    seed_checkpoint_path = _resolve_phase_checkpoint(seed_from)
    seed_phase_angles = _checkpoint_phase_angles(seed_checkpoint_path)
    seed_phase_genes, seed_reconstruction_error = _recover_phase_genes(
        seed_phase_angles
    )
    seed_geometry_genes, seed_geometry_path = _checkpoint_geometry_genes(
        seed_checkpoint_path
    )
    seed_solution = np.concatenate((seed_phase_genes, seed_geometry_genes))
    if seed_solution.size != num_genes:
        raise ValueError(
            f"Checkpoint produced {seed_solution.size} genes; expected {num_genes}."
        )
    initial_population[0] = seed_solution


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
    stage_name = getattr(ga_instance, "stage_name", "optimization")

    Path("best_phases").mkdir(exist_ok=True)
    Path("genetic_instances").mkdir(exist_ok=True)
    plot_dir = Path("plots")/instance_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    with (Path("best_phases")/f"{instance_name}.pkl").open("wb") as file:
        pkl.dump(phase_angles, file)
    with (Path("best_phases")/f"{instance_name}_geometry.yaml").open(
            "w", encoding="utf-8") as file:
        metadata = optical_train.metadata(stages)
        metadata["optimization_stage"] = stage_name
        if stage_name == "padded refinement":
            metadata["refinement_padding_factor"] = (
                refinement_padding_factor
            )
        yaml.safe_dump(metadata, file, sort_keys=False)
    ga_instance.save(filename=f"genetic_instances/{instance_name}")

    print(
        f"{stage_name} generation {ga_instance.generations_completed}: "
        f"best fitness {fitness:.8g}"
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
    mode_names = (
        [f"{name} (mirror={is_mirrored})"
         for name, is_mirrored in zip(knotType, mirror)]
        if isKnot else LG_modes
    )
    print("\n"+"="*80)
    print("KNOT SORTER CONFIGURATION")
    print("="*80)
    print(f"Configuration: {config_path}")
    print(f"Inputs: {mode_names}")
    print(f"Grid: {N} x {N}, pitch={pixel_pitch/UM:.3f} um, wavelength={wavelength/NM:.1f} nm")
    print(f"Phase planes: {num_of_phase_maps}")
    print(f"Propagation model: {optical_train.model}")
    print(f"Genes: {num_genes} ({phase_gene_count} phase + {optical_train.num_geometry_genes} geometry)")
    print(f"Start stage: {start_stage}")
    if seed_checkpoint_path is None:
        print("Seed checkpoint: none")
    else:
        print(f"Seed checkpoint: {seed_checkpoint_path}")
        print(
            "Seed phase reconstruction error after decode: "
            f"{seed_reconstruction_error:.3g} rad"
        )
        if seed_geometry_path is not None:
            print(f"Seed geometry: {seed_geometry_path}")

    displayed_geometry = (
        initial_population[0, phase_gene_count:]
        if optical_train.num_geometry_genes else None
    )
    initial_stages = optical_train.decode_geometry(displayed_geometry)
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
    if refinement_generations:
        if optical_train.model == "legacy_fft":
            refinement_size = N*int(refinement_padding_factor)
        else:
            refinement_size = padded_grid_size(
                N, refinement_padding_factor
            )
        saturation_text = (
            "disabled" if refinement_saturate is None
            else str(refinement_saturate)
        )
        print(
            f"Padded refinement: {refinement_generations} generations, "
            f"factor={refinement_padding_factor:g}, "
            f"grid={refinement_size} x {refinement_size}, "
            f"saturation={saturation_text}, metric=full"
        )
    else:
        print("Padded refinement: disabled")
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
    if refinement_generations:
        refinement_metrics = compute_sorting_performance(
            validation_maps, list_of_OAMs, validation_stages,
            padding_factor=refinement_padding_factor,
        )
        print("Padded-refinement validation efficiency matrix:")
        print(refinement_metrics[1])
        print("Padded-refinement validation assignment matrix:")
        print(refinement_metrics[3])
        print("Padded-refinement accepted detector efficiency per input:")
        print(refinement_metrics[5])
        print(
            "Padded-refinement balanced throughput: "
            f"{refinement_metrics[4]:.8g}"
        )
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

ga_instance_sorting = None
ga_instance_full = None
ga_instance_refinement = None

if start_stage == "warm_up":
    ga_instance_sorting = pygad.GA(
        num_generations=gen_start,
        fitness_func=fitness_func_sorting,
        initial_population=initial_population,
        on_stop=on_stop,
        **common_ga_arguments,
    )
    ga_instance_sorting.stage_name = "warm-up"
    ga_instance_sorting.run()
    full_initial_population = last_pop
elif start_stage == "full":
    full_initial_population = initial_population

if start_stage in {"warm_up", "full"}:
    ga_instance_full = pygad.GA(
        num_generations=num_generations,
        fitness_func=full_fitness,
        initial_population=full_initial_population,
        stop_criteria=f"saturate_{gen_saturate}",
        **common_ga_arguments,
    )
    ga_instance_full.stage_name = "full"
    ga_instance_full.run()

if refinement_generations and start_stage != "padded_refinement":
    refinement_population = ga_instance_full.population.copy()
    best_full_solution, _, _ = ga_instance_full.best_solution()
    # Guarantee that the best native-grid candidate survives the transition
    # even if it is not present in the final population ordering.
    refinement_population[0] = best_full_solution
elif start_stage == "padded_refinement":
    refinement_population = initial_population.copy()

if refinement_generations:
    refinement_arguments = dict(common_ga_arguments)
    if refinement_saturate is not None:
        refinement_arguments["stop_criteria"] = (
            f"saturate_{refinement_saturate}"
        )
    ga_instance_refinement = pygad.GA(
        num_generations=refinement_generations,
        fitness_func=fitness_func_padded_refinement,
        initial_population=refinement_population,
        **refinement_arguments,
    )
    ga_instance_refinement.stage_name = "padded refinement"
    ga_instance_refinement.run()
