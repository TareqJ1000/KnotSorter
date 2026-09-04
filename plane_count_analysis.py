"""Compare sorter detector efficiency and crosstalk across phase-plane counts."""

from pathlib import Path
import pickle

import numpy as np
import yaml

from optical_functions import (
    LG,
    cart2pol,
    oamModes,
    output_chan_circle,
    setKnotType,
)
from sorter_configuration import parse_optical_train_config
from supersampled_sorting import (
    DEFAULT_EVALUATION_GRID_SIZE,
    compute_supersampled_sorting_metrics,
)


NM = 1e-9
UM = 1e-6
MM = 1e-3
CM = 1e-2


def _load_saved_geometry(checkpoint_directory, instance_name, optical_train,
                         plane_count):
    """Use the same geometry-sidecar precedence as PhaseAnalyzer."""
    geometry_path = Path(checkpoint_directory) / f"{instance_name}_geometry.yaml"
    saved_geometry = {}
    if geometry_path.exists():
        with geometry_path.open("r", encoding="utf-8") as stream:
            saved_geometry = yaml.safe_load(stream) or {}

    saved_stages = saved_geometry.get("stages", [])
    use_saved_geometry = (
        saved_geometry.get("model") == optical_train.model
        and (
            optical_train.model == "legacy_fft"
            or len(saved_stages) == plane_count
        )
    )

    padding_factor = optical_train.padding_factor
    if use_saved_geometry:
        padding_factor = saved_geometry.get(
            "refinement_padding_factor",
            saved_geometry.get("padding_factor", padding_factor),
        )
        stages = [
            {
                "z_to_lens": stage["z_to_lens_cm"] * CM,
                "focal_length": stage["focal_length_cm"] * CM,
                "z_after_lens": stage["z_after_lens_cm"] * CM,
            }
            for stage in saved_stages
        ]
    else:
        initial_geometry = (
            optical_train.initial_normalized_geometry
            if optical_train.num_geometry_genes else None
        )
        stages = optical_train.decode_geometry(initial_geometry)

    return float(padding_factor), stages


def _create_input_modes(config, coordinates, radius, azimuth, detectors,
                        pixel_pitch, wavenumber):
    """Create the configured, ordered input alphabet."""
    detector_count = len(detectors)
    beam_waist = float(config["w0"]) * MM

    if config["isKnot"]:
        labels = list(config["knotType"])
        shape_parameters = list(config["shapeParams"])
        mirror = config.get("mirror", False)
        mirrors = [mirror] * len(labels) if isinstance(mirror, bool) else mirror
        if not (
            len(labels) == len(shape_parameters) == len(mirrors)
            == detector_count
        ):
            raise ValueError(
                "knotType, shapeParams, mirror, and detector counts must match."
            )
        modes = [
            oamModes(
                setKnotType(
                    radius, azimuth, beam_waist, knot_name, parameters,
                    mirror=is_mirrored,
                ),
                detector,
            )
            for knot_name, parameters, is_mirrored, detector in zip(
                labels, shape_parameters, mirrors, detectors
            )
        ]
        return labels, modes

    lg_modes = list(config["LG_modes"])
    if len(lg_modes) != detector_count:
        raise ValueError("The LG alphabet and detector counts must match.")
    labels = [f"LG({ell}, {radial_index})" for ell, radial_index in lg_modes]
    modes = [
        oamModes(
            LG(
                radius, azimuth, ell, radial_index, beam_waist,
                pixel_pitch, 0, wavenumber,
            ),
            detector,
        )
        for (ell, radial_index), detector in zip(lg_modes, detectors)
    ]
    return labels, modes


def evaluate_plane_count_config(
    config_path,
    checkpoint_directory=Path("best_phases"),
    evaluation_grid_size=DEFAULT_EVALUATION_GRID_SIZE,
):
    """Evaluate one checkpoint with supersampled detector accounting.

    ``efficiency_matrix`` contains incident-power fractions.  Its row-normalized
    ``assignment_matrix`` is the conditional crosstalk matrix used for plotting.
    The default 2048-square evaluation uses the finite-device convention from
    the sampling-convergence test in ``PropTest.ipynb``.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    grid_size = int(config["dim"])
    plane_count = int(config.get(
        "num_phase_planes",
        config.get("num_phase_maps_near", 0)
        + config.get("num_phase_maps_far", 0),
    ))
    detector_count = int(config["num_output_chans"])
    optical_train = parse_optical_train_config(config, plane_count)

    wavelength = float(config.get("wavelength_nm", 780.0)) * NM
    wavenumber = 2 * np.pi / wavelength
    pixel_pitch = float(config.get("pixel_pitch_um", 20.0)) * UM
    grid_side_length = pixel_pitch * grid_size
    coordinates = pixel_pitch * (np.arange(grid_size) - grid_size // 2)
    xx, yy = np.meshgrid(coordinates, coordinates)
    pupil_radius = np.sqrt(xx**2 + yy**2)

    if config.get("randomRotation", False):
        raise ValueError(
            f"{config_path}: randomRotation must be disabled for a repeatable "
            "plane-count comparison."
        )
    rotation_angle = (
        float(config.get("rot_angle", 0.0))
        if config.get("fixedRotation", True) else 0.0
    )
    xx_rotated = (
        np.cos(rotation_angle) * xx - np.sin(rotation_angle) * yy
    )
    yy_rotated = (
        np.sin(rotation_angle) * xx + np.cos(rotation_angle) * yy
    )
    radius, azimuth = cart2pol(xx_rotated, yy_rotated)

    detectors = output_chan_circle(
        coordinates,
        coordinates,
        float(config["output_chan_width"]) * MM,
        grid_side_length,
        detector_count,
        circle_radius=float(config["circle_radius"]),
        coordinate_mode=optical_train.output_coordinate_mode,
    )
    labels, input_modes = _create_input_modes(
        config,
        coordinates,
        radius,
        azimuth,
        detectors,
        pixel_pitch,
        wavenumber,
    )

    instance_name = config["ga_instance"]
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_path = checkpoint_directory / f"{instance_name}.pkl"
    with checkpoint_path.open("rb") as stream:
        phase_angles = np.asarray(pickle.load(stream))
    expected_shape = (plane_count, grid_size, grid_size)
    if phase_angles.shape != expected_shape:
        raise ValueError(
            f"{checkpoint_path} has shape {phase_angles.shape}; "
            f"expected {expected_shape}."
        )
    phase_maps = np.exp(1j * phase_angles)

    padding_factor, stages = _load_saved_geometry(
        checkpoint_directory, instance_name, optical_train, plane_count
    )

    mirror = config.get("mirror", False)
    mirrors = [mirror] * len(labels) if isinstance(mirror, bool) else mirror

    def rebuild_input(input_index, axis, sample_pitch):
        grid_x, grid_y = np.meshgrid(axis, axis)
        rotated_x = (
            np.cos(rotation_angle) * grid_x
            - np.sin(rotation_angle) * grid_y
        )
        rotated_y = (
            np.sin(rotation_angle) * grid_x
            + np.cos(rotation_angle) * grid_y
        )
        refined_radius, refined_azimuth = cart2pol(rotated_x, rotated_y)
        if config["isKnot"]:
            return setKnotType(
                refined_radius,
                refined_azimuth,
                float(config["w0"]) * MM,
                config["knotType"][input_index],
                config["shapeParams"][input_index],
                mirror=mirrors[input_index],
            )
        ell, radial_index = config["LG_modes"][input_index]
        return LG(
            refined_radius,
            refined_azimuth,
            ell,
            radial_index,
            float(config["w0"]) * MM,
            sample_pitch,
            0,
            wavenumber,
        )

    metrics = compute_supersampled_sorting_metrics(
        phase_maps,
        [mode.oamBeam for mode in input_modes],
        detectors,
        pixel_pitch,
        evaluation_grid_size=evaluation_grid_size,
        alpha=float(config.get("alpha", 1.0)),
        throughput_metric=config.get(
            "throughput_metric", "geometric_mean"
        ),
        optical_model=optical_train.model,
        field_sampler=rebuild_input,
        wavelength=wavelength,
        stages=stages,
        lens_radius=optical_train.lens_aperture_radius,
        config=config,
    )
    efficiency_matrix = metrics["efficiency_matrix"]
    accepted_efficiencies = metrics["accepted_efficiencies"]
    assignment_matrix = metrics["assignment_matrix"]

    return {
        "config_path": config_path,
        "instance": instance_name,
        "plane_count": plane_count,
        "labels": labels,
        "efficiency_matrix": efficiency_matrix,
        "accepted_efficiencies": accepted_efficiencies,
        "correct_detector_efficiencies": np.diag(efficiency_matrix),
        "assignment_matrix": assignment_matrix,
        "conditional_crosstalk": 1 - np.diag(assignment_matrix),
        "balanced_throughput": metrics["throughput"],
        "fitness": metrics["fitness"],
        "propagated_survival": metrics["propagated_survival"],
        "evaluation_grid_size": metrics["evaluation_grid_size"],
        "samples_per_pixel": metrics["samples_per_pixel"],
        "checkpoint_padding_factor": padding_factor,
    }


def plot_plane_count_sorting(
    config_indices=range(18, 22),
    experiment_name="Knot Sorting ga40",
    config_root=Path("configs"),
    checkpoint_directory=Path("best_phases"),
    evaluation_grid_size=DEFAULT_EVALUATION_GRID_SIZE,
    save_path=None,
    dpi=200,
):
    """Plot detector-efficiency trends and conditional crosstalk matrices."""
    import matplotlib.pyplot as plt

    config_paths = [
        Path(config_root) / experiment_name / f"ga{index}.yaml"
        for index in config_indices
    ]
    results = sorted(
        [
            evaluate_plane_count_config(
                config_path,
                checkpoint_directory,
                evaluation_grid_size=evaluation_grid_size,
            )
            for config_path in config_paths
        ],
        key=lambda result: result["plane_count"],
    )
    if not results:
        raise ValueError("At least one configuration index is required.")

    labels = results[0]["labels"]
    if any(result["labels"] != labels for result in results):
        raise ValueError(
            "All comparison configurations must use the same ordered alphabet."
        )
    plane_counts = np.asarray([
        result["plane_count"] for result in results
    ])

    figure = plt.figure(figsize=(17, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(1.15, 1.15, 1, 1))
    efficiency_axis = figure.add_subplot(grid[0, :2])
    crosstalk_axis = figure.add_subplot(grid[1, :2])
    matrix_grid = grid[:, 2:].subgridspec(2, 2, wspace=0.18, hspace=0.25)
    matrix_axes = [
        figure.add_subplot(matrix_grid[row, column])
        for row in range(2) for column in range(2)
    ]
    if len(results) > len(matrix_axes):
        raise ValueError(
            "The comparison figure supports at most four plane counts."
        )

    for mode_index, label in enumerate(labels):
        efficiency_axis.plot(
            plane_counts,
            100 * np.asarray([
                result["correct_detector_efficiencies"][mode_index]
                for result in results
            ]),
            marker="o",
            linewidth=2,
            label=label,
        )
        crosstalk_axis.plot(
            plane_counts,
            100 * np.asarray([
                result["conditional_crosstalk"][mode_index]
                for result in results
            ]),
            marker="o",
            linewidth=2,
            label=label,
        )

    efficiency_axis.plot(
        plane_counts,
        100 * np.asarray([
            result["balanced_throughput"] for result in results
        ]),
        marker="s",
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="Balanced accepted throughput",
    )
    efficiency_axis.set(
        title="Detector efficiency",
        xlabel="Number of phase planes",
        ylabel="Incident power in detector (%)",
        xticks=plane_counts,
    )
    crosstalk_axis.set(
        title="Conditional crosstalk",
        xlabel="Number of phase planes",
        ylabel="Power assigned to wrong detectors (%)",
        xticks=plane_counts,
    )
    for axis in (efficiency_axis, crosstalk_axis):
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9, ncol=2)

    matrix_image = None
    for axis, result in zip(matrix_axes, results):
        matrix = result["assignment_matrix"]
        matrix_image = axis.imshow(
            matrix, vmin=0, vmax=1, cmap="viridis", aspect="equal"
        )
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=10,
                )
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(f"{result['plane_count']} planes")
        axis.set_xlabel("Detected as")
        axis.set_ylabel("Input")
    for axis in matrix_axes[len(results):]:
        axis.set_visible(False)

    figure.colorbar(
        matrix_image,
        ax=matrix_axes[:len(results)],
        shrink=0.8,
        label="Conditional detector assignment",
    )
    figure.suptitle(
        f"{len(labels)}-mode sorter performance versus phase-plane count "
        f"({evaluation_grid_size} x {evaluation_grid_size} evaluation)"
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plane-count comparison to {save_path}")

    for result in results:
        correct_efficiencies = ", ".join(
            f"{label}={100 * value:.2f}%"
            for label, value in zip(
                labels, result["correct_detector_efficiencies"]
            )
        )
        print(
            f"{result['plane_count']} planes ({result['instance']}): "
            f"{correct_efficiencies}; balanced throughput="
            f"{100 * result['balanced_throughput']:.2f}%; "
            f"fitness={result['fitness']:.6g}"
        )

    plt.show()
    return results, figure
