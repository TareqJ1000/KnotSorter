"""High-resolution reevaluation of phase-mask sorter checkpoints.

The optimizer stores phase masks on their native device grid.  This module
keeps those device pixels fixed while evaluating the optical field on a finer
grid, matching the sampling-convergence experiment in ``PropTest.ipynb``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.ndimage import zoom

from optical_functions import (
    balanced_detector_throughput,
    build_fresnel_lens_kernels,
    propagate_fresnel_lens_train,
    propagate_legacy_fft_supersampled,
    shannon_entropy,
)


DEFAULT_EVALUATION_GRID_SIZE = 2048


def samples_per_device_pixel(native_grid_size, evaluation_grid_size):
    """Return the integer sampling refinement for a requested square grid."""
    native_grid_size = int(native_grid_size)
    evaluation_grid_size = int(evaluation_grid_size)
    if native_grid_size < 1 or evaluation_grid_size < native_grid_size:
        raise ValueError(
            "evaluation_grid_size must be at least the native grid size."
        )
    if evaluation_grid_size % native_grid_size:
        raise ValueError(
            "evaluation_grid_size must be an integer multiple of the native "
            "phase-mask grid size."
        )
    return evaluation_grid_size // native_grid_size


def supersample_complex_field(field, samples_per_pixel):
    """Interpolate an arbitrary native-grid field onto the refined grid.

    Analytically rebuilt fields can instead be supplied through ``field_sampler``
    in :func:`compute_supersampled_sorting_metrics`.  Interpolation is retained
    as the fallback needed by the robustness notebook, where the incoming field
    may already contain a rotation, translation, propagated offset, or phase
    aberration.
    """
    field = np.asarray(field)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("Every input field must be a square 2-D array.")
    if int(samples_per_pixel) != samples_per_pixel or samples_per_pixel < 1:
        raise ValueError("samples_per_pixel must be a positive integer.")
    samples_per_pixel = int(samples_per_pixel)
    if samples_per_pixel == 1:
        return np.asarray(field, dtype=np.complex128)

    # Cubic interpolation is applied to the real and imaginary parts because
    # scipy.ndimage.zoom does not interpolate complex values as one quantity.
    zoom_factor = (samples_per_pixel, samples_per_pixel)
    real = zoom(
        np.real(field), zoom_factor, order=3, mode="grid-constant",
        cval=0.0, prefilter=True, grid_mode=True,
    )
    imaginary = zoom(
        np.imag(field), zoom_factor, order=3, mode="grid-constant",
        cval=0.0, prefilter=True, grid_mode=True,
    )
    return real + 1j * imaginary


def legacy_detector_channels(output_channels, samples_per_pixel, plane_count):
    """Map native detector pixels onto the supersampled legacy output plane.

    This is the same ``detector_mode='device'`` convention used by PropTest:
    image-plane detector pixels are replicated, whereas a Fourier-plane device
    retains its native footprint in the centre of the larger Fourier grid.
    """
    channels = np.real(np.asarray(output_channels))
    if channels.ndim != 3 or channels.shape[1] != channels.shape[2]:
        raise ValueError(
            "output_channels must have shape (channels, N, N)."
        )
    samples_per_pixel = int(samples_per_pixel)
    native_size = channels.shape[1]
    evaluation_size = native_size * samples_per_pixel

    if plane_count % 2 == 0:
        return np.repeat(
            np.repeat(channels, samples_per_pixel, axis=1),
            samples_per_pixel,
            axis=2,
        )

    mapped = np.zeros(
        (channels.shape[0], evaluation_size, evaluation_size), dtype=float
    )
    start = (evaluation_size - native_size) // 2
    mapped[:, start:start + native_size, start:start + native_size] = channels
    return mapped


def _supersampled_fresnel_channels(output_channels, samples_per_pixel):
    """Preserve each physical detector pixel on a refined Fresnel grid."""
    channels = np.real(np.asarray(output_channels))
    return np.repeat(
        np.repeat(channels, samples_per_pixel, axis=1),
        samples_per_pixel,
        axis=2,
    )


def _sample_input_field(
    input_index,
    input_fields,
    field_sampler,
    evaluation_axis,
    evaluation_pixel_pitch,
    native_grid_size,
    samples_per_pixel,
):
    if field_sampler is not None:
        field = np.asarray(field_sampler(
            input_index, evaluation_axis, evaluation_pixel_pitch
        ))
        expected_shape = (len(evaluation_axis), len(evaluation_axis))
        if field.shape != expected_shape:
            raise ValueError(
                "field_sampler returned shape "
                f"{field.shape}; expected {expected_shape}."
            )
        return np.asarray(field, dtype=np.complex128)

    field = np.asarray(input_fields[input_index])
    native_shape = (native_grid_size, native_grid_size)
    evaluation_shape = (len(evaluation_axis), len(evaluation_axis))
    if field.shape == native_shape:
        return supersample_complex_field(field, samples_per_pixel)
    if field.shape == evaluation_shape:
        return np.asarray(field, dtype=np.complex128)
    raise ValueError(
        f"Input field {input_index} has shape {field.shape}; expected "
        f"{native_shape} or {evaluation_shape}."
    )


def _metric_summary(efficiency_matrix, alpha, throughput_metric):
    efficiency_matrix = np.asarray(efficiency_matrix, dtype=float)
    mode_count, detector_count = efficiency_matrix.shape
    if mode_count < 2 or mode_count != detector_count:
        raise ValueError(
            "The input alphabet and detector-channel counts must match and "
            "contain at least two modes."
        )

    accepted = efficiency_matrix.sum(axis=1, keepdims=True)
    assignment = np.divide(
        efficiency_matrix,
        accepted,
        out=np.zeros_like(efficiency_matrix),
        where=accepted > 0,
    )
    correct = np.diag(assignment)
    wrong_mean = (assignment.sum(axis=1) - correct) / (mode_count - 1)
    channel_contrast = correct - wrong_mean
    sorting_performance = (
        float(alpha) * np.min(channel_contrast)
        + (1.0 - float(alpha)) * np.mean(channel_contrast)
    )
    throughput, accepted_efficiencies = balanced_detector_throughput(
        efficiency_matrix, method=throughput_metric
    )
    symbol_error = 1.0 - np.mean(correct)
    secret_key = max(
        0.0,
        np.log2(mode_count) - 2.0 * shannon_entropy(symbol_error, mode_count),
    )
    return {
        "sorting_performance": float(sorting_performance),
        "efficiency_matrix": efficiency_matrix,
        "assignment_matrix": assignment,
        "coincidence_probabilities": assignment,
        "channel_contrast": channel_contrast,
        "secret_key": float(secret_key),
        "throughput": float(throughput),
        "accepted_efficiencies": accepted_efficiencies,
    }


def configured_fitness(metrics, config):
    """Reproduce the configured post-warm-up objective from ``run_ga.py``."""
    contrast = float(metrics["sorting_performance"])
    secret_key = float(metrics["secret_key"])
    assignment = np.asarray(metrics["assignment_matrix"])
    throughput = float(metrics["throughput"])
    fitness_name = config.get("fitness_func", "full")

    if fitness_name == "secret_key":
        return float(np.real(contrast * secret_key))
    if fitness_name not in {"bread", "full"}:
        raise ValueError(
            "fitness_func must be 'secret_key', 'bread', or 'full'."
        )

    gamma = float(config.get("gamma", 1.0))
    throughput_exponent = float(config.get("throughput_exponent", 0.0))
    distinguishability = abs(np.linalg.det(assignment)) ** gamma
    throughput_factor = (
        1.0 if throughput_exponent == 0.0
        else throughput ** throughput_exponent
    )
    return float(np.real(
        contrast * secret_key * distinguishability * throughput_factor
    ))


def compute_supersampled_sorting_metrics(
    phase_maps,
    input_fields: Sequence[np.ndarray],
    output_channels,
    native_pixel_pitch,
    *,
    evaluation_grid_size=DEFAULT_EVALUATION_GRID_SIZE,
    alpha=1.0,
    throughput_metric="geometric_mean",
    optical_model="legacy_fft",
    field_sampler: Callable | None = None,
    wavelength=None,
    stages=None,
    lens_radius=None,
    config=None,
):
    """Recompute efficiency, coincidence, and fitness on a refined grid.

    For ``legacy_fft`` this follows PropTest's supersampled finite-device model.
    For ``fresnel_lens_train`` it refines the field and physical device pixels
    over the same physical window before rebuilding the Fresnel kernels.
    """
    phase_maps = np.asarray(phase_maps)
    if phase_maps.ndim != 3 or phase_maps.shape[1] != phase_maps.shape[2]:
        raise ValueError("phase_maps must have shape (planes, N, N).")
    native_grid_size = phase_maps.shape[1]
    plane_count = len(phase_maps)
    samples_per_pixel = samples_per_device_pixel(
        native_grid_size, evaluation_grid_size
    )
    evaluation_grid_size = native_grid_size * samples_per_pixel
    evaluation_pixel_pitch = float(native_pixel_pitch) / samples_per_pixel
    evaluation_axis = evaluation_pixel_pitch * (
        np.arange(evaluation_grid_size) - evaluation_grid_size // 2
    )

    if len(input_fields) != len(output_channels):
        raise ValueError(
            "The input alphabet and detector-channel counts must match."
        )

    if optical_model == "legacy_fft":
        evaluation_channels = legacy_detector_channels(
            output_channels, samples_per_pixel, plane_count
        )
        fresnel_phase_maps = None
        fresnel_kernels = None
        evaluation_radius = None
    elif optical_model == "fresnel_lens_train":
        if wavelength is None or stages is None:
            raise ValueError(
                "wavelength and stages are required for Fresnel evaluation."
            )
        evaluation_channels = _supersampled_fresnel_channels(
            output_channels, samples_per_pixel
        )
        fresnel_phase_maps = np.repeat(
            np.repeat(phase_maps, samples_per_pixel, axis=1),
            samples_per_pixel,
            axis=2,
        )
        xx, yy = np.meshgrid(evaluation_axis, evaluation_axis)
        evaluation_radius = np.sqrt(xx**2 + yy**2)
        fresnel_kernels = build_fresnel_lens_kernels(
            (evaluation_grid_size, evaluation_grid_size),
            native_grid_size * float(native_pixel_pitch),
            float(wavelength),
            stages,
            evaluation_radius,
            lens_radius=lens_radius,
            padding_factor=1.0,
        )
    else:
        raise ValueError(
            "optical_model must be 'legacy_fft' or 'fresnel_lens_train'."
        )

    efficiency_matrix = np.zeros(
        (len(input_fields), len(output_channels)), dtype=float
    )
    propagated_survival = np.zeros(len(input_fields), dtype=float)
    for input_index in range(len(input_fields)):
        field = _sample_input_field(
            input_index,
            input_fields,
            field_sampler,
            evaluation_axis,
            evaluation_pixel_pitch,
            native_grid_size,
            samples_per_pixel,
        )
        input_power = float(np.sum(np.abs(field) ** 2))
        if not np.isfinite(input_power) or input_power <= 0:
            raise ValueError(f"Input field {input_index} has zero/invalid power.")

        if optical_model == "legacy_fft":
            output = propagate_legacy_fft_supersampled(
                field,
                phase_maps,
                samples_per_pixel=samples_per_pixel,
            )
        else:
            output = propagate_fresnel_lens_train(
                field,
                fresnel_phase_maps,
                native_grid_size * float(native_pixel_pitch),
                float(wavelength),
                stages,
                evaluation_radius,
                lens_radius=lens_radius,
                kernels=fresnel_kernels,
                padding_factor=1.0,
            )

        intensity = np.abs(output) ** 2
        propagated_survival[input_index] = intensity.sum() / input_power
        efficiency_matrix[input_index] = np.asarray([
            np.sum(intensity * channel) / input_power
            for channel in evaluation_channels
        ], dtype=float)

    metrics = _metric_summary(
        efficiency_matrix, alpha=alpha, throughput_metric=throughput_metric
    )
    metrics.update({
        "evaluation_grid_size": evaluation_grid_size,
        "samples_per_pixel": samples_per_pixel,
        "evaluation_pixel_pitch": evaluation_pixel_pitch,
        "propagated_survival": propagated_survival,
    })
    if config is not None:
        metrics["fitness"] = configured_fitness(metrics, config)
    return metrics
