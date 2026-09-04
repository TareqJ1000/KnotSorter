import numpy as np

from optical_functions import propagate_legacy_fft
from supersampled_sorting import (
    compute_supersampled_sorting_metrics,
    legacy_detector_channels,
)


def test_unit_sampling_matches_normalized_legacy_efficiencies():
    rng = np.random.default_rng(12)
    size = 8
    phase_maps = np.exp(1j * rng.normal(size=(2, size, size)))
    fields = [
        rng.normal(size=(size, size))
        + 1j * rng.normal(size=(size, size))
        for _ in range(2)
    ]
    channels = np.zeros((2, size, size))
    channels[0, 1:3, 1:3] = 1
    channels[1, 5:7, 5:7] = 1

    metrics = compute_supersampled_sorting_metrics(
        phase_maps,
        fields,
        channels,
        native_pixel_pitch=1.0,
        evaluation_grid_size=size,
        config={
            "fitness_func": "full",
            "gamma": 1.0,
            "throughput_exponent": 1.0,
        },
    )

    expected = np.zeros((2, 2))
    for input_index, field in enumerate(fields):
        output = propagate_legacy_fft(field, phase_maps)
        output *= np.sqrt(
            np.sum(np.abs(field) ** 2) / np.sum(np.abs(output) ** 2)
        )
        for channel_index, channel in enumerate(channels):
            expected[input_index, channel_index] = (
                np.sum(np.abs(output) ** 2 * channel)
                / np.sum(np.abs(field) ** 2)
            )

    np.testing.assert_allclose(
        metrics["efficiency_matrix"], expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(metrics["assignment_matrix"].sum(axis=1), 1)


def test_detector_mapping_follows_output_plane_parity():
    channel = np.zeros((1, 4, 4))
    channel[0, 1, 2] = 1

    image_plane = legacy_detector_channels(
        channel, samples_per_pixel=2, plane_count=2
    )
    assert image_plane.shape == (1, 8, 8)
    assert image_plane.sum() == 4

    fourier_plane = legacy_detector_channels(
        channel, samples_per_pixel=2, plane_count=1
    )
    assert fourier_plane.shape == (1, 8, 8)
    assert fourier_plane.sum() == 1
    assert fourier_plane[0, 3, 4] == 1
