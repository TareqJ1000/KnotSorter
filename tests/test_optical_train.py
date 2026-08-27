import unittest

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift

from optical_functions import (
    balanced_detector_throughput,
    cart2pol,
    center_crop,
    center_pad,
    complex_field_fidelity,
    fresnel_sampling_diagnostics,
    lens_phase,
    intensity_fidelity,
    output_chan_circle,
    padded_grid_size,
    propTF,
    propagate_fresnel_lens_train,
    propagate_legacy_fft,
)
from sorter_configuration import parse_optical_train_config


class FresnelPropagationTests(unittest.TestCase):
    def setUp(self):
        self.size = 64
        self.pitch = 20e-6
        self.side_length = self.size*self.pitch
        self.wavelength = 780e-9
        self.x = self.pitch*(np.arange(self.size)-self.size//2)
        self.xx, self.yy = np.meshgrid(self.x, self.x)
        self.rr, _ = cart2pol(self.xx, self.yy)
        self.field = np.exp(
            -((self.xx-0.12e-3)**2+(self.yy+0.08e-3)**2)/(0.14e-3)**2
        ).astype(complex)

    def test_zero_distance_returns_copy(self):
        propagated = propTF(
            self.field, self.side_length, self.wavelength, 0.0
        )
        np.testing.assert_allclose(propagated, self.field)
        self.assertIsNot(propagated, self.field)

    def test_fresnel_transfer_function_preserves_power(self):
        propagated = propTF(
            self.field, self.side_length, self.wavelength, 0.02
        )
        before = np.sum(np.abs(self.field)**2)
        after = np.sum(np.abs(propagated)**2)
        self.assertAlmostEqual(after/before, 1.0, places=12)

    def test_thin_lens_uses_voelz_quadratic_phase(self):
        focal_length = 0.08
        wave_number = 2*np.pi/self.wavelength
        lens = lens_phase(
            self.rr, 0.4e-3, wave_number, focal_length
        )
        expected = np.exp(
            -1j*wave_number*self.rr**2/(2*focal_length)
        )
        inside = self.rr <= 0.4e-3
        np.testing.assert_allclose(lens[inside], expected[inside])
        np.testing.assert_allclose(lens[~inside], 0.0)

    def test_front_to_back_focal_plane_matches_fft_intensity(self):
        # At the TF critical distance, the source and focal-plane grids use the
        # same sampling, so the explicit Fresnel/lens train matches a centered
        # FFT up to phase and normalization.
        focal_length = (
            self.side_length*self.pitch/self.wavelength
        )
        stages = [{
            "z_to_lens": focal_length,
            "focal_length": focal_length,
            "z_after_lens": focal_length,
        }]
        phase_maps = np.ones((1, self.size, self.size), dtype=complex)
        propagated = propagate_fresnel_lens_train(
            self.field,
            phase_maps,
            self.side_length,
            self.wavelength,
            stages,
            self.rr,
        )
        reference = fftshift(fft2(ifftshift(self.field)))
        actual_intensity = np.abs(propagated)**2
        reference_intensity = np.abs(reference)**2
        actual_intensity /= actual_intensity.sum()
        reference_intensity /= reference_intensity.sum()
        np.testing.assert_allclose(
            actual_intensity, reference_intensity, rtol=1e-11, atol=1e-13
        )
        self.assertAlmostEqual(
            complex_field_fidelity(propagated, reference), 1.0, places=12
        )
        self.assertAlmostEqual(
            intensity_fidelity(propagated, reference), 1.0, places=12
        )

    def test_three_phase_planes_return_three_intermediate_stages(self):
        distance = 0.02
        stages = [
            {
                "z_to_lens": distance,
                "focal_length": 0.08,
                "z_after_lens": distance,
            }
            for _ in range(3)
        ]
        phase_maps = np.ones((3, self.size, self.size), dtype=complex)
        output, intermediate = propagate_fresnel_lens_train(
            self.field,
            phase_maps,
            self.side_length,
            self.wavelength,
            stages,
            self.rr,
            return_intermediate=True,
        )
        self.assertEqual(output.shape, self.field.shape)
        self.assertEqual(len(intermediate), 3)
        self.assertIn("stage_output", intermediate[-1])

    def test_sampling_diagnostics_identify_critical_distance(self):
        critical = self.side_length*self.pitch/self.wavelength
        diagnostics = fresnel_sampling_diagnostics(
            self.side_length,
            self.size,
            self.wavelength,
            propagation_distance=critical,
        )
        self.assertAlmostEqual(diagnostics["tf_ratio"], 1.0, places=12)

    def test_centered_padding_round_trip_preserves_optical_axis(self):
        source = np.arange(25).reshape(5, 5)
        padded_size = padded_grid_size(5, 2.0)
        self.assertEqual(padded_size, 11)
        padded = center_pad(source, (padded_size, padded_size), fill_value=-1)
        self.assertEqual(padded[padded_size//2, padded_size//2], source[2, 2])
        np.testing.assert_array_equal(center_crop(padded, source.shape), source)

    def test_padded_train_propagates_on_expanded_grid_and_crops_detector(self):
        stages = [{
            "z_to_lens": 0.02,
            "focal_length": 0.08,
            "z_after_lens": 0.02,
        }]
        phase_maps = np.ones((1, self.size, self.size), dtype=complex)
        output, intermediate = propagate_fresnel_lens_train(
            self.field,
            phase_maps,
            self.side_length,
            self.wavelength,
            stages,
            self.rr,
            padding_factor=2.0,
            return_intermediate=True,
        )
        self.assertEqual(output.shape, self.field.shape)
        self.assertEqual(
            intermediate[0]["stage_output"].shape,
            (2*self.size, 2*self.size),
        )
        input_power = np.sum(np.abs(self.field)**2)
        padded_output_power = np.sum(
            np.abs(intermediate[0]["stage_output"])**2
        )
        self.assertAlmostEqual(padded_output_power/input_power, 1.0, places=12)

    def test_padding_expands_transfer_function_sampling_limit(self):
        unpadded = fresnel_sampling_diagnostics(
            self.side_length, self.size, self.wavelength,
            propagation_distance=0.02,
        )
        padded = fresnel_sampling_diagnostics(
            self.side_length, self.size, self.wavelength,
            propagation_distance=0.02, padding_factor=2.0,
        )
        self.assertAlmostEqual(
            padded["critical_distance"]/unpadded["critical_distance"],
            2.0,
        )
        self.assertAlmostEqual(padded["tf_ratio"]/unpadded["tf_ratio"], 0.5)


class LegacyFFTPropagationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260826)
        self.field = (
            rng.normal(size=(16, 16))+1j*rng.normal(size=(16, 16))
        )
        phase_angles = rng.uniform(-np.pi, np.pi, size=(7, 16, 16))
        self.phase_maps = np.exp(1j*phase_angles)

    def test_one_plane_matches_historical_forward_fft(self):
        expected = fftshift(fft2(self.field*self.phase_maps[0]))
        actual = propagate_legacy_fft(self.field, self.phase_maps[:1])
        np.testing.assert_array_equal(actual, expected)

    def test_two_planes_match_historical_forward_inverse_pair(self):
        first_plane = fftshift(fft2(self.field*self.phase_maps[0]))
        expected = ifft2(ifftshift(first_plane*self.phase_maps[1]))
        actual = propagate_legacy_fft(self.field, self.phase_maps[:2])
        np.testing.assert_array_equal(actual, expected)

    def test_three_planes_apply_third_mask_and_forward_fft(self):
        first_plane = fftshift(fft2(self.field*self.phase_maps[0]))
        second_plane = ifft2(ifftshift(first_plane*self.phase_maps[1]))
        expected = fftshift(fft2(second_plane*self.phase_maps[2]))
        actual = propagate_legacy_fft(self.field, self.phase_maps[:3])
        np.testing.assert_array_equal(actual, expected)

        changed_maps = self.phase_maps[:3].copy()
        changed_maps[2] *= np.exp(
            1j*0.2*np.arange(self.field.shape[1])[None, :]
        )
        changed = propagate_legacy_fft(self.field, changed_maps)
        self.assertLess(complex_field_fidelity(actual, changed), 0.99)

    def test_arbitrary_train_alternates_forward_and_inverse_transforms(self):
        expected = self.field
        for index, phase_map in enumerate(self.phase_maps):
            modulated = expected*phase_map
            expected = (
                fftshift(fft2(modulated))
                if index % 2 == 0
                else ifft2(ifftshift(modulated))
            )

        actual = propagate_legacy_fft(self.field, self.phase_maps)
        np.testing.assert_array_equal(actual, expected)


class SorterConfigurationTests(unittest.TestCase):
    def test_legacy_train_allows_arbitrary_positive_plane_count(self):
        train = parse_optical_train_config({}, 7)
        self.assertEqual(train.model, "legacy_fft")
        self.assertEqual(train.num_phase_planes, 7)

    def test_physical_train_retains_three_plane_limit(self):
        with self.assertRaisesRegex(ValueError, "at most three"):
            parse_optical_train_config({
                "optical_train": {"model": "fresnel_lens_train"}
            }, 4)

    def test_three_plane_geometry_has_nine_bounded_genes(self):
        stage = {
            "z_to_lens_cm": {"initial": 6.0, "min": 3.0, "max": 6.5},
            "focal_length_cm": {"initial": 6.0, "min": 5.5, "max": 10.0},
            "z_after_lens_cm": {"initial": 6.0, "min": 3.0, "max": 6.5},
        }
        config = {
            "optical_train": {
                "model": "fresnel_lens_train",
                "optimize_geometry": True,
                "lens_aperture_radius_mm": 1.0,
                "padding_factor": 2.0,
                "stages": [stage, stage, stage],
            }
        }
        train = parse_optical_train_config(config, 3)
        self.assertEqual(train.num_geometry_genes, 9)
        self.assertEqual(train.padding_factor, 2.0)
        decoded_min = train.decode_geometry(np.zeros(9))
        decoded_max = train.decode_geometry(np.ones(9))
        self.assertEqual(train.metadata(decoded_min)["padding_factor"], 2.0)
        self.assertAlmostEqual(decoded_min[0]["z_to_lens"], 0.03)
        self.assertAlmostEqual(decoded_max[2]["focal_length"], 0.10)

    def test_physical_output_channels_use_the_propagation_grid(self):
        size = 128
        pitch = 20e-6
        x = pitch*(np.arange(size)-size//2)
        channels = output_chan_circle(
            x, x, 0.1e-3, size*pitch, 2,
            circle_radius=0.5,
            coordinate_mode="physical",
        )
        xx, _ = np.meshgrid(x, x)
        centers = [
            np.sum(xx*np.real(channel))/np.sum(np.real(channel))
            for channel in channels
        ]
        self.assertAlmostEqual(centers[0], 0.5e-3, delta=pitch)
        self.assertAlmostEqual(centers[1], -0.5e-3, delta=pitch)


class FitnessMetricTests(unittest.TestCase):
    def test_geometric_throughput_penalizes_uniformly_low_efficiency(self):
        efficient = np.array([[0.495, 0.005], [0.005, 0.495]])
        inefficient = 1e-8*efficient

        efficient_score, efficient_rows = balanced_detector_throughput(efficient)
        inefficient_score, inefficient_rows = balanced_detector_throughput(
            inefficient
        )

        self.assertAlmostEqual(efficient_score, 0.5)
        self.assertAlmostEqual(inefficient_score/efficient_score, 1e-8)
        np.testing.assert_allclose(inefficient_rows/efficient_rows, 1e-8)

    def test_geometric_throughput_penalizes_a_weak_input(self):
        efficiency = np.array([[0.45, 0.05], [0.005, 0.12]])
        throughput, accepted = balanced_detector_throughput(efficiency)

        np.testing.assert_allclose(accepted, [0.5, 0.125])
        self.assertAlmostEqual(throughput, 0.25)

    def test_fidelity_distinguishes_phase_from_intensity_agreement(self):
        reference = np.ones((4, 4), dtype=complex)
        phase_only_change = reference.copy()
        phase_only_change[:, 2:] *= 1j

        self.assertAlmostEqual(
            intensity_fidelity(reference, phase_only_change), 1.0
        )
        self.assertLess(
            complex_field_fidelity(reference, phase_only_change), 1.0
        )


if __name__ == "__main__":
    unittest.main()
