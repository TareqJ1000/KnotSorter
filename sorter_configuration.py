"""Configuration helpers for physical knot-sorter lens trains."""

from dataclasses import dataclass

import numpy as np


CM = 1e-2
MM = 1e-3

_STAGE_FIELDS = (
    ("z_to_lens", "z_to_lens_cm"),
    ("focal_length", "focal_length_cm"),
    ("z_after_lens", "z_after_lens_cm"),
)


@dataclass(frozen=True)
class GeometryParameter:
    """One bounded, scalar optical-train parameter, stored in metres."""

    stage_index: int
    name: str
    initial: float
    minimum: float
    maximum: float

    @property
    def initial_normalized(self):
        if self.maximum == self.minimum:
            return 0.0
        return (self.initial-self.minimum)/(self.maximum-self.minimum)

    def decode(self, normalized_value):
        value = float(np.clip(normalized_value, 0.0, 1.0))
        return self.minimum + value*(self.maximum-self.minimum)


@dataclass(frozen=True)
class OpticalTrainConfig:
    """Validated optical-train configuration used by training and analysis."""

    model: str
    num_phase_planes: int
    optimize_geometry: bool
    stages: tuple
    geometry_parameters: tuple
    lens_aperture_radius: float | None
    padding_factor: float
    output_coordinate_mode: str
    geometry_mutation_probability: float
    geometry_mutation_scale: float

    @property
    def num_geometry_genes(self):
        return len(self.geometry_parameters) if self.optimize_geometry else 0

    @property
    def initial_normalized_geometry(self):
        return np.asarray(
            [parameter.initial_normalized for parameter in self.geometry_parameters],
            dtype=float,
        )

    def decode_geometry(self, normalized_values=None):
        """Return physical stage dictionaries with distances in metres."""
        decoded = [dict(stage) for stage in self.stages]
        if not self.optimize_geometry:
            return decoded

        values = np.asarray(normalized_values, dtype=float)
        if values.size != len(self.geometry_parameters):
            raise ValueError(
                f"Expected {len(self.geometry_parameters)} geometry genes, "
                f"received {values.size}."
            )

        for parameter, value in zip(self.geometry_parameters, values):
            decoded[parameter.stage_index][parameter.name] = parameter.decode(value)
        return decoded

    def metadata(self, stages):
        """Create YAML-safe metadata for a decoded candidate geometry."""
        return {
            "model": self.model,
            "num_phase_planes": self.num_phase_planes,
            "optimize_geometry": self.optimize_geometry,
            "lens_aperture_radius_mm": (
                None if self.lens_aperture_radius is None
                else float(self.lens_aperture_radius/MM)
            ),
            "padding_factor": self.padding_factor,
            "output_coordinate_mode": self.output_coordinate_mode,
            "stages": [
                {
                    "z_to_lens_cm": float(stage["z_to_lens"]/CM),
                    "focal_length_cm": float(stage["focal_length"]/CM),
                    "z_after_lens_cm": float(stage["z_after_lens"]/CM),
                }
                for stage in stages
            ],
        }


def _parse_stage_value(stage, key, stage_index, optimize_geometry):
    raw = stage.get(key)
    if raw is None:
        raise ValueError(f"optical_train.stages[{stage_index}].{key} is required.")

    if isinstance(raw, (int, float)):
        initial = minimum = maximum = float(raw)
    elif isinstance(raw, dict):
        required = {"initial", "min", "max"}
        missing = required-set(raw)
        if missing:
            raise ValueError(
                f"optical_train.stages[{stage_index}].{key} is missing "
                f"{', '.join(sorted(missing))}."
            )
        initial = float(raw["initial"])
        minimum = float(raw["min"])
        maximum = float(raw["max"])
    else:
        raise TypeError(
            f"optical_train.stages[{stage_index}].{key} must be a number "
            "or an {initial, min, max} mapping."
        )

    if minimum > maximum:
        raise ValueError(f"Minimum exceeds maximum for stage {stage_index} {key}.")
    if not minimum <= initial <= maximum:
        raise ValueError(f"Initial value lies outside the bounds for stage {stage_index} {key}.")
    if key != "focal_length_cm" and minimum < 0:
        raise ValueError("Free-space propagation distances cannot be negative.")
    if key == "focal_length_cm" and minimum <= 0 <= maximum:
        raise ValueError("A focal-length interval cannot include zero.")
    if optimize_geometry and minimum == maximum:
        # A fixed value is allowed in an optimized train; it simply receives no
        # gene and remains fixed while the other quantities are optimized.
        pass

    return initial*CM, minimum*CM, maximum*CM


def parse_optical_train_config(config, num_phase_planes):
    """Parse the new optical-train schema with a legacy FFT fallback."""
    if num_phase_planes < 1:
        raise ValueError("num_phase_planes must be at least 1.")

    raw_train = config.get("optical_train")
    if raw_train is None:
        return OpticalTrainConfig(
            model="legacy_fft",
            num_phase_planes=num_phase_planes,
            optimize_geometry=False,
            stages=tuple(),
            geometry_parameters=tuple(),
            lens_aperture_radius=None,
            padding_factor=1.0,
            output_coordinate_mode="legacy",
            geometry_mutation_probability=0.0,
            geometry_mutation_scale=0.0,
        )

    model = raw_train.get("model", "fresnel_lens_train")
    if model not in {"legacy_fft", "fresnel_lens_train"}:
        raise ValueError(
            "optical_train.model must be 'legacy_fft' or 'fresnel_lens_train'."
        )
    if model == "legacy_fft":
        return OpticalTrainConfig(
            model=model,
            num_phase_planes=num_phase_planes,
            optimize_geometry=False,
            stages=tuple(),
            geometry_parameters=tuple(),
            lens_aperture_radius=None,
            padding_factor=1.0,
            output_coordinate_mode="legacy",
            geometry_mutation_probability=0.0,
            geometry_mutation_scale=0.0,
        )

    if num_phase_planes > 3:
        raise ValueError(
            "The physical Fresnel lens train supports at most three phase planes."
        )

    optimize_geometry = bool(raw_train.get("optimize_geometry", False))
    raw_stages = raw_train.get("stages", [])
    if len(raw_stages) != num_phase_planes:
        raise ValueError(
            "A Fresnel lens train requires exactly one stage per phase plane; "
            f"expected {num_phase_planes}, received {len(raw_stages)}."
        )

    stages = []
    parameters = []
    for stage_index, raw_stage in enumerate(raw_stages):
        stage = {}
        for internal_name, yaml_name in _STAGE_FIELDS:
            initial, minimum, maximum = _parse_stage_value(
                raw_stage, yaml_name, stage_index, optimize_geometry
            )
            stage[internal_name] = initial
            if optimize_geometry and minimum != maximum:
                parameters.append(GeometryParameter(
                    stage_index=stage_index,
                    name=internal_name,
                    initial=initial,
                    minimum=minimum,
                    maximum=maximum,
                ))
        stages.append(stage)

    lens_radius_mm = raw_train.get("lens_aperture_radius_mm")
    lens_radius = None if lens_radius_mm is None else float(lens_radius_mm)*MM
    if lens_radius is not None and lens_radius <= 0:
        raise ValueError("lens_aperture_radius_mm must be positive or null.")

    padding_factor = float(raw_train.get("padding_factor", 1.0))
    if not np.isfinite(padding_factor) or padding_factor < 1:
        raise ValueError("optical_train.padding_factor must be at least 1.")

    coordinate_mode = raw_train.get("output_coordinate_mode", "physical")
    if coordinate_mode not in {"legacy", "physical"}:
        raise ValueError("output_coordinate_mode must be 'legacy' or 'physical'.")

    mutation_probability = float(
        raw_train.get("geometry_mutation_probability", 0.20)
    )
    mutation_scale = float(raw_train.get("geometry_mutation_scale", 0.05))
    if not 0 <= mutation_probability <= 1:
        raise ValueError("geometry_mutation_probability must lie in [0, 1].")
    if mutation_scale < 0:
        raise ValueError("geometry_mutation_scale cannot be negative.")

    return OpticalTrainConfig(
        model=model,
        num_phase_planes=num_phase_planes,
        optimize_geometry=optimize_geometry,
        stages=tuple(stages),
        geometry_parameters=tuple(parameters),
        lens_aperture_radius=lens_radius,
        padding_factor=padding_factor,
        output_coordinate_mode=coordinate_mode,
        geometry_mutation_probability=mutation_probability,
        geometry_mutation_scale=mutation_scale,
    )
