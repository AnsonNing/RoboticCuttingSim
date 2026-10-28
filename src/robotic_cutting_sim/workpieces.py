"""Continuous initial soft-material volumes, independent of tool and task goals.

The capsule is a rounded XY footprint extruded to the existing height field.
Masking takes place once during initial cell-center sampling; no solver state,
mass, cut history or rendering geometry is deleted during an episode. This does
not add imported-mesh or calibrated-material support to TLMPM.
"""
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .benchmark import BenchmarkSpec, HeightFieldSpec, MaterialParameters, sample_specimen
from .scene_frames import invert_transform, pose_matrix, transform_points, validate_pose_xyzw
from .tasks import CuttingTaskSpec


@dataclass(frozen=True)
class WorkpieceSpec:
    geometry: HeightFieldSpec
    footprint: str = "rectangle"
    world_from_workpiece_xyzw: tuple = (0., 0., 0., 0., 0., 0., 1.)
    material: MaterialParameters = field(default_factory=MaterialParameters)
    fixture_boxes_local: tuple = ()

    def __post_init__(self):
        if self.footprint not in ("rectangle", "capsule"):
            raise ValueError("Supported workpiece footprints are rectangle and capsule")
        if not isinstance(self.geometry, HeightFieldSpec) or not isinstance(self.material, MaterialParameters):
            raise ValueError("A workpiece requires an explicit height field and material recipe")
        if validate_pose_xyzw(self.world_from_workpiece_xyzw).shape != (7,):
            raise ValueError("A workpiece requires one rigid world pose")
        if self.footprint == "capsule" and self.geometry.length_m < self.geometry.width_m:
            raise ValueError("Capsule length along X must be at least its diameter along Y")

    @property
    def dimensions_m(self):
        return (self.geometry.length_m, self.geometry.width_m,
                self.geometry.base_height_m + max((b.height_m for b in self.geometry.bumps), default=0.))

    @property
    def T_world_workpiece(self):
        return pose_matrix(self.world_from_workpiece_xyzw)

    @property
    def analytic_base_volume_m3(self):
        length, width, height = self.geometry.length_m, self.geometry.width_m, self.geometry.base_height_m
        area = length * width if self.footprint == "rectangle" else (length - width) * width + np.pi * (width / 2)**2
        return float(area * height)

    def contains_xy(self, xy_local_m):
        xy = np.asarray(xy_local_m, float)
        if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError("Expected finite (...,2) workpiece-local XY points")
        x, y = np.abs(xy[..., 0]), np.abs(xy[..., 1])
        if self.footprint == "rectangle":
            return (x <= self.geometry.length_m / 2) & (y <= self.geometry.width_m / 2)
        radius = self.geometry.width_m / 2
        segment_half_length = (self.geometry.length_m - self.geometry.width_m) / 2
        return np.maximum(x - segment_half_length, 0.)**2 + y**2 <= radius**2

    def height(self, x_local_m, y_local_m):
        x, y = np.broadcast_arrays(x_local_m, y_local_m)
        return np.where(self.contains_xy(np.stack([x, y], axis=-1)), self.geometry.height(x, y), 0.)

    def points_to_world(self, local):
        return transform_points(self.T_world_workpiece, local)

    def points_to_local(self, world):
        return transform_points(invert_transform(self.T_world_workpiece), world)

    def sample_particles(self, spacing_m, *, seed=42, variant="CUT-01"):
        """LOCAL positions, positive reference masses, original lattice; no labels.

        Every retained cell has density*spacing**3 mass. Curved boundaries are a
        declared cell-center approximation; boundary cells are not fractionally
        weighted. A height bump can be clipped by the initial capsule boundary.
        """
        points, masses, lattice, _ = sample_specimen(BenchmarkSpec(variant, seed, self.geometry, self.material), spacing_m)
        if self.footprint == "rectangle":
            return points, masses, lattice
        inside = self.contains_xy(points[:, :2])
        return points[inside], masses[inside], lattice[inside]

    def to_dict(self):
        return {**asdict(self), "workpiece_spec_version": "continuous_footprint_v1",
                "dimensions_m": list(self.dimensions_m),
                "analytic_base_volume_m3": self.analytic_base_volume_m3,
                "analytic_base_mass_kg": self.analytic_base_volume_m3 * self.material.density_kg_m3,
                "analytic_total_volume_m3": None if self.geometry.bumps else self.analytic_base_volume_m3,
                "sampling": "Initial cell centers in footprint and one max-union height field; fixed per-cell reference mass",
                "boundary_volume_scope": "Discrete cell-center approximation, not fractional boundary volumes",
                "component_labels_supplied": False, "runtime_geometry_deletion": False,
                "material_support": "Provisional Plasticine-like TLMPM recipe; no universal or imported-mesh cutting claim"}


@dataclass(frozen=True)
class FootprintCuttingTaskSpec(CuttingTaskSpec):
    """Only opt-in footprints add a serialized field to the legacy TaskSpec.

    Keeping the base dataclass fields unchanged preserves old asdict receipts,
    strict source audits, samplers and fixed/randomized task signatures.
    """

    workpiece_kind: str = "capsule"

    def __post_init__(self):
        super().__post_init__()
        self.workpiece  # Validate footprint dimensions without sampling/stepping.


def with_workpiece_footprint(task, footprint):
    if footprint not in ("rectangle", "capsule"):
        raise ValueError("Supported workpiece footprints are rectangle and capsule")
    values = {item.name: getattr(task, item.name) for item in fields(CuttingTaskSpec)}
    if footprint == "rectangle":
        return task if task.workpiece_kind == "rectangle" else CuttingTaskSpec(**values)
    return FootprintCuttingTaskSpec(**values, workpiece_kind=footprint)
