"""Seeded continuous specimens and offline oracle diagnostics, outside observations.

Scenario generation does not execute a simulation. Particle-bin surface estimates
and source-reported component labels do not establish a validated cutting benchmark.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

FAMILIES = ("dome", "gaussian", "ellipsoid_cap", "ridge")


def array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Bump:
    x_m: float = 0.0
    y_m: float = 0.0
    radius_x_m: float = .015
    radius_y_m: float = .015
    height_m: float = .008
    family: str = "dome"

    def __post_init__(self):
        if (self.family not in FAMILIES or
                not np.isfinite([self.x_m, self.y_m, self.radius_x_m,
                                 self.radius_y_m, self.height_m]).all() or
                min(self.radius_x_m, self.radius_y_m, self.height_m) <= 0):
            raise ValueError("Bumps need a known family and finite positive size")

    def height(self, x, y):
        x, y = (np.asarray(x)-self.x_m)/self.radius_x_m, (np.asarray(y)-self.y_m)/self.radius_y_m
        r = np.sqrt(x*x+y*y)
        if self.family == "dome":
            shape = np.where(r < 1, np.cos(np.minimum(r, 1)*np.pi/2)**2, 0)
        elif self.family == "gaussian":
            shape = np.maximum((np.exp(-4*r*r)-np.exp(-4))/(1-np.exp(-4)), 0)
        elif self.family == "ellipsoid_cap":
            shape = np.sqrt(np.maximum(1-r*r, 0))
        else:
            shape = np.maximum(1-x*x, 0)**2*np.maximum(1-y*y, 0)**2
        return self.height_m*shape


@dataclass(frozen=True)
class HeightFieldSpec:
    length_m: float = .120
    width_m: float = .080
    base_height_m: float = .015
    bumps: tuple[Bump, ...] = (Bump(),)

    def __post_init__(self):
        if (not np.isfinite([self.length_m, self.width_m, self.base_height_m]).all()
                or min(self.length_m, self.width_m, self.base_height_m) <= 0):
            raise ValueError("Workpiece dimensions must be finite and positive")
        for bump in self.bumps:
            if (abs(bump.x_m)+bump.radius_x_m >= self.length_m/2 or
                    abs(bump.y_m)+bump.radius_y_m >= self.width_m/2):
                raise ValueError("Every bump must lie within the continuous base")

    def height(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        extra = np.zeros(x.shape, dtype=float)
        for bump in self.bumps:
            # Union of overlapping bumps over one base, never separate bodies.
            extra = np.maximum(extra, bump.height(x, y))
        return self.base_height_m+extra


@dataclass(frozen=True)
class MaterialParameters:
    density_kg_m3: float = 1500.0
    young_modulus_pa: float = 50000.0
    poisson_ratio: float = .30
    yield_stress_pa: float = 1000.0
    hardening_ratio: float = 0.0
    friction: float = .4
    failure_model: str = "not_configured"
    critical_stress_pa: float | None = None
    fracture_energy_j_m2: float | None = None
    calibration: str = "provisional_uncalibrated"

    def __post_init__(self):
        values = [self.density_kg_m3, self.young_modulus_pa, self.poisson_ratio,
                  self.yield_stress_pa, self.hardening_ratio, self.friction]
        if (not np.isfinite(values).all() or
                min(self.density_kg_m3, self.young_modulus_pa, self.yield_stress_pa) <= 0 or
                not 0 <= self.poisson_ratio < .5 or min(self.hardening_ratio, self.friction) < 0):
            raise ValueError("Invalid material recipe parameters")
        for value in (self.critical_stress_pa, self.fracture_energy_j_m2):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Configured failure thresholds must be finite and positive")


@dataclass(frozen=True)
class BenchmarkSpec:
    variant: str
    seed: int
    heightfield: HeightFieldSpec
    material: MaterialParameters = MaterialParameters()
    blade_half_thickness_m: float = .001
    camera_translation_m: tuple[float, float, float] = (0., 0., 0.)
    camera_rotation_deg: tuple[float, float, float] = (0., 0., 0.)

    @classmethod
    def generate(cls, seed=42, variant="D0"):
        if variant not in ("D0", "D1", "D2", "D3"):
            raise ValueError("Variant must be D0, D1, D2 or D3")
        rng = np.random.default_rng(seed)
        if variant == "D0":
            return cls(variant, seed, HeightFieldSpec())
        count = 1 if variant == "D1" else (3 if variant == "D2" else int(rng.integers(2, 6)))
        anchors = [(-.029, -.014), (.029, -.014), (0., .014), (-.029, .014), (.029, .014)]
        bumps = []
        for i in range(count):
            if variant == "D1":
                x, y = rng.uniform(-.025, .025), rng.uniform(-.010, .010)
                rx, ry = rng.uniform(.010, .015, 2)
            else:
                x, y = np.asarray(anchors[i])+rng.uniform(-.002, .002, 2)
                rx, ry = rng.uniform(.007, .011, 2)
            bumps.append(Bump(float(x), float(y), float(rx), float(ry),
                              float(rng.uniform(.005, .010)), str(rng.choice(FAMILIES))))
        material = MaterialParameters()
        blade, translation, rotation = .001, (0., 0., 0.), (0., 0., 0.)
        if variant == "D3":
            material = MaterialParameters(density_kg_m3=float(rng.uniform(1200, 1800)),
                young_modulus_pa=float(rng.uniform(30000, 90000)),
                poisson_ratio=float(rng.uniform(.25, .38)),
                yield_stress_pa=float(rng.uniform(750, 2500)),
                hardening_ratio=float(rng.uniform(0, .3)), friction=float(rng.uniform(.2, .6)))
            blade = float(rng.uniform(.0007, .0015))
            translation = tuple(float(v) for v in rng.uniform(-.005, .005, 3))
            rotation = tuple(float(v) for v in rng.uniform(-3, 3, 3))
        return cls(variant, seed, HeightFieldSpec(bumps=tuple(bumps)), material, blade,
                   translation, rotation)

    def to_dict(self):
        return asdict(self)


def sample_specimen(spec: BenchmarkSpec, spacing_m: float):
    """Return positions, mass, lattice indices, provenance; no component labels."""
    h = spec.heightfield
    if not np.isfinite(spacing_m) or not 0 < spacing_m <= h.base_height_m/2:
        raise ValueError("Spacing must provide at least two layers in the base")
    max_height = h.base_height_m+max((b.height_m for b in h.bumps), default=0.)
    dims = np.ceil(np.array([h.length_m, h.width_m, max_height])/spacing_m).astype(int)
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in dims), indexing="ij"), -1).reshape(-1, 3)
    q = (ijk+.5)*spacing_m-[h.length_m/2, h.width_m/2, 0]
    inside = ((q[:, 0] < h.length_m/2) & (q[:, 1] < h.width_m/2)
              & (q[:, 2] <= h.height(q[:, 0], q[:, 1])))
    q, ijk = q[inside].astype(np.float32), ijk[inside].astype(np.int32)
    mass = np.full(len(q), spec.material.density_kg_m3*spacing_m**3, dtype=np.float64)
    receipt = {"scope": "generated_initial_state_only", "variant": spec.variant,
               "seed": spec.seed, "spacing_m": spacing_m, "particle_count": len(q),
               "initial_mass_kg": float(mass.sum()), "particle_volume_m3": spacing_m**3,
               "continuous_base": True, "preassigned_components": False,
               "sampling": "cell centers below one max-union height field over a single base",
               "positions_sha256": array_sha256(q), "mass_sha256": array_sha256(mass),
               "lattice_sha256": array_sha256(ijk),
               "scenario_sha256": hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest(),
               "material_status": spec.material.calibration, "benchmark_ready": False}
    return q, mass, ijk, receipt


@dataclass(frozen=True)
class MetricConfig:
    length_m: float = .120
    width_m: float = .080
    target_height_m: float = .015
    density_kg_m3: float = 1500.
    surface_cell_size_m: float = .004
    particle_support_radius_m: float = 0.
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        values = list(asdict(self).values())
        if (not np.isfinite(values).all() or min(values[:-2]) <= 0
                or self.particle_support_radius_m < 0 or self.gravity_m_s2 <= 0):
            raise ValueError("Invalid metric dimensions, radius, density or gravity")


def _surface_proxy(points, masses, config):
    nx, ny = np.ceil([config.length_m/config.surface_cell_size_m,
                     config.width_m/config.surface_cell_size_m]).astype(int)
    footprint = ((np.abs(points[:, 0]) <= config.length_m/2)
                 & (np.abs(points[:, 1]) <= config.width_m/2) & (masses > 0))
    q = points[footprint]
    top = np.full(nx*ny, -np.inf)
    if len(q):
        ix = np.minimum(((q[:, 0]+config.length_m/2)*nx/config.length_m).astype(int), nx-1)
        iy = np.minimum(((q[:, 1]+config.width_m/2)*ny/config.width_m).astype(int), ny-1)
        np.maximum.at(top, ix+nx*iy, q[:, 2]+config.particle_support_radius_m)
    occupied = np.isfinite(top)
    top[~occupied] = 0.
    top = np.maximum(top, 0.)
    residual, deficit = np.maximum(top-config.target_height_m, 0.), np.maximum(config.target_height_m-top, 0.)
    return {"residual_height_proxy_m": float(residual.max()),
            "residual_height_p95_proxy_m": float(np.percentile(residual, 95)),
            "rms_target_plane_error_proxy_m": float(np.sqrt(np.mean((top-config.target_height_m)**2))),
            "surface_roughness_std_proxy_m": float(top[occupied].std()) if occupied.any() else None,
            "overcut_volume_proxy_m3": float(deficit.sum()*config.length_m*config.width_m/(nx*ny)),
            "occupied_bin_fraction": float(occupied.mean()), "grid_shape_xy": [int(nx), int(ny)],
            "particle_count_in_footprint": int(footprint.sum())}


def _component_budget(data, q0, qf, mass, velocity):
    unavailable = {"status": "NOT_AVAILABLE", "detached_mass_kg": None,
                   "substrate_mass_kg": None, "permanent_separation_proven": None,
                   "recontact_proven": None}
    if "component_ids" not in data:
        return {**unavailable, "reason": "No component IDs recorded; cut side codes are not component IDs"}, None
    if "component_provenance" not in data:
        return {**unavailable, "reason": "Component IDs lack explicit method provenance"}, None
    provenance_array = np.asarray(data["component_provenance"])
    if provenance_array.shape != () or provenance_array.dtype.kind not in "US":
        raise ValueError("component_provenance must be a scalar Unicode/byte string")
    provenance = str(provenance_array.item())
    if not provenance.strip():
        return {**unavailable, "reason": "Empty component method provenance"}, None
    labels = np.asarray(data["component_ids"])
    if labels.ndim == 2:
        labels = labels[-1]
    if labels.shape != (len(qf),) or labels.dtype.kind not in "iu" or np.any(labels < -1):
        raise ValueError("Component IDs must be integer particle labels; -1 means unclassified")
    if "substrate_anchor_mask" in data:
        anchors = np.asarray(data["substrate_anchor_mask"])
        if anchors.shape != (len(q0),) or anchors.dtype != np.bool_:
            raise ValueError("substrate_anchor_mask must be a Boolean particle mask")
        anchor_method = "explicit source mask; fixture attachment not independently verified"
    else:
        anchors = q0[:, 2] <= q0[:, 2].min()+1e-7
        anchor_method = "initial lowest particle layer; fixture attachment not independently verified"
    anchor_ids = np.unique(labels[anchors & (labels >= 0)])
    if not len(anchor_ids):
        return {**unavailable, "reason": "No classified substrate anchor component",
                "component_provenance": provenance}, None
    substrate = np.isin(labels, anchor_ids)
    unknown = labels < 0
    detached = (~substrate) & (~unknown)
    components = []
    for label in np.unique(labels[~unknown]):
        mask = labels == label
        m = float(mass[mask].sum())
        if m <= 0:
            continue
        components.append({"id": int(label), "mass_kg": m,
            "contains_reference_anchor": bool(np.any(anchors & mask)),
            "center_of_mass_m": ((mass[mask, None]*qf[mask]).sum(0)/m).tolist(),
            "linear_momentum_kg_m_s": None if velocity is None else (mass[mask, None]*velocity[mask]).sum(0).tolist()})
    components.sort(key=lambda row: row["mass_kg"], reverse=True)
    result = {"status": "SOURCE_REPORTED", "component_provenance": provenance,
              "anchor_method": anchor_method, "component_count": len(components),
              "detached_component_count": sum(not row["contains_reference_anchor"] for row in components),
              "substrate_mass_kg": float(mass[substrate].sum()),
              "detached_mass_kg": float(mass[detached].sum()),
              "unclassified_mass_kg": float(mass[unknown].sum()),
              "partition_residual_kg": float(mass.sum()-mass[substrate].sum()-mass[detached].sum()-mass[unknown].sum()),
              "largest_components": components[:16], "component_list_truncated": len(components)>16,
              "permanent_separation_proven": None, "recontact_proven": None}
    # Unknown particles remain visible to surface metrics; they are never silently removed.
    return result, substrate | unknown


def analyze_recording(path, config=None, run_metadata=None):
    """Measure a real recorded NPZ. Outputs are debug metrics, never observations.

    `component_ids` require explicit `component_provenance`. They report the
    source's mechanical grouping; this function cannot validate its algorithm.
    """
    config = config or MetricConfig()
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    q = np.asarray(data["positions"], dtype=float)
    times = np.asarray(data["time_s"], dtype=float)
    if q.ndim != 3 or q.shape[2] != 3 or min(q.shape[:2]) < 1 or not np.isfinite(q).all():
        raise ValueError("Recorded positions must be finite (T,N,3) with particles and frames")
    if (times.shape != (len(q),) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Recorded timestamps must match frames and increase strictly")
    m = np.asarray(data["mass_kg"], dtype=float)
    static_mass = m.ndim == 1
    if static_mass:
        if m.shape != (q.shape[1],):
            raise ValueError("Static particle mass must have shape(N,)")
        m = np.broadcast_to(m, q.shape[:2])
    if m.shape != q.shape[:2] or not np.isfinite(m).all() or np.any(m < 0) or m[0].sum() <= 0:
        raise ValueError("Mass must be finite nonnegative (N,) or (T,N), with positive initial total")
    mass_total = m.sum(1)
    velocity = None
    if "velocities" in data:
        velocity = np.asarray(data["velocities"], dtype=float)
        if velocity.shape != q.shape or not np.isfinite(velocity).all():
            raise ValueError("Velocities must match positions and be finite")
    if "component_ids" in data and np.asarray(data["component_ids"]).ndim == 2:
        if np.asarray(data["component_ids"]).shape != q.shape[:2]:
            raise ValueError("Temporal component IDs must match all frames and particles")
    component, substrate = _component_budget(data, q[0], q[-1], m[-1], None if velocity is None else velocity[-1])
    final_surface_mass = m[-1] if substrate is None else m[-1]*substrate
    initial_surface = _surface_proxy(q[0], m[0], config)
    final_surface = _surface_proxy(q[-1], final_surface_mass, config)
    outside = ((np.abs(q[-1, :, 0]) > config.length_m/2)
               | (np.abs(q[-1, :, 1]) > config.width_m/2))
    result = {"status": "PASS", "scope": "offline_oracle_diagnostics_only", "benchmark_ready": False,
              "formal_cutting_success": None, "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "metric_config": asdict(config), "frame_count": len(q), "particle_count": q.shape[1],
              "simulation_duration_s": float(times[-1]-times[0]),
              "mass": {"initial_kg": float(mass_total[0]), "final_kg": float(mass_total[-1]),
                       "max_relative_change": float(np.abs(mass_total-mass_total[0]).max()/mass_total[0]),
                       "temporal_measurement": "static masses reused; not an independent per-frame mass measurement" if static_mass else "per-frame masses recorded",
                       "outside_xy_footprint_mass_kg": float(m[-1, outside].sum()),
                       "outside_xy_footprint_reference_volume_m3": float(m[-1, outside].sum()/config.density_kg_m3)},
              "components": component, "initial_surface": initial_surface, "final_surface": final_surface,
              "surface_scope": "particle-bin maximum Z plus configured support radius; empty bins have height zero; not a reconstructed-mesh or RGB-D metric",
              "surface_particle_selection": "source-reported anchor components plus unclassified particles" if substrate is not None else "all particles; unclassified chips may occlude the substrate",
              "minimum_particle_z_m": float(q[:, :, 2].min()),
              "frame_sampled_max_speed_m_s": None if velocity is None else float(np.linalg.norm(velocity, axis=2).max()),
              "kinetic_energy_final_j": None if velocity is None else float((.5*m[-1]*np.sum(velocity[-1]**2, axis=1)).sum()),
              "gravitational_energy_final_j": float((m[-1]*config.gravity_m_s2*q[-1, :, 2]).sum()),
              "total_energy_balance_status": "NOT_EVALUATED; internal strain energy, fixture/tool work and dissipation absent",
              "input_fingerprints": {"initial_positions": array_sha256(q[0]), "initial_mass": array_sha256(m[0]),
                                     "timestamps": array_sha256(times)}}
    if "tool_pose_xyzw" in data:
        tool = np.asarray(data["tool_pose_xyzw"], dtype=float)
        if (tool.shape != (len(q), 7) or not np.isfinite(tool).all()
                or not np.allclose(np.linalg.norm(tool[:, 3:], axis=1), 1, atol=1e-4)):
            raise ValueError("Tool poses must be finite frame-aligned xyz+unit XYZW quaternions")
        result["tool_pose_status"] = "RECORDED"
        result["input_fingerprints"]["tool_poses"] = array_sha256(tool)
        if "tool_half_extents" in data:
            from scipy.spatial.transform import Rotation
            half = np.asarray(data["tool_half_extents"], dtype=float)
            if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                raise ValueError("Tool proxy half extents must be a positive finite 3-vector")
            penetration = 0.
            for points, pose in zip(q, tool):
                local = Rotation.from_quat(pose[3:]).inv().apply(points-pose[:3])
                penetration = max(penetration, float(np.maximum(np.min(half-np.abs(local), axis=1), 0).max()))
            result["tool_box_proxy_particle_center_penetration_m"] = penetration
            result["penetration_scope"] = "oriented bounding-box proxy; not particle-radius-aware or a solver contact certificate"
    else:
        result["tool_pose_status"] = "NOT_RECORDED"
    wall = (run_metadata or {}).get("wall_s")
    if wall is not None and (not np.isfinite(wall) or wall <= 0):
        raise ValueError("Supplied wall_s must be finite and positive")
    result["reported_wall_s"] = wall
    result["simulation_s_per_reported_wall_s"] = None if wall is None else result["simulation_duration_s"]/wall
    return result


def compare_recordings(recordings: dict[str, Path], config=None, metadata=None):
    """Independent reports plus input checks; never rank uncontrolled solver runs."""
    if not recordings:
        raise ValueError("At least one explicit recorded run is required")
    reports = {label: analyze_recording(path, config, (metadata or {}).get(label))
               for label, path in recordings.items()}
    fingerprints = {name: len({r["input_fingerprints"].get(name) for r in reports.values()}) == 1
                    and all(name in r["input_fingerprints"] for r in reports.values())
                    for name in ("initial_positions", "initial_mass", "timestamps", "tool_poses")}
    return {"status": "PASS", "scope": "explicit_recording_comparison_only", "benchmark_ready": False,
            "runs": reports, "recorded_inputs_match": fingerprints,
            "formal_paired_protocol": "NOT_ESTABLISHED; solver material, contact geometry and frozen protocol require separate evidence",
            "solver_ranking": None, "formal_cutting_success": None}
