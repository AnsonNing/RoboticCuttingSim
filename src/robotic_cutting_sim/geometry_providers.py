"""Versioned, opt-in workpiece geometry providers on the regular particle lattice (WP-12 Phase 4c, spec 4.5).

A scene workpiece without ``geometry_provider`` keeps ``heightfield_v1`` exactly: ``benchmark.HeightFieldSpec``
with ``benchmark.Bump`` rows sampled by ``benchmark.sample_specimen`` (every existing scene id, task id, receipt
and particle set is unchanged). The opt-in providers are:

``heightfield_v2``
    Still ONE continuous height field over a rectangular base box, z in [0, h(x, y)]. Adds, per protrusion row,
    ``yaw_deg`` (rotation of the bump's own x/y axes about local z: elliptical bumps at any heading) and
    ``peak_offset`` in [-0.8, 0.8] (the peak moves along the bump's own x axis by that fraction of
    ``radius_x_m``; each flank is the family profile stretched to its side, so the footprint is two half-ellipses
    joined at the peak, inside the declared rectangle [-radius_x, radius_x] x [-radius_y, radius_y] of the bump
    frame: a skewed / asymmetric bump with a steep and a shallow flank), the ``irregular`` family of ``benchmark.Bump`` (compact asymmetric cap), and an optional
    seeded low-amplitude ``surface_noise`` added to the whole top (sum of random-phase plane cosines,
    |noise| <= amplitude_m). Overlapping bumps are the max-union of ``HeightFieldSpec`` (one body, never two).
``sdf_csg_experimental_v1`` (EXPERIMENTAL)
    The base box minus a list of primitives (``cylinder`` along x or y, ``box``, ``sphere``), voxelised by
    cell-centre inside tests on the same lattice. It can express through-holes, tunnels, undercuts and
    overhangs that no height field can. It is not an imported-mesh sampler: only these analytic primitives,
    no protrusions, no fractional boundary volumes; goals that need a top height field (remove_protrusions,
    trim surface metrics) are unsupported on it.

What no provider here expresses: imported meshes / arbitrary closed surfaces, multiple separate bodies, curved
(non-rectangular) footprints other than the legacy capsule path (not wired into scenes), and sub-cell geometry
(every retained cell has the full density * spacing**3 mass; curved boundaries are cell-centre terraces).

Sampling contract (all providers): cell centres q = (ijk + 1/2) s - (L/2, W/2, 0) of the box lattice
``ceil((L, W, H_max) / s)``, the strict ``q_x < L/2, q_y < W/2`` rule of ``sample_specimen``, float32 points,
int32 lattice indices, fixed density * s**3 masses. ``heightfield_v2`` without yaw / offset / noise samples the
identical particle set to ``sample_specimen`` (tested). ``lattice_report`` checks 6-connectivity (no floating
particles), contact of the bottom layer with the support plane (k = 0 at z = s/2) and counts.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .benchmark import FAMILIES, Bump, HeightFieldSpec

DEFAULT_PROVIDER = "heightfield_v1"
HEIGHTFIELD_V2 = "heightfield_v2"
SDF_CSG = "sdf_csg_experimental_v1"
PROVIDERS = (DEFAULT_PROVIDER, HEIGHTFIELD_V2, SDF_CSG)
V2_FAMILIES = FAMILIES+("irregular",)
MAX_PEAK_OFFSET = .8
CSG_SHAPES = ("cylinder", "box", "sphere")

LIMITATIONS = {
    "heightfield_v1": "One max-union height field z <= h(x, y) over a rectangular base box; axis-aligned elliptical "
                      "bumps of families dome, gaussian, ellipsoid_cap, ridge.",
    "heightfield_v2": "Adds bump yaw, skew (peak_offset), the irregular family and seeded low-amplitude surface "
                      "noise. Still a height field: no undercuts, overhangs, holes, tunnels or separate bodies.",
    "sdf_csg_experimental_v1": "Base box minus cylinders/boxes/spheres on the lattice (holes, tunnels, undercuts, "
                               "overhangs). EXPERIMENTAL; no protrusions, no top height field, no imported meshes.",
    "not_supported_by_any_provider": ["imported triangle meshes or arbitrary closed SDFs",
                                      "several separate bodies in one workpiece",
                                      "non-rectangular scene footprints (capsule exists only for legacy tasks)",
                                      "fractional (sub-cell) boundary volumes",
                                      "workpiece yaw relative to the flat board (flat_board_v1 has no oriented "
                                      "TLMPM fixture boxes; the board must be co-rotated)"],
}


class GeometryProviderError(ValueError):
    """A geometry provider description is invalid (scene_config maps it to SceneConfigError/SceneInfeasible)."""


@dataclass(frozen=True)
class OrientedBump(Bump):
    """``heightfield_v2`` protrusion: a ``benchmark.Bump`` shape in its own rotated, optionally skewed frame."""

    yaw_deg: float = 0.
    peak_offset: float = 0.

    def __post_init__(self):
        if self.family not in V2_FAMILIES or not np.isfinite([self.x_m, self.y_m, self.radius_x_m, self.radius_y_m,
                                                               self.height_m, self.yaw_deg, self.peak_offset]).all() \
                or min(self.radius_x_m, self.radius_y_m, self.height_m) <= 0:
            raise ValueError("Bumps need a known family and finite positive size")
        if abs(self.peak_offset) > MAX_PEAK_OFFSET:
            raise ValueError(f"peak_offset must lie in [-{MAX_PEAK_OFFSET}, {MAX_PEAK_OFFSET}]")

    def half_extents_xy(self):
        """Conservative axis-aligned half extents of the footprint in the workpiece frame."""
        return protrusion_half_extents({"radius_x_m": self.radius_x_m, "radius_y_m": self.radius_y_m,
                                        "yaw_deg": self.yaw_deg, "peak_offset": self.peak_offset})

    def height(self, x, y):
        dx, dy = np.asarray(x)-self.x_m, np.asarray(y)-self.y_m
        yaw = math.radians(self.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        u, v = (c*dx+s*dy)/self.radius_x_m, (-s*dx+c*dy)/self.radius_y_m
        p = self.peak_offset
        if p:
            u = np.where(u >= p, (u-p)/(1-p), (u-p)/(1+p))
        return Bump(0., 0., 1., 1., self.height_m, self.family).height(u, v)


def _noise_modes(amplitude, wavelength, seed, modes):
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(0., 2*np.pi, modes)
    k = 2*np.pi/wavelength*rng.uniform(.75, 1.25, modes)
    phase = rng.uniform(0., 2*np.pi, modes)
    weight = rng.uniform(.5, 1., modes)
    return np.c_[k*np.cos(theta), k*np.sin(theta)], phase, weight*amplitude/weight.sum()


class ProviderGeometry:
    """Mixin marking a non-default provider geometry (``ConfiguredTaskSpec.sample_particles`` dispatches on it)."""

    provider = None

    def inside(self, q):
        raise NotImplementedError

    @property
    def max_height_m(self):
        raise NotImplementedError


@dataclass(frozen=True)
class HeightFieldSpecV2(ProviderGeometry, HeightFieldSpec):
    surface_noise_amplitude_m: float = 0.
    surface_noise_wavelength_m: float = .012
    surface_noise_seed: int = 0
    surface_noise_modes: int = 8
    provider: str = HEIGHTFIELD_V2

    def __post_init__(self):
        if (not np.isfinite([self.length_m, self.width_m, self.base_height_m]).all()
                or min(self.length_m, self.width_m, self.base_height_m) <= 0):
            raise ValueError("Workpiece dimensions must be finite and positive")
        for bump in self.bumps:
            hx, hy = bump.half_extents_xy() if isinstance(bump, OrientedBump) else (bump.radius_x_m, bump.radius_y_m)
            if abs(bump.x_m)+hx >= self.length_m/2 or abs(bump.y_m)+hy >= self.width_m/2:
                raise ValueError("Every bump must lie within the continuous base")
        a = self.surface_noise_amplitude_m
        if not np.isfinite(a) or a < 0 or not a < self.base_height_m:
            raise ValueError("surface_noise amplitude must be finite, >= 0 and below the base height")
        if a and (not self.surface_noise_wavelength_m > 0 or not 1 <= int(self.surface_noise_modes) <= 64):
            raise ValueError("surface_noise needs a positive wavelength and 1..64 modes")

    def noise(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        if not self.surface_noise_amplitude_m:
            return np.zeros(x.shape)
        k, phase, weight = _noise_modes(self.surface_noise_amplitude_m, self.surface_noise_wavelength_m,
                                        self.surface_noise_seed, int(self.surface_noise_modes))
        out = np.zeros(x.shape)
        for (kx, ky), ph, w in zip(k, phase, weight):
            out += w*np.cos(kx*x+ky*y+ph)
        return out

    def height(self, x, y):
        base = HeightFieldSpec.height(self, x, y)
        return base+self.noise(x, y) if self.surface_noise_amplitude_m else base

    @property
    def max_height_m(self):
        return self.base_height_m+max((b.height_m for b in self.bumps), default=0.)+self.surface_noise_amplitude_m

    @property
    def min_height_m(self):
        return self.base_height_m-self.surface_noise_amplitude_m

    def inside(self, q):
        return q[:, 2] <= self.height(q[:, 0], q[:, 1])


def _primitive_inside(primitive, q):
    shape = primitive[0]
    if shape == "cylinder":
        _, axis, center, radius = primitive
        a, b = (1, 2) if axis == "x" else (0, 2)
        return (q[:, a]-center[0])**2+(q[:, b]-center[1])**2 <= radius**2
    if shape == "box":
        _, center, half = primitive
        return np.all(np.abs(q-np.asarray(center))<=np.asarray(half), axis=1)
    _, center, radius = primitive
    return np.sum((q-np.asarray(center))**2, axis=1) <= radius**2


@dataclass(frozen=True)
class CSGSolidSpec(ProviderGeometry, HeightFieldSpec):
    """EXPERIMENTAL base box minus primitives. ``height`` is the box top (the upper envelope only where no
    primitive reaches the top face); it is informational, not a height-field description of the solid."""

    subtract: tuple = ()
    bumps: tuple = field(default=())
    provider: str = SDF_CSG

    def __post_init__(self):
        if (not np.isfinite([self.length_m, self.width_m, self.base_height_m]).all()
                or min(self.length_m, self.width_m, self.base_height_m) <= 0):
            raise ValueError("Workpiece dimensions must be finite and positive")
        if self.bumps:
            raise ValueError("sdf_csg_experimental_v1 takes no protrusions")
        for primitive in self.subtract:
            if primitive[0] not in CSG_SHAPES:
                raise ValueError(f"CSG primitives are {CSG_SHAPES}")

    def height(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        return np.full(x.shape, self.base_height_m)

    @property
    def max_height_m(self):
        return self.base_height_m

    @property
    def min_height_m(self):
        return self.base_height_m

    def inside(self, q):
        keep = q[:, 2] <= self.base_height_m
        for primitive in self.subtract:
            keep &= ~_primitive_inside(primitive, q)
        return keep


# ----------------------------------------------------------------------------- sampling and validation

def sample_lattice(geometry, spacing_m, density_kg_m3):
    """(points float32, masses float64, lattice int32) on the ``sample_specimen`` lattice; no labels."""
    s = float(spacing_m)
    if not np.isfinite(s) or not 0 < s <= geometry.min_height_m/2:
        raise ValueError("Spacing must provide at least two layers in the base")
    L, W = geometry.length_m, geometry.width_m
    dims = np.ceil(np.array([L, W, geometry.max_height_m])/s).astype(int)
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in dims), indexing="ij"), -1).reshape(-1, 3)
    q = (ijk+.5)*s-[L/2, W/2, 0]
    inside = (q[:, 0] < L/2) & (q[:, 1] < W/2) & geometry.inside(q)
    q, ijk = q[inside].astype(np.float32), ijk[inside].astype(np.int32)
    return q, np.full(len(q), density_kg_m3*s**3, dtype=np.float64), ijk


def lattice_report(lattice):
    """6-connected components of the sampled lattice, floating particles and bottom-layer (board) contact."""
    from scipy import ndimage
    lattice = np.asarray(lattice, np.int64)
    if not len(lattice):
        return {"particles": 0, "components": 0, "floating_particles": 0, "bottom_layer_particles": 0,
                "ok": False, "problems": ["empty_sample"]}
    grid = np.zeros(lattice.max(0)+1, bool)
    grid[tuple(lattice.T)] = True
    labels, count = ndimage.label(grid)   # default structure = 6-connectivity
    per = labels[tuple(lattice.T)]
    bottom = lattice[:, 2] == 0
    grounded = np.unique(per[bottom])
    floating = int((~np.isin(per, grounded)).sum())
    columns = np.unique(lattice[:, :2], axis=0)
    bottom_columns = np.unique(lattice[bottom][:, :2], axis=0)
    problems = []
    if count != 1:
        problems.append("disconnected_material")
    if floating:
        problems.append("floating_material")
    if not bottom.any():
        problems.append("no_support_contact")
    return {"particles": int(len(lattice)), "components": int(count), "floating_particles": floating,
            "bottom_layer_particles": int(bottom.sum()), "footprint_columns": int(len(columns)),
            "columns_touching_support": int(len(bottom_columns)),
            "columns_touching_support_fraction": float(len(bottom_columns)/len(columns)),
            "lattice_dims": (lattice.max(0)+1).tolist(), "ok": not problems, "problems": problems,
            "rule": "scipy.ndimage.label 6-connectivity on the occupied lattice; support contact = layer k = 0 "
                    "(cell centre z = spacing/2 above the support plane)"}


# ----------------------------------------------------------------------------- scene schema helpers

def _num(value, where, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise GeometryProviderError(f"{where} must be a finite number")
    if positive and not value > 0:
        raise GeometryProviderError(f"{where} must be positive")
    if nonnegative and value < 0:
        raise GeometryProviderError(f"{where} must be nonnegative")
    return float(value)


def _only(mapping, where, required=(), optional=()):
    if not isinstance(mapping, dict):
        raise GeometryProviderError(f"{where} must be a JSON object")
    unknown, missing = set(mapping)-set(required)-set(optional), [k for k in required if k not in mapping]
    if unknown or missing:
        raise GeometryProviderError(f"{where}: unknown keys {sorted(unknown)} / missing keys {missing}")


def provider_id(workpiece):
    """Declared provider id of a raw or normalized workpiece dict (absent = heightfield_v1)."""
    provider = workpiece.get("geometry_provider")
    if provider is None:
        return DEFAULT_PROVIDER
    if not isinstance(provider, dict) or provider.get("id") not in PROVIDERS:
        raise GeometryProviderError(f"workpiece.geometry_provider.id must be one of {PROVIDERS}")
    return provider["id"]


def validate_provider(provider, base):
    """Normalized ``workpiece.geometry_provider`` (None for heightfield_v1, so default scenes stay unchanged)."""
    if provider is None or (isinstance(provider, dict) and provider.get("id") == DEFAULT_PROVIDER and len(provider) == 1):
        return None
    pid = provider_id({"geometry_provider": provider})
    where = "workpiece.geometry_provider"
    if pid == DEFAULT_PROVIDER:
        raise GeometryProviderError(f"{where}: heightfield_v1 takes no parameters")
    if pid == HEIGHTFIELD_V2:
        _only(provider, where, ("id",), ("surface_noise",))
        out = {"id": pid}
        if "surface_noise" in provider:
            noise = provider["surface_noise"]
            _only(noise, f"{where}.surface_noise", ("amplitude_m", "wavelength_m", "seed"), ("modes",))
            seed = noise["seed"]
            modes = noise.get("modes", 8)
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise GeometryProviderError(f"{where}.surface_noise.seed must be a nonnegative integer")
            if isinstance(modes, bool) or not isinstance(modes, int) or not 1 <= modes <= 64:
                raise GeometryProviderError(f"{where}.surface_noise.modes must be an integer in [1, 64]")
            out["surface_noise"] = {"amplitude_m": _num(noise["amplitude_m"], f"{where}.surface_noise.amplitude_m",
                                                        nonnegative=True),
                                    "wavelength_m": _num(noise["wavelength_m"], f"{where}.surface_noise.wavelength_m",
                                                         positive=True),
                                    "seed": int(seed), "modes": int(modes)}
        return out
    _only(provider, where, ("id", "subtract"))
    rows = provider["subtract"]
    if not isinstance(rows, list) or not rows:
        raise GeometryProviderError(f"{where}.subtract must be a nonempty list")
    out = []
    for i, row in enumerate(rows):
        w = f"{where}.subtract[{i}]"
        shape = row.get("shape") if isinstance(row, dict) else None
        if shape == "cylinder":
            _only(row, w, ("shape", "axis", "center_m", "radius_m"))
            if row["axis"] not in ("x", "y"):
                raise GeometryProviderError(f"{w}.axis must be 'x' or 'y' (center_m = the other horizontal axis, z)")
            center = row["center_m"]
            if not isinstance(center, list) or len(center) != 2:
                raise GeometryProviderError(f"{w}.center_m must be [horizontal, z]")
            out.append({"shape": shape, "axis": row["axis"], "center_m": [_num(c, f"{w}.center_m") for c in center],
                        "radius_m": _num(row["radius_m"], f"{w}.radius_m", positive=True)})
        elif shape in ("box", "sphere"):
            extra = "half_extents_m" if shape == "box" else "radius_m"
            _only(row, w, ("shape", "center_m", extra))
            if not isinstance(row["center_m"], list) or len(row["center_m"]) != 3:
                raise GeometryProviderError(f"{w}.center_m must be 3 numbers")
            item = {"shape": shape, "center_m": [_num(c, f"{w}.center_m") for c in row["center_m"]]}
            if shape == "box":
                half = row["half_extents_m"]
                if not isinstance(half, list) or len(half) != 3:
                    raise GeometryProviderError(f"{w}.half_extents_m must be 3 numbers")
                item["half_extents_m"] = [_num(h, f"{w}.half_extents_m", positive=True) for h in half]
            else:
                item["radius_m"] = _num(row["radius_m"], f"{w}.radius_m", positive=True)
            out.append(item)
        else:
            raise GeometryProviderError(f"{w}.shape must be one of {CSG_SHAPES}")
    return {"id": pid, "subtract": out}


def protrusion_extra_keys(pid):
    return ("yaw_deg", "peak_offset") if pid == HEIGHTFIELD_V2 else ()


def protrusion_families(pid):
    return V2_FAMILIES if pid == HEIGHTFIELD_V2 else FAMILIES


def normalize_protrusion_extras(pid, row, where):
    if pid != HEIGHTFIELD_V2:
        return {}
    yaw = _num(row.get("yaw_deg", 0.), f"{where}.yaw_deg")
    offset = _num(row.get("peak_offset", 0.), f"{where}.peak_offset")
    if abs(offset) > MAX_PEAK_OFFSET:
        raise GeometryProviderError(f"{where}.peak_offset must lie in [-{MAX_PEAK_OFFSET}, {MAX_PEAK_OFFSET}]")
    return {"yaw_deg": yaw, "peak_offset": offset}


def protrusion_half_extents(row):
    """Conservative local axis-aligned (half x, half y) of a scene protrusion row.

    radius_x/y exactly unless the row declares a yaw; with a yaw, the rotated footprint ellipse (no peak offset)
    or the rotated bounding rectangle of the bump frame (peak offset: the two half-ellipses are inside it)."""
    yaw = row.get("yaw_deg", 0.)
    rx, ry = row["radius_x_m"], row["radius_y_m"]
    if not yaw:
        return rx, ry
    c, s = abs(math.cos(math.radians(yaw))), abs(math.sin(math.radians(yaw)))
    if row.get("peak_offset", 0.):
        return rx*c+ry*s, rx*s+ry*c
    return math.hypot(rx*c, ry*s), math.hypot(rx*s, ry*c)


def geometry_from_scene(workpiece):
    """HeightFieldSpecV2 / CSGSolidSpec of a normalized non-default workpiece (None for heightfield_v1)."""
    provider = workpiece.get("geometry_provider")
    if provider is None:
        return None
    base = workpiece["base_box_m"]
    if provider["id"] == HEIGHTFIELD_V2:
        bumps = tuple(OrientedBump(r["x_m"], r["y_m"], r["radius_x_m"], r["radius_y_m"], r["height_m"], r["family"],
                                   r["yaw_deg"], r["peak_offset"]) for r in workpiece["protrusions"])
        noise = provider.get("surface_noise") or {}
        return HeightFieldSpecV2(base["length_m"], base["width_m"], base["height_m"], bumps,
                                 surface_noise_amplitude_m=noise.get("amplitude_m", 0.),
                                 surface_noise_wavelength_m=noise.get("wavelength_m", .012),
                                 surface_noise_seed=noise.get("seed", 0), surface_noise_modes=noise.get("modes", 8))
    if workpiece["protrusions"]:
        raise GeometryProviderError("sdf_csg_experimental_v1 takes no protrusions")
    primitives = []
    for row in provider["subtract"]:
        if row["shape"] == "cylinder":
            primitives.append(("cylinder", row["axis"], tuple(row["center_m"]), row["radius_m"]))
        elif row["shape"] == "box":
            primitives.append(("box", tuple(row["center_m"]), tuple(row["half_extents_m"])))
        else:
            primitives.append(("sphere", tuple(row["center_m"]), row["radius_m"]))
    return CSGSolidSpec(base["length_m"], base["width_m"], base["height_m"], subtract=tuple(primitives))


def capability_receipt(geometry):
    pid = getattr(geometry, "provider", None) or DEFAULT_PROVIDER
    return {"provider": pid, "limitations": LIMITATIONS[pid],
            "not_supported_by_any_provider": LIMITATIONS["not_supported_by_any_provider"],
            "experimental": pid == SDF_CSG}
