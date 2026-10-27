"""One connected volume, sampled once; no presegmented bump or particle removal."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Specimen:
    length: float = 0.120
    width: float = 0.080
    base_height: float = 0.015
    bump_height: float = 0.008
    bump_radius: float = 0.015
    center_x: float = 0.0
    center_y: float = 0.0
    family: str = "dome"
    seed: int = 42

    def __post_init__(self):
        values = [self.length, self.width, self.base_height, self.bump_height, self.bump_radius]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Specimen dimensions must be positive and finite")
        if self.family not in ("dome", "gaussian", "ellipsoid_cap", "ridge"):
            raise ValueError(f"Unknown shape family: {self.family}")
        if abs(self.center_x) + self.bump_radius >= self.length / 2:
            raise ValueError("Bump must fit inside the base in X")
        if abs(self.center_y) + self.bump_radius >= self.width / 2:
            raise ValueError("Bump must fit inside the base in Y")

    @classmethod
    def generate(cls, seed=42, variant="D0", family="dome"):
        if variant == "D0":
            return cls(seed=seed, family=family)
        if variant != "D1":
            raise ValueError("Only D0 and D1 have been implemented")
        rng = np.random.default_rng(seed)
        return cls(seed=seed, family=family, center_x=float(rng.uniform(-0.025, 0.025)),
                   center_y=float(rng.uniform(-0.010, 0.010)),
                   bump_height=float(rng.uniform(0.005, 0.010)),
                   bump_radius=float(rng.uniform(0.010, 0.015)))

    def height(self, x, y):
        x = (np.asarray(x) - self.center_x) / self.bump_radius
        y = (np.asarray(y) - self.center_y) / self.bump_radius
        r = np.sqrt(x*x + y*y)
        if self.family == "dome":
            bump = np.where(r < 1, np.cos(np.minimum(r, 1)*np.pi/2)**2, 0)
        elif self.family == "gaussian":
            bump = np.maximum((np.exp(-4*r*r) - np.exp(-4)) / (1-np.exp(-4)), 0)
        elif self.family == "ellipsoid_cap":
            bump = np.sqrt(np.maximum(1-r*r, 0))
        else:
            bump = np.maximum(1-x*x, 0)**2 * np.maximum(1-y*y, 0)**2
        return self.base_height + self.bump_height*bump


def sample_particles(spec: Specimen, spacing: float, density: float):
    if not np.isfinite([spacing, density]).all() or min(spacing, density) <= 0:
        raise ValueError("Spacing and density must be positive and finite")
    if spacing > spec.base_height / 2:
        raise ValueError("At least two particle layers are required in the base")
    dims = np.floor(np.array([spec.length, spec.width,
                             spec.base_height + spec.bump_height]) / spacing).astype(int)
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in dims), indexing="ij"), -1).reshape(-1, 3)
    points = (ijk + 0.5)*spacing - [spec.length/2, spec.width/2, 0]
    inside = points[:, 2] <= spec.height(points[:, 0], points[:, 1])
    points, ijk = points[inside].astype(np.float32), ijk[inside].astype(np.int32)
    mass = np.full(len(points), density*spacing**3, dtype=np.float64)
    return points, mass, ijk


def specimen_dict(spec, points, mass, spacing):
    return {**asdict(spec), "units": "m,kg,s", "spacing_m": spacing,
            "particle_count": len(points), "initial_mass_kg": float(mass.sum()),
            "particle_volume_m3": spacing**3,
            "sampling": "uniform cell centers under one continuous height function"}
