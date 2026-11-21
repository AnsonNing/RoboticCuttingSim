"""Opt-in numerical profiles for the v2 Rubber strip-drag task."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RubberSpeedProfile:
    n_t: int
    cfl: float
    beta: float
    precision: str = "f64_shadow_v1"
    projection_momentum: str = "delta_scatter_v1"


# Nominal CFL 0.8 passed short drag/release runs but tripped the Prony guard
# on the stiff/light seed's longer closed-loop trajectory.
SPEED_PROFILES = {
    "rubber_fast_v1": RubberSpeedProfile(n_t=3, cfl=0.65, beta=10.0),
    "rubber_turbo_v1": RubberSpeedProfile(n_t=2, cfl=0.65, beta=5.0, precision="float32_v1"),
}
