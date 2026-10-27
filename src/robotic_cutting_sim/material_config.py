"""Dependency-free configuration for the experimental material solver."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TLMPMConfig:
    dx: float = .004
    dt: float = .0001
    density: float = 1500.0
    young: float = 50000.0
    poisson: float = .3
    yield_stress: float = 1000.0
    pic: float = .05
    contact_stiffness: float = .1
    contact_friction: float = .3
    particle_contact_stiffness: float = .05
    contact_radius_scale: float = .9
    cutting_energy: float = .05
    sharpness: float = .8
    blade_length: float = .012
    blade_height: float = .004
    blade_half_width: float = .025
    cuts: int = 2
    floor: bool = True
    fixture_walls: bool = True
    tool: bool = True
    material_contact: bool = True
    gravity: tuple = (0., 0., -9.81)
