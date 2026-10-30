"""``materials_wp12_v1``: interpretable, PROVISIONAL_UNCALIBRATED TLMPM material profiles (WP-12 Phase 4c, spec 4.6).

Opt-in via ``rcs_scene_v1`` ``workpiece.material_profile = "wp12_..."``. ``cutting_tasks_v1`` and the
``food_profiles_v1`` ids keep their exact behaviour. Each profile is a full ``benchmark.MaterialParameters``
recipe (density, E, nu, J2 yield, Coulomb friction, law-8 cutting energy; hardening fixed 0) with a one-line
interpretation, never a colour. ``admit`` enforces the declared elastic-wave CFL rule and refuses an unstable
(material dt, grid dx) combination before any solver is built.
"""
import hashlib
import json

import numpy as np

from .recording import ROOT

PATH = ROOT/"configs/materials/materials_wp12_v1.json"
VERSION = "materials_wp12_v1"
GRID_CFL_LIMIT = .2
RECIPE_KEYS = ("density_kg_m3", "young_modulus_pa", "poisson_ratio", "yield_stress_pa", "friction",
               "fracture_energy_j_m2")


class MaterialStabilityError(ValueError):
    """The requested material dt violates the profile's elastic-wave CFL admission rule."""


def document():
    return json.loads(PATH.read_text())


def profiles():
    return document()["profiles"]


def is_profile(name):
    return isinstance(name, str) and name in profiles()


def wave_speed(young, poisson, density):
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    return float(np.sqrt((lam+2*mu)/density))


def stability(name, grid_dx_m, material_dt_s, particle_spacing_m=None):
    p = profiles()[name]
    c = wave_speed(p["young_modulus_pa"], p["poisson_ratio"], p["density_kg_m3"])
    grid_cfl = float(material_dt_s*c/grid_dx_m)
    out = {"profile_id": name, "longitudinal_wave_speed_m_s": c, "grid_dx_m": float(grid_dx_m),
           "material_dt_s": float(material_dt_s), "grid_cfl": grid_cfl, "grid_cfl_limit": GRID_CFL_LIMIT,
           "max_admissible_dt_s": GRID_CFL_LIMIT*grid_dx_m/c, "declared_material_dt_s": p["material_dt_s"],
           "admitted": bool(grid_cfl <= GRID_CFL_LIMIT+1e-12 and material_dt_s <= p["material_dt_s"]+1e-15)}
    if particle_spacing_m is not None:
        out["particle_spacing_cfl_report_only"] = float(material_dt_s*c/particle_spacing_m)
    return out


def admit(name, grid_dx_m, material_dt_s, particle_spacing_m=None):
    """Stability receipt, or MaterialStabilityError if the combination is refused."""
    receipt = stability(name, grid_dx_m, material_dt_s, particle_spacing_m)
    if not receipt["admitted"]:
        raise MaterialStabilityError(
            f"{name}: material dt {material_dt_s:g} s gives grid CFL {receipt['grid_cfl']:.3f} on dx {grid_dx_m:g} m "
            f"(limit {GRID_CFL_LIMIT}; profile declares dt <= {receipt['declared_material_dt_s']:g} s, CFL-admissible "
            f"dt <= {receipt['max_admissible_dt_s']:.3g} s); refused")
    return receipt


def recipe(name, base):
    """``base`` (MaterialParameters) with the profile's recipe, hardening 0, labelled provisional_uncalibrated."""
    from dataclasses import replace
    p = profiles()[name]
    return replace(base, **{key: float(p[key]) for key in RECIPE_KEYS}, hardening_ratio=0.,
                   calibration="provisional_uncalibrated")


def receipt(name, grid_dx_m=.004, material_dt_s=None, particle_spacing_m=.001):
    p = profiles()[name]
    dt = p["material_dt_s"] if material_dt_s is None else material_dt_s
    return {"version": VERSION, "profile_id": name, "profile": p, "calibration": "PROVISIONAL_UNCALIBRATED",
            "config_sha256": hashlib.sha256(PATH.read_bytes()).hexdigest(),
            "stability": stability(name, grid_dx_m, dt, particle_spacing_m)}
