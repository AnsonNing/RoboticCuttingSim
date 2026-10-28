"""Explicit opt-in, uncalibrated food-like material profiles for v2 tasks."""
from dataclasses import replace
import hashlib
import json
import numpy as np
from .recording import ROOT

PATH = ROOT / 'configs/materials/food_profiles_v1.json'


def profiles():
    return json.loads(PATH.read_text())['profiles']


def cfl(young, poisson, density, dx, dt):
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    speed = float(np.sqrt((lam + 2 * mu) / density))
    return dict(longitudinal_wave_speed_m_s=speed, grid_cfl=float(dt * speed / dx))


def apply_profile(task, name):
    from .tasks import ProfileTaskSpec
    if not isinstance(task, ProfileTaskSpec):
        raise ValueError('Food profiles require opt-in v2 tasks; frozen v1 recipes stay unchanged')
    profile = profiles()[name]
    recipe = replace(task.material, **{key: value for key, value in profile.items() if key != 'material_dt_s'},
                     calibration='provisional_uncalibrated')
    return replace(task, material=recipe)


def receipt(name, dx=.004, dt=None):
    profile = profiles()[name]
    return dict(profile_id=name, profile=profile, calibration='NOT_VALIDATED',
                config_sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),
                **cfl(profile['young_modulus_pa'], profile['poisson_ratio'], profile['density_kg_m3'],
                      dx, profile['material_dt_s'] if dt is None else dt))
