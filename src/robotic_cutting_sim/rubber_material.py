"""``rubber_material_v1`` recipe: loader, validator and derived quantities (ADR 0023 §5, §6, §14.1; task R0.1).

A recipe is a JSON document (schema ``rcs_rubber_material_v1``, SI units). Inputs are the equilibrium shear
modulus ``mu_eq``, the numerical bulk ratio ``beta`` (kappa = beta * mu0), the density, optional isochoric Prony
terms and contact/numerics settings. Poisson's ratio is never an input (it is derived); unknown keys are refused at
every level; ``pic`` must be 0 (pure FLIP, ADR §7). Every value is provisional: there is no real rubber data on this
machine, so the only admissible status without a frozen held-out calibration is ``SIMULATOR_ONLY_UNCALIBRATED``.

Derived (never input): mu0 = mu_eq / g_inf, kappa = beta mu0, nu_eff, E_eff, c_s, c_p, dt_max(dx) = cfl dx / c_p,
the step rule (dt rounded down so that control_dt / dt is an integer, c_p,max over a declared randomisation range)
and the Prony update coefficients a_i, b_i (float64, ADR §10).
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "rcs_rubber_material_v1"
CALIBRATION_STATUSES = ("SIMULATOR_ONLY_UNCALIBRATED", "PILOT_FIT_NOT_HELDOUT", "REAL_CALIBRATED")
ELASTIC_MODEL = "neo_hookean_jbar_v1"
PRONY_MODEL = "neo_hookean_jbar_prony_v1"
MODELS = (ELASTIC_MODEL, PRONY_MODEL)
MAX_PRONY_TERMS = 4
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = ROOT/"configs"/"materials"/"rubber_generic_soft_v1.json"
CALIBRATION_DIR = ROOT/"configs"/"rubber"/"calibration"

# ADR 0023 §8 resolution profiles (ppc per axis, minimum cells through the thickness, label).
RESOLUTION_PROFILES = {
    "rubber_ref_v1": {"particles_per_cell_axis": 2, "min_cells_through_thickness": 4, "label": "full"},
    "rubber_rl_v1": {"particles_per_cell_axis": 2, "min_cells_through_thickness": 3, "label": "full"},
    "rubber_coarse_v1": {"particles_per_cell_axis": 2, "min_cells_through_thickness": 2,
                         "label": "reduced_fidelity: coarse_thickness_resolution"},
}

_TOP_KEYS = {"schema", "id", "calibration_status", "material_identity", "density_kg_m3", "model", "elastic",
             "viscoelastic", "contact", "numerics", "derived", "provenance", "calibration_manifest",
             "heldout_verdict"}
_REQUIRED_TOP = _TOP_KEYS - {"derived", "calibration_manifest", "heldout_verdict"}
_ELASTIC_KEYS = {"shear_modulus_eq_pa", "bulk_ratio_numerical", "bulk_modulus_physical_pa"}
_VISCO_KEYS = {"applies_to", "max_terms", "prony"}
_PRONY_KEYS = {"g", "tau_s"}
_CONTACT_KEYS = {"friction_table", "friction_pad", "friction_self"}
_NUMERICS_KEYS = {"resolution_profile", "particles_per_cell_axis", "min_cells_through_thickness", "cfl", "pic",
                  "contact_stiffness_fraction", "self_contact_stiffness_fraction", "contact_radius_scale",
                  "j_domain", "j_physical_tolerance"}
_IDENTITY_KEYS = {"family", "candidates", "evidence"}
_PROVENANCE_KEYS = {"defined", "values"}
_REFUSED_INPUTS = {"poisson_ratio", "nu", "young_modulus_pa", "lame_lambda", "bulk_modulus_pa"}


class RubberRecipeError(ValueError):
    """A recipe that violates ADR 0023 §14.1 (message names the rule)."""


def _refuse(message):
    raise RubberRecipeError(message)


def _check_keys(obj, allowed, where, required=None):
    if not isinstance(obj, dict):
        _refuse(f"{where} must be an object")
    for key in obj:
        if key in _REFUSED_INPUTS:
            _refuse(f"{where}.{key}: refused input (nu/E/lambda are derived from mu and beta, ADR 0023 §6)")
    unknown = set(obj) - set(allowed)
    if unknown:
        _refuse(f"{where}: unknown keys {sorted(unknown)}")
    missing = set(required if required is not None else allowed) - set(obj)
    if missing:
        _refuse(f"{where}: missing keys {sorted(missing)}")


def _finite_positive(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        _refuse(f"{where} must be a finite positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Ranged:
    """A value with an admissible (or randomisation) range [lo, hi]."""
    value: float
    lo: float
    hi: float

    @classmethod
    def parse(cls, obj, where):
        _check_keys(obj, {"value", "range"}, where)
        value = _finite_positive(obj["value"], f"{where}.value")
        rng = obj["range"]
        if not isinstance(rng, list) or len(rng) != 2:
            _refuse(f"{where}.range must be [lo, hi]")
        lo, hi = (_finite_positive(v, f"{where}.range") for v in rng)
        if not lo <= value <= hi:
            _refuse(f"{where}: value {value} outside its range [{lo}, {hi}]")
        return cls(value, lo, hi)

    def to_dict(self):
        return {"value": self.value, "range": [self.lo, self.hi]}


@dataclass(frozen=True)
class PronyTerm:
    g: float
    tau_s: float


@dataclass(frozen=True)
class RubberRecipe:
    """Validated ``rubber_material_v1`` recipe (immutable). Build with :func:`load_recipe` / :func:`parse_recipe`."""
    id: str
    calibration_status: str
    model: str
    density: Ranged
    shear_modulus_eq: Ranged
    bulk_ratio: Ranged
    bulk_modulus_physical_pa: float
    bulk_modulus_physical_status: str
    prony: tuple
    friction_table: Ranged
    friction_pad: Ranged
    friction_self: Ranged
    resolution_profile: str
    particles_per_cell_axis: int
    min_cells_through_thickness: int
    cfl: float
    pic: float
    contact_stiffness_fraction: float
    self_contact_stiffness_fraction: float
    contact_radius_scale: float
    j_domain: tuple
    j_physical_tolerance: float
    material_identity: dict
    provenance: dict
    calibration_manifest: str = None
    heldout_verdict: str = None
    source_sha256: str = None
    raw: dict = field(default=None, compare=False, repr=False)

    # ------------------------------------------------------------------ derived quantities (float64)

    @property
    def g_inf(self):
        return 1.0-sum(t.g for t in self.prony)

    def mu0(self, mu_eq=None):
        """Instantaneous shear modulus mu0 = mu_eq / g_inf (= mu_eq for the elastic model)."""
        return float(self.shear_modulus_eq.value if mu_eq is None else mu_eq)/self.g_inf

    def kappa(self, mu_eq=None, beta=None):
        return float(self.bulk_ratio.value if beta is None else beta)*self.mu0(mu_eq)

    def derived(self, mu_eq=None, density=None, beta=None, dx=None):
        mu0 = self.mu0(mu_eq)
        kappa = self.kappa(mu_eq, beta)
        rho = float(self.density.value if density is None else density)
        out = {"mu0_pa": mu0, "mu_eq_pa": mu0*self.g_inf, "g_inf": self.g_inf, "kappa_pa": kappa,
               "beta": kappa/mu0, "nu_effective": poisson_from(mu0, kappa), "young_effective_pa": young_from(mu0, kappa),
               "lame_lambda_pa": kappa-2*mu0/3, "c_s_m_s": math.sqrt(mu0/rho), "c_p_m_s": p_wave_speed(mu0, kappa, rho),
               "density_kg_m3": rho, "cfl": self.cfl,
               "bulk_modulus_physical_pa": self.bulk_modulus_physical_pa,
               "bulk_modulus_physical_status": self.bulk_modulus_physical_status}
        if dx is not None:
            out["dx_m"] = float(dx)
            out["dt_max_s"] = self.cfl*float(dx)/out["c_p_m_s"]
        return out

    def nominal(self):
        """The recipe's nominal per-env material values."""
        return MaterialValues(mu0=self.mu0(), kappa=self.kappa(), density=self.density.value,
                              friction_table=self.friction_table.value, friction_pad=self.friction_pad.value,
                              friction_self=self.friction_self.value)

    def values(self, mu_eq=None, density=None, beta=None, friction_table=None, friction_pad=None,
               friction_self=None):
        """Per-env values with optional overrides; every override must lie inside the recipe's range."""
        def pick(name, ranged, value):
            if value is None:
                return ranged.value
            value = _finite_positive(value, name)
            if not ranged.lo <= value <= ranged.hi:
                _refuse(f"{name} = {value} outside the recipe range [{ranged.lo}, {ranged.hi}]")
            return value
        mu_eq = pick("shear_modulus_eq_pa", self.shear_modulus_eq, mu_eq)
        beta = pick("bulk_ratio_numerical", self.bulk_ratio, beta)
        return MaterialValues(mu0=self.mu0(mu_eq), kappa=self.kappa(mu_eq, beta),
                              density=pick("density_kg_m3", self.density, density),
                              friction_table=pick("friction_table", self.friction_table, friction_table),
                              friction_pad=pick("friction_pad", self.friction_pad, friction_pad),
                              friction_self=pick("friction_self", self.friction_self, friction_self))

    def step_rule(self, dx, control_dt, *, mu_eq_max=None, density_min=None, beta=None, cfl=None):
        """ADR §7 step rule: dt <= cfl dx / c_p,max with c_p,max over the declared randomisation range
        (mu_eq max, density min; defaults: the recipe ranges), rounded down so that control_dt / dt is an integer."""
        cfl = self.cfl if cfl is None else float(cfl)
        mu_max = self.shear_modulus_eq.hi if mu_eq_max is None else float(mu_eq_max)
        rho_min = self.density.lo if density_min is None else float(density_min)
        return step_rule(dx, control_dt, mu0_max=self.mu0(mu_max), kappa_max=self.kappa(mu_max, beta),
                         density_min=rho_min, cfl=cfl)

    def prony_coefficients(self, dt):
        """Exact exponential-integrator coefficients (ADR §10) in float64: a_i = exp(-dt/tau_i),
        b_i = -g_i (tau_i/dt) expm1(-dt/tau_i)."""
        dt = _finite_positive(dt, "dt")
        return [(math.exp(-dt/t.tau_s), -t.g*(t.tau_s/dt)*math.expm1(-dt/t.tau_s)) for t in self.prony]

    def resolution(self):
        profile = RESOLUTION_PROFILES[self.resolution_profile]
        return {"resolution_profile": self.resolution_profile, **profile}

    def receipt(self):
        return {"schema": SCHEMA, "id": self.id, "model": self.model, "calibration_status": self.calibration_status,
                "source_sha256": self.source_sha256, "derived_nominal": self.derived(),
                "prony": [{"g": t.g, "tau_s": t.tau_s} for t in self.prony], "resolution": self.resolution()}


@dataclass(frozen=True)
class MaterialValues:
    """Concrete per-env material values (SI): instantaneous shear modulus, bulk modulus, density, frictions."""
    mu0: float
    kappa: float
    density: float
    friction_table: float = 0.8
    friction_pad: float = 1.0
    friction_self: float = 0.9

    @property
    def beta(self):
        return self.kappa/self.mu0

    def c_p(self):
        return p_wave_speed(self.mu0, self.kappa, self.density)

    def young_effective(self):
        return young_from(self.mu0, self.kappa)

    def poisson_effective(self):
        return poisson_from(self.mu0, self.kappa)

    def to_dict(self):
        return {"mu0_pa": self.mu0, "kappa_pa": self.kappa, "density_kg_m3": self.density, "beta": self.beta,
                "friction_table": self.friction_table, "friction_pad": self.friction_pad,
                "friction_self": self.friction_self, "nu_effective": self.poisson_effective(),
                "young_effective_pa": self.young_effective(), "c_p_m_s": self.c_p()}


def poisson_from(mu, kappa):
    return (3*kappa-2*mu)/(2*(3*kappa+mu))


def young_from(mu, kappa):
    return 9*kappa*mu/(3*kappa+mu)


def p_wave_speed(mu, kappa, rho):
    return math.sqrt((kappa+4*mu/3)/rho)


def step_rule(dx, control_dt, *, mu0_max, kappa_max, density_min, cfl=0.5):
    """dt = control_dt / ceil(control_dt / (cfl dx / c_p,max)); returns a receipt dict."""
    dx = _finite_positive(dx, "dx")
    control_dt = _finite_positive(control_dt, "control_dt")
    if not 0 < cfl <= 1.0:
        _refuse(f"cfl must be in (0, 1], got {cfl} (measured critical CFL 1.2-1.5, ADR §2 E2)")
    c_p = p_wave_speed(mu0_max, kappa_max, density_min)
    dt_max = cfl*dx/c_p
    substeps = int(math.ceil(control_dt/dt_max*(1-1e-12)))
    dt = control_dt/substeps
    return {"dx_m": dx, "control_dt_s": control_dt, "cfl_declared": cfl, "c_p_max_m_s": c_p, "dt_max_s": dt_max,
            "dt_s": dt, "substeps_per_control_step": substeps, "cfl_actual": dt*c_p/dx,
            "margin_to_measured_critical_1_2": 1.2/(dt*c_p/dx), "mu0_max_pa": mu0_max, "kappa_max_pa": kappa_max,
            "density_min_kg_m3": density_min}


# ----------------------------------------------------------------------------- parsing

def parse_recipe(obj, *, source_sha256=None, manifest_dir=None):
    """Validate a recipe dict (ADR §14.1 rules) and return a :class:`RubberRecipe`."""
    _check_keys(obj, _TOP_KEYS, "recipe", required=_REQUIRED_TOP)
    if obj["schema"] != SCHEMA:
        _refuse(f"schema must be {SCHEMA!r}")
    if not isinstance(obj["id"], str) or not obj["id"]:
        _refuse("id must be a non-empty string")
    status = obj["calibration_status"]
    if status not in CALIBRATION_STATUSES:
        _refuse(f"calibration_status {status!r} not in {CALIBRATION_STATUSES}")
    if "derived" in obj and not isinstance(obj["derived"], str):
        _refuse("derived values are computed by the loader and never input ('derived' may only be a note string)")
    _check_keys(obj["material_identity"], _IDENTITY_KEYS, "material_identity")
    _check_keys(obj["provenance"], _PROVENANCE_KEYS, "provenance")
    density = Ranged.parse(obj["density_kg_m3"], "density_kg_m3")
    model = obj["model"]
    if model not in MODELS:
        _refuse(f"model {model!r} not in {MODELS}")
    el = obj["elastic"]
    _check_keys(el, _ELASTIC_KEYS, "elastic")
    mu = Ranged.parse(el["shear_modulus_eq_pa"], "elastic.shear_modulus_eq_pa")
    beta = Ranged.parse(el["bulk_ratio_numerical"], "elastic.bulk_ratio_numerical")
    kphys = el["bulk_modulus_physical_pa"]
    _check_keys(kphys, {"value", "status"}, "elastic.bulk_modulus_physical_pa")
    kphys_value = _finite_positive(kphys["value"], "elastic.bulk_modulus_physical_pa.value")
    if kphys["status"] not in ("LITERATURE_TYPICAL_UNMEASURED", "MEASURED"):
        _refuse("elastic.bulk_modulus_physical_pa.status must be LITERATURE_TYPICAL_UNMEASURED or MEASURED")
    ve = obj["viscoelastic"]
    _check_keys(ve, _VISCO_KEYS, "viscoelastic")
    if ve["applies_to"] != "isochoric":
        _refuse("viscoelastic.applies_to must be 'isochoric' (no volumetric viscosity, ADR §10)")
    if ve["max_terms"] != MAX_PRONY_TERMS:
        _refuse(f"viscoelastic.max_terms must be {MAX_PRONY_TERMS}")
    if not isinstance(ve["prony"], list) or len(ve["prony"]) > MAX_PRONY_TERMS:
        _refuse(f"viscoelastic.prony must be a list of at most {MAX_PRONY_TERMS} terms")
    terms = []
    for i, term in enumerate(ve["prony"]):
        _check_keys(term, _PRONY_KEYS, f"viscoelastic.prony[{i}]")
        g, tau = term["g"], term["tau_s"]
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) or g < 0:
            _refuse(f"viscoelastic.prony[{i}].g must be finite and >= 0")
        terms.append(PronyTerm(float(g), _finite_positive(tau, f"viscoelastic.prony[{i}].tau_s")))
    if sum(t.g for t in terms) >= 1.0:
        _refuse("sum of Prony g_i must be < 1 (g_inf > 0)")
    if (model == PRONY_MODEL) != bool(terms):
        _refuse(f"model {model!r} does not match the Prony terms ({PRONY_MODEL} iff prony is non-empty)")
    ct = obj["contact"]
    _check_keys(ct, _CONTACT_KEYS, "contact")
    frictions = {k: Ranged.parse(ct[k], f"contact.{k}") for k in sorted(_CONTACT_KEYS)}
    nm = obj["numerics"]
    _check_keys(nm, _NUMERICS_KEYS, "numerics")
    profile = nm["resolution_profile"]
    if profile not in RESOLUTION_PROFILES:
        _refuse(f"numerics.resolution_profile {profile!r} not in {sorted(RESOLUTION_PROFILES)}")
    expected = RESOLUTION_PROFILES[profile]
    if nm["particles_per_cell_axis"] != expected["particles_per_cell_axis"]:
        _refuse("numerics.particles_per_cell_axis must be 2 (1 per cell is under-integrated, ADR §2 E1)")
    if nm["min_cells_through_thickness"] != expected["min_cells_through_thickness"]:
        _refuse(f"numerics.min_cells_through_thickness must be {expected['min_cells_through_thickness']} "
                f"for {profile}")
    if isinstance(nm["pic"], bool) or nm["pic"] != 0:
        _refuse("numerics.pic must be 0: rubber runs pure FLIP; PIC blending is dt-dependent numerical damping "
                "(ADR 0023 §2 E3, D4)")
    cfl = _finite_positive(nm["cfl"], "numerics.cfl")
    if cfl > 0.5:
        _refuse("numerics.cfl must be <= 0.5 (ADR §7 step rule; measured critical 1.2-1.5)")
    for key in ("contact_stiffness_fraction", "self_contact_stiffness_fraction"):
        v = _finite_positive(nm[key], f"numerics.{key}")
        if v > 1:
            _refuse(f"numerics.{key} must be in (0, 1]")
    radius = _finite_positive(nm["contact_radius_scale"], "numerics.contact_radius_scale")
    jd = nm["j_domain"]
    if (not isinstance(jd, list) or len(jd) != 2 or not 0 < float(jd[0]) < 1 < float(jd[1])
            or not all(math.isfinite(float(v)) for v in jd)):
        _refuse("numerics.j_domain must be [lo, hi] with 0 < lo < 1 < hi")
    jtol = _finite_positive(nm["j_physical_tolerance"], "numerics.j_physical_tolerance")
    manifest = obj.get("calibration_manifest")
    verdict = obj.get("heldout_verdict")
    if status == "REAL_CALIBRATED":
        if not manifest or verdict != "PASS":
            _refuse("REAL_CALIBRATED requires calibration_manifest and heldout_verdict 'PASS' (P2-5)")
        from .rubber_calibration import load_manifest
        directory = Path(manifest_dir) if manifest_dir is not None else CALIBRATION_DIR
        path = directory/f"{manifest}.json"
        if not path.is_file():
            _refuse(f"REAL_CALIBRATED: calibration manifest {manifest!r} not found in {directory}")
        loaded = load_manifest(path, check_files=False)
        if loaded["status"] != "FROZEN_FOR_HELDOUT":
            _refuse(f"REAL_CALIBRATED: manifest {manifest!r} has status {loaded['status']!r}, "
                    "not FROZEN_FOR_HELDOUT")
    elif verdict is not None and verdict not in ("PASS", "FAIL", "NOT_RUN"):
        _refuse("heldout_verdict must be PASS, FAIL or NOT_RUN")
    return RubberRecipe(
        id=obj["id"], calibration_status=status, model=model, density=density, shear_modulus_eq=mu,
        bulk_ratio=beta, bulk_modulus_physical_pa=kphys_value, bulk_modulus_physical_status=kphys["status"],
        prony=tuple(terms), friction_table=frictions["friction_table"], friction_pad=frictions["friction_pad"],
        friction_self=frictions["friction_self"], resolution_profile=profile,
        particles_per_cell_axis=int(nm["particles_per_cell_axis"]),
        min_cells_through_thickness=int(nm["min_cells_through_thickness"]), cfl=cfl, pic=0.0,
        contact_stiffness_fraction=float(nm["contact_stiffness_fraction"]),
        self_contact_stiffness_fraction=float(nm["self_contact_stiffness_fraction"]), contact_radius_scale=radius,
        j_domain=(float(jd[0]), float(jd[1])), j_physical_tolerance=jtol,
        material_identity=dict(obj["material_identity"]), provenance=dict(obj["provenance"]),
        calibration_manifest=manifest, heldout_verdict=verdict, source_sha256=source_sha256, raw=obj)


def load_recipe(path=DEFAULT_RECIPE, **kwargs):
    """Load and validate a recipe JSON file (receipts carry its SHA-256)."""
    data = Path(path).read_bytes()
    return parse_recipe(json.loads(data), source_sha256=hashlib.sha256(data).hexdigest(), **kwargs)


def resolution_check(recipe, thickness_min, dx, profile=None):
    """n_t = t_min / dx against the profile's minimum (ADR §8); returns the label or raises."""
    name = recipe.resolution_profile if profile is None else profile
    if name not in RESOLUTION_PROFILES:
        _refuse(f"unknown resolution profile {name!r}")
    n_t = float(thickness_min)/float(dx)
    required = RESOLUTION_PROFILES[name]["min_cells_through_thickness"]
    if n_t < required-1e-9:
        _refuse(f"{name}: {n_t:.3f} cells through the thickness < {required}")
    return {"resolution_profile": name, "cells_through_thickness": n_t,
            "label": RESOLUTION_PROFILES[name]["label"]}


def lattice_check(length, spacing):
    """Geometry on the particle lattice: ``length`` must be an integer multiple of ``spacing`` (ADR §8)."""
    k = float(length)/float(spacing)
    if abs(k-round(k)) > 1e-6 or round(k) < 1:
        _refuse(f"length {length} is not a positive multiple of the particle spacing {spacing}")
    return int(round(k))


def mass_per_particle(density, spacing):
    """rho s^3 in float64 then float32: one function for the single and the batched solver (same bits)."""
    return np.float32(float(density)*float(spacing)**3)
