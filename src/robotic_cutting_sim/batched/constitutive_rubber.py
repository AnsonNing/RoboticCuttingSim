"""Rubber constitutive registry for the batched rubber path (ADR 0023 §12, D9; task R1.7).

Separate from ``batched/constitutive.py`` (whose hash and v7_2 parameter table belong to the Phase 1 receipts), but
the same protocol: a model declares its id/version, the per-env parameter columns it reads from ``params[N, k]``, its
per-particle internal state (name, dtype, initial value ``identity`` / ``zero`` / ``snapshot``, multiplicity M), a
stress launcher, an energy description and its wave speed (the step rule). The masked reset and the vars()-complete
reset test are driven by ``per_particle_state``, so a later model (P2-3 ``neo_hookean_jbar_prony_v1`` with
``visco_q[P*M]`` and ``visco_s_prev[P]``) cannot be reset incompletely. Only ``neo_hookean_jbar_v1`` is registered.

Parameter table: float32 ``params[N, NUM_PARAMS]`` with columns RUBBER_PARAM_NAMES; columns 0 and 1 (mu0, kappa) are
also the diagnostics' per-env table. Values are float64 on the host, rounded once to float32 (the single solver
passes the same float64 values as float32 kernel scalars: identical bits).
"""
from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..rubber_constitutive import MODEL_ID, MODEL_VERSION, cell_jbar, cfl_effective, det_d, in_domain, nh_jbar_pk1
from .errors import UnsupportedBatchConfiguration

RUBBER_PARAM_NAMES = ("mu0", "kappa", "density", "friction_table", "friction_pad", "friction_self")
P_MU = wp.constant(0)
P_KAPPA = wp.constant(1)
P_RHO = wp.constant(2)
P_FRICTION_TABLE = wp.constant(3)
P_FRICTION_PAD = wp.constant(4)
P_FRICTION_SELF = wp.constant(5)
NUM_PARAMS = len(RUBBER_PARAM_NAMES)
DECLARED_LATER = {"neo_hookean_jbar_prony_v1": "P2-3 (ADR 0023 §10 design; needs probe evidence first)"}


def env_parameters(material):
    """float64 per-env values in RUBBER_PARAM_NAMES order from a ``rubber_material.MaterialValues``."""
    return (float(material.mu0), float(material.kappa), float(material.density), float(material.friction_table),
            float(material.friction_pad), float(material.friction_self))


def wave_speed(mu0, kappa, density):
    return math.sqrt((kappa+4*mu0/3)/density)


@dataclass(frozen=True)
class RubberStateVariable:
    name: str
    dtype: object
    initial: str          # "identity" | "zero" | "snapshot"
    multiplicity: int = 1


@wp.kernel
def _stress_b(F: wp.array(dtype=wp.mat33), nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64),
              stress: wp.array(dtype=wp.mat33), params: wp.array2d(dtype=float), env_of_particle: wp.array(dtype=int),
              lo: wp.float64, hi: wp.float64, violations: wp.array(dtype=int)):
    """``tlmpm_rubber_v1._stress`` with mu0, kappa of the particle's env and a per-env violation counter."""
    p = wp.tid()
    e = env_of_particle[p]
    f = F[p]
    jb = jbar[nodes[p, 0]]
    if not (in_domain(det_d(f), lo, hi) and in_domain(jb, lo, hi)):
        wp.atomic_add(violations, e, 1)
    stress[p] = nh_jbar_pk1(f, jb, params[e, P_MU], params[e, P_KAPPA])


@wp.kernel
def _cfl_guard_b(F: wp.array(dtype=wp.mat33), nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64),
                 mass: wp.array(dtype=float), params: wp.array2d(dtype=float), env_of_particle: wp.array(dtype=int),
                 volume: float, dt_over_dx: float, limit: float, cfl_violations: wp.array(dtype=int)):
    """``tlmpm_rubber_v1._cfl_guard`` with mu0, kappa of the particle's env and a per-env counter."""
    p = wp.tid()
    e = env_of_particle[p]
    c = cfl_effective(F[p], jbar[nodes[p, 0]], params[e, P_MU], params[e, P_KAPPA], mass[p]/volume, dt_over_dx)
    if not (c <= limit):
        wp.atomic_add(cfl_violations, e, 1)


@dataclass(frozen=True)
class RubberConstitutiveModel:
    model_id: str
    version: int
    param_names: tuple
    per_particle_state: tuple
    energy: str
    source: str

    def launch_stress(self, solver):
        """J-bar pass (deterministic cell gather over the flat batch) then the per-env stress pass."""
        wp.launch(cell_jbar, solver.total_cells, [solver.cell_node, solver.cell_offsets, solver.cell_particles,
                                                  solver.F, solver.jbar], device=solver.device)
        wp.launch(_stress_b, solver.total_particles, [solver.F, solver.nodes, solver.jbar, solver.stress,
                                                      solver.params, solver.env_of_particle,
                                                      wp.float64(solver.j_domain[0]), wp.float64(solver.j_domain[1]),
                                                      solver.violations], device=solver.device)

    def launch_guard(self, solver):
        """Opt-in stable-step guard pass (after the stress pass; reads F and the J-bar of this substep)."""
        wp.launch(_cfl_guard_b, solver.total_particles, [solver.F, solver.nodes, solver.jbar, solver.mass,
                                                         solver.params, solver.env_of_particle, solver.volume,
                                                         solver.dt/solver.dx, solver.cfl_guard_limit,
                                                         solver.cfl_violations], device=solver.device)

    def receipt(self):
        return {"model_id": self.model_id, "version": self.version, "param_names": list(self.param_names),
                "per_particle_state": {v.name: {"dtype": getattr(v.dtype, "__name__", str(v.dtype)),
                                                "initial": v.initial, "multiplicity": v.multiplicity}
                                       for v in self.per_particle_state},
                "energy": self.energy, "source": self.source, "registry": "rubber (batched/constitutive_rubber.py)"}


NEO_HOOKEAN_JBAR_V1 = RubberConstitutiveModel(
    model_id=MODEL_ID, version=MODEL_VERSION, param_names=("mu0", "kappa"), per_particle_state=(),
    energy="sum_p V mu/2 (J^-2/3 I1 - 3) + sum_c V_c kappa/4 (Jbar^2 - 1 - 2 ln Jbar) (segmented diagnostics)",
    source="rubber_constitutive.nh_jbar_pk1 + cell_jbar (shared with tlmpm_rubber_v1)")

REGISTRY = {NEO_HOOKEAN_JBAR_V1.model_id: NEO_HOOKEAN_JBAR_V1}
DEFAULT_MODEL_ID = NEO_HOOKEAN_JBAR_V1.model_id


def get_model(model_id):
    if model_id not in REGISTRY:
        later = DECLARED_LATER.get(model_id)
        raise UnsupportedBatchConfiguration("UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS",
                                            f"rubber model {model_id!r} not registered"
                                            + (f" (declared for {later})" if later else ""))
    return REGISTRY[model_id]


def initial_host_value(variable, count):
    shape = (count*variable.multiplicity,)
    if variable.initial == "identity":
        return np.tile(np.eye(3, dtype=np.float32), (shape[0], 1, 1))
    if variable.initial == "zero":
        return np.zeros(shape, np.float32)
    raise ValueError(f"initial value {variable.initial!r} needs a snapshot")
