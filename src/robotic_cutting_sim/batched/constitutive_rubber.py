"""Rubber constitutive registry for the batched rubber path (ADR 0023 §12, D9; task R1.7).

Separate from ``batched/constitutive.py`` (whose hash and v7_2 parameter table belong to the Phase 1 receipts), but
the same protocol: a model declares its id/version, the per-env parameter columns it reads from ``params[N, k]``, its
per-particle internal state (name, dtype, initial value ``identity`` / ``zero`` / ``snapshot``, multiplicity M), a
stress launcher, an energy description and its wave speed (the step rule). The masked reset and the vars()-complete
reset test are driven by ``per_particle_state``, so a model with internal state cannot be reset incompletely.
Registered: ``neo_hookean_jbar_v1`` (no state) and, since P2-3 (ADR 0023 §21.2.4), ``neo_hookean_jbar_prony_v1``:
float64 per-env table ``params64[N, 1 + 4M]`` (``param64_names``: g_inf, (g_i, tau_i, a_i, b_i) x M; a_i, b_i from
expm1 on the host at the batch dt), per-particle state ``visco_q`` (vec6d, multiplicity M) and ``visco_s_prev``
(vec6d), initial value ``zero``; stress launch = J-bar pass + ``tlmpm_rubber_visco_v1.stress_prony`` (the kernel the
single-env ``TLMPMRubberVisco`` launches), guard = ``cfl_guard_prony`` (``stable_step_guard_visco_v1``). M is uniform
per batch (1 <= M <= 4; padding terms have g = 0): ``prony_model(M)``.

Parameter table: float32 ``params[N, NUM_PARAMS]`` with columns RUBBER_PARAM_NAMES; columns 0 and 1 (mu0, kappa) are
also the diagnostics' per-env table. Values are float64 on the host, rounded once to float32 (the single solver
passes the same float64 values as float32 kernel scalars: identical bits).
"""
from dataclasses import dataclass, replace
import math

import numpy as np
import warp as wp

from ..rubber_constitutive import MODEL_ID, MODEL_VERSION, cell_jbar, cfl_effective, det_d, in_domain, nh_jbar_pk1
from .. import rubber_visco_constitutive as rvc
from .errors import UnsupportedBatchConfiguration

RUBBER_PARAM_NAMES = ("mu0", "kappa", "density", "friction_table", "friction_pad", "friction_self")
P_MU = wp.constant(0)
P_KAPPA = wp.constant(1)
P_RHO = wp.constant(2)
P_FRICTION_TABLE = wp.constant(3)
P_FRICTION_PAD = wp.constant(4)
P_FRICTION_SELF = wp.constant(5)
NUM_PARAMS = len(RUBBER_PARAM_NAMES)
DECLARED_LATER = {}


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


@dataclass(frozen=True)
class PronyConstitutiveModel(RubberConstitutiveModel):
    """``neo_hookean_jbar_prony_v1`` with M terms (module docstring; ADR 0023 §21.2)."""
    terms: int = 0
    param64_names: tuple = ()

    def launch_stress(self, solver):
        from ..tlmpm_rubber_visco_v1 import stress_prony
        wp.launch(cell_jbar, solver.total_cells, [solver.cell_node, solver.cell_offsets, solver.cell_particles,
                                                  solver.F, solver.jbar], device=solver.device)
        on = solver.visco_energy_on
        wp.launch(stress_prony, solver.total_particles,
                  [solver.F, solver.nodes, solver.jbar, solver.stress, solver.params, solver.env_of_particle,
                   solver.params64, self.terms, solver.visco_q, solver.visco_s_prev, wp.float64(solver.j_domain[0]),
                   wp.float64(solver.j_domain[1]), solver.violations, wp.float64(solver.dt),
                   wp.float64(solver.volume), int(on), solver.visco_pb if on else None,
                   solver.visco_dissipation if on else None, solver.visco_d_negative if on else None],
                  device=solver.device)

    def launch_guard(self, solver):
        from ..tlmpm_rubber_visco_v1 import cfl_guard_prony
        wp.launch(cfl_guard_prony, solver.total_particles,
                  [solver.F, solver.nodes, solver.jbar, solver.mass, solver.params, solver.env_of_particle,
                   solver.params64, self.terms, solver.visco_q, solver.visco_s_prev, solver.volume,
                   solver.dt/solver.dx, solver.cfl_guard_limit, solver.cfl_violations], device=solver.device)

    def receipt(self):
        out = super().receipt()
        out.update(terms=self.terms, param64_names=list(self.param64_names), guard=rvc.VISCO_GUARD_ID,
                   state_bytes_per_particle=48*(self.terms+1), adr="ADR 0023 §21.2")
        return out


NEO_HOOKEAN_JBAR_PRONY_V1 = PronyConstitutiveModel(
    model_id=rvc.MODEL_ID, version=rvc.MODEL_VERSION, param_names=("mu0", "kappa"), per_particle_state=(),
    energy=("Psi_eq = sum_p V g_inf mu0/2 (J^-2/3 I1 - 3) + sum_c V_c U(Jbar); Psi_v = sum_p V sum_i |Q_i|^2 / "
            "(4 g_i mu0); option visco_energy branch_work_v1: W_b, explicit dissipation D (ADR 0023 §21.2.7)"),
    source="rubber_visco_constitutive.prony_stress_particle (+ frozen nh_jbar_pk1; shared with TLMPMRubberVisco)")


def prony_model(M):
    """The Prony model with M uniform terms (state multiplicity M)."""
    M = int(M)
    if not 1 <= M <= rvc.MAX_TERMS:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_NONUNIFORM_BATCH", f"Prony terms M = {M} not in [1, 4]")
    return replace(NEO_HOOKEAN_JBAR_PRONY_V1, terms=M, param64_names=rvc.param64_names(M),
                   per_particle_state=(RubberStateVariable("visco_q", rvc.vec6d, "zero", M),
                                       RubberStateVariable("visco_s_prev", rvc.vec6d, "zero", 1)))


REGISTRY = {NEO_HOOKEAN_JBAR_V1.model_id: NEO_HOOKEAN_JBAR_V1,
            NEO_HOOKEAN_JBAR_PRONY_V1.model_id: NEO_HOOKEAN_JBAR_PRONY_V1}
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
        if variable.dtype is rvc.vec6d:
            return np.zeros(shape+(6,), np.float64)
        return np.zeros(shape, np.float32)
    raise ValueError(f"initial value {variable.initial!r} needs a snapshot")
