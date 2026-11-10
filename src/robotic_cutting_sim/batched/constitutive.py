"""Batched constitutive models (ADR 0021 §11 extension point).

A model declares its id/version, the per-env parameters it reads from the
batch ``params[N, k]`` table, its per-particle internal state (name, dtype,
trailing shape and declared initial value) and a stress kernel launched at
step 6 of the substep (ADR §3.3). The registry drives partial reset, snapshots,
receipts and the reset-equals-fresh test, so a future model with more internal
variables cannot be reset incompletely without failing that test.

The only model today is ``log_strain_von_mises_plastic_v5``: the frozen
``tlmpm_v5._stress`` body (Hencky log strain, perfect von Mises plasticity)
with lam, mu and yield stress read per env. It owns ``Fpi`` and ``plastic``.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

# Column layout of the per-env float32 parameter table params[N, NUM_PARAMS].
PARAM_NAMES = ("lam", "mu", "yield_stress", "contact_stiffness", "contact_friction", "cutting_energy",
               "sharpness", "slice_efficiency", "particle_contact_stiffness")
P_LAM = wp.constant(0)
P_MU = wp.constant(1)
P_YIELD = wp.constant(2)
P_STIFFNESS = wp.constant(3)
P_FRICTION = wp.constant(4)
P_GC = wp.constant(5)
P_SHARPNESS = wp.constant(6)
P_SLICE = wp.constant(7)
P_PARTICLE_STIFFNESS = wp.constant(8)
NUM_PARAMS = len(PARAM_NAMES)


def env_parameters(config, slice_efficiency):
    """float64 values computed exactly as ``tlmpm_v5.TLMPM.step`` does, in PARAM_NAMES order.

    Python floats are float64; the single solver passes them as float32 kernel
    scalars (round to nearest), and the batch stores ``np.float32`` of the same
    float64 values (also round to nearest), so both kernels see identical bits.
    """
    c = config
    lam = c.young*c.poisson/((1+c.poisson)*(1-2*c.poisson))
    mu = c.young/(2*(1+c.poisson))
    return (lam, mu, c.yield_stress, c.contact_stiffness, c.contact_friction, c.cutting_energy, c.sharpness,
            float(slice_efficiency), c.particle_contact_stiffness)


@dataclass(frozen=True)
class StateVariable:
    """A per-particle internal variable: flat [P, *shape] array with a declared initial value."""
    name: str
    dtype: object
    initial: str   # "identity" (mat33 I) or "zero"


@wp.kernel
def _stress_log_strain_v5(F: wp.array(dtype=wp.mat33), Fpi: wp.array(dtype=wp.mat33),
                          stress: wp.array(dtype=wp.mat33), plastic: wp.array(dtype=float),
                          params: wp.array2d(dtype=float), env_of_particle: wp.array(dtype=int)):
    """tlmpm_v5._stress (tlmpm_v5.py:77-99) with lam, mu, yield stress of the particle's env."""
    p = wp.tid()
    e = env_of_particle[p]
    lam = params[e, P_LAM]
    mu = params[e, P_MU]
    yield_stress = params[e, P_YIELD]
    fe = F[p]*Fpi[p]
    u, sigma, v = wp.svd3(fe)
    # Clamp only for finite evaluation; the host separately rejects inverted or
    # extreme total deformation, rather than silently accepting this safeguard.
    logs = wp.vec3(wp.log(wp.max(sigma[0], 0.05)), wp.log(wp.max(sigma[1], 0.05)),
                   wp.log(wp.max(sigma[2], 0.05)))
    tr = logs[0]+logs[1]+logs[2]
    dev = logs-wp.vec3(tr/3.0)
    norm = wp.length(dev)
    limit = wp.sqrt(2.0/3.0)*yield_stress/(2.0*mu)
    if norm > limit:
        delta = norm-limit
        logs = logs-delta*dev/norm
        corrected = u*wp.diag(wp.vec3(wp.exp(logs[0]), wp.exp(logs[1]), wp.exp(logs[2])))*wp.transpose(v)
        Fpi[p] = wp.inverse(F[p])*corrected
        plastic[p] = plastic[p]+delta
    tau = u*wp.diag(2.0*mu*logs+wp.vec3(lam*tr))*wp.transpose(u)
    stress[p] = tau*wp.transpose(wp.inverse(F[p]))


@dataclass(frozen=True)
class ConstitutiveModel:
    model_id: str
    version: int
    param_names: tuple
    per_particle_state: tuple   # StateVariable rows owned by the model
    stress_kernel: object
    stress_arguments: tuple    # solver attribute names, in kernel argument order
    source: str

    def launch_stress(self, solver):
        """Step 6 of the substep: (F, state..., params[env], env_of_particle) -> stress."""
        wp.launch(self.stress_kernel, solver.total_particles,
                  [getattr(solver, name) for name in self.stress_arguments], device=solver.device)

    def receipt(self):
        return {"model_id": self.model_id, "version": self.version, "param_names": list(self.param_names),
                "per_particle_state": {v.name: {"dtype": getattr(v.dtype, "__name__", str(v.dtype)),
                                                "initial": v.initial} for v in self.per_particle_state},
                "source": self.source}


LOG_STRAIN_VON_MISES_PLASTIC_V5 = ConstitutiveModel(
    model_id="log_strain_von_mises_plastic_v5", version=1, param_names=("lam", "mu", "yield_stress"),
    per_particle_state=(StateVariable("Fpi", wp.mat33, "identity"), StateVariable("plastic", float, "zero")),
    stress_kernel=_stress_log_strain_v5,
    stress_arguments=("F", "Fpi", "stress", "plastic", "params", "env_of_particle"),
    source="tlmpm_v5._stress (tlmpm_v5.py:77-99), per-env lam/mu/yield")

REGISTRY = {LOG_STRAIN_VON_MISES_PLASTIC_V5.model_id: LOG_STRAIN_VON_MISES_PLASTIC_V5}
DEFAULT_MODEL_ID = LOG_STRAIN_VON_MISES_PLASTIC_V5.model_id


def get_model(model_id):
    from .errors import UnsupportedBatchConfiguration
    if model_id not in REGISTRY:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS",
                                            f"unknown constitutive model {model_id!r}")
    return REGISTRY[model_id]


def initial_host_value(variable, count):
    """Host array of ``count`` particles holding the declared initial value."""
    if variable.initial == "identity":
        return np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    if variable.initial == "zero":
        return np.zeros(count, np.float32)
    raise ValueError(f"unknown initial value {variable.initial!r}")
