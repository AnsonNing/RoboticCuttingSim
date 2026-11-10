"""Batched rubber material ``BatchedRubberMaterial`` for ``rl_gpu_headless`` (ADR 0023 D1, D9, D10; task R1.7).

What ``LiveRubberMaterial.advance`` does for one env, for all N envs at once:

1. every group holds (start = end, one copy of the [N*G] tables), then the caller's group-motion END values
   (``set_group_motions``: rotation / translation / affine [N, G, ...], active [N, G]) and traction forces are written;
2. the control step's substeps run as CUDA graph chunks (``graph_chunk``, default 500, plus a remainder graph; captured
   once per batch shape) or eagerly on CPU (``parity_mode`` only); the device substep index carries across replays;
3. a kernel converts the per-group reaction impulses to WORLD = strip-local wrenches ``wrench[N, G, 6]`` (force the
   material exerts on the group, averaged over the step); there is no robot feedback in P2-1 (the 20 ms delayed
   handoff of a robot coupling belongs to P2-4 / P2-6);
4. fixed-order segmented diagnostics (``summary[N, 18]``, markers ``[N*K]``) and a validity kernel set the sticky
   ``invalid_reason[N]`` bits (``NONFINITE_STATE`` 1, ``RUBBER_J_DOMAIN`` 2, ``RUBBER_CFL_EXCEEDED`` 4 with the opt-in
   stable-step guard) and ``invalid_count[1]``; strict mode
   reads that one int and raises with env ids.

``reset(env_ids, seeds, materials)`` is a masked device reset (no allocation, no recapture) to the freshly packed state,
optionally with per-env material values (mu0, kappa, density, frictions) resampled inside the declared randomisation
ranges (staged in preallocated host-written buffers; the step rule's c_p,max covers the ranges). A geometry change on
partial reset is refused (``rebuild``). There is no wp.HashGrid in rubber v1 (no self-contact), so the process-level
device-HashGrid guard is not applicable. The ``open_vector_env`` dispatch to this material is deferred to P2-6.

P2-3 prerequisites (ADR 0023 §20): lives built with the P2-2 opt-in options (F8 projection, F9 grid traction work,
F10 float64 shadows) run batched; the masked reset also restores their state (``OPTION_RESET_ARRAYS``: shadows to the
reference, projection scratch and the discarded v1 traction work to zero) and, for an env reset with new material
values, the F8 block inverses are recomputed on the host with the single solver's ``projection_blocks`` (the mass
enters the consistent-mass blocks) and written into the preallocated inverse (no device allocation, no recapture).

P2-3 viscoelasticity (ADR 0023 §21.2.4-21.2.5; ``neo_hookean_jbar_prony_v1`` from the lives): the masked reset runs a
generic registry-state reset kernel per declared state variable of the model (``visco_q`` with multiplicity M,
``visco_s_prev``; initial value ``zero``), writes ``params64`` from its stage (``stage_params64``, host-staged float64
rows with a_i, b_i from expm1 at the batch dt) and clears the ``visco_energy`` accumulators, so an env never inherits
material memory (RUBBER-VISC-06). Per-reset randomisation of the Prony terms: declared ranges ``prony_g_scale``
(multiplies every g_i; sum g < 1 enforced) and ``prony_tau_scale`` (multiplies every tau_i), relative to the env's
packed terms. After the frozen diagnostics the visco diagnostics (``visco_summary[N, 8]``: Psi_eq, Psi_v, W_b, D, ...)
are launched (fixed order; the single solver's kernels).
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import warp as wp

from .. import tlmpm_rubber_v1 as T
from .. import rubber_visco_constitutive as rvc
from . import constitutive_rubber as cr
from . import rubber_layout
from .errors import UnsupportedBatchConfiguration
from .rubber_batch import BatchedTLMPMRubber

wp.set_module_options({"enable_backward": False})

RUNTIME = "rl_gpu_headless"
VERSION = "batched_rubber_material_v1"
INVALID_NONFINITE_STATE = 1
INVALID_RUBBER_J_DOMAIN = 2
INVALID_RUBBER_CFL_EXCEEDED = 4
INVALID_REASONS = {INVALID_NONFINITE_STATE: "NONFINITE_STATE", INVALID_RUBBER_J_DOMAIN: "RUBBER_J_DOMAIN",
                   INVALID_RUBBER_CFL_EXCEEDED: "RUBBER_CFL_EXCEEDED"}
DIAG_GROUPS = T.DIAG_GROUPS   # fixed reduction width shared with the single reference: same summation order
DEFAULT_GRAPH_CHUNK = 500

# vars()-complete reset classification (tests/test_rubber_batched.py)
SOLVER_RESET_ARRAYS = ("x", "v", "x_start", "impulse", "stress", "F", "mass", "gm", "gp", "gj", "old", "new_v", "jbar",
                       "group_active", "group_q0", "group_q1", "group_b0", "group_b1", "group_A0", "group_A1",
                       "reaction", "traction_force", "traction_mass", "work", "violations", "cfl_violations",
                       "params")
SOLVER_CONSTANT_ARRAYS = ("reference", "nodes", "weights", "gradients", "slot", "slot_code", "particle_traction",
                          "node_group", "node_mask", "node_X", "group_kind", "group_origin", "cell_node",
                          "cell_offsets", "cell_particles", "marker_offsets", "marker_particles", "plane", "box_c",
                          "box_h", "env_of_particle", "env_of_node", "env_of_group", "particle_offset", "node_offset",
                          "cell_env_offset")
SOLVER_CONSTANT_ARRAYS += ("bc_nodes",)
SOLVER_SHARED_ARRAYS = ("index",)
SOLVER_RECOMPUTED_ARRAYS = ("motion_M", "motion_b")      # bc_launch motion_table_v1: every control step
# option arrays (exist only when the lives were built with the option; ADR 0023 §20)
OPTION_RESET_ARRAYS = {"state_precision": ("F64", "x64"), "traction_work": ("work_scratch",),
                       "dirichlet": ("proj_rhs", "proj_vpresc", "proj_mu", "proj_mu_node", "proj_inv"),
                       "constraint_work": ("work_end_velocity",)}
OPTION_CONSTANT_ARRAYS = {"state_precision": ("reference64",), "traction_work": (), "constraint_work": (),
                          "dirichlet": ("proj_row_node", "proj_row_comp", "proj_row_group", "proj_row_block",
                                        "proj_row_local", "proj_block_offset", "proj_block_size", "proj_inv_offset",
                                        "env_of_row")}
MATERIAL_RESET_ARRAYS = ("invalid_reason", "step_count", "seed", "wrench")
MATERIAL_CONSTANT_ARRAYS = ("initial_params", "initial_mass", "initial_group_active", "env_of_traction",
                            "traction_counts", "initial_seed")
RESET_INPUTS = ("reset_mask", "reset_seed", "stage_params", "stage_mass")
RECOMPUTED_EVERY_CONTROL_STEP = ("invalid_count",)
DIAGNOSTICS_SCRATCH = ("pp", "pc", "pn", "summary", "marker_x", "marker_v")
# neo_hookean_jbar_prony_v1 (ADR 0023 §21.2.5; exist only for a Prony batch)
MODEL_STATE_RESET_ARRAYS = {"neo_hookean_jbar_prony_v1": ("visco_q", "visco_s_prev", "params64")}
VISCO_ENERGY_RESET_ARRAYS = ("visco_pb", "visco_branch_work", "visco_dissipation", "visco_d_negative")
VISCO_ENERGY_V2_RESET_ARRAYS = ("visco_df_prev", "visco_branch_work_left")   # branch_work_kick_average_v2 (§21.7)
VISCO_MATERIAL_CONSTANT_ARRAYS = ("initial_params64",)
VISCO_RESET_INPUTS = ("stage_params64",)
VISCO_DIAGNOSTICS_SCRATCH = ("vpp", "visco_summary")       # in ``visco_diagnostics`` (tlmpm_rubber_visco_v1)
PRONY_RANGE_KEYS = ("prony_g_scale", "prony_tau_scale")


class InvalidRubberEnvs(FloatingPointError):
    def __init__(self, env_ids, reasons):
        self.env_ids, self.reasons = list(env_ids), dict(reasons)
        super().__init__(f"Invalid rubber envs {self.env_ids}: {self.reasons}")


def decode_reason(value):
    return [name for bit, name in INVALID_REASONS.items() if int(value) & bit]


def batched_source_sha256():
    return dict(sorted(__import__("robotic_cutting_sim.batched.rubber_batch", fromlist=["x"]).source_sha256().items()))


# ----------------------------------------------------------------------------- kernels

@wp.kernel
def _wrench_b(reaction: wp.array(dtype=wp.spatial_vectord), control_dt: wp.float64, G: int,
              wrench: wp.array3d(dtype=wp.float64)):
    """-reaction / control_dt, written (r * -1) / dt: Warp's unary minus is 0 - x (+0.0 stays +0.0), NumPy's flips the
    sign bit, so the single reference's ``-reaction / control_dt`` gives -0.0 for a zero reaction (e.g. an inactive
    grip). The product by -1 is exact and flips the sign of zeros, so the result is byte-identical to the single
    reference for every finite value and both zeros (ADR 0023 §20.4); a NaN keeps its sign bit here (a NaN wrench sets
    NONFINITE_STATE)."""
    e, g = wp.tid()
    r = reaction[e*G+g]
    for k in range(6):
        wrench[e, g, k] = (r[k]*wp.float64(-1.0))/control_dt


@wp.kernel
def _validity_b(summary: wp.array2d(dtype=wp.float64), violations: wp.array(dtype=int),
                cfl_violations: wp.array(dtype=int), wrench: wp.array3d(dtype=wp.float64), G: int,
                invalid_reason: wp.array(dtype=int), invalid_count: wp.array(dtype=int),
                step_count: wp.array(dtype=wp.int64)):
    e = wp.tid()
    bits = invalid_reason[e]
    if summary[e, 0] != wp.float64(0.0):
        bits = bits | 1
    for g in range(G):
        for k in range(6):
            if not wp.isfinite(wrench[e, g, k]):
                bits = bits | 1
    if violations[e] > 0:
        bits = bits | 2
    if cfl_violations[e] > 0:
        bits = bits | 4
    invalid_reason[e] = bits
    step_count[e] = step_count[e]+wp.int64(1)
    if bits != 0:
        wp.atomic_add(invalid_count, 0, 1)


@wp.kernel
def _reset_particles_r(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int),
                       reference: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                       x_start: wp.array(dtype=wp.vec3), impulse: wp.array(dtype=wp.vec3),
                       stress: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33), mass: wp.array(dtype=float),
                       stage_mass: wp.array(dtype=float)):
    p = wp.tid()
    e = env_of_particle[p]
    if mask[e] != 0:
        x[p] = reference[p]
        v[p] = wp.vec3(0.0)
        x_start[p] = wp.vec3(0.0)
        impulse[p] = wp.vec3(0.0)
        stress[p] = wp.mat33(0.0)
        F[p] = wp.identity(n=3, dtype=float)
        mass[p] = stage_mass[e]


@wp.kernel
def _reset_nodes_r(mask: wp.array(dtype=int), env_of_node: wp.array(dtype=int), gm: wp.array(dtype=float),
                   gp: wp.array(dtype=wp.vec3), gj: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
                   new_v: wp.array(dtype=wp.vec3), jbar: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if mask[env_of_node[i]] != 0:
        gm[i] = 0.0
        gp[i] = wp.vec3(0.0)
        gj[i] = wp.vec3(0.0)
        old[i] = wp.vec3(0.0)
        new_v[i] = wp.vec3(0.0)
        jbar[i] = wp.float64(1.0)


@wp.kernel
def _reset_groups_r(mask: wp.array(dtype=int), env_of_group: wp.array(dtype=int), initial_active: wp.array(dtype=int),
                    active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd), q1: wp.array(dtype=wp.quatd),
                    b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d), A0: wp.array(dtype=wp.mat33d),
                    A1: wp.array(dtype=wp.mat33d), reaction: wp.array(dtype=wp.spatial_vectord)):
    g = wp.tid()
    if mask[env_of_group[g]] != 0:
        active[g] = initial_active[g]
        q0[g] = wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        q1[g] = wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        b0[g] = wp.vec3d()
        b1[g] = wp.vec3d()
        A0[g] = wp.identity(n=3, dtype=wp.float64)
        A1[g] = wp.identity(n=3, dtype=wp.float64)
        reaction[g] = wp.spatial_vectord()


@wp.kernel
def _reset_traction_r(mask: wp.array(dtype=int), env_of_traction: wp.array(dtype=int),
                      traction_counts: wp.array(dtype=float), stage_mass: wp.array(dtype=float),
                      force: wp.array(dtype=wp.vec3), traction_mass: wp.array(dtype=float)):
    t = wp.tid()
    e = env_of_traction[t]
    if mask[e] != 0:
        force[t] = wp.vec3(0.0)
        traction_mass[t] = traction_counts[t]*stage_mass[e]


@wp.kernel
def _reset_envs_r(mask: wp.array(dtype=int), stage_params: wp.array2d(dtype=float), params: wp.array2d(dtype=float),
                  ncols: int, work: wp.array(dtype=wp.float64), violations: wp.array(dtype=int),
                  cfl_violations: wp.array(dtype=int), invalid_reason: wp.array(dtype=int),
                  step_count: wp.array(dtype=wp.int64),
                  seed: wp.array(dtype=wp.int64), reset_seed: wp.array(dtype=wp.int64),
                  wrench: wp.array3d(dtype=wp.float64), G: int):
    e = wp.tid()
    if mask[e] != 0:
        for k in range(ncols):
            params[e, k] = stage_params[e, k]
        work[e] = wp.float64(0.0)
        violations[e] = 0
        cfl_violations[e] = 0
        invalid_reason[e] = 0
        step_count[e] = wp.int64(0)
        seed[e] = reset_seed[e]
        for g in range(G):
            for k in range(6):
                wrench[e, g, k] = wp.float64(0.0)


@wp.kernel
def _reset_shadow_r(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int),
                    reference64: wp.array(dtype=wp.vec3d), F64: wp.array(dtype=wp.mat33d),
                    x64: wp.array(dtype=wp.vec3d)):
    p = wp.tid()
    if mask[env_of_particle[p]] != 0:
        F64[p] = wp.identity(n=3, dtype=wp.float64)
        x64[p] = reference64[p]


@wp.kernel
def _reset_proj_rows_r(mask: wp.array(dtype=int), env_of_row: wp.array(dtype=int), rhs: wp.array(dtype=wp.float64),
                       vpresc: wp.array(dtype=wp.float64), mu: wp.array(dtype=wp.float64)):
    r = wp.tid()
    if mask[env_of_row[r]] != 0:
        rhs[r] = wp.float64(0.0)
        vpresc[r] = wp.float64(0.0)
        mu[r] = wp.float64(0.0)


@wp.kernel
def _reset_proj_nodes_r(mask: wp.array(dtype=int), env_of_node: wp.array(dtype=int),
                        mu_node: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    if mask[env_of_node[i]] != 0:
        for k in range(3):
            mu_node[i, k] = wp.float64(0.0)


@wp.kernel
def _reset_scratch_work_r(mask: wp.array(dtype=int), work_scratch: wp.array(dtype=wp.float64)):
    e = wp.tid()
    if mask[e] != 0:
        work_scratch[e] = wp.float64(0.0)


@wp.kernel
def _reset_state_vec6d_r(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int), multiplicity: int,
                         state: wp.array(dtype=rvc.vec6d)):
    """Generic registry-state reset (initial value ``zero``): entry i belongs to particle i // multiplicity."""
    i = wp.tid()
    if mask[env_of_particle[i//multiplicity]] != 0:
        state[i] = rvc.vec6d()


@wp.kernel
def _reset_params64_r(mask: wp.array(dtype=int), stage: wp.array2d(dtype=wp.float64),
                      params64: wp.array2d(dtype=wp.float64), ncols: int):
    e = wp.tid()
    if mask[e] != 0:
        for k in range(ncols):
            params64[e, k] = stage[e, k]


@wp.kernel
def _reset_visco_energy_r(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int),
                          pb: wp.array(dtype=wp.mat33d)):
    p = wp.tid()
    if mask[env_of_particle[p]] != 0:
        pb[p] = wp.mat33d()


@wp.kernel
def _reset_visco_energy_v2_r(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int),
                             df_prev: wp.array(dtype=wp.mat33d)):
    p = wp.tid()
    if mask[env_of_particle[p]] != 0:
        df_prev[p] = wp.mat33d()


@wp.kernel
def _reset_visco_env_v2_r(mask: wp.array(dtype=int), work_left: wp.array(dtype=wp.float64)):
    e = wp.tid()
    if mask[e] != 0:
        work_left[e] = wp.float64(0.0)


@wp.kernel
def _reset_visco_env_r(mask: wp.array(dtype=int), branch: wp.array(dtype=wp.float64),
                       dissipation: wp.array(dtype=wp.float64), d_negative: wp.array(dtype=int)):
    e = wp.tid()
    if mask[e] != 0:
        branch[e] = wp.float64(0.0)
        dissipation[e] = wp.float64(0.0)
        d_negative[e] = 0


# ----------------------------------------------------------------------------- material

class BatchedRubberMaterial:
    """N rubber envs in one process (see the module docstring).

    ``lives``: fresh single-env ``LiveRubberMaterial`` references built on device 'cpu' with a common step rule
    (pass ``step_material`` = the randomisation maxima so that dt is uniform). ``ranges`` (optional): declared
    per-reset randomisation ranges {"mu0": (lo, hi), "density": (lo, hi), "friction_table": ...}; each range must be
    covered by the batch's step rule. ``device='cpu'`` only with ``parity_mode=True`` (tests)."""
    runtime = RUNTIME

    def __init__(self, lives, *, device="cuda:0", parity_mode=False, strict=True, capture_graph=None,
                 graph_chunk=DEFAULT_GRAPH_CHUNK, ranges=None, seeds=None, cameras=None, recorders=None,
                 constitutive_model=None, bc_launch=None):
        if cameras or recorders:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_SENSOR" if cameras else "UNSUPPORTED_RECORDER",
                                                f"cameras={cameras!r} recorders={recorders!r}")
        device = str(wp.get_device(device))
        if device == "cpu" and not parity_mode:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_CPU_WITHOUT_PARITY_MODE",
                                                "device='cpu' is for parity tests (parity_mode=True)")
        start = time.perf_counter()
        self.device, self.parity_mode, self.strict = device, bool(parity_mode), bool(strict)
        self.capture_graph = wp.get_device(device).is_cuda if capture_graph is None else bool(capture_graph)
        lives = list(lives)
        self.packed = rubber_layout.pack_rubber(lives, constitutive_model=constitutive_model)
        pk = self.packed
        self.num_envs = N = pk.num_envs
        self.control_dt = float(pk.uniform["control_dt"])
        self.solver = s = BatchedTLMPMRubber(pk, device, **({} if bc_launch is None else {"bc_launch": bc_launch}))
        self._init_projection_host(lives)
        self._initial_prony = [tuple(live.material.prony) for live in lives] if s.M else None
        self.G, self.Gt, self.K = s.G, s.Gt, pk.uniform["markers"]
        self.ranges = self._check_ranges(ranges, lives)
        G1, Gt1 = max(self.G, 1), max(self.Gt, 1)
        seeds = np.arange(N, dtype=np.int64) if seeds is None else np.asarray(seeds, np.int64).reshape(N)
        trac_counts = (pk.env["traction_particle_counts"].astype(np.float32).reshape(-1) if self.Gt
                       else np.zeros(1, np.float32))
        with wp.ScopedDevice(device):
            self.wrench = wp.zeros((N, G1, 6), dtype=wp.float64)
            self.invalid_reason = wp.zeros(N, dtype=int)
            self.invalid_count = wp.zeros(1, dtype=int)
            self.step_count = wp.zeros(N, dtype=wp.int64)
            self.initial_seed = wp.array(seeds, dtype=wp.int64)
            self.seed = wp.array(seeds, dtype=wp.int64)
            self.reset_seed = wp.array(seeds, dtype=wp.int64)
            self.reset_mask = wp.zeros(N, dtype=int)
            self.initial_params = wp.array(pk.host["params"], dtype=float)
            self.stage_params = wp.array(pk.host["params"], dtype=float)
            self.initial_mass = wp.array(pk.env["mass_per_particle"], dtype=float)
            self.stage_mass = wp.array(pk.env["mass_per_particle"], dtype=float)
            self.initial_group_active = wp.clone(s.group_active)
            self.env_of_traction = wp.array(np.repeat(np.arange(N, dtype=np.int32), Gt1) if self.Gt
                                            else np.zeros(1, np.int32), dtype=int)
            self.traction_counts = wp.array(trac_counts, dtype=float)
            self.pp = wp.zeros((N*DIAG_GROUPS, 9), dtype=wp.float64)
            self.pc = wp.zeros((N*DIAG_GROUPS, 4), dtype=wp.float64)
            self.pn = wp.zeros((N*DIAG_GROUPS, 2), dtype=wp.float64)
            self.summary = wp.zeros((N, T.NDIAG), dtype=wp.float64)
            self.marker_x = wp.zeros(N*self.K, dtype=wp.vec3d)
            self.marker_v = wp.zeros(N*self.K, dtype=wp.vec3d)
            if s.M:
                self.initial_params64 = wp.array(pk.host["params64"], dtype=wp.float64)
                self.stage_params64 = wp.array(pk.host["params64"], dtype=wp.float64)
        self.visco_diagnostics = None
        if s.M:
            from ..tlmpm_rubber_visco_v1 import ViscoDiagnostics
            self.visco_diagnostics = ViscoDiagnostics(s, N)
        for module in (__name__, T.__name__):
            wp.load_module(module=module, device=device)
        s.prepare_motion_table()
        self.graph_chunk = max(1, min(int(graph_chunk), s.substeps))
        self.graphs = None
        self.capture_receipt = None
        if self.capture_graph:
            self._capture()
        self.build_wall_s = time.perf_counter()-start

    def _init_projection_host(self, lives):
        """F8: host copies of the initial per-env block inverses (exact single-solver bits) and the scenes, to rebuild
        an env's inverse when a reset changes its mass (``projection_blocks`` of the single solver)."""
        s = self.solver
        self._proj_scenes = [live.scene for live in lives] if s.projection else None
        self._proj_inv_initial = s.proj_inv.numpy().copy() if s.projection else None
        self._proj_inv_host = self._proj_inv_initial.copy() if s.projection else None
        self._proj_inv_cache = {}

    def _projection_inverse(self, e, mass):
        """Concatenated float64 inverses of env e's blocks for particle mass ``mass`` (float32), in row order."""
        key = (e, np.float32(mass).tobytes())
        if key not in self._proj_inv_cache:
            blocks = T.projection_blocks(self._proj_scenes[e], np.float32(mass))
            self._proj_inv_cache[key] = np.concatenate([b["inv"].reshape(-1) for b in blocks])
        return self._proj_inv_cache[key]

    def _check_ranges(self, ranges, lives):
        if not ranges:
            return {}
        names = cr.RUBBER_PARAM_NAMES+(PRONY_RANGE_KEYS if self.solver.M else ())
        out = {}
        for key, (lo, hi) in ranges.items():
            if key not in names or not 0 < lo <= hi:
                raise ValueError(f"randomisation range {key!r}: ({lo}, {hi})")
            out[key] = (float(lo), float(hi))
        if "prony_g_scale" in out:
            worst = max(sum(g for g, _ in live.material.prony) for live in lives)*out["prony_g_scale"][1]
            if worst >= 1.0:
                raise ValueError(f"prony_g_scale range gives sum g = {worst} >= 1")
        mu_max = out.get("mu0", (0, max(l.material.mu0 for l in lives)))[1]
        beta = max(l.material.beta for l in lives)
        rho_min = out.get("density", (min(l.material.density for l in lives), 0))[0]
        c_max = cr.wave_speed(mu_max, beta*mu_max, rho_min)
        cfl = self.solver.dt*c_max/self.solver.dx
        if cfl > 0.5+1e-9:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_NONUNIFORM_BATCH",
                                                f"randomisation ranges need CFL {cfl:.3f} > 0.5 at the batch dt; build "
                                                "the single references with step_material = the range maxima")
        return out

    def _capture(self):
        start = time.perf_counter()
        chunk = self.graph_chunk
        repeats, rest = divmod(self.solver.substeps, chunk)
        graphs = []
        for size in (chunk, rest):
            if size == 0:
                graphs.append(None)
                continue
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(size):
                    self.solver.step()
            graphs.append(capture.graph)
        self.graphs = (graphs[0], repeats, graphs[1])
        self.capture_receipt = {"graph_chunk_substeps": chunk, "chunk_replays": repeats, "remainder_substeps": rest,
                                "capture_wall_s": time.perf_counter()-start}
        # the capture ran the substeps: restore the fresh packed state of every env
        self.reset()

    # ------------------------------------------------------------------ inputs

    def begin_control_step(self):
        s = self.solver
        s.index.zero_()
        s.reaction.zero_()
        for a0, a1 in ((s.group_q0, s.group_q1), (s.group_b0, s.group_b1), (s.group_A0, s.group_A1)):
            wp.copy(a0, a1)

    def set_group_motions(self, *, rotation=None, translation=None, affine=None, active=None):
        """Host [N, G, ...] END values of this control step (call between begin and run, or pass to advance)."""
        s, N, G = self.solver, self.num_envs, self.G
        if rotation is not None:
            from scipy.spatial.transform import Rotation
            R = np.asarray(rotation, float).reshape(N*G, 3, 3)
            q = Rotation.from_matrix(R).as_quat()
            prev = s.group_q1.numpy()
            q[np.einsum("ij,ij->i", prev, q) < 0] *= -1
            s.group_q1.assign(q)
        if translation is not None:
            s.group_b1.assign(np.asarray(translation, float).reshape(N*G, 3))
        if affine is not None:
            s.group_A1.assign(np.asarray(affine, float).reshape(N*G, 3, 3))
        if active is not None:
            s.group_active.assign(np.asarray(active, np.int32).reshape(N*G))
        s.prepare_motion_table()

    def set_traction(self, forces):
        self.solver.traction_force.assign(np.asarray(forces, np.float32).reshape(-1, 3))

    def advance(self, *, rotation=None, translation=None, affine=None, active=None, traction=None):
        """One control step of every env; returns nothing (read ``wrench``, ``invalid_reason`` on the device)."""
        s = self.solver
        self.begin_control_step()
        self.set_group_motions(rotation=rotation, translation=translation, affine=affine, active=active)
        if traction is not None:
            self.set_traction(traction)
        with wp.ScopedDevice(self.device):
            if self.graphs is not None:
                main, repeats, rest = self.graphs
                for _ in range(repeats):
                    wp.capture_launch(main)
                if rest is not None:
                    wp.capture_launch(rest)
            else:
                for _ in range(s.substeps):
                    s.step()
            if self.G:
                wp.launch(_wrench_b, (self.num_envs, self.G), [s.reaction, wp.float64(self.control_dt), self.G,
                                                               self.wrench])
            self.launch_diagnostics()
            self.invalid_count.zero_()
            wp.launch(_validity_b, self.num_envs, [self.summary, s.violations, s.cfl_violations, self.wrench,
                                                   self.G, self.invalid_reason, self.invalid_count, self.step_count])
        if self.strict:
            self.check_valid()

    def launch_diagnostics(self):
        s, N = self.solver, self.num_envs
        with wp.ScopedDevice(self.device):
            wp.launch(T.rc.cell_jbar, s.total_cells, [s.cell_node, s.cell_offsets, s.cell_particles, s.F, s.jbar])
            wp.launch(T.diag_particles, (N, DIAG_GROUPS), [s.x, s.v, s.F, s.mass, s.reference, s.gravity, s.volume,
                                                           s.params, s.particle_offset, DIAG_GROUPS, s.nodes, s.jbar,
                                                           s.dt/s.dx, self.pp])
            wp.launch(T.diag_cells, (N, DIAG_GROUPS), [s.jbar, s.cell_node, s.cell_offsets, s.cell_env_offset,
                                                       DIAG_GROUPS, s.params, s.volume, self.pc])
            wp.launch(T.diag_nodes, (N, DIAG_GROUPS), [s.gm, s.new_v, s.node_offset, DIAG_GROUPS, self.pn])
            wp.launch(T.diag_combine, N, [self.pp, self.pc, self.pn, DIAG_GROUPS, DIAG_GROUPS, DIAG_GROUPS,
                                          s.particle_offset, s.cell_env_offset, self.summary])
            wp.launch(T.markers_kernel, N*self.K, [s.x, s.v, s.marker_offsets, s.marker_particles, self.marker_x,
                                                   self.marker_v])
        if self.visco_diagnostics is not None:
            self.visco_diagnostics.launch(self.summary, s.params, s.particle_offset)

    @property
    def visco_summary(self):
        return None if self.visco_diagnostics is None else self.visco_diagnostics.visco_summary

    def diagnostics_read(self):
        return [dict(zip(T.DIAG_COLUMNS, map(float, r))) for r in self.summary.numpy()]

    def visco_diagnostics_read(self):
        return None if self.visco_diagnostics is None else self.visco_diagnostics.read()

    def check_valid(self):
        if int(self.invalid_count.numpy()[0]):
            reasons = self.invalid_reason.numpy()
            ids = np.flatnonzero(reasons)
            raise InvalidRubberEnvs(ids.tolist(), {int(e): decode_reason(reasons[e]) for e in ids})
        return 0

    # ------------------------------------------------------------------ reset

    def reset(self, env_ids=None, seeds=None, materials=None, scene_keys=None, rng_ranges=False):
        """Masked partial reset (no allocation, no recapture). ``materials``: one ``MaterialValues`` per reset env
        (inside the declared ranges); ``rng_ranges=True`` samples them from the declared ranges with the env seed."""
        N = self.num_envs
        ids = np.arange(N) if env_ids is None else np.asarray(env_ids, np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= N or np.unique(ids).size != ids.size):
            raise ValueError(f"env_ids must be distinct and in [0, {N})")
        if scene_keys is not None:
            for e, key in zip(ids, scene_keys):
                if key != self.packed.scene_keys[e]:
                    raise UnsupportedBatchConfiguration("UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET",
                                                        f"env {int(e)}: use rebuild()")
        seed = self.seed.numpy().astype(np.int64)
        if seeds is not None:
            seed[ids] = np.asarray(seeds, np.int64).reshape(len(ids))
        params = self.initial_params.numpy().copy()
        mass = self.initial_mass.numpy().copy()
        params64 = self.initial_params64.numpy().copy() if self.solver.M else None
        if rng_ranges:
            if not self.ranges:
                raise ValueError("no randomisation ranges declared")
            materials = [self._sample(int(seed[e]), e) for e in ids]
        if materials is not None:
            from ..rubber_material import mass_per_particle
            for e, m in zip(ids, materials):
                self._check_in_range(m, int(e))
                params[e] = np.array(cr.env_parameters(m), np.float64).astype(np.float32)
                mass[e] = mass_per_particle(m.density, self.solver.spacing)
                if params64 is not None:
                    params64[e] = rvc.params64_row(m.prony, self.solver.dt)
        mask = np.zeros(N, np.int32)
        mask[ids] = 1
        if self.solver.projection:
            io = self.packed.env["proj_inv_env_offset"]
            initial_mass = self.initial_mass.numpy()
            for e in ids:
                e = int(e)
                inv = (self._proj_inv_initial[io[e]:io[e+1]] if mass[e] == initial_mass[e]
                       else self._projection_inverse(e, mass[e]))
                self._proj_inv_host[io[e]:io[e+1]] = inv
            self.solver.proj_inv.assign(self._proj_inv_host)
        self.reset_mask.assign(mask)
        self.reset_seed.assign(seed)
        self.stage_params.assign(params)
        self.stage_mass.assign(mass)
        if params64 is not None:
            self.stage_params64.assign(params64)
        self.reset_masked(full=len(ids) == N)

    def _sample(self, seed, e):
        from ..rubber_material import MaterialValues
        rng = np.random.default_rng(seed)
        base = dict(zip(cr.RUBBER_PARAM_NAMES, map(float, self.initial_params.numpy()[e])))
        beta = base["kappa"]/base["mu0"]
        vals = {k: (rng.uniform(*self.ranges[k]) if k in self.ranges else base[k]) for k in cr.RUBBER_PARAM_NAMES}
        prony = ()
        if self.solver.M:
            gs = rng.uniform(*self.ranges["prony_g_scale"]) if "prony_g_scale" in self.ranges else 1.0
            ts = rng.uniform(*self.ranges["prony_tau_scale"]) if "prony_tau_scale" in self.ranges else 1.0
            prony = scale_prony(self._initial_prony[e], gs, ts)
        return MaterialValues(mu0=vals["mu0"], kappa=beta*vals["mu0"], density=vals["density"],
                              friction_table=vals["friction_table"], friction_pad=vals["friction_pad"],
                              friction_self=vals["friction_self"], prony=prony)

    def _check_in_range(self, m, e=None):
        values = dict(zip(cr.RUBBER_PARAM_NAMES, cr.env_parameters(m)))
        prony = tuple(getattr(m, "prony", ()))
        if len(prony) != self.solver.M:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_NONUNIFORM_BATCH",
                                                f"material with {len(prony)} Prony terms in a batch with M = "
                                                f"{self.solver.M} (pad with rubber_material.pad_prony)")
        if self.solver.M and e is not None:
            values.update(prony_scales(self._initial_prony[e], prony))
        for key, (lo, hi) in self.ranges.items():
            if key in PRONY_RANGE_KEYS and key not in values:
                continue
            if not lo-1e-12*abs(lo) <= values[key] <= hi+1e-12*abs(hi):
                raise ValueError(f"{key} = {values[key]} outside the declared range [{lo}, {hi}]")
        if self.solver.M and e is not None:
            for key in PRONY_RANGE_KEYS:
                if key not in self.ranges and key in values and abs(values[key]-1.0) > 1e-12:
                    raise ValueError(f"{key} = {values[key]} but no {key} range is declared")
        c = cr.wave_speed(m.mu0, m.kappa, m.density)
        if self.solver.dt*c/self.solver.dx > 0.5+1e-9:
            raise ValueError("material violates the batch step rule (CFL > 0.5)")

    def reset_masked(self, full=False):
        s, N = self.solver, self.num_envs
        with wp.ScopedDevice(self.device):
            wp.launch(_reset_particles_r, s.total_particles, [self.reset_mask, s.env_of_particle, s.reference, s.x,
                                                              s.v, s.x_start, s.impulse, s.stress, s.F, s.mass,
                                                              self.stage_mass])
            wp.launch(_reset_nodes_r, s.total_nodes, [self.reset_mask, s.env_of_node, s.gm, s.gp, s.gj, s.old,
                                                      s.new_v, s.jbar])
            if self.G:
                wp.launch(_reset_groups_r, N*self.G, [self.reset_mask, s.env_of_group, self.initial_group_active,
                                                      s.group_active, s.group_q0, s.group_q1, s.group_b0, s.group_b1,
                                                      s.group_A0, s.group_A1, s.reaction])
            if self.Gt:
                wp.launch(_reset_traction_r, N*self.Gt, [self.reset_mask, self.env_of_traction, self.traction_counts,
                                                         self.stage_mass, s.traction_force, s.traction_mass])
            wp.launch(_reset_envs_r, N, [self.reset_mask, self.stage_params, s.params, cr.NUM_PARAMS, s.work,
                                         s.violations, s.cfl_violations, self.invalid_reason, self.step_count,
                                         self.seed, self.reset_seed, self.wrench, self.G])
            if s.state_precision == "f64_shadow_v1":
                wp.launch(_reset_shadow_r, s.total_particles, [self.reset_mask, s.env_of_particle, s.reference64,
                                                               s.F64, s.x64])
            if s.projection:
                wp.launch(_reset_proj_rows_r, s.proj_rows, [self.reset_mask, s.env_of_row, s.proj_rhs, s.proj_vpresc,
                                                            s.proj_mu])
                wp.launch(_reset_proj_nodes_r, s.total_nodes, [self.reset_mask, s.env_of_node, s.proj_mu_node])
            if s.traction_work == "grid_displacement_v1":
                wp.launch(_reset_scratch_work_r, N, [self.reset_mask, s.work_scratch])
            if s.kick_average:
                wp.launch(_reset_scratch_work_r, N, [self.reset_mask, s.work_end_velocity])
            if s.M:
                for var in s.model.per_particle_state:          # generic registry-state reset (initial "zero")
                    if var.initial != "zero" or var.dtype is not rvc.vec6d:
                        raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER",
                                                            f"state {var.name}: initial {var.initial!r} not in v1")
                    wp.launch(_reset_state_vec6d_r, s.total_particles*var.multiplicity,
                              [self.reset_mask, s.env_of_particle, var.multiplicity, getattr(s, var.name)])
                wp.launch(_reset_params64_r, N, [self.reset_mask, self.stage_params64, s.params64,
                                                 s.params64.shape[1]])
            if s.visco_energy_on:
                wp.launch(_reset_visco_energy_r, s.total_particles, [self.reset_mask, s.env_of_particle, s.visco_pb])
                wp.launch(_reset_visco_env_r, N, [self.reset_mask, s.visco_branch_work, s.visco_dissipation,
                                                  s.visco_d_negative])
                if s.visco_energy == "branch_work_kick_average_v2":
                    wp.launch(_reset_visco_energy_v2_r, s.total_particles, [self.reset_mask, s.env_of_particle,
                                                                            s.visco_df_prev])
                    wp.launch(_reset_visco_env_v2_r, N, [self.reset_mask, s.visco_branch_work_left])
            if full:
                s.index.zero_()

    def rebuild(self, lives, **kwargs):
        options = dict(device=self.device, parity_mode=self.parity_mode, strict=self.strict,
                       capture_graph=self.capture_graph, graph_chunk=self.graph_chunk, ranges=self.ranges or None,
                       bc_launch=self.solver.bc_launch)
        options.update(kwargs)
        replacement = type(self)(lives, **options)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)

    # ------------------------------------------------------------------ queries

    def markers(self):
        return self.marker_x.numpy().reshape(self.num_envs, self.K, 3)

    def device_arrays(self):
        out = {f"solver.{k}": v for k, v in self.solver.device_arrays().items()}
        out.update({k: v for k, v in vars(self).items() if isinstance(v, wp.array)})
        return out

    def memory_bytes(self):
        return int(sum(v.capacity for v in self.device_arrays().values()))

    def audits(self):
        host = {name: getattr(self.solver, name).numpy() for name in
                ("nodes", "cell_particles", "cell_offsets", "cell_node", "marker_particles", "marker_offsets",
                 "node_group", "particle_traction")}
        return rubber_layout.range_audit(self.packed, host)

    def receipt(self):
        return {"runtime": RUNTIME, "version": VERSION, "device": self.device, "parity_mode": self.parity_mode,
                "strict": self.strict, "graph": self.capture_receipt, "control_dt_s": self.control_dt,
                "layout": self.packed.layout_receipt(), "solver": self.solver.solver_receipt(),
                "envs": self.packed.receipts, "ranges": self.ranges, "memory_bytes": self.memory_bytes(),
                "build_wall_s": self.build_wall_s, "calibration": "SIMULATOR_ONLY_UNCALIBRATED",
                "device_hashgrid_guard": "not_applicable_no_hashgrid (rubber v1 has no self-contact)",
                "coupling": "kinematic Dirichlet groups (no robot feedback in P2-1)",
                "open_vector_env": "deferred to P2-6 (P1-5 owns the configured-scene dispatch)",
                "diagnostics_groups": DIAG_GROUPS}


def prony_scales(initial, prony):
    """(g scale, tau scale) of ``prony`` relative to ``initial`` (uniform over the terms, g = 0 terms skipped for g);
    raises ValueError if the terms are not a uniform scaling."""
    gs = [g/g0 for (g, _), (g0, _) in zip(prony, initial) if g0 != 0]
    ts = [t/t0 for (_, t), (_, t0) in zip(prony, initial)]
    if any(g != 0 for (g, _), (g0, _) in zip(prony, initial) if g0 == 0):
        raise ValueError("a padding term (g = 0) cannot be resampled to g != 0")
    out = {}
    for key, values in (("prony_g_scale", gs), ("prony_tau_scale", ts)):
        if values:
            if max(values)-min(values) > 1e-12*max(abs(v) for v in values):
                raise ValueError(f"{key}: the Prony terms are not scaled uniformly ({values})")
            out[key] = values[0]
    return out


def scale_prony(initial, g_scale, tau_scale):
    """Terms with every g_i multiplied by ``g_scale`` and every tau_i by ``tau_scale`` (sum g < 1 is validated)."""
    from ..rubber_material import validate_prony
    return validate_prony(tuple((g*g_scale, tau*tau_scale) for g, tau in initial))
