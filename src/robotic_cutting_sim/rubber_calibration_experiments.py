"""Forward models of the P2-5 calibration experiments (ADR 0023 §23.2 decision 6; plan §3.1, §3.3; task P5.2).

Every manifest experiment type maps to a ``rubber_strip_v1`` scene, a load schedule, a contact spec, a duration, an
observable extractor and a sensor transform:

====================  ===========================================  ==============================================
type                  scene / loading                              sensors (data channels)
====================  ===========================================  ==============================================
tensile_load_unload   case C (end_mask x), grip 1 axial at the     load cell (1 kHz, control-step mean of the
                      crosshead speed eps_dot L: lambda 1 ->       grip-1 reaction) + crosshead displacement
                      lambda_max -> lambda_min
stress_relaxation     case C, ramp to lambda_hold in T_ramp, hold  load cell + crosshead
cantilever_rebound    case A, body weight (traction, smootherstep  video markers (240 fps, the 9 cross-section
                      ramp) + tip load P (same ramp), tip load     centroids, z), linear interpolation between
                      released at t_rel, free rebound              control-step states
gravity_sag           case A, body weight ramp and hold            video markers
friction_sliding      free block on the table plane (contact law   force plate (1 kHz, plane wrench x / -z) +
                      of the protocol), weight (+ top load) ramp,  video (240 fps, block centroid x)
                      back-face push ramp to P_max, release
====================  ===========================================  ==============================================

Gravity is applied as a body traction ``M_g g`` per traction group (the solver's gravity is zero), so that loads can be
ramped (a step load on an undamped body overshoots); the gravity work is then part of the external (traction) work.
Applied forces (tip load, push, top load) are absolute inputs in N, independent of the model's density.

Population evaluation: (candidate theta_j, experiment k) -> one env of a ``BatchedRubberMaterial``. One batch per
experiment (every env of a batch has the same geometry, groups, traction groups, contact and dx; dt is uniform from the
declared step material = the parameter box's maxima), built once and re-used: a candidate set is written with the
masked reset (``materials=``), no rebuild and no recapture. ``PopulationBatch`` subclasses the batch only to replace
the randomisation-range check (any material of the parameter box, including per-term Prony values, is admitted; the
step rule and the Prony arity are still checked). The single-env reference ``run_single`` drives a fresh
``LiveRubberMaterial`` with the same schedule; on CPU the population batch is byte-identical to it (CAL-02).

Labels: ``SIMULATOR_ONLY_UNCALIBRATED``; ``NUMERICAL_VALIDATION_SYNTHETIC`` when the data are synthetic.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from . import rubber_scene as rs
from .rubber_material import MaterialValues, mass_per_particle, step_rule

TYPES = ("tensile_load_unload", "stress_relaxation", "cantilever_rebound", "gravity_sag", "friction_sliding")
CLASS_OF_TYPE = {"tensile_load_unload": "C", "stress_relaxation": "C", "cantilever_rebound": "A",
                 "gravity_sag": "A", "friction_sliding": "F"}
G_ACC = 9.81
CONTROL_DT = 1e-3
VIDEO_FPS = 240.0
DEFAULT_OPTIONS = {"dirichlet": "consistent_mass_projection_v1", "constraint_work": "kick_average_v1",
                   "traction_work": "grid_displacement_v1", "state_precision": "f64_shadow_v1"}
DEFAULT_VISCO_ENERGY = "branch_work_kick_average_v2"
DEFAULT_CONTACT_LAW = "grid_node_penalty_coulomb_v3"
DEFAULT_FIDELITIES = {"fit": {"n_t": 4, "cfl": 0.5, "resolution_profile": "rubber_ref_v1"},
                      "truth": {"n_t": 6, "cfl": 0.25, "resolution_profile": "rubber_ref_v1"}}
NOMINAL_DENSITY = 1150.0       # converts the declared applied loads (multiples of a nominal weight) to N
LABELS = {"calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
_HERE = Path(__file__).resolve()


def source_sha256():
    return {_HERE.name: hashlib.sha256(_HERE.read_bytes()).hexdigest()}


# ============================================================================ configuration

@dataclass(frozen=True)
class ModelConfig:
    """Uniform numerical configuration of a forward model (one fidelity)."""
    n_t: int = 4
    cfl: float = 0.5
    resolution_profile: str = "rubber_ref_v1"
    control_dt: float = CONTROL_DT
    beta: float = 20.0
    mu0_max: float = 8.0e5          # step material: the parameter box's instantaneous-modulus maximum
    density_min: float = 1000.0     # step material: declared minimum density
    contact_law: str = DEFAULT_CONTACT_LAW
    visco_energy: str = DEFAULT_VISCO_ENERGY
    stable_step_guard: float = 1.0

    def step_material(self):
        return MaterialValues(mu0=self.mu0_max, kappa=self.beta*self.mu0_max, density=self.density_min)


def model_config(common, fidelity):
    """``ModelConfig`` from a protocol ``common`` block and a fidelity name (``fit`` / ``truth``)."""
    fid = common["fidelities"][fidelity]
    return ModelConfig(n_t=int(fid["n_t"]), cfl=float(fid["cfl"]), resolution_profile=fid["resolution_profile"],
                       control_dt=float(common["control_dt_s"]), beta=float(common["bulk_ratio_beta"]),
                       mu0_max=float(common["step_material"]["mu0_max_pa"]),
                       density_min=float(common["step_material"]["density_min_kg_m3"]),
                       contact_law=common["contact_law"], visco_energy=common["visco_energy"],
                       stable_step_guard=float(common["stable_step_guard"]))


def material_from_theta(theta, density, beta=20.0):
    """theta = dict(mu_eq_pa, g1, tau1_s, g2, tau2_s, mu_table) -> ``MaterialValues`` (mu0 = mu_eq / g_inf,
    kappa = beta mu0; M = 2 Prony terms)."""
    g1, g2 = float(theta["g1"]), float(theta["g2"])
    mu0 = float(theta["mu_eq_pa"])/(1.0-g1-g2)
    return MaterialValues(mu0=mu0, kappa=beta*mu0, density=float(density), friction_table=float(theta["mu_table"]),
                          prony=((g1, float(theta["tau1_s"])), (g2, float(theta["tau2_s"]))))


# ============================================================================ scenes and schedules

def specimen_dims(exp):
    sp = exp["specimen"]
    return float(sp["length"]), float(sp["width"]), float(sp["thickness"])


def build_spec(exp, cfg):
    """``RubberStripSpec`` of an experiment at the configuration's n_t (dx = t / n_t)."""
    L, W, t = specimen_dims(exp)
    dx = t/cfg.n_t
    s = dx/2
    kind = CLASS_OF_TYPE[exp["type"]]
    common = dict(length=L, width=W, thickness=t, n_t=cfg.n_t, dx=dx, resolution_profile=cfg.resolution_profile)
    if kind == "C":
        return rs.RubberStripSpec(case="C", end_mask="x", **common)
    if kind == "A":
        regions = (("tip", (L-s+1e-9*L, -1.0, -1.0), (1.0, 1.0, 1.0)), ("body", (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
        return rs.RubberStripSpec(case="A", traction_regions=regions, **common)
    regions = (("push", (-1.0, -1.0, -1.0), (s, 1.0, 1.0)), ("top", (-1.0, -1.0, t-s), (1.0, 1.0, 1.0)),
               ("body", (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    return rs.RubberStripSpec(case="free", traction_regions=regions, **common)


def contact_spec(exp, cfg):
    if CLASS_OF_TYPE[exp["type"]] != "F":
        return None
    from .rubber_contact import ContactSpec
    return ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),), law=cfg.contact_law)


def smootherstep(x):
    x = min(max(float(x), 0.0), 1.0)
    return x*x*x*(x*(6.0*x-15.0)+10.0)


def nominal_weight(exp):
    L, W, t = specimen_dims(exp)
    return NOMINAL_DENSITY*L*W*t*G_ACC


def duration_s(exp):
    ld, kind = exp["loading"], exp["type"]
    if kind == "tensile_load_unload":
        L = specimen_dims(exp)[0]
        v = ld["strain_rate_1_s"]*L
        return ((ld["max_stretch"]-1.0)*L+(ld["max_stretch"]-ld["min_stretch"])*L)/v
    if kind == "stress_relaxation":
        return ld["ramp_s"]+ld["hold_s"]
    if kind == "cantilever_rebound":
        return ld["release_s"]+ld["record_after_release_s"]
    return ld["duration_s"]


def steps(exp, cfg):
    """Control steps of the record: ceil(duration / control_dt) (a schedule ending inside a step holds its end
    value for the rest of that step)."""
    return int(math.ceil(duration_s(exp)/cfg.control_dt-1e-9))


def crosshead(exp, t):
    """Grip-1 axial displacement at time t (tensile / relaxation)."""
    ld = exp["loading"]
    L = specimen_dims(exp)[0]
    if exp["type"] == "tensile_load_unload":
        v = ld["strain_rate_1_s"]*L
        umax, umin = (ld["max_stretch"]-1.0)*L, (ld["min_stretch"]-1.0)*L
        t_up = umax/v
        return v*t if t <= t_up else max(umax-v*(t-t_up), umin)
    return (ld["hold_stretch"]-1.0)*L*min(1.0, t/ld["ramp_s"])


class Schedule:
    """Per-control-step inputs of one experiment for one model: ``inputs(k)`` -> (translation [G, 3] or None,
    traction [Gt, 3] or None) for control step k = 1..K (motions are END values, tractions act over the step and are
    evaluated at the step's mid time). Group weights use the model's particle mass (gravity body force)."""

    def __init__(self, exp, scene, density, cfg):
        self.exp, self.cfg = exp, cfg
        self.kind = exp["type"]
        self.K = steps(exp, cfg)
        m = float(mass_per_particle(density, scene.spacing))
        counts = np.bincount(scene.particle_traction[scene.particle_traction >= 0],
                             minlength=len(scene.traction_groups))
        self.group_weight = np.array([float(c)*m*G_ACC for c in counts])
        self.names = [g["name"] for g in scene.traction_groups]
        self.G = len(scene.groups)
        self.Gt = len(scene.traction_groups)

    def inputs(self, k):
        k = min(int(k), self.K)                   # past its duration an env holds its last inputs
        dtc = self.cfg.control_dt
        t_end, t_mid = k*dtc, (k-0.5)*dtc
        ld = self.exp["loading"]
        if self.kind in ("tensile_load_unload", "stress_relaxation"):
            tr = np.zeros((self.G, 3))
            tr[1, 0] = crosshead(self.exp, t_end)
            return tr, None
        force = np.zeros((self.Gt, 3))
        if self.kind in ("cantilever_rebound", "gravity_sag"):
            ramp = smootherstep(t_mid/ld["weight_ramp_s"])
            force[:, 2] = -self.group_weight*ramp
            if self.kind == "cantilever_rebound" and t_mid < ld["release_s"]:
                force[self.names.index("tip"), 2] -= ld["tip_load_n"]*ramp
            return None, force
        ramp = smootherstep(t_mid/ld["settle_ramp_s"])
        force[:, 2] = -self.group_weight*ramp
        w_nom = nominal_weight(self.exp)
        top_load = ld["extra_top_load_over_nominal_weight"]*w_nom
        force[self.names.index("top"), 2] -= top_load*ramp
        t0, tp = ld["push_start_s"], ld["push_ramp_s"]
        if t0 <= t_mid <= t0+tp:
            pmax = ld["push_max_over_nominal_normal"]*(w_nom+top_load)
            force[self.names.index("push"), 0] += pmax*(t_mid-t0)/tp
        return None, force


# ============================================================================ lives, batch, raw records

def build_life(exp, material, cfg, *, device="cpu", capture_graph=False):
    from .live_rubber_material import LiveRubberMaterial
    visco = cfg.visco_energy if material.prony else "off_v1"
    return LiveRubberMaterial(build_spec(exp, cfg), material=material, control_dt=cfg.control_dt, cfl=cfg.cfl,
                              step_material=cfg.step_material(), gravity=(0.0, 0.0, 0.0), device=device,
                              capture_graph=capture_graph, strict=False, stable_step_guard=cfg.stable_step_guard,
                              visco_energy=visco, contact=contact_spec(exp, cfg), **DEFAULT_OPTIONS)


def check_step_rule(cfg, materials, dx):
    """Every material must satisfy the configuration's CFL at the batch dt (from the step material)."""
    rule = step_rule(dx, cfg.control_dt, mu0_max=cfg.mu0_max, kappa_max=cfg.beta*cfg.mu0_max,
                     density_min=cfg.density_min, cfl=cfg.cfl)
    from .batched.constitutive_rubber import wave_speed
    for m in materials:
        c = wave_speed(m.mu0, m.kappa, m.density)
        if rule["dt_s"]*c/dx > cfg.cfl*(1+1e-9):
            raise ValueError(f"material mu0 {m.mu0} rho {m.density} violates CFL {cfg.cfl} at dt {rule['dt_s']}")
    return rule


RAW_FIELDS = ("summary", "visco", "markers", "wrench", "work", "contact_work", "plane")


def _raw_single(live):
    s = live.solver
    out = {"summary": s.diagnostics.summary.numpy()[0].copy(),
           "visco": (s.visco_diagnostics.visco_summary.numpy()[0].copy()
                     if getattr(s, "visco_diagnostics", None) is not None else np.zeros(0)),
           "markers": s.diagnostics.marker_x.numpy().astype(np.float64).copy(),
           "wrench": np.asarray(live.last_wrench, np.float64).reshape(-1, 6).copy(),
           "work": s.work.numpy()[:1].copy(),
           "contact_work": s.ct_work.numpy()[0].copy() if hasattr(s, "ct_work") else np.zeros(0),
           "plane": (np.asarray(live.last_contact["plane_wrench"], np.float64).copy()
                     if getattr(live, "last_contact", None) else np.zeros((0, 6)))}
    return out


class PopulationBatchMixin:
    """Replaces the randomisation-range check of ``BatchedRubberMaterial``: a calibration candidate may set every
    Prony term independently. Still checked: the Prony arity M and CFL <= 0.5 at the batch dt (the caller checks the
    configuration's CFL with ``check_step_rule``)."""

    def _check_in_range(self, m, e=None):
        from .batched.constitutive_rubber import wave_speed
        from .batched.errors import UnsupportedBatchConfiguration
        if len(tuple(getattr(m, "prony", ()))) != self.solver.M:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_NONUNIFORM_BATCH",
                                                f"material with {len(m.prony)} Prony terms, batch M = {self.solver.M}")
        c = wave_speed(m.mu0, m.kappa, m.density)
        if self.solver.dt*c/self.solver.dx > 0.5+1e-9:
            raise ValueError("material violates the batch step rule (CFL > 0.5)")


def population_batch_class():
    from .batched.rubber_material_batch import BatchedRubberMaterial

    class PopulationBatch(PopulationBatchMixin, BatchedRubberMaterial):
        pass
    return PopulationBatch


class ExperimentBatch:
    """One experiment, P envs (one per candidate). ``run(materials)`` resets every env to its material and runs the
    schedule; returns one raw record per env."""

    def __init__(self, exp, density, cfg, population, *, device="cuda:0", parity_mode=False, placeholder=None):
        self.exp, self.cfg, self.density = exp, cfg, float(density)
        self.population = int(population)
        mat = placeholder or MaterialValues(mu0=cfg.mu0_max*0.5, kappa=cfg.beta*cfg.mu0_max*0.5, density=density,
                                            friction_table=0.5, prony=((0.1, 0.01), (0.1, 0.1)))
        lives = [build_life(exp, mat, cfg) for _ in range(self.population)]
        self.scene = lives[0].scene
        self.schedule = Schedule(exp, self.scene, density, cfg)
        cls = population_batch_class()
        self.batch = cls(lives, device=device, parity_mode=parity_mode, strict=False)
        self.receipt = {"experiment": exp["id"], "population": self.population, "particles_per_env": self.scene.n,
                        "substeps": int(self.batch.solver.substeps), "dt_s": float(self.batch.solver.dt),
                        "dx_m": float(self.scene.dx), "device": self.batch.device, "steps": self.schedule.K}
        del lives

    def run(self, materials):
        b, sch = self.batch, self.schedule
        if len(materials) != self.population:
            raise ValueError(f"{len(materials)} materials for a population of {self.population}")
        for m in materials:
            if abs(m.density-self.density) > 0:
                raise ValueError("a calibration candidate never changes the measured density")
        check_step_rule(self.cfg, materials, self.scene.dx)
        b.reset(materials=list(materials))
        N = self.population
        records = [_empty_record(self.schedule, self.scene) for _ in range(N)]
        G, Gt = b.G, b.Gt
        for k in range(1, sch.K+1):
            tr, force = sch.inputs(k)
            kw = {}
            if tr is not None:
                kw["translation"] = np.broadcast_to(tr, (N, G, 3)).copy()
            if force is not None:
                kw["traction"] = np.broadcast_to(force, (N, Gt, 3)).copy()
            b.advance(**kw)
            raw = _raw_batch(b)
            for e in range(N):
                _append(records[e], {name: raw[name][e] for name in RAW_FIELDS}, k)
        inv = b.invalid_reason.numpy()
        for e in range(N):
            records[e]["invalid_reason"] = int(inv[e])
        return records


def _raw_batch(b):
    s = b.solver
    N = b.num_envs
    out = {"summary": b.summary.numpy(),
           "visco": b.visco_summary.numpy() if b.visco_summary is not None else np.zeros((N, 0)),
           "markers": b.marker_x.numpy().astype(np.float64).reshape(N, b.K, 3),
           "wrench": b.wrench.numpy()[:, :b.G],
           "work": s.work.numpy().reshape(N, 1),
           "contact_work": s.ct_work.numpy() if b.contact is not None else np.zeros((N, 0)),
           "plane": (b.contact.plane_wrench.numpy()[:, :b.contact.P] if b.contact is not None
                     else np.zeros((N, 0, 6)))}
    return out


def _empty_record(schedule, scene):
    K = schedule.K
    return {"K": K, "summary": [None]*K, "visco": [None]*K, "markers": [None]*K, "wrench": [None]*K,
            "work": [None]*K, "contact_work": [None]*K, "plane": [None]*K,
            "marker_reference": scene.marker_reference.astype(np.float64)}


def _append(record, raw, k):
    if k > record["K"]:
        return
    for name in RAW_FIELDS:
        record[name][k-1] = np.array(raw[name], copy=True)


def finalize_record(record):
    """Stack the per-step lists into arrays (step k = row k-1)."""
    out = dict(record)
    for name in RAW_FIELDS:
        out[name] = np.stack(record[name])
    return out


def run_single(exp, material, cfg, *, device="cpu"):
    """Single-env reference: a fresh ``LiveRubberMaterial`` driven by the experiment's schedule."""
    check_step_rule(cfg, [material], specimen_dims(exp)[2]/cfg.n_t)
    live = build_life(exp, material, cfg, device=device, capture_graph=None if device != "cpu" else False)
    sch = Schedule(exp, live.scene, material.density, cfg)
    record = _empty_record(sch, live.scene)
    for k in range(1, sch.K+1):
        tr, force = sch.inputs(k)
        motions = {g: {"translation": tr[g]} for g in range(sch.G)} if tr is not None else None
        traction = {t: force[t] for t in range(sch.Gt)} if force is not None else None
        live.advance(motions, traction=traction)
        _append(record, _raw_single(live), k)
    record["invalid_reason"] = int(live.invalid_reason)
    return finalize_record(record)


def material_key(exp, m):
    """Physics-relevant identity of a material for one experiment: the table friction acts only in the friction
    experiment (no contact elsewhere), so candidates that differ only in mu_table share one env there."""
    key = (float(m.mu0), float(m.kappa), float(m.density), tuple(m.prony))
    return key+(float(m.friction_table),) if CLASS_OF_TYPE[exp["type"]] == "F" else key


class PopulationEvaluator:
    """Population forward model over a set of experiments: one ``ExperimentBatch`` per experiment (built lazily at the
    number of distinct materials of the first call, rebuilt larger if a later call needs more, padded otherwise).
    ``evaluate(candidates)`` with ``candidates[j][exp_id]`` = ``MaterialValues`` -> ``records[j][exp_id]``. Candidates
    with the same physics-relevant material for an experiment (``material_key``) share one env (exact: the same
    material gives the same record)."""

    def __init__(self, experiments, densities, cfg, population, *, device="cuda:0", parity_mode=False):
        self.experiments = list(experiments)
        self.densities = dict(densities)
        self.cfg, self.population, self.device, self.parity_mode = cfg, int(population), device, parity_mode
        self.batches = {}
        self.calls = 0
        self.env_evaluations = 0

    def batch(self, exp, size):
        b = self.batches.get(exp["id"])
        if b is None or b.population < size:
            self.batches[exp["id"]] = None
            b = self.batches[exp["id"]] = ExperimentBatch(exp, self.densities[exp["id"]], self.cfg, size,
                                                          device=self.device, parity_mode=self.parity_mode)
        return b

    def evaluate(self, candidates):
        n = len(candidates)
        if not 1 <= n <= self.population:
            raise ValueError(f"{n} candidates for a population of {self.population}")
        out = [dict() for _ in range(n)]
        for exp in self.experiments:
            eid = exp["id"]
            keys, unique = [], {}
            for c in candidates:
                k = material_key(exp, c[eid])
                if k not in unique:
                    unique[k] = len(unique)
                keys.append(unique[k])
            mats = [None]*len(unique)
            for c, k in zip(candidates, keys):
                mats[k] = mats[k] or c[eid]
            b = self.batch(exp, len(mats))
            recs = b.run(mats+[mats[0]]*(b.population-len(mats)))
            done = [finalize_record(recs[i]) for i in range(len(mats))]
            for j in range(n):
                out[j][eid] = done[keys[j]]
            self.env_evaluations += b.population
        self.calls += 1
        return out

    def receipts(self):
        return [b.receipt for b in self.batches.values()]


# ============================================================================ observables and sensors

def _col(name):
    from .tlmpm_rubber_v1 import DIAG_COLUMNS
    return DIAG_COLUMNS.index(name)


def _vcol(name):
    from .tlmpm_rubber_visco_v1 import VISCO_DIAG_COLUMNS
    return VISCO_DIAG_COLUMNS.index(name)


def observables(exp, record, cfg):
    """Noise-free observables at control-step ends (k = 0..K; row 0 is the reference state): time, crosshead,
    load-cell force, markers [K+1, 9, 3], plane wrench, volume ratio and the total dissipated energy
    D(t) = W_ext - (KE + Psi_eq + Psi_visc_stored) (W_ext: Dirichlet + traction work incl. the weight traction; with
    the books closed this is the viscous dissipation + the contact dissipation - the contact spring energy)."""
    K = record["K"]
    dtc = cfg.control_dt
    t = dtc*np.arange(K+1)
    markers = np.concatenate([record["marker_reference"][None], record["markers"]], axis=0)
    summ = record["summary"]
    vr = np.r_[1.0, summ[:, _col("volume_ratio")]]
    ke = summ[:, _col("kinetic_particle")]
    if record["visco"].size:
        psi = record["visco"][:, _vcol("psi_eq")]+record["visco"][:, _vcol("psi_visc_stored")]
    else:
        psi = summ[:, _col("elastic")]
    diss = np.r_[0.0, record["work"][:, 0]-(ke+psi)]
    out = {"t": t, "markers": markers, "volume_ratio": vr, "dissipated_energy_j": diss,
           "invalid_reason": record["invalid_reason"]}
    if exp["type"] in ("tensile_load_unload", "stress_relaxation"):
        out["crosshead_m"] = np.array([crosshead(exp, x) for x in t])
        out["force_n"] = np.r_[0.0, -record["wrench"][:, 1, 0]]
    if exp["type"] == "friction_sliding":
        out["plane_fx_n"] = np.r_[0.0, record["plane"][:, 0, 0]]
        out["plane_fn_n"] = np.r_[0.0, -record["plane"][:, 0, 2]]
    return out


def translation_check(obs):
    """Friction scope check (K2-1: law v3 is valid for translational sliding only): the largest lateral drift of the
    block centroid (y) and the largest change of the spread in y of the 9 cross-section centroids (a yaw / in-plane
    bending proxy) over the record, in m."""
    y = obs["markers"][:, :, 1]
    spread = y.max(1)-y.min(1)
    return {"max_centroid_y_drift_m": float(np.abs(y.mean(1)-y[0].mean()).max()),
            "max_marker_y_spread_change_m": float(np.abs(spread-spread[0]).max())}


def video_times(T_end, fps=VIDEO_FPS):
    n = int(math.floor(T_end*fps+1e-9))
    return np.arange(n+1)/fps


def _interp(t_new, t, y):
    return np.interp(t_new, t, y)


def sample_channels(exp, obs, sensors):
    """Sensor transform (noise-free): the experiment's data channels at the sensor timestamps.
    Load cell / crosshead / force plate: the control-step samples k = 1..K (the reaction is the step mean);
    video: frames at f / fps (f = 0..floor(T fps)) by linear interpolation between control-step states."""
    kind = exp["type"]
    t = obs["t"]
    out = {}
    if kind in ("tensile_load_unload", "stress_relaxation"):
        out["force"] = {"time_s": t[1:], "displacement_m": obs["crosshead_m"][1:], "force_n": obs["force_n"][1:]}
    elif kind in ("cantilever_rebound", "gravity_sag"):
        tv = video_times(t[-1], sensors["video"]["sample_rate_hz"])
        cols = {"time_s": tv}
        for m in range(obs["markers"].shape[1]):
            cols[f"m{m}_z_m"] = _interp(tv, t, obs["markers"][:, m, 2])
        out["video"] = cols
    else:
        out["force"] = {"time_s": t[1:], "friction_force_n": obs["plane_fx_n"][1:],
                        "normal_force_n": obs["plane_fn_n"][1:]}
        tv = video_times(t[-1], sensors["video"]["sample_rate_hz"])
        out["video"] = {"time_s": tv, "centroid_x_m": _interp(tv, t, obs["markers"][:, :, 0].mean(1))}
    return out


CHANNEL_SENSORS = {("force", "force_n"): "load_cell", ("force", "displacement_m"): "crosshead",
                   ("force", "friction_force_n"): "force_plate", ("force", "normal_force_n"): "force_plate",
                   ("video", "centroid_x_m"): "video"}


def sensor_of(role, column):
    if role == "video" and column != "time_s":
        return "video"
    return CHANNEL_SENSORS[(role, column)]


def add_noise(channels, sensors, rng):
    """Seeded sensor noise: Gaussian sigma then quantisation to the resolution (0 = none). Time stamps exact."""
    out = {}
    for role in sorted(channels):
        cols = {}
        for name, values in channels[role].items():
            if name == "time_s":
                cols[name] = np.array(values, copy=True)
                continue
            sen = sensors[sensor_of(role, name)]
            v = np.asarray(values, np.float64)+rng.normal(0.0, sen["noise_sigma"], size=len(values))
            if sen["resolution"] > 0:
                v = np.round(v/sen["resolution"])*sen["resolution"]
            cols[name] = v
        out[role] = cols
    return out


def effective_sigma(sensor):
    return math.sqrt(sensor["noise_sigma"]**2+sensor["resolution"]**2/12.0)


# ============================================================================ residuals and metrics

RESIDUAL_MARKERS = (4, 8)


def residuals(exp, pred, data, sensors):
    """Weighted residuals (pred - data) / sigma_eff of one fit experiment (noise-free prediction channels vs data
    channels). Tensile / relaxation: force; cantilever / sag: marker 4 and 8 z, frames f >= 1; friction: centroid x
    frames f >= 1 and friction force."""
    kind = exp["type"]
    parts = []
    if kind in ("tensile_load_unload", "stress_relaxation"):
        s = effective_sigma(sensors["load_cell"])
        parts.append((pred["force"]["force_n"]-data["force"]["force_n"])/s)
    elif kind in ("cantilever_rebound", "gravity_sag"):
        s = effective_sigma(sensors["video"])
        for m in RESIDUAL_MARKERS:
            parts.append((pred["video"][f"m{m}_z_m"][1:]-data["video"][f"m{m}_z_m"][1:])/s)
    else:
        sv, sf = effective_sigma(sensors["video"]), effective_sigma(sensors["force_plate"])
        parts.append((pred["video"]["centroid_x_m"][1:]-data["video"]["centroid_x_m"][1:])/sv)
        parts.append((pred["force"]["friction_force_n"]-data["force"]["friction_force_n"])/sf)
    return np.concatenate(parts)


def _moving_average(y, n):
    if n <= 1:
        return np.asarray(y, float)
    k = np.ones(n)/n
    pad = n//2
    yp = np.r_[np.full(pad, y[0]), y, np.full(pad, y[-1])]
    return np.convolve(yp, k, mode="valid")


def _parabolic(t, y, i):
    if i <= 0 or i >= len(y)-1:
        return t[i], y[i]
    y0, y1, y2 = y[i-1], y[i], y[i+1]
    den = y0-2*y1+y2
    if den == 0:
        return t[i], y1
    d = 0.5*(y0-y2)/den
    h = t[1]-t[0]
    return t[i]+d*h, y1-0.25*(y0-y2)*d


def rebound_features(t, z, t_release, smooth_frames, direction=1.0):
    """Rebound extraction (declared): moving average over ``smooth_frames`` frames; p0 = z at the release frame;
    p1 = first local maximum of ``direction`` z after release (the tip load acts in -z, so the rebound is upward:
    direction +1), p2 = the next local minimum; parabolic refinement. Rebound time = first crossing of the midpoint
    (p0 + p1) / 2 after release (linear interpolation), log decrement = 2 ln(|p1 - p0| / |p1 - p2|)."""
    zs = _moving_average(np.asarray(z, float), smooth_frames)
    i0 = int(np.searchsorted(t, t_release-1e-12))
    p0 = zs[i0]
    sign = 1.0 if direction > 0 else -1.0
    y = sign*zs
    peaks = []
    i = i0+1
    want_max = True
    while i < len(y)-1 and len(peaks) < 2:
        if want_max and y[i] >= y[i-1] and y[i] > y[i+1]:
            peaks.append(i)
            want_max = False
        elif not want_max and y[i] <= y[i-1] and y[i] < y[i+1]:
            peaks.append(i)
            want_max = True
        i += 1
    out = {"p0": float(p0), "found_peaks": len(peaks)}
    if not peaks:
        return {**out, "rebound_time_s": float("nan"), "log_decrement": float("nan")}
    t1, p1 = _parabolic(t, zs, peaks[0])
    mid = 0.5*(p0+p1)
    tc = float("nan")
    for j in range(i0, peaks[0]):
        a, b = zs[j]-mid, zs[j+1]-mid
        if a == 0 or a*b < 0:
            tc = float(t[j]+(t[j+1]-t[j])*(a/(a-b) if a != b else 0.0))-t_release
            break
    out.update(p1=float(p1), t1_s=float(t1-t_release), rebound_time_s=tc)
    if len(peaks) > 1:
        _, p2 = _parabolic(t, zs, peaks[1])
        h1, h2 = abs(p1-p0), abs(p1-p2)
        out.update(p2=float(p2), log_decrement=float(2.0*math.log(h1/h2)) if h1 > 0 and h2 > 0 else float("nan"))
    else:
        out["log_decrement"] = float("nan")
    return out


def slip_features(t, x, push_start, threshold, end_frames=5):
    """Slip distance = mean of the last ``end_frames`` frames - mean of the ``end_frames`` frames before push start;
    onset = first frame after push start whose displacement exceeds ``threshold`` (linear interpolation)."""
    i0 = int(np.searchsorted(t, push_start-1e-12))
    base = float(np.mean(x[max(0, i0-end_frames):max(i0, 1)]))
    d = np.asarray(x)-base
    dist = float(np.mean(x[-end_frames:])-base)
    onset = float("nan")
    for j in range(max(i0, 1), len(d)):
        if d[j] > threshold:
            a, b = d[j-1]-threshold, d[j]-threshold
            onset = float(t[j-1]+(t[j]-t[j-1])*(a/(a-b) if a != b else 0.0))
            break
    return {"slip_distance_m": dist, "onset_time_s": onset}


METRICS_OF_TYPE = {"tensile_load_unload": ("M1", "M2", "M8", "M9"), "stress_relaxation": ("M1", "M3", "M8", "M9"),
                   "cantilever_rebound": ("M4", "M5", "M6", "M8", "M9"), "gravity_sag": ("M4", "M8", "M9"),
                   "friction_sliding": ("M7_slip", "M7_onset", "M8", "M9")}


def metrics(exp, pred_ch, obs_ch, pred_oracle, obs_oracle, settings):
    """Metric values of one experiment: prediction channels vs observed channels (both possibly noisy); oracle
    metrics M8 / M9 from the noise-free records (``*_oracle``: dict volume_ratio [K+1], dissipated_energy_j [K+1]).
    ``settings``: the protocol's metric definitions (smoothing, windows, thresholds). Returns {metric: value}."""
    kind = exp["type"]
    ld = exp["loading"]
    out = {}
    if kind in ("tensile_load_unload", "stress_relaxation"):
        fp, fo = pred_ch["force"]["force_n"], obs_ch["force"]["force_n"]
        out["M1"] = float(np.sqrt(np.mean((fp-fo)**2))/np.max(np.abs(fo)))
        if kind == "tensile_load_unload":
            def area(f, u):
                return float(np.sum(0.5*(f[1:]+f[:-1])*np.diff(u)))
            ap = area(fp, pred_ch["force"]["displacement_m"])
            ao = area(fo, obs_ch["force"]["displacement_m"])
            out["M2"] = abs(ap-ao)/abs(ao)
        else:
            n = int(settings["M3_end_samples"])
            ep, eo = float(np.mean(fp[-n:])), float(np.mean(fo[-n:]))
            out["M3"] = abs(ep-eo)/abs(eo)
    elif kind in ("cantilever_rebound", "gravity_sag"):
        t = obs_ch["video"]["time_s"]
        zp, zo = pred_ch["video"]["m8_z_m"], obs_ch["video"]["m8_z_m"]
        if kind == "cantilever_rebound":
            sel = t >= ld["release_s"]-1e-12
            amp = float(np.max(zo[sel])-np.min(zo[sel]))
            out["M4"] = float(np.sqrt(np.mean((zp[sel]-zo[sel])**2))/amp)
            n = int(settings["rebound_smoothing_frames"])
            fp = rebound_features(t, zp, ld["release_s"], n)
            fo = rebound_features(t, zo, ld["release_s"], n)
            out["M5"] = abs(fp["rebound_time_s"]-fo["rebound_time_s"])*1e3
            out["M6"] = abs(fp["log_decrement"]-fo["log_decrement"])
        else:
            sel = t >= ld["static_window_s"][0]-1e-12
            z0 = 0.0      # reference tip z is the specimen mid-plane (t / 2)
            sag_o = float(np.mean(zo[sel]))
            ref = exp["specimen"]["thickness"]/2+z0
            out["M4"] = abs(float(np.mean(zp[sel]))-sag_o)/abs(sag_o-ref)
    else:
        t = obs_ch["video"]["time_s"]
        th = float(settings["slip_onset_threshold_m"])
        fp = slip_features(t, pred_ch["video"]["centroid_x_m"], ld["push_start_s"], th)
        fo = slip_features(t, obs_ch["video"]["centroid_x_m"], ld["push_start_s"], th)
        out["M7_slip"] = abs(fp["slip_distance_m"]-fo["slip_distance_m"])/abs(fo["slip_distance_m"])
        out["M7_onset"] = abs(fp["onset_time_s"]-fo["onset_time_s"])*1e3
    if pred_oracle is not None and obs_oracle is not None:
        out["M8"] = float(np.sqrt(np.mean((np.asarray(pred_oracle["volume_ratio"])
                                           -np.asarray(obs_oracle["volume_ratio"]))**2)))
        dp, do = float(pred_oracle["dissipated_energy_j"][-1]), float(obs_oracle["dissipated_energy_j"][-1])
        out["M9"] = abs(dp-do)/abs(do)
    return out


def oracle(obs):
    """The noise-free oracle quantities of a record used by M8 / M9 (synthetic data only)."""
    return {"volume_ratio": np.asarray(obs["volume_ratio"]).tolist(),
            "dissipated_energy_j": np.asarray(obs["dissipated_energy_j"]).tolist()}


# ============================================================================ CSV I/O

def write_channels(path, columns):
    names = list(columns)
    data = np.column_stack([np.asarray(columns[n], np.float64) for n in names])
    with open(path, "w") as fh:
        fh.write(",".join(names)+"\n")
        for row in data:
            fh.write(",".join(repr(float(v)) for v in row)+"\n")


def read_channels(path):
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        data = np.loadtxt(fh, delimiter=",", ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


COLUMN_UNITS = {"time_s": "s", "displacement_m": "m", "force_n": "N", "friction_force_n": "N", "normal_force_n": "N",
                "centroid_x_m": "m"}


def column_units(columns):
    return {n: COLUMN_UNITS.get(n, "m") for n in columns}
