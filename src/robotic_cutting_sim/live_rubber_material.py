"""Single-env rubber runtime wrapper ``LiveRubberMaterial`` (ADR 0023 §7, §13; task R1.4).

Builds the ``rubber_strip_v1`` scene and the ``tlmpm_rubber_v1`` solver from a recipe, applies the ADR §7 step rule
(dt rounded down so that control_dt / dt is an integer), and advances one control step at a time:

    advance(motions={g: {...}}, traction={t: force}) -> group wrenches [G, 6]

On CUDA the substeps of a control step run as CUDA graphs of ``graph_chunk`` substeps (default 500) replayed
``substeps // chunk`` times plus one remainder graph; the device substep index carries across replays, so the group
interpolation is continuous. On CPU the substeps run eagerly (same launches). No host work between chunks.

After each control step: segmented diagnostics, the sticky validity flags (``NONFINITE_STATE``, ``RUBBER_J_DOMAIN``,
``RUBBER_CFL_EXCEEDED`` with the opt-in ``stable_step_guard``; strict mode raises ``RubberInvalidState``, training mode
only records) and, if ``record=True``, one
``rcs_rubber_record_v1`` step record. The coupling surface for P2-6 Native hosting is ``reset()`` /
``advance(motions) -> wrench``. Material status: ``SIMULATOR_ONLY_UNCALIBRATED``.
"""
import math
import time

import numpy as np
import warp as wp

from . import rubber_material as rm
from . import rubber_scene as rs
from . import rubber_self_penetration as rsp
from .tlmpm_rubber_v1 import DIAG_COLUMNS, SOLVER_NAME, STATUS, TLMPMRubber

RECORD_SCHEMA = "rcs_rubber_record_v1"
INVALID_NONFINITE_STATE = 1
INVALID_RUBBER_J_DOMAIN = 2
INVALID_RUBBER_CFL_EXCEEDED = 4
INVALID_RUBBER_SELF_CONTACT_UNRESOLVED = rsp.INVALID_RUBBER_SELF_CONTACT_UNRESOLVED   # 8, S1 detector (ADR 0023 §23)
INVALID_REASONS = {INVALID_NONFINITE_STATE: "NONFINITE_STATE", INVALID_RUBBER_J_DOMAIN: "RUBBER_J_DOMAIN",
                   INVALID_RUBBER_CFL_EXCEEDED: "RUBBER_CFL_EXCEEDED",
                   INVALID_RUBBER_SELF_CONTACT_UNRESOLVED: rsp.REASON_NAME}
DEFAULT_GRAPH_CHUNK = 500
SPEED_COLUMN = DIAG_COLUMNS.index("max_speed_m_s")


class RubberInvalidState(FloatingPointError):
    def __init__(self, reasons, step):
        self.reasons, self.step = list(reasons), int(step)
        super().__init__(f"rubber state invalid after control step {step}: {self.reasons}")


def decode_reason(value):
    return [name for bit, name in INVALID_REASONS.items() if int(value) & bit]


class LiveRubberMaterial:
    """See the module docstring. ``material`` defaults to the recipe nominal values; ``step_material`` (default:
    ``material``) gives the c_p,max of the step rule (a batch passes its randomisation maxima).
    ``stable_step_guard`` (None = off, the v1 behaviour; a float = the limit of dt c_bound / dx, recommended
    ``rubber_constitutive.STABLE_STEP_GUARD_DEFAULT_LIMIT``) and ``deformation_reserve`` (>= 1, divides dt; 1 = v1)
    are the ADR 0023 §7 amendment options; ``dirichlet``, ``traction_work`` and ``state_precision`` are the P2-2 opt-in
    solver options (ADR 0023 §19, ``tlmpm_rubber_v1`` module docstring; defaults = v1); ``constraint_work``
    ("end_velocity_v1" default, "kick_average_v1" with the F8 projection only) is the bookkeeping option of ADR 0023
    §21.1 (``rubber_constraint_work``; the default builds the unchanged ``TLMPMRubber``). A material with Prony terms
    (``MaterialValues.prony``, model ``neo_hookean_jbar_prony_v1``) or ``visco_energy="branch_work_v1"`` builds the
    viscoelastic ``tlmpm_rubber_visco_v1.TLMPMRubberVisco`` (ADR 0023 §21.2); its visco diagnostics are read after every
    control step (``last_visco``). The step rule uses mu0 (the instantaneous modulus) of ``step_material``.
    ``self_penetration_check`` ("off_v1" default: no detector, no array, no launch; "surface_grid_v1") is the S1
    detector of ADR 0023 §23 (``rubber_self_penetration``): a check every ``self_penetration_check_substeps``
    substeps and after the last substep of the control step, inside the CUDA graph chunks (aligned to a multiple of
    the check interval); hits set the sticky bit ``RUBBER_SELF_CONTACT_UNRESOLVED`` (8); ``last_self_penetration``
    holds the per-step summary. It reads positions only, so the solver state is the same with it on or off."""

    def __init__(self, spec, *, recipe=None, material=None, control_dt=0.02, cfl=None, substeps=None,
                 step_material=None, gravity=(0.0, 0.0, -9.81), device="cpu", graph_chunk=DEFAULT_GRAPH_CHUNK,
                 capture_graph=None, record=False, strict=True, table_plane=None, fixture_boxes=None,
                 stable_step_guard=None, deformation_reserve=1.0, dirichlet="lumped_v1",
                 traction_work="particle_velocity_v1", state_precision="float32_v1",
                 constraint_work="end_velocity_v1", visco_energy="off_v1", contact=None,
                 self_penetration_check=rsp.DEFAULT_SELF_PENETRATION_CHECK,
                 self_penetration_check_substeps=rsp.DEFAULT_CHECK_SUBSTEPS):
        start = time.perf_counter()
        self.self_penetration_check, self.self_penetration_check_substeps = rsp.check_option(
            self_penetration_check, self_penetration_check_substeps)
        self.recipe = rm.load_recipe() if recipe is None else recipe
        self.material = self.recipe.nominal() if material is None else material
        self.spec = spec
        self.scene = rs.compile_strip(spec)
        self.control_dt = float(control_dt)
        cfl = self.recipe.cfl if cfl is None else float(cfl)
        sm = self.material if step_material is None else step_material
        self.step_rule = rm.step_rule(self.scene.dx, self.control_dt, mu0_max=sm.mu0, kappa_max=sm.kappa,
                                      density_min=sm.density, cfl=cfl, deformation_reserve=deformation_reserve)
        if substeps is not None:
            substeps = int(substeps)
            if substeps < self.step_rule["substeps_per_control_step"]:
                raise ValueError(f"substeps {substeps} below the step rule's "
                                 f"{self.step_rule['substeps_per_control_step']} (CFL {cfl})")
            self.step_rule = dict(self.step_rule, substeps_per_control_step=substeps, dt_s=self.control_dt/substeps,
                                  cfl_actual=self.control_dt/substeps*self.step_rule["c_p_max_m_s"]/self.scene.dx,
                                  substeps_override=True)
        self.substeps = int(self.step_rule["substeps_per_control_step"])
        self.dt = self.control_dt/self.substeps
        solver_cls, extra = TLMPMRubber, {}
        if contact is not None:
            # ADR 0023 §22: contact law grid_node_penalty_coulomb_v1 (opt-in; without a ContactSpec the classes below
            # are built unchanged)
            from .rubber_constraint_work import check_options
            from .tlmpm_rubber_contact_v1 import TLMPMRubberContact
            check_options(constraint_work, dirichlet)
            solver_cls, extra = TLMPMRubberContact, {"constraint_work": constraint_work, "visco_energy": visco_energy,
                                                     "contact": contact}
        elif getattr(self.material, "prony", ()) or visco_energy != "off_v1":
            # ADR 0023 §21.2: neo_hookean_jbar_prony_v1 (the elastic solver classes are used otherwise)
            from .rubber_constraint_work import check_options
            from .tlmpm_rubber_visco_v1 import TLMPMRubberVisco
            check_options(constraint_work, dirichlet)
            solver_cls, extra = TLMPMRubberVisco, {"constraint_work": constraint_work, "visco_energy": visco_energy}
        elif constraint_work != "end_velocity_v1":
            # ADR 0023 §21.1: bookkeeping option of the F8 projection (the frozen solver class is used otherwise)
            from .rubber_constraint_work import TLMPMRubberCW, check_options
            check_options(constraint_work, dirichlet)
            solver_cls, extra = TLMPMRubberCW, {"constraint_work": constraint_work}
        self.solver = solver_cls(self.scene, self.material, self.dt, substeps=self.substeps, gravity=gravity,
                                 device=device, j_domain=self.recipe.j_domain,
                                 contact_stiffness=self.recipe.contact_stiffness_fraction,
                                 contact_radius_scale=self.recipe.contact_radius_scale, table_plane=table_plane,
                                 fixture_boxes=fixture_boxes, stable_step_guard=stable_step_guard,
                                 dirichlet=dirichlet, traction_work=traction_work, state_precision=state_precision,
                                 **extra)
        self.device = self.solver.device
        self.strict, self.record = bool(strict), bool(record)
        self.capture_graph = wp.get_device(self.device).is_cuda if capture_graph is None else bool(capture_graph)
        self.graph_chunk = max(1, min(int(graph_chunk), self.substeps))
        self.self_penetration = None
        self.last_self_penetration = None
        if self.self_penetration_check != rsp.CHECK_OFF:
            s = self.solver
            positions = s.x64 if s.state_precision == "f64_shadow_v1" else s.x
            self.self_penetration = rsp.SelfPenetrationDetector(
                [self.scene], positions, s.reference, [0], device=self.device, dt=self.dt, substeps=self.substeps,
                check_substeps=self.self_penetration_check_substeps)
            with wp.ScopedDevice(self.device):
                self._self_penetration_invalid = wp.zeros(1, dtype=int)
            self.graph_chunk = rsp.aligned_chunk(self.graph_chunk, self.substeps, self.self_penetration_check_substeps)
        self.graphs = None
        self.capture_receipt = None
        self.step_count = 0
        self.invalid_reason = 0
        self.records = []
        self._last_violations = 0
        self._last_cfl_violations = 0
        self.total_mass = float(self.solver.mass_value)*self.scene.n
        if self.capture_graph:
            self._capture()
        self.build_wall_s = time.perf_counter()-start
        self.reset()

    # ------------------------------------------------------------------ graph

    def _capture(self):
        if not wp.get_device(self.device).is_cuda:
            raise ValueError("CUDA graph capture needs a CUDA device")
        start = time.perf_counter()
        chunk = self.graph_chunk
        repeats, rest = divmod(self.substeps, chunk)
        sp = self.self_penetration
        if sp is not None:
            sp.check()          # warm-up outside the capture (radix-sort scratch); reset() follows the capture
        graphs = []
        for size, first in ((chunk, 0), (rest, repeats*chunk)):
            if size == 0:
                graphs.append(None)
                continue
            with wp.ScopedCapture(device=self.device) as capture:
                if sp is None:
                    for _ in range(size):
                        self.solver.step()
                else:
                    sp.run_substeps(self.solver.step, size, first)
            graphs.append(capture.graph)
        self.graphs = (graphs[0], repeats, graphs[1])
        self.capture_receipt = {"graph_chunk_substeps": chunk, "chunk_replays": repeats, "remainder_substeps": rest,
                                "capture_wall_s": time.perf_counter()-start}

    def _run_substeps(self):
        if self.graphs is not None:
            main, repeats, rest = self.graphs
            for _ in range(repeats):
                wp.capture_launch(main)
            if rest is not None:
                wp.capture_launch(rest)
        elif self.self_penetration is not None:
            self.self_penetration.run_substeps(self.solver.step, self.substeps)
        else:
            for _ in range(self.substeps):
                self.solver.step()

    # ------------------------------------------------------------------ episode

    def reset(self):
        self.solver.reset()
        self.step_count = 0
        self.invalid_reason = 0
        self._last_violations = 0
        self._last_cfl_violations = 0
        self.last_visco = None
        self.last_contact = None
        if self.self_penetration is not None:
            self.self_penetration.reset_all()
            self._self_penetration_invalid.zero_()
            self.last_self_penetration = None
        self.records = [self.header()] if self.record else []
        return self.receipt_short()

    def advance(self, motions=None, traction=None, frames=None):
        """One control step. ``motions``: {group: dict(rotation=3x3, translation=3, affine=3x3, active=bool)} giving
        the END of the step (relative to the reference); unlisted groups hold. ``traction``: {group: force [3] N}.
        ``frames`` (contact solver only, ADR 0023 §22): {contact frame: dict(rotation=3x3, translation=3,
        active=bool)}, the END pose of each moving contact frame (pads); unlisted frames hold.
        Returns the group wrenches [G, 6] (force the material exerts on the group, averaged over the step); the
        contact wrenches of the step are in ``last_contact`` (planes [P, 6], boxes [F, B, 6])."""
        s = self.solver
        s.begin_control_step()
        for g, m in (motions or {}).items():
            s.set_group_motion(int(g), **m)
        if frames:
            if not hasattr(s, "set_frame_motion"):
                raise ValueError("frames need a contact solver (LiveRubberMaterial(contact=ContactSpec(...)))")
            for f, m in frames.items():
                s.set_frame_motion(int(f), **m)
        for t, f in (traction or {}).items():
            s.set_traction(int(t), f)
        self._run_substeps()
        self.step_count += 1
        wrench = s.end_control_step(self.control_dt)
        self.last_wrench = wrench
        diag = s.diagnostics.read()[0]
        if getattr(s, "visco_diagnostics", None) is not None:
            s.visco_diagnostics.launch(s.diagnostics.summary, s.dparams, s.particle_offset)
            self.last_visco = s.visco_diagnostics.read()[0]
        if hasattr(s, "contact_wrenches"):
            planes, boxes = s.contact_wrenches(self.control_dt)
            self.last_contact = {"plane_wrench": planes, "box_wrench": boxes, **s.contact_diagnostics()}
        violations = int(s.violations.numpy()[0])
        step_violations = violations-self._last_violations
        self._last_violations = violations
        if diag["nonfinite"] != 0.0 or not np.isfinite(wrench).all():
            self.invalid_reason |= INVALID_NONFINITE_STATE
        if violations > 0:
            self.invalid_reason |= INVALID_RUBBER_J_DOMAIN
        cfl_violations = int(s.cfl_violations.numpy()[0])
        step_cfl_violations = cfl_violations-self._last_cfl_violations
        self._last_cfl_violations = cfl_violations
        if cfl_violations > 0:
            self.invalid_reason |= INVALID_RUBBER_CFL_EXCEEDED
        sp = self.self_penetration
        if sp is not None:
            sp.post(s.diagnostics.summary, self._self_penetration_invalid, SPEED_COLUMN)
            self.invalid_reason |= int(self._self_penetration_invalid.numpy()[0])
            self.last_self_penetration = sp.read()[0]
        self.last_diagnostics = diag
        if self.record:
            self.records.append(self.step_record(diag, wrench, step_violations, step_cfl_violations))
        if self.strict and self.invalid_reason:
            raise RubberInvalidState(decode_reason(self.invalid_reason), self.step_count)
        return wrench

    # ------------------------------------------------------------------ records

    def header(self):
        sc = self.scene
        return {"schema": RECORD_SCHEMA, "kind": "header", "solver": SOLVER_NAME, "solver_status": STATUS,
                "calibration_status": self.recipe.calibration_status, "recipe_id": self.recipe.id,
                "scene": sc.receipt(), "material": self.material.to_dict(), "step_rule": self.step_rule,
                "surface_particle_ids": np.flatnonzero(sc.surface).tolist(),
                "marker_offsets": sc.marker_offsets.tolist(), "marker_particles": sc.marker_particles.tolist(),
                "marker_reference_m": sc.marker_reference.tolist(),
                "groups": [{"name": g["name"], "kind": g["kind"], "nodes": int(g["nodes"])} for g in sc.groups],
                "traction_groups": [dict(t) for t in sc.traction_groups]}

    def step_record(self, diag, wrench, step_violations, step_cfl_violations=0):
        s = self.solver
        work = float(s.work.numpy()[0])
        energies = {"kinetic_particle": diag["kinetic_particle"], "kinetic_grid": diag["kinetic_grid"],
                    "elastic": diag["elastic"], "gravity": diag["gravity"], "external_work": work}
        energies["total"] = energies["kinetic_particle"]+energies["elastic"]+energies["gravity"]-work
        return {"schema": RECORD_SCHEMA, "kind": "step", "step": self.step_count,
                "t_s": self.step_count*self.control_dt, "substeps": self.substeps,
                "markers_m": s.diagnostics.marker_x.numpy().tolist(),
                "marker_velocities_m_s": s.diagnostics.marker_v.numpy().tolist(),
                "j_stats": {k: diag[k] for k in ("J_min", "J_max", "Jbar_min", "Jbar_max", "volume_ratio",
                                                 "frac_cells_jbar_dev_gt_0_05")},
                "energies_j": energies, "bc_wrench": np.asarray(wrench).reshape(-1, 6).tolist(),
                "bc_active": s.group_active.numpy()[:s.G].tolist(),
                "traction_n": s.traction_force.numpy()[:s.Gt].tolist(), "invalid_reason": int(self.invalid_reason),
                "invalid_flags": decode_reason(self.invalid_reason), "j_domain_violations": int(step_violations),
                "max_speed_m_s": diag["max_speed_m_s"], "nonfinite": bool(diag["nonfinite"] != 0.0),
                "stable_step": {"cfl_eff_max_end_of_step": diag["cfl_eff_max"], "guard_limit": s.cfl_guard_limit,
                                "guard_violations_step": int(step_cfl_violations)},
                **({"visco": dict(self.last_visco)} if getattr(self, "last_visco", None) else {}),
                **({"contact": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in self.last_contact.items()}}
                   if getattr(self, "last_contact", None) else {}),
                **({"self_penetration": dict(self.last_self_penetration)}
                   if getattr(self, "last_self_penetration", None) else {})}

    # ------------------------------------------------------------------ queries

    def markers(self):
        return self.solver.diagnostics.marker_x.numpy()

    def receipt_short(self):
        return {"solver": SOLVER_NAME, "substeps": self.substeps, "dt_s": self.dt, "particles": self.scene.n}

    def receipt(self):
        return {"runtime": "live_rubber_material_v1", "solver": self.solver.solver_receipt(),
                "recipe": self.recipe.receipt(), "scene": self.scene.receipt(), "step_rule": self.step_rule,
                "control_dt_s": self.control_dt, "graph": self.capture_receipt, "strict": self.strict,
                "calibration_status": self.recipe.calibration_status, "memory_bytes": self.solver.memory_bytes(),
                "build_wall_s": self.build_wall_s, "invalid_reason": decode_reason(self.invalid_reason),
                "stable_step_guard": self.solver.solver_receipt()["stable_step_guard"],
                **({"self_penetration": self.self_penetration.receipt()} if self.self_penetration is not None
                   else {})}


# ----------------------------------------------------------------------------- record validation (RUBBER-PHYS-09)

_HEADER_KEYS = ("schema", "kind", "solver", "solver_status", "calibration_status", "scene", "material", "step_rule",
                "surface_particle_ids", "marker_offsets", "marker_particles", "marker_reference_m", "groups",
                "traction_groups")
_STEP_KEYS = ("schema", "kind", "step", "t_s", "substeps", "markers_m", "marker_velocities_m_s", "j_stats",
              "energies_j", "bc_wrench", "bc_active", "traction_n", "invalid_reason", "invalid_flags",
              "j_domain_violations", "max_speed_m_s", "nonfinite")
_J_KEYS = ("J_min", "J_max", "Jbar_min", "Jbar_max", "volume_ratio", "frac_cells_jbar_dev_gt_0_05")
_E_KEYS = ("kinetic_particle", "kinetic_grid", "elastic", "gravity", "external_work", "total")


def validate_records(records, *, expected_steps=None):
    """Check a record list against ``rcs_rubber_record_v1``; returns a list of problems (empty = valid)."""
    problems = []
    if not records or records[0].get("kind") != "header":
        return ["first record is not a header"]
    h = records[0]
    problems += [f"header missing {k}" for k in _HEADER_KEYS if k not in h]
    if h.get("schema") != RECORD_SCHEMA:
        problems.append("header schema")
    if h.get("calibration_status") != "SIMULATOR_ONLY_UNCALIBRATED":
        problems.append("header calibration_status")
    K = len(h.get("marker_offsets", []))-1
    if K != rs.MARKERS:
        problems.append(f"marker CSR has {K} markers")
    if not h.get("surface_particle_ids"):
        problems.append("no surface particles")
    G = len(h.get("groups", []))
    Gt = len(h.get("traction_groups", []))
    steps = records[1:]
    if expected_steps is not None and len(steps) != expected_steps:
        problems.append(f"{len(steps)} step records, expected {expected_steps}")
    for i, r in enumerate(steps, start=1):
        missing = [k for k in _STEP_KEYS if k not in r]
        if missing:
            problems.append(f"step {i} missing {missing}")
            continue
        if r["kind"] != "step" or r["schema"] != RECORD_SCHEMA or r["step"] != i:
            problems.append(f"step {i} kind/schema/index")
        for key in ("markers_m", "marker_velocities_m_s"):
            a = np.asarray(r[key], float)
            if a.shape != (K, 3) or not np.isfinite(a).all():
                problems.append(f"step {i} {key} shape {a.shape} or non-finite")
        w = np.asarray(r["bc_wrench"], float).reshape(-1, 6) if G else np.zeros((0, 6))
        if w.shape != (G, 6) or not np.isfinite(w).all():
            problems.append(f"step {i} bc_wrench")
        if len(r["bc_active"]) != G or len(r["traction_n"]) != Gt:
            problems.append(f"step {i} group arrays")
        for key, names in (("j_stats", _J_KEYS), ("energies_j", _E_KEYS)):
            block = r[key]
            if any(n not in block or not math.isfinite(float(block[n])) for n in names):
                problems.append(f"step {i} {key} incomplete or non-finite")
        if not math.isfinite(r["max_speed_m_s"]) or not isinstance(r["nonfinite"], bool):
            problems.append(f"step {i} speed / nonfinite flag")
    return problems
