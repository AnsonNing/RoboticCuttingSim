"""``RubberStripTaskFamily``: the P2-6 rubber task family behind ``open_vector_env(backend="kinematic_grip_batched")``.

ADR 0023 §23.2 decision 7; plan ``docs/reports/rl_ready/P2_REMAINING_PLAN.md`` §4.1-§4.4. The task config
``configs/rubber/rl/rubber_strip_drag_v1.json`` (SHA-pinned at P6.0) declares the scene (a case-A strip on the table
plane z = 0, a tip traction region, no contact frames), the slot set (L 100 / 125 / 150 mm x W 16 mm x t 4 / 5 / 6 mm
at dx 1 mm, ``rubber_ref_v1``), the solver profile ``rubber_rl_env_v1``, the per-reset randomisation (mu0, rho,
Prony g / tau scales, table friction inside the contact envelope, tip traction, seeded preload, goal), the action,
observation, reward, done and evaluator definitions.

This module is host-side: it loads and checks the config (recipe and envelope SHA pins, ranges inside the recipe,
the envelope and the batch step rule), selects slots, builds the single-env CPU references a
``BatchedRubberMaterial`` packs (``build_life``) and draws the per-reset values (``sample``). The vector env is
``rl.rubber_env.RubberVectorEnv``. Material status: ``SIMULATOR_ONLY_UNCALIBRATED``.
"""
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASK_CONFIG = ROOT/"configs"/"rubber"/"rl"/"rubber_strip_drag_v2.json"
TASK_CONFIG_SHA256 = "7395e6104cab16a247dc03afb133f8bd65c96d68a0d01d058bc59e6fbbf2addb"   # P6.0 v2 pin (400f5c5)
# superseded, never run formally (K2-1: the v3 contact law is wrong for spin / sideways drag); kept loadable by path
TASK_CONFIG_V1_SHA256 = "fba43424372e17fe926fad697ff8336ea7c5ed0c2d8ecdbdefb9978a994f4397"
FAMILY_VERSION = "rubber_strip_task_family_v1"
TASK_IDS = {"rubber_strip_drag_v2": TASK_CONFIG,
            "rubber_strip_drag_v1": ROOT/"configs"/"rubber"/"rl"/"rubber_strip_drag_v1.json"}
VARIANTS = ("cpu_parity_v1",)
RUBBER_CODES = ("UNSUPPORTED_BACKEND", "UNSUPPORTED_TASK", "UNSUPPORTED_OPTION", "UNSUPPORTED_SENSOR",
                "UNSUPPORTED_RECORDER", "UNSUPPORTED_RUNTIME", "UNSUPPORTED_DEVICE",
                "UNSUPPORTED_CPU_WITHOUT_PARITY_MODE", "UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET",
                "UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE", "UNSUPPORTED_NONUNIFORM_BATCH")
DRAW_ORDER = ("mu0_pa", "density_kg_m3", "prony_g_scale", "prony_tau_scale", "friction_table",
              "traction_horizontal_fraction", "traction_direction_rad", "traction_vertical_fraction",
              "preload_sweep_y_m", "preload_yaw_rad", "goal_radius_m", "goal_angle_rad")
G_STANDARD = 9.81


class UnsupportedRubberConfiguration(ValueError):
    """A rubber vector-env configuration that is not implemented / not allowed; never a silent fallback."""

    def __init__(self, code, detail=""):
        if code not in RUBBER_CODES:
            raise ValueError(f"unknown UNSUPPORTED code {code!r}")
        self.code, self.detail = code, str(detail)
        ValueError.__init__(self, code if not detail else f"{code}: {detail}")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task_config(config=TASK_CONFIG, expected_sha256=None):
    """(dict, path, sha256) of a task config given by id, path or dict (a dict has no SHA)."""
    if isinstance(config, dict):
        return copy.deepcopy(config), None, None
    path = TASK_IDS.get(str(config), Path(config))
    path = Path(path) if Path(path).is_absolute() else ROOT/path
    sha = _sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"task config {path} sha256 {sha} != pinned {expected_sha256}")
    return json.loads(path.read_text()), path, sha


class RubberStripTaskFamily:
    """Per-env slot selection plus the declared task (module docstring).

    ``slots``: one slot id per env (e.g. ``["L125_t4"] * 8``), or None with ``num_envs`` and ``slot_seed`` for the
    declared seeded selection. ``config``: the task id ``"rubber_strip_drag_v1"``, a path or a dict (tests).
    ``variant``: None (the declared task) or ``"cpu_parity_v1"`` (the config's CPU test configuration, labelled in
    every receipt). ``max_episode_steps`` overrides the horizon (recorded in the receipt)."""
    is_rubber_task_family = True        # duck-typed marker read by the open_vector_env dispatch (no import there)

    def __init__(self, slots=None, *, num_envs=None, slot_seed=0, config="rubber_strip_drag_v2", variant=None,
                 max_episode_steps=None, resolution=None, expected_sha256=TASK_CONFIG_SHA256):
        if config == "rubber_strip_drag_v1" and expected_sha256 == TASK_CONFIG_SHA256:
            expected_sha256 = TASK_CONFIG_V1_SHA256          # superseded v1 (never run formally), by id only
        cfg, path, sha = load_task_config(config, expected_sha256)
        if cfg.get("schema") != "rcs_rubber_rl_task_v1":
            raise ValueError("task config schema must be rcs_rubber_rl_task_v1")
        if variant is not None and variant not in VARIANTS:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", f"variant {variant!r}; supported {VARIANTS}")
        self.config, self.config_path, self.config_sha256, self.variant = cfg, path, sha, variant
        self.id = cfg["id"]
        v = cfg["test_configurations"][variant] if variant else {}
        sc = cfg["scene"]
        self.width = float(v.get("width_m", sc["width_m"]))
        self.dx = float(sc["dx_m"])
        self.jaw_length = float(sc["jaw_length_m"])
        self.tip_region = float(sc["traction_region"]["x_from_tip_m"])
        self.gravity = tuple(float(g) for g in sc["gravity_m_s2"])
        self.slot_set = {s["id"]: dict(s) for s in v.get("slots", cfg["slots"])}
        self.resolution, self.resolution_n_t = resolution, None
        if resolution is not None:
            rv = cfg.get("resolution_variants", {})
            if resolution not in rv or variant is not None:
                raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", f"resolution {resolution!r} (declared: "
                                                     f"{sorted(rv)}; not with a test variant)")
            self.slot_set = {k: dict(self.slot_set[k], n_t=int(rv[resolution]["n_t"])) for k in rv[resolution]["slots"]}
            self.resolution_n_t = int(rv[resolution]["n_t"])
        self.slot_ids = tuple(self.slot_set)
        for s in self.slot_set.values():
            n_t = s["thickness_m"]/self.dx
            if resolution is None and (abs(n_t-round(n_t)) > 1e-9 or int(round(n_t)) != s["n_t"]):
                raise ValueError(f"slot {s['id']}: n_t {s['n_t']} != t / dx")
        self.control_dt = float(v.get("control_dt_s", cfg["control_dt_s"]))
        pre = dict(cfg["preload"], **v.get("preload", {}))
        self.preload_steps, self.preload_lift = int(pre["steps"]), float(pre["lift_m"])
        act = dict(cfg["action"], **v.get("action_overrides", {}))
        self.max_translation = float(act["max_translation_m_per_step"])
        self.max_rotation = float(act["max_rotation_rad_per_step"])
        ws = dict(cfg["action"]["workspace"])
        if "grip_origin_offset_box_m" in v.get("action_overrides", {}):
            ws["grip_origin_offset_box_m"] = v["action_overrides"]["grip_origin_offset_box_m"]
        self.workspace_box = np.asarray(ws["grip_origin_offset_box_m"], np.float64)
        self.jaw_min_clearance = float(ws["jaw_min_clearance_m"])
        self.tilt_max = float(ws["tilt_max_rad"])
        ev = dict(cfg["evaluator"], **v.get("evaluator_overrides", {}))
        self.eval_window, self.eval_tolerance = int(ev["window_steps"]), float(ev["tolerance_m"])
        horizon = v.get("max_episode_steps", cfg["done"]["max_episode_steps"])
        self.max_episode_steps = int(horizon if max_episode_steps is None else max_episode_steps)
        if self.max_episode_steps <= self.preload_steps:
            raise ValueError("max_episode_steps must exceed the preload steps")
        # recipe, profile, contact law and envelope
        from .. import rubber_material as rm
        from ..rubber_contact_envelope import load_envelope
        rec = cfg["recipe"]
        self.recipe = rm.load_recipe(ROOT/rec["path"])
        if self.recipe.source_sha256 != rec["sha256"]:
            raise ValueError(f"recipe {rec['path']} sha256 {self.recipe.source_sha256} != {rec['sha256']}")
        if self.recipe.calibration_status != "SIMULATOR_ONLY_UNCALIBRATED":
            raise ValueError("P2-6 v1 runs SIMULATOR_ONLY_UNCALIBRATED recipes only")
        self.profile = dict(cfg["profile"])
        self.contact_cfg = dict(cfg["contact"])
        env_cfg = self.contact_cfg["envelope"]
        self.envelope = load_envelope(ROOT/env_cfg["path"], expected_sha256=env_cfg["sha256"])
        self.law = self.contact_cfg["law"]
        self.n_ts = tuple(sorted({int(s["n_t"]) for s in self.slot_set.values()}))
        # ranges: declared, inside the recipe, the envelope and the step rule
        rnd = cfg["randomisation"]
        if tuple(rnd["order"]) != DRAW_ORDER:
            raise ValueError("randomisation order differs from the declared DRAW_ORDER")
        self.ranges = {k: (float(lo), float(hi)) for k, (lo, hi) in rnd["ranges"].items()}
        self.ranges.update({k: (float(lo), float(hi)) for k, (lo, hi) in v.get("randomisation_overrides",
                                                                               {}).items()})
        lo, hi = self.envelope.friction_range(self.law, self.n_ts, lower=self.ranges["friction_table"][0],
                                              general_slip=True)
        if self.ranges["friction_table"][1] > hi or self.ranges["friction_table"][0] < lo:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE",
                                                 f"declared {self.ranges['friction_table']} vs envelope [{lo}, {hi}]")
        for n_t in self.n_ts:
            self.envelope.check_friction(self.law, n_t, self.ranges["friction_table"][1], general_slip=True)
        sm = cfg["step_material"]
        self.step_material = rm.MaterialValues(mu0=float(sm["mu0_pa"]), kappa=float(sm["bulk_ratio_beta"])*float(
            sm["mu0_pa"]), density=float(sm["density_kg_m3"]))
        mu_lo, mu_hi = self.ranges["mu0_pa"]
        rho_lo, rho_hi = self.ranges["density_kg_m3"]
        if mu_hi > self.step_material.mu0 or rho_lo < self.step_material.density:
            raise ValueError("randomisation ranges exceed the declared step material (batch step rule)")
        g_sum = sum(t.g for t in self.recipe.prony)
        for mu0 in (mu_lo, mu_hi):                       # mu_eq = mu0 g_inf at the corners of (mu0, g scale)
            for gs in self.ranges["prony_g_scale"]:
                mu_eq = mu0*(1.0-g_sum*gs)
                if not (g_sum*gs < 1.0 and self.recipe.shear_modulus_eq.lo <= mu_eq <= self.recipe.shear_modulus_eq.hi):
                    raise ValueError(f"mu_eq {mu_eq} (mu0 {mu0}, g scale {gs}) outside the recipe range")
        if not (self.recipe.density.lo <= rho_lo and rho_hi <= self.recipe.density.hi):
            raise ValueError("density range outside the recipe range")
        # slots per env
        if slots is None:
            if num_envs is None:
                raise ValueError("pass slots (one id per env) or num_envs with slot_seed")
            idx = np.random.default_rng(int(slot_seed)).integers(0, len(self.slot_ids), int(num_envs))
            slots = [self.slot_ids[i] for i in idx]
            self.slot_selection = {"rule": "seeded", "slot_seed": int(slot_seed)}
        else:
            slots = list(slots)
            if num_envs is not None and len(slots) != int(num_envs):
                raise ValueError("len(slots) != num_envs")
            self.slot_selection = {"rule": "explicit"}
        unknown = [s for s in slots if s not in self.slot_set]
        if unknown or not slots:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_TASK", f"unknown slots {unknown} (set {self.slot_ids})")
        self.env_slots = tuple(slots)
        self.num_envs = len(slots)
        self._geometry = {}

    # ------------------------------------------------------------------ scene

    def spec(self, slot, *, n_t=None):
        """RubberStripSpec of a slot (``n_t`` override: the RL-08 spacing rows only, dx = t / n_t)."""
        from ..rubber_scene import RubberStripSpec
        s = self.slot_set[slot]
        L, t = float(s["length_m"]), float(s["thickness_m"])
        if n_t is None and self.resolution_n_t is not None:
            n_t = self.resolution_n_t                     # labelled reduced-fidelity variant (RL-07 rows only)
        n = int(s["n_t"] if n_t is None else n_t)
        dx = self.dx if n_t is None else t/n
        return RubberStripSpec(length=L, width=self.width, thickness=t, n_t=n, dx=dx, case="A",
                               jaw_length=self.jaw_length if n_t is None else 2*dx,
                               traction_regions=(("tip", (L-self.tip_region, -1.0, -1.0), (1.0, 1.0, 1.0)),),
                               resolution_profile="rubber_rl_v1" if self.resolution == "rubber_rl_v1"
                               else "rubber_ref_v1")

    def contact_spec(self):
        from ..rubber_contact import ContactSpec
        lp = self.contact_cfg["law_parameters"]
        if self.contact_cfg.get("frames"):
            raise UnsupportedRubberConfiguration("UNSUPPORTED_TASK", "rubber_strip_drag has no contact frames")
        extra = ({"slip_rate_damping_ratio": lp["slip_rate_damping_ratio_zeta_d"]}
                 if "slip_rate_damping_ratio_zeta_d" in lp else {})
        return ContactSpec(planes=tuple(tuple(map(float, p)) for p in self.contact_cfg["planes"]), law=self.law,
                           normal_stiffness=lp["normal_stiffness_kappa_n"],
                           tangential_stiffness=lp["tangential_stiffness_kappa_t"],
                           normal_damping_ratio=lp["normal_damping_ratio_zeta_n"],
                           damping_ratio=lp["tangential_damping_ratio_zeta_t"], **extra)

    def build_material(self):
        """The build-time material: recipe nominal with friction_table = the midpoint of the declared range."""
        lo, hi = self.ranges["friction_table"]
        return dataclasses.replace(self.recipe.nominal(), friction_table=0.5*(lo+hi))

    def build_life(self, slot, material=None, *, device="cpu", capture_graph=None, cfl=None, n_t=None, strict=False,
                   record=False):
        """A single-env ``LiveRubberMaterial`` of a slot with the profile ``rubber_rl_env_v1`` (the batch packs fresh
        CPU lives; on CUDA it is the single GPU reference). ``cfl`` / ``n_t``: RL-08 dt / spacing rows only."""
        from ..live_rubber_material import LiveRubberMaterial
        p = self.profile
        return LiveRubberMaterial(
            self.spec(slot, n_t=n_t), recipe=self.recipe, material=self.build_material() if material is None
            else material, control_dt=self.control_dt, cfl=p["cfl"] if cfl is None else cfl,
            step_material=self.step_material, gravity=self.gravity, device=device, graph_chunk=p["graph_chunk"],
            capture_graph=capture_graph, record=record, strict=strict, stable_step_guard=p["stable_step_guard"],
            deformation_reserve=p["deformation_reserve"], dirichlet=p["dirichlet"], traction_work=p["traction_work"],
            state_precision=p["state_precision"], constraint_work=p["constraint_work"], visco_energy=p["visco_energy"],
            contact=self.contact_spec(), self_penetration_check=p["self_penetration_check"],
            self_penetration_check_substeps=p["self_penetration_check_substeps"])

    def batch_ranges(self):
        """Declared ranges in ``BatchedRubberMaterial`` names (the batch checks every reset against them)."""
        r = self.ranges
        return {"mu0": r["mu0_pa"], "density": r["density_kg_m3"], "friction_table": r["friction_table"],
                "prony_g_scale": r["prony_g_scale"], "prony_tau_scale": r["prony_tau_scale"]}

    def geometry(self, slot):
        """Host geometry of a slot: grip origin o, jaw-node bounding-box corners, tip reference, particle count,
        particle spacing (from the compiled scene; cached)."""
        if slot not in self._geometry:
            from ..rubber_scene import compile_strip
            sc = compile_strip(self.spec(slot))
            jaw = sc.node_X[(sc.node_group == 0) & sc.node_has_mass]
            lo, hi = jaw.min(0), jaw.max(0)
            corners = np.array([[(hi if (c >> k) & 1 else lo)[k] for k in range(3)] for c in range(8)], np.float64)
            s = self.slot_set[slot]
            self._geometry[slot] = {
                "origin": np.asarray(sc.groups[0]["origin"], np.float64), "jaw_corners": corners,
                "tip_reference": np.asarray(sc.marker_reference[-1], np.float64), "particles": int(sc.n),
                "spacing": float(sc.spacing), "length": float(s["length_m"]), "thickness": float(s["thickness_m"]),
                "n_t": int(s["n_t"]), "traction_particles": int(sc.traction_groups[0]["particles"]),
                "surface_particles": int(sc.surface.sum())}
        return self._geometry[slot]

    # ------------------------------------------------------------------ randomisation

    def sample(self, seed, slot):
        """Per-reset values of one env (declared draw order, np.random.default_rng(seed)); returns a dict with the
        raw draws, the ``MaterialValues``, the tip traction [3] (N), the preload (sweep_y, yaw) and the goal [3]."""
        from ..rubber_material import MaterialValues
        from ..batched.rubber_material_batch import scale_prony
        rng = np.random.default_rng(int(seed))
        d = {k: float(rng.uniform(*self.ranges[k])) for k in DRAW_ORDER}
        prony = scale_prony(self.recipe.prony_terms(), d["prony_g_scale"], d["prony_tau_scale"])
        mu0 = d["mu0_pa"]
        m = MaterialValues(mu0=mu0, kappa=20.0*mu0, density=d["density_kg_m3"], friction_table=d["friction_table"],
                           friction_pad=self.recipe.friction_pad.value, friction_self=self.recipe.friction_self.value,
                           prony=prony)
        geo = self.geometry(slot)
        weight = d["density_kg_m3"]*geo["particles"]*geo["spacing"]**3*G_STANDARD
        h, phi, v = d["traction_horizontal_fraction"], d["traction_direction_rad"], d["traction_vertical_fraction"]
        traction = weight*np.array([h*math.cos(phi), h*math.sin(phi), v], np.float64)
        L, t = geo["length"], geo["thickness"]
        r, a = d["goal_radius_m"], d["goal_angle_rad"]
        goal = np.array([L+r*math.cos(a), self.width/2+r*math.sin(a), t/2], np.float64)
        return {"seed": int(seed), "slot": slot, "draws": d, "material": m, "traction_n": traction,
                "weight_n": weight, "preload": np.array([d["preload_sweep_y_m"], d["preload_yaw_rad"]], np.float64),
                "goal_m": goal}

    def check_material(self, m, n_t):
        """Every value inside the declared range and the envelope (raises)."""
        self.envelope.check_friction(self.law, n_t, m.friction_table, general_slip=True)
        for key, value in (("mu0_pa", m.mu0), ("density_kg_m3", m.density), ("friction_table", m.friction_table)):
            lo, hi = self.ranges[key]
            if not lo <= value <= hi:
                raise ValueError(f"{key} = {value} outside [{lo}, {hi}]")

    # ------------------------------------------------------------------ receipts

    def resolution_label(self):
        """The resolution profile label of the slots (K2 Q1w conditions 1 / 2): read defensively from
        ``rubber_material.RESOLUTION_PROFILES`` (``measured_bias`` / ``known_limitation`` land with K2-2)."""
        from ..rubber_material import RESOLUTION_PROFILES
        name = "rubber_rl_v1" if self.resolution == "rubber_rl_v1" else "rubber_ref_v1"
        prof = RESOLUTION_PROFILES.get(name, {})
        return {"resolution_profile": name, "label": prof.get("label"),
                "measured_bias": prof.get("measured_bias", "PENDING_K2-2 (label not yet in RESOLUTION_PROFILES)"),
                "known_limitation": prof.get("known_limitation", "PENDING_K2-2"),
                "task_config_label": self.config["labels"].get("resolution_measured_bias")}

    def to_dict(self):
        return {"id": self.id, "version": FAMILY_VERSION, "config": None if self.config_path is None
                else str(Path(self.config_path).relative_to(ROOT)), "config_sha256": self.config_sha256,
                "variant": self.variant, "test_configuration": self.variant, "num_envs": self.num_envs,
                "resolution_variant": self.resolution,
                "reduced_fidelity": ("thickness_resolution_n_t3" if self.resolution == "rubber_rl_v1" else None),
                "slots": list(self.env_slots), "slot_selection": self.slot_selection}

    def receipt(self):
        out = self.to_dict()
        out.update(
            calibration_status="SIMULATOR_ONLY_UNCALIBRATED", labels=dict(self.config["labels"]),
            resolution=self.resolution_label(),
            recipe={"id": self.recipe.id, "sha256": self.recipe.source_sha256}, profile=self.profile,
            contact_law=self.law, envelope_rows={str(n): self.envelope.receipt_row(self.law, n) for n in self.n_ts},
            ranges={k: list(v) for k, v in self.ranges.items()}, control_dt_s=self.control_dt,
            preload={"steps": self.preload_steps, "lift_m": self.preload_lift},
            max_episode_steps=self.max_episode_steps,
            step_material=self.step_material.to_dict(),
            slot_geometry={s: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in self.geometry(s).items()} for s in sorted(set(self.env_slots))})
        return out


__all__ = ["RubberStripTaskFamily", "UnsupportedRubberConfiguration", "load_task_config", "TASK_CONFIG",
           "TASK_CONFIG_SHA256", "DRAW_ORDER", "FAMILY_VERSION", "RUBBER_CODES"]
