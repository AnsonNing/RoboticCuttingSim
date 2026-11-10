"""Shared helpers of ``rubber_visc_tests_v1`` (ADR 0023 §21.3): the protocol's common options and Prony sets, single
reference builders, and the byte-wise L0 comparison of a batched env against its single-env reference (RUBBER-VISC-06,
-07, -09). Used by ``scripts/rubber/p2_3/run_visc_v1.py`` (formal runs) and the CPU tests (small configurations).
"""
import hashlib
import json

import numpy as np

from . import rubber_scene as rs
from .rubber_material import ROOT, MaterialValues, pad_prony

PROTOCOL_PATH = ROOT/"configs"/"rubber"/"rubber_visc_tests_v1.json"
OPTIONS = {"dirichlet": "consistent_mass_projection_v1", "constraint_work": "kick_average_v1",
           "traction_work": "grid_displacement_v1", "state_precision": "f64_shadow_v1"}
VISCO_ENERGY = "branch_work_v1"
VISCO_ENERGY_V2 = "branch_work_kick_average_v2"      # rubber_visc_tests_v2 (ADR 0023 §21.7)
COMMON_ARRAYS = ("cell_node", "cell_offsets", "cell_particles", "index", "work", "violations", "cfl_violations")
OPTION_NAMES = ("work_end_velocity", "work_scratch", "F64", "x64", "proj_row_node", "proj_row_comp", "proj_row_group",
                "proj_row_block", "proj_row_local", "proj_rhs", "proj_vpresc", "proj_mu", "proj_block_offset",
                "proj_block_size", "proj_inv_offset", "proj_inv", "proj_mu_node")
VISCO_NAMES = ("visco_q", "visco_s_prev", "params64", "visco_pb", "visco_branch_work", "visco_dissipation",
               "visco_d_negative", "visco_df_prev", "visco_branch_work_left")


def load_protocol():
    raw = PROTOCOL_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def prony_set(name, protocol=None):
    """(mu0, terms) of a declared Prony set."""
    cfg = protocol or load_protocol()[0]
    ps = cfg["common"]["prony_sets"][name]
    return float(ps["mu0_pa"]), tuple((float(t["g"]), float(t["tau_s"])) for t in ps["terms"])


def material(mu0, terms, rho=1150.0, beta=20.0, M=None):
    terms = pad_prony(terms, M) if M is not None else tuple(terms)
    return MaterialValues(mu0=float(mu0), kappa=beta*float(mu0), density=float(rho), prony=terms)


def strip_spec(L=0.030, W=0.006, t=0.003, n_t=3, case="C", traction_regions=(), end_mask="x", dx=None):
    extra = {"end_mask": end_mask} if case == "C" else {}
    return rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case=case, dx=dx,
                              traction_regions=tuple(traction_regions), **extra)


def block_spec(size=0.004, dx=0.001):
    return rs.RubberStripSpec(length=size, width=size, thickness=size, dx=dx, case="block")


def make_live(spec, mat, *, control_dt=2e-4, gravity=(0.0, 0.0, 0.0), energy=VISCO_ENERGY, step_material=None,
              substeps=None, options=None, strict=False, device="cpu", guard=1.0, record=False):
    from .live_rubber_material import LiveRubberMaterial
    return LiveRubberMaterial(spec, material=mat, control_dt=control_dt, gravity=tuple(gravity),
                              stable_step_guard=guard, strict=strict, step_material=step_material, substeps=substeps,
                              visco_energy=energy, device=device, record=record, **(OPTIONS if options is None
                                                                                   else options))


def same_bytes(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_env(batch, e, live):
    """Byte-wise comparison (dtype, shape, raw bytes) of batched env ``e`` with its single reference ``live``; returns
    the list of mismatching names (empty = L0). Covers every solver array of the batch snapshot (particle, node,
    group, traction, cell CSR, index, work, violations, cfl_violations, params, option arrays, visco registry state,
    params64, visco energy arrays), the segmented diagnostics, the visco diagnostics, markers and wrenches."""
    from .batched import constitutive_rubber as cr
    from .batched.rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, TRACTION_ARRAYS
    snap = batch.solver.snapshot(e)
    s = live.solver
    bad = []
    names = PARTICLE_ARRAYS+NODE_ARRAYS+GROUP_ARRAYS+TRACTION_ARRAYS+COMMON_ARRAYS
    extra = [n for n in OPTION_NAMES+VISCO_NAMES if hasattr(s, n)]
    if set(extra) != {n for n in OPTION_NAMES+VISCO_NAMES if n in snap}:
        bad.append("option/visco array set")
    for name in names+tuple(extra):
        a = getattr(s, name).numpy()
        if name in GROUP_ARRAYS:
            a = a[:s.G]
        elif name in TRACTION_ARRAYS:
            a = a[:s.Gt]
        elif name.startswith("proj_row") or name in ("proj_rhs", "proj_vpresc", "proj_mu"):
            a = a[:s.proj_rows]
        if name not in snap or not same_bytes(snap[name], a):
            bad.append(name)
    if not same_bytes(snap["params"], np.array(cr.env_parameters(s.material), np.float64).astype(np.float32)):
        bad.append("params")
    if not same_bytes(batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]):
        bad.append("diagnostics")
    vb, vs = batch.visco_summary, getattr(s, "visco_diagnostics", None)
    if (vb is None) != (vs is None) or (vb is not None and not same_bytes(vb.numpy()[e],
                                                                          vs.visco_summary.numpy()[0])):
        bad.append("visco_diagnostics")
    K = batch.K
    if not same_bytes(batch.marker_x.numpy()[e*K:(e+1)*K], s.diagnostics.marker_x.numpy()):
        bad.append("markers")
    if s.G and not same_bytes(batch.wrench.numpy()[e, :s.G], live.last_wrench):
        bad.append("wrench")
    return bad
