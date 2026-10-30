#!/usr/bin/env python3
"""WP-12 Phase 3: ``cut_aware_surface_v1`` (B) vs ``particle_surface_v1`` (A) on S0-S9 under the frozen protocol.

Protocol: ``configs/wp12/surface_protocol_v1.json`` (frozen by commit b31553d). Its SHA256, the commit that froze it,
whether it is unchanged since, the case list and every evaluation reading of this driver (``DECLARATION``) are
written to ``<out>/protocol_record.json`` before any case runs. Nothing here changes a threshold.

Runners are the Phase 1 drivers (``scripts/run_wp12_phase1_geometry.py``: 1 mm particles, 4 mm grid, dt 0.1 ms,
tlmpm_v5, law 8, ruler_knife_v1, cutting energy 5 J/m^2, cut_events_v1 + cut_event_snapshots_v1 recorder on):
S1 = G3a (registered CUT-01-R, kinematic_env, seed 42) and G3b (20 deg yawed vertical cut), S4 = G4, S5 = G6
(slots_per_node 16), S7 = G5, S8 = G7, S9 = G8 + the G4 hold phase; new in Phase 3: S2 (disarmed 2 mm flat press,
then a vertical cut through the pressed zone), S3 (vertical cut whose edge yaws 25 deg/s during the stroke: per
substep rotation with the matching angular twist, ``sequence_blade_probe.OrientedPhase.angular_velocity_rad_s``),
S6 (four parallel slices of a 120 mm block touching the tray walls, probe_v5_multicut slices plan). S0 = the G1
static box, G2 = synthetic analytic planes (geometry unit test, never physics).

Surfaces on the same states: A = CuttingSurface ``current_volume_v1`` (GPU, Newton ParticleSurface; ``legacy``
reported alongside on repeat 0), B = CutAwareSurface with the recorder Hermite log and total-F embedding
(``cell_face_v1`` boundary), B_mid = the same with the labelled R4 midpoint fallback (no log).

Stages (each writes into a new directory, never overwriting):
  run      physics + evaluation: <out>/protocol_record.json, <case>_r<k>.json, <case>_r<k>_states.npz (positions,
           codes, F and the cut_events_v1 log per evaluation frame), <case>_r<k>_package.npz (debug viewer),
           summary.json (verdicts, threshold checks, comparison gate)
  perf     <run>/perf_NNN/perf.json from the saved states (fresh process: CPU RSS, GPU mempool for A)
  figures  PNGs from the saved packages via cut_debug_view (A vs B)

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase3_surface.py run --device cuda:1 \\
      --out artifacts/wp12/phase3_surface_001
  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase3_surface.py perf --device cuda:1 \\
      --out artifacts/wp12/phase3_surface_001
  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase3_surface.py figures \\
      --out artifacts/wp12/phase3_surface_001 --media docs/media/wp12/phase3
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
_spec = importlib.util.spec_from_file_location("wp12_phase1_geometry", ROOT/"scripts/run_wp12_phase1_geometry.py")
P1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1)

S, DX, DT, CUT_SPEED, MOVE_SPEED, ENERGY = P1.S, P1.DX, P1.DT, P1.CUT_SPEED, P1.MOVE_SPEED, P1.ENERGY
PROTOCOL_FILE = ROOT/"configs/wp12/surface_protocol_v1.json"
FREEZE_COMMIT = "b31553d"
A = "current_volume_v1"
LEGACY = "legacy"
B = "cut_aware_surface_v1"
B_MID = "cut_aware_surface_v1_midpoint_fallback"
ALL_CASES = ("S0", "G2", "S1_G3a", "S1_G3b", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
S3_YAW_RATE_DEG_S = 25.

DECLARATION = {
    "schema": "wp12_phase3_surface_run_declaration_v1",
    "protocol": "configs/wp12/surface_protocol_v1.json (thresholds used exactly as frozen; never edited here)",
    "methods": {
        A: "baseline A = particle_surface_v1 = CuttingSurface current_volume_v1 (voxel s/2, kernel 3 s, threshold 0.5, "
           "radius (s/2) cbrt(det F), unmeshable skip_presentation_v1)",
        LEGACY: "CuttingSurface legacy, reported alongside on repeat 0 only, never the comparison target",
        B: "candidate B = cut_aware_surface_v1, cut_events_v1 recorder Hermite log, total-F embedding, boundary "
           "cell_face_v1, QEF truncation 0.1, strict_audit off (audit errors are recorded as failures, not raised)",
        B_MID: "B with the labelled R4 link_midpoint_fallback (no log); measured against the same Hermite locus; "
               "never presented as recorder quality; only the G2 midpoint thresholds apply to it"},
    "metric_readings": {
        "distance": "surface_metrics_v1.surface_report distance_to_side_locus_plane over ALL region_rule_v1 cut-face "
                    "vertices (P50/P95/max, spacings); core (>= 3 s from the exterior) reported alongside",
        "current_space": "surface_report against the Hermite side locus q = x_p + F_p (P - X_p) (events = the run's log)",
        "reference_space": "same region_rule_v1 cut-face vertices; B: their pre-embedding positions X_r (exact) vs the "
                           "local plane of the Hermite P of the same cut within 2 mm; A: approximate pull-back "
                           "X_p + F_p^-1 (v - x_p) via the nearest same-component particle (diagnostic, as declared)",
        "bias_thickness_normal_coverage_rim": "current space (surface_report); |bias_toward_side2| and |thickness_bias| "
                                              "in spacings; normal = core triangles vs the local locus plane",
        "closed": "mesh_validity_v1 watertight (every edge has exactly two triangles) and 0 negative-volume shells",
        "shells_equal_components": "topology_consistency with extractor provenance (component_ranges)",
        "volume_reference_all": "B: |V(pre-embedding mesh) - sum m/rho| <= 0.5 s A_cut,ref + 1e-4 V, A_cut,ref = area of "
                                "B's cut triangles in reference; A: its current mesh volume vs sum m/rho with A_cut = its "
                                "current cut-face area",
        "volume_current_all": "|V(mesh) - sum (m/rho) det F| / sum (m/rho) det F <= 0.02",
        "temporal_fixed_state": "re-extraction with the cached extractor and with a fresh one; topology hash = faces + B "
                                "reference vertices (A: same_topology of temporal_stability); residual = max |dv| (A: "
                                "vertexwise if same topology, else max symmetric residual after transport)",
        "temporal_slow_motion": "S8 snapshot frames without code change (push/hold every 40 ms): temporal_stability "
                                "residual_after_transport P95",
        "S2": "S2_deformed_cut 'as S1 in current space': the S1 current-space thresholds with current_max 2.0 s decide "
              "the verdict; the S1 reference-space thresholds are reported as a strict reading",
        "S3": "S3_curved_trajectory 'as S1 reference space': reference_p95/max decide the verdict (plus the 'all' rows); "
              "the remaining S1 thresholds are reported as a strict reading; single-plane residual reported",
        "S4": "cut faces on non-conflicting links and slit extent from B's face_link vs the independent conflicting-link "
              "set (surface_metrics_v1.conflicting_links_per_cut); coverage = surface_report coverage; tip "
              "representatives shared = vertices used by cut faces of both sides exist, all lie in regions holding both "
              "side digits, and the mesh has 0 boundary edges; A: NOT_APPLICABLE where it has no link provenance",
        "S5": "S1 thresholds per cut digit (surface_report per_cut, per-cut reference distances and per-cut rim)",
        "S6": "components >= 5, every component >= 1 shell (provenance), per-component reference volume error bound",
        "S7": "skipped fragments = A's unmeshable_skipped fragments / B's components without surface; chip = every "
              "mechanical component except the largest; chip volume by provenance vs sum (m/rho) det F",
        "S8": "frames from the cut end through hold; vertices shared across conflicting codes = B regions whose "
              "particles carry both side digits of a cut (A: 0 by construction, per-component meshes); shell set = "
              "(shell count, component of every shell) constant over the frames; contact-stat penetration is not in "
              "the snapshots: NOT_AVAILABLE, minimum particle distance reported",
        "S9": "each G8 control and the G4 hold: 0 events, 0 B cut triangles, B faces + reference vertices identical to "
              "the uncut extraction and current vertices identical to the uncut extraction at the same state; Hermite "
              "points outside the swept region = 0 (no events)",
        "G2": "B vs the analytic plane (allowed only here): all cut-face vertices, side 2 offset by the rigid opening; "
              "non-crease normal = core triangles; numeric_tolerance_m added to the distance thresholds",
        "S0": "surface vs the analytic box (area-weighted P95 over centroids + vertices, max), volume vs the box"},
    "comparison_gate": {
        "bands": "distances 0.06 s, normals 1 deg, volume 0.005 (relative), as frozen",
        "metrics_without_declared_band": "coverage, counts, booleans: a difference is decided only when it exceeds the "
                                         "larger repeat spread of A and B and has the same sign in every repeat",
        "worse": "B worse than A on a metric when the difference beyond the band holds in ANY repeat (conservative)",
        "better": "B better than A when the difference beyond the band holds in EVERY repeat",
        "excluded": "roughness (smoothness is never a success criterion; reported with mesh/locus ratio)",
        "required_better": "current-space cut-face P95 in every case where both draw a cut face; coverage in S4 and S7; "
                           "topology (shells = components and no skipped fragments) in at least one case and never worse"},
    "repeats": ">= 2 for physical cases (same seed / deterministic sampling; GPU nondeterminism measured by repeats)",
    "never": "no target plane as a reference (G2 only); no physics edits; no threshold edits",
}


# ----------------------------------------------------------------------------------------------- utilities

def jsonable(v):
    return P1.jsonable(v)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def proc_memory():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                key, value = line.split(":")
                out[key+"_bytes"] = int(value.split()[0])*1024
    return out


def topo_hash(faces, vertex_reference=None):
    h = hashlib.sha256(np.ascontiguousarray(np.asarray(faces, np.int64)).tobytes())
    if vertex_reference is not None:
        h.update(np.ascontiguousarray(np.asarray(vertex_reference, np.float64)).tobytes())
    return h.hexdigest()


def check(value, op, threshold, *, note=None):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return {"value": None, "op": op, "threshold": threshold, "verdict": "NOT_EVALUABLE", **({"note": note} if note else {})}
    ok = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}[op](value, threshold)
    return {"value": value, "op": op, "threshold": threshold, "verdict": "PASS" if ok else "FAIL", **({"note": note} if note else {})}


def not_applicable(reason):
    return {"value": None, "verdict": "NOT_APPLICABLE", "note": reason}


def verdict_of(checks):
    vs = [c["verdict"] for c in checks.values()]
    if any(v == "FAIL" for v in vs):
        return "FAIL"
    if any(v == "NOT_EVALUABLE" for v in vs):
        return "NOT_EVALUABLE"
    return "PASS" if vs else "NOT_EVALUABLE"


def q(values, scale=S):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    return SM._quantiles(values, scale)


# ----------------------------------------------------------------------------------------------- physics runners

class SnapshotEvery:
    """PoseTrail want_snapshot predicate: snapshot every ``every``-th check() call (and nothing else)."""

    def __init__(self, every):
        self.every, self.calls = int(every), 0

    def __call__(self, recorder):
        self.calls += 1
        return self.calls % self.every == 0


def cut01r_run(device, seed=42, snapshot_every=40):
    """P1.cut01r_run with state snapshots (positions, codes, F) every ``snapshot_every`` control steps."""
    import warp as wp
    from robotic_cutting_sim import cut_event_snapshots as cs
    from robotic_cutting_sim.kinematic_env import KinematicConfig, KinematicCuttingBackend
    trail = cs.PoseTrail(stride=10, want_snapshot=SnapshotEvery(snapshot_every), snapshot_F=True)
    start_t = time.perf_counter()
    with cs.installed(hook=trail) as made:
        backend = KinematicCuttingBackend(KinematicConfig("CUT-01-R", seed=seed, particle_spacing_m=S, material_dt=DT,
                                                          device=device, cutting_law=8, solver="tlmpm_v5",
                                                          cut_events="cut_events_v1"))
        backend.reset(seed=seed)
        task, material = backend.task, backend.material
        steps, armed = 0, []
        while backend.time_s+task.control_dt_s <= task.duration_s+1e-9:
            backend.step()
            armed.append(int(material.last_diagnostics.get("cut_index", -1)))
            steps += 1
        wp.synchronize_device(device)
    wall = time.perf_counter()-start_t
    rec, bed = made[0], material.bed
    st = bed.snapshot()
    log = rec.log()
    out = {"reference": material.points.astype(float), "lattice": np.asarray(material.lattice),
           "masses": material.masses.astype(float), "density": float(task.material.density_kg_m3),
           "positions": st["positions"].astype(float), "codes": st["codes"], "F": st["F"].astype(float),
           "contact_work": st["contact_work_j"], "cut_progress": st["cut_progress"], "nodes": bed.nodes.numpy(),
           "weights": bed.weights.numpy(), "gradients": bed.gradients.numpy(), "cuts": 1, "log": log,
           "activation": rec.activation(), "metadata": rec.metadata(), "assign": rec.assignment_positions(log),
           "trail": trail, "half_length": float(bed.config.blade_half_width),
           "blade_polygon": np.asarray(bed.blade_profile_xz), "phase_substeps": [int(rec.substep.numpy()[0])],
           "labelled_after_phase": [int(np.count_nonzero(st["codes"]))],
           "digit_after_phase": [[int(np.count_nonzero(st["codes"]))]], "wall_s": wall,
           "runner": "kinematic_env.KinematicCuttingBackend (CUT-01-R scripted edge)", "seed": seed,
           "control_steps": steps, "armed_control_steps": int(np.sum(np.asarray(armed) >= 0)),
           "slot_diagnostics": bed.check_slots()}
    backend.close()
    return out


def sequence_host_trail(start, phases, stride=10, chunk=200):
    """Host poses of drive_blade_sequence phases (incl. rotating phases) via sequence_blade_probe.phase_trajectory."""
    from robotic_cutting_sim.sequence_blade_probe import phase_trajectory
    position = np.asarray(start, float)
    poses, cuts = [], []
    for phase in phases:
        steps = int(round(phase.duration_s/DT))
        rows = []
        for first in range(0, steps, chunk):
            count = min(chunk, steps-first)
            rows.append(phase_trajectory(position, phase, DT, first, count)[0].astype(float))
            position = position+np.asarray(phase.velocity_m_s, float)*DT*count
        chain = np.vstack(rows)
        keep = np.unique(np.r_[np.arange(0, len(chain), stride), len(chain)-1])
        poses.append(chain[keep].astype(np.float32))
        cuts.append(np.full(len(keep), phase.cut, np.int8))
    return poses, cuts


def probe(points, masses, phases, *, rotation=None, start, cuts=1, device, sequence=False, snapshot=None,
          slots_per_node=None):
    run = P1.probe_run(points, masses, phases, rotation=rotation, start=start, cuts=cuts, device=device,
                       sequence=sequence, snapshot=snapshot, slots_per_node=slots_per_node)
    if sequence:   # the Phase 1 host trail assumes a constant rotation per phase; rebuild it (rotating phases)
        run["trail"].poses, run["trail"].cuts = sequence_host_trail(start, phases)
    return run


def frames_of(run, names=None):
    """Evaluation frames: the snapshots (log restricted to events before the snapshot substep) and the final state."""
    frames = []
    snaps = run["trail"].snapshots if run.get("trail") is not None else []
    for i, snap in enumerate(snaps):
        if "F" not in snap:
            continue
        sub = int(snap["substep"])
        frames.append({"name": names[i] if names and i < len(names) else f"substep_{sub}", "substep": sub,
                       "positions": snap["positions"].astype(float), "codes": snap["codes"],
                       "F": snap["F"].astype(float), "log": run["log"][run["log"]["substep"] < sub]})
    total = int(np.sum(run["phase_substeps"]))
    frames.append({"name": "final", "substep": total, "positions": run["positions"], "codes": run["codes"],
                   "F": run["F"], "log": run["log"]})
    return frames


def case_s1a(args, r):
    run = cut01r_run(args.device, seed=42)
    return run, {"declared_plane": ((0, 0, 0), (1, 0, 0))}


def case_s1b(args, r):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.08, .03, top), spacing=S)
    rot = Rotation.from_euler("z", 20., degrees=True)*VERTICAL_CUT
    depth = top+.008
    phases = [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1)]
    run = probe(points, masses, phases, rotation=rot, start=[.0003, .0002, top+.004], device=args.device)
    run["phase_names"] = ["cut", "lift"]
    return run, {"yaw_deg": 20.}


def case_s2(args, r):
    """Disarmed flat press: bottom face 2 mm into the top over x in [-28, 8.6] mm (FLAT_SKIM, edge at x = 12 mm),
    hold, lift, move, reorient (clear of the material), vertical cut at x = -9.7 mm through the pressed zone, lift."""
    from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, block
    from robotic_cutting_sim.sequence_blade_probe import OrientedPhase as P
    top = .012
    points, masses = block(size_m=(.06, .024, top), spacing=S)
    flat, vert = tuple(FLAT_SKIM.as_quat()), tuple(VERTICAL_CUT.as_quat())
    x_edge, x_cut = .012, -.0097
    half_thickness = .00075                   # ruler_knife_v1 flat face at edge z -/+ 0.75 mm
    z0, z_press = top+.002, top-.002+half_thickness
    z_clear = top+.004
    depth = top+.008
    phases = [P((0, 0, -.01), (z0-z_press)/.01, -1, flat), P((0, 0, 0), .2, -1, flat),
              P((0, 0, .025), (z_clear-z_press)/.025, -1, flat), P(((x_cut-x_edge)/.5, 0, 0), .5, -1, flat),
              P((0, 0, 0), .05, -1, vert), P((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0, vert),
              P((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1, vert)]
    run = probe(points, masses, phases, start=[x_edge, 0, z0], device=args.device, sequence=True)
    run["phase_names"] = ["press", "hold", "lift_press", "move", "reorient", "cut", "lift"]
    return run, {"press": {"orientation": "FLAT_SKIM", "edge_x_m": x_edge, "flat_face_x_range_m": [x_edge-.04, x_edge-.00338],
                           "flat_face_depth_below_top_m": .002, "armed": False},
                 "cut_x_m": x_cut, "declared_plane": ((x_cut, 0, 0), (1, 0, 0))}


def case_s3(args, r):
    """Vertical cut whose edge yaws about world z at 25 deg/s during the 1 s stroke (-12.5 -> +12.5 deg; about 15 deg
    while the edge is inside the 12 mm block), per-substep rotation with the matching angular twist."""
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, block
    from robotic_cutting_sim.sequence_blade_probe import OrientedPhase as P
    top = .012
    points, masses = block(size_m=(.06, .024, top), spacing=S)
    depth = top+.008
    rate = np.radians(S3_YAW_RATE_DEG_S)
    r0 = Rotation.from_euler("z", -S3_YAW_RATE_DEG_S*depth/CUT_SPEED/2, degrees=True)*VERTICAL_CUT
    cut = P((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0, tuple(r0.as_quat()), (0., 0., float(rate)))
    phases = [cut, P((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1, cut.end_rotation_xyzw(DT))]
    run = probe(points, masses, phases, start=[.0003, 0, top+.004], device=args.device, sequence=True)
    run["phase_names"] = ["cut_yawing", "lift"]
    total = np.degrees(rate*depth/CUT_SPEED)
    inside = np.degrees(rate*top/CUT_SPEED)
    return run, {"yaw_rate_deg_s": S3_YAW_RATE_DEG_S, "yaw_total_stroke_deg": total, "yaw_inside_block_deg": inside,
                 "max_off_plane_motion_angle_at_block_edge_deg": float(np.degrees(np.arctan(rate*.012/CUT_SPEED)))}


def case_s4(args, r):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.06, .024, top), spacing=S)
    phases = [Phase((0, 0, -CUT_SPEED), .010/CUT_SPEED, 0), Phase((0, 0, 0), .2, 0), Phase((0, 0, MOVE_SPEED), .4, -1)]
    steps = np.cumsum([int(round(p.duration_s/DT)) for p in phases])
    want = lambda rec: int(rec.substep.numpy()[0]) in (int(steps[0]), int(steps[1]))  # noqa: E731
    run = probe(points, masses, phases, rotation=VERTICAL_CUT, start=[.0003, 0, top+.004], device=args.device,
                snapshot=want)
    run["phase_names"] = ["cut_half_depth", "hold_armed", "withdraw"]
    run["frame_names"] = ["cut_end", "hold_end"]
    return run, {"edge_depth_below_top_m": .006, "declared_plane": ((.0003, 0, 0), (1, 0, 0))}


def case_s5(args, r):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, block
    from robotic_cutting_sim.sequence_blade_probe import OrientedPhase as P
    top = .012
    points, masses = block(size_m=(.06, .04, top), spacing=S)
    r0 = tuple(VERTICAL_CUT.as_quat())
    r1 = tuple((Rotation.from_euler("z", 90, degrees=True)*VERTICAL_CUT).as_quat())
    depth = top+.008
    down, up = depth/CUT_SPEED, depth/MOVE_SPEED
    phases = [P((0, 0, -CUT_SPEED), down, 0, r0), P((0, 0, MOVE_SPEED), up, -1, r0), P((.024/.5, 0, 0), .5, -1, r0),
              P((0, 0, -CUT_SPEED), down, 1, r0), P((0, 0, MOVE_SPEED), up, -1, r0), P((-.012/.5, .0043/.5, 0), .5, -1, r0),
              P((0, 0, 0), .05, -1, r1), P((0, 0, -CUT_SPEED), down, 2, r1), P((0, 0, MOVE_SPEED), up, -1, r1)]
    run = probe(points, masses, phases, start=[-.0117, 0, top+.004], cuts=3, device=args.device, sequence=True,
                slots_per_node=16)
    run["phase_names"] = ["cut0", "lift0", "move0", "cut1", "lift1", "move1", "reorient", "cut2", "lift2"]
    return run, {"slots_per_node": 16}


def case_s6(args, r):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    top = .012
    xs = (-.0357, -.0117, .0123, .0363)    # 24 mm pitch, 0.3 mm off cell faces (as the other probes)
    points, masses = block(size_m=(.12, .024, top), spacing=S)   # end faces at the legacy tray walls (x = +-60 mm)
    depth = top+.008
    phases, names = [], []
    for k, x in enumerate(xs):
        phases += [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, k), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1)]
        names += [f"cut_{k}", f"lift_{k}"]
        if k+1 < len(xs):
            phases.append(Phase(((xs[k+1]-x)/.5, 0, 0), .5, -1))
            names.append(f"move_{k}")
    steps = np.cumsum([int(round(p.duration_s/DT)) for p in phases])
    lift_end = {int(steps[i]) for i, n in enumerate(names) if n.startswith("lift_")}
    want = lambda rec: int(rec.substep.numpy()[0]) in lift_end  # noqa: E731
    run = probe(points, masses, phases, rotation=VERTICAL_CUT, start=[xs[0], 0, top+.004], cuts=4, device=args.device,
                snapshot=want)
    run["phase_names"] = names
    run["frame_names"] = [f"after_cut_{k}" for k in range(len(xs))]
    return run, {"slice_x_m": list(xs), "block_m": [.12, .024, top], "walls": "legacy tray walls at x = +-60 mm"}


def case_s7(args, r):
    from robotic_cutting_sim.blade_probe import Phase
    top, dome_h, dome_r, edge_above = .010, .004, .009, .002
    grid = np.mgrid[-.04:.04-1e-9:S, -.012:.012-1e-9:S, 0:top+dome_h-1e-9:S]
    pts = grid.reshape(3, -1).T+S/2
    dome = (pts[:, 2] > top) & ((pts[:, 2]-top) <= dome_h*(1-(pts[:, 0]**2+pts[:, 1]**2)/dome_r**2))
    keep = (pts[:, 2] < top) | dome
    points = pts[keep].astype(np.float32)
    masses = np.full(len(points), 1500*S**3, np.float32)
    rot = Rotation.from_euler("y", 25., degrees=True)
    x0, x1 = -.018, .026
    phases = [Phase((CUT_SPEED, 0, 0), (x1-x0)/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), .4, -1)]
    run = probe(points, masses, phases, rotation=rot, start=[x0, 0, top+edge_above], device=args.device)
    run["phase_names"] = ["skim", "lift"]
    return run, {"edge_above_substrate_m": edge_above, "dome_height_m": dome_h, "rake_deg": 25.}


def case_s8(args, r):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.06, .024, top), spacing=S)
    depth = top+.008
    phases = [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1),
              Phase((.033/.5, 0, 0), .5, -1), Phase((0, 0, -CUT_SPEED), .5, -1),
              Phase((-.010, 0, 0), .6, -1), Phase((0, 0, 0), .2, -1), Phase((CUT_SPEED, 0, 0), .4, -1),
              Phase((0, 0, MOVE_SPEED), .4, -1)]
    steps = np.cumsum([int(round(p.duration_s/DT)) for p in phases])
    push_start, hold_end = steps[3], steps[5]

    def want(rec):
        sub = int(rec.substep.numpy()[0])
        return sub == steps[0] or sub == steps[3] or (push_start < sub <= hold_end and (sub-push_start) % 400 == 0)
    run = probe(points, masses, phases, rotation=VERTICAL_CUT, start=[.0003, 0, top+.004], device=args.device, snapshot=want)
    run["phase_names"] = ["cut", "lift", "move", "lower", "push", "hold", "withdraw", "lift_end"]
    return run, {"push_speed_m_s": .010, "push_travel_m": .006, "push_start_substep": int(push_start),
                 "hold_end_substep": int(hold_end), "declared_plane": ((.0003, 0, 0), (1, 0, 0))}


S9_CONTROLS = ("approach_armed", "flat_press_armed", "side_push_armed", "slide_disarmed", "plunge_disarmed")


def case_s9_control(args, name):
    from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.04, .024, top), spacing=S)
    controls = {
        "approach_armed": (VERTICAL_CUT, [0, 0, top+.010], [Phase((0, 0, -CUT_SPEED), .45, 0), Phase((0, 0, 0), .2, 0)]),
        "flat_press_armed": (FLAT_SKIM, [0, 0, top+.002], [Phase((0, 0, -.01), .4, 0), Phase((0, 0, 0), .2, 0),
                                                           Phase((0, 0, .01), .2, 0)]),
        "side_push_armed": (VERTICAL_CUT, [.025, 0, .006], [Phase((-.01, 0, 0), .7, 0), Phase((0, 0, 0), .1, 0)]),
        "slide_disarmed": (FLAT_SKIM, [-.03, 0, top-.0005], [Phase((CUT_SPEED, 0, 0), 1.25, -1)]),
        "plunge_disarmed": (VERTICAL_CUT, [.0003, 0, top+.004], [Phase((0, 0, -CUT_SPEED), .5, -1), Phase((0, 0, MOVE_SPEED), .4, -1)]),
    }
    rot, start, phases = controls[name]
    return probe(points, masses, phases, rotation=rot, start=start, device=args.device)


RUNNERS = {"S1_G3a": case_s1a, "S1_G3b": case_s1b, "S2": case_s2, "S3": case_s3, "S4": case_s4, "S5": case_s5,
           "S6": case_s6, "S7": case_s7, "S8": case_s8}


# ----------------------------------------------------------------------------------------------- surfaces

def new_b(run, log):
    from robotic_cutting_sim import cut_aware_surface as CAS
    from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition
    part = CurrentMechanicalPartition(run["nodes"], run["weights"], run["gradients"], run["cuts"], encoding="packed2")
    return CAS.CutAwareSurface(part, S, run["lattice"], run["reference"], cut_events_log=log,
                               masses_kg=run["masses"], density_kg_m3=run["density"], strict_audit=False)


def b_extract(surf, st, log):
    t = time.perf_counter()
    v, f, info = surf.extract(st["positions"], st["codes"], st["F"], cut_events_log=log)
    wall = time.perf_counter()-t
    last = {k: (np.array(x, copy=True) if isinstance(x, np.ndarray) else x) for k, x in surf.last.items()}
    return {"vertices": v, "faces": f.astype(np.int64), "info": info, "last": last, "wall_s": wall, "surface": surf}


def compact_b_info(info):
    keep = ("mechanical_components", "hermite", "qef", "regions", "faces", "embedding", "update", "timings_s",
            "boundary_hermite_profile")
    out = {k: info[k] for k in keep}
    a = info["audit"]
    out["audit"] = {k: a[k] for k in ("mechanical_components", "lattice_pieces", "mesh_shells", "piece_component_violations",
                                      "shell_component_violations", "errors", "ok")}
    out["audit"]["visually_separated_mechanically_bridged"] = len(a["visually_separated_mechanically_bridged"])
    out["audit"]["components"] = (a["components"] or [])[:12]
    mv = info["mesh_validity_reference"]
    out["mesh_validity_reference"] = mv
    return out


def provenance_volumes(vertices, faces, face_component, count):
    tri = np.asarray(vertices, float)[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))/6
    ok = face_component >= 0
    return np.bincount(face_component[ok], weights=signed[ok], minlength=count)


def reference_space(st, labels, locus, cls, v, f, Xref):
    """Reference-space cut-face distances (see DECLARATION); for B also coverage of P by its reference mesh."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    region = np.flatnonzero(cls["kind"] == 2)
    if not len(locus["cut"]) or not len(region):
        return {"status": "NO_CUT_FACE"}
    rows = cls["row"][region]
    if Xref is None:
        Xv = SM.pull_back(v[region], st["positions"], st["reference"], st["F"], labels, cls["component"][region])
        source = "A: approximate pull-back X_p + F_p^-1 (v - x_p), nearest same-component particle (diagnostic)"
    else:
        Xv = np.asarray(Xref, float)[region]
        source = "B: pre-embedding representative positions X_r (exact)"
    d = np.full(len(region), np.nan)
    for k in np.unique(locus["cut"]):
        vm = locus["cut"][rows] == k
        pm = locus["cut"] == k
        if vm.any():
            d[vm], _ = SM.local_plane_signed(Xv[vm], locus["P"][pm], locus["N"][pm], .002)
    core = cls["core"][region]
    out = {"source": source, "status": "MEASURED", "distance_to_locus_plane": q(np.abs(d)),
           "distance_to_locus_plane_core": q(np.abs(d[core])), "plane_coverage": float(np.isfinite(d).mean()),
           "per_cut": {str(int(k)): {"all": q(np.abs(d[locus["cut"][rows] == k])),
                                     "core": q(np.abs(d[(locus["cut"][rows] == k) & core]))} for k in np.unique(locus["cut"])}}
    if Xref is not None:
        rev = SM.point_mesh_distance(locus["P"], np.asarray(Xref, float), f)
        out["locus_to_mesh_distance"] = q(rev)
        out["coverage"] = float((rev <= S).mean())
    # single-plane residual per cut (S3: must not be forced to 0)
    sp = {}
    for k in np.unique(locus["cut"]):
        P = locus["P"][locus["cut"] == k]
        if len(P) < 4:
            continue
        c = P.mean(0)
        n = np.linalg.svd(P-c)[2][-1]
        row = {"locus_P_residual": q(np.abs((P-c)@n)), "plane_normal": n.tolist()}
        vm = locus["cut"][rows] == k
        if vm.any():
            row["mesh_reference_vertex_residual"] = q(np.abs((Xv[vm]-c)@n))
        sp[str(int(k))] = row
    out["single_plane_residual"] = sp
    return out


def per_cut_rim(locus, st, v, f):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    out = {}
    for k in np.unique(locus["cut"]):
        m = locus["cut"] == k
        sub = {"link_direction": locus["link_direction"][m], "pairs": locus["pairs"][m], "P": locus["P"][m]}
        out[str(int(k))] = SM.rim_continuity(sub, st["lattice"], st["reference"], st["positions"], st["F"], S, v, f)
    return out


def scalars(rep, ref, temporal=None):
    """Flat comparison scalars of one mesh report (lengths in spacings)."""
    def g(d, *keys):
        for key in keys:
            if d is None:
                return None
            d = d.get(key) if isinstance(d, dict) else None
        return d
    sb = rep.get("signed_bias") or {}
    mv = rep.get("mesh_validity") or {}
    topo = rep.get("topology") or {}
    vol = rep.get("volume") or {}
    out = {"current_p50_s": g(rep, "distance_to_side_locus_plane", "p50_spacings"),
           "current_p95_s": g(rep, "distance_to_side_locus_plane", "p95_spacings"),
           "current_max_s": g(rep, "distance_to_side_locus_plane", "max_spacings"),
           "current_core_p95_s": g(rep, "distance_to_side_locus_plane_core", "p95_spacings"),
           "reference_p95_s": g(ref, "distance_to_locus_plane", "p95_spacings"),
           "reference_max_s": g(ref, "distance_to_locus_plane", "max_spacings"),
           "locus_to_mesh_p95_s": g(rep, "locus_to_mesh_distance", "p95_spacings"),
           "coverage": rep.get("coverage"),
           "abs_bias_toward_side2_s": None if sb.get("bias_toward_side2_m") is None else abs(sb["bias_toward_side2_m"])/S,
           "abs_thickness_bias_s": None if sb.get("thickness_bias_m") is None else abs(sb["thickness_bias_m"])/S,
           "normal_p50_deg": g(rep, "normal_error_deg_core", "p50"), "normal_p95_deg": g(rep, "normal_error_deg_core", "p95"),
           "rim_corner_p95_s": g(rep, "rim", "corner_to_mesh_distance", "p95_spacings"),
           "rim_corner_max_s": g(rep, "rim", "corner_to_mesh_distance", "max_spacings"),
           "open_edges_near_junction": g(rep, "rim", "open_edges_near_junction"),
           "boundary_edges": mv.get("boundary_edges"), "non_manifold_edges": mv.get("non_manifold_edges"),
           "inconsistently_oriented_edges": mv.get("inconsistently_oriented_edges"),
           "degenerate_triangles": mv.get("degenerate_triangles"), "shells_negative_volume": mv.get("shells_negative_volume"),
           "self_intersections_sampled": g(mv, "self_intersection", "intersecting_pairs_sampled"),
           "closed": bool(mv.get("watertight")) and mv.get("shells_negative_volume", 1) == 0 if mv else None,
           "shells_equal_components": topo.get("shells_equal_components"),
           "shells_spanning_components": topo.get("shells_spanning_components"),
           "components_without_shell": len(topo.get("components_without_shell") or []) if topo else None,
           "faces_nearer_other_component": topo.get("faces_nearer_other_component"),
           "volume_current_abs_rel": None if vol.get("total_relative_error_current") is None else abs(vol["total_relative_error_current"]),
           "roughness_mesh_core_qm_s": None if g(rep, "roughness_mesh_core", "quadratic_mean") is None else g(rep, "roughness_mesh_core", "quadratic_mean")/S,
           "roughness_locus_core_qm_s": None if g(rep, "roughness_locus_core", "quadratic_mean") is None else g(rep, "roughness_locus_core", "quadratic_mean")/S}
    if temporal:
        out["fixed_state_topology_stable"] = temporal.get("topology_stable")
        out["fixed_state_residual_s"] = temporal.get("residual_max_s")
    return out


def volume_checks(method, rep, st, labels, mesh, ref):
    """'all' volume rows: reference and current (see DECLARATION)."""
    vref_mat = float((st["masses"]/st["density"]).sum())
    J = np.linalg.det(st["F"])
    vcur_mat = float((st["masses"]/st["density"]*J).sum())
    v, f = mesh["vertices"], mesh["faces"]
    tri = v[f]
    vcur = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()/6) if len(f) else 0.
    out = {"material_reference_m3": vref_mat, "material_current_m3": vcur_mat, "mesh_current_m3": vcur,
           "current_rel_error": (vcur-vcur_mat)/vcur_mat}
    if method in (B, B_MID):
        X = mesh["last"]["vertex_reference"]
        tr = X[f]
        vrm = float(np.einsum("ij,ij->i", tr[:, 0], np.cross(tr[:, 1], tr[:, 2])).sum()/6)
        from robotic_cutting_sim.cut_aware_surface import LINK_CUT
        area = .5*np.linalg.norm(np.cross(tr[:, 1]-tr[:, 0], tr[:, 2]-tr[:, 0]), axis=1)
        a_cut = float(area[mesh["last"]["face_kind"] == LINK_CUT].sum())/2   # both sides drawn: one cut area
        out.update(mesh_reference_m3=vrm, cut_area_reference_m2=a_cut, reference_abs_error_m3=abs(vrm-vref_mat),
                   reference_bound_m3=.5*S*a_cut+1e-4*vref_mat, reference_source="B pre-embedding mesh")
    else:
        a_cut = float(rep.get("cut_face_area_m2") or 0.)/2
        out.update(mesh_reference_m3=vcur, cut_area_reference_m2=a_cut, reference_abs_error_m3=abs(vcur-vref_mat),
                   reference_bound_m3=.5*S*a_cut+1e-4*vref_mat,
                   reference_source="A has no reference mesh: current mesh volume vs sum m/rho; A_cut = current cut-face area / 2")
    out["reference_rel_error"] = out["reference_abs_error_m3"]/vref_mat
    return out


def fixed_state_temporal(method, mesh, st, args, run, log):
    """Re-extraction at the same state: cached extractor and a fresh extractor."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    v0, f0 = mesh["vertices"], mesh["faces"]
    rows = {}
    if method in (B, B_MID):
        ref0 = mesh["last"]["vertex_reference"]
        h0 = topo_hash(f0, ref0)
        again = b_extract(mesh["surface"], st, log)
        fresh = b_extract(new_b(run, log), st, log)
        for name, m in (("cached_again", again), ("fresh_extractor", fresh)):
            same = m["faces"].shape == f0.shape and topo_hash(m["faces"], m["last"]["vertex_reference"]) == h0
            res = float(np.abs(m["vertices"]-v0).max()) if same and len(v0) else None
            rows[name] = {"topology_hash_equal": bool(same), "vertex_residual_max_m": res,
                          "update": m["info"]["update"], "wall_s": m["wall_s"]}
        stable = all(r["topology_hash_equal"] for r in rows.values())
        residual = max((r["vertex_residual_max_m"] or 0.) for r in rows.values()) if stable else None
        return {"rows": rows, "topology_stable": stable, "residual_max_s": None if residual is None else residual/S,
                "topology_hash": h0}
    surf = mesh["surface"]
    again = surf.extract(st["positions"], st["codes"], st["F"] if method == A else None)
    fresh = P1.extract_surfaces(st, args.device, profiles=(method,))[method]
    for name, (v, f) in (("cached_again", (again[0], again[1])), ("fresh_extractor", (fresh["vertices"], fresh["faces"]))):
        ts = SM.temporal_stability(v0, f0, np.asarray(v, float), np.asarray(f, np.int64), st["positions"], st["positions"], spacing=S)
        rows[name] = {"same_topology": ts["same_topology"], "vertexwise_max_m": ts.get("vertexwise_max_m"),
                      "residual_after_transport_max_m": ts["residual_after_transport"]["max"]}
    stable = all(r["same_topology"] for r in rows.values())
    residual = max((r["vertexwise_max_m"] if r["same_topology"] else r["residual_after_transport_max_m"]) or 0.
                   for r in rows.values())
    return {"rows": rows, "topology_stable": stable, "residual_max_s": residual/S}


def evaluate_frame(run, fr, args, *, legacy=False, temporal=True, light=False, b_surfaces=None):
    """All surfaces on one state; full surface_metrics_v1 reports unless ``light`` (topology/intersections only)."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    st = dict(run, positions=np.asarray(fr["positions"], float), codes=fr["codes"], F=np.asarray(fr["F"], float),
              log=fr["log"])
    st.pop("_rank", None)
    count, labels, lat_count, fractions = P1.components(st)
    rank = st["_rank"]
    locus = SM.material_locus(st["reference"], st["lattice"], st["positions"], st["codes"], cuts=st["cuts"], F=st["F"],
                              events=st["log"])
    out = {"frame": fr["name"], "substep": int(fr["substep"]), "mechanical_components": int(count),
           "lattice_components": int(lat_count), "mass_fractions": fractions[:10].tolist(),
           "conflicting_link_rows": int(len(locus["cut"])), "locus_rows_by_source": locus["rows_by_source"],
           "events_in_log": int(len(st["log"])), "codes_sha256": hashlib.sha256(np.asarray(st["codes"]).tobytes()).hexdigest()}
    meshes = {}
    profiles = (A, LEGACY) if legacy else (A,)
    for p, m in P1.extract_surfaces(st, args.device, profiles=profiles).items():
        meshes[p] = m
    b_surfaces = {} if b_surfaces is None else b_surfaces
    for name, log in ((B, st["log"]), (B_MID, None)):
        if light and name == B_MID:
            continue
        if name not in b_surfaces:
            b_surfaces[name] = new_b(run, log)
        m = b_extract(b_surfaces[name], st, log)
        m["face_component"] = P1.face_components(m["info"], rank, len(m["faces"]))
        meshes[name] = m
    ext = SM.exterior_samples(st["lattice"], st["positions"], S, st["F"])
    reports = {}
    for name, m in meshes.items():
        v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
        fc = m.get("face_component")
        if light:
            topo = SM.topology_consistency(v, f, st["positions"], labels, face_component=fc)
            sh = SM.shells(v, f)[1] if len(f) else None
            inter = SM.self_intersections(v, f, shell=sh)
            shell_component = sorted(int(r["component"]) for r in topo["shell_rows"])
            row = {"vertices": int(len(v)), "triangles": int(len(f)), "mesh_shells": topo["mesh_shells"],
                   "shell_components": shell_component, "shells_spanning_components": topo["shells_spanning_components"],
                   "faces_nearer_other_component": topo["faces_nearer_other_component"],
                   "inter_shell_intersections_sampled": inter["inter_shell_pairs_sampled"],
                   "self_intersections_sampled": inter["intersecting_pairs_sampled"], "extract_wall_s": m["wall_s"]}
            if name in (B, B_MID):
                row["topology_hash"] = topo_hash(f, m["last"]["vertex_reference"])
                row["mixed_side_regions"] = m["info"]["regions"]["mixed_side_regions"]
                row["cut_triangles"] = m["info"]["faces"]["cut_triangles"]
                row["audit_ok"] = m["info"]["audit"]["ok"]
            reports[name] = row
            continue
        rep = SM.surface_report(st["reference"], st["lattice"], st["positions"], st["codes"], v, f, spacing=S, labels=labels,
                                masses=st["masses"], density=st["density"], cuts=st["cuts"], F=st["F"], locus=locus,
                                face_component=fc, reference_space=False)
        cls = SM.classify_mesh_vertices(v, st["positions"], labels, locus, ext, spacing=S)
        ref = reference_space(st, labels, locus, cls, v, f, m["last"]["vertex_reference"] if name in (B, B_MID) else None)
        rep["reference_space"] = ref
        if st["cuts"] > 1 and len(locus["cut"]):
            rep["rim_per_cut"] = per_cut_rim(locus, st, v, f)
        vols = provenance_volumes(v, f, fc, count) if fc is not None else None
        J = np.linalg.det(st["F"])
        mat_cur = np.bincount(labels, weights=st["masses"]/st["density"]*J, minlength=count)
        mat_ref = np.bincount(labels, weights=st["masses"]/st["density"], minlength=count)
        rep["component_volumes_provenance"] = {"mesh_current_m3": None if vols is None else vols.tolist()[:64],
                                               "material_current_m3": mat_cur.tolist()[:64],
                                               "material_reference_m3": mat_ref.tolist()[:64]}
        if name in (B, B_MID):
            from robotic_cutting_sim.cut_aware_surface import LINK_CUT
            X = m["last"]["vertex_reference"]
            vr = provenance_volumes(X, f, fc, count)
            tr = X[f]
            area = .5*np.linalg.norm(np.cross(tr[:, 1]-tr[:, 0], tr[:, 2]-tr[:, 0]), axis=1)
            cut = m["last"]["face_kind"] == LINK_CUT
            a_cut = np.bincount(fc[cut], weights=area[cut], minlength=count) if cut.any() else np.zeros(count)
            rep["component_volumes_provenance"].update(mesh_reference_m3=vr.tolist()[:64],
                                                       cut_area_reference_m2=a_cut.tolist()[:64])
            rep["cut_aware_info"] = compact_b_info(m["info"])
        else:
            rep["unmeshable_skipped"] = m.get("skipped")
        rep["extract_wall_s"] = m["wall_s"]
        rep["volume_checks"] = volume_checks(name, rep, st, labels, m, ref)
        tmp = fixed_state_temporal(name, m, st, args, run, st["log"] if name == B else None) if temporal and name != LEGACY else None
        rep["temporal_fixed_state"] = tmp
        rep["scalars"] = scalars(rep, ref, tmp)
        rep["scalars"]["volume_reference_abs_rel"] = abs(rep["volume_checks"]["reference_rel_error"])
        reports[name] = rep
    out["reports"] = reports
    return out, meshes, labels, locus, st


def save_states(path, run, frames):
    """Per evaluation frame: positions, codes, F (float32), the frame's log; plus the static arrays."""
    from robotic_cutting_sim import cut_aware_surface as CAS
    nodes, weights, gradients = CAS.reference_stencil(np.asarray(run["reference"], np.float32), DX)
    recomputable = bool(np.array_equal(nodes, run["nodes"]) and np.array_equal(weights, run["weights"])
                        and np.array_equal(gradients, run["gradients"]))
    extra = {} if recomputable else {"nodes": run["nodes"], "weights": run["weights"], "gradients": run["gradients"]}
    np.savez_compressed(path, schema="wp12_phase3_states_v1", spacing_m=S, grid_dx_m=DX, cuts=run["cuts"],
                        reference=np.asarray(run["reference"], np.float32), lattice=np.asarray(run["lattice"], np.int32),
                        masses=np.asarray(run["masses"], np.float32), density=run["density"], log=run["log"],
                        frame_names=np.array([f["name"] for f in frames]), frame_substeps=np.array([f["substep"] for f in frames]),
                        positions=np.stack([np.asarray(f["positions"], np.float32) for f in frames]),
                        codes=np.stack([np.asarray(f["codes"]) for f in frames]),
                        F=np.stack([np.asarray(f["F"], np.float32) for f in frames]),
                        stencil_recomputable_from_reference=recomputable, **extra)
    return recomputable


def save_package(path, run, labels, meshes, locus, st):
    pm = {k: {"vertices": m["vertices"], "faces": m["faces"]} for k, m in meshes.items()}
    P1.save_package(path, dict(run, positions=st["positions"], codes=st["codes"], F=st["F"]), labels, pm, locus)


# ----------------------------------------------------------------------------------------------- thresholds

def protocol_thresholds():
    return json.loads(PROTOCOL_FILE.read_text())["thresholds_for_candidate_B"]


def s1_checks(sc, T, *, current_max=None, rows=None):
    t = T["S1_flat_cut"]
    allrows = {"reference_p95_s": check(sc["reference_p95_s"], "<=", t["reference_p95_s"]),
               "reference_max_s": check(sc["reference_max_s"], "<=", t["reference_max_s"]),
               "current_p95_s": check(sc["current_p95_s"], "<=", t["current_p95_s"]),
               "current_max_s": check(sc["current_max_s"], "<=", t["current_max_s"] if current_max is None else current_max),
               "abs_bias_toward_side2_s": check(sc["abs_bias_toward_side2_s"], "<=", t["abs_bias_toward_side2_s"]),
               "abs_thickness_bias_s": check(sc["abs_thickness_bias_s"], "<=", t["abs_thickness_bias_s"]),
               "normal_p95_deg": check(sc["normal_p95_deg"], "<=", t["normal_p95_deg"]),
               "coverage_min": check(sc["coverage"], ">=", t["coverage_min"]),
               "rim_corner_p95_s": check(sc["rim_corner_p95_s"], "<=", t["rim_corner_p95_s"]),
               "closed": check(sc.get("closed"), "==", True),
               "shells_equal_components": check(sc.get("shells_equal_components"), "==", True)}
    return allrows if rows is None else {k: v for k, v in allrows.items() if k in rows}


def all_checks(rep, T):
    t = T["all"]
    vc = rep["volume_checks"]
    tmp = rep.get("temporal_fixed_state") or {}
    out = {"volume_reference": check(vc["reference_abs_error_m3"], "<=", vc["reference_bound_m3"],
                                     note=f"bound 0.5 s A_cut + 1e-4 V; {vc['reference_source']}"),
           "volume_current_rel": check(abs(vc["current_rel_error"]), "<=", t["volume_current_rel"]),
           "temporal_fixed_state_topology": check(tmp.get("topology_stable"), "==", True),
           "temporal_fixed_state_residual_s": check(tmp.get("residual_max_s"), "<=", 1e-3)}
    return out


def s1_variant_rows(case):
    current_rows = ("current_p95_s", "current_max_s", "abs_bias_toward_side2_s", "abs_thickness_bias_s", "normal_p95_deg",
                    "coverage_min", "rim_corner_p95_s", "closed", "shells_equal_components")
    if case == "S2":
        return current_rows
    if case == "S3":
        return ("reference_p95_s", "reference_max_s")
    return None


def per_cut_scalars(rep, k):
    pc = rep.get("per_cut", {}).get(k, {})
    ref = (rep.get("reference_space") or {}).get("per_cut", {}).get(k, {})
    rim = (rep.get("rim_per_cut") or {}).get(k, {})
    sb = pc.get("signed_bias") or {}
    gp = lambda d, key: None if not d or d.get("count", 0) == 0 else d.get(key)  # noqa: E731
    return {"current_p95_s": gp(pc.get("distance_to_side_locus_plane"), "p95_spacings"),
            "current_max_s": gp(pc.get("distance_to_side_locus_plane"), "max_spacings"),
            "reference_p95_s": gp(ref.get("all"), "p95_spacings"), "reference_max_s": gp(ref.get("all"), "max_spacings"),
            "abs_bias_toward_side2_s": None if sb.get("bias_toward_side2_m") is None else abs(sb["bias_toward_side2_m"])/S,
            "abs_thickness_bias_s": None if sb.get("thickness_bias_m") is None else abs(sb["thickness_bias_m"])/S,
            "normal_p95_deg": gp(pc.get("normal_error_deg_core"), "p95"), "coverage": pc.get("coverage"),
            "rim_corner_p95_s": gp(rim.get("corner_to_mesh_distance"), "p95_spacings"),
            "closed": rep["scalars"]["closed"], "shells_equal_components": rep["scalars"]["shells_equal_components"]}


def case_checks(case, result, T):
    """Threshold rows per method (thresholds_for_candidate_B applied to B; to A and B_mid for information)."""
    final = result["frames"][-1]["reports"] if result.get("frames") else None
    out = {}
    for method in (A, B, B_MID):
        if final is None or method not in final:
            continue
        rep = final[method]
        sc = rep["scalars"]
        rows = {}
        strict = None
        if case in ("S1_G3a", "S1_G3b", "S2", "S3"):
            sel = s1_variant_rows(case)
            rows = s1_checks(sc, T, current_max=T["S2_deformed_cut"]["current_max_s"] if case == "S2" else None, rows=sel)
            if sel is not None:
                strict = s1_checks(sc, T, current_max=T["S2_deformed_cut"]["current_max_s"] if case == "S2" else None)
        elif case == "S4":
            rows = s4_rows(method, result, rep, T)
        elif case == "S5":
            for k in sorted(rep.get("per_cut", {})):
                for name, c in s1_checks(per_cut_scalars(rep, k), T).items():
                    rows[f"cut{k}_{name}"] = c
        elif case == "S6":
            rows = s6_rows(method, result, rep, T)
        elif case == "S7":
            rows = s7_rows(method, result, rep, T)
        elif case == "S8":
            rows = s8_rows(method, result, T)
        rows.update({f"all_{k}": v for k, v in all_checks(rep, T).items()})
        out[method] = {"checks": rows, "verdict": verdict_of(rows)}
        if strict is not None:
            strict = {**strict, **{f"all_{k}": v for k, v in all_checks(rep, T).items()}}
            out[method]["strict_all_S1_rows"] = {"checks": strict, "verdict": verdict_of(strict)}
    return out


def s4_rows(method, result, rep, T):
    t = T["S4_partial_cut"]
    rows = {"mechanical_components": check(result["frames"][-1]["mechanical_components"], "==", t["mechanical_components"]),
            "coverage_of_conflicting_links_min": check(rep.get("coverage"), ">=", t["coverage_of_conflicting_links_min"])}
    s4 = result.get("s4_provenance", {}).get(method)
    if s4 is None:
        for k in ("cut_faces_on_non_conflicting_links", "slit_extent_equals_conflicting_extent", "tip_representatives_shared"):
            rows[k] = not_applicable("A has no lattice-link provenance")
    else:
        rows["cut_faces_on_non_conflicting_links"] = check(s4["cut_face_links_not_conflicting"], "==", t["cut_faces_on_non_conflicting_links"])
        rows["slit_extent_equals_conflicting_extent"] = check(s4["slit_extent_equal"], "==", True)
        rows["tip_representatives_shared"] = check(s4["tip_shared"], "==", True)
    return rows


def s4_provenance(mesh, locus, codes, cuts):
    """B: links of its cut faces vs the independent conflicting-link set; crack-tip shared representatives."""
    from robotic_cutting_sim import cut_aware_surface as CAS
    from robotic_cutting_sim.cut_topology import side_masks
    surf = mesh["surface"]
    lat = surf.lattice
    last = mesh["last"]
    f = mesh["faces"]
    cut = last["face_kind"] == CAS.LINK_CUT
    links = np.unique(last["face_link"][cut])
    b_pairs = {tuple(sorted((int(lat.link_lo[k]), int(lat.link_hi[k])))) for k in links}
    c_pairs = {tuple(sorted(map(int, p))) for p in locus["pairs"]}
    # vertices of cut faces per side digit: the side particle's digit of the face's cut bits
    side = last["face_side_particle"][cut]
    bits = last["face_cutbits"][cut]
    a1, a2 = side_masks(np.asarray(codes, np.int64)[side])
    is1 = (a1 & bits) != 0
    is2 = (a2 & bits) != 0
    v1 = np.unique(f[cut][is1]) if is1.any() else np.zeros(0, int)
    v2 = np.unique(f[cut][is2]) if is2.any() else np.zeros(0, int)
    shared = np.intersect1d(v1, v2)
    corners = lat.corner_particle[last["vertex_cell"][shared]]
    mask = last["vertex_corner_mask"][shared]
    c = np.where(mask, np.asarray(codes, np.int64)[np.maximum(corners, 0)], 0)
    s1, s2 = side_masks(c)
    mixed = (np.bitwise_or.reduce(s1, axis=1) & np.bitwise_or.reduce(s2, axis=1)) != 0 if len(shared) else np.zeros(0, bool)
    boundary = mesh["info"]["mesh_validity_reference"]["boundary_edges"]
    return {"cut_face_links": len(b_pairs), "conflicting_links": len(c_pairs),
            "cut_face_links_not_conflicting": len(b_pairs-c_pairs), "conflicting_links_without_cut_face": len(c_pairs-b_pairs),
            "slit_extent_equal": b_pairs == c_pairs, "vertices_side1": int(len(v1)), "vertices_side2": int(len(v2)),
            "shared_tip_vertices": int(len(shared)), "shared_tip_vertices_in_mixed_regions": int(mixed.sum()),
            "boundary_edges_reference": int(boundary),
            "tip_shared": bool(len(shared) > 0 and bool(mixed.all()) and boundary == 0)}


def s6_rows(method, result, rep, T):
    t = T["S6_four_slices"]
    fr = result["frames"][-1]
    count = fr["mechanical_components"]
    rows = {"mechanical_components_min": check(count, ">=", t["mechanical_components_min"])}
    topo = rep["topology"]
    if method in (B, B_MID):
        per = [c["shells"] for c in rep["cut_aware_info"]["audit"]["components"]]
    else:
        per = [sum(r["component"] == c for r in topo["shell_rows"]) for c in range(count)]
    rows["shells_per_component_min"] = check(int(min(per)) if per else None, ">=", t["shells_per_component_min"])
    cv = rep["component_volumes_provenance"]
    mat = np.asarray(cv["material_reference_m3"])
    if method in (B, B_MID):
        mesh = np.asarray(cv["mesh_reference_m3"])
        area = np.asarray(cv["cut_area_reference_m2"])   # the component's own cut faces (reference)
        note = "B pre-embedding mesh per component (provenance); A_cut,component = its own cut-face area"
    else:
        mesh = np.asarray(cv["mesh_current_m3"])
        bs = result.get("_b_cut_area_per_component")
        area = None if bs is None else np.asarray(bs)
        note = ("A: current mesh per component (provenance) vs sum m/rho (A has no reference mesh); "
                "A_cut,component taken from B's reference cut faces of the same state")
    for c in range(min(count, len(mat))):
        err = abs(mesh[c]-mat[c])/mat[c]
        bound = (.5*S*area[c]/mat[c] if area is not None else np.nan)+1e-4
        rows[f"component{c}_reference_volume_error"] = check(float(err), "<=", float(bound), note=note)
    return rows


def s7_rows(method, result, rep, T):
    t = T["S7_thin_chip"]
    fr = result["frames"][-1]
    count = fr["mechanical_components"]
    if method in (B, B_MID):
        skipped = sum(c["shells"] == 0 for c in rep["cut_aware_info"]["audit"]["components"])
    else:
        skipped = int(((rep.get("unmeshable_skipped") or {}).get("fragments", 0)) or 0)
    rows = {"skipped_fragments": check(int(skipped), "==", t["skipped_fragments"])}
    cv = rep["component_volumes_provenance"]
    mesh, mat = np.asarray(cv["mesh_current_m3"] or [np.nan]*count), np.asarray(cv["material_current_m3"])
    chip = np.arange(1, count)       # ranked labels: 0 = largest body (substrate)
    if len(chip):
        err = abs(mesh[chip].sum()-mat[chip].sum())/mat[chip].sum()
        rows["chip_current_volume_rel_error"] = check(float(err), "<=", t["chip_current_volume_rel_error"],
                                                      note=f"{len(chip)} chip components, {mat[chip].sum():.3e} m^3")
    else:
        rows["chip_current_volume_rel_error"] = check(None, "<=", t["chip_current_volume_rel_error"], note="no chip detached")
    rows["non_manifold_edges_report"] = {"value": rep["scalars"]["non_manifold_edges"], "verdict": "REPORT_ONLY"}
    return rows


def s8_rows(method, result, T):
    frames = result.get("recontact_frames", [])
    rows = {}
    if method == B_MID:
        return rows
    vals = [f["meshes"][method] for f in frames if method in f["meshes"]]
    if method == B:
        shared = max((v["mixed_side_regions"] for v in vals), default=None)
        rows["vertices_shared_across_conflicting_codes"] = check(shared, "==", 0)
    else:
        rows["vertices_shared_across_conflicting_codes"] = {"value": 0, "verdict": "PASS",
                                                            "note": "by construction: one mesh per mechanical component"}
    sets = {(v["mesh_shells"], tuple(v["shell_components"])) for v in vals}
    rows["shell_set_unchanged_since_separation"] = check(len(sets) == 1 if vals else None, "==", True)
    res = [f["temporal"][method]["residual_after_transport"]["p95_spacings"] for f in frames
           if f.get("temporal") and method in f["temporal"]]
    rows["all_temporal_slow_motion_residual_p95_s"] = check(max(res) if res else None, "<=",
                                                            T["all"]["temporal_slow_motion_residual_p95_s"])
    return rows


def s9_rows(result):
    t = protocol_thresholds()["S9_negative_controls"]
    out = {}
    for name, ctl in result["controls"].items():
        rows = {"events": check(ctl["events"], "==", t["events"]),
                "cut_faces": check(ctl[B]["cut_triangles"], "==", t["cut_faces"]),
                "mesh_equals_S0_mesh_embedded": check(ctl[B]["equals_uncut_extraction"], "==", True),
                "hermite_points_outside_swept_region": check(0 if ctl["events"] == 0 else None, "==",
                                                             t["hermite_points_outside_swept_region"],
                                                             note="no events -> no Hermite points" if ctl["events"] == 0 else
                                                             "events present: swept-region audit not implemented here")}
        out[name] = {B: {"checks": rows, "verdict": verdict_of(rows)},
                     A: {"checks": {"events": rows["events"],
                                    "one_closed_shell": check(ctl[A]["shells"] == 1 and ctl[A]["watertight"], "==", True),
                                    "mesh_equals_S0_mesh_embedded": not_applicable("A is not lattice based")},
                         "verdict": None}}
        out[name][A]["verdict"] = verdict_of(out[name][A]["checks"])
    return out


# ----------------------------------------------------------------------------------------------- cases

def evaluate_case(case, r, run, info, args, T):
    t0 = time.perf_counter()
    frames = frames_of(run, run.get("frame_names"))
    recomputable = save_states(args.out/f"{case}_r{r}_states.npz", run, frames)
    result = {"case": case, "repeat": r, "runner": run["runner"], "physics_wall_s": run["wall_s"],
              "particles": int(len(run["codes"])), "setup": info, "recorder_audit": P1.recorder_audit(run),
              "slot_diagnostics": run["slot_diagnostics"], "states_file": f"{case}_r{r}_states.npz",
              "stencil_recomputable_from_reference": recomputable,
              "recorder_metadata": {k: run["metadata"][k] for k in ("events_logged", "event_overflow", "substeps",
                                                                  "unexpected_code_changes", "log_sha256", "lattice_sha256")}}
    if run.get("phase_names"):
        result["phases"] = P1.phase_activity(run, run["phase_names"])
    if "control_steps" in run:
        result["control_steps"], result["armed_control_steps"] = run["control_steps"], run["armed_control_steps"]
    result["frames"] = []
    b_surfaces = {}
    if case == "S8":
        result["recontact_frames"] = recontact_frames(run, frames[:-1], args, b_surfaces)
    if case == "S6":
        result["slice_frames"] = []
        for fr in frames[:-1]:
            ev, *_ = evaluate_frame(run, fr, args, light=True, temporal=False, b_surfaces=b_surfaces)
            result["slice_frames"].append(ev)
    if case == "S4":
        result["hold_frames"] = []
        for fr in frames[:-1]:
            ev, *_ = evaluate_frame(run, fr, args, light=True, temporal=False, b_surfaces=b_surfaces)
            result["hold_frames"].append(ev)
    ev, meshes, labels, locus, st = evaluate_frame(run, frames[-1], args, legacy=(r == 0), b_surfaces=b_surfaces)
    result["frames"].append(ev)
    if case == "S4":
        result["s4_provenance"] = {B: s4_provenance(meshes[B], locus, st["codes"], st["cuts"]),
                                   B_MID: s4_provenance(meshes[B_MID], locus, st["codes"], st["cuts"])}
        result["g4_hold_phase"] = g4_hold(run, result)
    if case == "S6":
        result["_b_cut_area_per_component"] = ev["reports"][B]["component_volumes_provenance"]["cut_area_reference_m2"]
    if r == 0:
        save_package(args.out/f"{case}_r{r}_package.npz", run, labels, meshes, locus, st)
    result["threshold_checks"] = case_checks(case, result, T)
    result["analysis_wall_s"] = time.perf_counter()-t0
    return result


def recontact_frames(run, frames, args, b_surfaces):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from scipy.spatial import cKDTree
    out, prev = [], None
    for fr in frames:
        ev, meshes, labels, locus, st = evaluate_frame(run, fr, args, light=True, temporal=False, b_surfaces=b_surfaces)
        row = {"frame": fr["name"], "substep": int(fr["substep"]), "time_s": fr["substep"]*DT,
               "components": ev["mechanical_components"], "labelled": int(np.count_nonzero(fr["codes"])),
               "meshes": ev["reports"]}
        if ev["mechanical_components"] == 2:
            d, _ = cKDTree(st["positions"][labels == 0]).query(st["positions"][labels == 1])
            row["min_particle_distance_between_pieces_m"] = float(d.min())
        row["contact_stat_penetration"] = "NOT_AVAILABLE (not in the snapshots)"
        if prev is not None and np.array_equal(prev["codes"], fr["codes"]):
            row["temporal"] = {}
            for name in (A, B):
                pv, pf = prev["meshes"][name]
                row["temporal"][name] = SM.temporal_stability(pv, pf, meshes[name]["vertices"], meshes[name]["faces"],
                                                              prev["positions"], st["positions"], spacing=S)
        out.append(row)
        prev = {"codes": fr["codes"], "positions": st["positions"],
                "meshes": {n: (np.asarray(meshes[n]["vertices"], float), np.asarray(meshes[n]["faces"], np.int64)) for n in (A, B)}}
    return out


def g4_hold(run, result):
    """S9 G4 hold phase: no events while armed and stationary; B topology and cut faces unchanged over the hold."""
    ph = result["phases"]["hold_armed"]
    hf = result["hold_frames"]
    b0, b1 = hf[0]["reports"][B], hf[1]["reports"][B]
    steps = np.cumsum(run["phase_substeps"])
    log = run["log"]
    hold_events = int(np.count_nonzero((log["substep"] >= steps[0]) & (log["substep"] < steps[1])))
    rows = {"events": check(hold_events, "==", 0),
            "new_digits": check(ph["newly_assigned"], "==", 0),
            "cut_faces_unchanged": check(b0["cut_triangles"] == b1["cut_triangles"], "==", True),
            "topology_hash_unchanged": check(b0["topology_hash"] == b1["topology_hash"], "==", True)}
    return {"checks": rows, "verdict": verdict_of(rows), "cut_triangles": [b0["cut_triangles"], b1["cut_triangles"]]}


def case_s0(args, T):
    """G1 static box 40 x 24 x 12 mm (no physics)."""
    from robotic_cutting_sim import surface_attribution_v1 as SA
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.blade_probe import block
    points, masses = block(size_m=(.04, .024, .012), spacing=S)
    lo, hi = points.min(0)-S/2, points.max(0)+S/2
    nodes, weights, gradients = SA.stencil(points, DX)
    n = len(points)
    run = {"reference": points.astype(float), "lattice": np.round((points-points.min(0))/S).astype(int),
           "masses": masses.astype(float), "density": 1500., "positions": points.astype(float),
           "codes": np.zeros(n, np.int64), "F": np.tile(np.eye(3), (n, 1, 1)), "nodes": nodes, "weights": weights,
           "gradients": gradients, "cuts": 1}
    count, labels, _, _ = P1.components(run)
    out = {"case": "S0", "box_m": (hi-lo).tolist(), "particles": n, "methods": {}}
    meshes = dict(P1.extract_surfaces(run, args.device))
    surf = new_b(run, None)
    meshes[B] = b_extract(surf, run, None)
    for name, m in meshes.items():
        v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
        row = {"validity": SM.mesh_validity_v1(v, f), "box": P1.box_metrics(v, f, lo, hi, S), "extract_wall_s": m["wall_s"]}
        if name == B:
            row["cut_aware_info"] = compact_b_info(m["info"])
            again = b_extract(surf, run, None)
            fresh = b_extract(new_b(run, None), run, None)
            row["temporal_fixed_state"] = {
                "topology_stable": all(topo_hash(x["faces"], x["last"]["vertex_reference"]) == topo_hash(f, m["last"]["vertex_reference"]) for x in (again, fresh)),
                "residual_max_s": max(float(np.abs(x["vertices"]-v).max()) for x in (again, fresh))/S}
        out["methods"][name] = row
    t = T["S0_uncut_box"]
    out["threshold_checks"] = {}
    for name in (A, B):
        row = out["methods"][name]
        val, box = row["validity"], row["box"]
        checks = {"whole_surface_p95_s": check(box["area_weighted_p95_spacings"], "<=", t["whole_surface_p95_s"]),
                  "whole_surface_max_s": check(box["max_m"]/S, "<=", t["whole_surface_max_s"]),
                  "volume_rel_error": check(abs(box["volume_relative_error"]), "<=", t["volume_rel_error"]),
                  "closed_oriented_manifold": check(val["boundary_edges"] == 0 and val["non_manifold_edges"] == 0
                                                    and val["inconsistently_oriented_edges"] == 0
                                                    and val["shells_negative_volume"] == 0, "==", t["closed_oriented_manifold"]),
                  "shells": check(val["shells"], "==", t["shells"]),
                  "euler": check(int(val["shell_rows"][0]["euler_characteristic"]) if val["shell_rows"] else None, "==", t["euler"])}
        out["threshold_checks"][name] = {"checks": checks, "verdict": verdict_of(checks)}
    s0 = {"faces": meshes[B]["faces"], "vertex_reference": meshes[B]["last"]["vertex_reference"],
          "lattice_sha256": surf.lattice.sha256()}
    save_package(args.out/"S0_r0_package.npz", dict(run, contact_work=np.zeros(n), cut_progress=np.zeros(n), log=None,
                                                     blade_polygon=np.zeros((0, 2)), half_length=0., trail=None),
                 labels, meshes, SM.material_locus(run["reference"], run["lattice"], run["positions"], run["codes"], cuts=1,
                                                   F=run["F"]), run)
    return out, s0


def case_g2(args, T):
    """Synthetic codes + Hermite data on analytic planes, rigid 3 mm opening (geometry unit test, never physics)."""
    from robotic_cutting_sim import surface_attribution_v1 as SA
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.blade_probe import block
    points, masses = block(size_m=(.04, .024, .016), spacing=S)
    reference = points.astype(float)
    lattice = np.round((points-points.min(0))/S).astype(int)
    nodes, weights, gradients = SA.stencil(points, DX)
    centre = reference.mean(0)+np.array([.0003, .0002, .0001])
    normals = {"n30_xz": [np.cos(np.radians(30)), 0, np.sin(np.radians(30))],
               "n45_xy": [np.cos(np.radians(45)), np.sin(np.radians(45)), 0],
               "n123": list(np.array([1., 2., 3.])/np.sqrt(14)), "axis_x": [1., 0, 0]}
    gap = .003
    t = T["G2_static_unit_test"]
    tol = t["numeric_tolerance_m"]
    out = {"case": "G2", "gap_m": gap, "plane_centre_m": centre.tolist(), "planes": {}, "threshold_checks": {}}
    for name, nrm in normals.items():
        nrm = np.asarray(nrm, float)
        d = (reference-centre)@nrm
        codes = np.where(d >= 0, 2, 1).astype(np.int64)
        log = P1.synthetic_events(reference, lattice, codes, centre, nrm)
        positions = reference+np.where(d[:, None] >= 0, gap*nrm, 0.)
        F = np.tile(np.eye(3), (len(points), 1, 1))
        run = {"reference": reference, "lattice": lattice, "positions": positions, "codes": codes, "F": F,
               "masses": masses.astype(float), "density": 1500., "nodes": nodes, "weights": weights,
               "gradients": gradients, "cuts": 1, "log": log}
        count, labels, _, _ = P1.components(run)
        locus = SM.material_locus(reference, lattice, positions, codes, cuts=1, F=F, events=log)
        ext = SM.exterior_samples(lattice, positions, S, F)
        meshes = {A: P1.extract_surfaces(run, args.device, profiles=(A,))[A]}
        meshes[B] = b_extract(new_b(run, log), run, log)
        meshes[B_MID] = b_extract(new_b(run, None), run, None)
        rows = {}
        for method, m in meshes.items():
            v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
            cls = SM.classify_mesh_vertices(v, positions, labels, locus, ext, spacing=S)
            cut = cls["kind"] == 2
            sd = (v-centre)@nrm-np.where(cls["side"] == 1, gap, 0.)
            nt, _, _ = SM.triangle_normals(v, f)
            core_tri = cls["core"][f].all(1)
            ang = np.degrees(np.arccos(np.clip(np.abs(nt[core_tri]@nrm), 0, 1)))
            val = SM.mesh_validity_v1(v, f)
            row = {"cut_face_vs_analytic": q(np.abs(sd[cut])), "core_vs_analytic": q(np.abs(sd[cls["core"]])),
                   "non_crease_normal_deg": q(ang), "non_manifold_edges": val["non_manifold_edges"],
                   "boundary_edges": val["boundary_edges"], "shells": val["shells"], "region": cls["counts"]}
            if method in (B, B_MID):
                X = m["last"]["vertex_reference"]
                row["reference_cut_face_vs_analytic"] = q(np.abs((X[cut]-centre)@nrm))
                row["qef"] = m["info"]["qef"]
                row["hermite_crossings_by_source"] = m["info"]["hermite"]["crossings_by_source"]
                clamped = m["last"]["vertex_clamped"]
                row["clamped_cut_face_vertices"] = int((clamped & cut).sum())
                row["cut_face_max_excluding_clamped"] = q(np.abs(sd[cut & ~clamped]))
                row["clamped_vs_analytic"] = q(np.abs(sd[cut & clamped]))
            rows[method] = row
        out["planes"][name] = rows
        for method, key in ((B, "hermite"), (B_MID, "midpoint_fallback"), (A, "hermite")):
            tt = t[key]
            r = rows[method]
            c = {"cut_face_p95_s": check(r["cut_face_vs_analytic"]["p95_spacings"], "<=", tt["cut_face_p95_s"]+tol/S),
                 "cut_face_max_s": check(r["cut_face_vs_analytic"]["max_spacings"], "<=", tt["cut_face_max_s"]+tol/S)}
            if key == "hermite":
                c["non_crease_normal_p95_deg"] = check(r["non_crease_normal_deg"]["p95"], "<=", tt["non_crease_normal_p95_deg"])
                c["non_manifold_edges"] = check(r["non_manifold_edges"], "==", tt["non_manifold_edges"])
            out["threshold_checks"].setdefault(method, {})[name] = {"checks": c, "verdict": verdict_of(c),
                                                                    "thresholds_set": key}
    for method, per in out["threshold_checks"].items():
        per["verdict"] = verdict_of({k: {"verdict": v["verdict"]} for k, v in per.items() if isinstance(v, dict)})
    return out


def case_s9(args, T, r, s0):
    """G8 controls; B must equal the uncut extraction (the S0 mesh embedded) at the same state."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    out = {"case": "S9", "repeat": r, "controls": {}}
    for name in S9_CONTROLS:
        run = case_s9_control(args, name)
        count, labels, lat, _ = P1.components(run)
        st = dict(run)
        a = P1.extract_surfaces(st, args.device, profiles=(A,))[A]
        av = SM.mesh_validity_v1(a["vertices"], a["faces"])
        surf = new_b(run, run["log"])
        b = b_extract(surf, run, run["log"])
        uncut = b_extract(new_b(run, None), dict(run, codes=np.zeros_like(run["codes"])), None)
        same_ref = (np.array_equal(b["faces"], uncut["faces"]) and np.array_equal(b["last"]["vertex_reference"],
                                                                                  uncut["last"]["vertex_reference"]))
        same_s0 = (s0 is not None and np.array_equal(b["faces"], s0["faces"])
                   and np.array_equal(b["last"]["vertex_reference"], s0["vertex_reference"]))
        same_cur = same_ref and np.array_equal(b["vertices"], uncut["vertices"])
        bv = SM.mesh_validity_v1(b["vertices"], b["faces"])
        out["controls"][name] = {
            "events": int(len(run["log"])), "labelled": int(np.count_nonzero(run["codes"])),
            "unexpected_code_changes": int(run["metadata"]["unexpected_code_changes"]), "mechanical_components": int(count),
            "lattice_components": int(lat), "max_displacement_m": float(np.linalg.norm(run["positions"]-run["reference"], axis=1).max()),
            "physics_wall_s": run["wall_s"],
            A: {"shells": av["shells"], "watertight": av["watertight"], "triangles": int(len(a["faces"]))},
            B: {"cut_triangles": b["info"]["faces"]["cut_triangles"], "shells": bv["shells"], "watertight": bv["watertight"],
                "equals_uncut_extraction": bool(same_cur), "faces_and_reference_equal_S0_box_mesh": bool(same_s0),
                "triangles": int(len(b["faces"])), "hermite_log_entries": b["info"]["hermite"]["log"]["entries"],
                "embedding": b["info"]["embedding"],
                "volume_current_rel_error": SM.volume_report(b["vertices"], b["faces"], run["positions"], labels,
                                                             run["masses"]/run["density"], run["F"])["total_relative_error_current"]}}
        print(f"[S9 r{r} {name}] events {len(run['log'])} B cut tris {b['info']['faces']['cut_triangles']} "
              f"equal-uncut {same_cur} S0 {same_s0}", flush=True)
    out["threshold_checks"] = s9_rows(out)
    return out


# ----------------------------------------------------------------------------------------------- comparison gate

GATE_METRICS = {   # name: (direction, band kind)
    "current_p50_s": ("lower", "distance"), "current_p95_s": ("lower", "distance"), "current_max_s": ("lower", "distance"),
    "current_core_p95_s": ("lower", "distance"), "reference_p95_s": ("lower", "distance"),
    "reference_max_s": ("lower", "distance"), "locus_to_mesh_p95_s": ("lower", "distance"),
    "coverage": ("higher", "spread"), "abs_bias_toward_side2_s": ("lower", "distance"),
    "abs_thickness_bias_s": ("lower", "distance"), "normal_p50_deg": ("lower", "normal"),
    "normal_p95_deg": ("lower", "normal"), "rim_corner_p95_s": ("lower", "distance"),
    "rim_corner_max_s": ("lower", "distance"), "open_edges_near_junction": ("lower", "spread"),
    "boundary_edges": ("lower", "spread"), "non_manifold_edges": ("lower", "spread"),
    "inconsistently_oriented_edges": ("lower", "spread"), "degenerate_triangles": ("lower", "spread"),
    "shells_negative_volume": ("lower", "spread"), "self_intersections_sampled": ("lower", "spread"),
    "closed": ("true", "spread"), "shells_equal_components": ("true", "spread"),
    "shells_spanning_components": ("lower", "spread"), "components_without_shell": ("lower", "spread"),
    "faces_nearer_other_component": ("lower", "spread"), "volume_current_abs_rel": ("lower", "volume"),
    "volume_reference_abs_rel": ("lower", "volume"), "fixed_state_topology_stable": ("true", "spread"),
    "fixed_state_residual_s": ("lower", "distance"),
}
SLOW_MOTION_METRIC = {"slow_motion_residual_p95_s": ("lower", "distance")}
BANDS = {"distance": .06, "normal": 1., "volume": .005}


def compare(case, rows_a, rows_b, metrics=None):
    """rows_*: list of scalar dicts, one per repeat (same order)."""
    out = {}
    for name, (direction, kind) in (GATE_METRICS if metrics is None else metrics).items():
        a = [r.get(name) for r in rows_a]
        b = [r.get(name) for r in rows_b]
        if any(x is None for x in a+b) or not a:
            out[name] = {"A": a, "B": b, "decision": "NOT_COMPARABLE"}
            continue
        if direction == "true":
            a, b = [float(bool(x)) for x in a], [float(bool(x)) for x in b]
        a, b = np.asarray(a, float), np.asarray(b, float)
        diff = (a-b) if direction in ("higher", "true") else (b-a)    # > 0: B worse
        if kind == "spread":
            band = max(np.ptp(a) if len(a) > 1 else 0., np.ptp(b) if len(b) > 1 else 0.)
        else:
            band = BANDS[kind]
        worse = bool(np.any(diff > band))
        better = bool(np.all(-diff > band))
        out[name] = {"A": a.tolist(), "B": b.tolist(), "band": float(band),
                     "decision": "B_WORSE" if worse else ("B_BETTER" if better else "NOT_DIFFERENT")}
    return out


def gate(results):
    per_case = {}
    for case, reps in results.items():
        if case in ("S0", "G2", "S9") or not isinstance(reps, list):
            continue
        rows_a = [r["frames"][-1]["reports"][A]["scalars"] for r in reps]
        rows_b = [r["frames"][-1]["reports"][B]["scalars"] for r in reps]
        per_case[case] = compare(case, rows_a, rows_b)
        if case == "S5":
            for k in sorted(reps[0]["frames"][-1]["reports"][B].get("per_cut", {})):
                pa = [per_cut_scalars(r["frames"][-1]["reports"][A], k) for r in reps]
                pb = [per_cut_scalars(r["frames"][-1]["reports"][B], k) for r in reps]
                per_case[f"S5_cut{k}"] = compare(f"S5_cut{k}", pa, pb)
        if case == "S8":
            def slow(r, m):
                vals = [f["temporal"][m]["residual_after_transport"]["p95_spacings"] for f in r["recontact_frames"] if f.get("temporal")]
                return {"slow_motion_residual_p95_s": max(vals) if vals else None}
            per_case["S8_temporal"] = compare("S8", [slow(r, A) for r in reps], [slow(r, B) for r in reps],
                                              metrics=SLOW_MOTION_METRIC)
    worse = {c: [m for m, d in rows.items() if d["decision"] == "B_WORSE"] for c, rows in per_case.items()}
    p95 = {c: rows["current_p95_s"]["decision"] for c, rows in per_case.items() if "current_p95_s" in rows}
    coverage = {c: per_case[c]["coverage"]["decision"] for c in ("S4", "S7") if c in per_case}
    topo_keys = ("shells_equal_components", "components_without_shell", "shells_spanning_components")
    topo = {c: {k: rows[k]["decision"] for k in topo_keys if k in rows} for c, rows in per_case.items()}
    topo_better = any(v == "B_BETTER" for d in topo.values() for v in d.values())
    topo_worse = any(v == "B_WORSE" for d in topo.values() for v in d.values())
    s7_skip = None
    if "S7" in results:
        s7_skip = [{m: r["threshold_checks"][m]["checks"]["skipped_fragments"]["value"] for m in (A, B)} for r in results["S7"]]
    measured_p95 = {c: d for c, d in p95.items() if d != "NOT_COMPARABLE"}
    parts = {"not_worse_on_any_metric": not any(worse.values()),
             "better_current_p95_every_case": bool(measured_p95) and all(d == "B_BETTER" for d in measured_p95.values()),
             "better_coverage_partial_and_chip": bool(coverage) and all(d == "B_BETTER" for d in coverage.values()),
             "better_topology": topo_better and not topo_worse}
    return {"per_case": per_case, "B_worse": worse, "current_p95_decisions": p95, "coverage_decisions": coverage,
            "topology_decisions": topo, "S7_skipped_fragments": s7_skip, "parts": parts,
            "verdict": "PASS" if all(parts.values()) else "FAIL"}


# ----------------------------------------------------------------------------------------------- stages

def stage_run(args):
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    import warp as wp
    import newton
    proto_commit = git("log", "-1", "--format=%H", "--", str(PROTOCOL_FILE.relative_to(ROOT)))
    proto_diff = git("status", "--porcelain", "--", str(PROTOCOL_FILE.relative_to(ROOT)))
    tests = args.tests.split(",")
    record = {**DECLARATION, "protocol_file": str(PROTOCOL_FILE.relative_to(ROOT)), "protocol_sha256": sha256_file(PROTOCOL_FILE),
              "protocol_last_commit": proto_commit, "protocol_freeze_commit_expected": FREEZE_COMMIT,
              "protocol_unchanged_since_freeze": proto_commit.startswith(FREEZE_COMMIT) and not proto_diff,
              "protocol_content": json.loads(PROTOCOL_FILE.read_text()),
              "cases": tests, "repeats": args.repeats, "command": " ".join([sys.executable, *sys.argv]),
              "git_head": git("rev-parse", "HEAD"), "git_dirty_paths": len(git("status", "--porcelain").splitlines()),
              "device": args.device, "device_name": wp.get_device(args.device).name, "warp": wp.__version__,
              "newton": newton.__version__, "numpy": np.__version__, "python": platform.python_version(),
              "runner_parameters": P1.PROTOCOL, "written_before_run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (args.out/"protocol_record.json").write_text(json.dumps(jsonable(record), indent=1))
    if not record["protocol_unchanged_since_freeze"]:
        print("WARNING: protocol file changed since the freeze commit; results are recorded against its current hash", flush=True)
    T = protocol_thresholds()
    results, summary, start = {}, {"cases": {}}, time.perf_counter()
    s0 = None
    for case in tests:
        t = time.perf_counter()
        try:
            if case == "S0":
                res, s0 = case_s0(args, T)
                results["S0"] = res
                (args.out/"S0_r0.json").write_text(json.dumps(jsonable(res), indent=1))
                summary["cases"]["S0"] = {m: res["threshold_checks"][m]["verdict"] for m in res["threshold_checks"]}
            elif case == "G2":
                res = case_g2(args, T)
                results["G2"] = res
                (args.out/"G2_r0.json").write_text(json.dumps(jsonable(res), indent=1))
                summary["cases"]["G2"] = {m: v["verdict"] for m, v in res["threshold_checks"].items()}
            elif case == "S9":
                reps = []
                for r in range(args.repeats):
                    res = case_s9(args, T, r, s0)
                    (args.out/f"S9_r{r}.json").write_text(json.dumps(jsonable(res), indent=1))
                    reps.append(res)
                results["S9"] = reps
                summary["cases"]["S9"] = [{c: {m: v[m]["verdict"] for m in v} for c, v in res["threshold_checks"].items()} for res in reps]
            else:
                reps = []
                for r in range(args.repeats):
                    run, info = RUNNERS[case](args, r)
                    res = evaluate_case(case, r, run, info, args, T)
                    (args.out/f"{case}_r{r}.json").write_text(json.dumps(jsonable(res), indent=1))
                    print(f"[{case} r{r}] components {res['frames'][-1]['mechanical_components']} | "
                          + " | ".join(f"{m} {v['verdict']}" for m, v in res["threshold_checks"].items())
                          + f" | physics {res['physics_wall_s']:.1f} s analysis {res['analysis_wall_s']:.1f} s", flush=True)
                    reps.append(res)
                    del run
                results[case] = reps
                summary["cases"][case] = [{m: v["verdict"] for m, v in res["threshold_checks"].items()} for res in reps]
                if case == "S4":
                    summary["cases"]["S9_G4_hold"] = [res["g4_hold_phase"]["verdict"] for res in reps]
            summary.setdefault("wall_s", {})[case] = time.perf_counter()-t
        except Exception as error:   # recorded, never hidden
            import traceback
            traceback.print_exc()
            summary["cases"][case] = {"ERROR": repr(error)}
        (args.out/"summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    try:
        summary["comparison_gate"] = gate(results)
    except Exception as error:
        import traceback
        traceback.print_exc()
        summary["comparison_gate"] = {"ERROR": repr(error)}
    summary["total_wall_s"] = time.perf_counter()-start
    (args.out/"summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    print(json.dumps(jsonable({"cases": summary["cases"], "gate": summary["comparison_gate"].get("verdict"),
                               "parts": summary["comparison_gate"].get("parts")}), indent=1))


def load_states(path):
    z = np.load(path, allow_pickle=False)
    from robotic_cutting_sim import cut_aware_surface as CAS
    reference = z["reference"].astype(np.float32)
    if bool(z["stencil_recomputable_from_reference"]):
        nodes, weights, gradients = CAS.reference_stencil(reference, float(z["grid_dx_m"]))
    else:
        nodes, weights, gradients = z["nodes"], z["weights"], z["gradients"]
    run = {"reference": reference.astype(float), "lattice": z["lattice"], "masses": z["masses"].astype(float),
           "density": float(z["density"]), "cuts": int(z["cuts"]), "nodes": nodes, "weights": weights, "gradients": gradients,
           "log": z["log"]}
    frames = [{"name": str(n), "substep": int(s), "positions": z["positions"][i].astype(float), "codes": z["codes"][i],
               "F": z["F"][i].astype(float)} for i, (n, s) in enumerate(zip(z["frame_names"], z["frame_substeps"]))]
    for fr in frames:
        fr["log"] = run["log"] if fr["name"] == "final" else run["log"][run["log"]["substep"] < fr["substep"]]
    return run, frames


def perf_state(run, frames, device, label):
    """Timings, mesh size, memory on one recorded state (+ its earlier frames for the incremental update)."""
    import warp as wp
    from robotic_cutting_sim import cut_aware_surface as CAS
    from robotic_cutting_sim.cutting_surface import CuttingSurface, CurrentMechanicalPartition
    final = frames[-1]
    pos, codes, F, log = final["positions"], final["codes"], final["F"], final["log"]
    out = {"state": label, "particles": int(len(codes)), "cuts": run["cuts"], "events": int(len(log)),
           "frames": [f["name"] for f in frames]}
    # A on the GPU
    wp.synchronize_device(device)
    mem0 = wp.get_mempool_used_mem_current(device)
    part_a = CurrentMechanicalPartition(run["nodes"], run["weights"], run["gradients"], run["cuts"], encoding="packed2")
    t = time.perf_counter()
    part_a.partition(codes)
    partition_s = time.perf_counter()-t
    surf_a = CuttingSurface(part_a, S, S/2, device, profile=A, unmeshable="skip_presentation_v1")
    times = []
    for _ in range(4):
        wp.synchronize_device(device)
        t = time.perf_counter()
        va, fa, ia = surf_a.extract(pos, codes, F)
        wp.synchronize_device(device)
        times.append(time.perf_counter()-t)
    out[A] = {"extract_s_first": times[0], "extract_s_median_warm": float(np.median(times[1:])),
              "mechanical_partition_first_s": partition_s, "vertices": int(len(va)), "triangles": int(len(fa)),
              "gpu_mempool_used_current_before_bytes": int(mem0),
              "gpu_mempool_used_current_after_bytes": int(wp.get_mempool_used_mem_current(device)),
              "gpu_mempool_used_high_bytes": int(wp.get_mempool_used_mem_high(device)),
              "note": "mempool high-water mark of this fresh perf process (A is the only GPU user here)"}
    # B on the CPU (NumPy/SciPy reference implementation)
    rss0 = proc_memory()
    t = time.perf_counter()
    part_b = CurrentMechanicalPartition(run["nodes"], run["weights"], run["gradients"], run["cuts"], encoding="packed2")
    surf = CAS.CutAwareSurface(part_b, S, run["lattice"], run["reference"], cut_events_log=log, masses_kg=run["masses"],
                               density_kg_m3=run["density"], strict_audit=False)
    setup_s = time.perf_counter()-t
    rss1 = proc_memory()
    full = []
    for i in range(3):
        t = time.perf_counter()
        vb, fb, ib = surf.extract(pos, codes, F, full_rebuild=True)
        full.append({"wall_s": time.perf_counter()-t, **ib["timings_s"], "update": ib["update"]})
    unchanged = []
    for i in range(3):
        t = time.perf_counter()
        _, _, iu = surf.extract(pos, codes, F)
        unchanged.append({"wall_s": time.perf_counter()-t, **iu["timings_s"], "update": iu["update"]})
    codes64 = np.asarray(codes).astype(np.int64)
    herm = []
    for _ in range(3):
        t = time.perf_counter()
        surf._hermite(codes64, log)
        herm.append(time.perf_counter()-t)
    t = time.perf_counter()
    surf.face_varying_normals(vb, fb)
    normals_s = time.perf_counter()-t
    lsq_surf = CAS.CutAwareSurface(part_b, S, run["lattice"], run["reference"], cut_events_log=log, strict_audit=False)
    t = time.perf_counter()
    _, _, il = lsq_surf.extract(pos, codes, None)
    lsq = {"wall_s": time.perf_counter()-t, **il["timings_s"], "embedding": il["embedding"]}
    rss2 = proc_memory()
    # incremental over the recorded frames vs a full rebuild per frame
    inc_surf = CAS.CutAwareSurface(part_b, S, run["lattice"], run["reference"], strict_audit=False)
    seq = []
    for fr in frames:
        t = time.perf_counter()
        vi, fi, ii = inc_surf.extract(fr["positions"], fr["codes"], fr["F"], cut_events_log=fr["log"])
        ti = time.perf_counter()-t
        t = time.perf_counter()
        vf, ff, iff = surf.extract(fr["positions"], fr["codes"], fr["F"], cut_events_log=fr["log"], full_rebuild=True)
        tf = time.perf_counter()-t
        seq.append({"frame": fr["name"], "incremental_wall_s": ti, "incremental_timings_s": ii["timings_s"],
                    "update": ii["update"], "full_wall_s": tf, "full_timings_s": iff["timings_s"],
                    "identical": bool(np.array_equal(vi, vf) and np.array_equal(fi, ff)),
                    "vertices": int(len(vi)), "triangles": int(len(fi))})
    med = lambda rows, k: float(np.median([r[k] for r in rows]))  # noqa: E731
    out[B] = {"setup_static_tables_s": setup_s, "full_rebuild_median_s": med(full, "wall_s"),
              "full_rebuild_breakdown_median_s": {k: med(full, k) for k in ("mechanical_partition_s", "topology_and_qef_s",
                                                                             "quads_s", "embedding_s", "audit_s", "total_s")},
              "cut_event_processing_hermite_median_s": float(np.median(herm)),
              "unchanged_frame_median_s": med(unchanged, "wall_s"),
              "unchanged_frame_breakdown_median_s": {k: med(unchanged, k) for k in ("topology_and_qef_s", "quads_s",
                                                                                     "embedding_s", "audit_s")},
              "face_varying_normals_s": normals_s, "lsq_embedding_no_F": lsq, "incremental_sequence": seq,
              "vertices": int(len(vb)), "triangles": int(len(fb)), "cut_triangles": ib["faces"]["cut_triangles"],
              "rss": {"before": rss0, "after_setup": rss1, "after_extractions": rss2,
                      "setup_delta_bytes": rss1["VmRSS_bytes"]-rss0["VmRSS_bytes"],
                      "extraction_delta_bytes": rss2["VmRSS_bytes"]-rss1["VmRSS_bytes"]},
              "device": "cpu (NumPy/SciPy reference; Warp port not done)"}
    out["usd_upload"] = {"status": "NOT_RUN", "reason": "measured by the Native integration later"}
    return out


def stage_perf(args):
    import warp as wp
    run_dir = args.out
    k = 1
    while (run_dir/f"perf_{k:03d}").exists():
        k += 1
    out_dir = run_dir/f"perf_{k:03d}"
    out_dir.mkdir()
    wp.init()
    report = {"command": " ".join([sys.executable, *sys.argv]), "git_head": git("rev-parse", "HEAD"),
              "device": args.device, "device_name": wp.get_device(args.device).name, "warp": wp.__version__,
              "rss_at_start": proc_memory(), "states": {}}
    for case in args.perf_cases.split(","):
        path = run_dir/f"{case}_states.npz"
        if not path.exists():
            report["states"][case] = {"status": "MISSING", "path": str(path)}
            continue
        run, frames = load_states(path)
        t = time.perf_counter()
        report["states"][case] = perf_state(run, frames, args.device, case)
        report["states"][case]["stage_wall_s"] = time.perf_counter()-t
        print(f"[perf {case}] A {report['states'][case][A]['extract_s_median_warm']:.3f} s, "
              f"B full {report['states'][case][B]['full_rebuild_median_s']:.3f} s", flush=True)
        (out_dir/"perf.json").write_text(json.dumps(jsonable(report), indent=1))
    report["rss_at_end"] = proc_memory()
    (out_dir/"perf.json").write_text(json.dumps(jsonable(report), indent=1))


FIGURES = {   # name: (package, digit, plane_point None = median side locus, section normal None = edge, window_m)
    "S1_G3b_oblique": ("S1_G3b_r0_package.npz", 0, None, None, .006),
    "S1_G3a_cut01r": ("S1_G3a_r0_package.npz", 0, None, None, .008),
    "S4_partial": ("S4_r0_package.npz", 0, None, None, .006),
    "S6_slices": ("S6_r0_package.npz", 1, None, None, .014),
    "S7_chip": ("S7_r0_package.npz", 0, "component:1", None, .008),
    "S8_recontact": ("S8_r0_package.npz", 0, None, None, .006),
    "S3_curved": ("S3_r0_package.npz", 0, None, None, .008),
    "S2_compress_cut": ("S2_r0_package.npz", 0, None, None, .008),
}


def _rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def stage_figures(args):
    from robotic_cutting_sim import cut_debug_view as V
    media = args.media if args.media.is_absolute() else ROOT/args.media
    media.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, (pkg_name, digit, point, normal, window) in FIGURES.items():
        path = args.out/pkg_name
        if not path.exists():
            continue
        pkg = V.load(path)
        if isinstance(point, str) and point.startswith("component:"):     # centre on a detached component (chip)
            sel = np.asarray(pkg["labels"]) == int(point.split(":")[1])
            point = np.median(np.asarray(pkg["positions"], float)[sel], 0) if sel.any() else None
        png = media/f"phase3_{name}_A_vs_B.png"
        if png.exists():
            raise SystemExit(f"Refusing to overwrite {png}")
        V.render_comparison(pkg, png, profiles=(A, B), digit=digit, window_m=window, plane_point=point, plane_normal=normal,
                            slab_m=1.2*S,
                            title=f"{name}: A current_volume_v1 vs B cut_aware_surface_v1 (r0)")
        written[name] = _rel(png)
    pkg = V.load(args.out/"S1_G3b_r0_package.npz")
    png = media/"phase3_reconstruction_vs_particles_S1_G3b.png"
    if not png.exists():
        V.render_reconstruction_vs_particles(pkg, png, profiles=(A, B), digit=0, window_m=.006, slab_m=1.2*S,
                                             title="S1 G3b: section through the cut, particles by side vs A and B")
        written["reconstruction_vs_particles"] = _rel(png)
    (media/"index.json").write_text(json.dumps({"source_run": _rel(args.out), "figures": written}, indent=1))
    print(json.dumps(written, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("run", "perf", "figures"))
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tests", default=",".join(ALL_CASES))
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--perf-cases", default="S1_G3a_r0,S6_r0,S5_r0")
    p.add_argument("--media", type=Path, default=Path("docs/media/wp12/phase3"))
    args = p.parse_args()
    args.out = args.out if args.out.is_absolute() else ROOT/args.out
    {"run": stage_run, "perf": stage_perf, "figures": stage_figures}[args.stage](args)


if __name__ == "__main__":
    main()
