"""Assemble the compact evidence of the P1-1b B2 grasp follow-up (docs/evidence/rl_ready/p1_1b_b2_001.json).

Reads the raw outputs under ``artifacts/rl_ready/p1_1b_b2/`` (not committed) and the WP-17 Native recordings, and
writes per-run summaries; nothing is recomputed from physics. Missing or crashed runs are listed, not dropped.

    PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_1b_b2_evidence.py --out docs/evidence/rl_ready/p1_1b_b2_001.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

A = Path("artifacts/rl_ready/p1_1b_b2")
NATIVE = "artifacts/wp17/native_optimized_001/seed_{seed}/box/env/recording/attempt_000/actions.jsonl"


NOTES = {
    "ood_gpu_huge_preguard_001": "KILLED by the operator after step i=2 (174 s/step against 0.6 s; step-rate collapse "
                                 "reproduced, no SIGSEGV); pre-guard code (src of 41d9405); first step jumped from the "
                                 "recorded initial edge to step 150 (probe start fixed in da32413)",
    "ood_cpu_none_001": "pre-guard; first step jumped from the recorded initial edge to step 150",
    "ood_cpu_far_001": "pre-guard; first step jumped from the recorded initial edge to step 150",
    "ood_cpu_huge_001": "pre-guard; first step jumped from the recorded initial edge to step 150",
    "ood_cpu_huge_postguard_001": "SUPERSEDED by _002: started from the recorded initial edge; the guard flagged that "
                                  "jump to step 150 in both envs (probe artefact, fixed in da32413)",
    "ood_cpu_sweep_postguard_001": "SUPERSEDED by _002 (same probe-start artefact)",
    "s3_b2grip_n2_001": "ran in the main worktree with the material guard written but not yet committed "
                        "(tracked_src_or_scripts_modified true); workload_b_sources_sha256 660af37c... equals the "
                        "committed sources at a930a8c; the guard set no flag",
    "robot_b2grip_n2_001": "sources equal to commit 41d9405 (probe imported before that commit was made; src "
                           "unchanged from a03c23d)",
}


def _load(run):
    path = A/run/"row.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _log_tail(run, n=6):
    path = A/run/"stdout.log"
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-n:]


def native_reference():
    out = {}
    for seed in (512, 15):
        slip, angle, status, fraction = [], [], set(), []
        with open(NATIVE.format(seed=seed)) as f:
            for line in f:
                g = json.loads(line)["grasp_status"]
                slip.append(g["slip_since_settling_m"])
                angle.append(g["rotation_since_settling_rad"])
                status.add(g["status"])
                fraction.append(g["bilateral_loaded_tool_fraction"])
        out[str(seed)] = {"source": NATIVE.format(seed=seed), "steps": len(slip), "max_slip_m": max(slip),
                          "max_rotation_rad": max(angle), "statuses": sorted(status),
                          "min_bilateral_fraction": min(fraction),
                          "contacts_per_step_note": "PhysX ContactSensor raw_contact_count 8 (4 per pad) at step 50"}
    return out


def diag_row(run):
    r = _load(run)
    if r is None:
        return {"run": run, "status": "MISSING", "log_tail": _log_tail(run)}
    s = r.get("summary", {})
    trace = np.load(A/run/"diag_trace.npz") if (A/run/"diag_trace.npz").is_file() else None
    row = {"run": run, "artifact": str(A/run), "label": r["label"], "device": r["device"], "N": r["N"],
           "scene_seeds": r["scene_seeds"], "steps_completed": r["steps_completed"], "grip_profile": r["grip_profile"],
           "diagnostic_solver_overrides": r["diagnostic_solver_overrides"],
           "contact_solver": r.get("contact_solver") or {"options": r["geom_receipt"]["options"]},
           "settle": r["grasp_initialization"]["status"],
           "settle_accumulators": r["grasp_initialization"].get("accumulators"),
           "error": r["error"], "termination_reason": r["termination_reason"],
           "max_slip_m": s.get("max_slip_m"), "max_rotation_rad": s.get("max_rotation_rad"),
           "slip_at_step": s.get("slip_at"), "rotation_at_step": s.get("rotation_at"),
           "first_lost_step": s.get("first_lost_step"), "final_monitor_status": s.get("final_status"),
           "mean_contacts_left_pad": s.get("mean_count_left"), "min_contacts_left_pad": s.get("min_count_left"),
           "mean_contacts_right_pad": s.get("mean_count_right"), "min_contacts_right_pad": s.get("min_count_right"),
           "max_cone_usage_left": s.get("max_usage_left"), "finger_q_range_m": s.get("finger_q_range"),
           "commit": r["commit"], "command": r.get("command"), "gpu_sharing": r.get("gpu_sharing")}
    if trace is not None:
        status = trace["monitor_status"]
        count = np.concatenate([trace["count_left"], trace["count_right"]], 1)
        row["monitor_status_counts"] = [{str(int(k)): int(v) for k, v in zip(*np.unique(status[:, e], return_counts=True))}
                                        for e in range(status.shape[1])]
        row["fraction_steps_4_contacts_per_pad"] = float(np.mean(count == 4))
        row["fraction_steps_le_2_contacts_per_pad"] = float(np.mean(count <= 2))
        row["max_tool_angular_speed_rad_s"] = float(np.linalg.norm(trace["tool_w"], axis=-1).max())
    return row


def lab_row(run):
    r = _load(run)
    if r is None:
        return {"run": run, "status": "FAIL_NO_ROW", "log_tail": _log_tail(run)}
    keys = ("probe", "N", "tool_mount", "driver", "material", "steps", "scene_seeds", "use_cuda_graph", "graph_captured",
            "deterministic_mode", "strict_mode", "reduced_fidelity", "reasons", "listed_differences", "gpu_sharing",
            "vram_own_peak_mib", "commit", "grasp_initialization", "grasp_trace_summary", "termination_reason",
            "reward_fractions", "engagement", "particles_per_env", "hashgrids", "g0_graph_safety")
    row = {"run": run, "artifact": str(A/run)}
    row.update({k: r[k] for k in keys if k in r})
    if "warm" in r:
        row["warm"] = {k: r["warm"].get(k) for k in ("steps", "step_wall_s_mean", "step_wall_s_p50", "step_wall_s_p95",
                                                      "env_steps_per_s")}
    if "telemetry" in r:
        row["foreign_sm_pct"] = (r["telemetry"] or {}).get("foreign_sm_pct")
    if "receipt" in r:
        row["contact_solver"] = r["receipt"].get("contact_solver")
        row["grip_material"] = r.get("grip_material")
        coupling = r["receipt"].get("coupling") or {}
        row["material_input_domain"] = coupling.get("input_domain")
    if "evaluation" in r:
        row["evaluation"] = [{k: e.get(k) for k in ("env", "task_id", "labelled", "labelled_by_digit",
                                                     "final_codes_sha256", "error")} | {"summary": e.get("summary")}
                             for e in r["evaluation"]]
    if (A/run/"grasp_trace.npz").is_file():
        g = np.load(A/run/"grasp_trace.npz")
        st = g["status"]
        row["monitor_status_counts"] = [{str(int(k)): int(v) for k, v in zip(*np.unique(st[:, e], return_counts=True))}
                                        for e in range(st.shape[1])]
        row["non_holding_steps"] = [(np.flatnonzero(st[:, e] != 0)+1).tolist()[:20] for e in range(st.shape[1])]
    return row


def ood_row(run):
    r = _load(run)
    if r is None:
        return {"run": run, "status": "MISSING", "log_tail": _log_tail(run)}
    steps = r.get("steps", [])
    return {"run": run, "artifact": str(A/run), "device": r["device"], "scenario": r["scenario"],
            "spacing": r["spacing"], "particles": r.get("particles"), "graph": r.get("graph"), "commit": r["commit"],
            "status": r.get("status", "INCOMPLETE (process ended before the last step; see log_tail)"),
            "steps": [{"i": s["i"], "wall_s": s["run"]["wall_s"], "invalid_reason": s["run"]["invalid_reason"],
                       "max_abs_x": s["run"]["max_abs_x"], "finite_x": s["run"]["finite_x"],
                       "pending_force_norm": [float(np.linalg.norm(w[:3])) for w in s["run"]["pending_wrench"]],
                       "others_bitwise": s.get("others_bitwise")} for s in steps],
            "log_tail": _log_tail(run, 3)}


def g4_row(run):
    path = A/run/"g4.json"
    if not path.is_file():
        return {"run": run, "status": "FAIL_NO_ROW", "log_tail": _log_tail(run, 4)}
    d = json.loads(path.read_text())
    info = d["info"]
    return {"run": run, "task_id": d["task_id"], "grip_contact_profile": info.get("grip_contact_profile"),
            "contact_solver": info.get("contact_solver"), "settle": info["grasp_initialization"].get("status"),
            "failed": info["grasp_initialization"].get("failed"), "strict": info.get("strict")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = sorted(x.name for x in A.iterdir() if x.is_dir())
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    evidence = {"schema": "rcs_rl_ready_p1_1b_b2_v1", "work_package": "P1-1b follow-up: B2 physical-grasp failure",
                "report": "docs/reports/rl_ready/P1_1B_ROBOT_BATCHING.md (section 'B2 grasp follow-up')",
                "evidence_commit": sha, "artifacts": str(A), "native_wp17_grasp": native_reference(),
                "diagnosis": [diag_row(r) for r in runs if r.startswith(("diag_", "prof_"))],
                "lab_rows": [lab_row(r) for r in runs if r.startswith(("robot_", "s3_", "g2mat_", "g0_"))],
                "settle_repeats": [g4_row(r) for r in runs if r.startswith("g4_")],
                "material_input_domain": [ood_row(r) for r in runs if r.startswith("ood_")],
                "material_results": "SIMULATOR_ONLY_UNCALIBRATED"}
    src = Path("src/robotic_cutting_sim")
    evidence["source_sha256"] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                                 (src/"batched/lab_scene.py", src/"batched/material_batch.py",
                                  src/"rl/lab_batched_env.py")}
    for section in ("diagnosis", "lab_rows", "settle_repeats", "material_input_domain"):
        for row in evidence[section]:
            if row["run"] in NOTES:
                row["note"] = NOTES[row["run"]]
    Path(args.out).write_text(json.dumps(evidence, indent=1, default=str)+"\n")
    print(f"wrote {args.out}: {len(evidence['diagnosis'])} diagnosis, {len(evidence['lab_rows'])} lab, "
          f"{len(evidence['material_input_domain'])} material-domain rows")


if __name__ == "__main__":
    main()
