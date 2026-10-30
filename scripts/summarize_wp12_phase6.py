#!/usr/bin/env python3
"""Read-only summary of the WP-12 Phase 6 readiness runs -> docs/evidence/wp12/phase6_readiness_001.json.

  .venv/bin/python scripts/summarize_wp12_phase6.py --run kinematic_mixed=artifacts/wp12/phase6_001 \
      --run native=artifacts/wp12/phase6_002 --run kinematic_full_physics=artifacts/wp12/phase6_003 \
      --tests tests_phase6.json --out docs/evidence/wp12/phase6_readiness_001.json

Per run: the phase6_report.json checks and episodes, per-episode StageProfiler stages (profile.json, when the run
used --profile), and the cut-aware surface timings of every extraction (surface_provenance.json rows). Nothing is
recomputed from physics; nothing is written outside --out.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("control_step", "backend_step", "material_advance", "surface_extract_usd", "camera_render_readback",
          "camera_policy_refresh", "grasp_contact_sample", "physx_step", "ik_controller", "proprioception_readback",
          "oracle_history", "observation", "action_validation", "collision_check", "policy_act", "camera_extrinsics")
EPISODE_STAGES = ("reset", "native_reset", "material_reset", "evaluation_topology", "recording_finish_io",
                  "dataset_export_io")


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    a = np.asarray(values)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_profile(directory):
    path = directory/"profile.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    per = doc.get("per_control_step", {})
    out = {"control_steps": doc.get("control_steps"), "control_step_total_s": doc.get("control_step_total_s"),
           "per_control_step_ms": {k: {"mean": per[k]["mean_ms"], "max": per[k]["max_ms"],
                                        "exclusive_mean": per[k]["exclusive_mean_ms"],
                                        "share": per[k]["share_of_control_step"], "steps_with_stage": per[k]["steps_with_stage"]}
                                    for k in STAGES if k in per}}
    level = doc.get("episode_level", {})
    out["episode_level_s"] = {k: level[k] for k in EPISODE_STAGES if k in level}
    return out


def surface_timings(directory):
    path = directory/"surface_provenance.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    rows = doc.get("rows") or []
    keys = sorted({k for r in rows for k in (r.get("timings_ms") or {})})
    return {"extractions": len(rows), "update_modes": doc["summary"].get("update_modes"),
            "audit_error_extractions": doc["summary"].get("audit_error_extractions"),
            "timings_ms": {k: _stats([(r.get("timings_ms") or {}).get(k) for r in rows]) for k in keys},
            "cut_events": {k: doc.get("cut_events", {}).get(k) for k in ("enabled", "log_sha256", "log_bytes",
                                                                          "device_bytes", "events_per_cut")},
            "triangles": _stats([r["mesh"]["triangles"] for r in rows])}


def recorded_leak_walk(directory):
    from robotic_cutting_sim.agentic_readiness import recorded_leak_walk as walk
    return walk(directory)


def step_timings(directory):
    """Per-step wall times recorded in actions.jsonl (task steps): the control step, the material (TLMPM) step and,
    on Native, the robot-physics loop; split by whether the policy camera rendered on that step."""
    rows = [json.loads(line) for line in (directory/"actions.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if (r.get("info") or {}).get("return_home_phase") is None]
    def diag(r, key):
        for block in ("privileged_diagnostics", "material_diagnostics"):
            value = (r.get(block) or {}).get(key)
            if value is not None:
                return value
        material = (r.get("privileged_diagnostics") or {}).get("material") or {}
        return material.get(key)
    out = {"steps": len(rows), "step_wall_ms": _stats([1e3*r["step_wall_s"] for r in rows if r.get("step_wall_s") is not None]),
           "material_wall_ms": _stats([1e3*diag(r, "material_wall_s") for r in rows if diag(r, "material_wall_s") is not None]),
           "robot_physics_wall_ms": _stats([1e3*diag(r, "robot_physics_wall_s") for r in rows
                                            if diag(r, "robot_physics_wall_s") is not None])}
    log = directory/"action_log.jsonl"
    if log.exists():
        logs = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        out["policy_env_step_wall_ms"] = _stats([1e3*r["wall_s"] for r in logs if r.get("wall_s") is not None])
    return out


def summarize_run(name, root):
    root = Path(root)
    report_path = root/"phase6_report.json"
    manifest = json.loads((root/"phase6_manifest.json").read_text())
    out = {"name": name, "root": str(root.relative_to(ROOT) if root.is_absolute() else root),
           "plan_id": manifest["config"]["plan"].get("plan_id"), "runtime": manifest["config"]["plan"]["runtime"],
           "runtime_profile": manifest["config"]["plan"].get("runtime_profile"),
           "surface_profile": manifest["config"]["plan"].get("surface_profile"),
           "runtime_options": manifest["config"]["runtime_options"], "git": {k: manifest.get(k) for k in ("git_sha", "git_dirty")},
           "args": manifest["config"]["args"]}
    if not report_path.exists():
        failure = root/"failure.json"
        out.update(status="NO_REPORT", failure=json.loads(failure.read_text()) if failure.exists() else None)
        return out
    report = json.loads(report_path.read_text())
    out.update(status=report["status"], checks=report["checks"], failing=report["failing"],
               memory=report["memory"], repeatability=report["repeatability"],
               attachment_drift=report["attachment_drift"], execution_failures=report["execution_failures"],
               wall_s=report["wall_s"], report_sha256=sha(report_path))
    episodes = []
    for row in report["episodes"]:
        d = root/row["episode_id"]
        meta = json.loads((d/"episode.json").read_text())
        result = meta.get("task_result") or {}
        outcome = result.get("outcome") or {}
        side = json.loads((d/"phase6.json").read_text())
        episodes.append({**row, "acceptance_valid": meta.get("acceptance_valid"),
                         "outcome_metrics": {k: outcome.get("metrics", {}).get(k) for k in (
                             "mechanical_components", "piece_mass_fractions", "substantial_pieces")
                             if k in outcome.get("metrics", {})},
                         "outcome_checks": outcome.get("checks"), "integrity": (result.get("integrity") or {}).get("status"),
                         "evaluation_wall_s": meta.get("evaluation_wall_s"), "reset_wall_s": (meta.get("reset_info") or {}).get("reset_wall_s"),
                         "steps_recorded": meta.get("steps_recorded"), "termination_reason": meta.get("termination_reason"),
                         "task_end_reason": meta.get("task_end_reason"),
                         "post_reset": side.get("post_reset"), "attachment_reset": side.get("attachment_reset"),
                         "attachment_end": side.get("attachment_end"),
                         "recorded_leak_walk": recorded_leak_walk(d), "wp12_leak_walk": side.get("wp12_leak_walk"),
                         "profile": episode_profile(d), "surface": surface_timings(d), "step_timings": step_timings(d)})
    out["episodes"] = episodes
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], help="NAME=DIR")
    p.add_argument("--tests", help="JSON with the recorded pytest commands and counts")
    p.add_argument("--extra", action="append", default=[], help="NAME=JSON file to embed (e.g. a goal-feature result)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from robotic_cutting_sim.recording import manifest
    git = manifest({}, "probe")
    doc = {"schema": "rcs_wp12_phase6_readiness_evidence_v1", "spec": "docs/WP12_SPECIFICATION.md Phase 6",
           "written_by": "scripts/summarize_wp12_phase6.py (read-only)",
           "git": {k: git.get(k) for k in ("git_sha", "git_dirty")}, "runs": {}, "extra": {}}
    for item in a.run:
        name, _, root = item.partition("=")
        doc["runs"][name] = summarize_run(name, root)
    for item in a.extra:
        name, _, path = item.partition("=")
        doc["extra"][name] = {"path": path, "sha256": sha(path), "content": json.loads(Path(path).read_text())}
    if a.tests:
        doc["tests"] = json.loads(Path(a.tests).read_text())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str)+"\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT/"src"))
    sys.exit(main())
