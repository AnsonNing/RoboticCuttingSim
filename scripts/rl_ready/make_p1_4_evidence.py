#!/usr/bin/env python3
"""Build ``docs/evidence/rl_ready/p1_4_001.json`` from the analyses of every P1-4 run (T4.7).

Every run directory listed below is analysed (``analyze_long_rollout.analyze``) and reduced to a compact form:
co-tenant PID lists become counts, per-episode rows keep class / verdict / labelled / canary / L2. Failed and
smoke runs are listed with their status; nothing is dropped. The output is created exclusively.

    .venv/bin/python scripts/rl_ready/make_p1_4_evidence.py --out docs/evidence/rl_ready/p1_4_001.json \\
        --extra-json tests=<json with test counts>
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "rl_ready"))

from analyze_long_rollout import analyze, compact  # noqa: E402

P14 = ROOT / "artifacts" / "rl_ready" / "p1_4"
V1 = ROOT / "configs" / "rl_benchmark" / "long_rollout_v1.json"
V2 = ROOT / "configs" / "rl_benchmark" / "long_rollout_v2.json"
RUNS = [  # (run dir, config, role)
    ("lr_a_001", V1, "LR-A main run (T4.4), evidence"),
    ("lr_b_001", V2, "LR-B reset stress (T4.5), evidence"),
    ("lr_c_001", V2, "LR-C rebuild cycles (T4.5), evidence"),
    ("lr_d_nan_001", V2, "LR-D probe nan_env:1@20: FAILED (harness bug: diagnostic evaluate() of a NaN env raised); preserved"),
    ("lr_d_nan_002", V2, "LR-D probe nan_env:1@20 rerun after the fix (968f522), evidence"),
    ("lr_d_pose_001", V2, "LR-D probe bad_pose:0@20 (T4.2 re-check), evidence"),
    ("lr_d_segv_001", V2, "LR-D probe sigsegv@20, evidence"),
    ("lr_d_hang_001", V2, "LR-D probe hang@20, evidence"),
    ("lr_d_oom_001", V2, "LR-D probe device_oom@20, evidence"),
    ("smoke_trim_001", P14 / "smoke_config_001.json", "GPU smoke of the trim backend (40-step episodes), never evidence"),
    ("t45_smoke_b_001", P14 / "smoke_config_v2_001.json", "GPU smoke LR-B code path, never evidence"),
    ("t45_smoke_c_001", P14 / "smoke_config_v2_001.json",
     "GPU smoke LR-C code path: exposed the post-close measurement bug (a884c86), never evidence"),
    ("t45_smoke_c_002", P14 / "smoke_config_v2_001.json", "GPU smoke LR-C after the fix, never evidence"),
]


def slim_telemetry(t):
    t = dict(t)
    for key in ("foreign_compute_pids", "foreign_graphics_pids"):
        if key in t:
            t[key.replace("_pids", "_process_count")] = len(t.pop(key))
    return t


PROBES = {  # run -> (declared probe, expected supervisor status, episode check)
    "lr_d_nan_001": ("nan_env:1@20", "COMPLETE", ("1", "TERMINATED", "MATERIAL_INVALID")),
    "lr_d_nan_002": ("nan_env:1@20", "COMPLETE", ("1", "TERMINATED", "MATERIAL_INVALID")),
    "lr_d_pose_001": ("bad_pose:0@20", "COMPLETE", ("0", "TERMINATED", "MATERIAL_INPUT_DOMAIN_GUARD")),
    "lr_d_segv_001": ("sigsegv@20", "FAILED_INFRA_CRASH_SIGNAL", None),
    "lr_d_hang_001": ("hang@20", "FAILED_INFRA_HANG", None),
    "lr_d_oom_001": ("device_oom@20", "FAILED_INFRA_OOM_DEVICE", None),
}


def probe_verdict(name, analysis):
    fault, status, episode = PROBES[name]
    checks = dict(supervisor_status=analysis["supervisor_status"] == status)
    if episode:
        env, cls, bit = episode
        rows = [r for r in analysis["episodes"]["rows"] if str(r["env"]) == env and r["cause"] == "terminated"]
        checks["faulted_env_terminated"] = any(r["end_class"] == cls and bit in (r["termination_names"] or [])
                                               for r in rows)
        others = [r for r in analysis["episodes"]["rows"] if str(r["env"]) != env]
        checks["other_env_not_terminated"] = all(r["cause"] != "terminated" for r in others)
    else:
        checks["env_exit_agrees"] = all(analysis["infrastructure"]["agrees_with_env_exit"])
    return dict(kind="fault_probe", fault=fault, expected_supervisor_status=status, checks=checks,
                verdict="PASS" if all(checks.values()) else "FAIL")


def slim(analysis):
    out = compact(analysis)
    rows = out["episodes"]["rows"]
    keep = [r for r in rows if r["end_class"] not in ("TRUNCATED_BY_FORCED_RESET", "TRUNCATED_RUN_END")]
    if len(rows) > 200:          # LR-B: hundreds of forced truncations -> counts only
        out["episodes"]["rows"] = keep
        out["episodes"]["rows_note"] = (f"{len(rows) - len(keep)} truncated (not evaluated) segments omitted; "
                                        "counts in per_env_classes")
    for a in out["attempts"]:
        a["telemetry"] = slim_telemetry(a.get("telemetry", {}))
        a["classification"].pop("stderr_tail", None)
        mr = a.get("mode_result") or {}
        if "reset_check_mismatch_detail" in mr:
            mr["reset_check_mismatch_detail"] = mr["reset_check_mismatch_detail"][:5]
    out["meta"] = {k: v for k, v in out["meta"].items() if k not in ("argv",)}
    out["infrastructure"]["env_exit"] = out["infrastructure"].get("env_exit")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra-json", action="append", default=[], help="key=path of a JSON file to embed")
    args = parser.parse_args()
    runs = {}
    for name, config, role in RUNS:
        run_dir = P14 / name
        if not run_dir.exists():
            runs[name] = dict(role=role, status="NOT_FOUND")
            continue
        full = analyze(run_dir, json.loads(Path(config).read_text()))
        result = slim(full)
        if name in PROBES:
            result["acceptance"] = probe_verdict(name, full)
        elif name.startswith(("smoke", "t45_smoke")):
            result["acceptance"] = dict(kind="smoke", verdict="N/A (code-path smoke, never evidence)",
                                        rollout_checks_for_reference=result["acceptance"])
        result["role"] = role
        result["config"] = str(Path(config).relative_to(ROOT))
        runs[name] = result
    evidence = dict(schema="rcs_rl_p1_4_evidence_v1", id="p1_4_001", runs=runs,
                    raw_output="artifacts/rl_ready/p1_4/<run>/ (supervisor.jsonl, telemetry.jsonl, attempt_NNN/ledger.jsonl)",
                    configs={"long_rollout_v1": str(V1.relative_to(ROOT)), "long_rollout_v2": str(V2.relative_to(ROOT))})
    for item in args.extra_json:
        key, _, path = item.partition("=")
        evidence[key] = json.loads(Path(path).read_text())
    with Path(args.out).open("x") as stream:
        json.dump(evidence, stream, indent=1, sort_keys=True, default=str)
        stream.write("\n")
    print(args.out, Path(args.out).stat().st_size)


if __name__ == "__main__":
    main()
