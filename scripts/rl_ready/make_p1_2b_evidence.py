"""Assemble the compact P1-2b evidence (``docs/evidence/rl_ready/p1_2b_001.json``) from the raw run outputs.

    PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_2b_evidence.py --runs smoke_a_n2_001 smoke_b1_n2_001 \
        --guard guard_domain_check_001 --tests tests.json --out docs/evidence/rl_ready/p1_2b_001.json

``--tests``: a JSON file with the recorded test counts per runtime (command, passed/failed/skipped, wall).
Raw outputs stay in ``artifacts/rl_ready/p1_2b/`` (not committed).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts/rl_ready/p1_2b"


def compact_row(name):
    path = RAW / name / "row.json"
    if not path.exists():
        return dict(run=name, status="NOT_RUN", reason="row.json missing")
    row = json.loads(path.read_text())
    t = row.get("telemetry") or {}
    warm = row.get("telemetry_warm") or {}
    contract = row.get("contract") or {}
    return dict(
        run=name, status=row.get("status"), backend=row.get("backend"), N=row.get("N"), tasks=row.get("tasks"),
        tool_mount=row.get("tool_mount"), particles_per_env=row.get("particles_per_env"),
        control_hz=row.get("control_hz"), material_dt_s=row.get("material_dt_s"),
        material_substeps_per_control_step=row.get("material_substeps_per_control_step"), cameras=row.get("cameras"),
        recording=row.get("recording"), evaluation_stream=row.get("evaluation_stream"),
        evaluation_stream_device_bytes=row.get("evaluation_stream_device_bytes"), strict_mode=row.get("strict_mode"),
        steps=row.get("steps"), reset_step=row.get("reset_step"), reset_envs=row.get("reset_envs"),
        cold=row.get("cold"), timing=row.get("timing"), warm_cold="warm steps exclude the first step",
        gpu=row.get("gpu"), gpu_sharing=row.get("gpu_sharing"), vram_peak_own_mib=row.get("vram_peak_own_mib"),
        foreign_sm_pct=t.get("foreign_sm_pct"), foreign_sm_pct_warm=warm.get("foreign_sm_pct"),
        foreign_compute_pids=t.get("foreign_compute_pids"), hashgrids=row.get("hashgrids"),
        fidelity=row.get("fidelity"), contract=dict(status=contract.get("status"), clauses=contract.get("clauses"),
                                                     final_step_count=contract.get("final_step_count"),
                                                     termination_reason=contract.get("termination_reason"),
                                                     evaluation_mode=contract.get("evaluation_mode")),
        evaluation=row.get("evaluation"), termination_reason_final=row.get("termination_reason_final"),
        input_guard_reason_final=row.get("input_guard_reason_final"), commit=row.get("commit"),
        error=row.get("error"), error_class=row.get("error_class"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--guard", default="guard_domain_check_001")
    parser.add_argument("--tests", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    guard_path = RAW / args.guard / "guard_domain_check.json"
    guard = json.loads(guard_path.read_text()) if guard_path.exists() else None
    if guard is not None:
        guard = dict(status=guard["status"], config=guard["config"], workload_a_replay=guard["workload_a_replay"],
                     workload_b={k: {e: v for e, v in r.items() if e != "domain_local_frame"} if isinstance(r, dict)
                                 else r for k, r in guard["workload_b"].items()}, run=args.guard)
    evidence = dict(work_package="P1-2b", raw_root="artifacts/rl_ready/p1_2b (not committed)",
                    rows=[compact_row(name) for name in args.runs], guard_domain_check=guard,
                    tests=json.loads(Path(args.tests).read_text()) if args.tests else None)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists (never overwritten)")
    out.write_text(json.dumps(evidence, indent=1, default=str) + "\n")
    print(out)


if __name__ == "__main__":
    main()
