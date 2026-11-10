"""Write ``docs/evidence/rl_ready/p2_3_visco_002.json`` (protocol ``rubber_visc_tests_v2``, ADR 0023 §21.7) from the
formal v2 run, its smoke run, the DIAGNOSTIC re-analysis of the v1 data and the CPU suite logs. The v1 FAILs are
recorded as preserved results (``p2_3_visco_001.json`` is not modified).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/write_visc_evidence_v2.py \
        --formal artifacts/rl_ready/p2_3/visc_v2_001 --smoke artifacts/rl_ready/p2_3/visc_v2_smoke_001 \
        --reanalysis artifacts/rl_ready/p2_3/visc_diag_v1_reanalysis_001 --cpu-tests artifacts/rl_ready/p2_3/cpu_tests_002 \
        --extra '{...}' --out docs/evidence/rl_ready/p2_3_visco_002.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ("src/robotic_cutting_sim/tlmpm_rubber_v1.py", "src/robotic_cutting_sim/rubber_constitutive.py",
          "src/robotic_cutting_sim/batched/constitutive.py", "src/robotic_cutting_sim/batched/errors.py",
          "configs/rubber/rubber_visc_tests_v1.json", "configs/rubber/rubber_visc_tests_v2.json",
          "configs/rubber/rubber_moving_grip_tests_v1.json", "configs/rubber/rubber_phys_tests_v4.json")


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def _tail(path):
    lines = [ln for ln in (ROOT/path).read_text().splitlines() if ln.strip()]
    return " | ".join(lines[-2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", required=True)
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--reanalysis", default=None)
    ap.add_argument("--cpu-tests", default=None, help="directory with pytest_hidden.log / pytest_gpu0.log")
    ap.add_argument("--extra", default="{}", help="JSON merged at the top level (commits, decision, not_run, ...)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    formal = json.loads((ROOT/args.formal/"summary.json").read_text())
    res = formal["results"]
    verdicts = {k: v["verdict"] for k, v in res.items() if isinstance(v, dict) and "verdict" in v}
    v1 = json.loads((ROOT/"docs"/"evidence"/"rl_ready"/"p2_3_visco_001.json").read_text())
    ev = {"schema": "rcs_evidence_v1", "id": out.stem, "adr": "docs/adr/0023-rubber-material-v1.md section 21.7",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED (illustrative Prony set P2; not fitted to any real rubber "
                                "data; RUBBER-VISC-03 b/c is a synthetic dry run of the P2-5 fit / held-out split, not a "
                                "calibration)",
          "protocol": {"path": formal["protocol"], "sha256": formal["protocol_sha256"],
                       "supersedes": {"path": "configs/rubber/rubber_visc_tests_v1.json",
                                      "sha256": formal["protocol_v1_sha256"], "scope": "RUBBER-VISC-03 and -05 only"}},
          "v1_results_preserved": {"evidence": "docs/evidence/rl_ready/p2_3_visco_001.json",
                                   "artifact": v1["formal"]["artifact"], "commit": v1["formal"]["commit"],
                                   "verdicts": v1["formal"]["verdicts"],
                                   "note": "the v1 FAILs of RUBBER-VISC-03 and -05 remain the v1 results; v2 adds results"},
          "formal": {"artifact": args.formal, "label": formal["label"], "commit": formal["commit"],
                     "dirty_tree": formal["dirty_tree"], "command": formal["command"], "device": formal["device"],
                     "workers": formal["workers"], "started": formal["started"], "finished": formal["finished"],
                     "wall_s": formal["wall_s"], "versions": formal["versions"], "verdicts": verdicts,
                     "validity_dynamics_identity_with_v1": res["validity_dynamics_identity_with_v1"]["status"],
                     "jobs": formal["jobs"], "results": res},
          "frozen_unchanged_sha256": {p: sha(p) for p in FROZEN}}
    if args.smoke:
        sm = json.loads((ROOT/args.smoke/"summary.json").read_text())
        ev["smoke"] = {"artifact": args.smoke, "label": sm["label"], "commit": sm["commit"],
                       "dirty_tree": sm["dirty_tree"], "command": sm["command"],
                       "identity_first_3_steps": sm["results"]["validity_dynamics_identity_with_v1"]["status"]}
    if args.reanalysis:
        ra = json.loads((ROOT/args.reanalysis/"summary.json").read_text())
        ev["diagnostic_reanalysis_v1"] = {"artifact": args.reanalysis, "label": ra["label"], "commit": ra["commit"],
                                          "command": ra["command"], "rates": ra["rates"],
                                          "identification_on_corrected_v1_data": ra["identification_on_corrected_v1_data"],
                                          "cantilever_books_diag_002": ra["cantilever_books_diag_002"]}
    if args.cpu_tests:
        ev["cpu_tests"] = {"suites": "tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_batched_*.py",
                           "artifact": args.cpu_tests,
                           "cuda_visible_devices_empty": _tail(Path(args.cpu_tests)/"pytest_hidden.log"),
                           "cuda_visible_devices_0": _tail(Path(args.cpu_tests)/"pytest_gpu0.log")}
    ev.update(json.loads(args.extra))
    out.write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps(verdicts))


if __name__ == "__main__":
    main()
