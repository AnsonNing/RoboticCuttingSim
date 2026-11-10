"""Write ``docs/evidence/rl_ready/p2_3_visco_001.json`` from the P2-3 visco artifacts (formal CPU run, tangent check,
GPU cost rows) and the CPU suite counts (ADR 0023 §21.4 R3.9).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/write_visc_evidence.py --formal artifacts/rl_ready/p2_3/<run> \
        --tangent artifacts/rl_ready/p2_3/<dir> --cost artifacts/rl_ready/p2_3/<dir> \
        --cpu-tests '{"cuda_visible_devices_empty": "...", "cuda_visible_devices_0": "..."}' --out docs/evidence/...
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ("src/robotic_cutting_sim/tlmpm_rubber_v1.py", "src/robotic_cutting_sim/rubber_constitutive.py",
          "src/robotic_cutting_sim/batched/constitutive.py", "src/robotic_cutting_sim/batched/errors.py",
          "configs/rubber/rubber_visc_tests_v1.json", "configs/rubber/rubber_moving_grip_tests_v1.json",
          "configs/rubber/rubber_phys_tests_v4.json")


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", required=True)
    ap.add_argument("--tangent", required=True)
    ap.add_argument("--cost", default=None)
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--cpu-tests", default="{}")
    ap.add_argument("--extra", default="{}", help="JSON merged at the top level (notes, commits)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    formal = json.loads((ROOT/args.formal/"summary.json").read_text())
    tangent = json.loads((ROOT/args.tangent/"summary.json").read_text())
    res = formal["results"]
    verdicts = {k: v["verdict"] for k, v in res.items() if isinstance(v, dict) and "verdict" in v}
    ev = {"schema": "rcs_evidence_v1", "id": out.stem, "adr": "docs/adr/0023-rubber-material-v1.md section 21.2-21.6",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED (illustrative Prony sets SLS / P2 and the recipe "
                                "rubber_generic_visco_v1; not fitted to any real rubber data; RUBBER-VISC-03 b/c is a "
                                "synthetic dry run of the P2-5 fit / held-out split, not a calibration)",
          "protocol": {"path": formal["protocol"], "sha256": formal["protocol_sha256"]},
          "gate_tangent_check": {"artifact": args.tangent, "commit": tangent["commit"],
                                 "dirty_tree": tangent["dirty_tree"], "command": tangent["command"],
                                 "samples": tangent["samples"], "max_ratio": tangent["max_ratio"],
                                 "median_ratio": tangent["median_ratio"], "p99_ratio": tangent["p99_ratio"],
                                 "premise_isochoric_material_part": tangent["premise_isochoric_material_part"],
                                 "verdict": tangent["verdict"]},
          "formal": {"artifact": args.formal, "label": formal["label"], "commit": formal["commit"],
                     "dirty_tree": formal["dirty_tree"], "command": formal["command"], "device": formal["device"],
                     "workers": formal["workers"], "started": formal["started"], "finished": formal["finished"],
                     "wall_s": formal["wall_s"], "versions": formal["versions"], "verdicts": verdicts,
                     "jobs": formal["jobs"], "results": res},
          "cpu_tests": json.loads(args.cpu_tests),
          "frozen_unchanged_sha256": {p: sha(p) for p in FROZEN}}
    if args.smoke:
        sm = json.loads((ROOT/args.smoke/"summary.json").read_text())
        ev["smoke"] = {"artifact": args.smoke, "label": sm["label"], "commit": sm["commit"]}
    if args.cost:
        meta = json.loads((ROOT/args.cost/"meta.json").read_text())
        rows = json.loads((ROOT/args.cost/"rows.json").read_text())
        prof = json.loads((ROOT/args.cost/"profile.json").read_text()) if (ROOT/args.cost/"profile.json").exists() \
            else None
        keep = ("variant", "model", "M", "visco_energy", "N", "n_t", "resolution_profile", "particles_per_env",
                "control_hz", "material_dt_s", "substeps", "beta", "cameras", "recording", "cold_wall_s",
                "warm_wall_s_median", "ms_per_substep", "env_steps_per_s", "aggregate_sim_time_per_wall",
                "registry_state_bytes_per_particle", "energy_option_bytes_per_particle",
                "vram_peak_mb_process_cumulative", "physics_verdict", "guard_violations")
        ev["gpu_cost_visc10"] = {"artifact": args.cost, "label": "RUBBER-VISC-10 (reported, not scored)",
                                 "meta": {k: meta.get(k) for k in ("command", "device_name", "driver", "commit",
                                                                   "dirty_tree", "protocol_sha256", "gpu_sharing",
                                                                   "co_tenant_pids_seen", "started", "finished",
                                                                   "warp", "options", "cuda_visible_devices")},
                                 "co_tenant_samples": [meta.get("co_tenant_before")]+meta.get("co_tenant_samples", [])
                                 + [meta.get("co_tenant_after")],
                                 "rows": [{k: r.get(k) for k in keep} for r in rows],
                                 "stage_split": {k: {kk: v[kk] for kk in ("N", "n_t", "substeps", "gpu_ms_per_substep",
                                                                           "stress_pass_ms_per_substep")}
                                                 for k, v in (prof or {}).items()}}
    ev.update(json.loads(args.extra))
    out.write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps(verdicts))


if __name__ == "__main__":
    main()
