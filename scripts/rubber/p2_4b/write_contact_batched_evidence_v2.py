"""Compact evidence ``docs/evidence/rl_ready/p2_4_contact_batched_002.json`` of P2-4 task B2-v2 (RUBBER-CONT-09 under
``rubber_contact_tests_v2`` with the v2 law, and the v2-law batched RUBBER-CONT-11 cost rows) from the raw outputs in
``artifacts/rl_ready/p2_4b`` (reads only; never re-runs anything; refuses to overwrite).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4b/write_contact_batched_evidence_v2.py \\
        --gpu artifacts/rl_ready/p2_4b/gpu_cont09_v2_001 --smoke artifacts/rl_ready/p2_4b/gpu_cont09_v2_smoke_001 \\
        --cpu-tests artifacts/rl_ready/p2_4b/cpu_tests_v2_001 --out docs/evidence/rl_ready/p2_4_contact_batched_002.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COST_KEYS = ("row", "workload", "boundary", "N", "particles_per_env", "nodes_per_env", "control_hz", "material_dt_s",
             "substeps_per_control_step", "cameras", "recording", "robot_dof", "contact_law", "contacts",
             "cold_wall_s", "warm_wall_s_median", "warm_steps", "ms_per_substep_warm", "env_steps_per_s_warm",
             "aggregate_sim_time_per_wall_warm", "reset_p95_ms", "reset_median_ms", "reset_kind",
             "vram_peak_mb_process_cumulative", "memory_bytes_arrays", "physics_verdict", "gpu_label",
             "co_tenant_pids", "gpu", "gpu_index_physical", "driver", "warp", "commit", "dirty_tree",
             "protocol_sha256", "smoke_reduced_fidelity")
PER_SCALAR_FALSE_REJECT = 5.5e-6     # protocol RUBBER-CONT-09.calibration (eps_rel = 0, worst case of 4 noise laws)


def _tail(path):
    p = Path(path)
    return [line for line in p.read_text().strip().splitlines() if "passed" in line or "failed" in line][-1:] \
        if p.exists() else ["MISSING"]


def _gpu_util(sample, index):
    return [g.split(",")[-1].strip() for g in sample["gpus"] if g.split(",")[0].strip() == str(index)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--cpu-tests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    gpu = json.loads((ROOT/a.gpu/"result.json").read_text())
    smoke = json.loads((ROOT/a.smoke/"result.json").read_text())
    meta, c09 = gpu["meta"], gpu["cont09"]
    proto = ROOT/"configs"/"rubber"/"rubber_contact_tests_v2.json"
    sha = hashlib.sha256(proto.read_bytes()).hexdigest()
    if sha != meta["protocol_sha256"]:
        raise SystemExit("protocol changed since the run")
    b = c09["b_perturbation_band"]
    noisy = b["scalars"]-b["scalars_with_s_zero"]
    v1 = json.loads((ROOT/"docs"/"evidence"/"rl_ready"/"p2_4_contact_batched_001.json").read_text())
    v1_cost = {r["row"]: r for r in v1["RUBBER-CONT-11_batched_cost_rows"]}
    rows = [{k: r.get(k) for k in COST_KEYS} | {
        "co_tenant_gpu_util_pct_before_after_row (GPU-wide, includes this process)":
            _gpu_util(r["co_tenant_before_row"], meta["physical_gpu_index"])
            + _gpu_util(r["co_tenant_after_row"], meta["physical_gpu_index"]),
        "v1_law_row_gpu_cont09_001_ms_per_substep_warm": v1_cost.get(r["row"], {}).get("ms_per_substep_warm")}
        for r in gpu["cost_rows"]]
    cpu = ROOT/a.cpu_tests
    ev = {
        "schema": "rcs_rl_ready_evidence_v1", "id": "p2_4_contact_batched_002",
        "phase": "P2-4 rubber contact, batched path, task B2-v2 (RUBBER-CONT-09 under the v2 rule and the v2 law)",
        "adr": "docs/adr/0023-rubber-material-v1.md section 22.9",
        "report": "docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md section 9",
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
        "protocol": {"path": "configs/rubber/rubber_contact_tests_v2.json", "sha256": sha,
                     "thresholds": meta["thresholds"], "contact_law": meta["contact_law"]},
        "supersedes": "nothing: p2_4_contact_batched_001 (RUBBER-CONT-09 v1 FAIL, gpu_cont09_001) stays preserved",
        "script": "scripts/rubber/p2_4b/gpu_contact_batched_check_v2.py",
        "RUBBER-CONT-09": {
            "verdict": c09["verdict"], "run": str(a.gpu), "commit": meta["commit"], "dirty_tree": meta["dirty_tree"],
            "command": ("flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 "
                        "TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python " + meta["command"]),
            "wall_s": meta["wall_s"], "contact_law_built": c09["contact_law_built"],
            "options_receipt": c09["options_receipt"],
            "N": c09["N"], "particles_per_env": c09["particles_per_env"], "nodes_per_env": c09["nodes_per_env"],
            "substeps": c09["substeps"], "dt_s": c09["dt_s"], "control_steps_per_episode": c09["control_steps_per_episode"],
            "reach": c09["reach"], "squeeze_depth_A_m": c09["squeeze_depth_A_m"],
            "squeeze_depth_B_m": c09["squeeze_depth_B_m"], "episode_order": c09["episode_order"],
            "a_zero_rows": c09["a_zero_rows"],
            "b_perturbation_band": {k: v for k, v in b.items() if k != "rows_worst_first"}
            | {"rows_worst_first_8": b["rows_worst_first"][:8]},
            "c_audits": c09["c_audits"], "validity": c09["validity"],
            "fwer_note": {
                "scalars_checked": b["scalars"], "scalars_nondegenerate_s_gt_0": noisy,
                "protocol_calibration_bound_assumed_scalars": 648,
                "union_bound_all_scalars": b["scalars"]*PER_SCALAR_FALSE_REJECT,
                "union_bound_nondegenerate_scalars": noisy*PER_SCALAR_FALSE_REJECT,
                "reading": "the protocol's calibration text assumed <= 648 scalars; the declared "
                           "observables (markers, summary, group wrench, plane / box / pad / pending wrenches) have "
                           f"{b['scalars']} scalars in 126 rows ({b['scalars_with_s_zero']} bitwise deterministic, "
                           "s = 0, which cannot false-reject under H0 unless d > floor). The verdict rule does not "
                           "depend on this count; the family-wise H0 false-reject bound is "
                           f"{b['scalars']*PER_SCALAR_FALSE_REJECT:.2%} (all) / {noisy*PER_SCALAR_FALSE_REJECT:.2%} "
                           "(s > 0), not 0.4 %"},
            "diagnostic_not_verdict": c09["diagnostic_not_verdict"],
            "pad_force_z_env1_last_step_N": {"A": [p[-1][1] for p in c09["pad_force_z"]["A"]],
                                             "B": [p[-1][1] for p in c09["pad_force_z"]["B"]]},
            "walls_s": c09["walls_s"]},
        "smoke_run": {"run": str(a.smoke), "smoke_reduced_fidelity": smoke["meta"]["smoke_reduced_fidelity"],
                      "geometry_m": smoke["meta"]["geometry_m"], "n_t": smoke["meta"]["n_t"],
                      "verdict_of_script_check_not_protocol": smoke["cont09"]["verdict"],
                      "commit": smoke["meta"]["commit"]},
        "RUBBER-CONT-11_batched_cost_rows_v2_law": rows,
        "gpu_meta": {k: meta.get(k) for k in ("device_name", "physical_gpu_index", "driver", "warp", "python", "pid",
                                              "shared_gpu", "label_gpu", "co_tenant_pids", "free_mb_at_start",
                                              "started", "finished", "co_tenant_before", "co_tenant_after",
                                              "smoke_reduced_fidelity")},
        "erratum_p2_4_contact_batched_001": "report section 7.3 named PID 3093155 (712 MB, at the end) a second "
                                            "co-tenant of gpu_cont09_001; it is that run's own process "
                                            "(meta.pid 3093155). The SHARED_GPU label stands (PID 642659)",
        "cpu_suites": {"run": str(a.cpu_tests),
                       "cuda_visible_devices_empty": _tail(cpu/"cpu_empty.log"),
                       "cuda_visible_devices_0": _tail(cpu/"cpu_gpu0.log"),
                       "meta": json.loads((cpu/"meta.json").read_text()) if (cpu/"meta.json").exists() else None},
        "not_done": ["B3 MaterialCoupling adapter (strip-to-world wrenches, finger recipients)",
                     "P1 contact kernel cost", "X2 (friction-normal flutter, defect 2)", "independent review (R1)"],
    }
    for key in ("cuda_visible_devices_empty", "cuda_visible_devices_0"):
        m = re.search(r"(\d+) passed", " ".join(ev["cpu_suites"][key]))
        ev["cpu_suites"][key+"_passed"] = int(m.group(1)) if m else None
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out)


if __name__ == "__main__":
    main()
