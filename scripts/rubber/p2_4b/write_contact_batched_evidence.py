"""Compact evidence ``docs/evidence/rl_ready/p2_4_contact_batched_001.json`` of P2-4 B1 / B2 (ADR 0023 §22.2, §22.5)
from the raw outputs in ``artifacts/rl_ready/p2_4b`` (reads only; never re-runs anything; refuses to overwrite).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4b/write_contact_batched_evidence.py \\
        --cont08 artifacts/rl_ready/p2_4b/cont08_cpu_001 --gpu artifacts/rl_ready/p2_4b/gpu_cont09_001 \\
        --cpu-tests artifacts/rl_ready/p2_4b/cpu_tests_001 --out docs/evidence/rl_ready/p2_4_contact_batched_001.json
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COST_KEYS = ("row", "workload", "boundary", "N", "particles_per_env", "nodes_per_env", "control_hz", "material_dt_s",
             "substeps_per_control_step", "cameras", "recording", "robot_dof", "contacts", "cold_wall_s",
             "warm_wall_s_median", "warm_steps", "ms_per_substep_warm", "env_steps_per_s_warm",
             "aggregate_sim_time_per_wall_warm", "reset_p95_ms", "reset_median_ms", "reset_kind",
             "vram_peak_mb_process_cumulative", "memory_bytes_arrays", "physics_verdict", "gpu_label", "gpu",
             "gpu_index_physical", "driver", "commit", "dirty_tree", "protocol_sha256", "smoke_reduced_fidelity")


def _tail(path, n=2):
    p = Path(path)
    return [line for line in p.read_text().strip().splitlines() if "passed" in line or "failed" in line][-n:] \
        if p.exists() else ["MISSING"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cont08", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cpu-tests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    c08 = ROOT/a.cont08
    meta08 = json.loads((c08/"meta.json").read_text())
    suite = ET.parse(c08/"junit.xml").getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    cases = [{"name": tc.get("name"), "outcome": ("failed" if tc.find("failure") is not None or tc.find("error")
                                                  is not None else "skipped" if tc.find("skipped") is not None
                                                  else "passed")} for tc in suite.iter("testcase")]
    n_pass = sum(c["outcome"] == "passed" for c in cases)
    gpu = json.loads((ROOT/a.gpu/"result.json").read_text())
    c09 = gpu["cont09"]
    diag = c09["diagnostic_not_verdict"]
    proto = ROOT/"configs"/"rubber"/"rubber_contact_tests_v1.json"
    cpu = ROOT/a.cpu_tests
    ev = {
        "schema": "rcs_rl_ready_evidence_v1", "id": "p2_4_contact_batched_001",
        "phase": "P2-4 rubber contact, batched path (tasks B1 / B2)",
        "adr": "docs/adr/0023-rubber-material-v1.md sections 22.2, 22.5, 22.8",
        "report": "docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md section 7",
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
        "protocol": {"path": "configs/rubber/rubber_contact_tests_v1.json",
                     "sha256": hashlib.sha256(proto.read_bytes()).hexdigest()},
        "implementation": {
            "solver": "batched/rubber_contact_batch.BatchedTLMPMRubberContact (tlmpm_rubber_contact_v1_batch)",
            "law_code": "the single solver's TLMPMRubberContact._launch_contact bound unchanged; rubber_contact "
                        "kernels shared (node_penalty_contact, contact_work_end, node_displacement, slider_update, "
                        "contact_energy)",
            "frictions": "params columns friction_table / friction_pad (strided view)",
            "outputs": "plane_wrench [N,P,6], box_wrench [N,F,B,6], pad_wrench [N,G*B,6] (strip-local), "
                       "pending_wrench [N,6] = 0; world transform + recipients = task B3 (not done)",
            "contact_off": "lives without a ContactSpec build the unchanged BatchedTLMPMRubber"},
        "RUBBER-CONT-08": {
            "verdict": "PASS" if cases and n_pass == len(cases) else "FAIL",
            "run": str(a.cont08), "commit": meta08["commit"], "dirty_tree": meta08["dirty_tree"],
            "command": meta08["command"], "warp": meta08["warp"], "tests": len(cases), "passed": n_pass,
            "cases": cases},
        "RUBBER-CONT-09": {
            "verdict": c09["verdict"], "run": str(a.gpu), "commit": gpu["meta"]["commit"],
            "dirty_tree": gpu["meta"]["dirty_tree"], "command": gpu["meta"]["command"],
            "N": c09["N"], "particles_per_env": c09["particles_per_env"], "substeps": c09["substeps"],
            "dt_s": c09["dt_s"], "control_steps_per_episode": c09["control_steps_per_episode"],
            "a_zero_rows": {"pass": c09["a_zero_rows"]["pass"],
                            "env1_contact_rows_nonzero_every_step": c09["a_zero_rows"][
                                "env1_contact_rows_nonzero_every_step"]},
            "b_perturbation_band": {k: c09["b_perturbation_band"][k] for k in
                                    ("pass", "env1_differs_A_vs_B", "repeat_A_vs_Aprime_bitwise_identical_all_envs",
                                     "rows_total", "rows_failing")}
            | {"failing_rows": [r for r in c09["b_perturbation_band"]["rows_worst_first"] if not r["pass"]]},
            "c_audits": c09["c_audits"],
            "diagnostic_not_verdict": {
                "note": diag["note"],
                "repeat_envelope_3A_2B": {k: diag["repeat_envelope_3A_2B"][k] for k in
                                          ("repeats_A", "repeats_B", "rows", "rows_within", "rows_outside")},
                "aggregated_over_steps_pass_all": diag["aggregated_over_steps"]["pass_all"],
                "aggregated_failing": [r for r in diag["aggregated_over_steps"]["rows"] if not r["pass"]],
                "zero_rows_including_diagnostic_episodes": diag["zero_rows_including_diagnostic_episodes"]},
            "pad_force_z_env1_A_last_step_N": c09["pad_force_z"]["A"][-1][1]},
        "RUBBER-CONT-11_batched_cost_rows": [{k: r.get(k) for k in COST_KEYS} for r in gpu["cost_rows"]],
        "gpu_meta": {k: gpu["meta"].get(k) for k in ("device_name", "physical_gpu_index", "driver", "warp",
                                                    "shared_gpu", "label_gpu", "free_mb_at_start", "started",
                                                    "finished", "co_tenant_before", "co_tenant_after",
                                                    "smoke_reduced_fidelity")},
        "cpu_suites": {"run": str(a.cpu_tests),
                       "cuda_visible_devices_empty": _tail(cpu/"cpu_empty.log"),
                       "cuda_visible_devices_0": _tail(cpu/"cpu_gpu0.log"),
                       "meta": json.loads((cpu/"meta.json").read_text()) if (cpu/"meta.json").exists() else None},
        "not_done": ["B3 MaterialCoupling adapter (strip-to-world wrenches, finger recipients)",
                     "P1 contact kernel cost (frame table, bounding-volume reject)", "S1 self-penetration detector",
                     "independent review (R1)"],
    }
    for key in ("cuda_visible_devices_empty", "cuda_visible_devices_0"):
        m = re.search(r"(\d+) passed", " ".join(ev["cpu_suites"][key]))
        ev["cpu_suites"][key+"_passed"] = int(m.group(1)) if m else None
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out)


if __name__ == "__main__":
    main()
