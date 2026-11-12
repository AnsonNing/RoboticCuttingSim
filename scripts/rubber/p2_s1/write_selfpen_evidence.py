"""Compact evidence ``docs/evidence/rl_ready/p2_s1_selfpen_001.json`` of task S1 (ADR 0023 §23.2 decision 2, §23.5) from
the raw outputs (reads only; never re-runs anything; refuses to overwrite): the GPU check ``rubber_selfpen_check_v1``
and the CPU suites in both CUDA modes.

    .venv/bin/python scripts/rubber/p2_s1/write_selfpen_evidence.py --gpu artifacts/rl_ready/p2_s1/gpu_selfpen_001 \\
        --cpu-tests artifacts/rl_ready/p2_s1/cpu_tests_s1_001 --out docs/evidence/rl_ready/p2_s1_selfpen_001.json
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_selfpen_check_v1.json"


def _tail(path, n=2):
    p = Path(path)
    return p.read_text().strip().splitlines()[-n:] if p.exists() else ["MISSING"]


def _r(x, n=4):
    return None if x is None else float(f"{x:.{n}g}")


def main():
    ap = argparse.ArgumentParser()
    for k in ("gpu", "cpu-tests", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    res = json.loads((ROOT/a.gpu/"result.json").read_text())
    meta, rows = res["meta"], res["rows"]
    cost = []
    for r in rows["COST"]:
        on, off = r["surface_grid_v1"], r["off_v1"]
        cost.append({"N": r["N"], "particles_per_env": on["particles_per_env"], "substeps": on["substeps"],
                     "dt_s": on["dt_s"], "control_hz": 50, "cameras": 0, "recording": False,
                     "graph_chunk_on": on["graph"]["graph_chunk_substeps"],
                     "graph_chunk_off": off["graph"]["graph_chunk_substeps"],
                     "ms_per_substep_on": _r(on["ms_per_substep"], 5), "ms_per_substep_off": _r(off["ms_per_substep"], 5),
                     "overhead_ms_per_substep": _r(r["overhead_ms_per_substep"]), "overhead_rel": _r(r["overhead_rel"]),
                     "env_steps_per_s_on": _r(on["env_steps_per_s"]), "env_steps_per_s_off": _r(off["env_steps_per_s"]),
                     "warm_walls_s_on": [_r(w) for w in on["warm_walls_s"]],
                     "warm_walls_s_off": [_r(w) for w in off["warm_walls_s"]],
                     "memory_bytes_on": on["memory_bytes"], "memory_bytes_off": off["memory_bytes"],
                     "detector_bytes": r["L2"]["detector_receipt"]["memory_bytes"],
                     "warm_cold": "1 cold + 5 warm control steps per arm, interleaved on/off; median warm"})
    l2 = {str(n): {k: v[k] for k in ("finite", "mass_unchanged", "audits_zero", "invalid_reason", "hits", "checks",
                                     "checks_expected", "blind_steps", "pass")}
          for n, v in rows["L2"]["per_N"].items()}
    trip = rows["TRIP"]
    nft = [{k: r.get(k) for k in ("scene", "scored", "particles", "substeps", "steps", "invalid_flags", "pass")}
           | {"hits": r["detector"]["hits"], "checks": r["detector"]["checks"],
              "first_hit_control_step": r["detector"]["first_hit_control_step"],
              "first_hit_substep": r["detector"]["first_hit_substep"], "blind_steps": r["detector"]["blind_steps"],
              "wall_s": _r(r["wall_s"])} for r in rows["NO_FALSE_TRIP"]["runs"]]
    ev = {"schema": "rcs_rl_ready_evidence_v1", "id": "p2_s1_selfpen_001",
          "work_package": "P2-4 completion task S1: self-penetration detector surface_grid_v1 (ADR 0023 section 23.2 "
                          "decision 2 and section 23.5; plan P2_REMAINING_PLAN.md section 2.2)",
          "report": "docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md section 12",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
          "self_contact_physics": "NOT_IMPLEMENTED_V1 (detector only, no force)",
          "commits": {"code_and_cpu_tests": "a1bdeac", "check_declaration": "5effd81", "gpu_script": "bca572d"},
          "declaration": {"file": str(PROTOCOL.relative_to(ROOT)),
                          "sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                          "pinned_in": "tests/test_rubber_s1_protocol.py"},
          "gpu_run": {"dir": a.gpu, "commit": meta["commit"], "dirty_tree": meta["dirty_tree"],
                      "protocol_sha256": meta["protocol_sha256"], "gpu": meta["gpu"],
                      "physical_gpu": meta["physical_gpu"], "warp": meta["warp"], "python": meta["python"],
                      "free_mb_at_start": meta["free_mb_at_start"], "gpu_label": meta["gpu_label"],
                      "co_tenant_pids": meta["co_tenant_pids"],
                      "co_tenant_samples": [{"t": s["t"], "gpu": s["gpu"], "apps": s["apps"]}
                                            for s in meta["co_tenant_samples"]],
                      "command": meta["command"], "started": meta["started"], "finished": meta["finished"],
                      "lock": "flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock (blocking)"},
          "verdict": res["verdict"],
          "cost_rows": cost,
          "L2": {"pass": rows["L2"]["pass"], "per_N": l2},
          "trip": {"pass": trip["pass"], "single_gpu": trip["single_gpu"], "single_cpu": trip["single_cpu"],
                   "batched_n3_gpu": trip["batched_n3_gpu"],
                   "note": "GPU P2G / MUSL use float atomics (order-nondeterministic), so the GPU hit time is reported "
                           "next to the CPU value, not scored"},
          "no_false_trip": {"pass": rows["NO_FALSE_TRIP"]["pass"], "runs": nft},
          "cpu_tests": {"dir": a.cpu_tests, "suites": "tests/test_rubber_*.py tests/test_live_rubber_material.py "
                                                     "tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py "
                                                     "tests/test_repo_naming.py tests/test_native_source_freeze.py "
                                                     "tests/test_wp12_phase5_frozen.py",
                        "cuda_visible_devices_empty": _tail(ROOT/a.cpu_tests/"cuda_none.log"),
                        "cuda_visible_devices_0": _tail(ROOT/a.cpu_tests/"cuda_0.log")},
          "limitations": [
              "detector only: a crossing is flagged (sticky bit 8, training mode terminates in P2-6), never resolved",
              "checks every 256 substeps (~1 ms) and at the end of each control step: a pass-through faster than about "
              "(t - s) / (2 dt_check) between two checks can be missed; blind_steps counts control steps where "
              "2 v_max dt_check >= t - s",
              "the RUBBER-PHYS-09 lambda 1.5 release scene trips bit 8 in the same control step as J_DOMAIN / CFL (the "
              "fold compresses distant material together); the bit order inside that control step is not resolved",
              "material parameters provisional; status SIMULATOR_ONLY_UNCALIBRATED"],
          "frozen": "unchanged: Phase 1 modules (batched/constitutive.py, errors.py, material_batch.py, "
                    "particle_contacts.py), configs/rl_benchmark/*, scripts/rl_ready/*, tlmpm_rubber_v1.py, "
                    "rubber_constitutive.py, tlmpm_rubber_visco_v1.py, rubber_visco_constitutive.py, "
                    "rubber_constraint_work.py, rubber protocols"}
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out)


if __name__ == "__main__":
    main()
