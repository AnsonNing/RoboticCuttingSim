#!/usr/bin/env python3
"""Compact evidence of the P1-6 R0 GPU smoke (``docs/evidence/rl_ready/p1_6_r0_<NNN>.json``) from a local
acceptance directory written by ``run_perf_acceptance_local.py`` (manifest, runs, judge verdict).

    .venv/bin/python scripts/rl_ready/make_p1_6_r0_evidence.py --acceptance artifacts/rl_ready/p1_6_r0/smoke_001 \\
        --extra extra.json --out docs/evidence/rl_ready/p1_6_r0_001.json
"""
import argparse
import json
from pathlib import Path


def compact_gate(g):
    value = g.get("value")
    if isinstance(value, list) and value and isinstance(value[0], dict) and "checks" in value[0]:
        value = [dict(run=x["run"], side=x["side"], seed=x["seed"], env=x["env"], checks=x["checks"],
                      values={k: v for k, v in x["values"].items() if k not in ("labelled", "labelled_native",
                                                                                 "samples")})
                 for x in value]
    return dict(id=g["id"], layer=g["layer"], status=g["status"], value=value, threshold=g.get("threshold"),
                detail=g.get("detail"), **({"affects": g["affects"]} if "affects" in g else {}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance", required=True)
    p.add_argument("--extra", default=None, help="JSON merged at top level (commands, tests, notes)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.acceptance)
    man = json.loads((root / "manifest.json").read_text())
    verdict = json.loads((root / "judge" / "verdict.json").read_text())
    runs = []
    for r in man["runs"]:
        res = json.loads(Path(r["result"]).read_text()) if r.get("result") else {}
        rounds = res.get("rounds") or []
        dumps = res.get("dumps") or []
        runs.append(dict(
            id=r["id"], side=r["side"], gate=r["gate"], mode=r["mode"], num_envs=r["num_envs"], seeds=r["seeds"],
            variant=res.get("variant"), rc=r["rc"], process_wall_s=r["wall_s"], status=res.get("status"),
            outer=r.get("outer"), clock=r.get("clock"), row=(res.get("rows") or [None])[-1],
            rounds=[dict(round=x.get("round"), env_seeds=x.get("env_seeds"), build_wall_s=x.get("build_wall_s"),
                         timing=x.get("timing"), reset_ms=x.get("reset_ms"), final_labelled=x.get("final_labelled"),
                         invalid_reason=x.get("invalid_reason"), guard_violations=x.get("guard_violations"),
                         device_hashgrids_after_build=x.get("device_hashgrids_after_build"),
                         device_hashgrids_after_release=x.get("device_hashgrids_after_release"),
                         particle_contact_mode=(x.get("particle_contact_receipt") or {}).get("mode"),
                         l2_candidate_reported_status=(x.get("l2_candidate_reported") or {}).get("status"),
                         evaluate_2_candidate_reported=[e.get("verdict") for e in
                                                        x.get("evaluate_2_frames_candidate_reported") or []],
                         samestate_rerun=(x.get("samestate") or {}).get("rerun")) for x in rounds],
            dumps=[dict(kind=d.get("kind"), path=d["path"], bytes=d["bytes"], sha256=d["sha256"]) for d in dumps],
            environment={k: (res.get("environment") or {}).get(k) for k in ("commit", "dirty", "package_root_inside_checkout",
                                                                             "harness_sha256", "python")},
            versions=res.get("versions"), gpu_sharing=res.get("gpu_sharing"), harness_wall_s=res.get("wall_s")))
    evidence = dict(
        schema="rcs_rl_p1_6_r0_evidence_v1", acceptance_dir=str(root), config=man["config"],
        config_sha256=man["config_sha256"], baseline=man["baseline"], candidate=man["candidate"],
        selection=man.get("selection"), cpu_tests=man.get("cpu_tests"), runs=runs,
        verdict=dict(layers=verdict["layers"], failed_gates=verdict["failed_gates"],
                     ratios=verdict["performance"]["ratios"], gates=[compact_gate(g) for g in verdict["gates"]],
                     rows=verdict["rows"], notes=verdict.get("notes"), limitations=verdict.get("limitations"),
                     judge=verdict["judge"], wall_s=verdict.get("wall_s")))
    if a.extra:
        evidence.update(json.loads(Path(a.extra).read_text()))
    out = Path(a.out)
    with out.open("x") as stream:
        json.dump(evidence, stream, indent=1, sort_keys=True, allow_nan=True)
        stream.write("\n")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
