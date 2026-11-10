#!/usr/bin/env python3
"""Review fix F2: derive position band v2 from the reference runs and re-classify every P1-0 / P1-2 / P1-3a run.

Method (pre-declared): ``configs/rl_benchmark/position_band_v2.json``. Reference runs: every
``artifacts/rl_ready/review_fix1/band_v2_s<seed>_g<gpu>_NNN`` with status COMPLETE, 1200 steps and invariants PASS
(none dropped for its value). Writes (never overwrites):

* ``--reference-out`` (default ``configs/rl_benchmark/position_band_v2_reference.json``, schema
  ``rcs_rl_position_band_reference_v2``): per-seed band values for consumers (``rl.position_band``);
* ``--classification-out``: the re-classification of the held-out runs (band v1 and band v2 status per run,
  per-package aggregate).

``PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_position_band_v2.py --classification-out <json>``
(``--dry-run`` prints without writing).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl import position_band as PB  # noqa: E402

ART = ROOT / "artifacts" / "rl_ready"
V1 = {512: (1.3910405714932552e-05, 3.8464872515711716e-05), 15: (0.00019569518905335448, 0.00046197226017502627)}
INVALID_MULTI_GRID = ("p1_3/episode_512_001", "p1_3/episode_15_001")


def load(path):
    return json.loads(Path(path).read_text())


def reference_runs():
    runs = {512: [], 15: []}
    rejected = []
    for result in sorted((ART / "review_fix1").glob("band_v2_s*_g*_[0-9][0-9][0-9]/result.json")):
        name = result.parent.name
        seed = int(name.split("_")[2][1:])
        d = load(result)
        ok = (d.get("status") == "COMPLETE" and d.get("steps_run") == 1200
              and (d.get("invariants") or {}).get("status") == "PASS")
        entry = dict(run=f"review_fix1/{name}", seed=seed, status=d.get("status"), steps_run=d.get("steps_run"),
                     metric_m=(d.get("parity") or {}).get(PB.METRIC), gpu_sharing=(d.get("telemetry") or {}).get(
                         "gpu_sharing"), final_codes_equal_native=((d.get("parity") or {}).get("final") or {}).get(
                         "codes_equal"), wall_s=d.get("wall_s"),
                     cuda_visible_devices=load(result.parent / "manifest.json").get("cuda_visible_devices"))
        (runs[seed] if ok else rejected).append(entry)
    return runs, rejected


def v1_status(seed, value):
    lo, hi = V1[seed]
    return "inside" if lo <= value <= hi else "above" if value > hi else "below"


def held_out():
    """(package, run, env, seed, metric or None, invalid, applicable, note) for every existing run."""
    rows = []
    for run in ("replay_s512_001", "band_replay_s512_001", "band_replay_s512_002", "band_replay_s512_003",
                "replay_s15_001", "band_replay_s15_001", "band_replay_s15_002", "band_replay_s15_003",
                "p1_2_single_ref_s512_001"):
        d = load(ART / "p1_0" / run / "result.json")
        rows.append(("P1-0 frozen single (band v1 runs + single ref)", f"p1_0/{run}", 0, d["seed"],
                     d["parity"][PB.METRIC], False, True, "single_env_headless, frozen LiveCuttingMaterial"))
    for run in ("smoke_replay_s512_001", "coldcache_replay_s512_001"):
        d = load(ART / "p1_0" / run / "result.json")
        rows.append(("P1-0 frozen single (band v1 runs + single ref)", f"p1_0/{run}", 0, d["seed"],
                     d["parity"][PB.METRIC], False, False, f"{d.get('steps_run')} steps only"))
    for run in ("m0_s512_001", "m0_s15_001", "m0_s512_postfix_001", "m0_s15_postfix_001"):
        d = load(ART / "p1_2" / run / "result.json")
        rows.append(("P1-2 M0 batched N=1 frozen_v1", f"p1_2/{run}", 0, d["seed"],
                     d["parity_vs_native"][PB.METRIC], False, True, "pre-fix" if "postfix" not in run else "post-fix"))
    rows.append(("P1-2 M0 batched N=1 frozen_v1", "p1_2/smoke_m0_001", 0, 512, None, False, False, "smoke (short)"))
    for run in ("m1_n4_001", "m1_n4_postfix_001"):
        for e, seed in enumerate((512, 15)):
            rows.append(("P1-2 M1/M2 (no Native position comparison recorded)", f"p1_2/{run}", e, seed, None, False,
                         True, "replay env; final codes equal Native; positions not compared"))
        for e, seed in ((2, 512), (3, 15)):
            rows.append(("P1-2 M1/M2 (no Native position comparison recorded)", f"p1_2/{run}", e, seed, None, False,
                         False, "policy-driven env (ideal tracking of commanded targets), not a Native replay"))
    for run, episodes in (("m2_n1_001", [(0, 512)]), ("m2_n1_002", [(0, 15)]),
                          ("m2_n8_001", [(e, 512 if e % 2 == 0 else 15) for e in range(8)] * 3),
                          ("m2_n16_001", [(e, 512 if e % 2 == 0 else 15) for e in range(16)])):
        for e, seed in episodes:
            rows.append(("P1-2 M1/M2 (no Native position comparison recorded)", f"p1_2/{run}", e, seed, None, False,
                         True, "episode; final codes equal Native; positions not compared"))
    p13 = "P1-3a alias_free_v1 batched"
    for run in ("episode_solo_512_001", "episode_solo_15_001", "episode_512_15_512_15_001",
                "hardened_episode_512_001", "hardened_episode_15_001"):
        d = load(ART / "p1_3" / run / "result.json")
        for env in d["results"]["alias_free_v1"]["envs"]:
            rows.append((p13, f"p1_3/{run}", env["env"], env["seed"], env["parity_vs_native"][PB.METRIC], False,
                         True, "alias_free_v1"))
    for run in INVALID_MULTI_GRID:
        d = load(ART / run / "result.json")
        for variant, res in d["results"].items():
            for env in res["envs"]:
                rows.append((p13 + " (INVALID_MULTI_GRID_PROCESS)", run, env["env"], env["seed"],
                             env["parity_vs_native"][PB.METRIC], True, True,
                             f"{variant}; two variants in one process (Warp descriptor hazard)"))
    return rows


def classify_all(reference):
    out = []
    for package, run, env, seed, metric, invalid, applicable, note in held_out():
        cls = PB.classify(seed, metric, reference, invalid=invalid, applicable=applicable)
        band = reference["seeds"][str(seed)]
        out.append(dict(package=package, run=run, env=env, seed=seed, metric_m=metric,
                        band_v1_status=None if metric is None or not applicable else v1_status(seed, metric),
                        band_v2_class=cls, band_v2_upper_m=band["upper_m"],
                        ratio_to_upper=None if metric is None else metric / band["upper_m"], note=note))
    packages = {}
    for row in out:
        packages.setdefault(row["package"], []).append(row["band_v2_class"])
    aggregates = {k: PB.aggregate(v) for k, v in packages.items()}
    return out, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-out", default=str(PB.REFERENCE_PATH))
    parser.add_argument("--classification-out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    method = PB.load_method()
    alpha = method["tolerance"]["alpha"]
    runs, rejected = reference_runs()
    reference = dict(schema=PB.REFERENCE_SCHEMA, id=PB.BAND_ID, metric=PB.METRIC,
                     method=str(PB.METHOD_PATH.relative_to(ROOT)),
                     method_sha256=hashlib.sha256(PB.METHOD_PATH.read_bytes()).hexdigest(),
                     seeds={str(seed): PB.derive([r["metric_m"] for r in rows], alpha=alpha)
                            for seed, rows in runs.items()},
                     reference_runs={str(seed): rows for seed, rows in runs.items()}, rejected_runs=rejected,
                     classes=list(PB.CLASSES),
                     usage="robotic_cutting_sim.rl.position_band.classify(seed, metric_m) -> class; "
                           "aggregate(classes) -> MET / NOT_MET (binomial rule)")
    print(json.dumps({s: {k: v for k, v in b.items() if k != "values_m"} for s, b in reference["seeds"].items()},
                     indent=1))
    rows, aggregates = classify_all(reference)
    for row in rows:
        if row["metric_m"] is not None:
            print(f"{row['run']:<45} env {row['env']} s{row['seed']:<4} {row['metric_m']:.4e} "
                  f"v1 {row['band_v1_status']:<6} v2 {row['band_v2_class']}")
    print(json.dumps(aggregates, indent=1))
    if args.dry_run:
        return
    with open(args.reference_out, "x") as stream:
        json.dump(reference, stream, indent=1)
        stream.write("\n")
    if args.classification_out:
        with open(args.classification_out, "x") as stream:
            json.dump(dict(schema="rcs_rl_position_band_v2_classification_v1", reference=reference,
                           rows=rows, aggregates=aggregates), stream, indent=1)


if __name__ == "__main__":
    main()
