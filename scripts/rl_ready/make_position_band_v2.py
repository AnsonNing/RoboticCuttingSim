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
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl import position_band as PB  # noqa: E402

ART = ROOT / "artifacts" / "rl_ready"
V1 = {512: (1.3910405714932552e-05, 3.8464872515711716e-05), 15: (0.00019569518905335448, 0.00046197226017502627)}
V1_FINAL = {512: (1.3910405714932552e-05, 3.8464872515711716e-05), 15: (6.268663952130912e-05, 0.000118195664666705)}
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
                     metric_m=(d.get("parity") or {}).get(PB.METRIC),
                     final_metric_m=((d.get("parity") or {}).get("final") or {}).get("max_position_diff_m"),
                     gpu_sharing=(d.get("telemetry") or {}).get(
                         "gpu_sharing"), final_codes_equal_native=((d.get("parity") or {}).get("final") or {}).get(
                         "codes_equal"), wall_s=d.get("wall_s"),
                     cuda_visible_devices=load(result.parent / "manifest.json").get("cuda_visible_devices"))
        (runs[seed] if ok else rejected).append(entry)
    return runs, rejected


def v1_status(seed, value, table=V1):
    lo, hi = table[seed]
    return "inside" if lo <= value <= hi else "above" if value > hi else "below"


def held_out():
    """(package, run, env, seed, metric or None, invalid, applicable, note) for every existing run."""
    rows = []
    for run in ("replay_s512_001", "band_replay_s512_001", "band_replay_s512_002", "band_replay_s512_003",
                "replay_s15_001", "band_replay_s15_001", "band_replay_s15_002", "band_replay_s15_003",
                "p1_2_single_ref_s512_001"):
        d = load(ART / "p1_0" / run / "result.json")
        rows.append(("P1-0 frozen single (band v1 runs + single ref)", f"p1_0/{run}", 0, d["seed"],
                     (d["parity"][PB.METRIC], d["parity"]["final"]["max_position_diff_m"]), False, True,
                     "single_env_headless, frozen LiveCuttingMaterial"))
    for run in ("smoke_replay_s512_001", "coldcache_replay_s512_001"):
        d = load(ART / "p1_0" / run / "result.json")
        rows.append(("P1-0 frozen single (band v1 runs + single ref)", f"p1_0/{run}", 0, d["seed"],
                     (d["parity"][PB.METRIC], (d["parity"].get("final") or {}).get("max_position_diff_m")), False,
                     False,
                     f"{d.get('steps_run')} steps only"))
    for run in ("m0_s512_001", "m0_s15_001", "m0_s512_postfix_001", "m0_s15_postfix_001"):
        d = load(ART / "p1_2" / run / "result.json")
        rows.append(("P1-2 M0 batched N=1 frozen_v1", f"p1_2/{run}", 0, d["seed"],
                     (d["parity_vs_native"][PB.METRIC], d["parity_vs_native"][PB.FINAL_METRIC]), False, True,
                     "pre-fix" if "postfix" not in run else "post-fix"))
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
            pv = env["parity_vs_native"]
            rows.append((p13, f"p1_3/{run}", env["env"], env["seed"], (pv[PB.METRIC], pv[PB.FINAL_METRIC]), False,
                         True, "alias_free_v1"))
    for run in INVALID_MULTI_GRID:
        d = load(ART / run / "result.json")
        for variant, res in d["results"].items():
            for env in res["envs"]:
                pv = env["parity_vs_native"]
                rows.append((p13 + " (INVALID_MULTI_GRID_PROCESS)", run, env["env"], env["seed"],
                             (pv[PB.METRIC], pv[PB.FINAL_METRIC]), True, True,
                             f"{variant}; two variants in one process (Warp descriptor hazard)"))
    return rows


def classify_all(reference):
    out = []
    for package, run, env, seed, values, invalid, applicable, note in held_out():
        metric, final = values if values is not None else (None, None)
        row = dict(package=package, run=run, env=env, seed=seed, note=note)
        for key, value, table in ((PB.METRIC, metric, V1), (PB.FINAL_METRIC, final, V1_FINAL)):
            band = reference["statistics"][key]["seeds"][str(seed)]
            row[key] = dict(value_m=value, band_v1_status=None if value is None or not applicable
                            else v1_status(seed, value, table),
                            band_v2_class=PB.classify(seed, value, reference, invalid=invalid, applicable=applicable,
                                                      metric=key),
                            band_v2_upper_m=band["upper_m"],
                            ratio_to_upper=None if value is None else value / band["upper_m"])
        out.append(row)
    aggregates = {}
    for key in PB.METRICS:
        packages = {}
        for row in out:
            packages.setdefault(row["package"], []).append(row[key]["band_v2_class"])
        aggregates[key] = {k: PB.aggregate(v) for k, v in packages.items()}
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
    reference = dict(schema=PB.REFERENCE_SCHEMA, id=PB.BAND_ID, primary_metric=PB.METRIC,
                     judge_metric=PB.JUDGE_METRIC, method=str(PB.METHOD_PATH.relative_to(ROOT)),
                     method_digest=PB.method_digest(method),
                     statistics={key: dict(seeds={str(seed): PB.derive([r[field] for r in rows], alpha=alpha)
                                                  for seed, rows in runs.items()})
                                 for key, field in ((PB.METRIC, "metric_m"), (PB.FINAL_METRIC, "final_metric_m"))},
                     reference_runs={str(seed): rows for seed, rows in runs.items()}, rejected_runs=rejected,
                     classes=list(PB.CLASSES),
                     usage="robotic_cutting_sim.rl.position_band.classify(seed, metric_m) -> class; "
                           "aggregate(classes) -> MET / NOT_MET (binomial rule)")
    print(json.dumps({key: {s: {k: v for k, v in b.items() if k != "values_m"} for s, b in st["seeds"].items()}
                      for key, st in reference["statistics"].items()}, indent=1))
    rows, aggregates = classify_all(reference)
    for row in rows:
        a, b = row[PB.METRIC], row[PB.FINAL_METRIC]
        if a["value_m"] is not None and b["value_m"] is not None:
            print(f"{row['run']:<45} e{row['env']} s{row['seed']:<4} over-time {a['value_m']:.3e} v1 "
                  f"{a['band_v1_status']:<6} v2 {a['band_v2_class']:<16} final {b['value_m']:.3e} v1 "
                  f"{b['band_v1_status']:<6} v2 {b['band_v2_class']}")
    print(json.dumps(aggregates, indent=1))
    if args.dry_run:
        return
    with open(args.reference_out, "x") as stream:
        json.dump(reference, stream, indent=1)
        stream.write("\n")
    # judge view in the method file (derived keys, excluded from the method digest)
    text = PB.METHOD_PATH.read_text()
    method = json.loads(text)
    method["position_band_m"] = PB.judge_table(reference)
    method["position_band_m_statistic"] = PB.JUDGE_METRIC
    method["position_band_m_by_statistic"] = {key: {s: {"high_m": b["upper_m"], "low_descriptive_m": b["lower_m"]}
                                                     for s, b in st["seeds"].items()}
                                              for key, st in reference["statistics"].items()}
    method["derived"] = dict(status="DERIVED", version=PB.BAND_ID, reference=str(PB.REFERENCE_PATH.relative_to(ROOT)),
                             method_digest=reference["method_digest"],
                             runs={s: [r["run"] for r in rows_] for s, rows_ in runs.items()},
                             summary={key: {s: {k: b[k] for k in ("n", "kind", "upper_m", "lower_m", "k",
                                                                  "nominal_false_reject_rate", "shapiro_wilk_p",
                                                                  "geometric_mean_m", "min_m", "max_m")}
                                            for s, b in st["seeds"].items()}
                                      for key, st in reference["statistics"].items()})
    method["version"] = PB.BAND_ID
    method["status"] = "DERIVED_FROM_PRE_DECLARED_METHOD"
    PB.METHOD_PATH.write_text(json.dumps(method, indent=1, ensure_ascii=False) + "\n")
    if args.classification_out:
        with open(args.classification_out, "x") as stream:
            json.dump(dict(schema="rcs_rl_position_band_v2_classification_v1", reference=reference,
                           rows=rows, aggregates=aggregates), stream, indent=1)


if __name__ == "__main__":
    main()
