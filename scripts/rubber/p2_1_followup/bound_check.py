"""Check of the stable_step_guard_v1 bound as implemented (rubber_constitutive.wave_modulus_bound_np, the float64
reference of the Warp function) against (1) the exact reference-frame acoustic tensor maximum (finite differences of
rubber_reference.pk1, 600 random directions + axes) for 300 random F with principal stretches in [0.3, 3], and
(2) the exact discrete-operator stiffening ratio r(F) of stability_spectrum.py (periodic block).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_1_followup/bound_check.py \
        --spectrum artifacts/rl_ready/p2_1/followup_stability_spectrum_n4_001/stability_spectrum.json --out <file.json>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robotic_cutting_sim import rubber_constitutive as rc  # noqa: E402
from stability_spectrum import acoustic_exact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    mu, kappa = 1.0, 20.0
    rng = np.random.default_rng(5)
    ratios = []
    for k in range(300):
        Q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        F = Q1@np.diag(np.exp(rng.uniform(math.log(0.3), math.log(3.0), 3)))@Q2
        if np.linalg.det(F) < 0:
            F = -F
        if np.linalg.det(F) < rc.J_EVAL_FLOOR:
            continue                         # below the evaluation floor the stress itself is floored (J domain)
        ratios.append(rc.wave_modulus_bound_np(F, mu, kappa)/acoustic_exact(F, mu, kappa, dirs=600, seed=k))
    spec = json.loads(Path(args.spectrum).read_text())
    base = kappa+4*mu/3
    disc = [{"name": r["name"], "r_exact_discrete": r["r_exact"],
             "r_bound": math.sqrt(rc.wave_modulus_bound_np(np.array(r["F"]), mu, kappa)/base)} for r in spec["rows"]]
    for d in disc:
        d["bound_over_exact"] = d["r_bound"]/d["r_exact_discrete"]
    res = {"label": "DIAGNOSTIC: stable_step_guard_v1 bound check", "beta": kappa/mu,
           "acoustic": {"samples": len(ratios), "min_bound_over_exact": min(ratios), "max": max(ratios),
                        "median": float(np.median(ratios))},
           "discrete": {"states": len(disc), "min_bound_over_exact": min(d["bound_over_exact"] for d in disc),
                        "max_bound_over_exact": max(d["bound_over_exact"] for d in disc), "rows": disc},
           "identity_speed_ratio": math.sqrt(rc.wave_modulus_bound_np(np.eye(3), mu, kappa)/base)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "discrete"} | {"discrete_min": res["discrete"][
        "min_bound_over_exact"], "discrete_max": res["discrete"]["max_bound_over_exact"]}))


if __name__ == "__main__":
    main()
