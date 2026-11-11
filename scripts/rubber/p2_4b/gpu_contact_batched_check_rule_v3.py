"""RUBBER-CONT-09 b, rule v3 (versioned check script; K2 review finding K2-3): the v2 band with a PER-QUANTITY tolerance
floor and without count columns.

Why. The v2 rule (``gpu_contact_batched_check_v2.py``, ``band_v2``) sets the floor eps_rel * max_{i,c} |A_i[c]| over
the whole observable vector. For the 18-column energies summary (``tlmpm_rubber_v1.DIAG_COLUMNS``) that maximum is the
integer column ``particles`` (41,220 in the -09 configuration; ``cells`` 5,175), so the floor is ~0.04 in mixed units,
10^2 - 10^3 x the elastic and kinetic energies of the out-of-reach envs: the 18 x 18 = 324 summary scalars of the
1,566 scored in ``gpu_cont09_v2_001`` / ``gpu_cont09_v3_001`` (18 columns x 6 steps x 3 envs) could not fail. The
PASS of those runs rests on the markers, the wrenches, criterion a (exact-zero contact rows), criterion c (audits) and
the CPU L1 byte tests (evidence notes ``k2_review_note_k2_3``).

Rule v3 (this file):
- every observable vector is split into QUANTITIES of one unit, each with its own floor eps_rel * max_{i,c in q}
  |A_i[c]|: the energies summary per column; a wrench [.., 6] into its force (first 3) and torque (last 3) parts per
  row; markers (m) stay one quantity;
- the integer / flag summary columns ``particles``, ``cells`` (counts) and ``nonfinite`` (flag) are NOT scored by the
  band: they must be bit-equal between every A and B episode and to the A reference (``exact_columns``), reported
  separately (a count is either equal or the env was rebuilt);
- d, s, lam and the PASS condition are those of v2 (d <= lam s + floor_q).

Status: a versioned check, not a declared protocol rule. No protocol declares rule v3 yet, so a run of this script is
DIAGNOSTIC until the next declared -09 version (planned for RL-04 / ENV2) names it; the v2 / v3 protocol verdicts of
``gpu_cont09_v2_001`` / ``gpu_cont09_v3_001`` are unchanged. The GPU run reuses the v2 script unchanged except for the
band function (and the protocol / law constants, as ``scripts/rubber/p2_4/gpu_contact_batched_check_v3.py`` does):

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \\
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python \\
        scripts/rubber/p2_4b/gpu_contact_batched_check_rule_v3.py --law v3 --out artifacts/rl_ready/p2_4b/<dir>

The raw per-episode observables of the earlier runs were not saved, so they cannot be re-scored with this rule.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RULE = "rubber_cont09_b_rule_v3_per_quantity_floor"
# tlmpm_rubber_v1.DIAG_COLUMNS (copied: this file must import without Warp; a CPU test checks the copy)
DIAG_COLUMNS = ("nonfinite", "J_min", "J_max", "volume_ratio", "Jbar_min", "Jbar_max", "frac_cells_jbar_dev_gt_0_05",
                "elastic_iso", "elastic_vol", "elastic", "kinetic_particle", "kinetic_grid", "gravity",
                "max_speed_m_s", "cells", "particles", "max_grid_speed_m_s", "cfl_eff_max")
EXACT_COLUMNS = ("nonfinite", "cells", "particles")
WRENCHES = ("wrench", "plane_wrench", "box_wrench", "pad_wrench", "pending_wrench")
LAWS = {"v2": ("rubber_contact_tests_v2.json", "89bc7680bb9c2ed6322279b198899cef0faf31c4751b6bd849b22ecd24a339d0",
               "grid_node_penalty_coulomb_v2"),
        "v3": ("rubber_contact_tests_v3.json", "36d70beaf63a0ef73195dcc47c62e522596779bb95b423a5544e5f7dc9e6a856",
               "grid_node_penalty_coulomb_v3")}


def _band(A, B, lam, eps_rel):
    d = np.maximum(A.max(0)-B.min(0), B.max(0)-A.min(0))
    s = np.maximum(np.ptp(A, axis=0), np.ptp(B, axis=0))
    floor = float(eps_rel*np.abs(A).max()) if A.size else 0.0
    limit = lam*s+floor
    return d, s, floor, limit


def quantities(name, A):
    """Split observable ``name`` with samples ``A`` [k, ...] into (quantity name, column selector of the flattened
    vector) pairs; exact (count / flag) columns are returned with kind "exact"."""
    k = A.shape[0]
    flat = A.reshape(k, -1)
    n = flat.shape[1]
    if name == "summary":
        if n != len(DIAG_COLUMNS):
            raise ValueError(f"summary has {n} columns, expected {len(DIAG_COLUMNS)}")
        return [(f"summary.{c}", np.array([i]), "exact" if c in EXACT_COLUMNS else "band")
                for i, c in enumerate(DIAG_COLUMNS)]
    if name in WRENCHES:
        if n % 6:
            raise ValueError(f"{name}: {n} components is not a multiple of 6")
        out = []
        for r in range(n//6):
            out.append((f"{name}[{r}].force", np.arange(6*r, 6*r+3), "band"))
            out.append((f"{name}[{r}].torque", np.arange(6*r+3, 6*r+6), "band"))
        return out
    return [(name, np.arange(n), "band")]


def band_v3(name, A, B, lam, eps_rel):
    """Rule v3 for one observable at one (step, env): per-quantity floors, exact count / flag columns."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    if A.shape[1:] != B.shape[1:]:
        raise ValueError("A and B have different shapes")
    fa, fb = A.reshape(len(A), -1), B.reshape(len(B), -1)
    out = []
    for qname, cols, kind in quantities(name, A):
        a, b = fa[:, cols], fb[:, cols]
        if kind == "exact":
            same = bool((a == a[0]).all() and (b == a[0]).all())
            out.append({"quantity": qname, "kind": "exact", "components": len(cols), "pass": same,
                        "failing_components": 0 if same else len(cols)})
            continue
        d, s, floor, limit = _band(a, b, lam, eps_rel)
        ok = d <= limit
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(limit > 0, d/np.where(limit > 0, limit, 1.0), np.where(d > 0, np.inf, 0.0))
        c = int(np.argmax(ratio))
        out.append({"quantity": qname, "kind": "band", "components": len(cols), "pass": bool(ok.all()),
                    "failing_components": int((~ok).sum()), "s_zero_components": int((s == 0).sum()),
                    "worst_component": c, "d": float(d[c]), "s": float(s[c]), "floor": floor,
                    "limit": float(limit[c]), "d_over_limit": float(ratio[c])})
    return out


def band_v3_all(runs_a, runs_b, envs, lam, eps_rel):
    """``band_v3`` over every (step, env in ``envs``, observable); same record layout as ``band_v2_all``. Returns
    (pass, band rows worst first, summary)."""
    rows, exact = [], []
    for k in range(len(runs_a[0]["obs"])):
        for e in envs:
            for name in runs_a[0]["obs"][k][e]:
                A = np.stack([np.asarray(r["obs"][k][e][name], np.float64) for r in runs_a])
                B = np.stack([np.asarray(r["obs"][k][e][name], np.float64) for r in runs_b])
                if A[0].size == 0:
                    continue
                for q in band_v3(name, A, B, lam, eps_rel):
                    q.update(step=k, env=e)
                    (exact if q["kind"] == "exact" else rows).append(q)
    rows.sort(key=lambda r: -r["d_over_limit"])
    summary = {"rule": RULE, "band_scalars": sum(r["components"] for r in rows),
               "band_scalars_failing": sum(r["failing_components"] for r in rows),
               "band_scalars_with_s_zero": sum(r["s_zero_components"] for r in rows),
               "band_quantities": len(rows), "exact_scalars": sum(q["components"] for q in exact),
               "exact_scalars_failing": sum(q["failing_components"] for q in exact),
               "max_d_over_limit": rows[0]["d_over_limit"] if rows else 0.0}
    ok = summary["band_scalars_failing"] == 0 and summary["exact_scalars_failing"] == 0
    return ok, rows, summary


def main():
    """GPU run: the v2 script with ``band_v2_all`` replaced by rule v3 (the result keeps the v2 layout; the rule-v3
    summary is in ``b_perturbation_band``; meta labelled DIAGNOSTIC_RULE_V3_NOT_DECLARED)."""
    law = sys.argv[sys.argv.index("--law")+1] if "--law" in sys.argv else "v3"
    if law not in LAWS:
        sys.exit(f"--law must be one of {sorted(LAWS)}")
    if "--law" in sys.argv:
        i = sys.argv.index("--law")
        del sys.argv[i:i+2]
    path = ROOT/"scripts"/"rubber"/"p2_4b"/"gpu_contact_batched_check_v2.py"
    spec = importlib.util.spec_from_file_location("gpu_contact_batched_check_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    proto, sha, law_name = LAWS[law]
    mod.PROTOCOL, mod.PROTOCOL_SHA256, mod.LAW = ROOT/"configs"/"rubber"/proto, sha, law_name
    d = list(mod.make_lives.__defaults__)
    d[-1] = law_name
    mod.make_lives.__defaults__ = tuple(d)

    def band_all(runs_a, runs_b, envs, lam, eps_rel):
        ok, rows, summary = band_v3_all(runs_a, runs_b, envs, lam, eps_rel)
        legacy = {"scalars": summary["band_scalars"]+summary["exact_scalars"],
                  "scalars_failing": summary["band_scalars_failing"]+summary["exact_scalars_failing"],
                  "scalars_with_s_zero": summary["band_scalars_with_s_zero"], "rows": len(rows),
                  "rows_failing": sum(not r["pass"] for r in rows), "max_d_over_limit": summary["max_d_over_limit"]}
        return ok, rows, legacy["scalars"], {**legacy, "rule_v3": summary}
    mod.band_v2_all = band_all
    out = Path(sys.argv[sys.argv.index("--out")+1])
    mod.main()
    res = out/"result.json"
    if res.exists():
        r = json.loads(res.read_text())
        r["meta"]["label_inherited"] = r["meta"]["label"]
        r["meta"]["label"] = (f"RUBBER-CONT-09 b rule v3 (per-quantity floor, count / flag columns exact), law "
                              f"{law_name}; DIAGNOSTIC_RULE_V3_NOT_DECLARED until a protocol declares rule v3")
        r["meta"]["rule"] = RULE
        r["meta"]["wrapper"] = "scripts/rubber/p2_4b/gpu_contact_batched_check_rule_v3.py"
        res.write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
