"""Compact evidence ``docs/evidence/rl_ready/p2_4_contact_x2_001.json`` of P2-4 task X2 (ADR 0023 §22.10) from the raw
DIAGNOSTIC outputs in ``artifacts/rl_ready/p2_4/x2_diag_001`` (reads only; never re-runs anything; refuses to
overwrite). Every summary value is looked up from a raw row by its exact arguments.

    .venv/bin/python scripts/rubber/p2_4/write_contact_x2_evidence.py --diag artifacts/rl_ready/p2_4/x2_diag_001 \
        --x1-diag artifacts/rl_ready/p2_4/x1_diag_001 --cpu-tests artifacts/rl_ready/p2_4/cpu_tests_x2_001 \
        --out docs/evidence/rl_ready/p2_4_contact_x2_001.json
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V2 = "grid_node_penalty_coulomb_v2"
INC2 = {"L": "0.024", "W": "0.012", "H": "0.006"}
M_INC = 1150.0*0.016*0.008*0.004            # INC block mass (protocol v2 material density x 16 x 8 x 4 mm)


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _tail(path, n=2):
    p = Path(path)
    return p.read_text().strip().splitlines()[-n:] if p.exists() else ["MISSING"]


def _r(x, n=5):
    return None if x is None else float(f"{x:.{n}g}")


def _compact(r):
    return {"slip_err_over_g_cos": _r(r.get("slip_err_gcos")), "dissipation_ratio_run": _r(r.get("D04_ratio_run")),
            "bottom_contact_fraction": _r(r.get("contact_fraction_bottom")),
            "KE_vib_mean_over_S": _r(r.get("KE_vib_mean_window_over_S")), "stick_disp_m": _r(r.get("stick_disp_m")),
            "books_max_rel": _r(r.get("books_max_rel")), "invalid": r.get("invalid"), "dt_s": _r(r.get("dt")),
            "nodes": r.get("nodes")}


class Rows:
    def __init__(self, diag):
        self.rows = {}
        for f in sorted(Path(diag).glob("run*.jsonl"))+sorted(Path(diag).glob("ts_*.jsonl")):
            for r in _jsonl(f):
                a = dict(r["args"])
                a.setdefault("law", V2)
                self.rows.setdefault(json.dumps(a, sort_keys=True), (f.stem, r))

    def get(self, **a):
        a = {k: str(v) for k, v in a.items()}
        a.setdefault("law", V2)
        key = json.dumps(a, sort_keys=True)
        if key not in self.rows:
            raise KeyError(key)
        return self.rows[key][1]

    def m(self, metric, **a):
        return _r(self.get(**a)[metric])


def _ramp_disp(r):
    s = r["series"]
    x = np.asarray(s["x_cm"])
    return {"ramp_phase_disp_m": _r(float(x[19]-x[0])), "after_ramp_disp_m": _r(float(x[-1]-x[19]))}


def _pattern(path):
    t = Path(path).read_text()
    g = lambda pat: re.search(pat, t)                                                   # noqa: E731
    return {"mean_bottom_contact_fraction": _r(float(g(r"mean contact frac ([0-9.e-]+)").group(1))),
            "vz_rms_m_s": _r(float(g(r"vz rms ([0-9.e-]+)").group(1))),
            "vx_rms_about_mean_m_s": _r(float(g(r"vx rms about mean ([0-9.e-]+)").group(1))),
            "mean_vx_m_s": _r(float(g(r"mean vx ([0-9.e-]+)").group(1))),
            "vz_centre_dominant_freq_hz": g(r"dominant freq \[([^\]]+)\]").group(1).split(),
            "x_wavenumber_spectrum_vz": g(r"x-wavenumber spectrum of vz \(k index 0\.\.\):\s*\[([^\]]+)\]").group(1).split()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--x1-diag", required=True)
    ap.add_argument("--cpu-tests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    diag = ROOT/a.diag
    R = Rows(diag)
    K = {" ".join(f"{k}={v}" for k, v in r["args"].items()): r for r in _jsonl(ROOT/a.x1_diag/"sweepK_v2_errormodel.jsonl")}
    sl, d4, cf, st = "slip_err_gcos", "D04_ratio_run", "contact_fraction_bottom", "stick_disp_m"

    def slide(mu, ratio, **kw):
        r = R.get(mu=mu, ratio=ratio, **kw)
        return {"slip_err": _r(r[sl]), "D04": _r(r[d4]), "contact_frac": _r(r[cf])}

    def hold(mu, **kw):
        return R.m(st, mu=mu, ratio=0.95, **kw)

    # ---------------------------------------------------------------- v2: flutter character and refinement
    v2_ref = {"mu_0.8_ratio_1.2": {
        "n_t_3_cfl_0.5": slide(0.8, 1.2, dev="x2", smode=0),
        "n_t_3_cfl_0.25": slide(0.8, 1.2, cfl=0.25), "n_t_3_cfl_0.125": slide(0.8, 1.2, cfl=0.125),
        "n_t_4": slide(0.8, 1.2, nt=4), "n_t_5": slide(0.8, 1.2, nt=5), "n_t_6": slide(0.8, 1.2, nt=6),
        "n_t_8": slide(0.8, 1.2, nt=8)},
        "mu_1.0_ratio_1.2": {
        "n_t_3_cfl_0.5_x1_sweepK": _r(K[f"mu=1.0 ratio=1.2 law={V2}"][sl]),
        "n_t_3_cfl_0.25": slide(1.0, 1.2, cfl=0.25), "n_t_4": slide(1.0, 1.2, nt=4), "n_t_6": slide(1.0, 1.2, nt=6)},
        "reading": "dt-converged (cfl 0.5 / 0.25 / 0.125 within 0.15 % of g cos theta at mu 0.8) and NOT dx-converged: the "
                   "friction deficit and the vibration energy grow with n_t at every mu tested"}
    pdir = diag/"pattern_b019684"
    pattern = {f.stem: _pattern(f) for f in sorted(pdir.glob("pat_*.txt"))}
    envelope = {
        "sliding_mu_0.6_INC": {"n_t_3_formal_contact_v2_001": "+0.78 / +0.42 / +0.38 % (INC2, PASS)",
                               "n_t_4": [slide(0.6, r, nt=4) for r in (1.1, 1.2, 1.3)],
                               "n_t_6": [slide(0.6, r, nt=6) for r in (1.1, 1.2, 1.3)],
                               "cfl_0.25_ratio_1.2": slide(0.6, 1.2, cfl=0.25)},
        "sliding_mu_0.6_INC2_n_t_4": [slide(0.6, r, nt=4, **INC2) for r in (1.1, 1.2, 1.3)],
        "sliding_mu_0.4_INC_n_t_6_ratio_1.2": slide(0.4, 1.2, nt=6),
        "sliding_mu_0.8_INC2_n_t_3": [slide(0.8, r, **INC2) for r in (1.1, 1.2, 1.3)],
        "hold_0.95_mu_0.6_INC_n_t_4_m": hold(0.6, nt=4), "hold_0.95_mu_0.8_INC_n_t_4_m": hold(0.8, nt=4),
        "hold_0.95_mu_0.8_INC2_n_t_3_m": hold(0.8, **INC2),
        "thresholds": "slip <= 1 % of g cos theta, |D04 - 1| <= 5 %, stick displacement <= 0.01 s = 6.7e-6 m (INC)",
        "reading": "the v2 declared envelope (sliding mu <= 0.6, static mu <= 0.8) holds at n_t 3 on INC / INC2 (the "
                   "formal rows), but not at the acceptance profile rubber_ref_v1 (n_t >= 4): mu 0.8 hold slides at "
                   "n_t 4, mu 0.6 sliding exceeds 1 % on INC2 at n_t 4 and on INC at n_t 6 (ratio 1.1)"}
    hf = {"prony_g0.15_tau100us": {
        "mu_0.8": {"INC_n_t_3": [slide(0.8, r, prony="0.15:0.0001") for r in (1.1, 1.2, 1.3)],
                   "INC_n_t_4": [slide(0.8, r, prony="0.15:0.0001", nt=4) for r in (1.1, 1.2)],
                   "INC2_n_t_3": [slide(0.8, r, prony="0.15:0.0001", **INC2) for r in (1.1, 1.2, 1.3)],
                   "hold_0.95_n_t_4_m": hold(0.8, prony="0.15:0.0001", nt=4)},
        "mu_1.0_ratio_1.2": slide(1.0, 1.2, prony="0.15:0.0001"), "mu_1.2_ratio_1.2": slide(1.2, 1.2, prony="0.15:0.0001"),
        "mu_1.5_ratio_1.2": slide(1.5, 1.2, prony="0.15:0.0001"),
        "hold_0.95_mu_1.0_m": hold(1.0, prony="0.15:0.0001")},
        "note": "D04 counts contact dissipation only; with a lossy material part of the interface work is booked as "
                "branch dissipation, so D04 < 1 is expected there (not scored, DIAGNOSTIC)",
        "reading": "a realistic HF loss in the material removes most of the mu 0.8 deficit (not INC2 ratio 1.1) and "
                   "restores the mu 0.8 hold at n_t 4; it does not fix mu >= 1.0 (continuum ill-posed regime)"}

    # ---------------------------------------------------------------- candidates
    cand = {}
    cand["nonlocal_cap_lattice_radius"] = {
        "law": "cap_i = mu sum_j w_ij F_n,j / Z_j over reference-lattice neighbours in contact with the same plane "
               "(Oden-Pires type, conservative in total); smode 1",
        "radius_1": {str(m): slide(m, 1.2, dev="x2", smode=1, radius=1) for m in (0.8, 1.0, 1.2, 1.5)},
        "radius_2": {str(m): slide(m, 1.2, dev="x2", smode=1, radius=2) for m in (0.8, 1.0, 1.2, 1.5)},
        "radius_3": {str(m): slide(m, 1.2, dev="x2", smode=1, radius=3) for m in (0.8, 1.0, 1.5)},
        "verdict": "REJECTED: partial at mu 0.8 (radius 2 / 3: 0.70 / 0.60 %), mu >= 1.0 not fixed; a lattice radius "
                   "shrinks under refinement"}
    cand["patch_level_cap"] = {
        "law": "cap_i = mu (sum_j F_n,j) m_i / sum_j m_j over the nodes in contact with the same obstacle; smode 6",
        "INC_n_t_3": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=6) for m in (0.8, 0.9) for r in (1.1, 1.2, 1.3)},
        "INC_mu_1.0_r_1.2": slide(1.0, 1.2, dev="x2", smode=6),
        "cfl_0.25_mu_0.8": [slide(0.8, r, dev="x2", smode=6, cfl=0.25) for r in (1.1, 1.2, 1.3)],
        "INC2_mu_0.8": [slide(0.8, r, dev="x2", smode=6, **INC2) for r in (1.1, 1.2, 1.3)],
        "holds_m": {f"mu_{m}": hold(m, dev="x2", smode=6) for m in (0.4, 0.8, 1.0, 1.2, 1.5)},
        "verdict": "REJECTED: mu 0.8 passes at n_t 3 / cfl 0.5 but fails at cfl 0.25 (ratio 1.1: 1.16 %), on INC2 "
                   "(ratio 1.1: D04 0.947) and at mu 0.9; mu >= 1.0 not fixed; hold lost at mu 1.2"}
    cand["patch_cap_lagged_total"] = {
        "law": "cap_i = mu L_tau(sum_j F_n,j) m_i / sum_j m_j (patch cap on a first-order-lagged patch normal force); "
               "smode 10",
        "tau_0.25ms": {"INC_n_t_3": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=10, tau=0.00025)
                                     for m in (0.8, 0.9, 1.0, 1.2, 1.5)},
                       "INC_n_t_4": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=10, tau=0.00025, nt=4)
                                     for m in (0.8, 0.9) for r in (1.1, 1.3)},
                       "INC_n_t_6": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=10, tau=0.00025, nt=6)
                                     for m in (0.8, 0.9) for r in (1.1, 1.3)},
                       "INC2": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=10, tau=0.00025, **INC2)
                                for m in (0.8, 0.9) for r in (1.1, 1.2, 1.3)},
                       "recovery_switch_40ms": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=10, tau=0.00025,
                                                                 switch_t=0.04) for m in (0.8, 0.9)},
                       "holds_m": {f"mu_{m}": hold(m, dev="x2", smode=10, tau=0.00025) for m in (0.4, 1.2, 1.5)}},
        "tau_0.1ms_INC_n_t_3": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=10, tau=0.0001)
                                for m in (0.8, 0.9) for r in (1.1, 1.2, 1.3)},
        "verdict": "REJECTED: exact at n_t 3 for mu <= 0.95 but not under dx refinement (n_t 4: mu 0.8 ratio 1.1 "
                   "+1.7 %, mu 0.9 +5.2 %; n_t 6: mu 0.9 +5.2 %), no recovery from a developed flutter, mu >= 1.0 not fixed, holds lost at "
                   "mu 1.2 / 1.5"}
    sym = {}
    for tau in (0.0005, 0.001, 0.003):
        sym[f"tau_{tau*1e3:g}ms"] = {
            "sliding_r_1.2": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=4, tau=tau) for m in (0.8, 1.0, 1.2, 1.5)},
            "holds_0.95_m": {f"mu_{m}": hold(m, dev="x2", smode=4, tau=tau) for m in (0.4, 0.8, 1.0, 1.5)}}
    sym["tau_0.25ms_sliding"] = {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=4, tau=0.00025)
                                 for m in (0.8, 0.9, 1.0) for r in (1.1, 1.2, 1.3)}
    sym["tau_0.5ms_n_t_4"] = {f"r_{r}": slide(0.8, r, dev="x2", smode=4, tau=0.0005, nt=4) for r in (1.1, 1.2, 1.3)}
    sym["tau_0.5ms_INC2"] = {f"r_{r}": slide(0.8, r, dev="x2", smode=4, tau=0.0005, **INC2) for r in (1.1, 1.2)}
    sym["recovery_switch_40ms"] = {
        "tau_0.5ms_mu_0.8": slide(0.8, 1.2, dev="x2", smode=4, tau=0.0005, switch_t=0.04),
        "tau_0.25ms_mu_0.8": slide(0.8, 1.2, dev="x2", smode=4, tau=0.00025, switch_t=0.04)}
    ramp = {"v2_mu_0.8": _ramp_disp(R.get(mu=0.8, ratio=0.95, ts=1))}
    for tau in (0.00025, 0.0005):
        for m in (0.8, 1.0):
            ramp[f"smode4_tau_{tau*1e3:g}ms_mu_{m}"] = _ramp_disp(R.get(mu=m, ratio=0.95, dev="x2", smode=4, tau=tau,
                                                                        ts=1))
    ramp["smode4_tau_1ms_mu_0.8"] = _ramp_disp(R.get(mu=0.8, ratio=0.95, dev="x2", smode=4, tau=0.001, ts=1))
    for tau in (0.00025, 0.0005, 0.001):
        ramp[f"smode10_tau_{tau*1e3:g}ms_mu_0.8"] = _ramp_disp(R.get(mu=0.8, ratio=0.95, dev="x2", smode=10, tau=tau,
                                                                     ts=1))
    ramp["smode10_tau_0.1ms_mu_0.8"] = _ramp_disp(R.get(mu=0.8, ratio=0.95, dev="x2", smode=10, tau=0.0001, ts=1))
    sym["hold_ramp_phase_displacement"] = ramp
    sym["law"] = ("cap_i = mu F_lag,i, F_lag first-order lag (time constant tau) of the node's normal force, "
                  "initialised at (re)contact (Prakash-Clifton type normal-stress memory); smode 4")
    sym["verdict"] = ("REJECTED: from rest it removes the flutter (tau 0.5 ms: mu 0.8 exact, also at n_t 4 / INC2; tau "
                      "1 / 3 ms: exact to mu 1.2 / 1.5) but (i) it slips under proportional loading: hold ramp-phase "
                      "displacement 10.5 um at tau 0.5 ms vs 0.19 um for v2 (57x), 37 um + 15 um after the ramp at 1 "
                      "ms, 0.2-0.5 mm at 3 ms; (ii) it does not recover from a developed flutter (switch at 40 ms: "
                      "over-friction -12 %); (iii) tau 0.25 ms gives over-friction -7 ... -20 % (the lag holds impact "
                      "peaks of bouncing nodes): a sharp, geometry-dependent threshold")
    cand["symmetric_lag_per_node"] = sym
    cand["memory_with_separation"] = {
        "law": "cap_i = mu F_lag,i with F_lag relaxing to 0 while the node is out of contact (memory kept across "
               "bounces); smode 8 / patch-gated smode 9",
        "smode8_tau_3ms": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=8, tau=0.003) for m in (0.8, 1.0, 1.2, 1.5)},
        "smode8_tau_1ms_mu_1.5": slide(1.5, 1.2, dev="x2", smode=8, tau=0.001),
        "smode8_recovery_switch_40ms": {f"mu_{m}_tau_3ms": slide(m, 1.2, dev="x2", smode=8, tau=0.003, switch_t=0.04)
                                        for m in (1.0, 1.5)},
        "smode8_holds_m": {f"mu_{m}_tau_{t*1e3:g}ms": hold(m, dev="x2", smode=8, tau=t) for m in (0.8, 1.0)
                           for t in (0.001, 0.003)},
        "smode9_tau_3ms": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=9, tau=0.003, vg=0.001) for m in (0.8, 1.0, 1.5)},
        "smode9_holds_m": {f"mu_{m}": hold(m, dev="x2", smode=9, tau=0.003, vg=0.001) for m in (0.8, 1.0)},
        "verdict": "REJECTED: friction collapse once nodes bounce (switch at 40 ms: slip error +75 % at mu 1.0; the "
                   "lagged force averages the zeros of the flight phases), hold slips at tau >= 1 ms; gated: mu 0.8 / "
                   "1.0 pass, mu 1.5 collapses (+99 %), hold at mu 1.0 lost"}
    cand["loading_safe_and_gated_memories"] = {
        "max_memory_smode2_tau_0.5ms": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=2, tau=0.0005) for m in (0.8, 1.0, 1.5)},
        "max_memory_contact_only_smode3_tau_0.5ms": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=3, tau=0.0005)
                                                     for m in (0.8, 1.0, 1.5)},
        "slip_distance_gate_smode5_tau_1ms_ds_10um": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=5, tau=0.001, ds="1e-5")
                                                      for m in (0.8, 1.0, 1.5)},
        "patch_slip_gate_smode7_tau_3ms": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=7, tau=0.003, vg=0.001)
                                           for m in (0.8, 1.0, 1.5)},
        "verdict": "REJECTED: a cap never below the instantaneous one (max) does not suppress the flutter (contact fraction "
                   "<= 0.04, mu 1.5 +18 %); gating the "
                   "memory by slip distance or patch macro-slip lets the flutter start before the memory acts and then "
                   "gives large errors (-15 ... +8 % of g cos theta)"}
    eq = {"law": ("v2 cap and slider; in SLIP only, outside the cap, a dashpot -2 zd sqrt(k_t m_i)(v_t,i - vbar) on the "
                  "deviation of the node's realised tangential velocity from the patch mean of the same substep "
                  "(smode 12: mean over all contact nodes; smode 13: over the nodes whose v2 trial slips, zero net damper "
                  "force over the slipping set, patch-total friction <= mu sum F_n); zd = 1 removes the whole deviation "
                  "per substep (2 zd sqrt(kappa_t) = 1)"),
          "smode11_previous_substep_mean_zd_1": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=11, zd="1.0")
                                                 for m in (0.8, 1.0, 1.5)},
          "smode12_zd_0.5": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=12, zd="0.5")
                             for m in (0.8, 1.0, 1.2, 1.5) for r in (1.1, 1.2, 1.3)},
          "smode12_zd_1": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=12, zd="1.0")
                           for m in (0.8, 1.0, 1.2, 1.5) for r in (1.1, 1.2, 1.3)},
          "smode13_zd_1": {
              "INC_n_t_3": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=13, zd="1.0")
                            for m in (0.8, 1.0, 1.2) for r in (1.1, 1.2, 1.3)},
              "INC_n_t_4": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=13, zd="1.0", nt=4)
                            for m in (0.8, 1.0) for r in (1.1, 1.2)},
              "INC_n_t_6_r_1.1": {f"mu_{m}": slide(m, 1.1, dev="x2", smode=13, zd="1.0", nt=6) for m in (0.6, 0.8, 1.0)},
              "INC_cfl_0.25": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=13, zd="1.0", cfl=0.25)
                               for m in (0.8, 1.0) for r in (1.1, 1.2)},
              "INC2_n_t_3": {f"mu_{m}_r_{r}": slide(m, r, dev="x2", smode=13, zd="1.0", **INC2)
                             for m in (0.8, 1.0) for r in (1.1, 1.2)},
              "INC2_n_t_4_mu_0.6_r_1.1": slide(0.6, 1.1, dev="x2", smode=13, zd="1.0", nt=4, **INC2),
              "recovery_switch_40ms": {f"mu_{m}": slide(m, 1.2, dev="x2", smode=13, zd="1.0", switch_t=0.04)
                                       for m in (0.8, 1.0)},
              "holds_0.95_m": {**{f"mu_{m}": hold(m, dev="x2", smode=13, zd="1.0") for m in (0.4, 0.8, 1.0, 1.2, 1.5)},
                               "mu_0.8_P2": hold(0.8, dev="x2", smode=13, zd="1.0", material="P2"),
                               "mu_0.8_n_t_4": hold(0.8, dev="x2", smode=13, zd="1.0", nt=4),
                               "mu_1.0_n_t_4": hold(1.0, dev="x2", smode=13, zd="1.0", nt=4)},
              "hold_ramp_phase_displacement": {f"mu_{m}": _ramp_disp(R.get(mu=m, ratio=0.95, dev="x2", smode=13,
                                                                            zd="1.0", ts=1)) for m in (0.8, 1.0)}},
          "verdict": ("PROMISING, NOT ADOPTED in X2 (time box): smode 13 zd 1 is exact (|slip err| <= 0.01 %, D04 >= "
                      "0.993) for mu 0.6 / 0.8 at n_t 3 / 4 / 6, cfl 0.25 and on INC2, recovers from a developed "
                      "flutter, keeps v2 hold behaviour (ramp-phase displacement 0.16 um vs 0.19 um for v2; holds at "
                      "0.95 mu to mu 1.5 within the stick limit, incl. mu 0.8 at n_t 4 where v2 slides 3.9 mm); "
                      "mu 1.0 exact at n_t 3 / cfl 0.25 / INC2 but NOT under dx refinement (n_t 4: +2.4 %, n_t 6: "
                      "+4.2 % at ratio 1.1); mu >= 1.2 sliding fails. It meets the requirement at mu 0.8, not at "
                      "mu 1.0; it needs a src implementation (two passes, per-patch accumulators, batched layout), "
                      "pinch / box-frame checks and a declared protocol before any formal claim (task X3)")}
    cand["slip_rate_equalisation_patch"] = eq
    cand["contact_parameters_v2"] = {
        "kappa_1": slide(0.8, 1.2, kn=1, kt=1, zn=0.5, zt=0.5), "zeta_1": slide(0.8, 1.2, zn=1, zt=1),
        "verdict": "not a fix (zeta 1 reduces mu 0.8 to 0.46 % at n_t 3; X1: no kappa / zeta removes it under refinement)"}

    # ---------------------------------------------------------------- -01 d books of the stick runs
    books = []
    for mu in (0.4, 0.8):
        for ratio in (0.6, 0.85):
            for cfl in (0.5, 0.25, 0.125):
                r = R.get(mu=mu, ratio=ratio, cfl=cfl, ts=1)
                s = {k: np.asarray(v, float) for k, v in r["series"].items()}
                S = float(max(s["KE"]+s["psi"])+max(abs(s["W_g"])))
                i = 19
                t = s["t"]
                Wc = s["W_n"]+s["W_t"]
                books.append({"mu": mu, "ratio": ratio, "cfl": cfl, "dt_s": _r(r["dt"]), "S_J": _r(S),
                              "max_abs_B_J": _r(float(abs(s["B"]).max())), "max_abs_B_over_S": _r(float(abs(s["B"]).max())/S),
                              "B_at_20ms_J": _r(float(s["B"][i])), "B_end_J": _r(float(s["B"][-1])),
                              "hold_dB_J": _r(float(s["B"][-1]-s["B"][i])),
                              "hold_dKE_J": _r(float(s["KE"][-1]-s["KE"][i])),
                              "hold_dpsi_J": _r(float(s["psi"][-1]-s["psi"][i])),
                              "hold_dW_g_J": _r(float(s["W_g"][-1]-s["W_g"][i])),
                              "hold_dW_n_J": _r(float(s["W_n"][-1]-s["W_n"][i])),
                              "hold_dW_t_J": _r(float(s["W_t"][-1]-s["W_t"][i])),
                              "hold_contact_power_W": _r(float(np.polyfit(t[20:], Wc[20:], 1)[0])),
                              "hold_KE_mean_J": _r(float(s["KE"][20:].mean())),
                              "hold_v_rms_m_s": _r(math.sqrt(2*float(s["KE"][20:].mean())/M_INC))})
    books_reading = (
        "stick runs of v2 (INC, n_t 3): max |B| is the end value B(140 ms) = 2.3-4.3e-12 J against S = 2.2-4.5e-10 J; "
        "B rises steadily through the static hold (by 2.0-3.6e-12 J after the ramp, 1.7-3.0e-11 W) and falls only by "
        "~40 % from cfl 0.5 to 0.125 (not first order in dt). The static block keeps a sustained jitter (v_rms 5-12 "
        "um/s, growing as dt shrinks) that the contact dashpots dissipate (net contact work -1.6e-11 ... -1.3e-10 W in "
        "the hold); the B drift is the part of that dissipation not matched by a booked energy change. It is a genuine "
        "residual of the discrete dynamics more than 10 orders of magnitude above the float64 resolution of the books "
        "sums (~1e-26 J here): an absolute floor tied to float64 resolution would not change any verdict, and the "
        "relative criterion is not measuring float64 noise. Its weakness for static rows is structural (the residual "
        "accumulates with the hold duration while S stays fixed), which argues for a drift-rate criterion derived "
        "a priori, not for relabelling the residual. No rule is declared (no v3 run); the -01 d FAIL of "
        "contact_v2_001 stands")

    meta = json.loads((diag/"README.json").read_text())
    ident = _jsonl(diag/"identity_b019684"/"rerun.jsonl")
    ident2 = _jsonl(diag/"identity_331550c"/"rerun.jsonl")
    ev = {
        "schema": "rcs_rl_ready_evidence_v1", "id": "p2_4_contact_x2_001",
        "phase": "P2-4 rubber contact, task X2 (defect 2: friction-normal flutter)",
        "adr": "docs/adr/0023-rubber-material-v1.md section 22.10",
        "report": "docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md section 9",
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
        "label": "DIAGNOSTIC (no formal run: no v3 law adopted, no protocol declared)",
        "decision": ("NEGATIVE RESULT for the X2 requirement (robust closure of sliding at mu 0.8 AND 1.0, supported "
                     "range ~0.6-1.5): no v3 law is adopted, no protocol v3 is declared, no formal run; "
                     "grid_node_penalty_coulomb_v2 and rubber_contact_tests_v2 stay as declared (v1 / v2 kernels and "
                     "every frozen file unchanged). The best candidate (patch slip-rate equalisation, smode 13) is "
                     "refinement-stable to mu 0.8 with v2 hold behaviour but not at mu 1.0; it is the recommended v3 "
                     "design for task X3. The declared mu envelope stays and is QUALIFIED: it holds at n_t 3 "
                     "(rubber_rl_v1) on the formal scenes, not at n_t >= 4 (rubber_ref_v1) or on larger blocks at mu 0.8"),
        "raw": {"dir": "artifacts/rl_ready/p2_4/x2_diag_001", "readme": meta},
        "identity_check": {"b019684": {"rows": len(ident), "result": "identical in every field except wall_s"},
                           "331550c": {"rows": len(ident2), "result": "identical in every field except wall_s (smode 0, 4, "
                                                                       "10, 11, 12, 13 against runA / C / E / J / P / Q / R)"}},
        "flutter_character_v2": {"pattern": pattern, "refinement": v2_ref},
        "envelope_qualification_v2": envelope,
        "hf_material_loss_probe": hf,
        "candidates": cand,
        "books_01d_stick_runs": {"rows": books, "reading": books_reading},
        "cpu_tests": {"no_cuda": _tail(ROOT/a.cpu_tests/"no_cuda.log"),
                      "cuda_visible_0": _tail(ROOT/a.cpu_tests/"cuda0.log")},
        "frozen": ("nothing in src, tests or configs changed; v1 / v2 contact kernels byte-identical (tests/"
                   "test_rubber_contact_v2.py pins); no Phase 1 module, configs/rl_benchmark/*, scripts/rl_ready/* or "
                   "frozen rubber module / protocol changed"),
        "rows_index": {k: {"batch": b, **_compact(r)} for k, (b, r) in sorted(R.rows.items())}}
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out)


if __name__ == "__main__":
    main()
