"""DIAGNOSTIC (not a protocol result): independent re-analysis of the RUBBER-VISC-03 / -05 FAILs of ``visc_v1_001``
from its raw data, the basis of the ADR 0023 §21.7 decision (rcs-xhigh). No simulation is run; the script reads
``artifacts/rl_ready/p2_3/visc_v1_001/raw`` and ``visc_diag_cantilever_books_002/summary.json``.

1. The declared estimator (``rubber_visco_reference.fourier_modulus`` on the last "full" cycle) reproduces the v1 E', E''.
2. Estimator leakage alone: exact steady-state small-strain samples on the protocol's time grid (strain at t_k, wrench
   as the exact control-step average), declared estimator vs a whole-period window (the smallest number K of periods
   with an integer number of control steps; sigma and eps mean-removed).
3. Start-up transient: the exact quasi-static linear response to eps = eps0 (1 - cos wt) from rest (partial fractions
   of E(t)), for the v1 window and for a whole-period window starting after 6 tau_max.
4. Drive-end inertia: exact steady state of the 1-D linear viscoelastic rod rho u'' = E* u'' with u(0) = 0,
   u(L) = U e^{iwt}; gauge strain between the actual marker reference layers a, b:
   sigma(L) / eps_gauge = E* k (b - a) cos(kL) / (sin kb - sin ka), sigma(0) / eps_gauge = E* k (b - a) / (sin kb - sin ka),
   k = w sqrt(rho / E*). Checked against the observed grip-1 minus grip-0 difference; de-embedded E* (fixed point).
5. Identification / held-out on the corrected data (the protocol's fit, DIAGNOSTIC).
6. Cantilever books (diag_002): kick-paired W_kp vs the branch's own energy D + Psi_v.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/diag_visc_v1_reanalysis.py \
        --out artifacts/rl_ready/p2_3/visc_diag_v1_reanalysis_001
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from robotic_cutting_sim import rubber_visco_reference as ref    # noqa: E402

RAW = ROOT/"artifacts"/"rl_ready"/"p2_3"/"visc_v1_001"/"raw"
DIAG = ROOT/"artifacts"/"rl_ready"/"p2_3"/"visc_diag_cantilever_books_002"/"summary.json"
MU0, KAPPA, RHO = 4.0e5, 8.0e6, 1150.0
P2 = ((0.2, 0.005), (0.15, 0.05))
L, W, TH = 0.030, 0.006, 0.003
DTC, EPS0, TAU_MAX = 2.0e-4, 0.002, 0.05
RATES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def load(job):
    meta = json.loads((RAW/f"{job}.json").read_text())
    data = np.load(RAW/f"{job}.npz")["data"]
    return meta, {c: data[:, i] for i, c in enumerate(meta["cols"])}


def periods_per_window(f):
    spp = 1.0/(f*DTC)
    for K in range(1, 65):
        if abs(K*spp-round(K*spp)) < 1e-9:
            return K
    raise ValueError(f)


def declared(sig, eps, tk, w):
    win = tk > tk[-1]-2*math.pi/w+1e-12
    return ref.fourier_modulus(sig[win], eps[win], tk[win]-DTC/2, tk[win], w, DTC), int(win.sum())


def whole(sig, eps, tk, w, K):
    n = int(round(K*2*math.pi/w/DTC))
    s, e, t = sig[-n:], eps[-n:], tk[-n:]
    x = w*DTC/2
    num = np.sum((s-s.mean())*np.exp(-1j*w*(t-DTC/2)))/(math.sin(x)/x)
    return complex(num/np.sum((e-e.mean())*np.exp(-1j*w*t))), n


def transfer(w, E, a, b, end):
    k = w*np.sqrt(RHO/complex(E))
    out = k*(b-a)/(np.sin(k*b)-np.sin(k*a))
    return complex(out*np.cos(k*L)) if end == 1 else complex(out)


def deembed(Emeas, w, a, b, end):
    E = Emeas
    for _ in range(100):
        new = Emeas/transfer(w, E, a, b, end)
        if abs(new-E) <= 1e-14*abs(E):
            return new
        E = new
    raise RuntimeError("de-embedding did not converge")


def exact_response(tk, w):
    """Control-step averages over (t_k - dtc, t_k] of the exact small-strain stress for eps0 (1 - cos wt) from rest."""
    N, D = ref._shear_polys(MU0, P2)
    r, p = ref._inverse_laplace_simple(9*KAPPA*MU0*N, np.poly1d([1.0, 0.0])*(3*KAPPA*D+MU0*N))
    out = np.zeros_like(tk)
    a, b = tk-DTC, tk
    for rk, pk in zip(r, p):
        I1 = (np.exp(1j*w*b)-np.exp(1j*w*a))/(1j*w)
        I2 = (np.exp(pk*b)-np.exp(pk*a))/pk if abs(pk) > 0 else (b-a)
        out += (EPS0*w*rk/(1j*w-pk)*(I1-I2)).imag/DTC
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    rows, sim = [], {}
    for f in RATES:
        meta, s = load(f"v03_strip_{f:g}hz")
        a, b = meta["info"]["marker_ref_x"]
        w, tk = 2*math.pi*f, s["t"]
        Eref = complex(ref.uniaxial_complex_modulus(w, MU0, KAPPA, P2))
        rel = lambda E: (E.real/Eref.real-1, E.imag/Eref.imag-1)       # noqa: E731
        sig1, sig0 = -s["wrench1_x"]/(W*TH), s["wrench0_x"]/(W*TH)
        K = periods_per_window(f)
        E_decl, n_decl = declared(sig1, s["eps"], tk, w)
        E_decl0, _ = declared(sig0, s["eps"], tk, w)
        E_w1, n_w = whole(sig1, s["eps"], tk, w, K)
        E_w0, _ = whole(sig0, s["eps"], tk, w, K)
        # 2. leakage alone (exact steady state on the same grid)
        Einf = float(ref.uniaxial_complex_modulus(0.0, MU0, KAPPA, P2).real)
        x = w*DTC/2
        sig_ss = Einf*EPS0-(Eref*EPS0*np.exp(1j*w*(tk-DTC/2))*math.sin(x)/x).real
        eps_ss = EPS0*(1-np.cos(w*tk))
        L_decl, _ = declared(sig_ss, eps_ss, tk, w)
        L_whole, _ = whole(sig_ss, eps_ss, tk, w, K)
        # 3. transient: v1 window and the whole-period window starting after 6 tau_max
        sig_tr = exact_response(tk, w)
        T_decl, _ = declared(sig_tr, eps_ss, tk, w)
        n_cyc = K*math.ceil((math.ceil(6*TAU_MAX*f)+K)/K)
        tk2 = DTC*np.arange(1, int(round(n_cyc/(f*DTC)))+1)
        T_whole, _ = whole(exact_response(tk2, w), EPS0*(1-np.cos(w*tk2)), tk2, w, K)
        # 4. inertia
        T1, T0 = transfer(w, Eref, a, b, 1), transfer(w, Eref, a, b, 0)
        E_de1, E_de0 = deembed(E_w1, w, a, b, 1), deembed(E_w0, w, a, b, 0)
        sim[f] = E_de1
        rows.append({
            "rate_hz": f, "marker_ref_x_m": [a, b], "samples_v1_window": n_decl, "samples_per_period": 1/(f*DTC),
            "whole_window_periods": K, "whole_window_samples": n_w,
            "v1_declared_E1_E2_rel": rel(E_decl),
            "leakage_only_declared_E1_E2_rel": rel(L_decl), "leakage_only_whole_E1_E2_rel": rel(L_whole),
            "transient_plus_leakage_v1_window_E1_E2_rel": rel(T_decl),
            "transient_plus_leakage_whole_after_6tau_E1_E2_rel": rel(T_whole), "whole_window_run_cycles": n_cyc,
            "inertia_exact_grip1_rel_E1": (Eref*T1).real/Eref.real-1,
            "inertia_exact_grip0_rel_E1": (Eref*T0).real/Eref.real-1,
            "inertia_implementer_3_8_rel_E1": -(3/8)*RHO*w*w*L*L/Eref.real,
            "grip1_minus_grip0_E1_observed_declared": (E_decl.real-E_decl0.real)/Eref.real,
            "grip1_minus_grip0_E1_observed_whole": (E_w1.real-E_w0.real)/Eref.real,
            "grip1_minus_grip0_E1_predicted_exact": ((Eref*T1).real-(Eref*T0).real)/Eref.real,
            "whole_not_deembedded_E1_E2_rel": rel(E_w1),
            "whole_deembedded_grip1_E1_E2_rel": rel(E_de1), "whole_deembedded_grip0_E1_E2_rel": rel(E_de0),
            "half_substep_stamp_E2_shift_if_applied": (w*meta["info"]["dt"]/2)/(E_de1.imag/E_de1.real)})
    # 5. identification on the corrected data (the protocol's fit; DIAGNOSTIC)

    def model(xv, f, M=2):
        return ref.uniaxial_complex_modulus(2*math.pi*f, MU0, KAPPA, [(xv[2*i], math.exp(xv[2*i+1])) for i in range(M)])

    def resid(xv, rates):
        res = []
        for f in rates:
            Em, Es = model(xv, f), sim[f]
            res += [Em.real/Es.real-1, Em.imag/Es.imag-1]
        return res
    fit = least_squares(lambda xv: resid(xv, (1.0, 4.0, 16.0)), [0.25, math.log(3e-3), 0.1, math.log(0.1)],
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    terms = sorted([(fit.x[0], math.exp(fit.x[1])), (fit.x[2], math.exp(fit.x[3]))], key=lambda z: z[1])
    xf = [terms[0][0], math.log(terms[0][1]), terms[1][0], math.log(terms[1][1])]
    ident = {"terms": [{"g_rel": g/d[0]-1, "tau_rel": t/d[1]-1} for (g, t), d in zip(terms, P2)],
             "heldout": [{"rate_hz": f, "E1_pred_rel": model(xf, f).real/sim[f].real-1,
                          "E2_pred_rel": model(xf, f).imag/sim[f].imag-1} for f in (2.0, 8.0, 32.0)]}
    # 6. cantilever books: kick-paired W_kp vs D + Psi_v (diag_002)
    cant = []
    if DIAG.exists():
        for r in json.loads(DIAG.read_text())["rows"]:
            cant.append({"variant": r["variant"], "BN_left_max_rel": r["BN_max_abs_rel"],
                         "BN_kick_paired_max_rel": r["kick_paired_BN_max_abs_rel"],
                         "W_kp_minus_D_minus_psi_v_end_rel": -r["D_minus_kick_paired_Wb_end_rel"]-r["psi_v_end_rel"],
                         "W_left_minus_D_minus_psi_v_end_rel": -r["D_minus_Wb_end_rel"]-r["psi_v_end_rel"]})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = {"label": "DIAGNOSTIC (re-analysis of visc_v1_001 raw data; not a protocol result)",
               "command": " ".join(sys.argv), "commit": commit, "inputs": [str(RAW.relative_to(ROOT)),
                                                                          str(DIAG.relative_to(ROOT))],
               "rates": rows, "identification_on_corrected_v1_data": ident, "cantilever_books_diag_002": cant}
    out.mkdir(parents=True)
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for r in rows:
        print(f"{r['rate_hz']:>4g} Hz v1 {r['v1_declared_E1_E2_rel'][0]:+.4%} | leak {r['leakage_only_declared_E1_E2_rel'][0]:+.4%}"
              f" | inertia {r['inertia_exact_grip1_rel_E1']:+.4%} (3/8: {r['inertia_implementer_3_8_rel_E1']:+.4%})"
              f" | g1-g0 obs {r['grip1_minus_grip0_E1_observed_whole']:+.4%} pred {r['grip1_minus_grip0_E1_predicted_exact']:+.4%}"
              f" | de-embedded E1 {r['whole_deembedded_grip1_E1_E2_rel'][0]:+.4%} E2 {r['whole_deembedded_grip1_E1_E2_rel'][1]:+.4%}"
              f" | transient whole {r['transient_plus_leakage_whole_after_6tau_E1_E2_rel'][0]:+.1e}"
              f" {r['transient_plus_leakage_whole_after_6tau_E1_E2_rel'][1]:+.1e}")
    print(json.dumps(ident, default=float))
    for c in cant:
        print(json.dumps(c))


if __name__ == "__main__":
    main()
