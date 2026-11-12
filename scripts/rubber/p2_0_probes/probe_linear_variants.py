import json
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_linear_cantilever import solve, beam
rho, mu = 1150., 5e5
rows = []
def run(L_over_t, W_over_t, beta, jbar, n_t, ppc):
    t = .004; L = L_over_t*t; W = W_over_t*t; kappa = beta*mu
    eb, timo, E, nu = beam(L, W, t, mu, kappa, rho)
    t0 = time.perf_counter()
    try:
        tip, n, nodes = solve(L, W, t, n_t, mu, kappa, rho, ppc=ppc, jbar=jbar)
    except Exception as exc:
        tip, n, nodes = float("nan"), -1, -1
        print("FAIL", exc)
    r = dict(L_over_t=L_over_t, W_over_t=W_over_t, beta=beta, nu=round(nu,4), jbar=jbar, n_t=n_t, ppc=ppc,
             particles=n, tip_m=tip, ratio_eb=tip/eb, ratio_timo=tip/timo, wall_s=round(time.perf_counter()-t0,2))
    rows.append(r)
    print(f"L/t={L_over_t} W/t={W_over_t} beta={beta} jbar={int(jbar)} n_t={n_t} ppc={ppc} P={n} tip/EB={r['ratio_eb']:.4f} tip/Timo={r['ratio_timo']:.4f} {r['wall_s']}s", flush=True)
# slender L/t=20, square section
for n_t in (2, 3, 4, 6):
    run(20, 1, 20., True, n_t, 2)
for n_t in (2, 4, 6):
    run(20, 1, 20., False, n_t, 2)
# quadrature density
for ppc in (1, 3):
    for n_t in (2, 3, 4):
        run(10, 1, 20., True, n_t, ppc)
# wide strip W = 4t
for n_t in (2, 3, 4):
    run(10, 4, 20., True, n_t, 2)
json.dump(rows, open("lin_variants.json", "w"), indent=1)
