import json
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_tlmpm_dynamic import Strip
n_t = int(sys.argv[1]); damp = int(sys.argv[2]); cfl = float(sys.argv[3]); T = float(sys.argv[4]); jbar = int(sys.argv[5]) if len(sys.argv) > 5 else 1
s = Strip(L=.1, W=.005, t=.005, n_t=n_t, mu=2e5, beta=20., rho=1150., cfl=cfl, pic=0.0, device="cpu", jbar=bool(jbar))
m = s.m.astype(np.float64); ke_prev = 0.
steps = int(round(T/s.dt))
for k in range(steps):
    s.step()
    if damp and k % 10 == 0:
        v = s.v.numpy().astype(np.float64); ke = .5*np.sum(m*np.sum(v*v, 1))
        if ke < ke_prev:
            s.v.zero_(); ke = 0.
        ke_prev = ke
    if k % 2000 == 0 or k == steps-1:
        e = s.energy()
        v = s.v.numpy()
        F = s.F.numpy().astype(np.float64)
        sv = np.linalg.svd(np.nan_to_num(F), compute_uv=False)
        print(json.dumps(dict(k=k, t=round(k*s.dt, 4), tip=e["tip"], J=[e["J_min"], e["J_max"]], Jbar=[e["Jbar_min"], e["Jbar_max"]],
                              vmax=float(np.abs(v).max()), smin=float(sv.min()), smax=float(sv.max()), finite=e["finite"])), flush=True)
        if not e["finite"]:
            break
