"""Grid-convergence arithmetic for the rubber refinement study (ADR 0023 §14.4, RUBBER-PHYS-11; task R2.2).

Celik et al. (2008, J. Fluids Eng. 130 078001, "Procedure for estimation and reporting of uncertainty due to
discretization in CFD applications") with Roache's grid convergence index. Three grids with representative spacings
h1 < h2 < h3 (fine, medium, coarse) and solutions phi1, phi2, phi3:

    r21 = h2 / h1, r32 = h3 / h2, e21 = phi2 - phi1, e32 = phi3 - phi2, s = sign(e32 / e21)
    p = |ln|e32 / e21| + q(p)| / ln r21,   q(p) = ln((r21^p - s) / (r32^p - s))      (fixed-point iteration)
    phi_ext = (r21^p phi1 - phi2) / (r21^p - 1)
    GCI_fine21 = Fs |e21 / phi1| / (r21^p - 1)                     (uncertainty of phi1)
    GCI_coarse32 = Fs |e32 / phi3| r32^p / (r32^p - 1)             (uncertainty of phi3, the coarse grid)

``s < 0`` is oscillatory convergence (reported as such, p computed from |e32 / e21| as in Celik). Pure functions, no
Warp, float64.
"""
import math

GCI_ID = "gci_celik2008_roache_v1"


def observed_order(h, phi, *, tol=1e-12, max_iter=200):
    """h = (h1, h2, h3) increasing; phi = (phi1, phi2, phi3). Returns dict(p, s, converged, r21, r32, e21, e32)."""
    h1, h2, h3 = map(float, h)
    f1, f2, f3 = map(float, phi)
    if not h1 < h2 < h3:
        raise ValueError("spacings must satisfy h1 < h2 < h3 (fine to coarse)")
    r21, r32 = h2/h1, h3/h2
    e21, e32 = f2-f1, f3-f2
    out = {"r21": r21, "r32": r32, "e21": e21, "e32": e32}
    if e21 == 0.0 or e32 == 0.0:
        return dict(out, p=None, s=0, converged=False, reason="zero difference between grids")
    s = 1 if e32/e21 > 0 else -1
    p = abs(math.log(abs(e32/e21)))/math.log(r21)
    if p < 0.1:                                  # starting guess away from the r^p = s singularity
        p = 1.0
    converged = False
    for _ in range(max_iter):
        try:
            q = math.log((r21**p-s)/(r32**p-s))
        except ValueError:
            return dict(out, p=None, s=s, converged=False, reason="q(p) undefined")
        p_new = abs(math.log(abs(e32/e21))+q)/math.log(r21)
        if abs(p_new-p) < tol:
            p, converged = p_new, True
            break
        p = p_new
    return dict(out, p=p, s=s, converged=converged, oscillatory=s < 0)


def extrapolated(phi1, phi2, r21, p):
    return (r21**p*phi1-phi2)/(r21**p-1)


def gci_fine(phi1, phi2, r21, p, fs=1.25):
    return fs*abs((phi2-phi1)/phi1)/(r21**p-1)


def gci_coarse(phi2, phi3, r32, p, fs=1.25):
    return fs*abs((phi3-phi2)/phi3)*r32**p/(r32**p-1)


def study(h, phi, *, p_bounds=(1.0, 2.0), fs_verified=1.25, fs_unverified=3.0, verified_window=(1.0, 3.0)):
    """Full triplet evaluation with the declared verdict rule of rubber_phys_tests_v4 RUBBER-PHYS-11:
    p_eff = clip(p_obs, *p_bounds); Fs = fs_verified if p_obs lies in ``verified_window`` and convergence is monotone,
    else fs_unverified. Returns the observed order, extrapolation, GCI of the fine and the coarse grid (both with
    p_eff), and the values with p_obs for reference."""
    o = observed_order(h, phi)
    res = {"id": GCI_ID, "h": list(map(float, h)), "phi": list(map(float, phi)), **o}
    if o["p"] is None:
        res.update(status="INCONCLUSIVE", gci_coarse=None, gci_fine=None)
        return res
    p_obs = o["p"]
    p_eff = min(max(p_obs, p_bounds[0]), p_bounds[1])
    verified = verified_window[0] <= p_obs <= verified_window[1] and not o["oscillatory"]
    fs = fs_verified if verified else fs_unverified
    f1, f2, f3 = map(float, phi)
    res.update(status="OSCILLATORY" if o["oscillatory"] else "MONOTONE", p_obs=p_obs, p_eff=p_eff,
               fs=fs, order_verified=verified, phi_ext=extrapolated(f1, f2, o["r21"], p_eff),
               phi_ext_p_obs=extrapolated(f1, f2, o["r21"], p_obs),
               gci_fine=gci_fine(f1, f2, o["r21"], p_eff, fs), gci_coarse=gci_coarse(f2, f3, o["r32"], p_eff, fs),
               gci_fine_p_obs=gci_fine(f1, f2, o["r21"], p_obs, 1.25),
               gci_coarse_p_obs=gci_coarse(f2, f3, o["r32"], p_obs, 1.25),
               range_rel=(max(phi)-min(phi))/abs(f1))
    return res
