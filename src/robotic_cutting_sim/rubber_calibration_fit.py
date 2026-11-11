"""P2-5 calibration fit API (ADR 0023 §23.2 decision 6; plan §3.1; task P5.3).

- ``ParameterSpace``: theta = (mu_eq_pa, g1, tau1_s, g2, tau2_s, mu_table) with the transforms u = log(mu_eq),
  logit(g1 / G), logit(g2 / (G - g1)) (stick-breaking with a declared total budget G < 1, so sum g < G and
  mu0 = mu_eq / (1 - sum g) stays under the step rule's mu0_max), log(tau1), log(tau2), mu_table (linear). Bounds are
  boxes in u. Density is measured (specimen mass and dimensions) and never fitted; beta is numerical (20), not fitted.
- ``Objective``: weighted residuals (pred - data) / sigma_eff built from **fit-split experiments only** of a v2
  manifest. It has no RL term and refuses any reward callable; a held-out experiment id, a sealed path or a manifest
  that is not v2 is refused.
- Methods: ``fit_trf`` (scipy ``least_squares`` method ``trf``, the FORMAL method; forward-difference Jacobian
  evaluated speculatively as ONE population of p + 1 candidates with the residual, several starts),
  ``CMAES`` (in-repo numpy reference, seeded; reported, not formal) and the ask/tell hook ``run_ask_tell(optimizer,
  ...)`` (``ask() -> u[lambda, p]``, ``tell(loss[lambda])``): the interface through which the external agentic
  optimizer or a BO method plugs in. No optimiser package is added.
- ``freeze_fit`` writes the fit receipt (theta_hat, covariance, sigma_hat, chi2, the fit experiments, the manifest
  freeze SHA, the commit) and its SHA-256; ``evaluate_heldout`` refuses unless the manifest is ``*FROZEN_FOR_HELDOUT``
  and every fit receipt names only fit-split experiments of that freeze; it is **one-shot per manifest freeze id**
  (ledger; ``HELDOUT_ALREADY_EVALUATED``; an audit re-run must be named and is labelled).
- ``guard_path`` refuses any input path under a ``sealed/`` directory.

Every output carries ``calibration_status: SIMULATOR_ONLY_UNCALIBRATED`` and, for synthetic data,
``validation_kind: NUMERICAL_VALIDATION_SYNTHETIC``. No recipe is promoted.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import rubber_calibration_experiments as rce
from . import rubber_calibration_v2 as rc2

PARAMS = ("mu_eq_pa", "g1", "tau1_s", "g2", "tau2_s", "mu_table")
LOG_PARAMS = ("mu_eq_pa", "tau1_s", "tau2_s")
LABELS = {"calibration_status": rc2.CALIBRATION_STATUS}
FIT_RECEIPT_SCHEMA = "rcs_rubber_fit_receipt_v1"
HELDOUT_SCHEMA = "rcs_rubber_heldout_receipt_v1"
_HERE = Path(__file__).resolve()


class CalibrationFitError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


def source_sha256():
    return {_HERE.name: hashlib.sha256(_HERE.read_bytes()).hexdigest()}


def guard_path(path):
    """Refuse any path with a ``sealed`` component (before or after resolving symlinks)."""
    p = Path(path)
    for candidate in (p, p.resolve()):
        if "sealed" in candidate.parts:
            raise CalibrationFitError("SEALED_PATH_REFUSED", f"{path} lies under a sealed/ directory")
    return p


def _sig(x):
    return 1.0/(1.0+math.exp(-x))


def _logit(p):
    if not 0.0 < p < 1.0:
        raise ValueError(f"logit argument {p} outside (0, 1)")
    return math.log(p/(1.0-p))


# ============================================================================ parameter space

class ParameterSpace:
    """See the module docstring. ``bounds``: {"mu_eq_pa": (lo, hi), "tau1_s": ..., "tau2_s": ..., "mu_table": ...,
    "g1": (lo, hi)} in theta, and ``g2_fraction``: (lo, hi) of g2 / (G - g1); ``g_total_max`` = G."""

    def __init__(self, bounds, g2_fraction=(0.02, 0.98), g_total_max=0.5):
        self.G = float(g_total_max)
        if not 0 < self.G < 1:
            raise ValueError("g_total_max must lie in (0, 1)")
        self.bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
        if set(self.bounds) != set(PARAMS)-{"g2"}:
            raise ValueError(f"bounds must give {sorted(set(PARAMS)-{'g2'})}")
        for k, (lo, hi) in self.bounds.items():
            if not 0 < lo < hi:
                raise ValueError(f"bounds {k}: ({lo}, {hi})")
        if not self.bounds["g1"][1] < self.G:
            raise ValueError("g1 upper bound must be < g_total_max")
        self.g2_fraction = (float(g2_fraction[0]), float(g2_fraction[1]))
        lb, ub = [], []
        for name in PARAMS:
            if name in LOG_PARAMS:
                lo, hi = self.bounds[name]
                lb.append(math.log(lo))
                ub.append(math.log(hi))
            elif name == "g1":
                lo, hi = self.bounds["g1"]
                lb.append(_logit(lo/self.G))
                ub.append(_logit(hi/self.G))
            elif name == "g2":
                lb.append(_logit(self.g2_fraction[0]))
                ub.append(_logit(self.g2_fraction[1]))
            else:
                lo, hi = self.bounds[name]
                lb.append(lo)
                ub.append(hi)
        self.lb, self.ub = np.array(lb), np.array(ub)
        self.p = len(PARAMS)

    def to_u(self, theta):
        g1 = float(theta["g1"])
        u = []
        for name in PARAMS:
            v = float(theta[name])
            if name in LOG_PARAMS:
                u.append(math.log(v))
            elif name == "g1":
                u.append(_logit(g1/self.G))
            elif name == "g2":
                u.append(_logit(v/(self.G-g1)))
            else:
                u.append(v)
        return np.array(u)

    def from_u(self, u):
        u = np.asarray(u, float)
        if u.shape != (self.p,):
            raise ValueError(f"u must have shape ({self.p},)")
        g1 = self.G*_sig(u[1])
        g2 = (self.G-g1)*_sig(u[3])
        return {"mu_eq_pa": math.exp(u[0]), "g1": g1, "tau1_s": math.exp(u[2]), "g2": g2, "tau2_s": math.exp(u[4]),
                "mu_table": float(u[5])}

    def jacobian_theta_u(self, u, h=1e-7):
        """d theta / d u (central differences of the closed-form transform)."""
        J = np.zeros((self.p, self.p))
        for i in range(self.p):
            e = np.zeros(self.p)
            e[i] = h
            a, b = self.from_u(u+e), self.from_u(u-e)
            J[:, i] = [(a[n]-b[n])/(2*h) for n in PARAMS]
        return J

    def inside(self, u, tol=1e-12):
        u = np.asarray(u, float)
        return bool(np.all(u >= self.lb-tol) and np.all(u <= self.ub+tol))

    def latin_hypercube(self, n, seed, box=None):
        """n points (u) of a seeded Latin hypercube over ``box`` (default the u bounds)."""
        lb, ub = (self.lb, self.ub) if box is None else box
        rng = np.random.default_rng(seed)
        out = np.empty((n, self.p))
        for i in range(self.p):
            perm = rng.permutation(n)
            out[:, i] = lb[i]+(ub[i]-lb[i])*(perm+rng.uniform(size=n))/n
        return out

    def mu0_max(self):
        return self.bounds["mu_eq_pa"][1]/(1.0-self.G)

    def receipt(self):
        return {"params": list(PARAMS), "bounds_theta": {k: list(v) for k, v in self.bounds.items()},
                "g2_fraction_of_remaining_budget": list(self.g2_fraction), "g_total_max": self.G,
                "transforms": {"mu_eq_pa": "log", "g1": "logit(g1 / G)", "g2": "logit(g2 / (G - g1))",
                               "tau1_s": "log", "tau2_s": "log", "mu_table": "linear"},
                "u_lower": self.lb.tolist(), "u_upper": self.ub.tolist(), "mu0_max_pa": self.mu0_max(),
                "not_fitted": {"density": "measured (specimen mass / measured dimensions)",
                               "beta": "numerical bulk ratio 20"}}


def space_from_protocol(protocol):
    ps = protocol["parameter_space"]
    return ParameterSpace({k: tuple(v) for k, v in ps["bounds_theta"].items()},
                          g2_fraction=tuple(ps["g2_fraction_of_remaining_budget"]), g_total_max=ps["g_total_max"])


# ============================================================================ objective

def experiment_definition(manifest_exp, specimen):
    """Forward-model experiment dict from a v2 manifest experiment and its specimen (nominal lattice geometry)."""
    return {"id": manifest_exp["experiment_id"], "type": manifest_exp["type"], "split": manifest_exp["split"],
            "specimen": dict(specimen["nominal_dimensions_m"]), "loading": dict(manifest_exp["loading"])}


def measured_density(specimen):
    d = specimen["dimensions_m"]
    return float(specimen["mass_kg"])/(float(d["length"])*float(d["width"])*float(d["thickness"]))


class Objective:
    """Weighted least-squares objective over the FIT split of a v2 manifest (module docstring).
    ``forward(thetas, experiments) -> [{exp_id: channels}]`` is the noise-free forward model (sensor transform
    applied); the default (``forward=None``) builds the population forward model of
    ``rubber_calibration_experiments`` on ``device``. ``experiments`` optionally restricts the fit to a subset of the
    fit split (the negative control). There is deliberately no reward argument."""

    def __init__(self, manifest_path, *, cfg=None, forward=None, population=7, device="cuda:0", experiments=None,
                 parity_mode=False, **refused):
        if refused:
            names = sorted(refused)
            code = "REWARD_TERM_REFUSED" if any("reward" in n.lower() for n in names) else "UNKNOWN_ARGUMENT"
            raise CalibrationFitError(code, f"Objective takes no {names} (no RL term in a calibration objective)")
        path = guard_path(manifest_path)
        manifest = rc2.load_manifest(path)
        self.manifest, self.manifest_path = manifest, path
        specimens = {s["specimen_id"]: s for s in manifest["specimens"]}
        fit_ids = list(manifest["splits"]["fit"])
        chosen = fit_ids if experiments is None else list(experiments)
        unknown = [e for e in chosen if e not in fit_ids]
        if unknown:
            raise CalibrationFitError("HELDOUT_EXPERIMENT_REFUSED",
                                      f"experiments {unknown} are not in the manifest's fit split")
        self.exp_ids = chosen
        by_id = {e["experiment_id"]: e for e in manifest["experiments"]}
        self.manifest_exps = [by_id[e] for e in chosen]
        self.experiments = [experiment_definition(by_id[e], specimens[by_id[e]["specimen_id"]]) for e in chosen]
        self.densities = {e["experiment_id"]: measured_density(specimens[e["specimen_id"]])
                          for e in self.manifest_exps}
        self.sensors = {e["experiment_id"]: e["sensors"] for e in self.manifest_exps}
        self.data = {}
        for e in self.manifest_exps:
            self.data[e["experiment_id"]] = {role: rce.read_channels(path.parent/guard_path(entry["file"]))
                                             for role, entry in e["data_files"].items()}
        self.cfg = cfg
        self.population = int(population)
        if forward is None:
            if cfg is None:
                raise ValueError("cfg (fit ModelConfig) is required for the default forward model")
            forward = PopulationForward(self.experiments, self.densities, cfg, self.population, device=device,
                                        parity_mode=parity_mode).bind_sensors(self.sensors)
        self.forward = forward
        self.n_residuals = None
        self.evaluations = 0
        self.candidates_evaluated = 0
        self.wall_s = 0.0

    def residuals_population(self, thetas):
        """Residual vectors of a list of theta dicts (one forward population)."""
        start = time.perf_counter()
        preds = self.forward(list(thetas), self.experiments)
        out = []
        for pred in preds:
            parts = []
            for exp in self.experiments:
                eid = exp["id"]
                ch = pred[eid]
                if ch.get("invalid_reason", 0):
                    n = rce.residuals(exp, self.data[eid], self.data[eid], self.sensors[eid]).size
                    parts.append(np.full(n, INVALID_RESIDUAL))
                else:
                    parts.append(rce.residuals(exp, ch, self.data[eid], self.sensors[eid]))
            out.append(np.concatenate(parts))
        self.n_residuals = out[0].size
        self.evaluations += 1
        self.candidates_evaluated += len(thetas)
        self.wall_s += time.perf_counter()-start
        return out

    def receipt(self):
        return {"manifest": str(self.manifest_path), "manifest_freeze_sha256": self.manifest["splits"]["sha256"],
                "dataset_id": self.manifest["dataset_id"], "fit_experiments": list(self.exp_ids),
                "densities_measured_kg_m3": self.densities, "rl_term": "none (refused)",
                "residual_definition": "(prediction - data) / sqrt(sigma^2 + resolution^2 / 12) per sensor sample",
                **LABELS, "validation_kind": self.manifest["validation_kind"]}


INVALID_RESIDUAL = 1.0e3     # residual of every sample of an experiment whose forward env set an invalid bit


class PopulationForward:
    """Default forward model: ``rce.PopulationEvaluator`` + observables + the noise-free sensor transform."""

    def __init__(self, experiments, densities, cfg, population, *, device="cuda:0", parity_mode=False, beta=None):
        self.experiments = experiments
        self.cfg = cfg
        self.beta = cfg.beta if beta is None else beta
        self.evaluator = rce.PopulationEvaluator(experiments, densities, cfg, population, device=device,
                                                 parity_mode=parity_mode)
        self.densities = densities
        self.sensor_rates = {}

    def channels(self, thetas, experiments, sensors_by_exp):
        cands = [{e["id"]: rce.material_from_theta(th, self.densities[e["id"]], self.beta) for e in experiments}
                 for th in thetas]
        records = self.evaluator.evaluate(cands)
        out = []
        for rec in records:
            d = {}
            for e in experiments:
                obs = rce.observables(e, rec[e["id"]], self.cfg)
                ch = rce.sample_channels(e, obs, sensors_by_exp[e["id"]])
                ch["invalid_reason"] = obs["invalid_reason"]
                ch["oracle"] = rce.oracle(obs)
                d[e["id"]] = ch
            out.append(d)
        return out

    def __call__(self, thetas, experiments):
        return self.channels(thetas, experiments, self.sensors)

    def bind_sensors(self, sensors):
        self.sensors = sensors
        return self

    def receipts(self):
        return self.evaluator.receipts()


# ============================================================================ TRF (formal method)

class SpeculativeJacobian:
    """fun / jac pair for scipy: every ``fun(u)`` evaluates the population [u, u + h e_1, ..., u + h e_p] at once and
    caches the forward-difference Jacobian for ``jac(u)`` at the same u (bitwise key)."""

    def __init__(self, objective, space, fd_step, log=None):
        self.objective, self.space, self.h = objective, space, float(fd_step)
        self.cache = {}
        self.history = []
        self.evaluated = []          # (u, r, J) of every evaluation (not written to receipts)
        self.log = log

    def _evaluate(self, u):
        u = np.asarray(u, float)
        pts = [u]
        steps = []
        for i in range(self.space.p):
            h = self.h if u[i]+self.h <= self.space.ub[i] else -self.h     # stay inside the box
            e = u.copy()
            e[i] += h
            pts.append(e)
            steps.append(h)
        res = self.objective.residuals_population([self.space.from_u(x) for x in pts])
        r0 = res[0]
        J = np.column_stack([(res[i+1]-r0)/steps[i] for i in range(self.space.p)])
        key = u.tobytes()
        self.cache = {key: (r0, J)}
        cost = 0.5*float(r0 @ r0)
        self.evaluated.append((u.copy(), r0, J))
        self.history.append({"u": u.tolist(), "theta": self.space.from_u(u), "cost": cost,
                             "chi2_reduced": float(r0 @ r0)/max(1, r0.size-self.space.p)})
        if self.log:
            self.log(f"eval {len(self.history)} cost {cost:.6e} theta {self.space.from_u(u)}")
        return r0, J

    def fun(self, u):
        key = np.asarray(u, float).tobytes()
        if key in self.cache:
            return self.cache[key][0]
        return self._evaluate(u)[0]

    def jac(self, u, *args):
        key = np.asarray(u, float).tobytes()
        if key not in self.cache:
            self._evaluate(u)
        return self.cache[key][1]


def fit_trf(objective, space, starts_u, *, max_nfev=40, xtol=1e-4, ftol=1e-8, gtol=1e-8, fd_step=1e-3, log=None,
            x_scale=1.0):
    """Formal method: scipy ``least_squares(method="trf")`` from every start (u); returns the per-start results and
    the best (lowest cost). Convergence: scipy's status > 0 with ``xtol`` (relative step) or the nfev budget."""
    from scipy.optimize import least_squares
    results = []
    for i, u0 in enumerate(starts_u):
        sj = SpeculativeJacobian(objective, space, fd_step, log=(lambda m, i=i: log(f"[start {i}] {m}")) if log
                                 else None)
        t0 = time.perf_counter()
        sol = least_squares(sj.fun, np.asarray(u0, float), jac=sj.jac, bounds=(space.lb, space.ub), method="trf",
                            x_scale=x_scale, xtol=xtol, ftol=ftol, gtol=gtol, max_nfev=max_nfev)
        r = sol.fun
        n, p = r.size, space.p
        results.append({"start": i, "u0": np.asarray(u0, float).tolist(), "theta0": space.from_u(u0),
                        "u": sol.x.tolist(), "theta": space.from_u(sol.x), "cost": float(sol.cost),
                        "chi2": float(r @ r), "chi2_reduced": float(r @ r)/(n-p), "n_residuals": n,
                        "status": int(sol.status), "message": str(sol.message), "nfev": int(sol.nfev),
                        "njev": int(sol.njev or 0), "optimality": float(sol.optimality),
                        "active_mask": sol.active_mask.tolist(), "jac": sol.jac, "residuals": r,
                        "converged_xtol": int(sol.status) in (2, 4), "budget_exhausted": int(sol.status) == 0,
                        "history": sj.history, "wall_s": time.perf_counter()-t0})
    best = min(results, key=lambda x: x["cost"])
    return {"method": "scipy.optimize.least_squares trf", "formal": True, "starts": results, "best": best["start"],
            "settings": {"max_nfev": max_nfev, "xtol": xtol, "ftol": ftol, "gtol": gtol, "fd_step_u": fd_step,
                         "jacobian": "forward differences in u, one population of p + 1 per evaluation"}}


class LockstepPool:
    """Evaluates the requests of several optimiser threads as ONE forward population: a worker's ``request(thetas)``
    blocks until every active worker has posted its request (or finished); the coordinator (the calling thread of
    ``run``) concatenates them, calls ``objective.residuals_population`` once and hands each worker its slice. The
    result of every worker is identical to running it alone (the forward model is per candidate)."""

    def __init__(self, objective, workers):
        import threading
        self.objective = objective
        self.cv = threading.Condition()
        self.pending = {}
        self.results = {}
        self.active = set(range(workers))
        self.errors = {}
        self.population_sizes = []

    def request(self, wid, thetas):
        with self.cv:
            self.pending[wid] = list(thetas)
            self.cv.notify_all()
            while wid not in self.results and not self.errors:
                self.cv.wait()
            if self.errors and wid not in self.results:
                raise RuntimeError(f"lockstep coordinator failed: {self.errors}")
            return self.results.pop(wid)

    def finish(self, wid, error=None):
        with self.cv:
            self.active.discard(wid)
            if error is not None:
                self.errors[wid] = error
            self.cv.notify_all()

    def run(self):
        while True:
            with self.cv:
                while self.active and set(self.pending) != self.active:
                    self.cv.wait()
                if not self.active:
                    return
                order = sorted(self.pending)
                batch = [(w, self.pending.pop(w)) for w in order]
            flat = [th for _, ths in batch for th in ths]
            try:
                res = self.objective.residuals_population(flat)
            except Exception as exc:             # recorded, re-raised in every worker
                with self.cv:
                    self.errors["coordinator"] = repr(exc)
                    self.cv.notify_all()
                raise
            self.population_sizes.append(len(flat))
            with self.cv:
                i = 0
                for w, ths in batch:
                    self.results[w] = res[i:i+len(ths)]
                    i += len(ths)
                self.cv.notify_all()


class _WorkerObjective:
    def __init__(self, pool, wid):
        self.pool, self.wid = pool, wid

    def residuals_population(self, thetas):
        return self.pool.request(self.wid, thetas)


def fit_trf_lockstep(objective, space, starts_u, **kw):
    """``fit_trf`` with every start in its own thread and every evaluation of all active starts run as one forward
    population (``LockstepPool``). Same per-start results as ``fit_trf``; fewer, larger forward calls."""
    import threading
    pool = LockstepPool(objective, len(starts_u))
    out = [None]*len(starts_u)

    def work(i):
        kwi = dict(kw)
        if kw.get("log"):
            kwi["log"] = lambda m, i=i: kw["log"](m.replace("[start 0]", f"[start {i}]", 1))
        try:
            out[i] = fit_trf(_WorkerObjective(pool, i), space, [starts_u[i]], **kwi)["starts"][0]
            out[i]["start"] = i
            pool.finish(i)
        except Exception as exc:
            pool.finish(i, repr(exc))
            raise
    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(starts_u))]
    for th in threads:
        th.start()
    pool.run()
    for th in threads:
        th.join()
    if pool.errors:
        raise RuntimeError(f"lockstep fit failed: {pool.errors}")
    best = min(out, key=lambda x: x["cost"])
    return {"method": "scipy.optimize.least_squares trf", "formal": True, "starts": out, "best": best["start"],
            "settings": {**out[0].get("settings", {}), "max_nfev": kw.get("max_nfev", 40),
                         "xtol": kw.get("xtol", 1e-4), "ftol": kw.get("ftol", 1e-8), "gtol": kw.get("gtol", 1e-8),
                         "fd_step_u": kw.get("fd_step", 1e-3),
                         "jacobian": "forward differences in u, one population of p + 1 per start and evaluation",
                         "lockstep": f"{len(starts_u)} starts evaluated together"},
            "population_sizes": pool.population_sizes}


def covariance(space, u_hat, J_u, residuals, rcond=1e-10):
    """Gauss-Newton covariance: cov_u = s^2 (J^T J)^-1, s^2 = chi2 / (n - p), by eigen-decomposition; directions with
    an eigenvalue below ``rcond`` x the largest are unidentifiable and get infinite variance (never the zero variance
    of a pseudo-inverse). cov_theta = D cov_u D^T with D = d theta / d u (delta method). Returns
    (cov_theta, sigma_theta, s2); an unidentifiable parameter has sigma = inf."""
    r = np.asarray(residuals, float)
    n, p = r.size, space.p
    s2 = float(r @ r)/(n-p)
    lam, V = np.linalg.eigh(J_u.T @ J_u)
    lam_max = max(float(lam.max()), 0.0)
    good = lam > rcond*lam_max if lam_max > 0 else np.zeros(p, bool)
    D = space.jacobian_theta_u(np.asarray(u_hat, float))
    DV = D @ V
    cov = (DV[:, good]*(s2/lam[good])) @ DV[:, good].T
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    for i in range(p):
        if np.any(np.abs(DV[i, ~good]) > 1e-12*max(1.0, float(np.abs(D[i]).max()))):
            sigma[i] = np.inf
            cov[i, :] = cov[:, i] = np.inf
    return cov, sigma, s2


# ============================================================================ CMA-ES and ask / tell

class CMAES:
    """Seeded (mu/mu_w, lambda) CMA-ES in u with box handling by projection (Hansen's standard parameter setting;
    numpy only). Implements the ask / tell interface (``ask() -> u[lambda, p]``, ``tell(loss[lambda])``)."""

    def __init__(self, x0, sigma0, popsize, seed, lb=None, ub=None):
        self.n = n = len(x0)
        self.mean = np.asarray(x0, float).copy()
        self.sigma = float(sigma0)
        self.lam = int(popsize)
        self.mu = self.lam//2
        w = np.log(self.mu+0.5)-np.log(np.arange(1, self.mu+1))
        self.w = w/w.sum()
        self.mueff = 1.0/np.sum(self.w**2)
        self.cc = (4+self.mueff/n)/(n+4+2*self.mueff/n)
        self.cs = (self.mueff+2)/(n+self.mueff+5)
        self.c1 = 2/((n+1.3)**2+self.mueff)
        self.cmu = min(1-self.c1, 2*(self.mueff-2+1/self.mueff)/((n+2)**2+self.mueff))
        self.damps = 1+2*max(0, math.sqrt((self.mueff-1)/(n+1))-1)+self.cs
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D, self.C = np.eye(n), np.ones(n), np.eye(n)
        self.chin = math.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
        self.rng = np.random.default_rng(seed)
        self.lb = None if lb is None else np.asarray(lb, float)
        self.ub = None if ub is None else np.asarray(ub, float)
        self.generation = 0
        self._pending = None
        self.best = (math.inf, None)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        y = z @ np.diag(self.D) @ self.B.T
        x = self.mean+self.sigma*y
        if self.lb is not None:
            x = np.clip(x, self.lb, self.ub)
        self._pending = x
        return x.copy()

    def tell(self, losses):
        if self._pending is None:
            raise RuntimeError("tell() without ask()")
        x, f = self._pending, np.asarray(losses, float)
        self._pending = None
        order = np.argsort(f, kind="stable")
        if f[order[0]] < self.best[0]:
            self.best = (float(f[order[0]]), x[order[0]].copy())
        old = self.mean.copy()
        xs = x[order[:self.mu]]
        self.mean = self.w @ xs
        y = (self.mean-old)/self.sigma
        invsqrt = self.B @ np.diag(1/self.D) @ self.B.T
        self.ps = (1-self.cs)*self.ps+math.sqrt(self.cs*(2-self.cs)*self.mueff)*(invsqrt @ y)
        hsig = (np.linalg.norm(self.ps)/math.sqrt(1-(1-self.cs)**(2*(self.generation+1)))/self.chin
                < 1.4+2/(self.n+1))
        self.pc = (1-self.cc)*self.pc+hsig*math.sqrt(self.cc*(2-self.cc)*self.mueff)*y
        artmp = (xs-old)/self.sigma
        self.C = ((1-self.c1-self.cmu)*self.C+self.c1*(np.outer(self.pc, self.pc)
                                                        + (1-hsig)*self.cc*(2-self.cc)*self.C)
                  + self.cmu*(artmp.T @ np.diag(self.w) @ artmp))
        self.sigma *= math.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chin-1))
        self.C = np.triu(self.C)+np.triu(self.C, 1).T
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.clip(d2, 1e-30, None))
        self.generation += 1


def run_ask_tell(optimizer, objective, space, generations, *, log=None):
    """Drive any ask / tell optimizer (``ask() -> u[lambda, p]``, ``tell(loss[lambda])``) against the objective; the
    loss is 1/2 |r|^2. This is the plug-in point of the external agentic optimizer or a BO method; their candidates are
    projected into the u box. Returns the history and the best point."""
    hist = []
    best = (math.inf, None)
    for g in range(int(generations)):
        U = np.asarray(optimizer.ask(), float)
        if U.ndim != 2 or U.shape[1] != space.p:
            raise CalibrationFitError("ASK_SHAPE", f"ask() returned shape {U.shape}, expected [lambda, {space.p}]")
        U = np.clip(U, space.lb, space.ub)
        res = objective.residuals_population([space.from_u(u) for u in U])
        losses = np.array([0.5*float(r @ r) for r in res])
        optimizer.tell(losses)
        i = int(np.argmin(losses))
        if losses[i] < best[0]:
            best = (float(losses[i]), U[i].copy())
        hist.append({"generation": g, "best_loss": float(losses.min()), "median_loss": float(np.median(losses)),
                     "best_theta": space.from_u(U[i])})
        if log:
            log(f"gen {g} best {losses.min():.6e} running best {best[0]:.6e}")
    return {"history": hist, "best_loss": best[0], "best_u": best[1].tolist(),
            "best_theta": space.from_u(best[1])}


# ============================================================================ freeze, held-out

def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def freeze_fit(path, *, objective, space, trf, provenance, extra=None):
    """Write the fit receipt (never overwrites) and return (receipt, sha256). The best TRF start gives theta_hat and
    the covariance; every start is listed without its Jacobian and residuals."""
    path = Path(path)
    if path.exists():
        raise CalibrationFitError("RECEIPT_EXISTS", f"{path} exists (no overwrite)")
    best = trf["starts"][trf["best"]]
    cov, sigma, s2 = covariance(space, best["u"], best["jac"], best["residuals"])
    starts = [{k: v for k, v in s.items() if k not in ("jac", "residuals")} for s in trf["starts"]]
    cfg = getattr(objective, "cfg", None)
    resolution = None if cfg is None else {
        "label": "RESOLUTION_SPECIFIC", "resolution_profile": cfg.resolution_profile, "n_t": cfg.n_t, "cfl": cfg.cfl,
        "statement": "theta_hat is specific to this fit-model resolution; rubber_ref_v1 (n_t 4) carries the measured "
                     "wide-strip sag bias Q1w +0.62 % (+0.50 ... +0.78 %; known_limitation "
                     "rubber_phys_11_v5_q1w_gci_2.16pct); never carry theta_hat to n_t 3 or 6 without the bias shift "
                     "(K2 review, Q1w condition 4)"}
    receipt = {"schema": FIT_RECEIPT_SCHEMA, **LABELS, "validation_kind": objective.manifest["validation_kind"],
               "resolution": resolution,
               "recipe_promoted": False, "objective": objective.receipt(), "parameter_space": space.receipt(),
               "method": trf["method"], "formal_method": True, "settings": trf["settings"],
               "theta_hat": best["theta"], "u_hat": best["u"], "covariance_theta": cov.tolist(),
               "sigma_hat": dict(zip(PARAMS, sigma.tolist())), "s2_reduced_chi2": s2,
               "chi2": best["chi2"], "chi2_reduced": best["chi2_reduced"], "n_residuals": best["n_residuals"],
               "best_start": trf["best"], "starts": starts, "provenance": provenance, **(extra or {})}
    text = json.dumps(_jsonable(receipt), indent=1, sort_keys=True)+"\n"
    path.write_text(text)
    return receipt, hashlib.sha256(text.encode()).hexdigest()


def evaluate_heldout(manifest_path, fit_receipt_paths, ledger_dir, *, predict, thresholds, metric_settings,
                     audit_rerun=None, provenance=None):
    """One-shot held-out evaluation of one or more frozen fits (the primary fit first, then e.g. the negative control)
    against the frozen manifest. ``predict(theta, experiments) -> {exp_id: channels incl. 'oracle'}`` is the fit
    model; ``thresholds``: {exp_id: {metric: tol}} (the SHA-pinned thresholds file). Refusals: manifest not
    ``*FROZEN_FOR_HELDOUT`` or freeze hash mismatch; a fit receipt for another freeze or naming a held-out experiment;
    a second evaluation of the same freeze (``HELDOUT_ALREADY_EVALUATED``) unless ``audit_rerun`` names it (then the
    result is labelled ``AUDIT_RERUN`` and is not the one-shot result); sealed paths."""
    mpath = guard_path(manifest_path)
    manifest = rc2.load_manifest(mpath)
    if manifest["status"] not in rc2.FROZEN_STATUSES:
        raise CalibrationFitError("MANIFEST_NOT_FROZEN", f"status {manifest['status']!r}")
    freeze = manifest["splits"]["sha256"]
    ledger_dir = guard_path(ledger_dir)
    ledger = Path(ledger_dir)/f"heldout_{freeze}.json"
    if ledger.exists() and not audit_rerun:
        raise CalibrationFitError("HELDOUT_ALREADY_EVALUATED", f"freeze {freeze[:12]}… was evaluated ({ledger})")
    fit_split, held = set(manifest["splits"]["fit"]), list(manifest["splits"]["heldout"])
    receipts = []
    for p in fit_receipt_paths:
        p = guard_path(p)
        rec = json.loads(Path(p).read_text())
        if rec.get("schema") != FIT_RECEIPT_SCHEMA:
            raise CalibrationFitError("FIT_RECEIPT_SCHEMA", f"{p}")
        if rec["objective"]["manifest_freeze_sha256"] != freeze:
            raise CalibrationFitError("FIT_RECEIPT_OTHER_FREEZE", f"{p} was fitted on another manifest freeze")
        bad = set(rec["objective"]["fit_experiments"])-fit_split
        if bad:
            raise CalibrationFitError("FIT_USED_HELDOUT", f"{p} names non-fit experiments {sorted(bad)}")
        receipts.append((p, rec, _sha_file(p)))
    specimens = {s["specimen_id"]: s for s in manifest["specimens"]}
    by_id = {e["experiment_id"]: e for e in manifest["experiments"]}
    exps = [experiment_definition(by_id[e], specimens[by_id[e]["specimen_id"]]) for e in held]
    data = {e: {role: rce.read_channels(mpath.parent/entry["file"]) for role, entry in by_id[e]["data_files"].items()}
            for e in held}
    oracle_file = manifest.get("synthetic", {}).get("truth_profile", {}).get("oracle_file")
    oracle_all = json.loads((mpath.parent/oracle_file).read_text()) if oracle_file else {}
    t0 = time.perf_counter()
    fits = []
    for p, rec, sha in receipts:
        pred = predict(rec["theta_hat"], exps)
        rows, verdict = [], "PASS"
        for exp in exps:
            eid = exp["id"]
            ch = pred[eid]
            m = rce.metrics(exp, ch, data[eid], ch.get("oracle"), oracle_all.get(eid), metric_settings)
            for name, value in sorted(m.items()):
                tol = thresholds[eid][name]
                ok = bool(np.isfinite(value) and value <= tol)
                rows.append({"experiment": eid, "metric": name, "value": float(value), "tolerance": float(tol),
                             "verdict": "PASS" if ok else "FAIL", "invalid_reason": int(ch.get("invalid_reason", 0))})
                if not ok:
                    verdict = "FAIL"
        fits.append({"fit_receipt": str(p), "fit_receipt_sha256": sha, "fit_experiments":
                     rec["objective"]["fit_experiments"], "theta_hat": rec["theta_hat"], "verdict": verdict,
                     "rows": rows})
    out = {"schema": HELDOUT_SCHEMA, **LABELS, "validation_kind": manifest["validation_kind"],
           "recipe_promoted": False, "manifest": str(mpath), "manifest_freeze_sha256": freeze,
           "dataset_id": manifest["dataset_id"], "heldout_experiments": held, "fits": fits,
           "verdict": fits[0]["verdict"] if fits else "NOT_RUN", "audit_rerun": audit_rerun,
           "label": "AUDIT_RERUN (not the one-shot result)" if audit_rerun else "ONE_SHOT",
           "wall_s": time.perf_counter()-t0, "provenance": provenance or {}}
    text = json.dumps(_jsonable(out), indent=1, sort_keys=True)+"\n"
    if audit_rerun:
        target = Path(ledger_dir)/f"heldout_{freeze}_audit_{audit_rerun}.json"
        if target.exists():
            raise CalibrationFitError("AUDIT_NAME_USED", str(target))
    else:
        target = ledger
    Path(ledger_dir).mkdir(parents=True, exist_ok=True)
    with open(target, "x") as fh:          # exclusive create: a race cannot evaluate twice
        fh.write(text)
    return out


# ============================================================================ v2: staged fit with a compute budget

class SubSpace:
    """A ParameterSpace restricted to ``active`` parameters (the others fixed at ``fixed`` theta values) and to a box
    in u (default: the parent's bounds). ``from_u`` returns the FULL theta dict, so the objective is unchanged."""

    def __init__(self, parent, active, fixed, box=None):
        self.parent, self.active = parent, list(active)
        self.idx = [PARAMS.index(n) for n in self.active]
        self.fixed_u = parent.to_u(fixed)
        lb, ub = (parent.lb, parent.ub) if box is None else box
        self.lb, self.ub = np.asarray(lb, float)[self.idx], np.asarray(ub, float)[self.idx]
        self.p = len(self.idx)

    def full_u(self, u):
        full = self.fixed_u.copy()
        full[self.idx] = np.asarray(u, float)
        return full

    def from_u(self, u):
        return self.parent.from_u(self.full_u(u))

    def to_u(self, theta):
        return self.parent.to_u(theta)[self.idx]

    def inside(self, u, tol=1e-12):
        u = np.asarray(u, float)
        return bool(np.all(u >= self.lb-tol) and np.all(u <= self.ub+tol))

    def latin_hypercube(self, n, seed):
        rng = np.random.default_rng(seed)
        out = np.empty((n, self.p))
        for i in range(self.p):
            perm = rng.permutation(n)
            out[:, i] = self.lb[i]+(self.ub[i]-self.lb[i])*(perm+rng.uniform(size=n))/n
        return out


class BudgetExhausted(RuntimeError):
    pass


class BudgetedObjective:
    """Wraps an objective: a new forward population is refused (``BudgetExhausted``) when the elapsed wall-clock time
    plus the duration of the previous evaluation would exceed ``deadline`` (time.monotonic seconds)."""

    def __init__(self, objective, deadline, min_evals=0):
        self.objective, self.deadline, self.last = objective, float(deadline), 0.0
        self.count, self.min_evals = 0, int(min_evals)

    def residuals_population(self, thetas):
        now = time.monotonic()
        if self.count >= self.min_evals and now+self.last > self.deadline:
            raise BudgetExhausted(f"wall-clock budget: {now:.0f} + {self.last:.0f} s > deadline {self.deadline:.0f}")
        out = self.objective.residuals_population(thetas)
        self.last = time.monotonic()-now
        self.count += 1
        return out


def _run_stage(objective, sub, starts, *, max_nfev, fd_step, xtol, ftol, gtol, x_scale, log, deadline, min_evals=0):
    """TRF over a SubSpace from one or more starts (lockstep); every evaluation is kept, the budget may end the stage
    early (best iterate kept). Returns (best (u_sub, r, J, cost, start), per-start records, budget_hit)."""
    import threading
    bobj = BudgetedObjective(objective, deadline, min_evals)
    pool = LockstepPool(bobj, len(starts))
    sjs = [SpeculativeJacobian(_WorkerObjective(pool, i), sub, fd_step,
                               log=(lambda m, i=i: log(f"[start {i}] {m}")) if log else None)
           for i in range(len(starts))]
    recs = [None]*len(starts)
    budget = {"hit": False}

    def work(i):
        from scipy.optimize import least_squares
        try:
            sol = least_squares(sjs[i].fun, np.asarray(starts[i], float), jac=sjs[i].jac, bounds=(sub.lb, sub.ub),
                                method="trf", x_scale=x_scale, xtol=xtol, ftol=ftol, gtol=gtol, max_nfev=max_nfev)
            recs[i] = {"status": int(sol.status), "message": str(sol.message), "nfev": int(sol.nfev)}
            pool.finish(i)
        except BudgetExhausted as exc:
            budget["hit"] = True
            recs[i] = {"status": "BUDGET_EXHAUSTED", "message": str(exc), "nfev": len(sjs[i].evaluated)}
            pool.finish(i)
        except RuntimeError as exc:
            if "lockstep coordinator failed" in str(exc) and "BudgetExhausted" in str(exc):
                budget["hit"] = True
                recs[i] = {"status": "BUDGET_EXHAUSTED", "message": str(exc), "nfev": len(sjs[i].evaluated)}
                pool.finish(i)
            else:
                pool.finish(i, repr(exc))
                raise
        except Exception as exc:
            pool.finish(i, repr(exc))
            raise
    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(starts))]
    for th in threads:
        th.start()
    try:
        pool.run()
    except BudgetExhausted:
        budget["hit"] = True
    for th in threads:
        th.join()
    hard = {k: v for k, v in pool.errors.items() if "BudgetExhausted" not in str(v) and k != "coordinator"}
    if hard:
        raise RuntimeError(f"stage failed: {hard}")
    best = None
    for i, sj in enumerate(sjs):
        recs[i] = {**(recs[i] or {}), "start": i, "u0": np.asarray(starts[i], float).tolist(),
                   "history": sj.history}
        for u, r, J in sj.evaluated:
            cost = 0.5*float(r @ r)
            if best is None or cost < best[3]:
                best = (u, r, J, cost, i)
    return best, recs, budget["hit"]


V2_STAGES = (
    {"name": "S1_elastic", "params": ["mu_eq_pa"], "experiments": ["T1"], "starts": 1, "max_nfev": 8},
    {"name": "S2_visco", "params": ["g1", "tau1_s", "g2", "tau2_s"], "experiments": ["R1", "T2"], "starts": 3,
     "max_nfev": 15},
    {"name": "S3_friction", "params": ["mu_table"], "experiments": ["F1"], "starts": 1, "max_nfev": 8},
    {"name": "S4_joint_polish", "params": list(PARAMS), "experiments": None, "starts": 1, "max_nfev": 6},
)


def fit_staged(make_objective, space, box, *, stages=V2_STAGES, start_theta, budget_s, seed=2513, fd_step=1e-3,
               xtol=1e-4, ftol=1e-8, gtol=1e-8, x_scale="jac", log=None):
    """v2 staged fit (protocol rubber_calib_tests_v2): the stages run in order, each a bounded TRF (``x_scale``,
    bounds = ``box`` in u, i.e. the prior box) over its parameters with the others fixed at the current estimate, on its
    experiments (``make_objective(experiment_ids or None)`` -> Objective; None = the whole fit split). Multi-start
    stages use a seeded Latin hypercube over their box (lockstep). The wall-clock budget ``budget_s`` covers every
    stage; when it is exhausted the best iterate so far of the current stage is kept and the remaining stages are
    skipped, except that the final stage always gets at least one evaluation (its first population gives the full
    residual and Jacobian at the current estimate). The result is the best evaluated point of the final stage."""
    t0 = time.monotonic()
    deadline = t0+float(budget_s)
    theta = dict(start_theta)
    out = {"stages": [], "budget_s": float(budget_s), "budget_hit": False}
    final = None
    for k, st in enumerate(stages):
        last = k == len(stages)-1
        if time.monotonic() > deadline and not last:
            out["stages"].append({"name": st["name"], "skipped": "BUDGET_EXHAUSTED"})
            out["budget_hit"] = True
            continue
        sub = SubSpace(space, st["params"], theta, box=box)
        starts = ([sub.to_u(theta)] if st["starts"] == 1
                  else list(sub.latin_hypercube(st["starts"], seed+k)))
        obj = make_objective(st["experiments"])
        ts = time.monotonic()
        stage_deadline = deadline
        best, recs, hit = _run_stage(obj, sub, starts, max_nfev=st["max_nfev"], fd_step=fd_step, xtol=xtol,
                                     ftol=ftol, gtol=gtol, x_scale=x_scale,
                                     log=(lambda m, n=st["name"]: log(f"{n} {m}")) if log else None,
                                     deadline=stage_deadline, min_evals=1 if last else 0)
        out["budget_hit"] |= hit
        theta = sub.from_u(best[0])
        out["stages"].append({"name": st["name"], "params": st["params"], "experiments": obj.exp_ids,
                              "starts": recs, "best_start": best[4], "best_cost": best[3], "theta": dict(theta),
                              "budget_hit": hit, "wall_s": time.monotonic()-ts,
                              "n_residuals": int(best[1].size)})
        if last:
            final = (sub, best, obj)
    sub, best, obj = final
    u_full = sub.full_u(best[0])
    r, J = best[1], best[2]
    n, p = r.size, space.p
    out.update(theta=space.from_u(u_full), u=u_full.tolist(), residuals=r, jac=J, cost=0.5*float(r @ r),
               chi2=float(r @ r), chi2_reduced=float(r @ r)/(n-p), n_residuals=n, objective=obj,
               wall_s=time.monotonic()-t0)
    return out


def freeze_fit_staged(path, *, result, space, provenance, protocol_id, extra=None):
    """Fit receipt of a v2 staged fit (same schema keys as ``freeze_fit`` plus ``stages``; never overwrites)."""
    obj = result["objective"]
    final = result["stages"][-1]
    trf = {"method": f"staged scipy.optimize.least_squares trf ({protocol_id})", "best": 0,
           "settings": {"stages": [{k: v for k, v in s.items() if k in ("name", "params", "experiments")}
                                   for s in result["stages"]], "budget_s": result["budget_s"],
                        "budget_hit": result["budget_hit"], "x_scale": "jac", "bounds": "prior box (u)"},
           "starts": [{"start": 0, "theta0": final["starts"][0]["history"][0]["theta"] if final.get("starts") else None,
                       "u": result["u"], "theta": result["theta"], "cost": result["cost"], "chi2": result["chi2"],
                       "chi2_reduced": result["chi2_reduced"], "n_residuals": result["n_residuals"],
                       "status": final["starts"][0].get("status") if final.get("starts") else None,
                       "message": final["starts"][0].get("message") if final.get("starts") else None,
                       "nfev": sum(len(s["history"]) for s in final.get("starts", [])),
                       "converged_xtol": final["starts"][0].get("status") in (2, 4) if final.get("starts") else False,
                       "budget_exhausted": bool(result["budget_hit"]), "jac": result["jac"],
                       "residuals": result["residuals"], "wall_s": result["wall_s"]}]}
    return freeze_fit(path, objective=obj, space=space, trf=trf, provenance=provenance,
                      extra={"stages": [{k: v for k, v in s.items()} for s in result["stages"]],
                             "budget_hit": result["budget_hit"], **(extra or {})})
