"""DIAGNOSTIC development sweep (P2-4 decision, ADR 0023 §22.1; NOT a protocol result): block 12 x 6 x 3 mm (n_t 3) on
the table z = 0 under tilted gravity ramped over 20 ms (``smooth=1``: updated every substep), 80 ms, Warp CPU.

    CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/rubber/p2_4/diag_contact_sweep.py law=pen kn=0.25 kt=0.25 \
        zeta=0.5 zetan=0.5 ratio=0.8 mu=0.5 [nt=3] [cfl=0.5] [smooth=1] [visco=1] [T=0.08]

Prints one JSON line: fitted acceleration on [0.04, T] vs the Coulomb value, creep after the ramp, vibration KE,
stick / slip node events, max penetration. The raw outputs of the 2026-11-11 sweeps (and the code state of each) are in
``artifacts/rl_ready/p2_4/contact_sweeps_dev_001``.
"""
import math, sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]/"src"))
import warp as wp
wp.config.log_level = wp.LOG_WARNING
from robotic_cutting_sim import rubber_scene as rs, rubber_material as rm
from robotic_cutting_sim.rubber_contact import ContactSpec, PENALTY_LAW, PROJECTION_LAW
from robotic_cutting_sim.tlmpm_rubber_contact_v1 import TLMPMRubberContact
a = dict(arg.split("=") for arg in sys.argv[1:])
law = PENALTY_LAW if a.get("law", "pen") == "pen" else PROJECTION_LAW
ratio, mu = float(a.get("ratio", 0.8)), float(a.get("mu", 0.5))
kn, kt, cfl, nt = float(a.get("kn", 0.25)), float(a.get("kt", 0.25)), float(a.get("cfl", 0.5)), int(a.get("nt", 3))
T = float(a.get("T", 0.08))
m = rm.MaterialValues(mu0=4e5, kappa=8e6, density=1150, friction_table=mu, prony=((0.2, 0.005), (0.15, 0.05)) if a.get("visco") == "1" else ())
sc = rs.compile_strip(rs.RubberStripSpec(length=0.012, width=0.006, thickness=0.003, n_t=nt, case="free"))
theta = math.atan(ratio*mu)
gvec = np.array([9.81*math.sin(theta), 0, -9.81*math.cos(theta)])
rule = rm.step_rule(sc.dx, 0.001, mu0_max=m.mu0, kappa_max=m.kappa, density_min=m.density, cfl=cfl)
s = TLMPMRubberContact(sc, m, 0.001/rule["substeps_per_control_step"], substeps=rule["substeps_per_control_step"], gravity=(0,0,0), contact=ContactSpec(planes=((0,0,1,0),), law=law, normal_stiffness=kn, tangential_stiffness=kt, damping_ratio=float(a.get("zeta", 0.0)), normal_damping_ratio=float(a.get("zetan", 0.0))))
s.reset()
M = s.mass_value*sc.n
xs = []
for k in range(int(round(T/1e-3))):
    tm = (k+0.5)*1e-3; r = 0.5*(1-math.cos(math.pi*tm/0.02)) if tm < 0.02 else 1.0
    s.gravity = wp.vec3(*(gvec*r).tolist())
    s.begin_control_step()
    for j in range(s.substeps):
        if a.get("smooth") == "1":
            ts_ = k*1e-3+(j+0.5)*s.dt; r_ = 0.5*(1-math.cos(math.pi*ts_/0.02)) if ts_ < 0.02 else 1.0
            s.gravity = wp.vec3(*(gvec*r_).tolist())
        s.step()
    xs.append(float(s.x.numpy()[:,0].astype(np.float64).mean()))
d = s.diagnostics.read()[0]; v = s.v.numpy().astype(np.float64); vcm = v.mean(0)
cd = s.contact_diagnostics()
ts = (np.arange(len(xs))+1)*1e-3; xa = np.array(xs)-xs[0]; sel = ts >= 0.04
c = np.polyfit(ts[sel], xa[sel], 2)
a_ref = 9.81*(math.sin(theta)-mu*math.cos(theta)) if ratio > 1 else 0.0
print(json.dumps({"args": a, "dt": s.dt, "a_fit": 2*c[0], "a_ref": a_ref, "Dfric": cd["friction_dissipation"], "muNd": mu*M*9.81*math.cos(theta)*(xa[-1]-xa[39]), "Dfric_window": None, "creep_after_ramp_m": xs[-1]-xs[19], "creep_last20ms_m": xs[-1]-xs[-21], "KEvib": d["kinetic_particle"]-0.5*M*float(vcm@vcm), "stick": cd["stick_events"], "slip": cd["slip_events"], "pen": cd["max_penetration_m"]}, default=float))
