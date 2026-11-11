"""DIAGNOSTIC (task X2, ADR 0023 §22.10): spatial / temporal pattern of the friction-normal flutter of
``grid_node_penalty_coulomb_v2`` on the INC block (16 x 8 x 4 mm, table z = 0, tilted gravity ramped per substep over
20 ms). Runs T seconds of control steps (1 ms), then NS substeps recording the bottom-face normal impulses and realised
velocities ``old``; prints the mean bottom-face contact fraction, the dominant frequencies of the centre node's v_z, the
mean x / y wavenumber spectra of v_z and the rms of v_z and of v_x about its face mean.

    CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/rubber/p2_4/diag_x2_pattern.py mu=0.8 nt=3 cfl=0.5 NS=260 T=0.1
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_4"))
import warp as wp
wp.config.log_level = wp.LOG_WARNING
import run_contact_v1 as R
from robotic_cutting_sim.rubber_contact import ContactSpec
a = dict(x.split("=",1) for x in sys.argv[1:])
mu=float(a.get("mu",0.8)); ratio=float(a.get("ratio",1.2)); G=9.81
theta=math.atan(ratio*mu) if "theta" not in a else float(a["theta"])
gvec=np.array([G*math.sin(theta),0,-G*math.cos(theta)])
mat=R._material(a.get("material","elastic"), friction_table=mu)
spec=R._spec(length=0.016,width=0.008,thickness=0.004,n_t=int(a.get("nt",3)),case="free")
live=R._live(spec,mat,1e-3,(0,0,0),ContactSpec(planes=((0,0,1,0),),law="grid_node_penalty_coulomb_v2"),cfl=float(a.get("cfl",0.5)))
s=live.solver
st=R.Stepper(live, lambda t:(gvec*R._ramp(t,0.02),None))
X=s.node_X.numpy(); gm=s.gm.numpy()
for k in range(int(float(a.get("T",0.08))/1e-3)): live.advance()
bottom=np.flatnonzero((np.abs(X[:,2])<1e-12)&(gm>0))
xs=np.unique(np.round(X[bottom,0],9)); ys=np.unique(np.round(X[bottom,1],9))
ix=np.searchsorted(xs,np.round(X[bottom,0],9)); iy=np.searchsorted(ys,np.round(X[bottom,1],9))
masks=[]; jns=[]; vz=[]; vx=[]
NS=int(a.get("NS",260))
for k in range(NS):
    g,_=st.loads(st.t+0.5*s.dt); s.gravity=wp.vec3(*map(float,g)); s.step(); st.t+=s.dt
    jn=s.ct_jn.numpy()[bottom,2]; m=np.zeros((len(xs),len(ys))); m[ix,iy]=jn; jns.append(m)
    o=s.old.numpy()[bottom]; q=np.zeros((len(xs),len(ys))); q[ix,iy]=o[:,2]; vz.append(q)
    q=np.zeros((len(xs),len(ys))); q[ix,iy]=o[:,0]; vx.append(q)
jns=np.array(jns); vz=np.array(vz); vx=np.array(vx)
print("grid", jns.shape, "mean contact frac", (jns>0).mean())
print("time-mean contact frac per node (x rows):")
np.set_printoptions(linewidth=200, precision=2, suppress=True)
print("label",a)
# temporal spectrum of vz at a centre node
c=vz[:, len(xs)//2, len(ys)//2]; f=np.abs(np.fft.rfft(c-c.mean())); fr=np.fft.rfftfreq(len(c), s.dt)
print("dt",s.dt,"vz centre dominant freq", fr[np.argsort(f)[-3:]], "period substeps", 1/fr[np.argsort(f)[-3:]]/s.dt)
# spatial spectrum of vz (x-direction) averaged
F=np.abs(np.fft.rfft(vz-vz.mean(axis=(1,2),keepdims=True),axis=1)).mean(axis=(0,2)); print("x-wavenumber spectrum of vz (k index 0..):",F/F.max())
F=np.abs(np.fft.rfft(vz-vz.mean(axis=(1,2),keepdims=True),axis=2)).mean(axis=(0,1)); print("y-wavenumber spectrum of vz:",F/F.max())
print("vz rms", vz.std(), "vx rms about mean", (vx-vx.mean(axis=(1,2),keepdims=True)).std(), "mean vx", vx.mean())
