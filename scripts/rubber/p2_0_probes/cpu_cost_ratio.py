"""Throwaway: CPU (Warp CPU device, serial kernels) cost per particle-substep of the frozen v7_2 step vs the lean
rubber probe step, as a work-ratio proxy (not a GPU timing)."""
import json, sys, time
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R/"tests")); sys.path.insert(0, str(R/"src")); sys.path.insert(0, str(R/"scripts/rubber/p2_0_probes"))
import warp as wp
wp.config.quiet = True
from batched_fixtures import trim_material, start_pose
from probe_tlmpm_dynamic import Strip
m = trim_material(512)
m.reset(start_pose(512), seed=42)
bed = m.bed
for _ in range(5): bed.step(advance=False)
t0 = time.perf_counter(); n = 100
for _ in range(n): bed.step(advance=False)
v72 = (time.perf_counter()-t0)/n
s = Strip(L=.2, W=.02, t=.004, n_t=2, mu=5e5, beta=20., rho=1150., device="cpu")
for _ in range(5): s.step()
t0 = time.perf_counter()
for _ in range(n): s.step()
rub = (time.perf_counter()-t0)/n
out = dict(v7_2=dict(particles=int(bed.n), nodes=int(bed.node_count), slots=int(bed.slots_per_node), s_per_substep=v72,
                     ns_per_particle_substep=1e9*v72/bed.n),
           rubber_lean=dict(particles=int(s.n), nodes=int(s.nn), slots=1, s_per_substep=rub, ns_per_particle_substep=1e9*rub/s.n))
out["ratio_per_particle_v72_over_rubber"] = out["v7_2"]["ns_per_particle_substep"]/out["rubber_lean"]["ns_per_particle_substep"]
print(json.dumps(out, indent=1)); json.dump(out, open("cpu_cost.json", "w"), indent=1)
