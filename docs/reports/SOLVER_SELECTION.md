# WP-02 — Initial solver comparison

Continuation: [independent TLMPM](PERSISTENT_CUTTING.md) now supplies online
persistent separation alongside official Newton's deformation baseline.
[Frozen GPU validation](TLMPM_VALIDATION.md) records actual cutting/withdrawal/
recontact and refinement limits. [CRESSim corrections](CRESSIM_FOLLOWUP.md)
restore stable controls without retained cut topology; both patches are applied
by the default build. The original investigation below is historical evidence.

State: **BLOCKED at persistent-separation gate; no formal solver selected**.
Newton is retained as the working deformation testbed. Neither candidate has
passed the required cut, withdrawal, recontact and independent-chip acceptance.

Both use the same specimen generator, seed42, SI units and reference CSV path.
They differ in constitutive law, collision geometry and time integration, so
the current experiments cannot support a controlled speed/quality ranking.

| Candidate | Actual execution | Hard-gate status |
|---|---|---|
| Newton1.6.1 implicit MPM | Official granular headless smoke; supported settling, indentation/unload, scrape; three grids; fresh reconstruction reset; surface export | Contact/deformation works. No implemented persistent discontinuity, failure labels or independent chip mass. G2 NOT_PROVEN. |
| CRESSim upstream `09aa5009b8580351f516b6df7660e87821fc5eb6` | CUDA12.8 sm_120 engine/C API and headless adapter build; original and corrected runs retained | Initial adapter bounds error and upstream scale/math issues invalidate early apparent successes. |
| CRESSim + tracked local four-part patch | GPU settling and scrape, 19,340 particles, dt=1/6000 s, 100 fine tool updates/frame, 120frames | Settling FAIL: 3.307 m/s peak and tray tunneling. Scrape FAIL: 5.685 m/s peak. Persistent history absent. |

The unmodified GUI `example_cutting` was read as the API reference; its executable
movie was NOT_RUN. The process-isolated headless QuadSlicer harness actually ran
the documented GPU solver and exports particles. It is not represented as an
unchanged official demo or as a successful cutting benchmark.

Reproduce current candidate (expected numerical FAIL / exit1 on this host):

```bash
bash scripts/build_cressim.sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MPLBACKEND=Agg .venv/bin/python -m robotic_cutting_sim.cli physics-probe --solver cressim --case settle --substeps 100 --output-dir artifacts/cressim_settle_fresh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MPLBACKEND=Agg .venv/bin/python -m robotic_cutting_sim.cli physics-probe --solver cressim --case through_bump_cut --substeps 100 --output-dir artifacts/cressim_scrape_fresh
```

The 2 s settling run took 5.021 s inside the native process; scraping took 6.923 s.
These failed runs are not usable performance comparisons. Sampled total physical
GPU0 memory peaked at 967 MiB including display/baseline. Particle mass error was
3.54e-8 relative, but mass remaining numerically present does not excuse invalid
dynamics. Solver stdout and all state frames are retained.

Source audit of the pinned CRESSim checkout found:

| Issue | Evidence / current handling |
|---|---|
| Wrong adapter grid bounds | C++ `Bounds3` takes min/max; C API struct uses center/extents. Fixed in adapter; original `cressim_scrape_001` retained as INVALID_ADAPTER. |
| Absolute nodal mass cutoff1e-4 kg | At rho1500/dx4mm, nominal nodal mass is9.6e-5kg. Patched to1e-12kg with zero rejected-node velocity in `src/solver/mls_mpm_algorithm.h`. |
| Co-rotational switch fallthrough | Missing break in `src/solver/constitutive_models.h`; patch restores selected law. Earlier case actually fell through to another elastic law. |
| APIC derivative missing inverse spacing | Shared MLS code used4/dx on an outer product containing physical distance; patch uses4/dx². Independent affine-reproduction calculation confirms dimensional error. |
| Local slicer normal used with world velocity | Patch rotates transfer normal in `src/solver/mpm_rigid_contact.h`. |
| Further matrix outer-product/SVD transpose errors | Actual-header host diagnostic fails: outer-product max error6 and SVD reconstruction error.01 for a.01 shear. These remain unpatched; causal share of instability is not isolated. |

Patch is committed at `scripts/patches/cressim_si_mls.patch`; native manifests
record upstream SHA and diff SHA256. This is an experimental downstream candidate,
not stock CRESSim. It still fails numerical validation and is not selected.
The small `scripts/cressim_math_audit.cpp` is a reproducible CPU failure diagnostic:

```bash
g++ -std=c++17 -O2 scripts/cressim_math_audit.cpp -I.deps/CRESSim-MPM/include/math -I.deps/CRESSim-MPM/include/core -I.deps/CRESSim-MPM/include/utils -I.deps/CRESSim-MPM/src/solver -I.deps/CRESSim-MPM/include/solver -o .deps/cressim-build/math_audit
.deps/cressim-build/math_audit
```

CRESSim's source rejects P2G/G2P transfers crossing the **current** finite slicer.
`mpm_rigid_contact.h::checkOnDifferentSides` receives current shape poses but no
particle cut history; withdrawal restores ordinary shared-grid support. This
source observation means persistence is unsupported, not that a withdrawal
experiment proved re-welding. Reflected G2P velocity also ignores blade-relative
normal velocity. Only elastic material laws are exposed. Kinematic shapes do
not accumulate native reaction force, hence NaN force CSV entries are explicit
unavailable data. C API has particle position/mass/velocity export but no failure
labels; a global factory and upstream non-thread-safety require process isolation.

Newton exposes yield/plastic history, collider setup and impulses, but no native
cut history or label-compatible grid degrees of freedom. A persistent extension
must alter `ImplicitMPMScratchpad._create_velocity_function_space`, particle
quadrature/binning, mass/momentum/strain assembly and G2P advection in
`newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py` and its kernels. Changing
yield alone does not create independent sides; `separate_worlds` isolates entire
environments and is not a chip-contact implementation.

T1–T8 coverage: Newton T1/T2/T3 trajectory execution and T7 fresh reconstruction
were exercised; T6 has three exploratory resolutions, T8 surface export passes
only for deformation. CRESSim has settling/T3 execution failures and export.
Formal T3 chip proof, T4 withdrawal persistence, T5 recontact/two-pass, converged
T6, and cutting-aware T8 are all NOT_PROVEN/NOT_RUN. No weighted ranking is issued.

Next experiment: a narrow, continuously sampled material split with persistent
side labels and recontact, backed by low-level transfer tests. Preserve the working
Newton deformation/camera pipeline while investigating this separately. Following
the user's scope clarification, static robot-scene and environment-API work proceeds
independently; it does not imply research-ready cutting around a failed solver.
