# WP-12 Phase 0.4 — Cut-surface technology review

Date: 2026-10-30. Branch `claude/wp12-physical-cut-surfaces` (base `fbe99bd`, HEAD `ce547f9` at review time).
Scope: [WP-12 specification](../WP12_SPECIFICATION.md) §0.4 (research review) and the design inputs for Phase 1
(§1.3–1.4) and Phase 3 (§3.1–3.7). Nothing in this report changes code, physics, thresholds or
evidence. It is a design and measurement-method document, and every threshold in §7 is a **candidate**.
Candidates must be committed in a protocol file before the first Phase 1/3 measurement that uses them.

Notation: *s* is the particle spacing (formal tasks use 1 mm), Δx is the TLMPM reference-grid spacing
(4 mm, `material_config.py:7`), K ≤ 15 is the number of packed cut digits, and N is the particle count
(about 10⁴–10⁵: 97 044 in the WP-11 TRIM profiling run, `docs/reports/WP11_PHASE_H_PROFILING.md`).

---

## 1. Sources and verification status

| Source | What was read | Status |
|---|---|---|
| Yang, Chen, Huang, Masterjohn, Yang, Jiang, *BladeMaster: Real-Time Robotic Cutting Simulation with Online-Generated Persistent Discontinuities*, arXiv:2609.27342v2, 28 Sep 2026. https://arxiv.org/abs/2609.27342 | Full PDF (7 pages): §III-A to §III-G, Eq. (1)–(18), §IV, appendix | **Verified from the full text** |
| BladeMaster project page, https://jango6324.github.io/blademaster/ | Whole page | Verified. It has **no** reconstruction details, no grid resolution and no code link |
| Wu, Dick, Westermann, *Interactive High-Resolution Boundary Surfaces for Deformable Bodies with Changing Topology*, VRIPHYS 2011, pp. 29–38 ([TUM PDF](https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2011/Interactive_High-Resolution_Boundary_Surfaces/VRIPHYS2011HighResolutionBoundarySurfaces.pdf)). This is BladeMaster ref. [23], the linked-volume representation | Full PDF: §3.1, §3.2.1, §3.2.2, §4 | **Verified from the full text** |
| Ju, Losasso, Schaefer, Warren, *Dual Contouring of Hermite Data*, SIGGRAPH 2002 (ACM TOG 21(3) 339–346). https://people.eecs.berkeley.edu/~jrs/meshpapers/JuLosassoSchaeferWarren.pdf | Full PDF: §2.1–2.4, §3, §4.1–4.2 | **Verified from the full text** |
| Zeng, Courtecuisse, *Dynamic Cutting Simulation Using Elastic Snapping for Mesh Quality Optimization*, Computer Graphics Forum 44 (2025), https://doi.org/10.1111/cgf.70005 | Wiley and Eurographics DL returned **HTTP 403**. Only abstract-level search snippets were available | **Full text NOT verified**. Everything about this paper below is abstract-level only |
| NVIDIA Isaac Lab 3.0, *Implicit MPM*, https://docs.nvidia.com/isaaclab/3.0/concepts/implicit-mpm | Whole page | Verified |

Note: the specification cites the snapping paper as "Dynamic Cutting with Elastic Snapping". Its published
title, under the same DOI, is "Dynamic Cutting Simulation Using Elastic Snapping for Mesh Quality Optimization".

---

## 2. BladeMaster Section III-F, element by element

### 2.1 What III-F actually says (verbatim core)

> "For visualization, we reconstruct the visible boundary on a uniform reference grid using the linked-volume
> representation [23] and local dual contouring [24]. Intersections between primal-grid links and the original
> boundary or blade-generated cut surfaces are stored as Hermite data, each comprising a point and a normal.
> Sweeping the finite blade edge disconnects the intersected links. Within each incident cell, connected
> components of the remaining links define local material regions. We update only affected cells and duplicate
> surface representatives across distinct regions to form the two coincident faces of a cut. Surface motion is
> driven by TLMPM through one-way coupling. Each surface representative inherits the cut-side code of its local
> material region and is embedded using fixed trilinear weights in the corresponding compatible TLMPM motion
> field. It therefore follows the deformation of its side without introducing additional dynamics. Link
> connectivity determines the visual topology, while the side-aware embedding keeps the surface motion
> consistent with the mechanical discontinuity." (BladeMaster §III-F)

§III-A adds: "The same blade sweep disconnects intersected links in the reference grid and triggers local
surface reconstruction."

III-F is one paragraph. It does **not** give:
- the resolution of the "uniform reference grid", including whether it is the TLMPM grid or a finer one;
- the link-intersection test;
- where the Hermite normal comes from;
- what "compatible TLMPM motion field" means numerically.

No code is published (project page). The implementation details below therefore come from ref. [23] (Wu
et al. 2011, which describes the same pipeline in full) and ref. [24] (Ju et al. 2002). Where BladeMaster is
silent, that is stated.

Context from the rest of the paper, needed for the mapping:
- Particle state includes the reference position X_p (§III-A).
- The stencils N_ip = N_i(X_p) and ∇₀N_ip are evaluated at fixed reference positions, Eq. (1).
- Side assignment uses the blade-plane distance s_{p,k} = r_{p,k}·n_k, Eq. (3), inside the finite band of
  Eq. (4). Only material behind the edge is assigned (f_{p,k} ≤ 0), with digit 1 if s < 0 and 2 otherwise,
  Eq. (6).
- The ternary code is c_p = Σ d_{p,k} 3^k, Eq. (7), with the conflict rule ∃k: (d_k(a), d_k(b)) ∈ {(1,2),(2,1)}.
- Compatible logical grids couple by antisymmetric pairwise exchange, Eq. (8).
- The band half-width is h_k = 6Δx.
- Compact slot allocation is described in §III-G.
- Reconstruction is excluded from the reported timings (§IV-D).

### 2.2 The nine elements

**(1) Reference-grid linked-volume representation.**
- *Paper:* a uniform reference grid with links between neighbouring primal cells (III-F). Wu et al. §3.1 gives
  the full definition: "decompose the object into a set of hexahedral cells, using a uniform hexahedral grid.
  Face-adjacent cells are connected via links, with six links emanating from each cell. Cuts are modeled by
  marking links as disconnected when they are intersected by the virtual cutting blade". Links at the original
  object boundary are also disconnected, and cells outside the object are removed.
- *Data needed:* a cell index per material cell, 6 neighbour links, and one connected/disconnected bit per link.
- *Ours:* this already exists implicitly.
  - Every sampler places one particle at each retained cell centre of an integer lattice:
    `q = (ijk+.5)*spacing` (`benchmark.py:163-164`), with a full-cell mass `density*spacing**3`.
  - The lattice is returned as `lattice` by `task.sample_particles()` (`tasks.py:173-179`,
    `workpieces.py:71-82`) and is recorded as `reference_indices` (`live_env.py:499,512,522,549`).
  - The 6-neighbour links are already built by `cut_topology.reference_components_packed`
    (`cut_topology.py:239`) and `law8_validation.lattice_bonds` (`law8_validation.py:323`).
  - So **our particles are the linked-volume cells, and the lattice bonds are the links.**

**(2) Surface and blade intersection Hermite data.**
- *Paper:* one point and one normal per intersected link, for both the original boundary and the cut
  surfaces (III-F). Wu §3.2.1 records "the distances between the intersection point to the link's endpoints as
  well as the normal of the blade at the intersection point ... whenever a link is cut by the blade". This is
  ordinary DC Hermite data (Ju §2.2).
- *Data needed:* for each disconnected link, the crossing fraction t ∈ [0,1] and the unit normal; per cut, also
  the cut id.
- *Ours:* nothing is recorded today; see §3.3–§4 for how to obtain it.

**(3) Link disconnection.**
- *Paper:* "Sweeping the finite blade edge disconnects the intersected links" (III-F, III-A). This is a
  **geometric** test of the blade sweep against links. The paper does not state that it is derived from the side
  labels.
- *Ours (required change of rule):* a geometric sweep test would be a second source of cut truth, independent of
  TLMPM. Spec §3.2 Layer 3 forbids that. We define instead:
  **link (i, j) is disconnected iff `conflicting_packed(code_i, code_j)`** (`cut_topology.py:187`). The set of
  cuts that sever it is the bit set `(A1_i & A2_j) | (A2_i & A1_j)`. The blade sweep enters only as Hermite
  geometry and as an audit (§6.4), never as topology.

**(4) Local connected material regions.**
- *Paper:* "Within each incident cell, connected components of the remaining links define local material
  regions" (III-F). Wu §3.2.1: each cut dual cell gets "as many [duplicated representative vertices] as there are
  disconnected material parts in this cell". Patterns are precomputed in a 2¹² LUT, one bit per each of the 12
  links of a dual cell.
- *Ours:*
  - The DC dual cell is the 2×2×2 block of lattice points (particle centres). Its 12 edges are lattice links.
  - The local regions are the connected components of the present corners under the uncut edges of that cell.
  - Computation is a 12-bit edge mask plus an 8-bit presence mask, which is LUT-able or a tiny union-find.

**(5) Local dual contouring.**
- *Paper:* DC ref. [24], updated only in affected cells (III-F).
- Ju §2.2: one vertex per cell, at the minimiser of the QEF E[x] = Σ(nᵢ·(x−pᵢ))², Eq. (1). Then "for each edge
  that exhibits a sign change, generate a quad connecting the minimizing vertices of the four cubes containing
  the edge".
- Ju §2.3 on solving the QEF:
  - The normal-equation form, Eq. (2), is numerically unstable in float. Use the QR form, Eq. (3), or doubles.
  - Singular values below 0.1 are truncated. The solution is pulled toward the mass point (the centroid of the
    pᵢ).
- Ju §2.4: multi-material contouring. "Index changes" play the role of sign changes, and the vertex lands at the
  junction when three or more materials meet.
- The octree (Ju §3) is a memory optimisation and is **not required**. DC is first defined on uniform grids
  (§2.2).
- *Ours:*
  - We have no scalar sign field. "Sign change" becomes "disconnected link": a boundary link (present → absent
    lattice site) or a cut link (conflicting codes).
  - Our uniform lattice is small (≤ ~10⁵ cells), so no octree is needed.

**(6) Duplicated cut-face representatives.**
- *Paper:* "duplicate surface representatives across distinct regions to form the two coincident faces of a
  cut" (III-F). In Wu §3.2.1 each duplicate "is bound to the nearest vertex of the respective part in the dual
  cell".
- *Ours:* one representative per (dual cell, local region). A cut link (i, j) is an edge of four dual cells and
  emits **two** quads:
  - the i-side quad uses, in each of the four cells, the representative of the region containing i;
  - the j-side quad uses the representatives of the regions containing j.
- A boundary link emits one quad on the material side.
- At an open crack front, i and j are still in one region of a cell through uncut links (via digit-0
  particles). Both quads then share that representative, so the slit closes at its tip.

**(7) Cut-side-aware surface embedding.**
- *Paper:* "Each surface representative inherits the cut-side code of its local material region and is embedded
  using fixed trilinear weights in the corresponding compatible TLMPM motion field" (III-F). Wu §3.2.2: each
  surface vertex is bound to exactly one primal cell and deformed "via tri-linear interpolation of the primal cell
  vertices".
- *Ours:* §6.5 recommends a particle-based embedding restricted to the region's own particles. This is the
  "compatible motion field" sampled where it actually lives (at 1 mm), not on the 4 mm logical-field nodes.

**(8) Motion-field interpolation.**
- *Paper:* fixed trilinear weights, one-way coupling, "without introducing additional dynamics" (III-F).
- *Ours:* the weights are fixed at the representative's reference position. The current position is recomputed
  from the Layer-1 state whenever a surface is needed. There is no integration and no feedback to physics.

**(9) Dynamic surface update.**
- *Paper:* "We update only affected cells" (III-F); "triggers local surface reconstruction" (III-A).
- *Ours:* topology changes only when codes change, and codes change only when a digit goes 0 → 1/2 (monotone).
  A dual cell is dirty iff one of its edges became disconnected since the last update (§6.7).

### 2.3 Other differences between BladeMaster and our `tlmpm_v5` that matter for surfaces

| Aspect | BladeMaster (paper) | Ours (`tlmpm_v5`, law 8) | Consequence for surfaces |
|---|---|---|---|
| Side test | Blade-plane distance s = r·n_k, Eq. (3)/(6) | Swept-surface distance `_swept_side` (`tlmpm_v5.py:329-344`) with normal ∝ (segment motion × edge axis), falling back to local z | The Hermite normal must be the **swept** normal the solver actually used, not the blade-plane normal |
| Band | \|s\| ≤ h = 6Δx, edge, rear and lead extents, Eq. (4); assign only f ≤ 0 | Same band 6Δx = 24 mm, rear 2Δx, persist 2Δx (`tlmpm_v5.py:711-715`, `_assign_band` 616-641). Law 8 additionally pre-assigns ahead of the edge (0 < f ≤ 2Δx, Δx < \|s\| ≤ 6Δx) on plowing segments (`_assign_lead` 644-665, `_segment_plow` 372-395) | Lead-assigned particles have \|s\| > 4 mm. Sign changes, and therefore Hermite crossings, come from band assignment near s = 0 |
| Encoding | Ternary Eq. (7) | Packed 2-bit digits with the same semantics (`tlmpm_v5.py:10-15`, `cut_topology.py:128-191`) | None |
| Tool pose in substeps | Fixed between rigid-solver updates (§III-G) | Interpolated per substep: slerp and linear (`coupled_robot.py:105-113`, `live_material.py:263-267`); 200 substeps per 20 ms control step (`coupled_robot.py:22-23`) | Per-substep blade poses are deterministic given the recorded measured poses |
| Reconstruction | Linked volume + local DC | Newton `ParticleSurface` per mechanical component (`cutting_surface.py:83-166`) | This is the subject of WP-12 Phase 3 |

---

## 3. Facts verified in our code

### 3.1 Per-particle reference positions — yes

- `TLMPM.__init__` stores `self.reference = wp.array(points)` (`tlmpm_v5.py:769`). Today it is used only for
  the self-contact reference exclusion (3Δx, `_particle_contacts`, `tlmpm_v5.py:472-497`, launched at 854-856).
- The trilinear reference stencil is precomputed once from the reference points (`tlmpm_v5.py:734-748`): 8 nodes,
  weights and gradients per particle, i.e. TLMPM Eq. (1).
- The host copy `material.points` (local) and `world_points` (`live_material.py:101-104`) are recorded as
  `reference_positions` (`live_env.py:484,548`).

### 3.2 A regular reference lattice usable as the linked-volume grid — yes

- Particles are cell centres of an integer lattice (`benchmark.py:156-176`). Each represents a full cubic cell of
  volume s³ (`SURFACE_GEOMETRY_VALIDATION.md`, "Quadrature error ...").
- The lattice indices are unique and recorded.
- **This is exactly Wu's linked volume (§3.1)**: cells = particles, links = 6-neighbour bonds, dual grid = lattice
  of particle centres.

Two lattice-dependent facts:

1. For a box whose dimensions are multiples of s, the outermost cell faces coincide with the analytic faces.
   (Centres at (ijk+.5)s − L/2 with q < L/2 put the last centre at L/2 − s/2.)
2. For curved height fields (D0 dome, capsule footprint), the full-cell volume is a staircase quadrature. At 1 mm
   it differs from the analytic D0 volume by −0.0175%; at 2 mm by +6.2% (`SURFACE_GEOMETRY_VALIDATION.md`).

### 3.3 Blade information available when a digit is assigned

In `_assign_band` (`tlmpm_v5.py:616-641`) and `_assign_lead` (644-665) the kernel holds:
- the particle's **current** position x_p (after G2P of this substep: assignment is the last physics launch
  before `_advance`, `tlmpm_v5.py:874-890`);
- v_p, the edge pose `poses[i]` and twist `twists[i]` at this substep's index;
- the cut digit k, the edge-local coordinates (f, ℓ, z), the swept signed distance `side`, and implicitly the
  swept normal (`_swept_side`);
- the segment index and its `seg_ok` / `seg_plow` flags.

Nothing of this is stored except the resulting digit.

Two consequences:
- The cut surface the solver actually decided is the zero set of the swept signed distance at the assignment
  instant, evaluated in **current** space.
- Particle positions are recorded only once per control step (`history_state`, `live_material.py:328-334`:
  positions, velocities, codes; **F is not recorded**). Positions at the assignment substep are lost.

### 3.4 Lattice links refine mechanical connectivity (never the reverse)

Claim: for two lattice-adjacent particles with compatible codes, `mechanical_components_packed`
(`cut_topology.py:194-234`) puts them in the same mechanical component.

Proof:
- Particles 1 mm apart in a 4 mm grid share at least four stencil nodes with positive weight. Our cell-centre
  fractions are 0.125/0.375/0.625/0.875, never 0, because the grid origin is `floor(min/Δx)Δx − 2Δx`
  (`tlmpm_v5.py:734`).
- Two compatible codes at a shared node are unioned (`cut_topology.py:216-226`).
- This assumes no slot overflow, which raises in any case (`tlmpm_v5.py:898-906`).

Hence every mechanical component is a union of lattice-connected components. A lattice-based surface can show
**more** separation than the mechanics, through code-0 or diagonal stencil bridges (`law8_validation.bridge_report`,
`law8_validation.py:351`). It can never show less. This asymmetry drives the audit in §6.4.

### 3.5 The existing pipeline is Approach A

`CuttingSurface` (`cutting_surface.py:83-166`) extracts one Newton `ParticleSurface` isosurface per current
mechanical component. There is no intercomponent weld (`combine_component_meshes`, 53-68). Profiles:
- `legacy` (H0): 12.06% volume inflation;
- `current_volume_v1` (H2): half-spacing voxels, kernel radius 3s, threshold 0.5, radius (s/2)·∛J;
  0.42–0.64% volume error and p95 0.16–0.22 mm on affine 1 mm boxes (`SURFACE_GEOMETRY_VALIDATION.md`);
- `unmeshable="skip_presentation_v1"`: omits fragments that ParticleSurface cannot mesh (WP-11 live_003).

The Isaac Lab Implicit-MPM documentation uses the same Newton `ParticleSurface` as an optional visualization pass
that "does not change particle motion, collisions, or material behavior". It recommends `kernel_radius` near three
times the particle spacing, matching our H1/H2, and has no cutting or fracture support. So Approach A is the
standard toolchain route, and nothing cut-aware is available upstream.

---

## 4. How to obtain Hermite data without changing physics

| Option | Mechanism | Exactness of t and n | Cost | Verdict |
|---|---|---|---|---|
| **R1 — in-loop read-only recorder (`cut_events_v1`)** | In the material wrapper's substep loop (inside the captured CUDA graph, `live_material.py:224-226`; CPU loop 275-276): `copy(code_prev, code)`; `bed.step()`; `record(...)`. `tlmpm_v5.py` is untouched. Assignment is the last code-writing launch of `step()`, so after `step()` `x` and `code` are exactly the state the assignment saw; the recorder reads the pose and twist at `index-1` | **Exact w.r.t. the solver's own decision.** Both endpoint sides are evaluated at the same instant, against the same swept plane | One N-thread kernel per substep with early exit, plus an N-int copy. Memory: an append log (§6.2), < 1 MB per cut at our scale. Estimated < 2% of the 1.30–1.34 ms/substep solver cost (ADR 0014 Phase M) — **estimate, to be measured** | **Recommended** |
| R1′ — recorder inside a new solver subclass | Duplicate `step()` in `tlmpm_v5_events.py` with extra launches | Same as R1 | Same as R1 | Acceptable, but duplicates the frozen launch order. R1 is preferred because it touches no solver source |
| R2 — offline from recorded control-step history | Find the control step in which a link became conflicting (code diff); interpolate particle positions linearly over the 200 substeps; reproduce substep poses with `interpolate_measured_poses` | Inexact: particles move ~0.2–1 mm per 20 ms control step at 10–50 mm/s cutting speeds, comparable to s. Sign may not reproduce the solver's digit | Needs per-step positions (stored) and recomputation per conflicting link | Diagnostic only |
| R3 — deterministic replay with recorder | Re-run the episode on CPU (bitwise deterministic, ADR 0012 review note) with R1 enabled | Exact for CPU/kinematic runs | Full re-simulation | Valid for CPU and kinematic baselines. **Not valid for GPU or Native episodes**: GPU atomics are nondeterministic (ADR 0012: equal final codes, \|dx\| ≤ 0.06 mm), so a replay is a different run |
| R4 — label-only (no blade data) | t = ½ (link midpoint). Normal from the local pattern of conflicting links (mean signed link direction, as the draft `surface_metrics_v1.py` does), mapped by F where available | Position error along the link ≤ ½s·\|n_a\| per point; normal is an estimate | None | **Fallback** for episodes recorded without R1, including all frozen WP-11 and WP-12 baselines |

**R1 recorder specification (proposed, `cut_events_v1`, opt-in, default off):**

1. Before the step: `code_prev ← code`. After `step()`:
   - for each p whose digit k went 0 → d ∈ {1, 2}, recompute the swept normal **n** exactly as `_swept_side` does,
     including its blade-z fallback, at pose and twist `[index-1]`;
   - compute s_p = (x_p − o)·n, with o the edge origin.
2. For each lattice neighbour q of p (precomputed N×6 table, −1 = absent) whose digit k equals 3 − d:
   - compute s_q = (x_q − o)·n at the same instant;
   - if q was also newly assigned this substep, write the link only from min(p, q);
   - if the signs agree with the digits, t = s_p/(s_p − s_q); otherwise t = NaN with a flag (the host uses t = ½
     and counts these links).
3. Map to the reference configuration:
   - Hermite point P = X_p + t(X_q − X_p). The crossing fraction along a link is invariant under any locally
     affine map.
   - Normal N ∝ F̄ᵀn with F̄ = ½(F_p + F_q) (Nanson: n da = J F⁻ᵀN dA).
4. Append (p, q, k, t, N, n, o, global substep, flags) to a log with an atomic counter. Overflow is counted and
   raised, like slot overflow.
5. Also log per-substep cut activation: `stroke`, `stroke_engaged`, `stroke_any_motion`, and the counts of
   `segment_ok` and `segment_plow`. All are read-only.
6. Physics neutrality is tested, not assumed:
   - CPU: `snapshot()` must be bitwise equal with the recorder on and off;
   - GPU: final codes equal and positions within the documented nondeterminism band;
   - default path: no new launches.

---

## 5. Comparison of approaches (our scale: N ≈ 10⁴–10⁵, s = 1 mm, Δx = 4 mm, K ≤ 15)

Definitions:
- **A** = improve ParticleSurface (kernels, current-volume radii, resolution, mesh processing).
- **B** = cut-aware reference-lattice reconstruction (linked volume + local DC + side-aware embedding).
- **C** = ParticleSurface outer surface + cut-aware internal faces.
- **D** = alternative mesh optimisation (elastic snapping, other contouring or refinement).

| Criterion | A | B | C | D |
|---|---|---|---|---|
| Expected geometric fidelity | Aligned boxes: p95 0.16–0.23 s, volume error 0.4–0.8% (measured, H1/H2). Kernel convolution rounds every sharp edge, including cut/boundary creases ("H1 rounds sharp corners", validation doc). Oblique cuts give a smoothed lattice staircase (amplitude ≤ ½s before smoothing; not measured). Thin (1–2 layer) fragments can fall below the 0.5 density threshold and vanish (WP-11 `skip_presentation_v1`) | Planar cut with R1 Hermite: QEF reproduces planes exactly up to float, expected cut-face p95 ≲ 0.05 s; creases kept sharp (Ju §2.2–2.4). With R4 fallback: ≤ ½s per Hermite point, expected p95 ≈ 0.2–0.35 s with visible steps (normals up to ~25°). Boxes exact. Curved outer boundary: staircase unless boundary Hermite uses the analytic reference shape (§6.3) | Cut faces as B; outer as A. Fidelity is limited at the seam, where A's rounded, offset isosurface meets B's sharp faces | Mesh quality, not position accuracy, is the target (abstract-level). Fidelity depends on what the vertices are snapped to |
| Topological consistency | Global topology = mechanical components (consistent). **Partial cuts and open crack fronts are invisible** within one component: the kernel radius 3s exceeds a blade-thickness gap. Crack faces appear only after the gap opens past ~2–3 s | Derived from the same codes: a face exists iff a link conflicts. Crack fronts, intersections and recontact are represented. Can show more separation than the mechanics through code-0 or diagonal stencil bridges (§3.4), so this must be audited and flagged | Mixed: two topology sources meet at the seam, the most likely place for holes or overlaps | Topology from separate geometric operations; risk of a second truth |
| Physical consistency | No physics change; reconstruction follows current particles only | No physics change. One-way particle embedding. Faces only where the solver wrote conflicting digits | Same as its parts. Trimming A near cut faces risks acting like a mask | Snapping moves vertices by an auxiliary elastic problem. In TLMPM that is not material motion, so it must stay a reference-space geometry step and never touch particles (spec D) |
| Implementation complexity | Low: parameters and profiles | Medium: recorder kernel, NumPy reference DC (like `cut_topology`'s NumPy references), optional Warp port, embedding, bridge normals | High: B plus seam stitching and trimming | High: an auxiliary FEM solve and a tetrahedral or surface mesh |
| GPU memory | Voxel grid per component: at 0.5 mm voxels, a 160×30×20 mm bar is 7.7×10⁵ voxels, ~5–50 MB (estimate) | Neighbour table N×6 int32 (2.3 MB at 97k); Hermite log ~3·A_cut/s² entries per cut (< 1 MB); feature-cell arrays ~10 MB (estimate) | A + B | Not determinable (full text unavailable) |
| Runtime | Measured: surface + USD update 49–79 ms per control step in the WP-11 Native TRIM (Phase H); 3.4–5.5 ms for small fixtures | Estimates, to be measured: full rebuild CPU NumPy 50–200 ms, Warp ~5–10 ms; incremental per-frame embedding 1–5 ms; recorder < 2% of solver | > A | Iterative elastic solve per update; "real-time" claimed (abstract-level, unverified) |
| Dynamic multi-cut support | Yes (per component); cuts inside a component are hidden | Native: per-link cut set from the packed bits; intersections by multi-normal QEF; K ≤ 15 | As B | Progressive cuts claimed (abstract-level) |
| Integration risk | Low | Medium: new opt-in modules; recorder hook in the material wrapper; bridge needs face-varying normals for creases | High | High |
| Validation difficulty | Medium (existing harness, `surface_distance`, affine controls) | Medium-high. Strong CPU invariants (faces ⇔ conflicting links, no weld, per-shell mass) and exact G2 oracle unit tests | Highest (seam correctness) | High |

**Conclusion:**
- B is the primary approach. It is the only one whose topology is derived from the solver's side codes, so it can
  represent partial cuts, crack fronts and thin chips without hiding or inventing material.
- A remains the frozen `particle_surface_v1` baseline and the fallback for any workpiece without a reference
  lattice.
- C is only needed if B's outer boundary is unacceptable. Analytic boundary Hermite (§6.3) removes the main reason
  for C.
- D is not recommended for v1.

---

## 6. Recommended architecture — `cut_aware_surface_v1`

### 6.1 Layer 1 — physical material state (unchanged)

`tlmpm_v5` with law 8 is the only source of truth: x, v, F, Fpi, packed codes, the stencil, slots and contact.
`cut_aware_surface_v1` reads `reference`, `x`, `F`, `code`, the lattice and the masses, and writes nothing back.

### 6.2 Layer 2 — cut-event tracking (`cut_events_v1`, R1 of §4)

- Per substep: the activation summary (stroke, engagement, motion, plow counts).
- Per newly conflicting link: (p, q, k, t, N_ref, n_cur, edge origin, substep, flags).
- Consistency audits:
  - every logged link must conflict in cut k in the final codes (monotone codes make this an invariant);
  - every Hermite point of cut k must lie inside the actual finite swept region of cut k, using
    `law8_validation.swept_region` / `label_region_report` (`law8_validation.py:281-320`) on the recorded edge
    poses, with a tolerance declared in the protocol.
- The swept region is an **audit**, never a generator.

### 6.3 Layer 3 — reference-lattice cut topology

**Reference grid: the particle lattice (1 mm).** The alternatives are rejected:
- **4 mm MPM grid:** four particles per cell edge. A 1 mm chip or a cut 1–3 mm from a boundary is unrepresentable.
  Its nodes carry several logical fields, so a "node code" is undefined.
- **Finer surface grid:** sub-particle sites have no code. Inventing codes by interpolation makes the topology
  independent of the solver. Resolution beyond the particle quadrature has no material meaning.

**Links:**
- Cut iff the endpoint codes conflict, with the cut set from the packed bits.
- Boundary iff a neighbour site is absent.
- Uncut otherwise, including every 0–1 and 0–2 pair. Digit-0 material is compatible with both sides, so crack
  fronts, residual bridges and partial cuts follow automatically.

**Boundary Hermite (declare one profile before measuring):**
- `cell_face_v1`: t = ½, axis normal. This is consistent with the full-cell mass and exact for boxes, but
  terraced on curved shapes.
- `analytic_reference_v1`: the crossing of the link with the workpiece's analytic **initial** shape (height field
  plus footprint, known at sampling), with the gradient normal. This is initial geometry, not target-cut
  information. Its represented volume differs from Σm/ρ by the quadrature error (−0.0175% for D0 at 1 mm); report
  it, never scale it away.

### 6.4 Layer 4 — local dual contouring

1. **Regions.** For each dual cell (2×2×2 lattice block with at least one present corner), compute the connected
   components of present corners under its uncut edges. Record per region its particle set, its code set and
   their mechanical-component labels.
2. **QEF per region.**
   - Inputs: the Hermite points of every disconnected edge incident to a region corner (cut and boundary). A
     link severed by several cuts uses, for each side, the crossing nearest that side's endpoint, and is counted
     as `multi_crossing_links`.
   - Solve in float64 or QR form (Ju §2.3 Eq. (2)/(3)), truncating singular values below 0.1 relative, toward
     the mass point.
   - Clamp to the dual cell and count clamps.
   - Intersecting cuts and cut/boundary junctions then give sharp edges and corners (Ju §2.4).
3. **Quads.**
   - Per disconnected link, one quad per material side, using that side's region representative in each of the
     four incident dual cells.
   - Orientation: the normal points from the side's endpoint toward the other endpoint.
   - **No vertex is ever shared between regions.** Vertices are welded only within a region, so conflicting
     sides are never welded.
4. **Special cases:**
   - **Partial cut / open crack front:** cut edges whose endpoints remain in one region of that cell. Both side
     quads use the same representative, giving a closed slit tip. No face is generated beyond the last
     conflicting link.
   - **Intersecting cuts:** up to four regions per cell; multi-normal QEF.
   - **Thin chips (one particle thick):** the chip's top and bottom faces fall in different dual cells (offset by
     half a cell), so each gets its own representative. A one-particle fragment becomes a small closed cell. This
     replaces the `skip_presentation_v1` omission, and no represented mass is dropped.
   - **Near-coincident faces:** two cuts severing the same link or adjacent links leave no particle between
     them. The zero-volume sliver is reported (`multi_crossing_links`, volume metric), not repaired.
   - **Original-boundary intersection:** the same QEF mixes boundary and cut normals into a crease. No stitching
     is needed.
   - **Ambiguous configurations:** face-diagonal-only connectivity inside a cell splits into separate regions.
     Count non-manifold and quasi-manifold violations (Ju §4.1–4.2). Never silently drop them.
5. **Topology audit (every extraction):**
   - each mesh shell maps to exactly one mechanical component; violation is an error, impossible per §3.4;
   - mechanical components with more than one shell are listed as `visually_separated_mechanically_bridged`,
     with the bridge report. This is the "Remaining Bridges" metric and is never hidden;
   - global piece counts always come from `mechanical_components_packed`, never from the mesh.

### 6.5 Layer 5 — cut-side-aware embedding

Each representative r has a fixed reference position X_r. Let R(r) be the region's particles, i.e. its present,
mutually linked corners: same side by construction.

```
x_r = Σ_{p∈R(r)} w_p [ x_p + F_p (X_r − X_p) ]  /  Σ w_p
```

- w_p are the trilinear weights of X_r w.r.t. the dual-cell corners, restricted to R(r) and renormalised.
- Why the F term: restricting trilinear weights to a subset destroys linear precision. Adding the first-order term
  restores exactness for any affine motion (compression, shear, rotation; consistent with the affine controls)
  with any subset.
- No integration, so there is no drift, and the result is a pure function of the current Layer-1 state.
- Recontact cannot weld because opposite sides never share particles.
- Crack-tip representatives (regions spanning digits 1, 0 and 2) are pulled by both sides, which is the physically
  connected tip.
- F is total deformation (validated: `affine_surface_controls.tlmpm_total_F_receipt`, `affine_surface_controls.py:30`).
  Guard against det F ≤ 0 by falling back to translation-only and counting it.
- **Recordings without F** (all current histories, `live_material.py:328`): fit a local affine map by least
  squares to R(r) ∪ (linked same-region neighbours within 2 links). Use translation-only when fewer than four
  non-coplanar particles are available. Record which estimator was used.

The paper's literal variant samples the logical-field nodal velocity at 4 mm (`new_v[node*S+slot]`) and
integrates. It is not recommended:
- it needs a kernel inside the substep loop;
- a representative's exact code may be absent at a node, and the code-0 field is "compatible" with both sides, so
  reading it would drag a face with the opposite side;
- it smooths 1 mm features at 4 mm and accumulates integration drift.

### 6.6 Layer 6 — USD and rendering

- Use the existing `DynamicSurfaceBridge.update` (`usd_surface_bridge.py:38-67`): strictly increasing
  timestamps, `ChangeBlock`, double-sided.
- Its contract clears normals. Sharp DC creases need **face-varying normals**, computed per region with a crease
  angle and never averaged across regions. This requires an additive, versioned bridge method. Keep the current
  method unchanged.
- Metadata per extraction: `surface_profile: "cut_aware_surface_v1"`, the codes hash, the Hermite-log hash and
  count, the boundary-Hermite profile, the embedding estimator counts, the audit results, and
  `renderer_partition_changes_physics: false`.

### 6.7 Incremental update

- New conflicting links per frame = the new log entries (or the code diff). Mark their four incident dual cells
  dirty, and recompute regions, QEF and quads only there. All other representatives keep their fixed X_r, so
  topology is temporally stable by construction.
- Re-embed all representatives every surface frame (O(#reps)).
- Compare a full rebuild against the incremental one every M frames in tests: they must match exactly.
- Keep `surface_update="on_capture"` (Phase H) semantics available.

### 6.8 Fallback and hybrid

- No lattice (future imported meshes) → `particle_surface_v1`.
- No R1 log → R4 midpoint Hermite. It is clearly labelled, never presented as recorder-quality, and has its own
  thresholds.
- `hybrid_surface_v1` (Approach C) is built only if Phase 1/3 data show B's outer boundary failing a frozen
  metric that A passes.

### 6.9 Provenance to record

Record:
- solver, law and parameters, `slots_per_node`;
- the lattice and reference-position SHA256, the stencil SHA256, the codes SHA256 per frame;
- the `cut_events_v1` log and its SHA256, with recorder on/off and the overflow count;
- the boundary-Hermite profile, QEF truncation and clamp counts, multi-crossing and sign-inconsistent link counts;
- embedding estimator counts and det F fallbacks;
- the mechanical audit (shells per component, bridged components);
- timings (extraction, events, topology, embedding, USD), mesh size and GPU memory.

### 6.10 Pitfalls (hard rules)

1. Never use the target or declared cut plane: not for Hermite points, normals, clamping, evaluation alignment,
   or "flattening".
2. Never generate a face on a link whose endpoint codes do not conflict. No sweep-only, contact-only or
   gap-only faces (S9/G8).
3. Never edit, move, delete or duplicate particles, and never feed the mesh back into physics.
4. Never derive global separation from the mesh.
5. Never weld vertices across regions.
6. Never drop a fragment silently.
7. Debug and oracle data (codes, Hermite log, regions) stay out of policy observations.

---

## 7. Quantitative evaluation method (Phase 1 and Phase 3)

### 7.1 Spaces and references (spec §1.4: separate material-reference space from current space)

| Space | Reference geometry | Measures | Never compared with |
|---|---|---|---|
| **Material reference (X)** | *Material cut locus*: per conflicting link, the R1 Hermite point P (or the midpoint with R4), and the local least-squares plane of locus points within a declared patch radius (removes lattice quantisation) | Reconstruction error e_recon of B's pre-embedding mesh; the topology invariants | Target plane |
| **Assignment-time current space** | Actual swept surface of the recorded edge trajectory at each assignment substep (R1) | Labelling/physics deviation e_label: locus vs swept surface. **Descriptive physics attribution, not a reconstruction gate** (B's Hermite comes from the same sweep, so this check is circular for B) | — |
| **Current (x)** | *Side locus*: q = x_p + F_p(P − X_p) per side (the draft `surface_metrics_v1` uses the midpoint form) and its local plane | e_embed+recon for B; total error for A | Target plane |
| Analytic (G1/G2 only) | Exact box SDF (`affine_box_sdf`) and an oracle static cut plane in a geometry unit test, with codes set by a geometric rule outside any physics run | Algorithmic correctness | — |

- ParticleSurface (A) has no reference-space mesh. Compare A and B **in current space against the same side locus**.
- Pull-back of A to reference space (nearest same-component particle, F⁻¹) is diagnostic only and must be declared
  approximate.
- Physical deformation after cutting (current locus vs assignment-time locus) is **reported as material behaviour,
  never as error**.

Metric definitions (spec §1.4):
- **Distance:** P50/P95/max, area-weighted over triangle centroids plus all vertices (as `surface_distance.directed`),
  with no trimming.
- **Signed error:** per side, toward the side's digit.
- **Normal error:** triangle normal vs local Hermite normal. Crease faces (a QEF region with more than one normal
  cluster) are reported separately because the normal is undefined there; they are still included in the distance
  statistics.
- **Roughness:** RMS residual to the local plane, for the mesh and for the locus. The difference is what
  reconstruction adds.
- **Mesh validity:** degenerate, non-manifold and boundary edges, orientation, shells, Euler characteristic.
- **Boundary continuity:** no boundary edges at cut/outer junctions.
- **Topological consistency:** the §6.4 audit.
- **Detached mass and remaining bridges:** from mechanical components and the bridge report.
- **Volume:** mesh vs Σm/ρ (reference) and Σm·J/ρ (current).
- **Temporal stability:** topology hash and vertex residual between frames.

### 7.2 Expected discretization error per method (planar cut, interior of the face)

| Method | Expected P95 | Expected max | Basis |
|---|---|---|---|
| B + R1 Hermite | ≲ 0.05 s | ≲ 0.25 s (clamped or truncated cells) | QEF exact for coplanar Hermite data (Ju §2.2). Residual from non-affine strain within one link and float32 positions |
| B + R4 midpoint, exact normal | ≈ 0.2–0.35 s | 0.5 s | Each point is offset ≤ ½s·\|n_a\|. The single-normal QEF minimiser is the mean offset, so it is bounded by ½s |
| B + R4 midpoint, label-estimated normal | ≈ 0.3–0.5 s | ≤ (√3/2)s (cell clamp) | Normal noise added |
| A (H2, current_volume_v1) | 0.16–0.23 s on aligned box faces (measured); cut faces unmeasured | Corner rounding ~ kernel support | Validation doc; convolution |
| Midpoint locus itself vs true crossing | uniform ±½s\|n_a\| → RMS ≈ 0.29 s\|n_a\| | 0.5 s | Lattice quantisation, so compare against local planes |

Measurement uncertainty:
- GPU run-to-run nondeterminism, |dx| ≤ 0.06 mm (ADR 0012), i.e. ±0.06 s on current-space metrics. Require ≥ 2
  repeats and report the spread.
- float32 positions are negligible (≈10⁻¹⁰ m at our scale).

### 7.3 Candidate thresholds (freeze in a protocol file before the first measurement)

| Test | Gate (candidate) |
|---|---|
| G1 uncut box (B, `cell_face_v1`) | Whole-surface P95 ≤ 0.02 s, max ≤ 0.05 s; \|ΔV\|/V ≤ 10⁻⁴; closed oriented manifold, 1 shell, χ = 2 (A's H1/H2 figures reported alongside, not gated) |
| G2 static oracle cut, held-out normals (e.g. 30°, 45°, (1,2,3)/√14), exact Hermite | Cut-face P95 ≤ 0.05 s, max ≤ 0.25 s; non-crease normal P95 ≤ 2°; 0 non-manifold edges. R4 variant reported against §7.2 (P95 ≤ 0.35 s, max ≤ 0.5 s) to quantify the recorder's value |
| G3 / S1 physical flat cut (R1) | Reference: P95 ≤ 0.1 s, max ≤ 0.5 s vs the Hermite local plane. Current: P95 ≤ 0.5 s, max ≤ 1.5 s vs the side locus; \|signed mean\| ≤ 0.1 s; normal P95 ≤ 10° |
| S2 deformed | As S1 in current space, max ≤ 2 s (heterogeneous strain); embedding fallback counts reported |
| S3 curved path | As S1 (reference); single-plane residual reported (must not be forced to 0) |
| S4 partial | 0 cut faces on non-conflicting links; slit extent equals the conflicting-link extent exactly; mechanical components = 1; tip representatives shared |
| S5 intersecting | Per-cut distances as S1; quasi-manifold violations reported (v1: report-only) |
| S6 four slices | ≥ 5 mechanical components, each with ≥ 1 shell; per-component represented-volume error ≤ ½s·A_cut,component / V_component + 10⁻⁴ |
| S7 thin chip | 0 skipped fragments; chip current volume vs Σm·J/ρ ≤ 5%; non-manifold edges report-only in v1 |
| S8 recontact | 0 vertices shared across conflicting codes; shell set unchanged since separation; penetration reported from contact stats |
| S9 / G8 negatives (side push, flat press, stop, withdrawal, partial cut, blunt handle) | 0 conflicting links ⇒ 0 cut faces and an empty Hermite log; mesh equal to the uncut G1 mesh embedded; all Hermite points inside the finite swept region |
| All | Volume, reference: \|V_mesh − Σm/ρ\| ≤ ½s·A_cut + 10⁻⁴V. Volume, current: ≤ 2%. On frames without code change: topology hash constant and vertex residual vs the embedding function ≤ 10⁻³ s |
| Phase 3 comparison gate (plan) | B not worse than A on every frozen metric, and better on cut-face current-space P95 and topological consistency |

These numbers come from the analytical expectations in §7.2 and the measured A controls, not from any cut result.
- Phase 1 controls may justify changing a candidate. If so, record the reason, and freeze the change before any
  S-test.
- Smoothness alone is never a success criterion (spec §3.6). Physical indentation and deformation must survive in
  current space.
- Performance (not a gate): target full rebuild ≤ A's measured 49–79 ms per step and per-frame embedding ≤ 5 ms at
  10⁵ particles on GPU. Report full rebuild vs incremental.

---

## 8. Risks and open questions

1. **Visual over-separation (§3.4).** A lattice crack bridged by code-0 stencil coupling renders as separate shells
   while the mechanics still couples them. It is flagged, never hidden. Large counts point to a law-level issue
   (Phase 4), not a surface fix.
2. **Sign-inconsistent links.** An earlier-assigned endpoint can read on the "wrong" side at the later instant. These
   use t = ½ and are counted. A large count would indicate band or lead behaviour worth a Phase 4 look.
3. **Plowed and torn material** (WP-11 Phase C: torn top-layer material carried ~61 mm). Embedding under very large,
   heterogeneous F is validated only for affine strain. S2 and S7 must report fallback counts.
4. **Render mesh ≠ contact surface** (spec §3.3). Contact uses particle spheres. Report the mesh-to-particle-surface
   offset; do not claim the mesh is the collider.
5. **Curved outer boundaries.** Choose `cell_face_v1` or `analytic_reference_v1` before measuring. Their volumes
   differ by the reported quadrature error.

---

## 9. Summary

**Recommendation:** implement `cut_aware_surface_v1` as Approach B, adapted to our solver:
- Wu et al.'s linked volume on the **1 mm particle lattice**, where particles are the cells and 6-neighbour bonds
  the links.
- Link disconnection **derived from conflicting packed codes**: one cut truth, the solver's.
- Hermite data from an **opt-in, read-only `cut_events_v1` recorder** in the material wrapper's substep loop. It
  needs no `tlmpm_v5.py` change and is exact with respect to the solver's own swept-side decision. A labelled
  midpoint fallback (R4) covers existing recordings.
- Local multi-region dual contouring with per-region duplicated representatives and no cross-region welding.
- A one-way, stateless embedding from each region's own particles with the first-order F term.
- Incremental per-cell updates.
- `particle_surface_v1` remains the frozen baseline and lattice-free fallback. Hybrid C is built only if Phase 1/3
  data require it. Elastic snapping (D) is not adopted.
- Evaluation separates reference-space reconstruction error, assignment-time labelling deviation and current-space
  embedding error. The thresholds in §7.3 are expressed in particle spacings and must be frozen before measurement.

**Claims not verified from primary sources:**
- BladeMaster's reference-grid resolution, link-intersection test, Hermite-normal source and the numerical meaning
  of "compatible TLMPM motion field": not stated in the paper. No code is published. The details above come from
  Wu et al. 2011 and are our interpretation.
- The elastic snapping paper (CGF 2025): full text unavailable (HTTP 403). Everything stated about it is from the
  abstract and search snippets.
- The Gerstner–Pajarola manifold test cited by Ju §4.1 and later manifold-DC work: not reviewed.
- Every runtime and memory figure for B, C and D, and the recorder's < 2% overhead, are **estimates** to be
  measured. Only A's figures are measured (validation doc; WP-11 Phase H).
