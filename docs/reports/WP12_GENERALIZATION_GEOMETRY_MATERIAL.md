# WP-12 Phase 4c — workpiece geometry and material generalization (spec §4.5–4.6)

**Status: DRAFT, pending xhigh review.** The orchestrator merges this into `WP12_CUTTING_PHYSICS.md` after review.
Every result here is **kinematic ideal tracking** (`runtime_source = kinematic_ideal_tracking`) on
`blade_master_flat_board_v1`. Nothing was run in Isaac Sim / Isaac Lab. All material values are
**PROVISIONAL_UNCALIBRATED**.

Evidence: [`docs/evidence/wp12/phase4c_geometry_material_001.json`](../evidence/wp12/phase4c_geometry_material_001.json).
Figures: `docs/media/wp12/phase4c/`. Runs: `artifacts/wp12/phase4c_002/<case>/` (`result.json`,
`final_state.npz`, `cut_aware_mesh.npz`). Preflight: `artifacts/wp12/phase4c_005/preflight.json`.

## 1. What changed (all opt-in, versioned)

| Item | Where | Default behaviour |
| --- | --- | --- |
| `workpiece.geometry_provider` (`heightfield_v1` default, `heightfield_v2`, `sdf_csg_experimental_v1`) | `src/robotic_cutting_sim/geometry_providers.py`, `scene_config.validate_scene / scene_heightfield / _check_geometry / scene_top_m / _footprint_local_xy`, `ConfiguredTaskSpec.sample_particles` | Absent key = today's path. An explicit `{"id": "heightfield_v1"}` normalizes to the absent key, so the scene JSON and task id are identical. |
| Per-protrusion `yaw_deg`, `peak_offset`, family `irregular` | `heightfield_v2` rows only | `heightfield_v1` rows still reject these keys and this family. |
| Planner footprint of a yawed protrusion | `goal_planner._support_along` | Rows without `yaw_deg` use exactly `radius_x/radius_y` as before. |
| `materials_wp12_v1` profiles | `configs/materials/materials_wp12_v1.json`, `src/robotic_cutting_sim/materials_wp12.py`, `scene_config.scene_material` | `cutting_tasks_v1` and `food_profiles_v1` unchanged. |
| Stability admission (elastic-wave CFL) | `materials_wp12.admit`, called by `scene_config.make_configured_kinematic_env` for a wp12 profile | It raises `MaterialStabilityError` before any solver is built. Other profiles are not checked there (unchanged). |

No solver (`tlmpm*.py`), cutting law, threshold, frozen record or evaluator changed. `fixture_contacts`,
`live_material`, `kinematic_env` and `live_env` are untouched by this work. Other agents' uncommitted edits to
`kinematic_env.py`, `live_material.py` and the surface modules were present in the tree during the runs (see §7).

**Identity checks.** `compile_task` of `bar_flat_board + sever_mid` and `trim01_flat_board + trim_bump_rho10` gives
`CFG-bar_flat_board-sever_mid-74cbdae6` and `CFG-trim01_flat_board-trim_bump_rho10-ec69e5d0` both with a `git
archive HEAD` copy of `src/` and with the working tree. The pin tests `tests/test_scene_goal_config.py` and
`tests/test_wp12_flat_board_workcell.py` pass, and so do 208 tests in 15 further suites that import
`scene_config`/`goal_planner`. `heightfield_v2` without yaw, offset or noise samples bit-identical particles,
masses and lattice to `sample_specimen` (test).

## 2. Geometry providers

All providers sample cell centres of the same box lattice as `benchmark.sample_specimen`:
`q = (ijk + ½) s − (L/2, W/2, 0)`, strict `q_x < L/2`, `q_y < W/2`, float32 points, int32 lattice and fixed
`ρ s³` masses. The `cut_aware_surface_v1` lattice links and the `cut_events_v1` recorder therefore work
unchanged.

* **`heightfield_v1`** (default): one max-union height field over a rectangular base box, with axis-aligned
  elliptical bumps of the families dome, gaussian, ellipsoid_cap and ridge. Bar and block sizes, thin slabs, aspect
  ratios, several protrusions and **overlapping** bumps (max-union, one body) were already expressible.
* **`heightfield_v2`** adds three things.
  * Bump `yaw_deg`: rotated elliptical bumps.
  * `peak_offset ∈ [−0.8, 0.8]`: the peak moves along the bump's own x axis and each flank is the family profile
    stretched to its side. This gives a steep and a shallow flank. The footprint is two half-ellipses joined at the
    peak, inside the declared rectangle. It is **not** the original ellipse.
  * The `irregular` family, and optional seeded low-amplitude `surface_noise`: a sum of random-phase plane cosines
    with |noise| ≤ `amplitude_m`, deterministic from `seed`.

  It is still a height field.
* **`sdf_csg_experimental_v1`** (EXPERIMENTAL): the base box minus cylinders (along x or y), boxes and spheres,
  tested at cell centres. It expresses through-holes, tunnels, undercuts and overhangs. It is not an
  imported-mesh or arbitrary-SDF sampler and takes no protrusions. It has no top height field, so
  `remove_protrusions` and the TRIM surface metrics are unsupported on it.

**Not supported by any provider:**

* imported meshes or arbitrary closed surfaces;
* several separate bodies;
* non-rectangular scene footprints (the capsule exists only on the legacy task path);
* sub-cell (fractional) boundary volumes: curved boundaries are cell-centre terraces;
* **workpiece yaw relative to the flat board.** `flat_board_v1` has no oriented TLMPM fixture boxes, so the
  workpiece yaw must equal the board yaw. A yawed workpiece needs a co-rotated board (`cutting_board.yaw_deg`
  override), and that board must still fit on the table.

**This is not arbitrary-object support.**

**Validation at scene compile (non-default providers).**

* `geometry_providers.lattice_report` runs on the sampled lattice. It requires one 6-connected body, 0
  floating particles and a nonempty bottom layer `k = 0` (cell centre s/2 above the board top).
* Every protrusion's conservative axis-aligned footprint must lie inside the base. For a yawed bump this is the
  rotated ellipse, or the rotated rectangle when it has a peak offset.
* The spacing must give two layers under the lowest point of the noise.
* The existing `check_placement` keeps the footprint on the board (5 mm margin) and the workpiece bottom equal to
  the board top.

Declared refusals, all refused with the expected reason (`preflight.json` → `refusals`):

| Refusal case | Reason |
| --- | --- |
| 30° workpiece on the unrotated board | `workpiece_yaw_differs_from_board` |
| translated 160 mm in x | `workpiece_off_board` |
| SDF box slot cutting the bar in two | `disconnected_material` |
| noise amplitude 2.5 mm on a 4 mm base | `spacing_too_coarse` |
| yaw-90 ellipse wider than the bar | `protrusion_outside_base` |
| `wp12_stiff_v1` at 100 µs | `MaterialStabilityError` |

## 3. Run protocol

* Command per case: `TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4c.py run --case <C>
  --device cuda:1 --out artifacts/wp12/phase4c_002/<C>`, queued two at a time on cuda:1. The GPU was shared with the
  resolution study, so wall times are contended. Logs are `artifacts/wp12/logs/phase4c_002_*.log`.
* Setup: `blade_master_flat_board_v1`, `ruler_knife_v2_y`, `tlmpm_v5` cutting law 8 (unchanged), 1 mm particles,
  4 mm grid, material dt 0.1 ms (50 µs for `wp12_stiff_v1`), seed 42, `cut_events_v1` recorder on, history stride
  10. Software: Warp 1.17.0, NumPy 2.5.4, SciPy 1.18.1. Runs started from git HEAD `b012902` with a dirty tree; HEAD
  moved to `9157725` (another agent's Phase 3 commit) during the queue. Per-run sha and runner sha256 are in each
  `result.json`.
* **CUT** = `configs/wp11/goals/sever_mid.json` thresholds at local x = 0. **TRIM** =
  `configs/wp11/goals/trim_bump_rho10.json` thresholds: remove every protrusion, ρ = 10°, `path_v2`. The goal
  planner's own plan drives the edge (WaypointPolicy). Episode duration = plan end + 1 s, rounded up to 0.5 s.
* **Measured per run:**
  * mechanical components and mass fractions;
  * direct code-0 bridges per digit (`analyze_cut_bridges.py` rule);
  * goal verdict with integrity / outcome;
  * min det F (every 5 steps and every armed step);
  * max particle speed;
  * blade reaction, armed mean and max;
  * deformation near the cut. CUT: reference distance to the plane < 3 mm, at the last HOLD step. TRIM: substrate
    within 3 mm below the plane, at the last SKIM step. Reported as |x − X| percentiles, |det F − 1| p95 and
    top-surface sag;
  * settle state at the first armed step;
  * `cut_aware_surface_v1` on the final state with the recorder log (audit, shells vs components);
  * wall time.
* Classification rules were declared in `scripts/summarize_wp12_phase4c.py` before the summary and are copied into
  the evidence JSON.
* **Repeat noise.** `M01_reference_*` repeats `G01_bar_ref_CUT` / `G11_dome_TRIM` with the identical recipe under
  the `wp12_reference_v1` id. The codes and backend parameters are identical. Final positions differ by at most
  0.006 mm (CUT) and 0.05 mm (TRIM): GPU float atomics.

## 4. Geometry capability matrix

One run per cell. VALIDATED / PARTIAL / FAIL / UNSUPPORTED / N/A (not applicable) / ERROR, per the rules in
the evidence JSON.

| Geometry | Provider | Particles | Compile | Sampling | Settle | CUT | TRIM | Cut-aware (shells = components) | Goal eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 bar 80×24×16 (reference) | v1 | 30720 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G02 bar short 50×24×16 | v1 | 19200 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G03 bar long 150×24×16 | v1 | 57600 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G04 bar narrow 80×12×16 | v1 | 15360 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G05 bar wide 80×48×16 | v1 | 61440 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G06 bar tall 60×24×32 | v1 | 46080 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G07 bar low 80×24×8 | v1 | 15360 | VALIDATED | VALIDATED | VALIDATED | **FAIL** | N/A | VALIDATED | VALIDATED |
| G08 block 50×40×30 | v1 | 60000 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G09 thin slab 100×40×4 | v1 | 16000 | VALIDATED | VALIDATED | VALIDATED | **FAIL** | N/A | **FAIL** (audit ok; 2 shells / 1 component) | VALIDATED |
| G10 rod 140×10×10 | v1 | 14000 | VALIDATED | VALIDATED | VALIDATED | **FAIL** | N/A | VALIDATED | VALIDATED |
| G11 dome (r 10, h 6) | v1 | 23588 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED |
| G12 gaussian bump | v1 | 23472 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED |
| G13 elliptical dome 14×7 | v1 | 23580 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED |
| G14 three protrusions (dome, gaussian, ellipsoid cap) | v1 | 36220 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | **FAIL** (audit ok; 13 shells / 7 components) | VALIDATED |
| G15 two overlapping gaussians | v1 | 23804 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | **ERROR** (slot overflow) | VALIDATED (CUT only) | VALIDATED (CUT only) |
| G15b same, `slots_per_node` 16 (follow-up) | v1 | 23804 | VALIDATED | VALIDATED | VALIDATED | N/A | **FAIL** (0 % removed) | **FAIL** (audit ok; 16 shells / 1 component) | VALIDATED |
| G16 asymmetric: yaw 30°, peak offset 0.5 | v2 | 29340 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED |
| G17 irregular family | v2 | 23598 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | **FAIL** (audit ok; 3 shells / 2 components) | VALIDATED |
| G18 dome + surface noise 0.5 mm / λ 12 mm | v2 | 23590 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | VALIDATED | **FAIL** (audit ok; 3 shells / 2 components) | VALIDATED |
| G19 translated (+80, +40) mm on the board | v1 | 30720 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G20 yaw 30° with co-rotated board | v1 | 30720 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | N/A | VALIDATED | VALIDATED |
| G21 SDF tunnel (Ø 8 mm along y) | SDF exp. | 29472 | VALIDATED | VALIDATED | VALIDATED | **FAIL** | UNSUPPORTED | VALIDATED | VALIDATED |
| G22 SDF arch / bottom undercut (16 × 5 mm) | SDF exp. | 28800 | VALIDATED | VALIDATED | VALIDATED | VALIDATED | UNSUPPORTED | VALIDATED | VALIDATED |
| Workpiece yaw relative to the board | — | — | refused | — | — | — | — | — | — |
| Imported mesh / arbitrary SDF / several bodies | — | — | UNSUPPORTED | — | — | — | — | — | — |

Per-run measurements (Comp. = components; Bridges = direct code-0 bridges, CUT kerf rule; force = blade reaction
while armed; Settle = max |x − X| at the first armed step):

| Run | Verdict | Comp. | Bridges | Removed | min det F | max speed (m/s) | force mean / max (N) | shells / comp. | Settle (mm) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 CUT | PASS | 2 | 0 | | 0.955 | 0.053 | 0.0045 / 0.034 | 2 / 2 | 0.11 | 146 |
| G02 CUT | PASS | 2 | 0 | | 0.954 | 0.046 | 0.0041 / 0.032 | 2 / 2 | 0.11 | 97 |
| G03 CUT | PASS | 2 | 0 | | 0.962 | 0.056 | 0.0048 / 0.034 | 2 / 2 | 0.12 | 231 |
| G04 CUT | PASS | 2 | 0 | | 0.960 | 0.059 | 0.0022 / 0.016 | 2 / 2 | 0.11 | 108 |
| G05 CUT | PASS | 2 | 0 | | 0.952 | 0.045 | 0.0094 / 0.070 | 2 / 2 | 0.11 | 152 |
| G06 CUT | PASS | 2 | 0 | | 0.959 | 0.062 | 0.0112 / 0.047 | 2 / 2 | 0.25 | 128 |
| G07 CUT | FAIL | 1 | 8 | | 0.949 | 0.064 | 0.0014 / 0.022 | 1 / 1 | 0.07 | 71 |
| G08 CUT | PASS | 2 | 0 | | 0.949 | 0.063 | 0.0163 / 0.051 | 2 / 2 | 0.23 | 158 |
| G09 CUT | FAIL | 1 | 4 | | 0.972 | 0.113 | 0.0008 / 0.021 | 2 / 1 | 0.06 | 68 |
| G10 CUT | FAIL | 1 | 8 | | 0.950 | 0.042 | 0.0012 / 0.007 | 1 / 1 | 0.08 | 82 |
| G11 CUT / TRIM | PASS / PASS | 2 / 2 | 0 / 0 | 0.942 | 0.968 / 0.908 | 0.047 / 0.299 | 0.0027 / 0.027 ; 0.0126 / 0.047 | 2/2 ; 2/2 | 0.10 | 100 / 75 |
| G12 CUT / TRIM | PASS / PASS | 2 / 2 | 0 / 0 | 0.921 | 0.969 / 0.920 | 0.050 / 0.330 | 0.0031 / 0.030 ; 0.0092 / 0.038 | 2/2 ; 2/2 | 0.16–0.19 | 87 / 73 |
| G13 CUT / TRIM | PASS / PASS | 2 / 2 | 0 / 0 | 0.944 | 0.965 / 0.921 | 0.052 / 1.136 | 0.0032 / 0.027 ; 0.0116 / 0.044 | 2/2 ; 2/2 | 0.10 | 95 / 98 |
| G14 CUT / TRIM | PASS / PASS | 2 / 7 | 0 / (n.a.) | 0.956 / 0.935 / 0.991 | 0.969 / 0.879 | 0.065 / 1.025 | 0.0034 / 0.030 ; 0.0210 / 0.067 | 2/2 ; 13/7 | 0.16–0.18 | 214 / 335 |
| G15 CUT / TRIM | PASS / ERROR | 2 / — | 0 / — | — | 0.964 / — | 0.049 / — | 0.0033 / 0.027 ; — | 2/2 ; — | 0.10 | 148 / 52 |
| G15b TRIM (slots 16) | FAIL | 1 | (n.a.) | 0.000 / 0.000 | 0.822 | 0.796 | 0.0165 / 0.063 | 16 / 1 | 0.11 | 247 |
| G16 CUT / TRIM | PASS / PASS | 2 / 2 | 0 / 0 | 0.946 | 0.958 / 0.853 | 0.062 / 0.330 | 0.0057 / 0.036 ; 0.0119 / 0.038 | 2/2 ; 2/2 | 0.16–0.17 | 98 / 85 |
| G17 CUT / TRIM | PASS / PASS | 2 / 2 | 0 / 0 | 0.948 | 0.968 / 0.915 | 0.049 / 0.501 | 0.0033 / 0.028 ; 0.0118 / 0.046 | 2/2 ; 3/2 | 0.10 | 87 / 74 |
| G18 CUT / TRIM | PASS / PASS | 2 / 2 | 0 / 0 | 0.925 | 0.968 / 0.845 | 0.048 / 0.423 | 0.0027 / 0.027 ; 0.0073 / 0.033 | 2/2 ; 3/2 | 0.10 | 87 / 76 |
| G19 CUT | PASS | 2 | 0 | | 0.955 | 0.054 | 0.0045 / 0.034 | 2 / 2 | 0.11 | 196 |
| G20 CUT | PASS | 2 | 0 | | 0.955 | 0.055 | 0.0045 / 0.034 | 2 / 2 | 0.11 | 200 |
| G21 CUT | FAIL | 1 | 12 | | 0.940 | 0.157 | 0.0065 / 0.034 | 1 / 1 | 0.12 | 195 |
| G22 CUT | PASS | 2 | 0 | | 0.955 | 0.065 | 0.0023 / 0.032 | 2 / 2 | 0.21 | 191 |

The direct-bridge count is a CUT-kerf rule. In multi-digit TRIM runs it also counts unlabelled particles next to
chips that lie on the substrate, so it is not interpreted for TRIM ("n.a.").

### 4.1 Findings

1. **Thin pieces fail the kinematic CUT. Cause found for the controller; mechanism not established.**
   * **Failures:** G07 (8 mm), G09 (4 mm), G10 (10 × 10 mm rod) and G21 (4 mm ligament under the tunnel) each
     end as one component with 4–12 direct code-0 bridges.
   * **Where:** all bridges are in the bottom layer (reference z = 0.5 mm). G07, G10 and G21 also have
     unconflicted kerf links in layer 0 only: 20/24, 8/10 and 20/24 conflicting. Every other layer is fully
     conflicting.
   * **Controller cause:** `sever_mid` uses the frozen v1 controller profile, whose ARMED stroke ends with the
     edge 0.4 mm above the board. The last 0.4 mm happens in the disarmed HOLD. That is just outside the measured
     kinematic clean-cut window [0, 0.375] mm (WP12_WORKCELL.md §5), and it is the same for every geometry.
   * **Diagnostic (declared after the failures, labelled `DIAGNOSTIC_CONTROLLER_VARIANT`):** the same four scenes
     with `planner.controller_profile = path_v2` (armed stroke ends at 0.2 mm) all PASS. Each gives 2 components,
     0 bridges, and 2 shells for 2 components.
   * **Not established:** why ≥ 12 mm pieces pass at 0.4 mm while ≤ 10 mm pieces fail. Bottom-layer settle
     heights differ by only 0.03 mm (0.41–0.45 mm). This is for the Phase 4 xhigh analysis.
   * **Matrix status:** the matrix cells stay FAIL. The default configured sever plan on the flat board is
     marginal at the bottom layer.
2. **G22 arch passes.** Under the cut plane its ligament ends in air above the notch, not on the board. This is
   consistent with finding 1, where every failure sits in a ligament resting on the board.
3. **Overlapping bumps (G15) are not TRIM-able with `remove_protrusions`.**
   * **G15 (8 slots per node):** raised `SlotOverflowError` at 52 s (1115 entries). The failed result is kept.
   * **G15b (opt-in `slots_per_node = 16`):** ran clean (max 9 slots occupied) but removed 0 % of both bumps.
   * **Cause, a planner limitation:** `goal_planner._protrusion_passes` clips each pass at the mid-gap between
     neighbours, so pass p0 ends and pass p1 starts at x = 0 inside the merged bump. The planner itself warns
     "pass crosses the footprint". No chip can exit.
   * **Diagnostic `G15_overlapping_TRIMPLANE_DIAG`** (one `trim_plane` pass at the base top over |x| ≤ 20 mm,
     `DIAGNOSTIC_GOAL_VARIANT`): PASS, 0.935 of the above-plane mass removed, substrate share 1.0, 8 slots
     sufficient.
   * Overlapping bumps are expressible and CUT-able; per-protrusion TRIM of overlapping bumps is not supported by
     the current planner.
4. **The cut-aware surface audit flags real mechanical bridges.** The FAIL cells in that column all have
   `audit_ok = True` with 0 audit errors. They report a mechanical component whose mesh has several lattice pieces
   ("visually separated, mechanically bridged").
   * G09: the halves look separate but are joined by the 4 bottom-layer bridges.
   * G14, G17, G18 and G15b: chip fragments stay stencil-connected to another piece.
   * So "shells = components" does not hold in those runs, and the surface correctly says so.
5. **Yaw and translation.**
   * Translation on the board (G19) and a co-rotated board at 30° (G20) reproduce G01. They have the same
     verdict, the same force within 0.0001 N and the same min det F.
   * Workpiece yaw relative to the board is refused at compile.
6. **Settle.** Every DONE run met the settle rule at its first armed step: max |x − X| 0.06–0.25 mm, speed
   < 10 mm/s, min particle z ≥ 0.41 mm above the board, centroid xy drift < 0.5 mm. Taller pieces compress more
   (G06 0.25 mm, G08 0.23 mm).
7. **Cost.** Wall time ranged 68–335 s per run on the contended cuda:1 (180–560 ms per control step at 14–61k
   particles, including the per-step host copies of this runner). The stiff profile at 50 µs took 840–910 ms per
   step.

## 5. Material profiles (`materials_wp12_v1`, PROVISIONAL_UNCALIBRATED)

Each profile is a full `MaterialParameters` recipe and each is a one- or two-factor change from the reference.
The solver mapping is in the config:

* density → particle mass;
* E, ν → elastic predictor;
* yield → perfect J2 return mapping;
* friction → Coulomb coefficient of **both** blade and board contact;
* G_c → the law-8 edge-work threshold;
* hardening is fixed at 0 (`LiveCuttingMaterial` rejects hardening).

Stability rule: longitudinal wave speed c = √((λ+2μ)/ρ). The grid CFL dt·c/dx must be ≤ 0.2 on the 4 mm grid, the
same rule as `food_profiles_v1`. The 1 mm particle-spacing CFL is reported only: the frozen reference itself is
0.67 there, and TLMPM integrates on the grid.

| Profile | Interpretation | ρ (kg/m³) | E (kPa) | ν | σy (Pa) | μ | G_c (J/m²) | c (m/s) | grid CFL @ 100 µs | run dt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wp12_reference_v1 | control = frozen cutting_tasks_v1 | 1500 | 50 | 0.3 | 1000 | 0.3 | 0.05 | 6.70 | 0.167 admit | 100 µs |
| wp12_soft_elastic_v1 | E ÷ 2.5, yield effectively off | 1500 | 20 | 0.3 | 1e6 | 0.3 | 0.05 | 4.24 | 0.106 admit | 100 µs |
| wp12_clay_plastic_v1 | yield ÷ 2.5 (ρgH = 235 Pa for 16 mm) | 1500 | 50 | 0.3 | 400 | 0.3 | 0.05 | 6.70 | 0.167 admit | 100 µs |
| wp12_firm_v1 | stiffest admissible at 100 µs | 1500 | 70 | 0.3 | 3000 | 0.3 | 0.05 | 7.93 | 0.198 admit | 100 µs |
| wp12_stiff_v1 | E × 3, yield × 5 | 1500 | 150 | 0.3 | 5000 | 0.3 | 0.05 | 11.60 | **0.290 REFUSED** | 50 µs (CFL 0.145) |
| wp12_low_friction_v1 | slippery | 1500 | 50 | 0.3 | 1000 | 0.05 | 0.05 | 6.70 | 0.167 admit | 100 µs |
| wp12_high_friction_v1 | sticky | 1500 | 50 | 0.3 | 1000 | 0.8 | 0.05 | 6.70 | 0.167 admit | 100 µs |
| wp12_low_cut_resistance_v1 | G_c ÷ 5 | 1500 | 50 | 0.3 | 1000 | 0.3 | 0.01 | 6.70 | 0.167 admit | 100 µs |
| wp12_high_cut_resistance_v1 | G_c × 10 | 1500 | 50 | 0.3 | 1000 | 0.3 | 0.5 | 6.70 | 0.167 admit | 100 µs |
| wp12_light_v1 | ρ 1100 | 1100 | 50 | 0.3 | 1000 | 0.3 | 0.05 | 7.82 | 0.196 admit | 100 µs |
| wp12_dense_v1 | ρ 2200 | 2200 | 50 | 0.3 | 1000 | 0.3 | 0.05 | 5.53 | 0.138 admit | 100 µs |

Standard runs:

* CUT = G01 (80×24×16 bar, sever at x = 0).
* TRIM = G11 (same bar at 12 mm with a dome r 10 / h 6 mm).
* `|J−1|` = |det F − 1| p95 near the cut (CUT: within 3 mm of the plane; TRIM: substrate within 3 mm below the
  plane).
* sag = max top-surface sag next to the kerf.

| Case | CUT verdict, comp. / bridges | CUT force mean / max (N) | CUT \|J−1\| | CUT sag (mm) | CUT min det F | CUT max speed | TRIM verdict, removed | TRIM force mean / max (N) | TRIM \|J−1\| | TRIM min det F | TRIM max speed | Wall CUT / TRIM (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01 reference | PASS 2 / 0 | 0.0045 / 0.034 | 0.0213 | 0.53 | 0.955 | 0.053 | PASS 0.942 | 0.0126 / 0.047 | 0.0118 | 0.908 | 0.306 | 190 / 151 |
| M02 soft elastic | PASS 2 / 0 | 0.0080 / 0.034 | 0.0425 | 0.43 | 0.948 | 0.172 | PASS 0.942 | 0.0142 / 0.034 | 0.0150 | 0.889 | 0.383 | 197 / 154 |
| M03 clay plastic | PASS 2 / 0 | 0.0072 / 0.029 | 0.0135 | 0.76 | 0.970 | 0.035 | PASS 0.923 | 0.0093 / 0.032 | 0.0115 | 0.883 | 0.381 | 201 / 156 |
| M04 firm | PASS 2 / 0 | 0.0033 / 0.034 | 0.0336 | 0.52 | 0.947 | 0.096 | PASS 0.953 | 0.0191 / 0.074 | 0.0142 | 0.909 | 0.598 | 197 / 165 |
| M05 stiff (50 µs) | PASS 2 / 0 | 0.0034 / 0.075 | 0.0529 | 0.31 | 0.903 | 0.108 | PASS 0.953 | 0.0502 / 0.162 | 0.0132 | 0.889 | 0.620 | 391 / 297 |
| M06 low friction | PASS 2 / 0 | 0.0026 / 0.029 | 0.0223 | 0.52 | 0.948 | 0.049 | PASS 0.942 | 0.0121 / 0.047 | 0.0111 | 0.909 | 0.274 | 195 / 182 |
| M07 high friction | PASS 2 / 0 | 0.0050 / 0.034 | 0.0216 | 0.54 | 0.955 | 0.052 | PASS 0.942 | 0.0126 / 0.047 | 0.0120 | 0.907 | 0.600 | 237 / 149 |
| M08 low cut resistance | PASS 2 / 0 | 0.0044 / 0.034 | 0.0214 | 0.53 | 0.955 | 0.052 | PASS 0.942 | 0.0126 / 0.047 | 0.0119 | 0.908 | 0.320 | 235 / 157 |
| M09 high cut resistance | PASS 2 / 0 | 0.0056 / 0.046 | 0.0219 | 0.54 | 0.955 | 0.055 | PASS 0.942 | 0.0125 / 0.047 | 0.0118 | 0.908 | 0.376 | 227 / 196 |
| M10 light | PASS 2 / 0 | 0.0022 / 0.020 | 0.0243 | 0.47 | 0.944 | 0.075 | PASS 0.945 | 0.0111 / 0.042 | 0.0117 | 0.921 | 0.362 | 198 / 191 |
| M11 dense | PASS 2 / 0 | 0.0114 / 0.056 | 0.0182 | 0.67 | 0.960 | 0.041 | PASS 0.942 | 0.0155 / 0.060 | 0.0122 | 0.892 | 0.431 | 192 / 193 |

Changes in the final state against the reference profile: codes that differ, and final-position p95 / max
(repeat noise: CUT max 0.006 mm, TRIM max 0.05 mm).

| Profile | CUT codes differ | CUT final position p95 / max (mm) | TRIM codes differ | TRIM final position p95 (mm) |
| --- | --- | --- | --- | --- |
| soft elastic | 84 | 0.23 / 1.52 | 34 | 0.16 |
| clay plastic | 84 | 0.35 / 1.37 | 42 | 0.12 |
| firm | 68 | 0.53 / 2.54 | 10 | 0.17 |
| stiff (50 µs) | 324 | 1.44 / 3.97 | 34 | 0.18 |
| low friction | 4 | 0.10 / 0.50 | 24 | 0.03 |
| high friction | **0** | 0.01 / 0.06 | **0** | 0.03 |
| low / high cut resistance | **0 / 0** | (force only) | **0 / 0** | — |
| light | 12 | 0.30 / 0.65 | 2 | 0.05 |
| dense | 48 | 0.24 / 0.62 | 4 | 0.06 |

TRIM final-position maxima (30–40 mm) are the detached chip landing in different places; they are not substrate
deformation.

### 5.1 Findings

1. **Elastic and plastic profiles produce measurably different mechanics.** Near-kerf |det F − 1| p95 orders with
   the elastic strain the material can carry before yielding (σy/E):

   | Profile | σy/E | CUT \|J−1\| p95 |
   | --- | --- | --- |
   | clay plastic | 0.008 | 0.0135 |
   | reference | 0.02 | 0.0213 |
   | stiff | 0.033 | 0.0529 |
   | firm | 0.043 | 0.0336 |
   | soft elastic | (no yield) | 0.0425 |

   The ordering of stiff vs firm does not follow σy/E, so this is an observed trend, not a law.
   * The plastic profile keeps the largest permanent top-surface sag at the kerf (0.76 mm vs 0.53 mm) and removes
     slightly less of the dome (0.923 vs 0.942).
   * The soft elastic profile rebounds: max speed 0.17 m/s vs 0.05 m/s, less sag (0.43 mm).
   * The stiff profile has 4× the TRIM blade force (0.050 N vs 0.013 N) and the lowest min det F (0.903).
2. **Density changes the blade reaction roughly in proportion:** CUT mean 0.0022 N (light) / 0.0045 N (reference)
   / 0.0114 N (dense). The CUT blade reaction in this setup is dominated by the material's weight and inertia on
   the edge rather than by a cutting resistance.
3. **Friction has a weak effect in these two tasks.**
   * μ 0.05 lowers the CUT mean force by 42 % (0.0026 N vs 0.0045 N).
   * μ 0.8 is indistinguishable from μ 0.3 in codes, positions and forces, apart from a +11 % CUT mean force. The
     doubled TRIM max speed (0.60 vs 0.31 m/s) is a single flying chip particle; TRIM max speed varies between
     repeats of other rows too, so it is not treated as a friction effect.
   * Both standard tasks are pure edge-normal motions; friction-dominated tasks (sawing, pushing) were not run.
4. **"Cutting resistance" (G_c) does not change separation under law 8.** G_c 0.01 / 0.05 / 0.5 give **identical**
   labelled sets and codes in both CUT and TRIM.
   * Law 8 assigns codes geometrically (swept band, plow gate).
   * G_c only enters the per-particle edge-work progress that removes the in-plane resisting impulse. It changes
     only the CUT blade force: max 0.034 → 0.046 N at 10× G_c, and 0.034 N at G_c ÷ 5.
   * The low and high cut-resistance profiles are therefore distinct parameter sets, but **not** distinct
     separation behaviours. A material whose separation depends on toughness would need a cutting-law change,
     which is out of scope here.
5. **Every profile PASSES both standard tasks** (2 components and 0 bridges for CUT, removal ≥ 0.923 for TRIM).
   The profiles change deformation and forces, not task success, at this geometry.
6. **Stability.** The admission rule refused `wp12_stiff_v1` at 100 µs (CFL 0.290). At 50 µs it ran finite with
   min det F 0.889–0.903 at twice the wall time (391 s CUT). No admitted run produced a non-finite state. Light
   (0.196) and firm (0.198) sit at the limit, so at 100 µs E ≈ 70 kPa (ρ 1500) or ρ ≈ 1100 (E 50 kPa) is the
   stiffest or lightest that can be admitted.

## 6. Figures

* `docs/media/wp12/phase4c/geometry_gallery.png`: sampled reference lattice of every geometry, top surface and
  y = 0 section.
* `docs/media/wp12/phase4c/geometry_cut_final.png`, `geometry_trim_final.png`: final states coloured by mechanical
  component.
* `docs/media/wp12/phase4c/material_cut_final.png`: final CUT state per profile.
* `docs/media/wp12/phase4c/material_comparison.png`: per-profile CUT/TRIM blade force, near-cut |det F − 1|, sag,
  speed, removal and wall time; the dashed line is the reference profile.

## 7. Open issues

1. **Bottom-layer marginality of the default sever plan.** Armed stroke ends 0.4 mm above the board. Thin pieces
   (≤ 10 mm) fail and ≥ 12 mm pieces pass. `path_v2` (0.2 mm) passes all four. The mechanism behind the height
   dependence needs the Phase 4 xhigh analysis. A versioned default change (planner or controller) is a
   decision, not made here.
2. **Overlapping-protrusion TRIM.** Planner mid-gap clipping makes `remove_protrusions` fail. G15 at 8 slots also
   overflowed. `trim_plane` works. A planner rule for merged footprints is future work.
3. **Material cutting resistance is not a separation property under law 8** (finding 5.1.4). Friction effects
   are weak in edge-normal tasks.
4. **SDF provider is experimental.** It has only analytic primitives, no imported meshes and no TRIM. Its CUT is
   validated on the arch (G22) only; the tunnel (G21) failed under the default plan and passed with `path_v2`.
5. **No live (Isaac) run and one seed per cell.** Repeat noise was measured only once (G01/G11 vs M01).
6. **Tree state during runs.** Other agents' uncommitted edits were in the tree:
   * `kinematic_env.py` and `live_env.py`: Layer-6 `surface_profile`, unused here (default None);
   * `live_material.py`: an opt-in recorder summary capacity, default unchanged;
   * HEAD moved to `9157725` mid-queue.

   Physics code (`tlmpm_v5.py`, `fixture_contacts.py`) was unchanged.
7. **Runner revisions.** `G15b`, the `_pathv2_DIAG` rows and the `TRIMPLANE_DIAG` row were added to
   `scripts/run_wp12_phase4c.py` after the failures they follow up (time-stamped in the code). Each result records
   its runner sha256. Preflights `phase4c_001` and `phase4c_003` predate those additions and the bump-extent fix
   (§2). `phase4c_005` is the final one, and every run's task id equals it.

## 8. Reproduce

```
TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4c.py preflight --out artifacts/wp12/phase4c_005/preflight.json
TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4c.py run --case <case> --device cuda:1 --out artifacts/wp12/phase4c_002/<case>
.venv/bin/python scripts/summarize_wp12_phase4c.py --preflight artifacts/wp12/phase4c_005/preflight.json \
    --runs artifacts/wp12/phase4c_002 --out docs/evidence/wp12/phase4c_geometry_material_001.json --media docs/media/wp12/phase4c
.venv/bin/python -m pytest -q tests/test_wp12_geometry_material.py tests/test_scene_goal_config.py \
    tests/test_wp12_flat_board_workcell.py tests/test_repo_naming.py     # 82 passed
```
