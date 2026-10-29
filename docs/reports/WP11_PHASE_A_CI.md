# WP-11 Phase A — CI and reproducibility

**Branch:** `claude/wp11-cutting-refinement`.
**Commits:** `97d56d3` (tests, fixture, extra, workflow) and `cce395e`
(full-history checkout).
**Starting point:** CI run 40915870753 on main `3e26d6b` had 5 failed,
748 passed, 27 skipped and 5 xfailed.

## Result

| Run | Where | Result |
| --- | --- | --- |
| 40922921407 (`97d56d3`) | GitHub Actions | **PASS**: 755 passed, 27 skipped, 5 xfailed, 0 failed (18 m 51 s) |
| 40926922358 (`cce395e`, full history) | GitHub Actions; AMD EPYC 7763, OpenBLAS `Haswell` | **FAIL** (33 m 07 s): 819 passed, **1 skipped**, 5 xfailed, **1 failed**. Every Phase A test passes on this AVX2 runner and the 26 law-regression tests now run. The one failure is `tests/test_tool_profile_v2.py::test_v1_profile_outputs_are_pinned_bitwise`, added by Phase B in `a8e4d25`. It is the same root cause; see the section on the open failure below. |
| Baseline `9bd8a29`, clean venv, `OPENBLAS_CORETYPE=Haswell` | local | 5 failed, 774 passed, 1 skipped, 5 xfailed (14 m 02 s). The same 5 failures as CI. |
| `97d56d3`, clean venv, `OPENBLAS_CORETYPE=Haswell` | local | **PASS**: 781 passed, 1 skipped, 5 xfailed (14 m 28 s) |
| `cce395e`, clean venv, native kernel (SkylakeX), full history | local | **PASS**: 820 passed, 1 skipped, 5 xfailed (15 m 34 s). This includes the Phase B tests committed in the meantime. |

The clean venv is `/mnt/data/ningan/tmp/ci_repro_venv`. It was installed from
a fresh clone exactly as the workflow installs it:
`python -m pip install -c requirements.lock -e '.[policy]' pytest usd-core warp-lang`
before the change, and `-e '.[policy,test]'` after it. The suite was run with
`python -m pytest -q -rsx -p no:cacheprovider --basetemp=...` and
`TMPDIR=/mnt/data/ningan/tmp`. Logs are in
`artifacts/wp10_claude/wp11_phase_a_ci/`.

## Root cause 1: the rollout hashes depended on the BLAS kernel

**Versions are identical.** CI, the local `.venv` and the clean venv all have:
- NumPy 2.5.4 and SciPy 1.18.1;
- gymnasium 1.4.0, warp-lang 1.17.0 and usd-core 26.8;
- PyYAML 6.0.3 and pytest 9.1.1.

Python is 3.12.15 on CI and 3.12.12 locally. The only difference that
matters is the CPU.

**What differs.** NumPy's bundled OpenBLAS 0.3.34 picks its kernel at run
time from the CPU it finds:
- The local machine is an AMD Ryzen Threadripper PRO 9975WX with AVX-512, so
  OpenBLAS selects `SkylakeX`.
- Setting `OPENBLAS_CORETYPE=Haswell` (or `Zen`, `Sandybridge`, `Nehalem` or
  `Prescott`) on the local machine reproduces the **four CI digests exactly**.
  For example, the saw rollout gives `2dacfbb0…cdcc0207` and SLICE-PUSH v1
  gives `7b35da0e…5540781a`.
- So the failing CI runner used an AVX2 OpenBLAS kernel. This is inferred
  from the digests: that run did not log its CPU.
- Run 40922921407 now logs its environment. It ran on an AMD EPYC 9V74
  (AVX-512) and OpenBLAS reported `Core: SkylakeX`, so the old pins would
  have passed on that runner.

GitHub's `ubuntu-latest` hardware varies between runs, and the pins were
hardware-dependent, not just environment-dependent.

**Mechanism.** `cblas_ddot` on 3-vectors rounds differently in the two
kernels. This affects `v @ w` and `np.linalg.norm`. In the sawing rollout at
step 40, `v @ v` is `2.2156845360086963e-07` under SkylakeX and
`2.2156845360086965e-07` under Haswell, one ulp apart. Quaternion and
rotation-vector code (SciPy `Rotation`) is not involved: rotation columns
are bit-identical.

**First differing step and size** (per-step action dump, SkylakeX vs Haswell):

| Rollout | First differing step | Steps that differ | max \|Δ action\| | max \|Δ edge position\| | Waypoints, cut flags, termination |
| --- | --- | --- | --- | --- | --- |
| CUT-01-R default | none | 0/600 | 0 | 0 | identical |
| CUT-01-R saw 3 mm + tilt 10° | 40 (x, z) | 117/600 | 8.7e-19 m | 3.7e-18 m | identical |
| v1 SLICE-PUSH-01 | 344 (x, z) | 15/800 | 4.3e-19 m | 0 | identical |
| v1 TRIM-01-R | 194 (z) | 1/800 | 4.3e-19 m | 0 | identical |
| path_v2 SLICE-PUSH-01 | 732 (x, z) | 3/800 | 4.3e-19 m | 0 | identical |
| v1 CUT-01-R, path_v2 TRIM-01-R, path_v2 CUT-01-R | none | 0 | 0 | 0 | identical |

**Verdict: a numeric-environment difference, not a policy-behaviour
difference.**
- The differences are at the 1e-19 m level, in components that are near
  zero.
- Rotation columns, waypoint switching, cut flags and the done step are
  identical.
- Controller state and time do not differ: the timestamps are `k*dt`, and
  the waypoint index sequence is identical.
- The plan hashes passed on CI. They are invariant under all seven OpenBLAS
  kernels tested (Prescott, Nehalem, Sandybridge, Haswell, Zen, SkylakeX,
  Cooperlake).

## Fix 1: reference trajectory with a derived tolerance, plus behavioural checks

The pinned digests are **not** replaced. `tests/rollout_reference.py` keeps
them in `HISTORIC_ROLLOUT_SHA256`, unchanged. They now pin a stored
reference, `tests/fixtures/scripted_rollout_reference_v1.json` (19 kB, with
zlib+base64 float64 actions).

**What the tests check:**

1. **Provenance (environment independent).** The decoded fixture bytes must
   hash to the historic digest. The generator refuses to write a fixture
   unless every rollout reproduces its digest bit for bit, and it never
   overwrites an existing file. Under Haswell it refuses.
2. **Reference match.**
   - `cut_enabled`, `cut_index` and the waypoint-index sequence (stored
     run-length encoded) must match exactly.
   - Continuous action components must satisfy |Δ| ≤ `rollout_tolerance(N)`,
     with rtol = 0.
3. **Behaviour** (`check_behaviour`):
   - Action bounds: translation is at most the waypoint step (the per-step
     limit and `speed_m_s·dt`). Sawing is bounded along and across the edge
     line. Tracked steps are bounded by the step plus the cross-track
     command. Rotation norm is at most π.
   - Waypoint progress: the index is monotone. Every passed waypoint was
     reached within `position_tolerance_m` and `rotation_tolerance_rad`, and
     holds lasted at least `hold_s`.
   - Cut flags follow the active waypoint.
   - Trajectory error:
     - path_v2 cross-track error ≤ tol on the ideal plant;
     - v1 cut steps stay within `position_tolerance_m` of the planned
       segment, ignoring the saw stroke along the edge line.
   - Termination: every waypoint is reached inside the rollout, then the
     motion is zero and cutting is off. The done step equals the
     reference's.

**Bit-identical checks kept** where they are environment independent:
- the protocol, receipt, scenario and plan hashes;
- a new same-process check that the default `WaypointPolicy` equals the
  explicit `v1` profile, bit for bit (`np.array_equal`);
- `_rollout` and `ideal_rollout` produce the same actions.

**Tolerance, defined before use:** `rollout_tolerance(N) = N × 16 × eps`,
with eps = 2.22e-16.
- Every compared quantity is at most 1 in SI units: edge |x| ≤ 0.11 m,
  translation ≤ 2 mm, rotation ≤ 0.18 rad.
- One control step has about 16 rounding-sensitive operations: 3-vector
  dot products and norms, quaternion products and rotation-vector
  conversions. Each can change by at most one ulp of a quantity ≤ 1 between
  BLAS or libm implementations.
- The controllers do not amplify these differences:
  - v1 commands `target − edge`, with unit gain, so an earlier error is
    removed rather than integrated;
  - path_v2 has cross-track gain 0.5.
- The linear worst case over N steps is therefore N·16·eps: 2.1e-12 for
  600 steps and 2.8e-12 for 800 steps.
- That is about 3·10⁶ times the largest observed difference (8.7e-19) and
  about 10⁸ times below the smallest behavioural scale (the 0.3 mm path_v2
  position tolerance).

**Negative controls** (run locally):

| Change | Result |
| --- | --- |
| +1e-10 m on one action | detected |
| `max_translation_m` 1.9 mm | detected |
| v1 plan run with `path_tracking=True` | detected (6.6e-7 m) |
| saw period 0.41 s | detected |
| corrupted fixture digest | detected |
| `position_tolerance_m` 0.45 mm | not detected: the rollout is bit-identical |
| `cross_track_gain` 0.4 | not detected: Δ = 3.5e-18, the ideal plant has no cross-track error to correct |

The cross-track gain is still pinned by the profile tests and exercised by
the laggy-plant tests in `tests/test_controller_path_v2.py`.

**Cross-kernel check.** With the new tests, `test_blademaster_demo.py`,
`test_wrist_depth_survey.py`, `test_controller_path_v2.py` and the polygon
test pass under the native, Haswell, Prescott and Sandybridge kernels
(87/87 each).

## Root cause 2: matplotlib

`tests/test_tool_profile.py::test_profile_sdf_matches_ruler_polygon_geometry`
used `matplotlib.path.Path.contains_points` as a point-in-polygon oracle.
matplotlib is in the `tools` extra, which CI does not install. It is a
plotting dependency of scripts, not something the tests need.

**Fix:**
- The oracle is now a small NumPy even-odd (crossing-number) test,
  `_points_in_polygon`. It is independent of the Warp SDF under test, and it
  agrees with `contains_points` on 10⁶ random points (5 seeds, 0
  mismatches).
- A new `test` extra (`pytest>=8`, `usd-core<26.9`, `warp-lang==1.17.0`)
  declares the CPU test environment. The workflow installs
  `-c requirements.lock -e '.[policy,test]'`, which resolves to the same
  packages as before.
- matplotlib stays in `tools`.

## Skips

All 27 skips in runs 40915870753 and 40922921407 are listed below. None is
a GPU or Isaac skip: those tests are not collected by default.

| Count | Test (line) | Reason | Legitimate? |
| --- | --- | --- | --- |
| 2 | `test_cutting_law_v5.py:128` (law-4 bitwise vs committed pre-law-5 module, `3c1f4b4`) | `git show` fails on the depth-1 checkout | **No.** A historical benchmark was not run. Fixed by `fetch-depth: 0` (`cce395e`). |
| 4 | `test_cutting_law_v6.py:132` (`ebd7395`) | same | **No.** Fixed. |
| 9 | `test_cutting_law_v7.py:145, :152` (`9f57e41`) | same | **No.** Fixed. |
| 11 | `test_cutting_law_v8.py:152, :159` (`b529929`) | same | **No.** Fixed. |
| 1 | `test_blademaster_demo_runner.py:629` (`final_v1` dry run vs recorded live receipts) | The receipts are local evidence under `artifacts/`, which is gitignored and not on the runner | **Yes.** It also skips in a clean local clone and runs in the main working tree. |

Locally, with full history, the 26 law-regression tests run and pass
(`97d56d3`: 781 passed, 1 skipped).

## Open failure: the same root cause in a Phase B pin (NOT fixed here)

`tests/test_tool_profile_v2.py:33` hashes
`json.dumps(make_profile_tool_spec(RULER_KNIFE_V1).to_dict())`.

**Mechanism.**
- `principal_axes_xyzw` comes from an eigen-decomposition of the inertia
  tensor (LAPACK through OpenBLAS). It differs by one ulp between kernels:
  - SkylakeX: `[0.7071067811865471, 0, 0, 0.7071067811865479]`;
  - Haswell: `[0.7071067811865472, 0, 0, 0.7071067811865478]`.
- Every other field is identical.

**Reproduction.** With `OPENBLAS_CORETYPE=Haswell` the test fails locally;
under the native kernel it passes.

**Status.** The file belongs to the concurrent Phase B work, and Phase A was
told not to edit it, so it is reported here rather than changed.

**Suggested fix.** Keep the bitwise digest for `to_dict()` with
`principal_axes_xyzw` (and, for safety, `principal_moments_kg_m2`) removed.
Then compare those two fields to stored values with a tolerance of a few
ulp, for example `atol = 8*eps`. Treat the quaternion up to sign. The mesh
bytes and the other digests in that test are kernel independent.

**Rule for future pins.** Do not hash floats that come from BLAS or LAPACK
(dot, norm, matmul, eig, svd). Hash only stored data or exactly computed
data. Compare computed trajectories and tensors with a derived tolerance, as
`tests/rollout_reference.py` does.

## Changed files

- `tests/rollout_reference.py`: new. Reference cases, ideal integrator,
  tolerance, comparison, behavioural checks and generator.
- `tests/fixtures/scripted_rollout_reference_v1.json`: new reference data.
- `tests/test_blademaster_demo.py`:
  - `test_waypoint_policy_default_matches_v1_reference[default|saw3mm_tilt10]`;
  - `test_waypoint_policy_default_is_bit_identical_to_v1_profile_and_estimator_is_used`.
- `tests/test_wrist_depth_survey.py`:
  `test_v1_and_path_v2_plans_and_rollouts_are_pinned`. Plan hashes are
  kept; rollouts are checked against the reference.
- `tests/test_tool_profile.py`: NumPy polygon oracle.
- `pyproject.toml`: the `test` extra.
- `.github/workflows/tests.yml`:
  - `fetch-depth: 0`;
  - `.[policy,test]`;
  - an environment record (Python, `pip freeze`, CPU, SIMD flags, OpenBLAS
    core, `numpy.show_runtime()`);
  - `pytest -q -rsx`.

No physics, thresholds, task definitions, v1 records or `tlmpm*.py` files
were changed.
