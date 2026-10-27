# CRESSim follow-up: corrected math and the remaining cutting boundary

2026-10-28, Asia/Seoul. **Numerical stability improved; persistent physical cutting remains unsupported.**

The pinned CRESSim source is
[`09aa5009b8580351f516b6df7660e87821fc5eb6`](https://github.com/yafei-ou/CRESSim-MPM/tree/09aa5009b8580351f516b6df7660e87821fc5eb6).
This follow-up adds a three-part matrix-layout correction after the existing
four-part SI/MLS correction. It runs in a fresh source archive under
`artifacts/CRESSim_candidate/`; `.deps/CRESSim-MPM` and `.deps/cressim-build`
were not changed. This candidate is not stock upstream or a plasticine model.

## Reproduce

CPU diagnostics need a C++17 compiler, `git`, `patch`, and the pinned source
checkout. GPU probes also need the project Python environment, CUDA toolkit,
CMake, `nvidia-smi`, matplotlib and ffmpeg. Use a new output directory every time.

```bash
.venv/bin/python scripts/verify_cressim_candidate.py \
  --output-dir artifacts/CRESSim_candidate/cpu_fresh

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MPLBACKEND=Agg \
  .venv/bin/python scripts/verify_cressim_candidate.py \
  --output-dir artifacts/CRESSim_candidate/gpu_fresh \
  --gpu --nvcc /usr/local/cuda-12.8/bin/nvcc --cuda-arch 120
```

The CUDA path above is the toolkit used on the audited host; pass the local
toolkit path elsewhere. No package, driver or operating-system changes occur.
The script archives the pinned commit, preserves its license headers, applies
patches using `patch(1)`, compiles the actual internal headers, retains expected
baseline failures, and independently classifies optional GPU sanity probes.
Its final PASS covers these numerical controls only. `persistent_cutting_status`
remains `UNSUPPORTED` regardless of the process exit code.

## Actual-header root-cause controls

`scripts/cressim_low_level_verify.cpp` calls CRESSim's actual functions. It does
not reproduce a replacement solver in Python. The new patch is
[`cressim_math_layout.patch`](../../scripts/patches/cressim_math_layout.patch).

| Defect | Controlled finding before correction | Additional correction / finding afterward |
|---|---|---|
| `Vec3::outerProduct` passes rows into a column-major constructor | `a⊗b` maximum error 6; affine G2P returns the transposed gradient, error 1.900005 s⁻¹ on the four-patch candidate | Pass constructor columns in column order; outer-product error 0 and affine-gradient error 7.7784e-6 s⁻¹ |
| `svd3` wrapper passes column-major A to a scalar routine expecting row order | Reconstruction returns Aᵀ; reconstruction error 0.2000004; a pure rigid rotation creates 26,971.5 Pa stress | Correct input argument order; reconstruction error 2.3842e-7 and rigid-rotation stress error 0.002654 Pa |
| `Mat3x3f::inverse` assembles inverse rows as columns | `M * inverse(M)` identity error 0.2 | Transpose the assembled adjugate; identity error 0 |

The inverse defect affects code paths using `inverse`, including PB-MPM's
constitutive correction. MLS-MPM does not call this inverse in the tested path;
it is not attributed as a cause of its settling instability. The outer-product
and SVD defects directly affect MLS velocity gradients and co-rotational stress.
Individual GPU ablations were not run; the stable GPU result belongs to the
combined, analytically tested correction set.

All 16 host controls passed after the two patches: outer product, inverse,
SVD reconstruction and orthogonality, rigid-rotation zero stress, analytic
simple-shear stress, affine velocity/gradient reproduction, 100-step freefall
position/velocity, transfer mass, Coulomb plane contact, particle projection,
moving-plane relative contact, current-slicer crossing and withdrawal semantics.
The raw upstream version failed eight controls; the earlier four-part candidate
failed six. Baseline failures remain in the evidence rather than being skipped.

Additional numerical results: simple-shear stress error 0.009949 Pa; freefall
velocity error 5.2154e-8 m/s and position error 3.7253e-9 m; P2G mass error
5.4003e-8 relative. Tolerances are in the JSONL and diagnostic source.

## Actual GPU experiments

Two fresh 2 s settling/scrape pairs were run on physical GPU0, RTX5090,
driver 570.211.01, CUDA12.8, GCC11.4. Configuration: seed42, connected D0 specimen,
19,340 particles, density1500 kg/m³, particle spacing2 mm, grid4 mm,
frame dt1/60 s, solver dt1/6000 s, 100 fine blade updates per frame, 120 frames.
Material is uncalibrated co-rotational elasticity, E50,000 Pa, ν0.3; no plasticity.
The existing headless native adapter and its reference blade paths were used
unchanged. This is not the unchanged GUI `example_cutting` executable.

| Fresh run | Settling frame-sampled peak speed | Scrape frame-sampled peak speed | Minimum particle Z, settle / scrape | Native wall time, settle / scrape |
|---|---:|---:|---:|---:|
| `verification_001` | 0.02427345 m/s | 0.03343749 m/s | 0.970842 / 0.970572 mm | 1.866 / 1.880 s |
| `verification_002` | 0.02427346 m/s | 0.03397112 m/s | 0.970842 / 0.970572 mm | 1.936 / 1.994 s |

Both runs passed the existing broad engineering guards: finite exported states,
frame-sampled maximum speed below2 m/s, particle Z at least−1 mm, and XY within
±150 mm. These guards screen numerical failures; they do not establish calibrated
material behavior, absence of all substep penetration, fracture, or convergence.
The previously retained four-part candidate failed supported settling at
3.307 m/s and scraping at5.685 m/s. These are corrections of a defective candidate,
not a controlled performance comparison to Newton.

Mass error relative to the float64 input mass was 3.5371e-8 in every run; particle
count/mass were not changed. The second pair sampled GPU memory every100 ms:
peak967 MiB on the entire selected physical device, including display/baseline.
It is not a solver-only allocation measurement. Energy diagnostics were NOT_RUN.
Kinematic reaction forces remain unavailable and are recorded as NaN trace fields.

Same-seed repeats were **not bitwise deterministic**. Across all121 exported frames,
settling position/velocity maximum differences were6.4261e-8 m /2.1587e-6 m/s;
scraping differences were1.0160e-5 m /0.0134506 m/s. Mass and commanded blade poses
were identical. CUDA atomic summation and contact sensitivity are plausible causes;
they were not independently isolated. No formal tolerance-based reproducibility
gate has been frozen or passed.

## Why this is not persistent cutting

The actual-header withdrawal control places two points on opposite sides of a
QuadSlicer, calls `checkOnDifferentSides`, moves the slicer outside their support,
and calls it again: **true → false**. The function takes current shape poses and
geometry only; no particle cut-history argument exists. P2G rejects a currently
crossing transfer and G2P uses a reflected ghost velocity for the current slicer.
After withdrawal, the shared grid once again accepts that transfer.

This proves memoryless transfer visibility at this function boundary. It does
not claim a full continuum withdrawal experiment measured re-welding. Persistent
independent fields, failure history, chip labels and recontact were not implemented
by the layout patch. The G2P reflected velocity also uses world particle velocity,
without the blade-relative normal velocity; the successful moving-plane contact
control exercises regular rigid contact, not this separate slicer ghost rule.

| Requested physical test | Candidate status |
|---|---|
| T1 supported settling | PASS for the stated 2 s numerical guards |
| T2 plastic indentation/unload | UNSUPPORTED by this elastic material path |
| T3 permanent separation, T4 withdrawal, T5 chip recontact | UNSUPPORTED; no cut history/multi-field dynamics |
| T6 second/different-direction cut | NOT_RUN |
| T7 grid/time refinement | NOT_RUN |
| T8 deterministic reset | Bitwise repeat FAIL; tolerance-based formal acceptance NOT_RUN |
| T9 mass / energy | Global particle mass accounting PASS; chip budget and energy NOT_RUN |
| T10 true-cut surface consistency | NOT_RUN; no verified true-cut material state |

The corrected candidate is useful for studying stationary-slicer contact and
deformation. It remains unsuitable as the final persistent-cutting backend without
a validated persistent multi-field or equivalent discontinuity method. Further
parameter tuning cannot add the missing state or degrees of freedom.

## Evidence and preservation

Committed compact receipt:
[`cressim_followup_evidence.json`](cressim_followup_evidence.json), including all
baseline/corrected CPU results, GPU metrics, repeat differences, patch hashes and
raw artifact hashes. GPU-run patch SHA256:
`bbb5c1fc1ef4e3f92b9a20ffa16bb7e61e2d7d81215d0655ded5744df6a81b83`.
Whitespace-only context formatting was then normalized for repository checks;
current patch SHA256 is
`1388b9011d305afe3249c976c91e2a45e7fff9295bba327d69dc8b938ab48e28`.
Fresh CPU controls in `verification_003_cpu` passed and all five modified source
files were byte-identical to the GPU candidate. The originally executed patch
is preserved alongside each GPU run.

Raw local evidence is retained under `artifacts/CRESSim_candidate/verification_001`
and `verification_002`: pinned source tar, corrected source, build/commands/logs,
JSONL diagnostics, native adapter, per-case NPZ particle positions/velocities/mass,
blade trajectory, CSV trace, PNG frames, MP4 diagnostic video and memory CSV.
`repeat_comparison_001.json` records the repeat check. Existing failed CRESSim
experiments remain untouched. Large evidence and third-party binaries are ignored
by Git; they are not redistributed in the public repository.

The patch retains upstream BSD notices through the source archive. `svd3.h`
contains additional original BSD/Apache-2.0 notices; those remain in the archive.
Sources: [official repository](https://github.com/yafei-ou/CRESSim-MPM),
[pinned MLS implementation](https://github.com/yafei-ou/CRESSim-MPM/blob/09aa5009b8580351f516b6df7660e87821fc5eb6/src/solver/mls_mpm_algorithm.h),
[pinned contact implementation](https://github.com/yafei-ou/CRESSim-MPM/blob/09aa5009b8580351f516b6df7660e87821fc5eb6/src/solver/mpm_rigid_contact.h).
