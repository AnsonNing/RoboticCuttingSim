# WP-07 deployment validation — 2026-10-28

**Clean-environment deployment passed on the original host**, with status
`PASS_DEPLOYMENT_WITH_EXPECTED_TRIM_SURFACE_FAILURE`. A fresh public clone
installed managed Python and three isolated environments, then completed both
Isaac runtime smokes and four full CUT/TRIM episodes with independent audits.
Strict TRIM surface metrics remain **FAIL**; `full_benchmark_ready=false`.
Independent cross-host reproduction remains **NOT_RUN**.

The [JSON receipt](evidence/portable_deployment_validation.json) retains versions,
commands, source/configuration and raw-artifact hashes. Reusable setup commands
are in the [deployment workflow](PORTABLE_DEPLOYMENT.md).

## Preserved attempts

| Attempt | Installation | Subsequent actual validation |
| --- | --- | --- |
| 001 | **FAIL**, 229.483 s; base installed, large Torch download exceeded the original 120 s HTTP timeout | Lab environment and GPU tests NOT_RUN; partial environments and logs preserved |
| 002 | **PASS_INSTALL_METADATA**, 802.164 s; new venvs using an explicit existing Python 3.12.12 | Native stage loading failed at the old 300-update cap; native full cases BLOCKED. Lab full CUT passed; TRIM separated its chip with strict surface FAIL. Overall deployment validation failed. |
| 003 | **PASS_INSTALL_METADATA**, 516.067 s; another fresh clone, new managed Python and new venvs | Both smokes and all four full runtime/separation contracts PASS; CUT strict PASS / TRIM strict surface FAIL |

Attempt 001 used `3d6cb9f3164b0b3294f4f59d225096f28bd2231b` in
`/mnt/data/ningan/RoboticCuttingSim-wp07-clean-20261028`. Attempt 002 installed at
`a72cffcad5261be70bf8b7793b8e659681c4c759` and validated at
`674c463912e18de8039250ce078d8eb5bcbf2cb3` in the sibling `...-clean-002-20261028`
clone. Their `bootstrap_all_001`, `bootstrap_all_002` and second-clone
`deployment_full_001` artifacts remain preserved.

Download settings and editable-origin checks were corrected before the final
attempt. Native loading now uses actual stage readiness, a 300 s wall-clock
deadline and an event-loop yield. The final readiness loop passed in 4.483 s / 112 updates after the synchronous
stage-open call; this is not the total Kit/shader startup time. Terminal-reset
stage checks passed too. Stage readiness alone does not
certify physics or RGB-D.

## Final source and installation boundary

All six final episode manifests record clean source
`f2bf2403a90cc12e182399e80d3a91a4b80d23c3`. The final clone is
`/mnt/data/ningan/RoboticCuttingSim-wp07-clean-003-20261028`.

Installation started with `PATH=/usr/bin:/bin` and downloaded project-local
CPython 3.12.12. Its executable SHA256 is
`4304de9fdcd8465bbf4bf0814bb7abadd6d33971bab26ae38d1080c355fec983`, different
from the earlier explicit interpreter. New base, Sim and Lab venvs and a fresh
pinned Lab checkout were created. No old venv or site-packages was copied.
Verified package-download cache from attempt 001 and the verified official
shader archive were reused. Official USD files were downloaded into the new
clone: all 13 locked SHA256 checks passed, and scenes were regenerated with
new local references. No old official-asset directory or generated scene was
copied. Host Kit/Warp caches may be reused; a cold user cache is not claimed.

| Component | Measured recipe |
| --- | --- |
| Host / selected GPU | Ubuntu 22.04.5 x86_64, Linux 6.8.0-138; physical RTX 5090 GPU 0, 32,607 MiB VRAM, on the original two-card host |
| Driver / Python | Unchanged NVIDIA 570.211.01; managed CPython 3.12.12 in three distinct environments |
| Isaac Sim | 6.2.0.0 / Kit 110.4; native PhysX and camera extension 1.4.0 |
| Isaac Lab | v3.0.0 source `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`; distributions `isaaclab` 28.0.0 / `isaaclab-newton` 9.1.4 |
| Newton / Warp / MuJoCo / MuJoCo Warp | 1.6.1 / 1.17.0 / 3.12.0 / 3.12.0 |
| Torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| USD providers | Base `usd-core` 26.8; Sim bundled USD 25.11; Lab `usd-exchange` 3.0.1 |

Doctor required-metadata checks passed for all environments. Base `pip check`
passed; Sim's unused converter/OSQP dependency gaps and Lab's unused OVStage gap
remain `WARN_EXPECTED_OPTIONAL_METADATA`, with raw nonzero output preserved.
Exact versions, warning lines and lock hashes are in the receipt. Isolated
metadata probes require the same project checkout and actual installed Lab pin.

## Recorded commands and protocol

These historical commands ran from the final clone; the cache path deliberately
points to attempt 001. Choose fresh checkout/output paths for a new reproduction.

```bash
PATH=/usr/bin:/bin python3 scripts/run_logged.py artifacts/bootstrap_all_003.log -- \
  bash scripts/bootstrap.sh --runtime all --managed-python \
  --cache-dir /mnt/data/ningan/RoboticCuttingSim-wp07-clean-20261028/.deps/uv-cache \
  --accept-eula --output-dir artifacts/bootstrap_all_003

OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
python3 scripts/run_logged.py artifacts/deployment_full_003.outer.log -- \
  python3 scripts/run_deployment_validation.py --tier full --runtime both \
  --gpu auto --allow-expected-trim-geometry-fail \
  --output-dir artifacts/deployment_full_003
```

Bootstrap recorded timeout/retries/concurrent downloads of 900 s / 8 / 4.
The validator ran doctor, CPU tests, static USD, two-step smokes, four full
cases and independent camera/task audits. Every subprocess argv is retained in
`protocol.json`. Seed was **42**; full cases used 400 × 20 ms steps (8 s), the
existing 4 mm material grid and `current_volume_v1` surface profile. CUT used
1 mm particles; TRIM used the explicit 2 mm override. Native used
`native_single_view` with noise off, 4,000 actual PhysX substeps, no standalone
robot dynamics and no replay. Lab used actual scene/environment/task APIs and
NewtonManager. No physics, trajectory or Task threshold was tuned for deployment.

## Full results

| Runtime / task | Particles | Final masses after 1 → 2 components | Recorder wall time | Independent outcome |
| --- | --- | --- | --- | --- |
| Native Sim CUT | 96,000 | 72 g + 72 g | 288.292 s | Runtime/separation and strict CUT metrics **PASS** |
| Native Sim TRIM | 12,136 | 1.608 g chip + 144.024 g base | 110.975 s | Runtime/separation **PASS**; strict surface **FAIL** |
| Lab CUT | 96,000 | 72 g + 72 g | 489.048 s | Runtime/separation and strict CUT metrics **PASS** |
| Lab TRIM | 12,136 | 1.608 g chip + 144.024 g base | 198.636 s | Runtime/separation **PASS**; strict surface **FAIL** |

Every full case retained particle counts and immutable reference-mass budgets,
had zero cut-history digit rewrites, and preserved separation for 2.4 s. Both
TRIM cases detached 98.5294% of initially above-plane particle mass. Independent
source-byte checks and score recomputation passed.

| Strict TRIM surface metric | Native Sim | Lab |
| --- | --- | --- |
| p95 / RMS plane error | 1.6756 / 0.5518 mm — FAIL | 1.6727 / 0.6152 mm — FAIL |
| Worst footprint gouge / deficit | 20 mm — FAIL | 20 mm — FAIL |
| Overcut fraction of initial base volume | 2.0188% — FAIL | 2.0411% — FAIL |

All original footprint samples, missing rays and tails remain scored. The
explicit orchestration flag accepts only known surface failures after physical
invariants pass; it never relabels TRIM Task PASS. Without it, strict full
validation deliberately returns nonzero.

The complete validation including audits took **1,594.296 s**, returning 0 with
the scoped deployment status. GPU 0 device-wide samples ranged 980–6,971 MiB,
including baseline/other processes; this is not an allocator peak, isolated
performance result or hardware minimum. Recorder times include I/O/compression.
The frozen run passed **276 CPU tests**. Publication source
`156b2623edaa001cba2fc90924690b282f1f249b` separately passed **277** locally
(2.19 s) and in [CI](https://github.com/AnsonNing/RoboticCuttingSim/actions/runs/40672294114)
(5.88 s). These are separate CPU checks. All 51 recorded implementation-file
hashes in each full episode match the final publication source byte for byte.

## RGB-D, reset and remaining limits

Each full case has 41 synchronized frames per camera and measured poses at 401
control states. Actual hand-mounted `wrist_d405_0/1` are 848×480;
world-fixed close-view `global_d455` is 640×480 in Native Sim and 1280×960
in Lab. Independent FK, timestamps,
calibration, finite/range/mask and required RGB-D availability checks passed.
Wrists remain `ideal_d405_prototype`. Native Global uses actual NVIDIA
SingleView with version-specific registration; physical D455 calibration/noise
fidelity remains unvalidated. Lab Global is explicit ideal Warp RGB-D. Native
raw depth was never replaced by RTX reference depth; noise-off deployment runs
do not establish noise-model fidelity.

Minimum full-frame valid-depth fractions across the 41 captures are below;
these are availability measurements, not sensor-accuracy acceptance. Per-frame
material/knife pixels and calibration errors are retained in the JSON receipt.

| Runtime / task | Wrist 0 | Wrist 1 | Global |
| --- | --- | --- | --- |
| Native CUT | 88.59% | 91.67% | 27.05% |
| Native TRIM | 88.59% | 91.71% | 27.23% |
| Lab CUT | 85.08% | 91.87% | 32.23% |
| Lab TRIM | 85.08% | 91.87% | 32.32% |

Native reset restored all nine DOFs, body poses, material, codes and clock
exactly, but exact camera repeatability **fails**. Native CUT reset depth
maxima were 37.30 / 33.24 mm for the wrists and 760.70 mm for Global, with 212
Global mask changes. Lab restored joints/material/codes exactly; masks matched
and depth differences were ≤1.20×10⁻⁷ m. Its reset receipt does not record the
clock. Physical reset and camera repeatability remain separate measurements.
TRIM secondary-wrist knife visibility after withdrawal, native whole-frame
accuracy/holes, strict mount/flange clearance, coupling-work, material calibration
and convergence limits remain open. R570 denoiser fallback and missing optional
housing MDL warnings persist; gray fallback is not complete material fidelity.

## One visible CUDA device and cross-host boundary

```bash
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
python3 scripts/run_deployment_validation.py --tier smoke --runtime both \
  --gpu 0 --skip-cpu-tests --output-dir artifacts/single_visible_gpu_smoke_001
```

This actual additional run passed static USD and both two-step physics/RGB-D/
reset contracts: `PASS_REQUESTED_DEPLOYMENT_SCOPE`. CPU tests were explicitly
skipped because the full run already recorded 276. It exposes one CUDA device
on the existing two-card host; physical single-card reproduction is NOT_RUN.

Three same-owner Linux peers rejected batch SSH authentication. No remote
command, hardware query, install or simulation ran; global SSH configuration
was unchanged. Independent cross-host status is
**NOT_RUN_SSH_AUTHENTICATION_UNAVAILABLE**. Hardware camera calibration and a
new full D0–D3 GPU matrix were NOT_RUN; earlier evidence and CPU sampler tests
remain separate. No complete scientific benchmark or second-host PASS is claimed.

## Evidence

Local raw receipts, protocols, source snapshots, RGB-D, robot/material states,
reset and independent audits remain under final-clone
`artifacts/deployment_full_003`; their hashes are in the public JSON receipt.
Fresh native [CUT views](media/deployment_cut01/synchronized_multiview.mp4) /
[Global RGB](media/deployment_cut01/global_d455.mp4) and
[TRIM views](media/deployment_trim01/synchronized_multiview.mp4) /
[Global RGB](media/deployment_trim01/global_d455.mp4) include paired depth,
transforms and provenance. All 14 public videos passed full decoding. Large
raw artifacts, official assets, venvs and wheels remain ignored. No driver,
RSI repository or other project changed, and no policy training was performed.
