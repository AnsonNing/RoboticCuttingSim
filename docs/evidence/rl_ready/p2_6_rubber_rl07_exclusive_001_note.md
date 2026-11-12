# Supplementary exclusive-GPU RL-07 throughput (p2_6_rubber_rl07_exclusive_001)

**Supplementary measurement for Phase 2 acceptance review Z1 / M2.** It re-scores nothing. The formal RL-07 row stays
`gpu_rl_b_001`, labelled SHARED_GPU, verdict PASS.

**Run.**

- RL-07 runner and settings exactly as declared (`run_rubber_rl_gpu.py --rows RL07`, protocol `rubber_rl_tests_v2`).
- Commit `c21ae8d`, clean tree.
- Hardware: RTX 5090 GPU1, bus F1:00.0, driver 570.211.01, under a blocking flock on
  `rcs_rl_runner/gpu1.lock`, with `OPENBLAS_NUM_THREADS=1`.
- Window: 15:17:38–15:26:45 KST, 2026-11-12.
- Output: `artifacts/rl_ready/p2_6/gpu_rl07_exclusive_001`.

**Label: EXCLUSIVE_GPU.**

- Before the run, GPU1 had no compute process.
- A continuous sampler queried nvidia-smi every 2 s: 275 samples, maximum gap 2.0 s. No sample showed a foreign
  process on GPU1. The only processes seen were the runner and its own Phase 1 workload-A children, identified by
  process ancestry.

**Settings common to every row.** 50 Hz control; dt 3.827 µs with 5,226 substeps (3,920 for `rubber_rl_v1`); no
cameras; no recording. Each row runs 1 cold step and 5 warm steps; env-steps/s is N / median(warm step wall).

| Set | N | env-steps/s exclusive | env-steps/s shared (formal) |
| --- | --- | --- | --- |
| full slot set | 1 | 0.648 | 0.658 |
| full slot set | 8 | 0.942 | 0.960 |
| uniform L125_t4 | 1 | 0.624 | 0.624 |
| uniform L125_t4 | 8 | 1.144 | 1.160 |
| `rubber_rl_v1` (reduced fidelity, n_t 3) | 1 | 1.195 | 1.221 |
| `rubber_rl_v1` (reduced fidelity, n_t 3) | 8 | 3.236 | 3.277 |

The other N 4 / N 16 rows, the VRAM figures and the cost arms are in the JSON.

**N8 / N1 ratio.** The ratio uses matched windows: the same 5 warm steps at each N, as median and as mean.

| Set | Exclusive | Shared |
| --- | --- | --- |
| full slot set | **1.454** (mean 1.455) | 1.459 |
| uniform L125_t4 | 1.835 | 1.860 |
| `rubber_rl_v1` | 2.709 | 2.684 |

**Reading.**

- Exclusive and shared numbers agree within about 2 %. The formal RL-07 row was not materially inflated or deflated
  by its co-tenant.
- VRAM at N 8 is 1,228 MiB, the same as in the formal row.
- Phase 1 workload A, run as children of this process: 7.06 env-steps/s at N 1 and 10.00 at N 8. Those child
  processes label themselves SHARED_GPU only because their parent (this process) holds memory on the same GPU.
