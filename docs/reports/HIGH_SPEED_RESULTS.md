# High-speed simulation research

## Initial Rubber timing

The fresh Reference run on physical GPU 1 used RTX 5090, driver 570.211.01,
Warp 1.17.0, Torch 2.11.0+cu128 and one OpenBLAS thread. It held the GPU1
lock. Compute-PID samples showed no other compute jobs.

| Profile | N | Grid | Substeps per 20 ms | Warm batch wall | Aggregate env-steps/s | Sampled process MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rubber_rl_env_v2`, full slots, CUDA Graph | 8 | 1 mm | 5226 | 8.371 s | 0.956 | 1228 |

The five timed steps ran during the seeded preload and were finite with zero
invalid bits. This row gives early throughput feedback. Its scratch wrapper
then failed when the reduced profile rejected full-slot geometry. The quick
result and the failure remain recorded. The longer runs time 50 steps after
the 10-step preload. They record full-slot Reference separately and compare
Reference and the existing reduced profile with identical alternating
`L100_t4` and `L150_t4` slots and seeds.

See the [quick baseline receipt](../evidence/high_speed/rubber_quick_baseline.json).

```bash
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/perf/benchmark_rubber.py \
    --slots paired --num-envs 8 --gpu-index 1 --out artifacts/rubber_fresh_pair
```

Add `--resolution rubber_rl_v1` for the existing reduced profile, or use
`--slots full` for the full-slot Reference. The runner records synchronized
step walls, cold build/reset timing, table contact, invalid bits and GPU
compute co-tenants. Each invocation requires a fresh output directory.

## Timestep bucketing estimate

The Reference wave-speed rule applied to 8,192 reset seeds gives 4,030
median per-environment substeps versus the fixed 5,226. Random groups of
eight average 4,755.88 substeps at the within-group maximum, a 9.0% reduction.
Perfectly sorted groups of eight average 4,005.29, a 23.4% reduction. These
are analytical estimates before deformation, independently timed resets,
and graph scheduling. Bucketing stays outside the first implementation.
See the [step-rule analysis](../evidence/high_speed/rubber_step_rule_distribution.json).

## Full-slot Reference drag

50 synchronized steps after the 10-step seeded preload. All slots recorded table
contact in all 50 steps and zero invalid bits. The controller applies a fixed
open-loop drag and is used for contact timing.

| N | Batch step wall | Aggregate env-steps/s | Aggregate sim s/wall s | 60-step run wall | Sampled process MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.518 s | 0.659 | 0.01317 | 92.96 s | 812 |
| 8 | 8.297 s | 0.964 | 0.01928 | 501.82 s | 1228 |

See the [contact baseline receipts](../evidence/high_speed/rubber_contact_baseline.json).
