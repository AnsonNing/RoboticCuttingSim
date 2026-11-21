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
