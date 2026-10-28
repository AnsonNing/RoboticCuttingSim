# WP-07 deployment validation

Status: **IN_PROGRESS**. This report is updated from preserved installation and
actual-run receipts. No clean-environment or independent cross-host PASS is
claimed before the requested runs complete.

## Scope and first actual attempt

A new public HTTPS clone at source commit
`3d6cb9f3164b0b3294f4f59d225096f28bd2231b` was created at
`/mnt/data/ningan/RoboticCuttingSim-wp07-clean-20261028`. It had no venv, dependency
checkout or local asset cache. An existing CPython3.12.12 interpreter created
new independent venvs; no old research environment's site-packages was reused.

```bash
git clone --depth 1 https://github.com/AnsonNing/RoboticCuttingSim.git \
  /mnt/data/ningan/RoboticCuttingSim-wp07-clean-20261028
cd /mnt/data/ningan/RoboticCuttingSim-wp07-clean-20261028
python3 scripts/run_logged.py artifacts/bootstrap_all_001.log -- \
  bash scripts/bootstrap.sh --runtime all \
  --python /home/ningan/.local/bin/python3.12 \
  --cache-dir "$PWD/.deps/uv-cache" --accept-eula \
  --output-dir artifacts/bootstrap_all_001
```

The base environment installed and passed its pinned metadata check. Sim's
large Torch2.11/cu128 wheel timed out at the original120s HTTP timeout; bootstrap
returned2 after229.483s. Lab source had been fetched at the exact pinned SHA,
but Lab environment installation and all physics/camera tests were **NOT_RUN**.
This is an actual failed installation, not a runtime or cutting failure. The
partial environments, all logs and bootstrap receipt are retained in that clone.

A subsequent bootstrap change increases large-wheel timeout and reduces download
concurrency, respecting explicit environment overrides. Reproduction retries
use a new clone and new venvs; only verified package download cache can be reused.
The final receipt will identify installation/validation commits separately.

## Verification boundaries

- Original host: Ubuntu22.04.5, RTX5090, unchanged driver570.211.01; cu128 recipe.
- Clean-clone complete installation: **NOT_RUN_TO_COMPLETION** at this checkpoint.
- Static / live-runtime smoke / full task reproduction: **NOT_RUN** in the fresh clone at this checkpoint.
- Independent Ubuntu/RTX second host: **NOT_RUN**; no such host has been accessed.
- Managed Python download with no discoverable3.12 interpreter: **NOT_RUN**; the actual attempt used an explicit interpreter.
- Host Kit/user shader caches may be reused on the original host. This workflow
  does not assert a cold user cache or independent machine.
- TRIM's known strict surface failure, native depth limits, visual mount
  clearance failure and provisional material parameters remain unchanged.

The final machine-readable receipt is published under `docs/evidence/` after
actual execution. Large package archives, official assets, venvs, full RGB-D and
material traces remain ignored local artifacts rather than Git content.
