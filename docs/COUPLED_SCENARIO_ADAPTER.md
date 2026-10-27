# Coupled scenario recipes and explicit cut activation

The physics adapter now accepts an optional `BenchmarkSpec` recipe without
changing the frozen default D0 experiment. The original no-recipe D0/D1 route
uses `Specimen` and the original TLMPM defaults, including friction 0.3.
No-recipe D2/D3 automatically call `BenchmarkSpec.generate` and `sample_specimen`.
An explicit recipe can also supply D0–D3:

```python
from robotic_cutting_sim.benchmark import BenchmarkSpec
from robotic_cutting_sim.coupled_robot import CoupledCutting, CouplingConfig

spec = BenchmarkSpec.generate(seed=42, variant="D3")
scene = CoupledCutting(CouplingConfig(variant="D3", benchmark_spec=spec))
proprio, setup = scene.reset(seed=42)
print(setup["parameter_receipt"])
```

The reset seed must match a supplied recipe's seed. Supplying a recipe means
supported material values are deliberately applied, including its friction.
For example an explicit default D0 `BenchmarkSpec` requests friction 0.4; this
is a different experiment from the preserved no-recipe D0 friction 0.3 case.
Receipts identify the scenario selection, sampled continuous material, applied
backend parameters, unsupported fields and fields deferred to the sensor adapter.

| Recipe field | Adapter behavior |
|---|---|
| Density, Young modulus, Poisson ratio, yield stress | Applied to material sampling and TLMPM constitutive parameters |
| Friction | Applied to TLMPM tool/material contact friction |
| Optional fracture energy | Applied to the experimental edge-work cutting-energy parameter |
| Hardening ratio | UNSUPPORTED; TLMPM remains perfect Hencky J2 plasticity |
| Critical stress | UNSUPPORTED; it is not the active failure criterion |
| Recipe failure-model name | NOT_APPLIED; actual experimental logical-grid cut law is identified explicitly |
| Rectangular blade half thickness | UNSUPPORTED; no exact mapping to the matched triangular wedge |
| Camera translation/rotation | Deferred to sensor adapter; never claimed applied by physics |

`setup["parameter_receipt"]`, `scene.parameter_receipt`,
`scene.sampling_receipt` and `scene.benchmark_spec` expose these boundaries.
Randomization of a recipe field does not establish its backend application.
No recipe parameter is relabeled calibrated.

`step` also accepts `cut_index`:

```python
state, debug = scene.step(world_edge_pose, cut_index=0)   # active outside script window
state, debug = scene.step(world_edge_pose, cut_index=-1)  # no new cut assignment
state, debug = scene.step(world_edge_pose)               # original reference window
```

`cutting=False` disables new assignment even with `cut_index=0`. This minimal
adapter retains one cut digit; `cut_index=1` is rejected. It does not promise
arbitrary multiple cutting planes using one retained digit. Forces and material
state remain outside returned proprioception.

The probe supports `--variant D0|D1|D2|D3` and `--benchmark-recipe`. A short smoke
can be reproduced as follows; it checks backend state, not successful cutting:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/run_coupled_cutting.py --variant D3 --duration .2 --output-dir artifacts/coupled_D3_short_fresh
.venv/bin/python -m pytest tests/test_robot_contract.py tests/test_coupled_robot_contract.py -q
```

Actual 10-step, 0.2 s GPU state checks are preserved at
`artifacts/coupled_benchmark_adapter_001` and summarized in
[coupled_scenario_adapter_validation.json](evidence/coupled_scenario_adapter_validation.json):

- D2 seed 42: 19,386 particles, finite evolving state, minimum det(F)=0.998175.
- D3 seed 42: 19,303 particles, finite evolving state, minimum det(F)=0.998935.
- Both exercised explicit cut-index enable/disable. The knife is still above
  material in this short check, so it establishes no cutting success.
- Nineteen CPU contract tests passed, including supported/unapplied parameter
  mapping and unchanged default D0 initial arrays/material settings.

The source hashes for these short API checks were collected after execution and
are labeled accordingly. The separate frozen 8 s coupled D0 results and their
launch-time source receipts remain immutable.
