# Current cutting surfaces → full workcell RTX RGB-D

Actual runs on2026-10-28 connect the experimental solver's recorded material and
robot states to programmable RTX RGB-D. This is sensor/replay validation; physical
acceptance is evaluated separately by the solver tests.

Canonical coupled source: `artifacts/coupled_robot_cut_002`. It contains81 material
snapshots and401 measured articulated robot/tool poses over8s. The camera replay
uses41 material meshes at0.2s intervals while retaining all401 robot/tool poses.
The current reference-stencil mechanical graph first splits at4.10s. Initial
material is one component containing all19,340particles; final material has two
components, including a138particle chip. No final component labels are used to
partition any earlier frame.

| Evidence | Actual result |
|---|---|
| Component/mesh contracts |4 CPU tests PASS; current code0 bridges remain connected, nonzero stress gradients count, no mesh welding across components, exact outward watertight wedge |
| `coupled_surface_002` |PASS:current-code graph and independent ParticleSurface extraction per current component; initial53,592triangles, final56,480triangles;10.77s history/extraction/export section |
| `ovrtx_coupled_002` |PASS:41 actual RTX RGB640×480×3 uint8 and depth480×640 float32 frames from both native D455 optical prims plus overview Global Camera |
| `coupled_rgbd_qa_002` |PASS:8,119 /7,203 /1,324 changed material depth pixels above10µm; chip visible in up to58 /60 /9pixels respectively |
| Mesh/depth agreement |PASS:per-frame visible-material95th-percentile error remains below0.1mm acceptance tolerance; measured worst below1µm |
| Force/oracle observations |Absent; current component labels and independent mesh-depth checks are saved only under QA `oracle_debug/` |
| Live Isaac6.2 / Lab3.0 |Still BLOCKED/not executed; standalone OVRTX rendering is a distinct supported path |

Every surface frame records source physics timestamp, current code hash, actual
reference-stencil hash, topology/surface hashes and component vertex/triangle
ranges. The exporter reconstructs the same eight-node reference interpolation by
constructing a CPU TLMPM instance without advancing it. Both positive shape
weights and nonzero stress gradients enter the mechanical connectivity audit.
Unchanged code arrays reuse cached graph results.

Newton `ParticleSurface` runs separately for each **current** mechanical
component. Meshes are concatenated without shared vertices or triangles across
components. This prevents a detached chip from becoming a welded reconstructed
surface when it approaches the substrate again. It does not alter the solver,
particle positions/masses, cut labels or contact response. The first generated
surface is also verified as one connected triangle mesh. The exporter never reads
the recorded final `component_ids` array.

The blade is the solver's actual finite triangular extrusion:local x/z vertices
`(0,0),(-.012,.004),(-.012,-.0004)`, y±.025m, origin at the cutting edge. Robot poses
and exact edge poses retain their own measured timestamps. The old box proxy and
duplicate robot replay knife are replaced/disabled in the composed replay only;
the single rendered wedge follows the recorded articulated tool pose.

Earlier `coupled_surface_001` / `ovrtx_coupled_001` are preserved intermediate
renders. Their prism positions and physical sources are the same; the exporter's
facet orientation was subsequently corrected to outward winding, with positive
signed volume1.32×10⁻⁶m³. Canonical002 was exported, actually rendered and checked
again after that correction. No physical result was invalidated by this render
orientation fix. `robot.wedge_mesh` independently has the same positive volume.

## Reproduce

Use fresh output directories. These commands consume recorded measured physics;
they do not rerun the solver or initialize Isaac Lab:

```bash
.venv/bin/python -m pytest -q tests/test_cutting_surface.py tests/test_sensor_bridge.py
.venv/bin/python scripts/export_cutting_replay.py \
  artifacts/coupled_robot_cut_002 \
  artifacts/research_handoff_001/workcell/workcell.usda artifacts/coupled_surface_fresh \
  --robot-replay artifacts/coupled_robot_cut_002/dual_franka_replay.usda \
  --samples 41 --device cuda:0
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py \
  artifacts/coupled_surface_fresh/cutting_workcell_replay.usda artifacts/coupled_rtx_fresh \
  --samples 41 --warmup 5 --gpu 0
.venv/bin/python scripts/check_cutting_rgbd.py \
  artifacts/coupled_surface_fresh artifacts/coupled_rtx_fresh artifacts/coupled_depth_qa_fresh
```

The QA computes independent CPU first-hit triangle IDs from all visible workcell
geometry. Only current material-component face ranges classify those hits. It
checks actual RTX depths against independently ray-traced material meshes, counts
changed pixels that see material in both compared frames, and verifies both
native D455 views can see detached chip surfaces. These debug masks are never
used to generate the cut, hide material, or create an observation image.

For the closeup debug view, retain both native D455 views and select the additional
camera only for `global_debug/`:

```bash
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py \
  artifacts/coupled_surface_fresh/cutting_workcell_replay.usda artifacts/coupled_closeup_fresh \
  --samples 41 --warmup 5 --gpu 0 --global-camera-path /World/CutCloseupCamera
ffmpeg -n -loglevel error -framerate 5 \
  -i artifacts/coupled_closeup_fresh/global_debug/global_%04d.png \
  -c:v libx264 -pix_fmt yuv420p artifacts/coupled_closeup_fresh/coupled_cutting_closeup.mp4
```

Completed local videos: `artifacts/ovrtx_cut_closeup_001/cut_closeup.mp4` (33KB),
`artifacts/ovrtx_recontact_closeup_001/recontact_closeup.mp4` (34KB),
`artifacts/ovrtx_coupled_001/coupled_cutting_overview.mp4` (21KB), and
`artifacts/ovrtx_coupled_closeup_001/coupled_cutting_closeup.mp4` (101KB).
These earlier diagnostic videos retain the intermediate prism facet winding;
the canonical002 arrays and geometry QA use the corrected outward version.
All files are actual RTX captures, with simple material overrides, not generated
illustrations. Media and large USD/NPZ files remain ignored; small actual PNGs
are committed under `docs/evidence/`.

The two separate kinematic physics sources were also actually reconstructed,
rendered and checked:

| Source → surface → RTX → QA | Changed material depth pixels (D4550 /D4551 /Global) | Max visible chip pixels |
|---|---|---|
| `tlmpm_cut_003` → `cut_surface_001` → `ovrtx_cut_001` → `cut_rgbd_qa_001` |10,656 /8,031 /1,548 |52 /67 /10 |
| `tlmpm_recontact_001` → `recontact_surface_001` → `ovrtx_recontact_001` → `recontact_rgbd_qa_001` |9,919 /7,960 /1,465 |53 /65 /12 |

The current depth-agreement acceptance rule was also applied to saved measured
rows of those intermediate QA runs and retained in `acceptance_check.json`; this
does not represent a new physical or render trial.

## Live adapter and remaining limits

```python
from robotic_cutting_sim.cutting_surface import ComponentSurface
surface = ComponentSurface(spacing=bed.spacing, device="cuda:0", voxel_size=.001)
vertices_m, triangles, info = surface.extract(bed)
```

`ComponentSurface` copies a new bed's reference stencil once, caches connectivity
by current code hash, and rebuilds when the bed identity changes on reset. Lower
level recorded-state interfaces are `CurrentMechanicalPartition` and
`CuttingSurface.extract(positions_m,current_codes)`. No final labels or forces are
accepted by these surface interfaces.

Density-surface reconstruction at1mm voxels is a visualization approximation,
not an exact constitutive boundary. During a partial cut that still has an uncut
ligament, the current material is one global mechanical component; sub-voxel gaps
inside that connected component can still be bridged by density reconstruction.
At recontact, independent component meshes may geometrically overlap rather than
weld. Neither case repairs or changes physical contact. Rare small disconnected
surface islands are retained and disclosed, never removed to improve appearance.

RTX capture here uses registered **ideal** depth. SingleView simulated stereo
calibration/registration is covered separately in [WP-03-next](WP-03-next.md) and
the subsequent registration audit. Actual hardware D455 calibration, live solver
↔RTX timestamp scheduling, millimeter-level hardware observability, material
calibration and complete physics convergence are separate remaining validations.
