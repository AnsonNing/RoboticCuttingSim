"""Pre-declared CUT workload ``cut_v7_2_v1`` (P1-5 option (b), ``configs/rl_benchmark/cut_v7_2_v1.json``).

The workload is the WP-12 Demo 1 CUT task (CUT-01-R bar on the flat board, frozen ``sever_mid`` goal at 14 s,
``workcell_live_v1`` planner overrides) on the versioned opt-in scene ``bar_flat_board_tlmpm_v7_2_v1``, which
differs from the frozen ``configs/wp12/scenes/bar_flat_board.json`` only in ``solver.module`` (tlmpm_v5 ->
tlmpm_v7_2) and its description. Everything else is read from the declaration file, whose SHA256 pins are
verified on load; nothing here changes a frozen scene, goal, evaluator or threshold.

``workload_task_args()`` returns the ``compile_task`` arguments, so a configured-scene task family (T5.2) can
compile the same task per env slot; ``compile_workload_task()`` / ``compile_source_task()`` compile the v7_2
task and its frozen tlmpm_v5 counterpart (CPU only, no Warp).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKLOAD_PATH = ROOT / "configs" / "rl_benchmark" / "cut_v7_2_v1.json"
SCHEMA = "rcs_rl_cut_workload_v1"


class WorkloadPinError(ValueError):
    """A pinned file of the declaration differs from its SHA256 (the declaration is stale or a file changed)."""


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_workload(path=WORKLOAD_PATH, *, root=ROOT, verify=True):
    """The declaration dict; ``verify`` checks the scene, source-scene and goal SHA256 pins."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} declaration")
    if data.get("status") != "PREDECLARED":
        raise ValueError("Only a PREDECLARED CUT workload may be loaded")
    if verify:
        pins = [(data["scene"]["path"], data["scene"]["sha256"]),
                (data["scene"]["source_scene"]["path"], data["scene"]["source_scene"]["sha256"]),
                (data["goal"]["path"], data["goal"]["sha256"])]
        for rel, expected in pins:
            actual = _sha256(Path(root) / rel)
            if actual != expected:
                raise WorkloadPinError(f"{rel}: sha256 {actual} differs from the declared {expected}")
    return data


def workload_task_args(workload=None, *, root=ROOT):
    """``compile_task`` arguments of the workload (scene and goal as absolute paths)."""
    w = workload or load_workload(root=root)
    return dict(scene=str(Path(root) / w["scene"]["path"]), goal=str(Path(root) / w["goal"]["path"]),
                planner_options=dict(w["planner_options"]), seed=int(w["episode"]["sim_seed"]))


def compile_workload_task(workload=None, *, root=ROOT, particle_spacing_m=None):
    """The compiled tlmpm_v7_2 ``ConfiguredTaskSpec`` (``particle_spacing_m`` != None is a reduced-fidelity
    override for CPU tests only; the declared task uses the scene's 1 mm)."""
    from ..scene_config import compile_task
    args = workload_task_args(workload, root=root)
    return compile_task(args["scene"], args["goal"], planner_options=args["planner_options"], seed=args["seed"],
                        particle_spacing_m=particle_spacing_m)


def compile_source_task(workload=None, *, root=ROOT, particle_spacing_m=None):
    """The frozen WP-12 tlmpm_v5 counterpart (source scene, same goal and options) for equality checks."""
    from ..scene_config import compile_task
    w = workload or load_workload(root=root)
    args = workload_task_args(w, root=root)
    return compile_task(str(Path(root) / w["scene"]["source_scene"]["path"]), args["goal"],
                        planner_options=args["planner_options"], seed=args["seed"],
                        particle_spacing_m=particle_spacing_m)


def scene_differences(workload=None, *, root=ROOT):
    """Dotted keys at which the workload scene differs from the frozen source scene."""
    w = workload or load_workload(root=root)
    new = json.loads((Path(root) / w["scene"]["path"]).read_text())
    old = json.loads((Path(root) / w["scene"]["source_scene"]["path"]).read_text())
    out = []

    def walk(a, b, prefix):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key)
        elif a != b:
            out.append(prefix)
    walk(new, old, "")
    return out


def receipt(workload=None, *, root=ROOT, path=WORKLOAD_PATH):
    """Compact identity of the declaration for run manifests."""
    w = workload or load_workload(path, root=root)
    return dict(id=w["id"], schema=w["schema"], status=w["status"], declaration_sha256=_sha256(path),
                scene=w["scene"]["path"], scene_sha256=w["scene"]["sha256"], goal=w["goal"]["path"],
                goal_sha256=w["goal"]["sha256"], planner_options=w["planner_options"], task_id=w["compiled"]["task_id"],
                success_rule=w["success_rule"], physics_label=w["physics_label"])
