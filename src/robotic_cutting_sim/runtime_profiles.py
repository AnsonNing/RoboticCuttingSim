"""Versioned, opt-in runtime profiles of the policy env (WP-11 Phase H4).

``configs/wp11/runtime_profiles_v1.json`` declares ``full_physics_v1`` (the formal
resolution; the only profile whose episodes can be acceptance-valid) and
``research_fast_v1`` (F3 'coarse' numerics, fewer cameras and less recording;
every episode ``acceptance_valid: false``). ``open_cutting_env(...,
preset="cutting_research_v1", runtime_profile=...)`` applies one; without
``runtime_profile`` nothing changes.
"""
import copy
import json

from .recording import ROOT

PROFILES_PATH = ROOT/"configs/wp11/runtime_profiles_v1.json"
HIGH_SPEED_PROFILES_PATH = ROOT/"configs/high_speed/cutting_runtime_profiles_v1.json"
PROFILE_SCHEMA = "rcs_runtime_profiles_v1"
FIDELITIES = ("reference", "turbo")
NATIVE_50HZ_RECORDING = {
    "camera_snapshot": True,
    "steady_render_passes": 1,
    "async_camera_write": True,
    "buffer_grasp_trace": True,
    "stream_physics": True,
}
_TURBO_BY_BACKEND = {
    "kinematic": "research_fast_4mm_v1",
    "kinematic_ideal_tracking_batched": "research_fast_4mm_v1",
    "isaac_sim": "research_fast_4mm_v1",
    "kinematic_grip_batched": "rubber_turbo_v1",
}


def fidelity_profile(fidelity, backend):
    """Resolve the public fidelity selection to its existing backend-specific profile."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"Unknown fidelity {fidelity!r}; choose reference or turbo")
    if fidelity == "reference":
        return None
    if backend not in _TURBO_BY_BACKEND:
        raise NotImplementedError(f"fidelity='turbo' is unavailable for {backend!r}")
    return _TURBO_BY_BACKEND[backend]


def native_50hz_cadence(camera_every=None, record_every=None):
    if any(value is not None and (type(value) is not int or value != 1)
           for value in (camera_every, record_every)):
        raise ValueError("Native Vision fidelity requires camera_every=1 and record_every=1")
    return 1, 1


def load_runtime_profiles(path=PROFILES_PATH):
    doc = json.loads(open(path).read())
    if doc.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"{path} is not an {PROFILE_SCHEMA} document")
    return doc


def runtime_profile(name, path=PROFILES_PATH):
    """The declared profile ``name`` (a dict with its id and the config version)."""
    paths = (PROFILES_PATH, HIGH_SPEED_PROFILES_PATH) if path == PROFILES_PATH else (path,)
    declared = set()
    for candidate in paths:
        doc = load_runtime_profiles(candidate)
        if name in doc["profiles"]:
            return {"profile_id": name, "config_version": doc["version"], **copy.deepcopy(doc["profiles"][name])}
        declared.update(doc["profiles"])
    raise ValueError(f"Unknown runtime profile {name!r}; declared: {sorted(declared)}")


def profile_scene(scene, profile):
    """The scene config with the profile's solver override written into it (recorded in the task id)."""
    from .scene_config import resolve_scene
    scene = copy.deepcopy(resolve_scene(scene))
    override = profile.get("solver_override") or {}
    for key, value in override.items():
        if key not in ("particle_spacing_m", "grid_dx_m"):
            raise ValueError(f"Unsupported solver override {key!r}")
        scene["solver"][key] = float(value)
    return scene


def check_caller_overrides(profile, *, particle_spacing_m=None, backend_options=None):
    """Refuse caller numerics that the profile fixes (an acceptance profile is not silently changed)."""
    given = {"particle_spacing_m": particle_spacing_m, **(backend_options or {})}
    bad = sorted(key for key in profile.get("forbid_caller_overrides", ()) if given.get(key) is not None)
    if bad:
        raise ValueError(f"runtime_profile {profile['profile_id']!r} fixes {bad}; do not pass them as well")


def check_task(profile, task, material_dt):
    """Verify the compiled task and material dt meet the profile's declared resolution."""
    require = profile.get("require") or {}
    actual = {"particle_spacing_m": float(task.particle_spacing_m), "grid_dx_m": float(task.grid_dx_m),
              "control_dt_s": float(task.control_dt_s), "material_dt_s": float(material_dt)}
    wrong = {key: {"required": value, "actual": actual[key]} for key, value in require.items()
             if abs(actual[key]-float(value)) > 1e-12}
    if wrong:
        raise ValueError(f"runtime_profile {profile['profile_id']!r} requires {wrong}")
    return actual


def profile_receipt(profile, resolution):
    """What every reset info / evaluation / dataset episode records."""
    return {"profile_id": profile["profile_id"], "config_version": profile["config_version"],
            "acceptance_valid": bool(profile["acceptance_valid"]), "resolution": resolution,
            "native": profile.get("native") or {}, "preset_overrides": profile.get("preset_overrides") or {},
            "recording": profile.get("recording") or {}, "history_stride": profile.get("history_stride"),
            "fidelity_limits": profile.get("fidelity_limits")}


def batch_material_dt(name, particle_spacing_m, material_dt_s):
    """Resolve the numerical portion of a runtime profile for the headless vector API."""
    profile = runtime_profile(name)
    check_caller_overrides(profile, particle_spacing_m=particle_spacing_m)
    dt = float(profile["material_dt_s"])
    if material_dt_s not in (0.0001, dt):
        raise ValueError(f"runtime_profile {name!r} fixes material_dt_s to {dt:g}")
    return dt


def compile_batch_task(scene, goal, *, name, material_dt_s, **options):
    """Compile with the profile's scene numerics. Camera and recorder settings are Native-only."""
    from .scene_config import compile_task
    profile = runtime_profile(name)
    task = compile_task(profile_scene(scene, profile), goal, **options)
    check_task(profile, task, material_dt_s)
    return task


def compile_profiled_trim_seed(scene_seed, *, name, policy_seed, sim_seed, distribution, output_dir):
    """Compile an opt-in Native TRIM task using the frozen seeded scene and goal."""
    from pathlib import Path

    from .geometry_providers import lattice_report
    from .scene_config import (SceneInfeasible, canonical_json, compile_task,
                               configured_task_payload, sha256_json)
    from .seeded_trim import compile_seed, digest_array

    reference, manifest = compile_seed(
        scene_seed, policy_seed=policy_seed, sim_seed=sim_seed, distribution=distribution)
    profile = runtime_profile(name)
    material_dt_s = float(profile["material_dt_s"])
    scene = profile_scene(manifest["sampled_scene"], profile)
    task = compile_task(scene, reference.goal, seed=int(sim_seed))
    resolution = check_task(profile, task, material_dt_s)
    points, masses, lattice = task.sample_particles()
    connectivity = lattice_report(lattice)
    if not connectivity["ok"]:
        raise SceneInfeasible("initial_connectivity", canonical_json(connectivity))
    height_above_base = points[:, 2] - task.geometry.base_height_m
    counts = [
        int(((height_above_base > 0) & (
            bump.height(points[:, 0], points[:, 1]) > height_above_base)).sum())
        for bump in task.geometry.bumps
    ]
    if min(counts) < 20:
        raise SceneInfeasible("unresolved_protrusion", str(counts))
    manifest.update(
        runtime_profile=name, runtime_resolution=resolution,
        acceptance_valid=bool(profile["acceptance_valid"]),
        particle_count=len(points), target_particle_counts=counts,
        initial_lattice_connectivity=connectivity,
        scene_hash=sha256_json(scene), solver_hash=sha256_json(scene["solver"]),
        task_hash=task.config_sha256, initial_particles_sha256=digest_array(points),
        initial_masses_sha256=digest_array(masses), initial_lattice_sha256=digest_array(lattice),
        sampled_scene=scene,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for label, data in (
        ("scene", scene), ("goal", task.goal), ("scene_manifest", manifest),
        ("compiled_task", json.loads(configured_task_payload(task))),
    ):
        (out / f"{label}.json").write_text(json.dumps(data, indent=2) + "\n")
    return task, manifest
