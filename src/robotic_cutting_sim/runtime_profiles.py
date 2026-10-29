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
PROFILE_SCHEMA = "rcs_runtime_profiles_v1"


def load_runtime_profiles(path=PROFILES_PATH):
    doc = json.loads(open(path).read())
    if doc.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"{path} is not an {PROFILE_SCHEMA} document")
    return doc


def runtime_profile(name, path=PROFILES_PATH):
    """The declared profile ``name`` (a dict with its id and the config version)."""
    doc = load_runtime_profiles(path)
    if name not in doc["profiles"]:
        raise ValueError(f"Unknown runtime profile {name!r}; declared: {sorted(doc['profiles'])}")
    return {"profile_id": name, "config_version": doc["version"], **copy.deepcopy(doc["profiles"][name])}


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
