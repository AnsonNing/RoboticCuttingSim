"""WP-12 Phase 3 Layer 6: live surface_profile selection (surface_runtime.py), recording and leak checks.

CPU only (Warp "cpu", 4 mm particles, material_dt 0.5 ms) on the real kinematic backend with the configured
blade_master_flat_board_v1 bar (tlmpm_v5, law 8): one through-cut driven along the compiled reference path. The
Native Isaac Sim path is exercised here only through the runner's surface method with a real in-memory USD stage
(no Kit launch); the live Native runs are in docs/reports/WP12_SURFACE_LIVE.md.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest

from robotic_cutting_sim import surface_runtime as sr

SCENE = "configs/wp12/scenes/bar_flat_board.json"
GOAL = "sever_mid"
SURFACE_KEYS = frozenset({"surface", "surface_provenance", "hermite", "cut_events", "codes_sha256", "audit",
                          "surface_selection", "face_varying_normals_world"})


def walk_keys(tree, keys, path=""):
    found = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            if key in keys:
                found.append(f"{path}/{key}")
            found += walk_keys(value, keys, f"{path}/{key}")
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            found += walk_keys(value, keys, f"{path}[{i}]")
    return found


# ----------------------------------------------------------------------------- configuration (no simulation)

def test_kinematic_config_default_is_unchanged_and_cut_aware_turns_the_recorder_on():
    from robotic_cutting_sim.kinematic_env import KinematicConfig, KinematicCuttingBackend
    default = KinematicCuttingBackend(KinematicConfig())
    assert "surface_profile" not in default.receipt()["config"] and "cut_events" not in default.receipt()["config"]
    assert default.surface_selection() is None and default.surface_provenance() is None
    assert default.surface_frame() is None
    with pytest.raises(ValueError, match="tlmpm_v5"):
        KinematicConfig(surface_profile="cut_aware_surface_v1")
    with pytest.raises(ValueError, match="surface_profile must be one of"):
        KinematicConfig(surface_profile="cut_aware")
    config = KinematicConfig(solver="tlmpm_v5", cutting_law=8, surface_profile="cut_aware_surface_v1")
    assert config.cut_events == "cut_events_v1"
    receipt = KinematicCuttingBackend(config).receipt()["config"]
    assert receipt["surface_profile"] == "cut_aware_surface_v1" and receipt["cut_events"] == "cut_events_v1"


def test_isaac_config_particle_surface_alias_records_the_same_config_and_cut_aware_needs_v5():
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
    from robotic_cutting_sim.scene_config import compile_task, configured_task_payload
    assert asdict(IsaacSimCuttingConfig(surface_profile="particle_surface_v1")) == asdict(IsaacSimCuttingConfig())
    assert IsaacSimCuttingConfig().surface_profile == "legacy"
    with pytest.raises(ValueError, match="configured tlmpm_v5"):
        IsaacSimCuttingConfig(surface_profile="cut_aware_surface_v1")
    with pytest.raises(ValueError, match="surface_profile must be"):
        IsaacSimCuttingConfig(surface_profile="cut_aware")

    def configured(task, **extra):
        return IsaacSimCuttingConfig(task_id=task.task_id, configured_task=configured_task_payload(task),
                                     tool_attachment_mode="physical_grasp", max_cuts=task.max_cuts,
                                     camera_mode="native_single_view", camera_layout="wrist_d405_global_d455", **extra)
    v5 = compile_task(SCENE, GOAL)
    default = configured(v5)
    assert default.surface_profile == "current_volume_v1"
    assert asdict(configured(v5, surface_profile="particle_surface_v1")) == asdict(default)
    assert configured(v5, surface_profile="cut_aware_surface_v1").surface_profile == "cut_aware_surface_v1"
    v4 = compile_task("configs/wp11/scenes/bar_160x30x20.json", GOAL)
    with pytest.raises(ValueError, match="configured tlmpm_v5"):
        configured(v4, surface_profile="cut_aware_surface_v1")


def test_run_wp10_live_surface_flags_are_absent_unless_given():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rcs_run_wp10_live_flags", root/"scripts/run_wp10_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import argparse
    seen = {}

    def parse(argv):
        original = argparse.ArgumentParser.parse_args

        def capture(self, args=None, namespace=None):
            seen["a"] = original(self, args, namespace)
            raise SystemExit(0)
        argparse.ArgumentParser.parse_args = capture
        try:
            with pytest.raises(SystemExit):
                module.main(argv)
        finally:
            argparse.ArgumentParser.parse_args = original
        return vars(seen["a"])
    base = ["--runtime", "isaac_sim", "--out", "unused"]
    assert "surface_profile" not in parse(base) and "surface_timing_log" not in parse(base)
    given = parse(base+["--surface-profile", "cut_aware_surface_v1", "--surface-timing-log"])
    assert given["surface_profile"] == "cut_aware_surface_v1" and given["surface_timing_log"] is True


# ----------------------------------------------------------------------------- kinematic CPU episodes

def _drive_cut(env, task):
    """Teleport above the bar (cutting off), then follow the compiled CUT reference with cut 0 armed."""
    times = np.asarray(task._times())
    labels = task.reference_phase_labels
    start = max(t for t, label in zip(times, labels) if label.startswith("PRE_CUT"))
    end = max(t for t, label in zip(times, labels) if label.startswith("CUT"))
    env.step({"mode": "cartesian_edge_pose_world", "target": task.reference_edge_pose_world(start), "cut_enabled": False})
    k = 1
    while start+k*task.control_dt_s <= end+.1:
        env.step({"mode": "cartesian_edge_pose_world", "target": task.reference_edge_pose_world(start+k*task.control_dt_s),
                  "cut_enabled": True, "cut_index": 0})
        k += 1


@pytest.fixture(scope="module")
def episodes():
    """The same through-cut with particle_surface_v1 (A, recorder off) and cut_aware_surface_v1 (B, recorder on)."""
    pytest.importorskip("warp")
    from robotic_cutting_sim.scene_config import compile_task, make_configured_kinematic_env
    task = compile_task(SCENE, GOAL, particle_spacing_m=.004)
    out = {}
    for profile in (None, sr.CUT_AWARE):
        env = make_configured_kinematic_env(task, material_dt=.0005, device="cpu", surface_profile=profile)
        env.reset(seed=42)
        initial = None
        if profile is not None:
            vertices, faces = env.backend.surface
            initial = (vertices.copy(), faces.copy(), dict(env.backend.cut_aware_surface.rows[-1]))
        _drive_cut(env, task)
        out[profile] = (env, task, initial)
    yield out
    for env, _, _ in out.values():
        env.close()


@pytest.mark.cpu_sim
def test_cut_aware_surface_does_not_change_physics_bitwise_on_cpu(episodes):
    a, b = episodes[None][0].backend, episodes[sr.CUT_AWARE][0].backend
    assert a.material.cut_event_recorder is None and b.material.cut_event_recorder is not None
    state_a, state_b = a.material.debug_state(), b.material.debug_state()
    for key in state_a:
        assert state_a[key].tobytes() == state_b[key].tobytes(), key
    assert np.count_nonzero(state_b["codes"]) > 0
    # the surface reads only: extracting it again leaves the solver state untouched
    b._surface_time = None
    b._extract_surface()
    for key, value in b.material.debug_state().items():
        assert value.tobytes() == state_b[key].tobytes(), key


@pytest.mark.cpu_sim
def test_cut_aware_live_surface_uses_recorder_hermite_and_world_frame(episodes):
    env, task, _ = episodes[sr.CUT_AWARE]
    backend = env.backend
    vertices, faces = backend.surface
    info = backend.surface_metadata
    assert info["mechanical_components"] == 2 and info["audit"]["ok"]
    h = info["hermite"]
    assert h["source"] == "cut_events_v1" and h["crossings_by_source"]["recorder"] == h["cut_links"] > 0
    assert h["crossings_by_source"]["midpoint_fallback"] == 0
    assert h["log"]["log_sha256"] == backend.material.cut_event_recorder.sha256()
    assert info["faces"]["cut_triangles"] == 4*h["cut_links"]
    # WORLD vertices = rigid workpiece pose of the local extraction; normals unit, 3 per triangle
    local = backend.cut_aware_surface.surface.last
    assert vertices.dtype == np.float32 and len(info["face_varying_normals_world"]) == 3*len(faces)
    assert np.allclose(np.linalg.norm(info["face_varying_normals_world"], axis=1), 1, atol=1e-5)
    world_reference = task.points_to_world(local["vertex_reference"])
    assert np.abs(world_reference-vertices).max() < .01      # current vs reference: the 1.5 mm wedge spread, small
    # global piece counts from the mechanical partition; the evaluator sees per-triangle component provenance
    assert sum(r["triangles"] for r in info["component_ranges"]) == len(faces)
    assert info["partition_rule"].startswith("current compatible reference-node")
    row = backend.surface_frame()
    assert row["version"] == sr.PROVENANCE_VERSION and row["codes_sha256"] == info["codes_sha256"]
    assert {"extract_total", "topology_and_qef", "embedding", "normals", "cut_events_log_fetch"} <= set(row["timings_ms"])


@pytest.mark.cpu_sim
def test_frame_rotation_route_equals_rigid_transform_of_the_local_extraction(episodes):
    """A yawed/rotated workpiece: extracting in WORLD with frame_rotation equals extracting LOCAL then rotating."""
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim.cut_aware_surface import CutAwareSurface, face_varying_normals
    env, _, _ = episodes[sr.CUT_AWARE]
    material = env.backend.material
    live = env.backend.cut_aware_surface.surface
    bed = material.bed
    R = Rotation.from_euler("zyx", [35., 4., -3.], degrees=True).as_matrix()
    t = np.array([.31, -.12, .05])
    x, codes, F = bed.x.numpy().astype(np.float64), bed.code.numpy(), bed.F.numpy().astype(np.float64)
    log = material.cut_event_recorder.log()
    fresh = lambda: CutAwareSurface(live.partition, live.spacing, material.lattice, bed.reference.numpy(),  # noqa: E731
                                    strict_audit=False)
    v_local, f_local, _ = fresh().extract(x, codes, F, cut_events_log=log)
    v_world, f_world, _ = fresh().extract(x@R.T+t, codes, F, cut_events_log=log, frame_rotation=R)
    assert np.array_equal(f_local, f_world)
    assert np.abs(v_local@R.T+t-v_world).max() < 1e-12
    n_local = face_varying_normals(v_local, f_local)
    n_world = face_varying_normals(v_world, f_world)
    assert np.abs(n_local@R.T-n_world).max() < 1e-9


@pytest.mark.cpu_sim
def test_reset_starts_a_fresh_topology_no_stale_cut_faces(episodes):
    env, _, initial = episodes[sr.CUT_AWARE]
    backend = env.backend
    before = backend.cut_aware_surface
    assert backend.surface_metadata["faces"]["cut_triangles"] > 0
    env.reset(seed=42)
    after = backend.cut_aware_surface
    assert after is not before and after.surface is not before.surface and after.rows == []
    vertices, faces = backend.surface
    row = after.rows[-1]
    assert row["extraction"] == 0 and row["update"]["mode"] == "full"
    assert row["mesh"]["cut_triangles"] == 0 and row["cut_events"]["entries"] == 0
    assert row["audit"]["mechanical_components"] == 1
    assert np.array_equal(faces, initial[1]) and np.array_equal(vertices, initial[0])
    assert row["codes_sha256"] == initial[2]["codes_sha256"]
    _drive_cut(env, env.task)          # the episode after the reset cuts again from a clean state
    assert backend.surface_metadata["faces"]["cut_triangles"] > 0 and backend.surface_metadata["audit"]["ok"]


@pytest.mark.cpu_sim
def test_temporal_rows_report_vertex_motion_at_unchanged_codes(episodes):
    env, task, _ = episodes[sr.CUT_AWARE]
    backend = env.backend
    first = backend.surface_frame()
    hold = {"mode": "cartesian_edge_pose_world", "target": np.asarray(backend.edge), "cut_enabled": False}
    env.step(hold)
    second = backend.surface_frame()
    assert second["extraction"] == first["extraction"]+1
    temporal = second["temporal"]
    assert temporal["previous"] and temporal["codes_unchanged"] and temporal["topology_unchanged"]
    assert second["update"]["mode"] == "incremental" and second["update"]["dirty_cells"] == 0
    # a vertex is an affine combination of its region's particle motion: no flicker beyond the material motion
    assert temporal["vertex_step_max_m"] <= temporal["particle_step_max_m"]*1.5+1e-12
    summary = backend.surface_provenance()["summary"]
    assert summary["audit_error_extractions"] == 0 and summary["max_shell_component_violations"] == 0
    assert summary["temporal_unchanged_codes"]["frames"] >= 1


# ----------------------------------------------------------------------------- policy env, recording, leaks

@pytest.fixture(scope="module")
def policy_env(tmp_path_factory):
    pytest.importorskip("warp")
    from robotic_cutting_sim.live_env import open_cutting_env
    root = tmp_path_factory.mktemp("surface_policy")
    with open_cutting_env(runtime="kinematic", output_dir=root/"run", preset="cutting_research_v1",
                          particle_spacing_m=.004, material_dt=.0005, device="cpu", scene=SCENE, goal=GOAL,
                          surface_profile=sr.CUT_AWARE, recording={"root": str(root/"episodes"), "every": 2}) as env:
        yield env, root


@pytest.mark.cpu_sim
def test_policy_env_cut_aware_keeps_surface_data_out_of_observations_and_info(policy_env):
    from robotic_cutting_sim.policy_env import find_privileged_info_keys, find_privileged_keys
    env, _ = policy_env
    observation, info = env.reset(seed=42)
    assert env.surface_profile == sr.CUT_AWARE and info["surface_profile"] == sr.CUT_AWARE
    assert find_privileged_info_keys(info) == [] and walk_keys(info, SURFACE_KEYS) == []
    steps = [env.step({"type": "edge_delta", "target": [0., 0., -.001, 0., 0., 0.], "frame": "world",
                       "cut_enabled": True}) for _ in range(3)]
    for observation, _, _, _, step_info in steps:
        assert find_privileged_keys(observation) == [] and walk_keys(observation, SURFACE_KEYS) == []
        assert find_privileged_info_keys(step_info) == [] and walk_keys(step_info, SURFACE_KEYS) == []
    assert walk_keys(env.observe(), SURFACE_KEYS) == []
    with pytest.raises(PermissionError):
        env.surface_provenance()


@pytest.mark.cpu_sim
def test_recording_writes_surface_provenance_and_the_hermite_log(policy_env):
    from robotic_cutting_sim import dataset as ds
    from robotic_cutting_sim.cut_events import LOG_DTYPE
    env, root = policy_env
    env.reset(seed=42)
    for _ in range(4):
        env.step({"type": "edge_delta", "target": [0.]*6})
    directory = env.recorder.output
    env.finish_recording({"status": "TEST"})
    provenance = json.loads((directory/"surface_provenance.json").read_text())
    assert provenance["version"] == sr.PROVENANCE_VERSION
    selection = provenance["selection"]
    assert selection["surface_profile"] == sr.CUT_AWARE and selection["cut_events"] == "cut_events_v1"
    assert selection["writes_physics_state"] is False and selection["is_contact_surface"] is False
    assert set(selection["source_sha256"]) >= {"cut_aware_surface.py", "cut_events.py", "surface_runtime.py"}
    assert provenance["cut_events"]["enabled"] and provenance["cut_events_log"] == "cut_events_log.npy"
    log = np.load(directory/"cut_events_log.npy", allow_pickle=False)
    assert log.dtype == LOG_DTYPE
    frames = json.loads((directory/"frame_manifest.json").read_text())
    assert len(provenance["frames"]) == len(frames) and all(f["codes_match_extraction"] for f in provenance["frames"])
    assert [f["time_s"] for f in provenance["frames"]] == [f["time_s"] for f in frames]
    assert provenance["rows"] and provenance["summary"]["extractions"] == len(provenance["rows"])
    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["config"]["config"]["surface_profile"] == sr.CUT_AWARE
    # the deployable export carries no surface data
    path, summary = ds.export_episode(directory)
    assert not [key for key in summary["keys"] if "surface" in key or "hermite" in key]


@pytest.mark.cpu_sim
def test_default_recording_writes_no_surface_files(tmp_path):
    pytest.importorskip("warp")
    from robotic_cutting_sim.live_env import open_cutting_env
    with open_cutting_env(runtime="kinematic", output_dir=tmp_path/"run", preset="cutting_research_v1",
                          particle_spacing_m=.004, material_dt=.0005, device="cpu", scene=SCENE, goal=GOAL,
                          recording={"root": str(tmp_path/"episodes"), "every": 2}) as env:
        _, info = env.reset(seed=42)
        assert "surface_profile" not in info and env.surface_profile == "particle_surface_v1"
        env.step({"type": "edge_delta", "target": [0.]*6})
        directory = env.recorder.output
        env.finish_recording({"status": "TEST"})
        assert env.backend.surface_selection() is None
    assert not (directory/"surface_provenance.json").exists() and not (directory/"cut_events_log.npy").exists()
    manifest = json.loads((directory/"manifest.json").read_text())
    assert "surface_profile" not in manifest["config"]["config"]


# ----------------------------------------------------------------------------- Native runner surface method (no Kit)

@pytest.mark.cpu_sim
def test_runner_stages_world_mesh_with_face_varying_normals_double_sided(episodes):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from robotic_cutting_sim.isaac_sim_runner import LiveIsaacSimCutting
    from robotic_cutting_sim.usd_surface_bridge import DynamicSurfaceBridge
    env, task, _ = episodes[None]          # the A episode's material (recorder off: midpoint fallback, labelled)
    material = env.backend.material
    runner = LiveIsaacSimCutting.__new__(LiveIsaacSimCutting)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    runner.bridge, runner.material, runner.task = DynamicSurfaceBridge(stage), material, task
    runner.time_s, runner.step_index, runner.surface_trace = .5, 25, []
    runner.cut_aware_surface = sr.LiveCutAwareSurface(material, to_world=task.points_to_world,
                                                      density_kg_m3=task.material.density_kg_m3)
    runner._update_surface()
    mesh = runner.bridge.mesh
    assert mesh.GetDoubleSidedAttr().Get() is True
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    points = np.asarray(mesh.GetPointsAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    normals = np.asarray(mesh.GetNormalsAttr().Get())
    vertices, faces = runner.surface_arrays
    assert np.array_equal(points, vertices) and np.all(counts == 3) and np.array_equal(indices, faces)
    # right-handed winding, outward: each staged corner normal agrees with its triangle's winding normal
    v = points[indices].astype(np.float64)
    winding = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    assert (np.einsum("ij,ij->i", np.repeat(winding, 3, axis=0), normals) > 0).mean() > .999
    assert mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.rightHanded
    row = runner.cut_aware_surface.rows[-1]
    assert row["hermite"]["source"] == "midpoint_fallback_R4" and row["cut_events"] is None
    assert "usd_upload" in row["timings_ms"] and row["bridge"]["bridge_method"] == "update_face_varying_v1"
    assert runner.surface_trace[-1]["surface_profile"] == sr.CUT_AWARE
    assert runner.surface_frame() is row
    provenance = runner.surface_provenance()
    assert provenance["selection"]["surface_profile"] == sr.CUT_AWARE and provenance["cut_events"]["enabled"] is False


@pytest.mark.cpu_sim
def test_zero_area_triangles_get_unit_reference_normals_not_a_crash(episodes):
    """Native surface_live_001 failure: a zero-area CURRENT triangle (collapsed slit-tip quad) has a zero
    face-varying normal, which update_face_varying_v1 rightly rejects. The runtime substitutes the triangle's
    reference-space normal rotated to WORLD (counted); vertices are unchanged."""
    from robotic_cutting_sim.cut_aware_surface import face_varying_normals
    env, task, _ = episodes[sr.CUT_AWARE]
    surface = env.backend.cut_aware_surface
    vertices, faces = env.backend.surface
    world = vertices.astype(np.float64)
    collapsed = world.copy()
    tri = 7
    collapsed[faces[tri]] = collapsed[faces[tri][0]]           # one triangle with zero area in current space
    normals = face_varying_normals(collapsed, faces)
    assert (np.linalg.norm(normals, axis=1) < .5).any()        # the raw function leaves zero vectors
    repaired, counts = surface._repair_degenerate_normals(normals, faces)
    assert np.allclose(np.linalg.norm(repaired, axis=1), 1, atol=1e-9)
    assert counts["corners"] >= 3 and counts["reference_normal_used"] == counts["corners"]
    reference = task.points_to_world(surface.surface.last["vertex_reference"])[faces[tri]]
    expected = np.cross(reference[1]-reference[0], reference[2]-reference[0])
    assert np.allclose(repaired[3*tri], expected/np.linalg.norm(expected), atol=1e-9)
    untouched = np.linalg.norm(normals, axis=1) > .5
    assert np.array_equal(repaired[untouched], normals[untouched])
    assert env.backend.cut_aware_surface.rows[-1]["degenerate_normal_corners"]["corners"] >= 0


def test_recorder_summary_capacity_covers_long_live_episodes():
    """Native surface_live_004 (Test B, 1850 steps x 200 substeps) overflowed the 2^18 default and raised."""
    from types import SimpleNamespace
    long = SimpleNamespace(duration_s=37., control_dt_s=.02)
    assert sr.recorder_summary_capacity(long, 1e-4) == (1850+sr.RECORDER_MARGIN_STEPS)*200 > 1850*200
    assert sr.recorder_summary_capacity(SimpleNamespace(duration_s=1., control_dt_s=.02), 1e-4) == 1 << 18
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    with pytest.raises(ValueError, match="cut_events_summary_capacity"):
        LiveCuttingMaterial(cut_events_summary_capacity=10)


@pytest.mark.cpu_sim
def test_kinematic_cut_aware_recorder_uses_the_episode_summary_capacity(episodes):
    env, task, _ = episodes[sr.CUT_AWARE]
    recorder = env.backend.material.cut_event_recorder
    assert recorder.summary_capacity == sr.recorder_summary_capacity(task, .0005)
