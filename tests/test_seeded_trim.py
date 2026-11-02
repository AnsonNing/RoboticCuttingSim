"""CPU invariance, diversity and deployable boundary for agentic control robot research."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.agentic_trim_env import AgenticTrimEnv
from robotic_cutting_sim.research_env import vision_observation
from robotic_cutting_sim.scene_config import SceneInfeasible, scene_heightfield
from robotic_cutting_sim.seeded_trim import compile_seed, digest_array, generate_scene
from robotic_cutting_sim.geometry_providers import sample_lattice


def test_seeded_real_particle_diversity():
    scenes = [generate_scene(seed) for seed in range(12)]
    assert {len(s['workpiece']['protrusions']) for s in scenes} == {1, 2}
    assert {b['family'] for s in scenes for b in s['workpiece']['protrusions']} == {
        'dome', 'gaussian', 'ridge', 'irregular'}
    hashes = []
    for scene in scenes:
        points, _, _ = sample_lattice(scene_heightfield(scene), .001, 1500.)
        assert np.isfinite(points).all()
        hashes.append(digest_array(points))
    assert len(set(hashes)) == len(scenes)
    assert generate_scene(7) == scenes[7]


def test_same_seed_full_compile_and_separate_rngs():
    task_a, manifest_a = compile_seed(2, policy_seed=11, sim_seed=42)
    task_b, manifest_b = compile_seed(2, policy_seed=91, sim_seed=43)
    for key in ('scene_hash', 'geometry_hash', 'material_hash', 'solver_hash',
                'initial_particles_sha256', 'initial_lattice_sha256'):
        assert manifest_a[key] == manifest_b[key]
    for a, b in zip(task_a.sample_particles(), task_b.sample_particles()):
        assert np.array_equal(a, b)
    assert manifest_a['policy_seed'] != manifest_b['policy_seed']
    assert manifest_a['sim_seed'] != manifest_b['sim_seed']
    assert manifest_a['initial_lattice_connectivity']['components'] == 1
    assert min(manifest_a['target_particle_counts']) >= 20
    assert task_a.max_cuts == task_b.max_cuts == 2


def camera_payload():
    rgb = np.ones((4, 6, 3), np.uint8)
    wrist = dict(rgb=rgb, depth_m=np.ones((4, 6), np.float32), surface_timestamp_s=0., physics_timestamp_s=0.)
    return {'wrist_d405_0': wrist,
            'global_d455': dict(rgb=rgb.copy(), modalities=['rgb'], surface_timestamp_s=0., physics_timestamp_s=0.)}


def test_optional_depth_is_absent_not_fabricated():
    frames = camera_payload()
    obs = vision_observation(frames, {k: {} for k in frames}, 0., 'vision_only', include_global_camera=True)
    assert set(obs['rgb']) == {'global_d455', 'wrist_d405_0'}
    assert set(obs['depth_m']) == set(obs['valid_mask']) == {'wrist_d405_0'}
    del frames['global_d455']['modalities']
    with pytest.raises(KeyError):
        vision_observation(frames, {k: {} for k in frames}, 0., 'vision_only', include_global_camera=True)


def test_optional_camera_stale_frame_is_rejected():
    frames = camera_payload()
    frames['global_d455']['physics_timestamp_s'] = .02
    with pytest.raises(ValueError, match='timestamps'):
        vision_observation(frames, {k: {} for k in frames}, 0., 'vision_only', include_global_camera=True)


def test_live_rgb_option_preserves_stereo_guard():
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
    opts = dict(camera_rig_profile='wrist_camera_rig_v14', camera_mode='native_single_view',
                camera_layout='wrist_d405_global_d455')
    with pytest.raises(NotImplementedError, match='BLOCKED'):
        IsaacSimCuttingConfig(**opts)
    config = IsaacSimCuttingConfig(**opts, global_rgb_only=True,
                                  active_camera_ids=('global_d455', 'wrist_d405_0'))
    assert config.global_rgb_only
    with pytest.raises(ValueError, match='active_camera_ids'):
        IsaacSimCuttingConfig(**opts, global_rgb_only=True, active_camera_ids=('made_up_camera',))


def test_no_reference_no_gripper_release_and_no_truth_metadata():
    adapter = object.__new__(AgenticTrimEnv)
    adapter.env = SimpleNamespace(action_limits=SimpleNamespace(to_dict=lambda: {}))
    with pytest.raises(ValueError, match='Explicit'):
        adapter.step(None)
    spec = adapter.task_spec()
    assert spec['gripper'] == 'hold'
    assert not {'scene', 'protrusions', 'reference', 'world_from_workpiece_xyzw', 'scene_seed'} & set(spec)
    assert spec['controlled_arm'] == 'arm_0'
    assert spec['max_cuts'] == 2


def test_observation_strips_second_arm_and_hidden_task():
    adapter = object.__new__(AgenticTrimEnv)
    frames = camera_payload()
    cameras = vision_observation(frames, {k: {} for k in frames}, 0., 'vision_only', include_global_camera=True)
    cameras.pop('timestamp_s')
    original = dict(timestamp_s=0., cameras=cameras, proprio=dict(q=np.zeros((2, 7)),
        qdot=np.zeros((2, 7)), hand_pose=np.zeros(7), finger_q=np.ones((2, 2)), tool_truth=np.ones(7)),
        ee_pose_world_xyzw=np.zeros(7), edge_command_world_xyzw=np.zeros(7), gripper={},
        camera_timestamp_s=0., task={'protrusions': [{'x_m': .123}]})
    adapter.env = SimpleNamespace(observe=lambda: copy.deepcopy(original),
                                 action_limits=SimpleNamespace(to_dict=lambda: {}))
    public = adapter.observe()
    assert public['proprio']['q'].shape == (7,)
    assert public['proprio']['finger_q'].shape == (2,)
    assert 'tool_truth' not in public['proprio']
    assert 'protrusions' not in public['task']
    assert 'x_m' not in str(public['task'])
    assert 'wrist_d405_1' not in public['cameras']['rgb']


@pytest.mark.parametrize('seed', [-1, True, 1.5])
def test_invalid_seeds_are_not_replaced(seed):
    with pytest.raises(ValueError):
        generate_scene(seed)


def test_challenge_dataset_export_keeps_optional_modalities_and_active_arm(tmp_path):
    import json
    from robotic_cutting_sim.dataset import export_episode
    rec = tmp_path/'attempt_000'
    (rec/'observations').mkdir(parents=True)
    (rec/'provenance.json').write_text(json.dumps({'version': 'rcs_trim_random_sim_v1'}))
    (rec/'frame_manifest.json').write_text(json.dumps([{'time_s': 0.}]))
    row = dict(time_s=.02, info={}, submitted={}, observation_proprio={
        'q': [[0.]*7, [1.]*7], 'qdot': [[0.]*7, [1.]*7], 'finger_q': [[.01,.01],[.04,.04]],
        'tool_pose': [1.]*7})
    (rec/'actions.jsonl').write_text(json.dumps(row)+'\n')
    for name, frame in camera_payload().items():
        np.savez_compressed(rec/'observations'/f'{name}_0000.npz', **frame)
    path, _ = export_episode(rec)
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays['proprio__q'].shape == (1, 7)
        assert arrays['proprio__finger_q'].shape == (1, 2)
        assert 'proprio__tool_pose' not in arrays
        assert 'camera__global_d455__depth_m' not in arrays
        assert 'camera__global_d455__valid_mask' not in arrays
        assert arrays['camera__wrist_d405_0__depth_m'].shape == (1, 4, 6)
