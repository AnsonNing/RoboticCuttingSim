"""ADR 0021 §3-4, §10: batch layout, isolation audits, int32 guards, uniformity and every UNSUPPORTED code."""
import re
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from batched_fixtures import start_pose, trim_material
from robotic_cutting_sim import tlmpm_v7_2
from robotic_cutting_sim.batched import constitutive, layout, tlmpm_v7_2_batch
from robotic_cutting_sim.batched.errors import CODES, MATERIAL_CODES, ROBOT_CODES, UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.material_batch import BatchedCuttingMaterial
from robotic_cutting_sim.batched.tlmpm_v7_2_batch import BatchedTLMPMv7_2

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def two_envs():
    singles, specs = [], []
    for seed in (512, 15):
        single = trim_material(seed)
        single.reset(start_pose(seed), seed=42)
        singles.append(single)
        specs.append(layout.EnvSpec(trim_material(seed), start_pose(seed), 7))
    packed = layout.pack(specs)
    return packed, singles


def test_offsets_and_concatenation(two_envs):
    packed, singles = two_envs
    beds = [s.bed for s in singles]
    assert packed.num_envs == 2
    assert packed.particle_counts.tolist() == [b.n for b in beds]
    assert packed.node_counts.tolist() == [b.node_count for b in beds]
    assert packed.particle_offset.tolist() == [0, beds[0].n, beds[0].n+beds[1].n]
    assert packed.node_offset.tolist() == [0, beds[0].node_count, beds[0].node_count+beds[1].node_count]
    h = packed.host
    n0, nn0 = beds[0].n, beds[0].node_count
    np.testing.assert_array_equal(h["nodes"][:n0], beds[0].nodes.numpy())
    np.testing.assert_array_equal(h["nodes"][n0:], beds[1].nodes.numpy()+nn0)       # global node ids
    np.testing.assert_array_equal(h["env_of_particle"], np.repeat([0, 1], packed.particle_counts))
    l0 = len(beds[0].link_neighbours)
    np.testing.assert_array_equal(h["link_neighbours"][l0:], beds[1].link_neighbours.numpy()+n0)
    np.testing.assert_array_equal(h["link_offsets"][nn0:], beds[1].link_offsets.numpy()+l0)
    assert len(h["link_offsets"]) == packed.total_nodes+1
    S = packed.uniform["slots"]
    assert len(h["slot_code"]) == packed.total_nodes*S
    for name, width in (("stats_sum", 7), ("stats_max", 7), ("contact_totals_sum", 12), ("contact_totals_max", 12)):
        assert h[name].shape == (2*width,)
    assert h["params"].shape == (2, constitutive.NUM_PARAMS) and h["params"].dtype == np.float32
    expected = [np.float32(v) for v in constitutive.env_parameters(beds[0].config, beds[0].slice_efficiency)]
    np.testing.assert_array_equal(h["params"][0], expected)
    assert packed.env["seed"].tolist() == [7, 7]
    assert packed.scene_keys == [s.task.config_sha256 for s in singles]
    receipt = packed.layout_receipt()
    assert receipt["total_particles"] == packed.total_particles and receipt["int32_guards"]["limit"] == 2**31
    assert all(r["solver_receipt"]["solver"] == "tlmpm_v7_2" for r in packed.receipts)


def test_range_audits_pass_and_detect_leaks(two_envs):
    packed, _ = two_envs
    h = packed.host
    args = lambda **o: [o.get(k, h[k]).copy() for k in ("nodes", "link_offsets", "link_neighbours", "pending_q")]
    clean = layout.range_audit(packed, *args(), h["env_of_particle"])
    assert set(clean.values()) == {0}
    nodes = h["nodes"].copy()
    nodes[0, 0] = packed.node_offset[1]                    # a stencil node of env 1 used by env 0
    assert layout.range_audit(packed, *args(nodes=nodes), h["env_of_particle"])["node_out_of_env"] == 1
    neighbours = h["link_neighbours"].copy()
    neighbours[0] = packed.particle_offset[1]              # env 0's CSR names a particle of env 1
    assert layout.range_audit(packed, *args(link_neighbours=neighbours),
                              h["env_of_particle"])["neighbour_out_of_env"] == 1
    pending = h["pending_q"].copy()
    pending[0] = packed.particle_offset[1]
    assert layout.range_audit(packed, *args(pending_q=pending), h["env_of_particle"])["pending_out_of_env"] == 1


def test_snapshot_at_pack_equals_frozen_singles(two_envs):
    from batched_fixtures import compare_env
    packed, singles = two_envs
    solver = BatchedTLMPMv7_2(packed, "cpu")
    for e, single in enumerate(singles):
        assert compare_env(solver, e, single.bed) == []
    assert solver.memory_bytes() > sum(s.bed.memory_bytes() for s in singles)  # adds env_of_particle, offsets


def test_int32_guards():
    layout.check_int32(2**28-1, 1000, 8, 10)
    with pytest.raises(UnsupportedBatchConfiguration, match="^UNSUPPORTED_INT32_OVERFLOW"):
        layout.check_int32(2**28, 1000, 8, 10)
    with pytest.raises(UnsupportedBatchConfiguration, match="^UNSUPPORTED_INT32_OVERFLOW"):
        layout.check_int32(10, 2**28, 8, 10)
    with pytest.raises(UnsupportedBatchConfiguration, match="^UNSUPPORTED_INT32_OVERFLOW"):
        layout.check_int32(10, 10, 8, 2**31)


def test_no_shared_index_zero_atomics_in_env_kernels():
    """ADR §4: env-state targets are indexed by env; the only [0] access is the shared substep index."""
    source = Path(tlmpm_v7_2_batch.__file__).read_text()
    arrays = set(re.findall(r"(\w+): wp\.array", source))
    assert {"stats_sum", "reaction", "stroke", "previous", "engaged"} <= arrays
    zero_index = set(re.findall(r"(\w+)\[0\]", source)) & arrays
    assert zero_index == {"index"}, zero_index
    assert not re.search(r"wp\.atomic_\w+\(\w+,\s*0\s*,", source)
    material = Path(tlmpm_v7_2_batch.__file__).with_name("material_batch.py").read_text()
    assert set(re.findall(r"wp\.atomic_\w+\((\w+),\s*0\s*,", material)) <= {"invalid_count"}


def test_uniformity_keys_cover_all_kernel_scalars():
    signature, tool_keys = layout.uniform_signature(_tiny_bed(), .02)
    for key in ("dt", "dx", "spacing", "trajectory", "slots", "max_cuts", "cutting_law", "segments", "boxes",
                "material_contact", "pic", "gravity", "floor", "blade_profile_xz", "control_dt"):
        assert key in signature
    assert "blade_profile_xz" in tool_keys


def _tiny_bed(**config):
    from robotic_cutting_sim.blade_probe import block
    from robotic_cutting_sim.material_config import TLMPMConfig
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y
    points, masses = block(size_m=(.016, .008, .008), spacing=.004)
    polygon, half = RULER_KNIFE_V2_Y.tlmpm_profile()
    base = dict(cuts=2, floor=False, fixture_walls=False, blade_half_width=float(half))
    base.update(config)
    return tlmpm_v7_2.TLMPM(points, masses, .004, TLMPMConfig(**base), "cpu", blade_profile_xz=polygon,
                            cutting_law=8)


def _expect(code, call):
    with pytest.raises(UnsupportedBatchConfiguration) as info:
        call()
    assert info.value.code == code and str(info.value).startswith("UNSUPPORTED"), str(info.value)


def _pack_pair(first, second):
    return lambda: layout.pack([layout.EnvSpec(first, start_pose(512), 42),
                                layout.EnvSpec(second, start_pose(512), 42)])


def _stepped_bed():
    bed = _tiny_bed()
    bed.step()
    return bed


def _material(**options):
    def make():
        return BatchedCuttingMaterial([layout.EnvSpec(trim_material(512), start_pose(512), 42)], device="cpu",
                                      parity_mode=True, **options)
    return make


def test_every_unsupported_code_is_raised():
    from dataclasses import replace
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from robotic_cutting_sim.tool_profile import PROFILES
    from batched_fixtures import trim_task

    task = trim_task(512)
    config = CouplingConfig(control_dt=.02, material_dt=1e-4, dx=task.grid_dx_m, device="cpu")
    other_profile = next(p for name, p in PROFILES.items() if p != task.tool_profile)
    cases = {
        "UNSUPPORTED_SOLVER": lambda: layout.pack([layout.EnvSpec(
            LiveCuttingMaterial(config, task=task, max_cuts=2, cutting_law=8, solver="tlmpm_v7_1"),
            start_pose(512))]),
        "UNSUPPORTED_V7_2_OPTIMIZATIONS_DISABLED": lambda: layout.pack([layout.EnvSpec(
            LiveCuttingMaterial(config, task=task, max_cuts=2, cutting_law=8, solver="tlmpm_v7_2",
                                v7_2_optimizations=False), start_pose(512))]),
        "UNSUPPORTED_NONUNIFORM_BATCH": _pack_pair(
            trim_material(512),
            LiveCuttingMaterial(replace(config, material_dt=2e-4), task=task, max_cuts=2, cutting_law=8,
                                solver="tlmpm_v7_2")),
        "UNSUPPORTED_PER_ENV_TOOL_PROFILE": _pack_pair(
            trim_material(512),
            LiveCuttingMaterial(config, task=task, max_cuts=2, cutting_law=8, solver="tlmpm_v7_2",
                                tool_profile=other_profile)),
        "UNSUPPORTED_LEGACY_FLOOR": lambda: layout.pack_from_beds([_tiny_bed(floor=True)]),
        "UNSUPPORTED_RECORDER": lambda: layout.pack([layout.EnvSpec(
            LiveCuttingMaterial(config, task=task, max_cuts=2, cutting_law=8, solver="tlmpm_v7_2",
                                cut_events="cut_events_v1"), start_pose(512))]),
        "UNSUPPORTED_SENSOR": _material(cameras=["wrist_d405"]),
        "UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET": lambda: _material()().reset([0], scene_keys=["other"]),
        "UNSUPPORTED_WORKPIECE_MOTION": lambda: _material()().set_workpiece_pose([0, 0, 0, 0, 0, 0, 1]),
        "UNSUPPORTED_BACKEND": _material(backend="physx_rigid"),
        "UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS": _material(constitutive_model="neo_hookean_v1"),
        "UNSUPPORTED_CPU_WITHOUT_PARITY_MODE": lambda: BatchedCuttingMaterial(
            [layout.EnvSpec(trim_material(512), start_pose(512), 42)], device="cpu"),
        "UNSUPPORTED_INT32_OVERFLOW": lambda: layout.check_int32(2**31, 1, 8, 1),
        "UNSUPPORTED_EMPTY_BATCH": lambda: layout.pack([]),
        "UNSUPPORTED_SINGLE_ENV_STATE": lambda: layout.pack_from_beds([_stepped_bed()]),
    }
    assert set(cases) == set(MATERIAL_CODES), set(MATERIAL_CODES) ^ set(cases)
    for code, call in cases.items():
        _expect(code, call)
    # Workload B codes are raised by the robot path (tests/test_rl_lab_*); here only their registration.
    assert set(CODES) == set(MATERIAL_CODES) | set(ROBOT_CODES) and len(set(CODES)) == len(CODES)
    for code in ROBOT_CODES:
        assert UnsupportedBatchConfiguration(code, "x").code == code
    # Recorders are also refused at the material level.
    _expect("UNSUPPORTED_RECORDER", _material(recorders=["cut_aware_surface_v1"]))
    with pytest.raises(ValueError, match="unknown UNSUPPORTED code"):
        UnsupportedBatchConfiguration("NOT_A_CODE")


def test_pack_requires_cpu_single_solvers():
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from batched_fixtures import trim_task
    task = trim_task(512)
    material = LiveCuttingMaterial(CouplingConfig(control_dt=.02, material_dt=1e-4, dx=task.grid_dx_m,
                                                  device="cuda:0"), task=task, max_cuts=2, cutting_law=8,
                                   solver="tlmpm_v7_2")
    with pytest.raises(ValueError, match="CPU device"):
        layout.pack([layout.EnvSpec(material, start_pose(512))])


def test_device_arrays_have_expected_dtypes(two_envs):
    packed, _ = two_envs
    solver = BatchedTLMPMv7_2(packed, "cpu")
    assert solver.nodes.dtype == wp.int32 and solver.nodes.ndim == 2
    assert solver.gradients.dtype == wp.vec3 and solver.gradients.shape == (packed.total_particles, 8)
    assert solver.contact_totals_sum.dtype == wp.float64
    assert solver.params.shape == (2, constitutive.NUM_PARAMS)
