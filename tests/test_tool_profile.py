"""Ruler-knife profile geometry and the tlmpm_v2 blade SDF on Warp's CPU backend.

The v1 module itself cannot compile for CPU (a kernel argument named ``new``),
so v2's legacy-wedge equivalence is checked against a renamed copy of the
frozen v1 source and against v1's own ``_wedge_sdf`` function.
"""
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.physical_tool import make_tool_spec
from robotic_cutting_sim.robot import wedge_mesh
from robotic_cutting_sim.scene_frames import pose_matrix
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.tool_profile import (RULER_KNIFE_V1, ToolProfile, legacy_profile,
                                              make_profile_tool_spec, require_profile_consistency)

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_profile_is_the_retained_wedge():
    for task_id in ("CUT-01", "TRIM-01"):
        task = make_task(task_id)
        vertices, _ = legacy_profile(task).blade_mesh_edge()
        expected, _ = wedge_mesh(task.blade_length_m, task.blade_height_m, task.blade_half_width_m)
        assert np.abs(np.sort(vertices, 0)-np.sort(expected, 0)).max() < 1e-7
        # The legacy ToolSpec's blade collider is the same extruded polygon.
        spec = make_tool_spec(task)
        require_profile_consistency(spec, legacy_profile(task))


def test_ruler_profile_geometry_mass_and_mount():
    p = RULER_KNIFE_V1
    section = p.cross_section_xz()
    assert section.shape == (5, 2) and np.allclose(section[0], 0)
    assert np.isclose(section[:, 1].max()-section[:, 1].min(), p.thickness_m)
    assert np.isclose(np.degrees(2*np.arctan(p.thickness_m/2/p.bevel_length_m)), 25.)
    spec = make_profile_tool_spec(p)
    require_profile_consistency(spec, p)
    blade_mass = p.edge_length_m*p.thickness_m*(p.blade_depth_m-p.bevel_length_m/2)*p.blade_density_kg_m3
    assert np.isclose(spec.shapes[1].mass_properties[1], blade_mass, rtol=1e-9)
    assert .13 < spec.mass_kg < .17  # about 97 g polymer handle + 53 g steel blade
    T = pose_matrix(p.hand_from_edge())
    assert np.allclose(T[:3, 0], [0, 0, 1]) and np.allclose(T[:3, 2], [0, 1, 0])
    assert np.allclose(spec.T_hand_tool@spec.T_tool_edge, T, atol=1e-12)
    # Edge center lies on the grasp axis: cutting force adds no lever arm across the fingers.
    assert np.allclose(T[:2, 3], 0)
    # The handle width across the fingers matches the existing grasp configuration.
    assert np.isclose(p.handle_half_extents_hand_m[1], make_tool_spec(make_task("CUT-01")).handle_half_extents_m[1])
    tampered = make_profile_tool_spec(ToolProfile("other", .121, .04, .0015))
    with pytest.raises(NotImplementedError):
        require_profile_consistency(tampered, p)
    with pytest.raises(ValueError):
        ToolProfile("bad", .1, .002, .0015, bevel_included_angle_deg=5.)


def _renamed_v1():
    """Frozen v1 source with only the C++-keyword argument renamed, as a real file."""
    import tempfile
    name = "robotic_cutting_sim._tlmpm_v1_cpu_test"
    if name in sys.modules:
        return sys.modules[name]
    source = re.sub(r"\bnew\b", "new_v", (ROOT/"src/robotic_cutting_sim/tlmpm.py").read_text())
    path = Path(tempfile.mkdtemp(prefix="tlmpm_v1_cpu_"))/"_tlmpm_v1_cpu_test.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bar(spacing=.004):
    grid = np.mgrid[-.03:.03+1e-9:spacing, -.012:.012+1e-9:spacing, 0:.012+1e-9:spacing]
    points = grid.reshape(3, -1).T.astype(np.float32)+np.float32(spacing/2)
    return points, np.full(len(points), 1500.*spacing**3, np.float32)


def _drive(solver, profile=None, *, rotation=Rotation.identity(), steps=600, speed=.05, cut=0):
    """Push an edge downward (world -z) through the bar, edge-frame +x = advance."""
    import warp as wp
    points, masses = _bar()
    config = TLMPMConfig(cuts=1, blade_half_width=.04 if profile is None else float(profile[1]))
    kwargs = {} if profile is None else {"blade_profile_xz": profile[0]}
    bed = solver.TLMPM(points, masses, .004, config, "cpu", **kwargs)
    base = rotation*Rotation.from_euler("y", 90, degrees=True)
    q = base.as_quat()
    advance = base.apply([1, 0, 0])
    start = np.array([.002, 0, .002])-advance*.016
    poses = np.array([np.r_[start+advance*speed*config.dt*i, q] for i in range(steps+1)], np.float32)
    twists = np.tile(np.r_[advance*speed, 0, 0, 0], (steps+1, 1)).astype(np.float32)
    bed.set_trajectory(poses, twists, np.full(steps+1, cut, np.int32))
    for _ in range(steps):
        bed.step()
    wp.synchronize_device("cpu")
    return bed.snapshot(), bed.reaction.numpy()


def test_v2_legacy_wedge_matches_v1_bitwise_on_cpu():
    pytest.importorskip("warp")
    from robotic_cutting_sim import tlmpm_v2
    v1, v2 = (_drive(solver, steps=600, speed=.3) for solver in (_renamed_v1(), tlmpm_v2))
    for key in ("positions", "velocities", "codes", "F", "cut_progress"):
        assert np.array_equal(v1[0][key], v2[0][key]), key
    assert np.array_equal(v1[1], v2[1]) and np.count_nonzero(v2[0]["codes"]) > 0


def test_profile_sdf_reproduces_v1_wedge_function():
    wp = pytest.importorskip("warp")
    from robotic_cutting_sim import tlmpm_v2
    v1 = _renamed_v1()
    rng = np.random.default_rng(3)
    points = rng.uniform([-.02, -.03, -.006], [.006, .03, .008], (4000, 3)).astype(np.float32)
    triangle = np.array([[0, 0], [-.012, .004], [-.012, -.0004]], np.float32)

    @wp.kernel
    def both(p: wp.array(dtype=wp.vec3), profile: wp.array(dtype=wp.vec2),
             a: wp.array(dtype=wp.vec4), b: wp.array(dtype=wp.vec4)):
        i = wp.tid()
        a[i] = v1._wedge_sdf(p[i], 0.012, 0.004, 0.025)
        b[i] = tlmpm_v2._profile_sdf(p[i], profile, 3, 0.025)

    a = wp.zeros(len(points), dtype=wp.vec4, device="cpu")
    b = wp.zeros(len(points), dtype=wp.vec4, device="cpu")
    wp.launch(both, len(points), [wp.array(points, dtype=wp.vec3, device="cpu"),
              wp.array(triangle, dtype=wp.vec2, device="cpu"), a, b], device="cpu")
    assert np.array_equal(a.numpy(), b.numpy())


def test_profile_sdf_matches_ruler_polygon_geometry():
    wp = pytest.importorskip("warp")
    from robotic_cutting_sim import tlmpm_v2
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    rng = np.random.default_rng(4)
    probes = rng.uniform([-.05, -.07, -.004], [.01, .07, .004], (6000, 3)).astype(np.float32)

    @wp.kernel
    def sdf(p: wp.array(dtype=wp.vec3), profile: wp.array(dtype=wp.vec2), n: int, h: float,
            out: wp.array(dtype=wp.vec4)):
        i = wp.tid()
        out[i] = tlmpm_v2._profile_sdf(p[i], profile, n, h)

    out = wp.zeros(len(probes), dtype=wp.vec4, device="cpu")
    wp.launch(sdf, len(probes), [wp.array(probes, dtype=wp.vec3, device="cpu"),
              wp.array(polygon, dtype=wp.vec2, device="cpu"), len(polygon), float(half), out], device="cpu")
    values = out.numpy()
    from matplotlib.path import Path as Polygon
    inside = Polygon(polygon).contains_points(probes[:, [0, 2]]) & (np.abs(probes[:, 1]) < half)
    certain = np.abs(values[:, 3]) > 1e-6
    assert np.array_equal(values[certain, 3] < 0, inside[certain])
    assert np.allclose(np.linalg.norm(values[:, :3], axis=1), 1, atol=1e-5)


@pytest.mark.parametrize("tilt_deg", [0., 25.])
def test_ruler_blade_severs_a_bar_on_cpu(tilt_deg):
    pytest.importorskip("warp")
    from robotic_cutting_sim import tlmpm_v2
    from robotic_cutting_sim.cut_topology import reference_components
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    # Rotate about the edge line (world y): an oblique cut plane.
    state, _ = _drive(tlmpm_v2, (polygon, half), rotation=Rotation.from_euler("y", tilt_deg, degrees=True),
                      steps=3200, speed=.1)
    points, _ = _bar()
    lattice = np.round((points-points.min(0))/.004).astype(int)
    count, component = reference_components(points, lattice, state["codes"], cuts=1)
    sizes = np.sort(np.bincount(component))[::-1]
    assert count >= 2 and sizes[1] > .3*len(points)
    # The separating surface follows the tilted blade plane through the bar center.
    normal = Rotation.from_euler("y", tilt_deg, degrees=True).apply([1, 0, 0])
    side = (points-np.array([.002, 0, .002]))@normal > 0
    big = component == np.argmax(np.bincount(component))
    assert min(np.mean(side[big]), 1-np.mean(side[big])) < .05
