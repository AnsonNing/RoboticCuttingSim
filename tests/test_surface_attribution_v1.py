"""CPU tests of surface_attribution_v1 (WP-12 Phase 1; NumPy/SciPy only, no Newton or Warp)."""
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import surface_attribution_v1 as A
from robotic_cutting_sim import surface_metrics_v1 as M
from robotic_cutting_sim.cut_events import LOG_DTYPE

H = .001
VERTICAL = Rotation.from_euler("y", 90, degrees=True)   # blade_probe.VERTICAL_CUT: advance -z, edge y, normal +x


def test_imports_without_newton_or_warp():
    code = ("import sys; import robotic_cutting_sim.surface_attribution_v1; "
            "print(any(m in sys.modules for m in ('newton', 'warp', 'pxr')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def vertical_chain(x0=.0003, z0=.016, z1=-.004, steps=200):
    z = np.linspace(z0, z1, steps)
    q = np.tile(VERTICAL.as_quat(), (steps, 1))
    return [(np.c_[np.full(steps, x0), np.zeros(steps), z, q], np.zeros(steps, int))]


def test_edge_sweep_signed_distance_follows_the_blade_normal():
    chains = vertical_chain()
    v, f, n, strips = A.edge_sweep_mesh(chains, 0, .03, footprint=np.array([[-.01, -.01, 0], [.01, .01, .012]]))
    assert strips == 1 and len(f) > 0 and np.allclose(n, [1, 0, 0])
    pts = np.array([[.0013, 0, .005], [-.0007, .002, .008], [.0003, -.003, .001]])
    d = A.signed_distance_to_mesh(pts, v, f, n)
    assert np.allclose(d, [.001, -.001, 0.], atol=1e-12)
    # other digits and disarmed poses produce no sweep
    assert A.edge_sweep_mesh(chains, 1, .03)[3] == 0


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3))
    r = Rotation.from_euler("xyz", [5, -20, 33], degrees=True).as_matrix()
    R, t = A.kabsch(a, a@r.T+[1, 2, 3])
    assert np.allclose(R, r) and np.allclose(t, [1, 2, 3])


def synthetic_cut(nx=14, ny=8, nz=8, x0=.0063):
    lattice = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    reference = lattice*H
    n = np.array([1., 0, 0])
    d = (reference[:, 0]-x0)
    codes = np.where(d >= 0, 2, 1).astype(np.int64)
    pairs, _ = M.conflicting_links_per_cut(lattice, codes, cuts=1)
    a, b = pairs.T
    t = d[a]/(d[a]-d[b])
    log = np.zeros(len(a), LOG_DTYPE)
    log["p"], log["q"], log["t"] = a, b, t
    log["hermite_reference"] = reference[a]+t[:, None]*(reference[b]-reference[a])
    log["normal_reference"] = n
    log["normal_current"] = n
    return lattice, reference, codes, log, x0


def test_attribution_of_an_undeformed_exact_cut_is_zero_except_the_staircase():
    lattice, reference, codes, log, x0 = synthetic_cut()
    gap = np.where(reference[:, :1] >= x0, [[.002, 0, 0]], 0.)    # rigid opening after the cut (geometry only)
    positions = reference+gap
    labels = (reference[:, 0] >= x0).astype(int)
    locus = M.material_locus(reference, lattice, positions, codes, cuts=1, events=log)
    mid = M.material_locus(reference, lattice, positions, codes, cuts=1)
    ha = log["hermite_reference"].astype(float)          # at the assignment instant x = X
    out = A.attribute_cut(0, locus=locus, lattice=lattice, reference=reference, positions=positions, labels=labels,
                          spacing=H, chains=vertical_chain(x0=x0, z0=.012, z1=-.004), half_length_m=.03, log=log,
                          hermite_assign=ha, assign_endpoints=(reference[log["p"]], reference[log["q"]]),
                          declared_plane=((x0, 0, 0), (1, 0, 0)), midpoint_locus=mid)
    assert out["status"] == "MEASURED" and out["links"] == 8*8
    assert out["L2_assignment_vs_sweep"]["all"]["max"] < 1e-9
    assert out["L3_pre_assignment_displacement"]["all"]["max"] < 1e-9
    assert out["L1_path_vs_declared_plane"]["max"] < 1e-12
    assert out["L5_post_cut_deformation"]["normal_residual_all"]["max"] < 1e-9     # rigid opening is not deformation
    assert out["L5_post_cut_particle_residual"]["normal_residual_core"]["max"] < 1e-9
    # axis plane 0.2 mm from the cell face: the staircase (midpoint vs Hermite) is the constant 0.2 mm offset
    assert out["L4_lattice_staircase_reference"]["all"]["p50"] == pytest.approx(.0002, abs=1e-7)
    b = out["budget_quadratic_mean_patch_rms_m"]
    assert b["sweep"] < 1e-12 and b["assignment"] < 1e-9 and b["final_physics"] < 1e-9


def test_post_cut_bending_is_reported_as_deformation():
    lattice, reference, codes, log, x0 = synthetic_cut()
    positions = reference.copy()
    right = reference[:, 0] >= x0
    positions[right, 0] += .0004*np.sin(reference[right, 1]/.0006)    # wavy face after the cut (4 mm wavelength)
    labels = right.astype(int)
    locus = M.material_locus(reference, lattice, positions, codes, cuts=1, events=log)
    out = A.attribute_cut(0, locus=locus, lattice=lattice, reference=reference, positions=positions, labels=labels,
                          spacing=H, chains=vertical_chain(x0=x0), half_length_m=.03, log=log,
                          hermite_assign=log["hermite_reference"].astype(float),
                          assign_endpoints=(reference[log["p"]], reference[log["q"]]))
    assert out["L5_post_cut_particle_residual"]["normal_residual_core"]["max"] > 1e-4
    assert out["budget_increments_m"]["L5_post_cut_physics"] > 5e-5


def test_increments_are_signed_quadrature_differences():
    inc = A.increments({"sweep": 0., "assignment": .3, "final_physics": .5, "final_lattice": .5, "mesh_x": .4})
    assert inc["L2_assignment"] == pytest.approx(.3) and inc["L5_post_cut_physics"] == pytest.approx(.4)
    assert inc["L4_lattice_in_current_space"] == 0 and inc["L6_reconstruction_x"] == pytest.approx(-.3)


def test_topology_layer_counts_code_zero_bridges():
    lattice = np.stack(np.meshgrid(np.arange(12), np.arange(4), np.arange(4), indexing="ij"), -1).reshape(-1, 3)
    reference = (lattice*H).astype(np.float32)
    codes = np.where(lattice[:, 0] < 6, 1, 2).astype(np.int64)
    codes[(lattice[:, 0] == 6) & (lattice[:, 2] == 0)] = 0          # an unlabelled bottom row: lattice-adjacent to both sides
    nodes, weights, grads = A.stencil(reference, .004)
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    count, labels = mechanical_components_packed(nodes, weights, codes, 1, gradients=grads)
    out = A.topology_layer(nodes, weights, grads, codes, 1, lattice=lattice, labels=labels, masses=np.ones(len(codes)),
                           reference=reference)
    assert count == 1 and out["mechanical_components"] == 1
    assert out["per_cut"]["0"]["code0_touching_both_sides_on_lattice"] == 4
    assert out["per_cut"]["0"]["bridge_report"]["direct_code_zero_bridges"] > 0
