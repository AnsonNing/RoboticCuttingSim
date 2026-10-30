"""root_cause_diagnostics (WP-12 Phase 4 verification): pure-NumPy read-only metrics, checked on exact cases."""
import numpy as np
import pytest

from robotic_cutting_sim import root_cause_diagnostics as rd
from robotic_cutting_sim.cut_topology import conflicting_packed


def box(a, b, c, s=1e-3):
    lat = np.stack(np.meshgrid(np.arange(a), np.arange(b), np.arange(c), indexing="ij"), -1).reshape(-1, 3)
    return (lat+.5)*s, lat


def test_conflict_matches_cut_topology():
    codes = np.arange(0, 64)
    a, b = np.meshgrid(codes, codes, indexing="ij")
    valid = ((a & 0x2A) >> 1 & a & 0x15) == 0      # no digit 3 in the first three digits
    valid &= ((b & 0x2A) >> 1 & b & 0x15) == 0
    ours = rd.conflict_packed(a[valid], b[valid])
    ref = np.array([conflicting_packed(int(x), int(y)) for x, y in zip(a[valid], b[valid])])
    assert np.array_equal(ours, ref)


def test_lattice_links_count_and_direction():
    X, lat = box(4, 3, 2)
    links = rd.lattice_links(lat)
    assert sum(len(p) for p, _, _ in links) == 3*3*2+4*2*2+4*3*1
    for p, q, axis in links:
        e = np.zeros(3, int)
        e[axis] = 1
        assert np.array_equal(lat[q]-lat[p], np.tile(e, (len(p), 1)))


def test_affine_motion_is_consistent_and_drift_explains_the_rest():
    X, lat = box(6, 4, 4)
    A = np.array([[1.1, .05, 0.], [0., .95, .02], [.01, 0., 1.03]])
    F = np.tile(A, (len(X), 1, 1))
    x = X@A.T+np.array([.01, -.02, .003])
    dist = lambda P: np.abs(P[:, 0]-3e-3)  # noqa: E731
    res = rd.link_consistency(X, lat, x, F, np.zeros(len(X), int), 1e-3, dist)
    assert all(res[k]["max"] < 1e-9 for k in res if k.endswith("s") and res[k]["n"])
    rng = np.random.default_rng(0)
    d = rng.normal(0, 2e-4, X.shape)
    res = rd.link_consistency(X, lat, x+d, F, np.zeros(len(X), int), 1e-3, dist, drift=d,
                              assign_substep=np.zeros((len(X), 1), int))
    first = res["0-1s"]
    assert first["identity"]["explained_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert first["p99"] > .1 and first["same_assignment_history"]["n"] == first["n"]


def test_conflicting_links_are_excluded():
    X, lat = box(4, 1, 1)
    codes = np.array([1, 1, 2, 2])                  # link 1-2 conflicts
    x = X.copy()
    x[2:, 0] += .01                                 # opening at the conflicting link only
    res = rd.link_consistency(X, lat, x, np.tile(np.eye(3), (4, 1, 1)), codes, 1e-3, lambda P: np.abs(P[:, 0]-2e-3))
    assert sum(res[k]["n"] for k in res if k.endswith("s")) == 2
    assert res["0-1s"]["max"] < 1e-12 if res["0-1s"]["n"] else True


def test_elastic_state_bounds():
    E, nu, sy = 5e4, .3, 1e3
    st = rd.elastic_state(np.tile(np.eye(3), (2, 1, 1)), np.tile(np.eye(3), (2, 1, 1)), E, nu, sy)
    assert np.allclose(st["psi"], 0)
    F = np.tile(np.diag([1.5, 1/np.sqrt(1.5), 1/np.sqrt(1.5)]), (1, 1, 1))   # isochoric, far beyond yield
    st = rd.elastic_state(F, np.tile(np.eye(3), (1, 1, 1)), E, nu, sy)
    assert st["psi_dev"][0] == pytest.approx(st["psi_dev_bound"], rel=1e-9)
    assert st["psi_dev_bound"] == pytest.approx(sy**2/(6*E/(2*(1+nu))))
    assert st["von_mises_kirchhoff_pa"][0] == pytest.approx(sy, rel=1e-9)


def test_series_fit_reproduces_review_numbers():
    f_inf, dt0 = rd.series_fit(.05, .0327, .025, .0521)
    assert dt0 == pytest.approx(.0504, abs=5e-4) and f_inf == pytest.approx(.065, abs=1e-3)
    assert rd.series_predict(f_inf, dt0, .05) == pytest.approx(.0327, rel=1e-9)


def test_reference_stencil_matches_tlmpm_v5():
    wp = pytest.importorskip("warp")  # noqa: F841
    from robotic_cutting_sim.blade_probe import block
    from robotic_cutting_sim.tlmpm_v5 import TLMPM
    points, masses = block()
    bed = TLMPM(points, masses, .004, device="cpu")
    nodes, weights = rd.reference_stencil(points, .004)
    assert np.array_equal(nodes, bed.nodes.numpy())
    np.testing.assert_allclose(weights, bed.weights.numpy(), atol=1e-6)


def test_mis_sided_and_visibility_on_a_two_column_toy():
    # two particle columns x = -0.5, +0.5 mm sharing every node; plane x = 0; digit 1 left, 2 right; a code-0 bridge
    X = np.array([[-.5e-3, 0, .5e-3], [.5e-3, 0, .5e-3], [-.5e-3, 0, 1.5e-3], [.5e-3, 0, 1.5e-3]])
    nodes = np.zeros((4, 8), int)
    nodes[:, 1:] = np.arange(1, 8)
    weights = np.full((4, 8), 1/8)
    codes = np.array([0, 2, 1, 2])
    s = X[:, 0]
    r = rd.mis_sided_bridges(codes, nodes, weights, s)
    assert r["bridges"] == 1 and r["need_mis_sided_label"] == 0 and r["in_core_wedge"] == 1
    codes_bad = np.array([0, 2, 2, 1])               # digit 2 on the left, 1 on the right: mis-sided
    r = rd.mis_sided_bridges(codes_bad, nodes, weights, s)
    assert r["bridges"] == 1 and r["need_mis_sided_label"] == 1 and r["mis_sided_labelled_particles"] == 2
    new, lab = rd.visibility_relabel(X, codes, nodes, weights, [0, 0, 0], [1, 0, 0], 0.)
    assert lab["labels"] == 1 and (new[0] & 3) == 1
    new, lab = rd.visibility_relabel(X, codes, nodes, weights, [0, 0, 0], [1, 0, 0], 1e-3)   # crossing below h
    assert lab["labels"] == 0 and np.array_equal(new, codes)


def test_chip_links_and_energy():
    X, lat = box(4, 4, 6)
    region = X[:, 2] > 3e-3
    x = X.copy()
    x[region, 0] += np.where(X[region, 0] > 2e-3, 3e-3, 0.)   # positions torn apart, F = I
    F = np.tile(np.eye(3), (len(X), 1, 1))
    r = rd.chip_links(X, lat, x, F, np.zeros(len(X), int), 1e-3, region)
    assert r["long_links"] > 0 and r["long_with_F_length_below_rule"] == 1.0
    assert r["endpoint_max_principal_stretch"]["max"] == pytest.approx(1.0)
    e = rd.chip_energy(X, F, F, np.zeros(len(X)), region, 1e-3, dict(young=5e4, poisson=.3, yield_stress=1e3))
    assert e["griffith"]["chip_thickness_max_m"] == pytest.approx(3e-3)
    assert e["griffith"]["thin_particles_with_psi_t_ge_gc"] == 0


def test_cut_face_particles_and_griffith_gating():
    X, lat = box(4, 1, 3)
    codes = np.where(X[:, 0] < 2e-3, 1, 2)                     # a cut between columns 1 and 2
    face = rd.cut_face_particles(lat, codes)
    assert np.array_equal(face, np.isin(np.round(X[:, 0]*1e4).astype(int), (15, 25)))
    F = np.tile(np.diag([1.2, 1.2, 1.2]), (len(X), 1, 1))       # dilatation: tensile energy everywhere
    hist = {"psi_tensile_max_j_m3": np.where(face, 100., 1.), "psi_total_max_j_m3": np.where(face, 100., 1.)}
    e = rd.chip_energy(X, F, np.tile(np.eye(3), (len(X), 1, 1)), np.zeros(len(X)), np.ones(len(X), bool), 1e-3,
                       dict(young=5e4, poisson=.3, yield_stress=1e3), history=hist, codes=codes, lattice=lat)
    g = e["griffith"]
    assert g["thin_particles_with_psi_t_ge_gc"] == 0 and g["cut_face_thin_reported"]["with_psi_t_ge_gc"] == face.sum()
