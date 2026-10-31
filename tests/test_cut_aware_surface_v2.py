"""cut_aware_surface_v2 (decision review section 2, M1-M5) on synthetic zero-physics / affine data (CPU, no Warp).

Codes and exact Hermite crossings come from geometric rules outside any physics run (as tests/test_cut_aware_surface.py);
no surface_protocol threshold is used. The v1 tests (tests/test_cut_aware_surface.py) are unchanged and must keep
passing: v1 stays byte-identical.
"""
import numpy as np
import pytest

from robotic_cutting_sim import cut_aware_surface as cas
from robotic_cutting_sim import cut_aware_surface_v2 as cv2
from robotic_cutting_sim.cut_topology import pack_digits
from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition

from test_cut_aware_surface import DX, EXACT, S, box, exact_log, identity_F, plane_digits, progressive_frames


def partition(X, cuts=1):
    return CurrentMechanicalPartition(*cas.reference_stencil(X, DX), cuts, encoding="packed2")


def both(ijk, X, cuts=1, fixes=("M1", "M2", "M3", "M4"), **options):
    v1 = cas.CutAwareSurface(partition(X, cuts), S, ijk, X, **options)
    v2 = cv2.CutAwareSurfaceV2(partition(X, cuts), S, ijk, X, fixes=fixes, **options)
    return v1, v2


def same(a, b):
    va, fa, ia = a
    vb, fb, ib = b
    return fa.shape == fb.shape and np.array_equal(fa, fb) and np.array_equal(va, vb)


def oblique_case(dims=(14, 10, 8), normal=(1., 2, 3), gap=.003):
    ijk, X, _, _ = box(dims)
    n = np.asarray(normal, float)/np.linalg.norm(normal)
    origin = X.mean(0)+np.array([.0003, .0002, .0001])
    digits = plane_digits(X, origin, n)[:, None]
    log = exact_log(ijk, X, digits, [(origin, n)])
    pos = X+np.where(digits == 2, gap*n, 0.)
    return ijk, X, n, origin, digits, log, pos


# ----------------------------------------------------------------------------- v1 path untouched

@pytest.mark.parametrize("case", ["uncut", "axis_cut", "oblique_cut", "affine_shear"])
def test_v2_without_fixes_equals_v1_exactly(case):
    if case == "oblique_cut":
        ijk, X, n, origin, digits, log, pos = oblique_case()
        codes, F = pack_digits(digits), identity_F(len(X))
    else:
        ijk, X, _, _ = box((10, 7, 6))
        log, codes, pos, F = None, np.zeros(len(X), np.int64), X.copy(), identity_F(len(X))
        if case == "axis_cut":
            digits = plane_digits(X, np.array([.0003, 0, 0]), np.array([1., 0, 0]))[:, None]
            log, codes = exact_log(ijk, X, digits, [(np.array([.0003, 0, 0]), np.array([1., 0, 0]))]), pack_digits(digits)
            pos = X+np.where(digits == 2, [.002, 0, 0], 0.)
        if case == "affine_shear":
            A = np.array([[1, .25, 0], [0, 1, .1], [0, 0, 1.]])
            pos, F = X@A.T+[.1, 0, .2], np.tile(A, (len(X), 1, 1))
    v1, v2 = both(ijk, X, fixes=(), cut_events_log=log)
    a, b = v1.extract(pos, codes, F), v2.extract(pos, codes, F)
    assert same(a, b)
    assert np.array_equal(v1.last["vertex_reference"], v2.last["vertex_reference"])
    assert b[2]["surface_profile"] == cv2.PROFILE and a[2]["surface_profile"] == cas.PROFILE


@pytest.mark.parametrize("motion", ["static", "affine", "rotation_frame"])
def test_v2_equals_v1_on_uncut_states_s0_s9_like(motion):
    """No cut, no clamp, no mixed region, no fold, no separation: every fix is inactive and v2 == v1 bit for bit."""
    from scipy.spatial.transform import Rotation
    ijk, X, _, _ = box((9, 6, 5))
    F, pos, rot = identity_F(len(X)), X.copy(), None
    if motion == "affine":
        A = np.diag([1.05, 1.02, .9])@Rotation.from_rotvec([.1, -.2, .3]).as_matrix()
        F, pos = np.tile(A, (len(X), 1, 1)), X@A.T+[.01, .02, .03]
    if motion == "rotation_frame":
        rot = Rotation.from_rotvec([0, 0, 1.1]).as_matrix()
        pos = X@rot.T+[.4, .1, .8]
    v1, v2 = both(ijk, X)
    codes = np.zeros(len(X), np.int64)
    a = v1.extract(pos, codes, F, frame_rotation=rot)
    b = v2.extract(pos, codes, F, frame_rotation=rot)
    assert same(a, b)
    m = b[2]["v2"]
    assert m["M1_constrained_qef"]["constrained"] == 0 and m["M2_hinge"]["moved"] == 0
    assert m["M3_fold_guard"]["flipped_triangles_before"] == 0 and m["M4_unlabelled_separation"]["fallback_particles"] == 0


def test_affine_motion_with_a_cut_is_still_embedded_exactly():
    ijk, X, n, origin, digits, log, _ = oblique_case(normal=(1., .4, .2))
    A, c = np.array([[1, .25, 0], [0, 1, .1], [0, 0, 1.]]), np.array([.1, -.02, .3])
    x = X@A.T+c
    _, v2 = both(ijk, X, cut_events_log=log)
    v, f, info = v2.extract(x, pack_digits(digits), np.tile(A, (len(X), 1, 1)))
    assert np.abs(v-(v2.last["vertex_reference"]@A.T+c)).max() < EXACT
    assert info["v2"]["M3_fold_guard"]["flipped_triangles_before"] == 0


# ----------------------------------------------------------------------------- M1

def test_box_constrained_qef_is_the_exact_constrained_optimum():
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(7)
    R, K = 60, 6
    normals = rng.normal(size=(R, K, 3))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    points = rng.uniform(-.5, 1.5, size=(R, K, 3))
    weights = (rng.random((R, K)) < .8).astype(float)
    weights[:, 0] = 1.
    x, q, qclip, _ = cv2.solve_qef_box(points, normals, weights, truncation=.1)
    assert np.all((x >= 0) & (x <= 1))
    assert np.all(q <= qclip+1e-9)
    # the same truncated QEF solved by an independent bounded least-squares solver
    w = weights[..., None]
    mass = (points*w).sum(1)/weights.sum(1)[:, None]
    A = normals*w
    rhs = (A*(points-mass[:, None])).sum(-1)
    for r in range(R):
        U, Sv, Vt = np.linalg.svd(A[r], full_matrices=False)
        keep = Sv > .1*Sv[0]
        Ak, bk = (U*np.where(keep, Sv, 0))@Vt, U@(keep*(U.T@rhs[r]))
        ref = lsq_linear(Ak, bk, bounds=(-mass[r], 1-mass[r]), method="bvls", tol=1e-14)
        assert q[r] <= 2*ref.cost+1e-9          # lsq_linear cost = 0.5 ||r||^2


def test_m1_keeps_the_v1_solution_inside_the_cell_and_reduces_the_crease_error():
    ijk, X, n, origin, digits, log, pos = oblique_case()
    v1, v2 = both(ijk, X, cut_events_log=log, strict_audit=False)
    _, f1, i1 = v1.extract(pos, pack_digits(digits), identity_F(len(X)))
    _, f2, i2 = v2.extract(pos, pack_digits(digits), identity_F(len(X)))
    assert np.array_equal(f1, f2)                          # topology unchanged
    X1, X2 = v1.last["vertex_reference"], v2.last["vertex_reference"]
    clamped = v1.last["vertex_clamped"]
    assert clamped.any() and np.array_equal(clamped, v2.last["vertex_m1_constrained"])
    assert np.array_equal(X1[~clamped], X2[~clamped])     # bit-identical where v1 was inside the cell
    cut = np.unique(f2[v2.last["face_kind"] == cas.LINK_CUT])
    e1, e2 = np.abs((X1[cut]-origin)@n).max(), np.abs((X2[cut]-origin)@n).max()
    assert e2 < e1
    assert i2["v2"]["M1_constrained_qef"]["constrained"] == i1["qef"]["clamped"]
    for X_ in (X2,):    # stays inside its dual cell
        u = (X_-v2.lattice.cell_origin[v2.last["vertex_cell"]])/S
        assert np.all((u >= -1e-9) & (u <= 1+1e-9))


# ----------------------------------------------------------------------------- M2

def partial_cut(opening=.0015):
    ijk, X, _, _ = box((12, 8, 6))
    origin, normal = np.array([0, .0002, 0]), np.array([0., 1, 0])
    top = ijk[:, 2] >= 3                                    # rows z >= 3 cut, rows below joined (digit 0)
    digits = np.where(top, plane_digits(X, origin, normal), 0)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    pos = X.copy()
    pos[digits[:, 0] == 2] += opening*normal                # the notch opens; the joined hinge stays (hinge links
    #                                                         1.8 s long: below the M4 2 s length, so M4 stays off)
    return ijk, X, origin, normal, digits, log, pos


def test_m2_moves_the_shared_tip_toward_the_joined_hinge_in_plane():
    ijk, X, origin, normal, digits, log, pos = partial_cut()
    v1, v2 = both(ijk, X, cut_events_log=log)
    a = v1.extract(pos, pack_digits(digits), identity_F(len(X)))
    b = v2.extract(pos, pack_digits(digits), identity_F(len(X)))
    assert np.array_equal(a[1], b[1])
    assert b[2]["v2"]["M4_unlabelled_separation"]["fallback_particles"] == 0
    moved = v2.last["vertex_m2_hinge"]
    assert moved.any() and b[2]["v2"]["M2_hinge"]["moved"] == int(moved.sum())
    assert b[2]["v2"]["M2_hinge"]["mixed_without_hinge_corner"] == 0
    X1, X2 = v1.last["vertex_reference"], v2.last["vertex_reference"]
    # in plane: the distance to the Hermite plane is unchanged; deeper (toward the code-0 row z < 3)
    assert np.abs((X2[moved]-origin)@normal-(X1[moved]-origin)@normal).max() < 1e-12
    assert np.all(X2[moved, 2] < X1[moved, 2])
    assert np.array_equal(X1[~moved], X2[~moved])
    # the embedded tip now follows the joined material: no longer the side-1 / side-2 average across the notch
    tip_y1 = (a[0][moved]-origin)@normal
    tip_y2 = (b[0][moved]-origin)@normal
    assert np.abs(tip_y2).max() < np.abs(tip_y1).min()
    # one shared tip vertex per region, closed slit: S4 provenance semantics kept
    cut = v2.last["face_kind"] == cas.LINK_CUT
    side_digit = digits[v2.last["face_side_particle"], 0]
    shared = np.intersect1d(b[1][cut & (side_digit == 1)], b[1][cut & (side_digit == 2)])
    assert len(shared) and np.all(moved[shared])
    assert b[2]["mesh_validity_reference"]["boundary_edges"] == 0
    assert b[2]["faces"]["cut_triangles"] == a[2]["faces"]["cut_triangles"] == 4*len(log)


def test_m2_does_nothing_on_a_closed_hold_state():
    """A held, uncut-front state (no mixed region) is unchanged; a repeated extraction at the same state is identical."""
    ijk, X, origin, normal, digits, log, pos = partial_cut(opening=0.)
    _, v2 = both(ijk, X, cut_events_log=log)
    a = v2.extract(pos, pack_digits(digits), identity_F(len(X)))
    b = v2.extract(pos, pack_digits(digits), identity_F(len(X)))
    assert same(a, b)
    assert np.array_equal(a[0], b[0])
    assert np.abs(a[0]-v2.last["vertex_reference"]).max() < EXACT   # zero opening: the mesh is the reference mesh


# ----------------------------------------------------------------------------- M3

def test_m3_fold_guard_detects_and_repairs_an_F_extrapolation_fold():
    """Positions unmoved, but the top layer's F stretches z 6x (F inconsistent with the positions, as near the cut):
    total-F extrapolation folds the side walls; the guard re-embeds those representatives with lsq_affine, which is
    exact here because the positions are affine."""
    ijk, X, _, _ = box((8, 6, 6))
    F = identity_F(len(X))
    F[ijk[:, 2] == 5] = np.diag([1., 1., 6.])
    v1, v2 = both(ijk, X)
    codes = np.zeros(len(X), np.int64)
    a = v1.extract(X, codes, F)
    b = v2.extract(X, codes, F)
    g = b[2]["v2"]["M3_fold_guard"]
    assert g["flipped_triangles_before"] > 0 and g["representatives_lsq"] > 0
    assert g["flipped_triangles_after"] == 0
    stage = v2.last["vertex_fold_guard_stage"]
    assert np.all(stage[stage > 0] == 1)
    assert np.abs(b[0][stage == 1]-v2.last["vertex_reference"][stage == 1]).max() < EXACT
    assert np.array_equal(b[0][stage == 0], a[0][stage == 0])
    assert np.array_equal(v2.last["vertices_before_fold_guard"], a[0])


def test_m3_never_triggers_where_det_F_is_nonpositive():
    ijk, X, _, _ = box((8, 6, 6))
    F = identity_F(len(X))
    F[ijk[:, 2] == 5] = np.diag([1., 1., -6.])
    _, v2 = both(ijk, X)
    _, _, info = v2.extract(X, np.zeros(len(X), np.int64), F)
    g = info["v2"]["M3_fold_guard"]
    assert g["representatives_lsq"] == 0 and g["inversion_skipped_triangles"] > 0
    assert info["embedding"]["det_F_nonpositive_particles"] == int((ijk[:, 2] == 5).sum())


# ----------------------------------------------------------------------------- M4

def test_m4_flags_an_unlabelled_tear_and_draws_labelled_particle_cells():
    ijk, X, _, _ = box((8, 6, 5))
    pos = X.copy()
    corner = int(np.flatnonzero((ijk == [7, 5, 4]).all(1))[0])
    pos[corner] += [.005, .004, .003]                       # moved ~7 s without F or codes recording it
    masses = np.full(len(X), 1500*S**3)
    v1, v2 = both(ijk, X, masses_kg=masses, density_kg_m3=1500., strict_audit=False)
    a = v1.extract(pos, np.zeros(len(X), np.int64), identity_F(len(X)))
    v, f, info = v2.extract(pos, np.zeros(len(X), np.int64), identity_F(len(X)))
    m4 = info["v2"]["M4_unlabelled_separation"]
    # both endpoints of each of the 3 flagged links (no guess which side moved): the corner and its 3 neighbours
    assert m4["flagged_links"] == 3 and m4["fallback_particles"] == 4 and m4["label"] == cv2.FALLBACK_LABEL
    fb = v2.last["particle_cell_fallback"]
    assert fb[corner] and fb.sum() == 4
    cell = v2.last["face_particle_cell"]
    assert cell.sum() == 4*12 and info["faces"]["cut_triangles"] == 0
    mv = info["mesh_validity_reference"]
    assert mv["boundary_edges"] == 0 and mv["non_manifold_edges"] == 0 and mv["inconsistently_oriented_edges"] == 0
    assert info["audit"]["mesh_shells"] == 1+4 and info["audit"]["ok"]
    # reference volume exact (cavity + reference cube); the cell's current volume is (m/rho) det F by construction
    assert abs(cas.signed_volume(v2.last["vertex_reference"], f)-len(X)*S**3) < 1e-15
    assert abs(cas.signed_volume(v, f[cell])-4*S**3) < 1e-15
    centres = v[f[cell]].reshape(4, 12*3, 3).mean(1)
    assert np.abs(centres-pos[np.flatnonzero(fb)]).max() < EXACT
    # v1 drew a membrane: its stretched triangles span the tear
    assert a[2]["audit"]["mesh_shells"] == 1


def test_m4_does_not_flag_a_stretch_recorded_by_F():
    ijk, X, _, _ = box((8, 6, 5))
    A = np.diag([3., 1., 1.])
    v1, v2 = both(ijk, X)
    a = v1.extract(X@A.T, np.zeros(len(X), np.int64), np.tile(A, (len(X), 1, 1)))
    b = v2.extract(X@A.T, np.zeros(len(X), np.int64), np.tile(A, (len(X), 1, 1)))
    m4 = b[2]["v2"]["M4_unlabelled_separation"]
    assert m4["fallback_particles"] == 0 and m4["long_links_consistent_with_F_not_flagged"] > 0
    assert same(a, b)


def test_m4_keeps_the_cut_face_on_the_kept_side():
    ijk, X, _, _ = box((10, 6, 5))
    origin, normal = np.array([.0003, 0, 0]), np.array([1., 0, 0])
    digits = plane_digits(X, origin, normal)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    pos = X+np.where(digits == 2, [.002, 0, 0], 0.)
    torn = int(np.flatnonzero((ijk == [5, 5, 4]).all(1))[0])   # side 2, on the cut face, top corner
    assert digits[torn, 0] == 2
    pos[torn] += [.004, .004, .004]
    v1, v2 = both(ijk, X, cut_events_log=log, strict_audit=False)
    a = v1.extract(pos, pack_digits(digits), identity_F(len(X)))
    b = v2.extract(pos, pack_digits(digits), identity_F(len(X)))
    cut1 = lambda surf: int(((surf.last["face_kind"] == cas.LINK_CUT)  # noqa: E731
                             & (digits[surf.last["face_side_particle"], 0] == 1)).sum())
    assert cut1(v2) == cut1(v1)                              # side-1 cut faces all kept, still labelled cut
    fb = v2.last["particle_cell_fallback"]
    n_fb = b[2]["v2"]["M4_unlabelled_separation"]["cut_links_with_fallback_endpoint"]
    assert fb[torn] and n_fb == int((fb & (ijk[:, 0] == 5)).sum()) >= 1   # fallback sites on the side-2 cut face
    assert b[2]["faces"]["cut_triangles"] == a[2]["faces"]["cut_triangles"]-2*n_fb   # only their own side quads
    mv = b[2]["mesh_validity_reference"]
    assert mv["boundary_edges"] == 0 and mv["non_manifold_edges"] == 0


def test_m4_is_disabled_and_counted_without_F():
    ijk, X, _, _ = box((6, 5, 4))
    pos = X.copy()
    pos[0] -= [.005, .005, .005]
    _, v2 = both(ijk, X)
    _, _, info = v2.extract(pos, np.zeros(len(X), np.int64), None)
    m4 = info["v2"]["M4_unlabelled_separation"]
    assert m4["evaluated"] is False and m4["fallback_particles"] == 0


# ----------------------------------------------------------------------------- M5 (labelled variant)

def test_m5_position_consistent_is_a_separate_label_and_follows_the_positions():
    ijk, X, _, _ = box((8, 6, 5))
    F = np.tile(1.3*np.eye(3), (len(X), 1, 1))              # det F 2.2 while the positions do not move
    pc = cv2.CutAwareSurfaceV2(partition(X), S, ijk, X, embedding="position_consistent_v1")
    v, f, info = pc.extract(X, np.zeros(len(X), np.int64), F)
    assert info["surface_profile"] == cv2.PROFILE_POSITION_CONSISTENT
    m5 = info["v2"]["M5_position_consistent"]
    assert m5["particles_lsq"] == m5["particles"]
    assert np.abs(v-pc.last["vertex_reference"]).max() < EXACT
    # affine (consistent) motion: no LSQ term
    A = np.diag([1.05, 1., .95])
    v, _, info = pc.extract(X@A.T, np.zeros(len(X), np.int64), np.tile(A, (len(X), 1, 1)))
    assert info["v2"]["M5_position_consistent"]["particles_lsq"] == 0
    with pytest.raises(ValueError):
        cv2.CutAwareSurfaceV2(partition(X), S, ijk, X, embedding="smooth")


# ----------------------------------------------------------------------------- general

@pytest.mark.parametrize("use_log", [True, False])
def test_v2_incremental_update_equals_full_rebuild(use_log):
    ijk, X, _, _ = box((10, 6, 5))
    origin, normal = np.array([0, .0002, 0]), np.array([0., 1, 0])
    surf = cv2.CutAwareSurfaceV2(partition(X), S, ijk, X)
    x = X+.0002*np.sin(X*900)                               # non-affine current positions
    modes = []
    for codes, log in progressive_frames(ijk, X, origin, normal, (0, 2, 5, 5, 8, 10)):
        log = log if use_log else None
        inc = surf.extract(x, codes, identity_F(len(X)), cut_events_log=log)
        full = cv2.CutAwareSurfaceV2(partition(X), S, ijk, X).extract(x, codes, identity_F(len(X)), cut_events_log=log)
        assert same(inc, full)
        assert inc[2]["v2"]["M2_hinge"] == full[2]["v2"]["M2_hinge"] and inc[2]["qef"] == full[2]["qef"]
        modes.append(inc[2]["update"])
    assert modes[0]["mode"] == "full" and all(m["mode"] == "incremental" for m in modes[1:])
    assert modes[3]["dirty_cells"] == 0


def test_v2_inputs_are_never_modified():
    ijk, X, origin, normal, digits, log, pos = partial_cut()
    pos[3] += [.006, 0, 0]
    codes, F = pack_digits(digits), identity_F(len(X))
    copies = [a.copy() for a in (ijk, X, pos, codes, F)]
    log_copy = log.copy()
    surf = cv2.CutAwareSurfaceV2(partition(X), S, ijk, X, cut_events_log=log, strict_audit=False)
    surf.extract(pos, codes, F)
    for a, b in zip((ijk, X, pos, codes, F), copies):
        assert np.array_equal(a, b)
    assert np.array_equal(log, log_copy)


def test_v2_metadata_records_every_fix_and_the_frozen_parameters():
    ijk, X, origin, normal, digits, log, pos = partial_cut()
    surf = cv2.CutAwareSurfaceV2(partition(X), S, ijk, X, cut_events_log=log)
    _, _, info = surf.extract(pos, pack_digits(digits), identity_F(len(X)))
    m = info["v2"]
    for key in ("M1_constrained_qef", "M2_hinge", "M3_fold_guard", "M4_unlabelled_separation", "M5_position_consistent"):
        assert key in m
    p = m["frozen_parameters"]
    assert p["M4_unlabelled_separation"]["length_s"] == 2.0 and p["M4_unlabelled_separation"]["inconsistency_s"] == 1.0
    assert p["M2_hinge"]["fraction"] == 1.0 and info["writes_physics_state"] is False
    assert info["profile_version"] == "cut_aware_surface_v2" and info["v1_module_profile"] == "cut_aware_surface_v1"
