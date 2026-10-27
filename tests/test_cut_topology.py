import numpy as np

from robotic_cutting_sim.cut_topology import compatibility, conflicting, couple, sweep_labels, mechanical_components


def test_persistent_history_retains_both_sequential_cuts():
    x = np.array([[-.001, 0, -.001], [-.001, 0, .001], [.01, 0, .001]])
    original = np.zeros(3, np.int32)
    first = sweep_labels(x, original, np.zeros(3), np.eye(3), .001)
    assert first.tolist() == [1, 2, 0]
    withdrawn = sweep_labels(x, first, np.array([.1, 0, 0]), np.eye(3), .001)
    np.testing.assert_array_equal(withdrawn, first)
    second = sweep_labels(x, first, np.zeros(3), np.eye(3), .001, cut=1)
    np.testing.assert_array_equal(second % 3, first)
    assert second.tolist() == [4, 8, 0]
    np.testing.assert_array_equal(original, 0)


def test_finite_sweep_has_no_preassigned_far_particle():
    x = np.array([[-.001, .1, .001], [-.001, 0, .1], [-.02, 0, .001], [.001, 0, .001]])
    assert not sweep_labels(x, np.zeros(4, int), np.zeros(3), np.eye(3), .001).any()


def test_ternary_compatibility_accumulates_conflict():
    c = compatibility(2)
    assert c.shape == (9, 9)
    assert np.array_equal(c, c.T)
    assert c[0].all() and np.diag(c).all()
    assert not c[1, 2] and not c[4, 8]
    assert conflicting(4, 7)  # same first cut, opposite second cut
    assert c[1, 4]  # unset second digit remains compatible


def test_compatible_grid_exchange_conserves_momentum_for_all_codes():
    rng = np.random.default_rng(4)
    mass = rng.uniform(.1, 2, 9)
    momentum = rng.normal(size=(9, 3))
    np.testing.assert_allclose(couple(momentum, mass, compatibility()).sum(0), momentum.sum(0), atol=2e-15)
    common = mass[:, None]*[.1, -.4, 2]
    np.testing.assert_allclose(couple(common, mass, compatibility()), common, atol=2e-15)


def test_opposite_faces_have_independent_dofs_after_blade_withdrawal():
    # No unassigned material at this node: opposing sides never exchange momentum.
    mass = np.array([0, 2., 3.])
    momentum = np.array([[0., 0, 0], [1., 0, 0], [-2., 0, 0]])
    np.testing.assert_array_equal(couple(momentum, mass, compatibility(1)), momentum)


def test_mechanical_components_detect_unassigned_stencil_bridge():
    nodes = np.array([[0, 1], [1, 2], [1, 2]])
    weights = np.full((3, 2), .5)
    n, _ = mechanical_components(nodes, weights, np.array([1, 2, 0]), 1)
    assert n == 1  # zero code mediates both sides
    n, component = mechanical_components(nodes, weights, np.array([1, 2, 1]), 1)
    assert n == 2 and component[0] == component[2] != component[1]
