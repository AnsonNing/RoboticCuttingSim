import numpy as np
import pytest

from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition, combine_component_meshes, wedge_mesh


def stencil(particles):
    nodes = np.tile(np.arange(8), (particles, 1))
    weights = np.full((particles, 8), 1 / 8, np.float32)
    gradients = np.zeros((particles, 8, 3), np.float32)
    return nodes, weights, gradients


def test_partition_starts_connected_and_uses_only_current_history():
    partition = CurrentMechanicalPartition(*stencil(3), cuts=1)
    first, labels, proof = partition.partition(np.array([0, 0, 0]))
    assert first == 1 and np.array_equal(labels, [0, 0, 0])
    assert proof["future_component_ids_used"] is False
    # Code0 still couples both sides: labeling material in the future cannot
    # presegment the present continuous solid or erase the uncut ligament.
    assert partition.partition(np.array([1, 0, 2]))[0] == 1
    count, labels, _ = partition.partition(np.array([1, 1, 2]))
    assert count == 2 and labels[0] == labels[1] and labels[1] != labels[2]
    assert partition.partition(np.array([0, 0, 0]))[0] == 1


def test_nonzero_stress_gradients_keep_zero_weight_bridge():
    nodes, weights, gradients = stencil(2)
    weights[:] = 0
    nodes[1] += 8
    nodes[1, 0] = nodes[0, 0]
    gradients[:, 0, 0] = 1
    partition = CurrentMechanicalPartition(nodes, weights, gradients, cuts=1)
    assert partition.partition(np.array([0, 0]))[0] == 1
    assert partition.partition(np.array([1, 2]))[0] == 2


def test_nearby_components_remain_distinct_meshes_without_welding():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    combined, triangles, ranges = combine_component_meshes([(0, vertices, faces, 10), (1, vertices, faces, 5)])
    assert combined.shape == (6, 3)
    assert np.array_equal(triangles, [[0, 1, 2], [3, 4, 5]])
    assert [r["particles"] for r in ranges] == [10, 5]
    with pytest.raises(ValueError, match="No surface"):
        combine_component_meshes([(0, vertices, np.empty((0, 3), np.int32), 10)])


def test_wedge_matches_physics_edge_frame_and_is_closed():
    vertices, triangles = wedge_mesh()
    assert vertices.shape == (6, 3) and triangles.shape == (8, 3)
    assert np.allclose(vertices[[0, 3]][:, [0, 2]], 0)
    assert np.isclose(vertices[:, 0].min(), -.012)
    assert np.isclose(vertices[:, 2].max(), .004)
    assert np.isclose(vertices[:, 2].min(), -.0004)
    edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    signed_volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert np.isclose(signed_volume, .5 * .012 * .0044 * .05)  # Outward facing watertight prism.
