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


# --- WP-11 Phase E live_003: tiny unmeshable fragments are skipped from the presentation mesh only ---------------

def _block(n, origin, spacing=.001):
    grid = np.stack(np.meshgrid(*[np.arange(n) * spacing] * 3, indexing="ij"), -1).reshape(-1, 3)
    return (grid + np.asarray(origin)).astype(np.float32)


class _FixedPartition:
    """Stand-in mechanical partition: CuttingSurface reads only ``nodes`` (length) and ``partition``."""
    def __init__(self, labels):
        self.labels = np.asarray(labels, np.int64)
        self.nodes = np.zeros((len(labels), 8), np.int64)

    def partition(self, codes):
        return int(self.labels.max()) + 1, self.labels.copy(), {"codes_sha256": "fixture", "partition_rule": "fixture"}


def _digest(vertices, faces, info):
    import hashlib
    import json
    return hashlib.sha256(np.ascontiguousarray(vertices).tobytes() + np.ascontiguousarray(faces).tobytes()
                          + json.dumps(info, sort_keys=True).encode()).hexdigest()


def _extract(positions, labels, profile):
    pytest.importorskip("newton")
    from robotic_cutting_sim.cutting_surface import CuttingSurface
    surface = CuttingSurface(_FixedPartition(labels), .001, .0005, "cpu", profile=profile)
    F = np.tile(np.eye(3, dtype=np.float32), (len(positions), 1, 1)) if profile == "current_volume_v1" else None
    return surface.extract(positions, np.zeros(len(positions), np.int64), F)


# sha256 of (vertices, faces, diagnostics) from the pre-fix extractor (commit dcf393e) on the same input, CPU warp.
_PRE_FIX_DIGEST = {"legacy": "a2dc7cc71f95ca9d813883cf8b31fabf33c2b56bc1e2ffdf9208213ce6f04aab",
                   "current_volume_v1": "19e7c56b34c9f9eb171567f7bb5ff56eb8dbfa64888d6018bacf218f52e1aa5c"}


@pytest.mark.parametrize("profile", ["legacy", "current_volume_v1"])
def test_all_meshable_components_are_byte_identical_to_pre_fix(profile):
    positions = np.vstack([_block(6, (0, 0, 0)), _block(4, (.02, 0, 0))])
    labels = np.r_[np.zeros(216, int), np.ones(64, int)]
    vertices, faces, info = _extract(positions, labels, profile)
    assert "unmeshable_skipped" not in info
    assert [row["mechanical_component"] for row in info["component_ranges"]] == [0, 1]
    assert _digest(vertices, faces, info) == _PRE_FIX_DIGEST[profile]


def test_tiny_fragments_are_skipped_and_recorded_not_raised():
    # current_volume_v1 is the configured-task (live) profile; its iso threshold leaves loose particles without a
    # surface (the legacy profile meshes them, so nothing is skipped there).
    profile = "current_volume_v1"
    substrate, chip = _block(6, (0, 0, 0)), _block(4, (.02, 0, 0))
    # Live_002 failure shape: an 11-particle sparse fragment, plus a single loose particle.
    sparse = (_block(3, (0, 0, 0))[:11] * np.float32(5) + np.float32([0, .03, 0])).astype(np.float32)
    single = np.float32([[.04, .04, .04]])
    positions = np.vstack([substrate, sparse, chip, single])
    labels = np.r_[np.zeros(216, int), np.full(11, 1), np.full(64, 2), [3]]
    vertices, faces, info = _extract(positions, labels, profile)
    assert info["mechanical_components"] == 4   # the physical count is reported unchanged
    assert [row["mechanical_component"] for row in info["component_ranges"]] == [0, 2]
    skipped = info["unmeshable_skipped"]
    assert skipped["fragments"] == 2 and skipped["particles"] == 12
    assert skipped["components"] == [{"mechanical_component": 1, "particles": 11},
                                     {"mechanical_component": 3, "particles": 1}]
    # The meshable components' surfaces are exactly those of the all-meshable input (no merge, no change).
    reference_vertices, reference_faces, _ = _extract(np.vstack([substrate, chip]),
                                                      np.r_[np.zeros(216, int), np.ones(64, int)], profile)
    assert np.array_equal(vertices, reference_vertices) and np.array_equal(faces, reference_faces)


def test_no_meshable_component_still_raises():
    with pytest.raises(RuntimeError, match="no mesh for any"):
        _extract(np.float32([[0, 0, 0], [.03, .03, .03]]), np.array([0, 1]), "current_volume_v1")
