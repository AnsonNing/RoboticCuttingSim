import numpy as np
import pytest

pytest.importorskip("warp")
from scripts.describe_represented_mesh_distance import describe_meshes
from scripts.probe_flatbar_heightfield import analytic_closed_box


def test_exact_mesh_self_and_known_translation_keep_all_samples_and_volume():
    vertices, faces = analytic_closed_box([.16, .03, .02])
    self_result, self_samples = describe_meshes(vertices, faces, vertices, faces)
    assert self_result["actual_to_target"]["maximum_all_sampled_distance_m"] < 1e-7
    assert self_result["actual_to_target"]["all_referenced_vertices"] == 8
    assert len(self_samples["actual_to_target_triangle_centroid_distance_m"]) == 12
    result, _ = describe_meshes(vertices+np.array([0, 0, .0015]), faces, vertices, faces)
    for direction in ["actual_to_target", "target_to_actual"]:
        assert np.isclose(result[direction]["maximum_all_sampled_distance_m"], .0015, atol=1e-7)
    assert result["volume_comparison_has_closed_oriented_inputs"]
    assert result["absolute_volume_difference_fraction_of_fixed_target"] < 1e-6
    assert result["new_pass_thresholds"] is None
