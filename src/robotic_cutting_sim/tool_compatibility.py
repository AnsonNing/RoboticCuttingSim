"""Reject a visual/dynamic blade that differs from the retained contact wedge."""
import numpy as np
from scipy.spatial import cKDTree

from .robot import wedge_mesh
from .scene_frames import pose_matrix, transform_points


def require_matching_blade(spec, hand_from_edge_xyzw, dimensions):
    """Custom handle/mass models may reuse the current blade contact contract.

    A different cutting head needs a matching material-contact implementation
    and task/controller recipe. It must never silently use the old wedge.
    """
    actual_dimensions = [spec.blade_length_m,spec.blade_height_m,spec.blade_half_width_m]
    if not np.allclose(actual_dimensions,dimensions,atol=1e-12,rtol=0):
        raise NotImplementedError("Custom tool blade dimensions require a matching material/task recipe")
    if not np.allclose(spec.T_hand_tool @ spec.T_tool_edge,pose_matrix(hand_from_edge_xyzw),atol=1e-10,rtol=0):
        raise NotImplementedError("Custom tool nominal hand-to-edge frame differs from the IK/task recipe")
    blades = [s for s in spec.shapes if s.role == "blade"]
    expected = transform_points(spec.T_tool_edge,wedge_mesh(*dimensions)[0])
    if len(blades) != 1:
        raise NotImplementedError("The retained TLMPM contact contract requires one matching wedge blade")
    vertices = np.asarray(blades[0].vertices_tool_m)
    if (len(vertices) != len(expected) or cKDTree(vertices).query(expected)[0].max() > 1e-9
            or cKDTree(expected).query(vertices)[0].max() > 1e-9):
        raise NotImplementedError("Dynamic/visible blade geometry differs from the retained TLMPM wedge")
