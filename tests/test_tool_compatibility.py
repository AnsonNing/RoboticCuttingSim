"""A custom rigid visual must not bypass the material blade contract."""
from dataclasses import replace

import numpy as np
import pytest

from robotic_cutting_sim.physical_tool import make_tool_spec
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.tool_compatibility import require_matching_blade


@pytest.mark.parametrize("task_id",["CUT-01","TRIM-01"])
def test_real_heads_match_their_shared_dynamic_and_material_geometry(task_id):
    t=make_task(task_id);s=make_tool_spec(t)
    require_matching_blade(s,t.hand_from_edge_xyz_xyzw,[t.blade_length_m,t.blade_height_m,t.blade_half_width_m])


def test_different_mass_is_allowed_but_different_contact_geometry_is_rejected():
    t=make_task();s=make_tool_spec(t);dims=[t.blade_length_m,t.blade_height_m,t.blade_half_width_m]
    shapes=tuple(replace(shape,density_kg_m3=shape.density_kg_m3*1.1) for shape in s.shapes)
    require_matching_blade(replace(s,shapes=shapes),t.hand_from_edge_xyz_xyzw,dims)
    with pytest.raises(NotImplementedError,match="dimensions"):
        require_matching_blade(replace(s,blade_length_m=s.blade_length_m*2),t.hand_from_edge_xyz_xyzw,dims)
    vertices=np.array(s.shapes[-1].vertices_tool_m);vertices[:,0]+=.001
    bad_blade=replace(s.shapes[-1],vertices_tool_m=vertices)
    with pytest.raises(NotImplementedError,match="geometry"):
        require_matching_blade(replace(s,shapes=s.shapes[:-1]+(bad_blade,)),t.hand_from_edge_xyz_xyzw,dims)


def test_ik_nominal_frame_cannot_silently_differ_from_the_actual_head():
    t=make_task();s=make_tool_spec(t)
    mount=list(t.hand_from_edge_xyz_xyzw);mount[2]+=.01
    with pytest.raises(NotImplementedError,match="frame"):
        require_matching_blade(s,mount,[t.blade_length_m,t.blade_height_m,t.blade_half_width_m])
