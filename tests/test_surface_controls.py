import numpy as np

from robotic_cutting_sim.surface_controls import box_particles,box_sdf,mesh_measurements


CASE={"dimensions_m":[.048,.036,.018],"euler_xyz_rad":[0,0,0],"translation_m":[0,0,0]}


def test_full_cell_box_mass_boundary_and_rotated_sdf():
    points,masses=box_particles(CASE,.0015,1500.)
    assert np.isclose(masses.sum()/1500.,np.prod(CASE["dimensions_m"]))
    assert points[:,2].max() < .009
    np.testing.assert_allclose(box_sdf([[0,0,.009],[0,0,.010],[0,0,0]],CASE),[0,.001,-.009],atol=1e-15)


def test_signed_box_volume_and_manifold_metrics_detect_missing_face():
    vertices=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    case={**CASE,"dimensions_m":[2,2,2]}
    metrics=mesh_measurements(vertices,faces,case,.1,8.)
    assert metrics["closed_oriented_manifold"]
    assert metrics["mesh_components"]==1 and metrics["euler_characteristic"]==2
    assert np.isclose(metrics["mesh_volume_m3"],8.)
    assert not mesh_measurements(vertices,faces[:-1],case,.1,8.)["closed_oriented_manifold"]
