import numpy as np
import pytest
from scipy.optimize import lsq_linear

from robotic_cutting_sim.affine_surface_controls import affine_box_sdf,current_volume_radii,affine_mesh_measurements


def test_current_volume_radius_uses_total_jacobian_and_rejects_inversion():
    matrices=np.array([np.diag([1,1,.8]),np.diag([1,1,1.2])])
    radius=current_volume_radii(.001,matrices)
    np.testing.assert_allclose(8*radius**3,.001**3*np.array([.8,1.2]),rtol=1e-12)
    with pytest.raises(ValueError,match="inverted"):
        current_volume_radii(.001,np.diag([1,1,-1]))


def test_sheared_box_sdf_matches_independent_bounded_least_squares():
    case={"dimensions_m":[2.,2.,2.],"euler_xyz_rad":[0,0,0],"translation_m":[0,0,0],
          "affine_local":[[1.,0,.25],[0,1.,0],[0,0,.8]]}
    points=np.array([[1.8,.2,.9],[-1.4,.3,-1.],[.3,1.6,.1],[0,0,0]])
    result=affine_box_sdf(points,case)
    matrix=np.asarray(case["affine_local"])
    for i,point in enumerate(points[:3]):
        solved=lsq_linear(matrix,point,bounds=(-np.ones(3),np.ones(3)),tol=1e-13)
        assert np.isclose(result[i],np.linalg.norm(matrix@solved.x-point),atol=1e-10)
    assert np.isclose(result[-1],-.8)


def test_affine_closed_mesh_volume_matches_current_mass_density():
    case={"dimensions_m":[2.,2.,2.],"euler_xyz_rad":[0,0,0],"translation_m":[0,0,0],
          "affine_local":[[1.,0,.25],[0,1.,0],[0,0,.8]]}
    reference=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    vertices=reference@np.asarray(case["affine_local"]).T
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    result=affine_mesh_measurements(vertices,faces,case,.1,np.float64(8.))
    assert result["closed_oriented_manifold"]
    assert np.isclose(result["analytic_volume_m3"],6.4)
    assert np.isclose(result["mass_over_density_m3"],6.4)
    assert result["relative_volume_error_vs_mass"]<1e-12
    assert result["full_vertex_sdf"]["max_absolute_m"]<1e-12
