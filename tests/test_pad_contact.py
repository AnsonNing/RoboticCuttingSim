"""Independent geometry/units controls, not a physical grasp result."""
import numpy as np
import pytest
from robotic_cutting_sim.pad_contact import rectangular_mean_radius, patch_receipt

def test_quadrature_matches_independent_closed_form_rectangle():
    a, b = .008, .00825
    expected = (2*a*b*np.sqrt(a*a+b*b)+a**3*np.arcsinh(b/a)+b**3*np.arcsinh(a/b))/(6*a*b)
    assert np.isclose(rectangular_mean_radius(a, b), expected, atol=3e-10, rtol=0)

def test_finite_patch_uses_handle_pad_overlap_and_scales_with_friction():
    normal = patch_receipt((.008, .011, .020), .6)
    weak = patch_receipt((.008, .011, .020), .02)
    assert normal["nominal_overlap_half_xz_m"] == [.008, .00825]
    assert np.isclose(normal["torsional_friction_length_m"], .0037305973699669637)
    assert np.isclose(normal["torsional_friction_length_m"]/weak["torsional_friction_length_m"], 30)
    assert normal["torsional_friction_length_m"] < .005
    assert normal["adhesion"] == 0 and not normal["rolling_resistance_enabled"]

def test_patch_radius_scales_with_geometry_and_rejects_nonphysical_inputs():
    assert np.isclose(rectangular_mean_radius(.02, .04), 2*rectangular_mean_radius(.01, .02))
    assert patch_receipt((.008, .011, .020), 0)["torsional_friction_length_m"] == 0
    with pytest.raises(ValueError):rectangular_mean_radius(-1, 1)
    with pytest.raises(ValueError):patch_receipt((.008, .011, .020), np.inf)
