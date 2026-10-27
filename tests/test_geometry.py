import numpy as np
import pytest
from scipy.ndimage import label

from robotic_cutting_sim.geometry import Specimen,sample_particles


@pytest.mark.parametrize("family",["dome","gaussian","ellipsoid_cap","ridge"])
def test_one_connected_volume_and_mass(family):
    spec=Specimen(family=family)
    points,masses,ijk=sample_particles(spec,.002,1500)
    occupied=np.zeros(ijk.max(axis=0)+1,dtype=bool); occupied[tuple(ijk.T)]=True
    _,components=label(occupied)
    assert components==1
    assert np.any(points[:,2]>spec.base_height)
    assert np.all(points[:,2]<=spec.height(points[:,0],points[:,1])+1e-7)
    assert np.isclose(masses.sum(),len(points)*1500*.002**3)
    # Every sampled bump column is connected all the way down to the same base.
    for i,j,k in ijk[points[:,2]>spec.base_height]:
        assert occupied[i,j,:k+1].all()


def test_seed_determinism_and_variation():
    a=Specimen.generate(11,"D1"); b=Specimen.generate(11,"D1"); c=Specimen.generate(12,"D1")
    assert a==b and a!=c
    assert np.array_equal(sample_particles(a,.002,1500)[0],sample_particles(b,.002,1500)[0])


@pytest.mark.parametrize("kwargs",[{"bump_radius":0},{"center_x":.059},{"family":"unknown"},{"base_height":float("nan")}])
def test_illegal_specimen_rejected(kwargs):
    with pytest.raises(ValueError):
        Specimen(**kwargs)
