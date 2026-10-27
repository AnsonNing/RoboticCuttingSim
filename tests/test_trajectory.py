import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.trajectory import Trajectory,reference_trajectory
from robotic_cutting_sim.recording import new_output


def test_csv_roundtrip_and_known_midpoint(tmp_path):
    t=reference_trajectory("through_bump_cut"); t.save(tmp_path/"t.csv"); loaded=Trajectory.load(tmp_path/"t.csv")
    pose,twist=loaded.at(.8)
    np.testing.assert_allclose(pose[:3],[0,0,.0255],atol=1e-12)
    np.testing.assert_allclose(twist[:3],[.09,0,0],atol=1e-12)
    np.testing.assert_allclose(loaded.at(3)[1],0)


def test_slerp_not_linear_quaternion():
    q=Rotation.from_euler("z",120,degrees=True).as_quat()
    t=Trajectory([[0,0,0,0,0,0,0,1],[2,0,0,0,*q]])
    pose,twist=t.at(1)
    np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_euler("xyz",degrees=True),[0,0,60],atol=1e-8)
    np.testing.assert_allclose(twist[3:],[0,0,np.pi/3],atol=1e-8)


def test_invalid_times_and_quaternions():
    with pytest.raises(ValueError):
        Trajectory([[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1]])
    with pytest.raises(ValueError):
        Trajectory([[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,2]])


def test_preserve_old_experiment(tmp_path):
    out=new_output(tmp_path/"run"); (out/"evidence").write_text("keep")
    with pytest.raises(FileExistsError):
        new_output(out)
    assert (out/"evidence").read_text()=="keep"
