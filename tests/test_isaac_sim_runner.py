import numpy as np
import pytest

from robotic_cutting_sim.isaac_sim_runner import (
    IsaacSimCuttingConfig, calibration_from_replicator, pose_from_wxyz, require_sim_62,
)


def test_live_runner_rejects_older_or_missing_runtime_and_no_ideal_fallback():
    with pytest.raises(RuntimeError,match="6.2.0.0"):
        require_sim_62({"isaacsim":"6.1.0.0"})
    with pytest.raises(ValueError,match="no ideal fallback"):
        IsaacSimCuttingConfig(camera_mode="ideal_rgbd")
    with pytest.raises(ValueError,match="positive"):
        IsaacSimCuttingConfig(record_every=0)


def test_native_wxyz_pose_becomes_project_xyzw_without_changing_translation():
    np.testing.assert_allclose(pose_from_wxyz([1,2,3],[.5,.5,.5,.5]),[1,2,3,.5,.5,.5,.5])
    with pytest.raises(ValueError,match="unit quaternion"):
        pose_from_wxyz([1,2,3],[2,0,0,0])


def test_runtime_projection_calibration_preserves_aspect_and_offsets():
    projection=np.eye(4)
    projection[0,0],projection[1,1]=2.,3.
    projection[2,0],projection[2,1]=.1,-.2
    parameters={"renderProductResolution":[640,480],"cameraProjection":projection.ravel(),
                "cameraViewTransform":np.eye(4).ravel(),"metersPerSceneUnit":1.}
    calibration=calibration_from_replicator(parameters)
    np.testing.assert_allclose(calibration["K"],[[640,0,288],[0,720,192],[0,0,1]])
    np.testing.assert_allclose(calibration["T_world_camera"],np.diag([1,-1,-1,1]))
