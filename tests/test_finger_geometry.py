import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.finger_geometry import discover_pads, pad_boxes_edge, pad_plane_clearance


def test_pad_collider_scale_and_actual_body_motion_are_used():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/hand')
    for name in ('left', 'right'):
        body = UsdGeom.Xform.Define(stage, '/hand/' + name)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        cube = UsdGeom.Cube.Define(stage, '/hand/' + name + '/' + name + '_finger_pad')
        cube.CreateSizeAttr(2.)
        cube.AddTranslateOp().Set(Gf.Vec3d(0., 0., .04))
        cube.AddScaleOp().Set(Gf.Vec3f(.01, .005, .008))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    pads = discover_pads(stage, '/hand')
    poses = {'/hand/left': [0., .02, .1, 0., 0., 0., 1.],
             '/hand/right': [0., -.02, .1, 0., 0., 0., 1.]}
    edge = [0., 0., 0., 0., 0., 0., 1.]
    boxes = pad_boxes_edge(pads, poses, edge)
    assert np.allclose(boxes[:, 0], [[0., .02, .14], [0., -.02, .14]])
    assert np.allclose(boxes[:, 1], [.01, .005, .008])
    poses['/hand/left'] = [0., .03, .1, *Rotation.from_euler('z', 90, degrees=True).as_quat()]
    moved = pad_boxes_edge(pads, poses, edge)
    assert np.allclose(moved[0, 0], [0., .03, .14])
    assert np.allclose(moved[0, 1], [.005, .01, .008])
    assert np.allclose(moved[1], boxes[1])
    assert np.allclose(list(pad_plane_clearance(pads, poses, [0., 0., .1], [0., 0., 1.]).values()), .032)


def test_blunt_pad_impulse_has_a_separate_equal_reaction():
    import warp as wp
    from robotic_cutting_sim.tlmpm_v2 import _obstacle_contacts
    wp.init()
    with wp.ScopedDevice('cpu'):
        x = wp.array([[.001, 0., 0.]], dtype=wp.vec3)
        v = wp.zeros(1, dtype=wp.vec3)
        mass = wp.array([.001], dtype=float)
        impulse = wp.zeros(1, dtype=wp.vec3)
        poses = wp.array([[0., 0., 0., 0., 0., 0., 1.]], dtype=wp.transform)
        twists = wp.zeros(1, dtype=wp.spatial_vector)
        index = wp.zeros(1, dtype=int)
        total = wp.zeros(1, dtype=wp.spatial_vector)
        boxes = wp.zeros(2, dtype=wp.vec3)
        halves = wp.array([[.005, .005, .005]] * 2, dtype=wp.vec3)
        separate = wp.zeros(2, dtype=wp.spatial_vector)
        wp.launch(_obstacle_contacts, dim=1, inputs=[x, v, mass, impulse, poses, twists,
            index, total, .0005, .0001, 3000., .3, boxes, halves, 1, separate])
        assert np.linalg.norm(impulse.numpy()) > 0.
        assert np.allclose(total.numpy()[0, :3], -impulse.numpy()[0])
        assert np.array_equal(separate.numpy()[0], np.zeros(6))
        assert np.array_equal(separate.numpy()[1], total.numpy()[0])
