"""Independent PhysX compound cutting tool; no robot attachment is authored.

Only reset initializes its world transform. Runtime poses/velocities are read
from RigidPrim; all following motion belongs to the rigid contact solver.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .scene_frames import compose_pose, matrix_pose, pose_matrix

TOOL_PATH = '/World/Tools/CuttingTool_0'


def author_tool(stage, spec, world_pose, grasp, finger_paths):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade
    UsdGeom.Xform.Define(stage, '/World/Tools')
    body = UsdGeom.Xform.Define(stage, TOOL_PATH)
    body.AddTranslateOp().Set(Gf.Vec3d(*world_pose[:3]))
    body.AddOrientOp().Set(Gf.Quatf(float(world_pose[6]), Gf.Vec3f(*world_pose[3:6])))
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    rigid.CreateVelocityAttr(Gf.Vec3f(0))
    rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(float(spec.mass_kg))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*spec.com_tool_m))
    moments, axes = np.linalg.eigh(np.asarray(spec.inertia_com_tool_kg_m2))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    q = Rotation.from_matrix(axes).as_quat()
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    native = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    native.CreateSolverPositionIterationCountAttr(32)
    native.CreateSolverVelocityIterationCountAttr(8)
    native.CreateLinearDampingAttr(0.)
    native.CreateAngularDampingAttr(0.)
    material = UsdShade.Material.Define(stage, '/World/Tools/GripMaterial')
    phys = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys.CreateStaticFrictionAttr(float(grasp.friction))
    phys.CreateDynamicFrictionAttr(float(grasp.friction))
    phys.CreateRestitutionAttr(0.)
    for shape in spec.shapes:
        mesh = UsdGeom.Mesh.Define(stage, TOOL_PATH+'/'+shape.name)
        mesh.CreatePointsAttr(np.asarray(shape.vertices_tool_m).tolist())
        faces = np.asarray(shape.triangles)
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*((.12, .13, .16) if shape.role == 'handle' else (.64, .68, .73)))])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        offsets = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        offsets.CreateContactOffsetAttr(.0002)
        offsets.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
    # Match the two actual pad materials to the declared finite coefficient.
    # No finger geometry, joints, tool pose, or constraints are altered here.
    for path in finger_paths:
        from pxr import Usd
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    edge = UsdGeom.Xform.Define(stage, TOOL_PATH+'/CuttingEdge')
    offset = spec.tool_from_edge_xyz_xyzw
    edge.AddTranslateOp().Set(Gf.Vec3d(*offset[:3]))
    edge.AddOrientOp().Set(Gf.Quatf(float(offset[6]), Gf.Vec3f(*offset[3:6])))
    body.GetPrim().SetCustomDataByKey('research:tool_attachment_mode', 'physical_grasp')
    body.GetPrim().SetCustomDataByKey('research:contact_owner', 'PhysX rigid pairs; TLMPM owns only soft-material contact')
    return TOOL_PATH


def nominal_tool_pose(edge_pose, spec):
    return matrix_pose(pose_matrix(edge_pose) @ np.linalg.inv(spec.T_tool_edge))


def tool_state(rigid, spec):
    from .isaac_sim_runner import numpy_data, pose_from_wxyz
    from .physical_tool import measured_edge_pose
    p, q = rigid.get_world_poses()
    pose = pose_from_wxyz(numpy_data(p)[0], numpy_data(q)[0])
    linear_native, angular_native = rigid.get_velocities()
    linear = numpy_data(linear_native)[0]
    angular = numpy_data(angular_native)[0]
    return {'tool_body_pose_xyzw': pose, 'tool_pose': measured_edge_pose(pose, spec),
            'tool_linear_velocity_world_m_s': linear, 'tool_angular_velocity_world_rad_s': angular}
