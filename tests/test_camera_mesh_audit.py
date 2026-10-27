import numpy as np
import warp as wp
from pxr import Usd,UsdGeom

from robotic_cutting_sim.visual_geometry import non_render_purpose_paths
from scripts.audit_live_camera_mesh import reference_rays


def test_instanced_guide_collision_mesh_is_excluded_from_camera_reference():
    stage=Usd.Stage.CreateInMemory()
    source=UsdGeom.Xform.Define(stage,"/Source")
    guide=UsdGeom.Mesh.Define(stage,"/Source/Collision")
    guide.CreatePurposeAttr("guide")
    UsdGeom.Mesh.Define(stage,"/Source/Visual")
    instance=UsdGeom.Xform.Define(stage,"/World/Instance").GetPrim()
    instance.GetReferences().AddInternalReference(str(source.GetPath()))
    instance.SetInstanceable(True)
    assert stage.GetPrimAtPath("/World/Instance/Collision").IsInstanceProxy()
    excluded=non_render_purpose_paths(stage)
    assert "/World/Instance/Collision" in excluded
    assert "/World/Instance/Visual" not in excluded


def test_reference_ray_respects_usd_sidedness_and_optical_near_plane():
    wp.init()
    vertices=wp.array([[-1,-1,1],[1,-1,1],[0,1,1],[-1,-1,1.5],[0,1,1.5],[1,-1,1.5]],dtype=wp.vec3,device="cpu")
    mesh=wp.Mesh(vertices,wp.array([0,1,2,3,4,5],dtype=int,device="cpu"))
    k=wp.mat33(1.,0.,.5,0.,1.,.5,0.,0.,1.)
    for near in (.2,1.1):
        depth=wp.empty((1,1),dtype=float,device="cpu");labels=wp.empty((1,1),dtype=int,device="cpu")
        wp.launch(reference_rays,dim=(1,1),inputs=[mesh.id,wp.mat44(np.eye(4)),k,near,2.,
            wp.array([3,4],dtype=int,device="cpu"),wp.array([1,1],dtype=int,device="cpu"),depth,labels],device="cpu")
        assert np.isclose(depth.numpy()[0,0],1.5)
        assert labels.numpy()[0,0]==4
