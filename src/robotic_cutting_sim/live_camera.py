"""Live ideal RGB-D of material, tray, table and actual kinematic blade pose."""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from .camera import calibration,capture


def box_mesh(center,half,quat=(0,0,0,1)):
    corners=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)*half
    corners=Rotation.from_quat(quat).apply(corners)+center
    triangles=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                        [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],dtype=np.int32)
    return corners.astype(np.float32),triangles


class LiveCameras:
    def __init__(self,spacing,device,fixture_boxes):
        from newton.geometry import ParticleSurface
        self.device=device
        self.calibration={"ideal_0":calibration([0,-.5,.55]),
                          "ideal_1":calibration([.5,-.4,.25]),
                          "global":calibration([.8,-.8,.8],hfov_deg=65)}
        self.surface=ParticleSurface(voxel_size=min(spacing,.0015),smooth_lambda=0,
                                     mesh_smooth_iterations=0,device=device)
        self.boxes=[(*box,[.35,.38,.42]) for box in fixture_boxes]
        self.boxes.append(([0,0,-.03],[.4,.3,.02],[.28,.31,.35]))

    def capture(self,positions,radii,tool_pose,tool_half):
        with wp.ScopedDevice(self.device):
            vertices,indices,_=self.surface.extract(positions,radii)
            if vertices is None or indices is None:
                raise RuntimeError("Material surface extraction returned empty mesh")
            chunks=[vertices.numpy()]; faces=[indices.numpy().reshape(-1,3)]
            colors=[np.tile([.72,.32,.13],(len(faces[0]),1))]
            offset=len(chunks[0])
            boxes=self.boxes+[(tool_pose[:3],tool_half,[.38,.65,.85])]
            for j,(center,half,color) in enumerate(boxes):
                v,i=box_mesh(center,half,tool_pose[3:] if j==len(boxes)-1 else (0,0,0,1))
                chunks.append(v); faces.append(i+offset); offset+=len(v)
                colors.append(np.tile(color,(len(i),1)))
            verts=wp.array(np.vstack(chunks),dtype=wp.vec3,device=self.device)
            tris=wp.array(np.vstack(faces).ravel(),dtype=int,device=self.device)
            face_colors=wp.array(np.vstack(colors),dtype=wp.vec3,device=self.device)
            result={}
            for name,calib in self.calibration.items():
                rgb,depth=capture(verts,tris,calib,self.device,face_colors)
                result[name]={"rgb":rgb,"depth_m":depth,"valid_mask":depth>0}
            return result
