"""Current-volume and exact affine-box geometry controls, independent of cuts."""
import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from .surface_controls import box_particles, mesh_measurements, weighted_quantile


def affine_matrix(case):
    return Rotation.from_euler("xyz",case["euler_xyz_rad"]).as_matrix() @ np.asarray(case["affine_local"],float)


def affine_particles(case,spacing,density):
    reference,mass=box_particles({**case,"euler_xyz_rad":[0,0,0],"translation_m":[0,0,0]},spacing,density)
    matrix=affine_matrix(case)
    return (reference.astype(float)@matrix.T+case["translation_m"]).astype(np.float32),mass,matrix


def current_volume_radii(spacing,total_F):
    deformation=np.asarray(total_F,dtype=float)
    if deformation.shape[-2:]!=(3,3) or not np.isfinite(deformation).all():
        raise ValueError("Current-volume radii require finite total deformation gradients")
    jacobian=np.linalg.det(deformation)
    if not np.isfinite(jacobian).all() or np.any(jacobian<=0):
        raise ValueError("Current-volume radii reject inverted/nonfinite deformation instead of clamping")
    return (spacing/2)*np.cbrt(jacobian)


def tlmpm_total_F_receipt(device="cuda:0"):
    """Exercise actual deformation/plasticity kernels on independent arrays."""
    import warp as wp
    from .tlmpm import _deformation,_stress
    target=np.diag([1.1,.9,.8]).astype(np.float32)
    corners=np.array([[(c>>i)&1 for i in range(3)] for c in range(8)],np.float32)
    gradients=(2*corners-1)*.25
    velocity=corners@(target-np.eye(3,dtype=np.float32)).T
    F=wp.array(np.eye(3,dtype=np.float32)[None],dtype=wp.mat33,device=device)
    Fpi=wp.array(np.eye(3,dtype=np.float32)[None],dtype=wp.mat33,device=device)
    stress=wp.zeros(1,dtype=wp.mat33,device=device)
    plastic=wp.zeros(1,dtype=float,device=device)
    wp.launch(_deformation,dim=1,inputs=[F,wp.zeros(1,dtype=int,device=device),
        wp.array(np.arange(8,dtype=np.int32)[None],dtype=int,device=device),
        wp.array(gradients[None],dtype=wp.vec3,device=device),wp.array(velocity,dtype=wp.vec3,device=device),8,1.],device=device)
    before=F.numpy().copy()
    wp.launch(_stress,dim=1,inputs=[F,Fpi,stress,plastic,28846.153846,19230.769231,1.],device=device)
    after,inverse_plastic=F.numpy().copy(),Fpi.numpy().copy()
    result={"semantics":"F is total reference-to-current deformation; Fe=F*Fpi with separately stored inverse plastic part",
        "expected_affine_F":target.tolist(),"measured_F":after[0].tolist(),
        "total_F_error_max":float(abs(after[0]-target).max()),
        "plastic_return_total_F_change_max":float(abs(after-before).max()),
        "inverse_plastic_change_max":float(abs(inverse_plastic[0]-np.eye(3)).max()),
        "accumulated_plastic_strain":float(plastic.numpy()[0]),"det_total_F":float(np.linalg.det(after[0])),
        "det_elastic_F":float(np.linalg.det(after[0]@inverse_plastic[0])),
        "device":str(device),"scope":"Actual TLMPM kernels on one independent affine reference cell; no cutting dynamics"}
    if result["total_F_error_max"]>1e-6 or result["plastic_return_total_F_change_max"]!=0 or result["inverse_plastic_change_max"]<1e-4:
        raise AssertionError("Total-F/plastic-factor semantics control failed")
    return result


def affine_box_sdf(positions,case):
    """Exact Euclidean signed distance to an affine image of a reference box.

    Exterior closest points enumerate face/edge/vertex active sets of the
    box-constrained least-squares problem. Interior distance is the smallest
    physical plane distance. Inverse-affine coordinate norm is NOT an SDF.
    """
    matrix=affine_matrix(case)
    inverse=np.linalg.inv(matrix)
    delta=np.asarray(positions,float)-case["translation_m"]
    local=delta@inverse.T
    half=np.asarray(case["dimensions_m"])/2
    outside=np.any(abs(local)>half,axis=-1)
    result=-np.min((half-abs(local))/np.linalg.norm(inverse,axis=1),axis=-1)
    target=delta[outside]
    if not len(target):
        return result
    best=np.full(len(target),np.inf)
    for state in itertools.product((-1,0,1),repeat=3):
        active=np.flatnonzero(state)
        if not len(active):
            continue
        free=np.flatnonzero(np.asarray(state)==0)
        fixed=np.asarray(state)[active]*half[active]
        residual=target-(matrix[:,active]@fixed)
        if len(free):
            solved=residual@np.linalg.pinv(matrix[:,free]).T
            admissible=np.all(abs(solved)<=half[free]+1e-12,axis=1)
            residual=residual-solved@matrix[:,free].T
        else:
            admissible=np.ones(len(target),bool)
        distance=np.linalg.norm(residual,axis=1)
        best=np.minimum(best,np.where(admissible,distance,np.inf))
    result[outside]=best
    return result


def affine_mesh_measurements(vertices,triangles,case,spacing,reference_volume):
    # Reuse physical signed-volume/topology checks. All analytic distance/face
    # fields are replaced below using the actual sheared/compressed shape.
    reference_volume=float(reference_volume)
    metrics=mesh_measurements(vertices,triangles,case,spacing,reference_volume)
    matrix=affine_matrix(case)
    jacobian=float(np.linalg.det(matrix))
    analytic_volume=float(np.prod(case["dimensions_m"])*jacobian)
    current_volume=reference_volume*jacobian
    metrics.update({"analytic_volume_m3":analytic_volume,"mass_over_density_m3":current_volume,
        "reference_volume_m3":reference_volume,"current_volume_from_total_F_m3":current_volume,
        "total_F":matrix.tolist(),"det_total_F":jacobian,
        "relative_volume_error_vs_analytic":abs(metrics["mesh_volume_m3"]-analytic_volume)/analytic_volume,
        "relative_volume_error_vs_mass":abs(metrics["mesh_volume_m3"]-current_volume)/current_volume})
    v=np.asarray(vertices,float);tri=v[triangles]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
    sdf=affine_box_sdf(v,case)
    sdf_centroids=affine_box_sdf(tri.mean(1),case)
    metrics["full_vertex_sdf"]={"signed_mean_m":float(sdf.mean()),"p95_absolute_m":float(np.quantile(abs(sdf),.95)),
        "max_absolute_m":float(max(abs(sdf)))}
    metrics["full_area_sdf"]={"area_m2":float(area.sum()),"signed_mean_m":float(np.sum(sdf_centroids*area)/area.sum()),
        "rms_m":float(np.sqrt(np.sum(sdf_centroids**2*area)/area.sum())),
        "p95_absolute_m":weighted_quantile(abs(sdf_centroids),area,.95)}
    inverse=np.linalg.inv(matrix)
    local=(v-case["translation_m"])@inverse.T
    half=np.asarray(case["dimensions_m"])/2
    faces=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        central=np.all(abs(local[:,other])<half[other]-3*spacing,axis=1)
        for sign in (-1,1):
            selected=central&(local[:,axis]*sign>half[axis]-2*spacing)
            physical_distance=(local[selected,axis]*sign-half[axis])/np.linalg.norm(inverse[axis])
            faces.append({"axis":axis,"sign":sign,"samples":int(selected.sum()),
                "signed_mean_m":float(physical_distance.mean()) if physical_distance.size else None,
                "max_absolute_m":float(max(abs(physical_distance))) if physical_distance.size else None})
    metrics["central_faces"]=faces
    metrics["mesh_local_extents_m"]=[local.min(0).tolist(),local.max(0).tolist()]
    metrics["physical_box_local_half_extents_m"]=half.tolist()
    metrics["boundary_metric"]="Exact Euclidean affine-box SDF, all triangles; reference coordinates used only for fixed face-center diagnostic"
    return metrics
