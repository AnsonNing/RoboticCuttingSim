"""Analytic closed-volume controls, independent of cutting/task geometry."""
import numpy as np
from scipy.spatial.transform import Rotation


def box_particles(case, spacing, density):
    dimensions = np.asarray(case["dimensions_m"], float)
    counts = np.rint(dimensions / spacing).astype(int)
    if not np.allclose(counts * spacing, dimensions, atol=1e-10, rtol=0):
        raise ValueError("Closed-box controls require integer full quadrature cells")
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in counts), indexing="ij"), -1).reshape(-1, 3)
    local = (ijk+.5)*spacing - dimensions/2
    rotation = Rotation.from_euler("xyz", case["euler_xyz_rad"]).as_matrix()
    positions = local @ rotation.T + case["translation_m"]
    masses = np.full(len(positions), density*spacing**3)
    return positions.astype(np.float32), masses


def box_local(positions, case):
    rotation = Rotation.from_euler("xyz", case["euler_xyz_rad"]).as_matrix()
    return (np.asarray(positions, float) - case["translation_m"]) @ rotation


def box_sdf(positions, case):
    delta = abs(box_local(positions, case)) - np.asarray(case["dimensions_m"])/2
    return np.linalg.norm(np.maximum(delta,0),axis=-1) + np.minimum(delta.max(-1),0)


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(quantile*cumulative[-1], cumulative, values[order]))


def mesh_measurements(vertices, triangles, case, spacing, mass_over_density):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    v, f = np.asarray(vertices, float), np.asarray(triangles, int)
    tri = v[f]
    signed_volume = float(np.einsum("ij,ij->i", tri[:,0], np.cross(tri[:,1],tri[:,2])).sum()/6)
    areas = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
    edges = np.concatenate([f[:,[0,1]], f[:,[1,2]], f[:,[2,0]]])
    sorted_edges = np.sort(edges,axis=1)
    unique, inverse, counts = np.unique(sorted_edges,axis=0,return_inverse=True,return_counts=True)
    signs = np.where(edges[:,0] < edges[:,1],1,-1)
    balance = np.bincount(inverse,weights=signs,minlength=len(unique))
    graph = coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(v),len(v))).tocsr()
    components = connected_components(graph,directed=False,return_labels=False)
    sdf = box_sdf(v,case)
    centroids_sdf = box_sdf(tri.mean(1),case)
    local = box_local(v,case)
    half = np.asarray(case["dimensions_m"])/2
    faces = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        central = np.all(abs(local[:,other]) < half[other]-3*spacing,axis=1)
        for sign in (-1,1):
            mask = central & (local[:,axis]*sign > half[axis]-2*spacing)
            errors = local[mask,axis]*sign-half[axis]
            faces.append({"axis":axis,"sign":sign,"samples":int(mask.sum()),
                "signed_mean_m":float(errors.mean()) if errors.size else None,
                "max_absolute_m":float(max(abs(errors))) if errors.size else None})
    analytic_volume = float(np.prod(case["dimensions_m"]))
    return {"vertices":len(v),"triangles":len(f),"finite":bool(np.isfinite(v).all()),
        "analytic_volume_m3":analytic_volume,"mass_over_density_m3":float(mass_over_density),
        "oriented_volume_m3":signed_volume,"mesh_volume_m3":abs(signed_volume),
        "relative_volume_error_vs_analytic":float(abs(abs(signed_volume)-analytic_volume)/analytic_volume),
        "relative_volume_error_vs_mass":float(abs(abs(signed_volume)-mass_over_density)/mass_over_density),
        "closed_oriented_manifold":bool(np.all(counts == 2) and np.all(balance == 0)),
        "boundary_edges":int(np.sum(counts == 1)),"nonmanifold_edges":int(np.sum(counts > 2)),
        "mesh_components":int(components),"euler_characteristic":len(v)-len(unique)+len(f),
        "full_vertex_sdf": {"signed_mean_m":float(sdf.mean()),"p95_absolute_m":float(np.quantile(abs(sdf),.95)),
            "max_absolute_m":float(max(abs(sdf)))},
        "full_area_sdf": {"area_m2":float(areas.sum()),
            "signed_mean_m":float(np.sum(centroids_sdf*areas)/areas.sum()),
            "rms_m":float(np.sqrt(np.sum(centroids_sdf**2*areas)/areas.sum())),
            "p95_absolute_m":weighted_quantile(abs(centroids_sdf),areas,.95)},
        "central_faces":faces,"physical_box_local_half_extents_m":half.tolist(),
        "mesh_local_extents_m":[local.min(0).tolist(),local.max(0).tolist()]}
