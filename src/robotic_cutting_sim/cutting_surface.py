"""Surface reconstruction partitioned only by the current mechanical cut graph.

This module visualizes solver state. It does not assign cuts, change masses,
remove particles, or infer a cut from a rendered gap. Future/final component IDs
are deliberately absent from its interface.
"""
import hashlib

import numpy as np

from .cut_topology import mechanical_components, mechanical_components_packed, validate_packed
from .usd_surface_bridge import validate_surface


class CurrentMechanicalPartition:
    def __init__(self, nodes, weights, gradients, cuts, *, encoding="ternary"):
        self.nodes = np.asarray(nodes)
        self.weights = np.asarray(weights)
        self.gradients = np.asarray(gradients)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 8 or self.weights.shape != self.nodes.shape:
            raise ValueError("Expected the solver's eight-node reference stencil")
        if self.gradients.shape != (*self.nodes.shape, 3):
            raise ValueError("Reference stress gradients must accompany the stencil")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.gradients).all():
            raise ValueError("Reference stencil must be finite")
        if encoding not in ("ternary", "packed2"):
            raise ValueError("Cut-code encoding is 'ternary' (v1-v4) or 'packed2' (tlmpm_v5)")
        self.cuts, self.encoding = cuts, encoding
        self.cache = {}
        self.stencil_sha256 = hashlib.sha256(self.nodes.tobytes() + self.weights.tobytes()
                                             + self.gradients.tobytes()).hexdigest()

    def partition(self, current_codes):
        codes = np.asarray(current_codes)
        if codes.shape != (len(self.nodes),) or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("Need exactly one current integer cut-history code per particle")
        if self.encoding == "packed2":
            if not validate_packed(codes, self.cuts):
                raise ValueError("Cut-history code is not a packed code of the solver's cuts")
        elif np.any(codes < 0) or np.any(codes >= 3 ** self.cuts):
            raise ValueError("Cut-history code exceeds the solver field count")
        key = hashlib.sha256(codes.tobytes()).hexdigest()
        if key not in self.cache:
            components = mechanical_components_packed if self.encoding == "packed2" else mechanical_components
            count, labels = components(self.nodes, self.weights, codes, self.cuts, gradients=self.gradients)
            self.cache[key] = count, labels.copy()
        count, labels = self.cache[key]
        return count, labels.copy(), {"codes_sha256": key, "stencil_sha256": self.stencil_sha256,
            "partition_rule": "current compatible reference-node MPM stencils, including nonzero stress gradients",
            "future_component_ids_used": False}


def combine_component_meshes(meshes):
    """Concatenate surfaces without welding vertices across mechanical components."""
    vertices, faces, rows = [], [], []
    vertex_offset, face_offset = 0, 0
    for label, points, triangles, particle_count in meshes:
        points, triangles = validate_surface(points, triangles)
        if not len(triangles):
            raise ValueError(f"No surface triangles for current mechanical component {label}")
        vertices.append(points); faces.append(triangles + vertex_offset)
        rows.append({"mechanical_component": int(label), "particles": int(particle_count),
                     "first_vertex": vertex_offset, "vertices": len(points),
                     "first_triangle": face_offset, "triangles": len(triangles)})
        vertex_offset += len(points); face_offset += len(triangles)
    if not meshes:
        raise ValueError("No current mechanical components to reconstruct")
    return np.vstack(vertices), np.vstack(faces), rows


class CuttingSurface:
    """Newton ParticleSurface per current component, with no intercomponent weld."""
    def __init__(self, partition, spacing_m, voxel_size_m=.001, device="cuda:0", *, profile="legacy"):
        from newton.geometry import ParticleSurface
        self.partition = partition
        self.spacing_m, self.voxel_size_m, self.device = spacing_m, voxel_size_m, device
        if profile not in ("legacy", "current_volume_v1"):
            raise ValueError("Unknown surface profile")
        if profile == "current_volume_v1" and not np.isclose(voxel_size_m, spacing_m/2):
            raise ValueError("Validated current-volume profile requires half-spacing voxels")
        self.profile = profile
        self.surface_type = ParticleSurface
        self.surfaces = {}

    def extract(self, positions_m, current_codes, total_F=None):
        import warp as wp
        positions = np.asarray(positions_m, np.float32)
        if positions.shape != (len(self.partition.nodes), 3) or not np.isfinite(positions).all():
            raise ValueError("Expected current finite particle positions matching the mechanical graph")
        count, labels, provenance = self.partition.partition(current_codes)
        if self.profile == "current_volume_v1":
            from .affine_surface_controls import current_volume_radii
            deformation = np.asarray(total_F)
            if deformation.shape != (len(positions), 3, 3):
                raise ValueError("Current-volume reconstruction needs one total deformation gradient per particle")
            radii_all = current_volume_radii(self.spacing_m, deformation)
            provenance["current_volume_jacobian_min_max"] = [float(np.linalg.det(deformation).min()),
                                                             float(np.linalg.det(deformation).max())]
            provenance["profile_validation_scope"] = "Independent closed affine boxes; J=.8/1.2, shear=.25, held-out rigid rotations; not calibrated fracture surfaces"
        chunks = []
        with wp.ScopedDevice(self.device):
            for label in range(count):
                selected = np.flatnonzero(labels == label)
                key = hashlib.sha256(selected.tobytes()).hexdigest()
                if key not in self.surfaces:
                    profile_args = ({"threshold": .5, "kernel_scale": .5, "kernel_radius": 3*self.spacing_m,
                        "anisotropic": False, "field_smooth_iterations": 0} if self.profile == "current_volume_v1" else {})
                    self.surfaces[key] = self.surface_type(voxel_size=self.voxel_size_m, smooth_lambda=0,
                                           mesh_smooth_iterations=0, device=self.device, **profile_args)
                points = wp.array(positions[selected], dtype=wp.vec3, device=self.device)
                radii = (wp.array(radii_all[selected].astype(np.float32), dtype=float, device=self.device)
                    if self.profile == "current_volume_v1" else
                    wp.full(len(selected), self.spacing_m / 2, dtype=float, device=self.device))
                vertices, indices, _ = self.surfaces[key].extract(points, radii)
                if vertices is None or indices is None:
                    raise RuntimeError(f"ParticleSurface returned no mesh for {len(selected)} physical particles")
                chunks.append((label, vertices.numpy().copy(), indices.numpy().reshape(-1, 3).copy(), len(selected)))
        vertices, faces, ranges = combine_component_meshes(chunks)
        return vertices, faces, {**provenance, "mechanical_components": count, "component_ranges": ranges,
                                 "renderer_partition_changes_physics": False, "surface_profile": self.profile}


class ComponentSurface:
    """Live-bed convenience adapter; reference stencil is copied only once."""
    def __init__(self, spacing, device="cuda:0", voxel_size=.001, *, profile="legacy"):
        self.spacing, self.device, self.voxel_size = spacing, device, voxel_size
        self.profile = profile
        self.surface = None
        self.bed_identity = None

    def extract(self, bed):
        if self.surface is None or self.bed_identity != id(bed):
            partition = CurrentMechanicalPartition(bed.nodes.numpy(), bed.weights.numpy(),
                                                    bed.gradients.numpy(), bed.config.cuts,
                                                    encoding=getattr(bed, "code_encoding", "ternary"))
            self.surface = CuttingSurface(partition, self.spacing, self.voxel_size, self.device, profile=self.profile)
            self.bed_identity = id(bed)
        return self.surface.extract(bed.x.numpy(), bed.code.numpy(),
                                    bed.F.numpy() if self.profile == "current_volume_v1" else None)


def wedge_mesh(length=.012, height=.004, half_width=.025):
    """Exact finite triangular wedge used by TLMPM, with origin at its cutting edge."""
    vertices = np.array([[0, -half_width, 0], [-length, -half_width, height], [-length, -half_width, -.0004],
                         [0, half_width, 0], [-length, half_width, height], [-length, half_width, -.0004]], np.float32)
    faces = np.array([[0, 1, 2], [3, 5, 4], [0, 4, 1], [0, 3, 4],
                      [1, 5, 2], [1, 4, 5], [2, 3, 0], [2, 5, 3]], np.int32)
    return vertices, faces
