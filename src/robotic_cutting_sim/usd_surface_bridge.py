"""Solver-neutral dynamic mesh staging and full USD geometry for ideal cameras.

Updating a mesh is a renderer bridge, never evidence of physical cutting. Callers
must supply the solver's actual surface, including any separated components.
"""
import hashlib

import numpy as np


def validate_surface(vertices_m, triangles):
    vertices = np.asarray(vertices_m, dtype=np.float32)
    faces = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices_m must be finite Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("triangles must be integer Mx3")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Triangle index outside vertex range")
    if faces.size and np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 0] == faces[:, 2])
                             | (faces[:, 1] == faces[:, 2])):
        raise ValueError("Triangle contains repeated vertex indices")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces, dtype=np.int32)


class DynamicSurfaceBridge:
    """Author all topology arrays together, clearing stale normals on every update."""
    def __init__(self, stage, prim_path="/World/ContinuousSpecimen"):
        from pxr import UsdGeom
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise ValueError("Dynamic surfaces require an SI stage with metersPerUnit=1")
        self.stage = stage
        self.mesh = UsdGeom.Mesh.Define(stage, prim_path)
        self.mesh.CreateSubdivisionSchemeAttr("none")
        self.mesh.CreateDoubleSidedAttr(True)
        self.last_timestamp_s = -np.inf

    def update(self, vertices_m, triangles, physics_timestamp_s, *, surface_timestamp_s=None,
               time_code=None):
        from pxr import Gf, Sdf, Usd, Vt
        vertices, faces = validate_surface(vertices_m, triangles)
        timestamp = float(physics_timestamp_s)
        surface_time = timestamp if surface_timestamp_s is None else float(surface_timestamp_s)
        if not np.isfinite([timestamp, surface_time]).all() or timestamp < 0:
            raise ValueError("Surface timestamps must be finite and nonnegative")
        if abs(timestamp - surface_time) > 1e-9:
            raise ValueError("Refusing to stage a stale surface")
        if timestamp <= self.last_timestamp_s:
            raise ValueError("Surface timestamps must increase strictly")
        tc = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(float(time_code))
        extent = ([Gf.Vec3f(*map(float, vertices.min(0))), Gf.Vec3f(*map(float, vertices.max(0)))]
                  if len(vertices) else [])
        # ChangeBlock prevents downstream notices from exposing half-updated topology.
        with Sdf.ChangeBlock():
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices), tc)
            self.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)), tc)
            self.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.ravel()), tc)
            self.mesh.GetNormalsAttr().Clear()
            self.mesh.GetNormalsAttr().Block()  # Also suppress normals from weaker source layers.
            self.mesh.CreateExtentAttr().Set(extent, tc)
            for name, value in [("physicsTimestampS", timestamp), ("surfaceTimestampS", surface_time)]:
                self.mesh.GetPrim().CreateAttribute("research:" + name, Sdf.ValueTypeNames.Double).Set(value, tc)
        self.last_timestamp_s = timestamp
        return {"physics_timestamp_s": timestamp, "surface_timestamp_s": surface_time,
                "vertices": len(vertices), "triangles": len(faces),
                "topology_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
                "surface_sha256": hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()}

    def update_face_varying_v1(self, vertices_m, triangles, face_varying_normals, physics_timestamp_s, *,
                               surface_timestamp_s=None, time_code=None):
        """Additive, versioned variant of ``update`` (WP-12 cut_aware_surface_v1): same timestamp, ChangeBlock
        and topology contract, but stages one unit normal per triangle corner with ``faceVarying``
        interpolation, so sharp dual-contouring creases survive. ``update`` is unchanged and still clears and
        blocks normals. The caller computes the normals (``cut_aware_surface.face_varying_normals``) per
        material region; this bridge never averages them."""
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
        vertices, faces = validate_surface(vertices_m, triangles)
        normals = np.asarray(face_varying_normals, dtype=np.float32)
        if normals.shape != (3 * len(faces), 3) or not np.isfinite(normals).all():
            raise ValueError("face_varying_normals must be finite (3 * triangles) x 3")
        if len(normals) and np.abs(np.linalg.norm(normals, axis=1) - 1).max() > 1e-4:
            raise ValueError("face_varying_normals must be unit vectors")
        timestamp = float(physics_timestamp_s)
        surface_time = timestamp if surface_timestamp_s is None else float(surface_timestamp_s)
        if not np.isfinite([timestamp, surface_time]).all() or timestamp < 0:
            raise ValueError("Surface timestamps must be finite and nonnegative")
        if abs(timestamp - surface_time) > 1e-9:
            raise ValueError("Refusing to stage a stale surface")
        if timestamp <= self.last_timestamp_s:
            raise ValueError("Surface timestamps must increase strictly")
        tc = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(float(time_code))
        extent = ([Gf.Vec3f(*map(float, vertices.min(0))), Gf.Vec3f(*map(float, vertices.max(0)))]
                  if len(vertices) else [])
        normals = np.ascontiguousarray(normals)
        with Sdf.ChangeBlock():
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices), tc)
            self.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)), tc)
            self.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.ravel()), tc)
            self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals), tc)
            self.mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            self.mesh.CreateExtentAttr().Set(extent, tc)
            for name, value in [("physicsTimestampS", timestamp), ("surfaceTimestampS", surface_time)]:
                self.mesh.GetPrim().CreateAttribute("research:" + name, Sdf.ValueTypeNames.Double).Set(value, tc)
        self.last_timestamp_s = timestamp
        return {"physics_timestamp_s": timestamp, "surface_timestamp_s": surface_time,
                "vertices": len(vertices), "triangles": len(faces), "bridge_method": "update_face_varying_v1",
                "normals_interpolation": "faceVarying",
                "topology_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
                "surface_sha256": hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
                "normals_sha256": hashlib.sha256(normals.tobytes()).hexdigest()}


def usd_camera_calibration(stage, camera_path, width=640, height=480, time_code=None,
                           near_m=None, far_m=None):
    from pxr import Usd, UsdGeom
    tc = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(float(time_code))
    camera = UsdGeom.Camera.Get(stage, camera_path)
    if not camera:
        raise ValueError(f"No USD Camera at {camera_path}")
    if camera.GetProjectionAttr().Get(tc) != "perspective":
        raise ValueError("Only perspective cameras are supported")
    horizontal = float(camera.GetHorizontalApertureAttr().Get(tc))
    vertical = float(camera.GetVerticalApertureAttr().Get(tc))
    focal = float(camera.GetFocalLengthAttr().Get(tc))
    if min(horizontal, vertical, focal, width, height) <= 0:
        raise ValueError("Invalid camera lens or resolution")
    # USD lens offsets move the image plane center, hence subtract at projection.
    cx = width * (.5 - float(camera.GetHorizontalApertureOffsetAttr().Get(tc)) / horizontal)
    cy = height * (.5 + float(camera.GetVerticalApertureOffsetAttr().Get(tc)) / vertical)
    k = [[width * focal / horizontal, 0, cx], [0, height * focal / vertical, cy], [0, 0, 1]]
    transform = np.array(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(tc)).T
    transform[:3, :3] = transform[:3, :3] @ np.diag([1, -1, -1])
    clipping = camera.GetClippingRangeAttr().Get(tc)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    transform[:3, 3] *= meters_per_unit
    return {"width": width, "height": height, "K": k,
            "T_world_camera": transform.tolist(), "prim_path": camera_path,
            "near_m": float(clipping[0]) * meters_per_unit if near_m is None else float(near_m),
            "far_m": float(clipping[1]) * meters_per_unit if far_m is None else float(far_m),
            "convention": "OpenCV +Z forward, +Y down", "depth": "optical Z, meters; invalid=0",
            "sensor_model": "ideal pinhole from native USD lens; no D455 post-processing"}


def stage_triangles(stage, time_code=None, excluded_prefixes=()):
    """Flatten visible Mesh/Cube geometry including instance proxies to SI arrays.

    This diagnostic uses polygon fans and display colors, without MDL, textures,
    subdivision or RTX. The workcell's official Panda assets contain triangle and
    convex polygon geometry. Unsupported primitives are explicitly reported.
    """
    from pxr import Usd, UsdGeom
    tc = Usd.TimeCode.Default() if time_code is None else Usd.TimeCode(float(time_code))
    xforms = UsdGeom.XformCache(tc)
    vertices, triangles, colors, rows, unsupported = [], [], [], [], []
    offset = 0
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if any(path == prefix or path.startswith(prefix + "/") for prefix in excluded_prefixes):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility(tc) == "invisible":
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(tc), float)
            if not len(points):
                continue
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(tc), int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(tc), int)
            if counts.sum() != len(indices) or np.any(counts < 3):
                raise ValueError(f"Invalid polygon topology: {path}")
            faces, cursor = [], 0
            for count in counts:
                polygon = indices[cursor:cursor + count]
                faces.extend((polygon[0], polygon[j], polygon[j + 1]) for j in range(1, count - 1))
                cursor += count
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get(tc)) / 2
            points = half * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
            quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6),
                     (1, 2, 6, 5), (3, 0, 4, 7)]
            faces = [(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads]
        else:
            if prim.IsA(UsdGeom.Gprim):
                unsupported.append({"prim_path": path, "type": prim.GetTypeName()})
            continue
        transform = np.array(xforms.GetLocalToWorldTransform(prim)).T
        points = (points @ transform[:3, :3].T + transform[:3, 3]) * meters_per_unit
        points, faces = validate_surface(points, np.asarray(faces, np.int32))
        display = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get(tc)
        color = np.asarray(display[0], float) if display else np.array([.8, .82, .85])
        if "ContinuousSpecimen" in path:
            color = np.array([.72, .32, .13])
        elif "Blade" in path:
            color = np.array([.38, .65, .85])
        vertices.append(points); triangles.append(faces + offset)
        colors.append(np.tile(color, (len(faces), 1)))
        rows.append({"prim_path": path, "triangles": len(faces), "first_triangle": sum(len(x) for x in triangles[:-1])})
        offset += len(points)
    if not triangles:
        raise ValueError("USD stage has no supported visible geometry")
    return (np.vstack(vertices), np.vstack(triangles), np.vstack(colors).astype(np.float32),
            {"prims": rows, "unsupported_geometry": unsupported, "meters_per_unit": meters_per_unit})
