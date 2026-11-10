"""``rubber_strip_v1`` scene compiler (ADR 0023 §8, §9, §13; task R1.3).

A strip L x W x t in a strip-local frame (x along the length, z through the thickness, gravity -z) on the particle
lattice: spacing s = dx / ppc, particles at cell-centred lattice points, the minimum corner on a grid line. The free
length runs over x in [0, L]; boundary cases:

- ``"A"`` clamp / grip jaw: the strip extends to x = -jaw (default 2 dx) and every component of every grid node with
  mass and reference x <= 0 belongs to group 0 (a clamp when it stays, a kinematic grip when it moves);
- ``"C"`` two end-face grips: group 0 = nodes with mass on the face x = 0, group 1 = on the face x = L, component
  mask ``end_mask`` (``"x"``: axial only, plus y/z pins on the face-centre node and a y pin on the node
  (W/2, 0) when those lie on grid nodes: rigid modes pinned on symmetry lines; ``"xyz"``: full clamp);
- ``"block"``: a block centred at ``center`` with one AFFINE group holding every node with mass (patch and homogeneous
  tests: prescribed x = A X + b);
- ``"free"``: no group.

The stencil is the frozen TLMPM formula (trilinear weights, gradients (2 bit - 1) f f / dx, ``tlmpm_v5.py:734-750``)
evaluated with exact lattice fractions (0.25 / 0.75 at 2 particles per cell). Also compiled: the constant cell CSR of
the J-bar gather (cell = stencil corner-0 node, cells in order of first appearance, particles ascending), K = 9 marker
cross-sections (CSR), surface-particle flags, traction groups and the scene receipt (schema ``rcs_rubber_scene_v1``).
"""
from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

SCENE_SCHEMA = "rcs_rubber_scene_v1"
SCENE_FAMILY = "rubber_strip_v1"
CASES = ("A", "C", "block", "free")
MASK_X, MASK_Y, MASK_Z = 1, 2, 4
MASK_XYZ = 7
MARKERS = 9
GROUP_RIGID, GROUP_AFFINE = 0, 1


@dataclass(frozen=True)
class RubberStripSpec:
    length: float
    width: float
    thickness: float
    n_t: int = 3
    dx: float = None                 # grid spacing; default thickness / n_t (a batch passes its uniform dx)
    case: str = "A"
    jaw_length: float = None         # case A: default 2 dx
    end_mask: str = "x"              # case C
    ppc: int = 2
    center: tuple = (0.0, 0.0, 0.0)  # case "block": block centre
    traction_regions: tuple = ()     # ((name, (lo xyz), (hi xyz)), ...) in strip coordinates
    resolution_profile: str = "rubber_rl_v1"

    def grid_dx(self):
        return float(self.thickness)/int(self.n_t) if self.dx is None else float(self.dx)


def _lattice(length, spacing, what):
    k = float(length)/float(spacing)
    if abs(k-round(k)) > 1e-6 or round(k) < 1:
        raise ValueError(f"{what} {length} m is not a positive multiple of the particle spacing {spacing} m "
                         "(geometry on the particle lattice, ADR 0023 §8)")
    return int(round(k))


@dataclass
class RubberStripScene:
    spec: RubberStripSpec
    dx: float
    spacing: float
    corner: np.ndarray               # minimum corner (strip coordinates), float64
    lattice: tuple                   # particles per axis
    X: np.ndarray                    # [n, 3] float32 reference positions
    lattice_index: np.ndarray        # [n, 3] int32
    nodes: np.ndarray                # [n, 8] int32
    weights: np.ndarray              # [n, 8] float32
    gradients: np.ndarray            # [n, 8, 3] float32
    grid_origin: np.ndarray          # float64
    grid_dims: np.ndarray            # int64 [3]
    node_X: np.ndarray               # [nodes, 3] float64 reference node positions
    node_has_mass: np.ndarray        # [nodes] bool
    cell_node: np.ndarray            # [C] int32
    cell_offsets: np.ndarray         # [C + 1] int32
    cell_particles: np.ndarray       # [n] int32
    cell_count: np.ndarray           # [C] int32
    node_group: np.ndarray           # [nodes] int32 (-1 free)
    node_mask: np.ndarray            # [nodes] int32 component mask bits
    groups: list                     # dicts: name, kind, origin, nodes
    particle_traction: np.ndarray    # [n] int32 (-1 none)
    traction_groups: list            # dicts: name, particles
    marker_offsets: np.ndarray       # [K + 1] int32
    marker_particles: np.ndarray     # int32
    marker_reference: np.ndarray     # [K, 3] float64
    surface: np.ndarray              # [n] bool
    free_length: float
    receipt_extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.X)

    @property
    def node_count(self):
        return int(np.prod(self.grid_dims))

    @property
    def volume(self):
        return self.spacing**3

    @property
    def tip_particles(self):
        return self.marker_particles[self.marker_offsets[-2]:self.marker_offsets[-1]]

    def interior_nodes(self):
        """Nodes whose 8 adjacent cells are all occupied (patch-test interior)."""
        dims = self.grid_dims
        occupied = np.zeros(self.node_count, bool)
        occupied[self.cell_node] = True
        idx = np.arange(self.node_count)
        ix, iy, iz = idx//(dims[1]*dims[2]), (idx//dims[2]) % dims[1], idx % dims[2]
        inside = (ix > 0) & (iy > 0) & (iz > 0)
        result = inside.copy()
        for dxi in (0, 1):
            for dyi in (0, 1):
                for dzi in (0, 1):
                    c = ((ix-dxi)*dims[1]+(iy-dyi))*dims[2]+(iz-dzi)
                    ok = inside & (c >= 0)
                    hit = np.zeros(self.node_count, bool)
                    hit[ok] = occupied[c[ok]]
                    result &= hit
        return np.flatnonzero(result)

    def receipt(self):
        s = self.spec
        sha = hashlib.sha256()
        for a in (self.X, self.nodes, self.weights, self.gradients, self.cell_offsets, self.cell_particles,
                  self.node_group, self.node_mask, self.marker_offsets, self.marker_particles):
            sha.update(np.ascontiguousarray(a).tobytes())
        return {"schema": SCENE_SCHEMA, "family": SCENE_FAMILY, "case": s.case,
                "geometry_m": {"length": s.length, "width": s.width, "thickness": s.thickness,
                               "jaw_length": float(self.receipt_extra.get("jaw_length", 0.0))},
                "dx_m": self.dx, "particle_spacing_m": self.spacing, "particles_per_cell_axis": s.ppc,
                "cells_through_thickness": s.thickness/self.dx, "resolution_profile": s.resolution_profile,
                "lattice": list(self.lattice), "particles": self.n, "nodes": self.node_count,
                "nodes_with_mass": int(self.node_has_mass.sum()), "cells": len(self.cell_node),
                "grid_origin_m": self.grid_origin.tolist(), "grid_dims": [int(v) for v in self.grid_dims],
                "groups": [{"name": g["name"], "kind": g["kind"], "nodes": int(g["nodes"]),
                            "origin_m": list(map(float, g["origin"])), "pins": g.get("pins", [])} for g in self.groups],
                "traction_groups": [{"name": t["name"], "particles": int(t["particles"])} for t in self.traction_groups],
                "markers": {"count": MARKERS, "particles": int(len(self.marker_particles)),
                            "reference_x_m": self.marker_reference[:, 0].tolist()},
                "surface_particles": int(self.surface.sum()), "free_length_m": self.free_length,
                "array_sha256": sha.hexdigest()}


def compile_strip(spec):
    """Compile a :class:`RubberStripSpec` into a :class:`RubberStripScene` (numpy, host)."""
    if spec.case not in CASES:
        raise ValueError(f"case {spec.case!r} not in {CASES}")
    if spec.ppc not in (2, 3):
        raise ValueError("rubber_strip_v1 uses 2 particles per cell per axis (ADR 0023 §8); 3 only as the labelled "
                         "quadrature check of the refinement study (§14.4)")
    dx = spec.grid_dx()
    s = dx/spec.ppc
    jaw = 0.0
    if spec.case == "A":
        jaw = 2*dx if spec.jaw_length is None else float(spec.jaw_length)
        _lattice(jaw, dx, "jaw length (grid lines)")
    nx = _lattice(spec.length+jaw, s, "length")
    ny = _lattice(spec.width, s, "width")
    nz = _lattice(spec.thickness, s, "thickness")
    if spec.case == "block":
        corner = np.asarray(spec.center, float)-0.5*np.array([spec.length, spec.width, spec.thickness])
    else:
        corner = np.array([-jaw, 0.0, 0.0])
    ii = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    X64 = corner+(ii+0.5)*s
    n = len(ii)
    # Exact stencil: particle lattice index i sits in cell i // ppc + 2 at fraction (i % ppc + 0.5) / ppc.
    cells = ii//spec.ppc+2
    frac = (ii % spec.ppc+0.5)/spec.ppc
    dims = cells.max(0)+4                      # = ceil((max - origin)/dx) + 3 of the frozen formula
    origin = corner-2*dx
    if int(np.prod(dims)) >= 2**31 or n*8 >= 2**31:
        raise ValueError("scene exceeds int32 indexing")
    nodes = np.empty((n, 8), np.int64)
    weights = np.empty((n, 8), np.float64)
    gradients = np.empty((n, 8, 3), np.float64)
    for corner_id in range(8):
        bit = np.array([(corner_id >> j) & 1 for j in range(3)])
        xyz = cells+bit
        nodes[:, corner_id] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
        f = np.where(bit, frac, 1-frac)
        weights[:, corner_id] = f.prod(1)
        for axis in range(3):
            gradients[:, corner_id, axis] = (2*bit[axis]-1)*f[:, (axis+1) % 3]*f[:, (axis+2) % 3]/dx
    node_count = int(np.prod(dims))
    idx = np.arange(node_count)
    node_ijk = np.stack([idx//(dims[1]*dims[2]), (idx//dims[2]) % dims[1], idx % dims[2]], 1)
    node_X = origin+node_ijk*dx
    has_mass = np.zeros(node_count, bool)
    has_mass[nodes.reshape(-1)] = True
    # cell CSR (J-bar gather): cell = corner-0 node, cells in order of first appearance, particles ascending
    cell_of = nodes[:, 0]
    _, first = np.unique(cell_of, return_index=True)
    cell_node = cell_of[np.sort(first)]
    rank = np.empty(node_count, np.int64)
    rank[cell_node] = np.arange(len(cell_node))
    order = np.argsort(rank[cell_of], kind="stable")
    cell_count = np.bincount(rank[cell_of], minlength=len(cell_node))
    cell_offsets = np.r_[0, np.cumsum(cell_count)]
    # groups
    node_group = np.full(node_count, -1, np.int64)
    node_mask = np.zeros(node_count, np.int64)
    groups = []
    gi = node_ijk-2                                # node index relative to the corner's grid line
    if spec.case == "A":
        jaw_cells = int(round(jaw/dx))
        sel = has_mass & (gi[:, 0] <= jaw_cells)
        node_group[sel], node_mask[sel] = 0, MASK_XYZ
        groups.append({"name": "jaw", "kind": "rigid", "origin": np.array([0.0, spec.width/2, spec.thickness/2]),
                       "nodes": int(sel.sum())})
    elif spec.case == "C":
        cells_x = spec.length/dx
        if abs(cells_x-round(cells_x)) > 1e-6:
            raise ValueError("case C needs L to be a multiple of dx (end faces on grid lines)")
        cy, cz = spec.width/2/dx, spec.thickness/2/dx
        on_y, on_z = abs(cy-round(cy)) < 1e-6, abs(cz-round(cz)) < 1e-6
        for g, face in enumerate((0, int(round(cells_x)))):
            sel = has_mass & (gi[:, 0] == face)
            node_group[sel] = g
            pins = []
            if spec.end_mask == "xyz":
                node_mask[sel] = MASK_XYZ
            elif spec.end_mask == "x":
                node_mask[sel] = MASK_X
                if on_y and on_z:
                    centre = sel & (gi[:, 1] == round(cy)) & (gi[:, 2] == round(cz))
                    node_mask[centre] = MASK_XYZ
                    pins.append("centre_yz")
                if on_y:
                    low = sel & (gi[:, 1] == round(cy)) & (gi[:, 2] == 0)
                    node_mask[low] |= MASK_Y
                    pins.append("mid_bottom_y")
            else:
                raise ValueError("end_mask is 'x' or 'xyz'")
            groups.append({"name": f"grip{g}", "kind": "rigid", "pins": pins, "nodes": int(sel.sum()),
                           "origin": np.array([face*dx, spec.width/2, spec.thickness/2])})
    elif spec.case == "block":
        node_group[has_mass], node_mask[has_mass] = 0, MASK_XYZ
        groups.append({"name": "affine_all", "kind": "affine", "origin": np.asarray(spec.center, float),
                       "nodes": int(has_mass.sum())})
    # traction groups
    particle_traction = np.full(n, -1, np.int64)
    traction_groups = []
    for t, (name, lo, hi) in enumerate(spec.traction_regions):
        sel = np.all((X64 >= np.asarray(lo)) & (X64 <= np.asarray(hi)), axis=1) & (particle_traction < 0)
        if not sel.any():
            raise ValueError(f"traction region {name!r} holds no particle")
        particle_traction[sel] = t
        traction_groups.append({"name": name, "particles": int(sel.sum())})
    # markers: K cross-sections at x_k = k L / 8 of the free length (strip x in [0, L]; block: its own x extent)
    x_lo = corner[0] if spec.case == "block" else 0.0
    free_length = float(spec.length)
    layer_x = corner[0]+(np.arange(nx)+0.5)*s
    candidates = np.flatnonzero(layer_x >= x_lo-1e-12)
    marker_lists, marker_ref = [], []
    for k in range(MARKERS):
        xk = x_lo+k*free_length/(MARKERS-1)
        # nearest layer; exact ties (layers at x_k +- s/2) go deterministically to the lower layer
        layer = candidates[np.argmin(np.abs(layer_x[candidates]-(xk-1e-6*s)))]
        ids = np.flatnonzero(ii[:, 0] == layer)
        marker_lists.append(ids)
        marker_ref.append(X64[ids].mean(0))
    marker_offsets = np.r_[0, np.cumsum([len(m) for m in marker_lists])]
    surface = np.any((ii == 0) | (ii == np.array([nx-1, ny-1, nz-1])), axis=1)
    return RubberStripScene(
        spec=spec, dx=dx, spacing=s, corner=corner, lattice=(nx, ny, nz), X=X64.astype(np.float32),
        lattice_index=ii.astype(np.int32), nodes=nodes.astype(np.int32), weights=weights.astype(np.float32),
        gradients=gradients.astype(np.float32), grid_origin=origin, grid_dims=dims.astype(np.int64),
        node_X=node_X, node_has_mass=has_mass, cell_node=cell_node.astype(np.int32),
        cell_offsets=cell_offsets.astype(np.int32), cell_particles=order.astype(np.int32),
        cell_count=cell_count.astype(np.int32), node_group=node_group.astype(np.int32),
        node_mask=node_mask.astype(np.int32), groups=groups, particle_traction=particle_traction.astype(np.int32),
        traction_groups=traction_groups, marker_offsets=marker_offsets.astype(np.int32),
        marker_particles=np.concatenate(marker_lists).astype(np.int32), marker_reference=np.asarray(marker_ref),
        surface=surface, free_length=free_length, receipt_extra={"jaw_length": jaw})


_RECEIPT_KEYS = {"schema", "family", "case", "geometry_m", "dx_m", "particle_spacing_m", "particles_per_cell_axis",
                 "cells_through_thickness", "resolution_profile", "lattice", "particles", "nodes", "nodes_with_mass",
                 "cells", "grid_origin_m", "grid_dims", "groups", "traction_groups", "markers", "surface_particles",
                 "free_length_m", "array_sha256"}


def validate_scene_receipt(receipt):
    """Schema check of a scene receipt (``rcs_rubber_scene_v1``); raises ValueError."""
    missing = _RECEIPT_KEYS-set(receipt)
    if missing:
        raise ValueError(f"scene receipt missing {sorted(missing)}")
    if receipt["schema"] != SCENE_SCHEMA or receipt["family"] != SCENE_FAMILY or receipt["case"] not in CASES:
        raise ValueError("scene receipt schema/family/case")
    if receipt["markers"]["count"] != MARKERS or not receipt["particles"] > 0:
        raise ValueError("scene receipt markers/particles")
    if not all(math.isfinite(v) for v in receipt["grid_origin_m"]):
        raise ValueError("scene receipt grid origin")
    return True
