"""``cut_aware_surface_v1``: reference-lattice linked volume + local dual contouring + cut-side-aware embedding.

WP-12 Phase 3, Layers 3-6 (design: ``docs/reports/WP12_SURFACE_TECH_REVIEW.md`` §2.2, §3.4, §6).
Approach B of the review. ``particle_surface_v1`` (``cutting_surface.CuttingSurface``) stays the frozen
baseline and the lattice-free fallback; nothing here changes it.

Hard rules (review §6.10), enforced by construction and by ``tests/test_cut_aware_surface.py``:

* The surface **reads** Layer-1 state (reference lattice, current positions, total F, packed codes, the
  optional ``cut_events_v1`` log) and writes nothing back. Inputs are never modified.
* A cut face exists only across a lattice link whose endpoint codes conflict
  (``cut_topology.conflicting_packed`` semantics, bitmask ``(A1_a & A2_b) | (A2_a & A1_b)``). A boundary face
  exists only across a link to an absent lattice site. Nothing else emits a face: no target plane, no swept
  volume, no contact or gap test. The swept region is an audit only (``cut_events``/``law8_validation``).
* Vertices are duplicated per local material region and never shared between regions, so the two sides of a
  cut are never welded, even when they recontact.
* Global piece counts always come from ``mechanical_components_packed`` (via the partition); the mesh only
  reports how its shells map onto those components (``audit``).
* Fragments are never dropped (a one-particle fragment is a small closed cell).

Layer 3 — topology (``ReferenceLattice``)
    Particles are the linked-volume cells (they sit at the integer lattice cell centres, ``benchmark.py``
    ``sample_specimen``), the 6-neighbour bonds are the links. A link is *cut* iff its endpoint codes conflict
    (its cut set is the conflicting bit set), *boundary* iff exactly one endpoint site is present, else *uncut*
    (including every 0-1 / 0-2 pair, so crack fronts and digit-0 ligaments stay connected).
    Boundary Hermite profiles: ``cell_face_v1`` (t = 1/2, axis normal; exact for boxes, terraced on curved
    shapes) and ``analytic_reference_v1`` (crossing of the link with the workpiece's analytic **initial** shape,
    ``AnalyticBox`` / ``AnalyticHeightfield`` / ``analytic_shape_from_workpiece``; never a target cut).
    Cut Hermite: the ``cut_events_v1`` log (t along the link and the reference normal N, per cut id) when given;
    otherwise the labelled R4 midpoint fallback (t = 1/2, normal = normalised sum of the signed directions,
    digit 1 -> digit 2, of the same cut's conflicting links within ``fallback_normal_radius_spacings``).
    The two sources are counted separately in the metadata.

Layer 4 — local dual contouring
    Dual cells are 2x2x2 blocks of lattice sites. Regions = connected components of present corners under the
    cell's uncut edges (min-label propagation over the 12 edges). One representative per (cell, region):
    QEF over the Hermite data of the cell's disconnected edges incident to a region corner, solved in float64 by
    SVD in cell-local coordinates with singular values < ``qef_truncation`` x sigma_max truncated toward the mass
    point (Ju et al. 2002, §2.3), clamped to the cell (counted). A link severed by several cuts uses, for each
    side, the crossing nearest that side's endpoint (``multi_crossing_links``). Every disconnected link emits
    one quad per present side, from that side's region representative in the four incident cells, oriented from
    the side endpoint toward the other endpoint. At an open crack front both side quads use the same tip
    representative (closed slit tip); no face is emitted beyond the last conflicting link.

Layer 5 — embedding
    x_r = sum_p w_p [x_p + F_p (X_r - X_p)] / sum_p w_p over the region's own particles, with the trilinear
    weights of X_r restricted to the region and renormalised (exact for any affine motion with any subset).
    det F_p <= 0 -> translation-only term for that particle (counted). Without F: per-particle least-squares
    affine fit over the particle's code-compatible neighbours within two uncut links (documented variant of
    review §6.5: the neighbourhood never crosses a cut), translation-only when fewer than four non-coplanar
    particles are available (counted). Stateless: a pure function of the current Layer-1 state.

Layer 6 — rendering
    ``face_varying_normals`` (crease angle; averaging only among triangles sharing a vertex, and vertices are
    per region, so normals are never averaged across regions) and the additive bridge method
    ``usd_surface_bridge.DynamicSurfaceBridge.update_face_varying_v1``.

Incremental update
    The per-link signature (state, cut set, both side Hermite crossings) is recomputed every extraction (cheap,
    vectorised); dual cells incident to a changed link are dirty and only their regions and QEFs are recomputed.
    Quads are re-emitted from the cached per-cell tables, and every representative is re-embedded. A full
    rebuild gives exactly the same arrays (tested).

Warp port (not done here; CPU NumPy/SciPy reference): the region labelling and the QEF are independent per dual
cell (one thread per dirty cell, 3x3 SVD via ``wp.svd3`` on AᵀA in float64 or a Givens QR), quad emission is
independent per disconnected link, the embedding is one thread per representative (8 corner reads), and the LSQ
moments are per-particle atomics over the two-link neighbour table. The static tables (cells, links, link->cell
incidence, cell-edge -> link) are built once on the host and uploaded.
"""
import hashlib
import time

import numpy as np

from .cut_topology import side_masks, ternary_to_packed, validate_packed

PROFILE = "cut_aware_surface_v1"
BASELINE_PROFILE = "particle_surface_v1"
SURFACE_PROFILES = (BASELINE_PROFILE, PROFILE)
BOUNDARY_PROFILES = ("cell_face_v1", "analytic_reference_v1")

# cut_events_v1 flag bits (cut_events.py; duplicated so this module does not import Warp; equality is tested)
FLAG_SIGN_INCONSISTENT = 1
FLAG_DEGENERATE_REFERENCE_NORMAL = 16

CORNER_OFFSETS = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], np.int64)
# 12 cell edges (lower corner i, upper corner j = i + 2^axis, axis)
CELL_EDGES = np.array([(i, i | (1 << a), a) for a in range(3) for i in range(8) if not (i >> a) & 1], np.int64)
# counter-clockwise (seen from +axis) offsets (o1, o2) of the four dual cells around a link along axis a, in the
# (a+1, a+2) axes: cell min corner g = lower - o1 e_(a+1) - o2 e_(a+2)
_RING = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], np.int64)

LINK_UNCUT, LINK_BOUNDARY, LINK_CUT = 0, 1, 2
SOURCE_NAMES = ("boundary_cell_face", "boundary_analytic", "boundary_analytic_inconsistent_cell_face",
                "recorder", "recorder_sign_inconsistent_midpoint", "recorder_point_fallback_normal",
                "midpoint_fallback")
(SRC_CELL_FACE, SRC_ANALYTIC, SRC_ANALYTIC_INCONSISTENT, SRC_RECORDER, SRC_RECORDER_SIGN,
 SRC_RECORDER_NORMAL_FALLBACK, SRC_MIDPOINT) = range(len(SOURCE_NAMES))


class CutAwareTopologyError(RuntimeError):
    """A mesh shell or lattice piece spans several mechanical components (impossible per review §3.4)."""


def _sha(array):
    """Same digest as cut_events._sha (shape, dtype and bytes) so a recorder log hash can be compared."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def reference_stencil(points, dx):
    """The tlmpm_v5 eight-node reference stencil (nodes, weights, gradients) of reference ``points``.

    A NumPy copy of ``TLMPM.__init__`` (tlmpm_v5.py, grid origin floor(min/dx) dx - 2 dx) for recordings that
    stored the reference points but not the stencil. Equality with the solver's arrays is tested on CPU.
    """
    points = np.asarray(points, np.float32)
    origin = np.floor(points.min(0)/dx)*dx-2*dx
    dims = np.ceil((points.max(0)-origin)/dx).astype(int)+3
    cells = np.floor((points-origin)/dx).astype(int)
    fraction = (points-origin)/dx-cells
    n = len(points)
    nodes = np.empty((n, 8), np.int32)
    weights = np.empty((n, 8), np.float32)
    gradients = np.empty((n, 8, 3), np.float32)
    for corner in range(8):
        bit = np.array([(corner >> j) & 1 for j in range(3)])
        xyz = cells+bit
        nodes[:, corner] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
        f = np.where(bit, fraction, 1-fraction)
        weights[:, corner] = f.prod(1)
        for axis in range(3):
            gradients[:, corner, axis] = (2*bit[axis]-1)*f[:, (axis+1) % 3]*f[:, (axis+2) % 3]/dx
    return nodes, weights, gradients


# ----------------------------------------------------------------------------- analytic initial shapes

class AnalyticBox:
    """Axis-aligned box [low, high] in the reference frame (initial workpiece shape, never a cut target)."""

    kind = "box"

    def __init__(self, low, high):
        self.low, self.high = np.asarray(low, np.float64), np.asarray(high, np.float64)
        if self.low.shape != (3,) or self.high.shape != (3,) or np.any(self.high <= self.low):
            raise ValueError("AnalyticBox needs low < high (3-vectors)")

    def inside(self, X):
        X = np.asarray(X, np.float64)
        return np.all((X >= self.low) & (X <= self.high), axis=-1)

    def normal(self, X):
        X = np.asarray(X, np.float64)
        terms = np.concatenate([self.low-X, X-self.high], axis=-1)        # (..., 6), max is the active face
        active = np.argmax(terms, axis=-1)
        out = np.zeros(X.shape)
        axis, sign = active % 3, np.where(active < 3, -1., 1.)
        np.put_along_axis(out, axis[..., None], sign[..., None], axis=-1)
        return out

    def to_dict(self):
        return {"kind": self.kind, "low_m": self.low.tolist(), "high_m": self.high.tolist()}


class AnalyticHeightfield:
    """Workpiece-local initial shape {footprint(x, y), 0 <= z <= h(x, y)} of ``workpieces.WorkpieceSpec``.

    ``inside`` reproduces the sampler's tests (rectangle: strict |x| < L/2, |y| < W/2; capsule: closed disc
    union; z <= h). The normal is the gradient of the active term (bottom, top height field, footprint side);
    the height-field gradient is a central difference with step ``gradient_step_m``.
    """

    kind = "heightfield"

    def __init__(self, height, length_m, width_m, *, footprint="rectangle", gradient_step_m=1e-7, name=None):
        if footprint not in ("rectangle", "capsule"):
            raise ValueError("footprint is 'rectangle' or 'capsule'")
        self.height_fn, self.length, self.width = height, float(length_m), float(width_m)
        self.footprint, self.step, self.name = footprint, float(gradient_step_m), name

    def _footprint_terms(self, x, y):
        if self.footprint == "rectangle":
            return np.maximum(np.abs(x)-self.length/2, np.abs(y)-self.width/2)
        radius, half = self.width/2, (self.length-self.width)/2
        return np.hypot(np.maximum(np.abs(x)-half, 0.), y)-radius

    def inside(self, X):
        X = np.asarray(X, np.float64)
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        if self.footprint == "rectangle":
            foot = (np.abs(x) < self.length/2) & (np.abs(y) < self.width/2)
        else:
            foot = self._footprint_terms(x, y) <= 0
        return foot & (z >= 0) & (z <= self.height_fn(x, y))

    def normal(self, X):
        X = np.asarray(X, np.float64)
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        h = self.height_fn(x, y)
        terms = np.stack([-z, z-h, self._footprint_terms(x, y)], -1)
        active = np.argmax(terms, axis=-1)
        e = self.step
        hx = (self.height_fn(x+e, y)-self.height_fn(x-e, y))/(2*e)
        hy = (self.height_fn(x, y+e)-self.height_fn(x, y-e))/(2*e)
        top = np.stack([-hx, -hy, np.ones_like(hx)], -1)
        if self.footprint == "rectangle":
            ax = np.abs(x)-self.length/2 >= np.abs(y)-self.width/2
            side = np.stack([np.where(ax, np.sign(x), 0.), np.where(ax, 0., np.sign(y)), np.zeros_like(x)], -1)
        else:
            half = (self.length-self.width)/2
            dx = np.sign(x)*np.maximum(np.abs(x)-half, 0.)
            side = np.stack([dx, y, np.zeros_like(x)], -1)
        bottom = np.broadcast_to(np.array([0., 0., -1.]), X.shape)
        out = np.where((active == 0)[..., None], bottom, np.where((active == 1)[..., None], top, side))
        length = np.linalg.norm(out, axis=-1, keepdims=True)
        return out/np.where(length > 0, length, 1.)

    def to_dict(self):
        return {"kind": self.kind, "length_m": self.length, "width_m": self.width, "footprint": self.footprint,
                "gradient_step_m": self.step, "name": self.name}


def analytic_shape_from_workpiece(workpiece):
    """``AnalyticHeightfield`` of a ``workpieces.WorkpieceSpec`` (its sampled initial geometry, local frame)."""
    g = workpiece.geometry
    return AnalyticHeightfield(g.height, g.length_m, g.width_m, footprint=workpiece.footprint,
                               name="workpieces.WorkpieceSpec initial height field")


def analytic_crossings(shape, inside_points, outside_points, iterations=60):
    """Bisection t in [0, 1] along inside -> outside where ``shape.inside`` changes, and the shape normal there.

    Returns (t, normal, consistent); ``consistent`` is False where the endpoints do not straddle the shape.
    """
    a, b = np.asarray(inside_points, np.float64), np.asarray(outside_points, np.float64)
    consistent = shape.inside(a) & ~shape.inside(b)
    lo, hi = np.zeros(len(a)), np.ones(len(a))
    for _ in range(iterations):
        mid = .5*(lo+hi)
        inside = shape.inside(a+mid[:, None]*(b-a))
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    t = .5*(lo+hi)
    return t, shape.normal(a+t[:, None]*(b-a)), consistent


# ----------------------------------------------------------------------------- Layer 3: lattice topology

class ReferenceLattice:
    """Static linked-volume tables of one reference particle lattice (built once per workpiece)."""

    def __init__(self, lattice, reference_positions, spacing_m, *, check=True):
        lattice = np.asarray(lattice)
        if lattice.ndim != 2 or lattice.shape[1] != 3 or not np.issubdtype(lattice.dtype, np.integer) or not len(lattice):
            raise ValueError("The reference lattice must be a non-empty (N, 3) integer array")
        reference = np.asarray(reference_positions, np.float64)
        if reference.shape != lattice.shape or not np.isfinite(reference).all():
            raise ValueError("Need one finite reference position per lattice site")
        self.n, self.spacing = len(lattice), float(spacing_m)
        if not self.spacing > 0:
            raise ValueError("spacing_m must be positive")
        self.lattice = lattice.astype(np.int64)
        self.reference = reference
        self.low = self.lattice.min(0)
        self.shifted = self.lattice-self.low+1                         # >= 1: every neighbour/cell index >= 0
        self.dims = self.shifted.max(0)+2
        keys = self.key(self.shifted)
        self.order = np.argsort(keys, kind="stable")
        self.sorted_keys = keys[self.order]
        if self.n > 1 and np.any(self.sorted_keys[1:] == self.sorted_keys[:-1]):
            raise ValueError("Reference lattice indices must be unique")
        self.origin = (reference-self.lattice*self.spacing).mean(0)    # X = origin + lattice * spacing
        residual = np.abs(reference-(self.origin+self.lattice*self.spacing)).max()
        if check and residual > 1e-3*self.spacing:
            raise ValueError(f"Reference points are not origin + lattice * spacing (residual {residual:.3g} m)")
        self.lattice_residual_m = float(residual)
        self._build_links()
        self._build_cells()

    # -- indexing
    def key(self, v):
        v = np.asarray(v, np.int64)
        return (v[..., 0]*self.dims[1]+v[..., 1])*self.dims[2]+v[..., 2]

    def decode(self, k):
        k = np.asarray(k, np.int64)
        return np.stack([k//(self.dims[1]*self.dims[2]), (k//self.dims[2]) % self.dims[1], k % self.dims[2]], -1)

    def lookup(self, v):
        """Particle index at shifted lattice sites ``v`` (..., 3); -1 where absent."""
        v = np.asarray(v, np.int64)
        valid = np.all((v >= 0) & (v < self.dims), axis=-1)
        k = self.key(np.where(valid[..., None], v, 0))
        position = np.minimum(np.searchsorted(self.sorted_keys, k), self.n-1)
        found = valid & (self.sorted_keys[position] == k)
        return np.where(found, self.order[position], -1)

    def site_position(self, v):
        """Reference position of shifted lattice sites (float64)."""
        return self.origin+(np.asarray(v, np.int64)-1+self.low)*self.spacing

    # -- static tables
    def _build_links(self):
        candidates = []
        for axis in range(3):
            e = np.eye(3, dtype=np.int64)[axis]
            for lower in (self.shifted, self.shifted-e):
                candidates.append(self.key(lower)*3+axis)
        self.link_keys = np.unique(np.concatenate(candidates))
        self.link_axis = self.link_keys % 3
        self.link_lower = self.decode(self.link_keys//3)
        upper = self.link_lower+np.eye(3, dtype=np.int64)[self.link_axis]
        self.link_lo, self.link_hi = self.lookup(self.link_lower), self.lookup(upper)
        self.link_lo_reference = self.site_position(self.link_lower)
        self.link_hi_reference = self.site_position(upper)
        self.n_links = len(self.link_keys)

    def _build_cells(self):
        corner_keys = self.key(self.shifted[:, None, :]-CORNER_OFFSETS[None])
        self.cell_keys = np.unique(corner_keys)
        self.cell_min = self.decode(self.cell_keys)
        self.n_cells = len(self.cell_keys)
        self.cell_origin = self.site_position(self.cell_min)                 # reference min corner
        self.corner_particle = self.lookup(self.cell_min[:, None, :]+CORNER_OFFSETS[None])   # (M, 8)
        edge_keys = self.key(self.cell_min[:, None, :]+CORNER_OFFSETS[CELL_EDGES[:, 0]][None])*3+CELL_EDGES[:, 2]
        position = np.minimum(np.searchsorted(self.link_keys, edge_keys), self.n_links-1)
        self.cell_edge_link = np.where(self.link_keys[position] == edge_keys, position, -1)   # (M, 12)
        # four dual cells around every link, counter-clockwise seen from +axis, and the endpoint corner indices
        a = self.link_axis
        a1, a2 = (a+1) % 3, (a+2) % 3
        eye = np.eye(3, dtype=np.int64)
        cells, corner_lo = [], []
        for o1, o2 in _RING:
            g = self.link_lower-o1*eye[a1]-o2*eye[a2]
            k = self.key(g)
            position = np.minimum(np.searchsorted(self.cell_keys, k), self.n_cells-1)
            if not np.all(self.cell_keys[position] == k):
                raise AssertionError("Every link with a present endpoint is an edge of four present cells")
            cells.append(position)
            corner_lo.append((o1 << a1) | (o2 << a2))
        self.link_cells = np.stack(cells, 1)                                 # (L, 4)
        self.link_corner_lo = np.stack(corner_lo, 1)                         # (L, 4)
        self.link_corner_hi = self.link_corner_lo | (1 << a)[:, None]

    def neighbours(self):
        """(N, 6) particle index of each 6-neighbour (-x, +x, -y, +y, -z, +z); -1 absent (cut_events order)."""
        out = np.full((self.n, 6), -1, np.int64)
        for j, axis in enumerate((0, 0, 1, 1, 2, 2)):
            step = np.eye(3, dtype=np.int64)[axis]*(1 if j % 2 else -1)
            out[:, j] = self.lookup(self.shifted+step)
        return out

    def sha256(self):
        return {"lattice_sha256": _sha(self.lattice.astype(np.int32)), "reference_sha256": _sha(self.reference)}


def link_states(lattice, codes):
    """Per link: state (uncut / boundary / cut) and the conflicting cut bit set (bit 2k for cut k)."""
    lo, hi = lattice.link_lo, lattice.link_hi
    both = (lo >= 0) & (hi >= 0)
    a1, a2 = side_masks(codes)
    cutbits = np.zeros(lattice.n_links, np.int64)
    a, b = lo[both], hi[both]
    cutbits[both] = (a1[a] & a2[b]) | (a2[a] & a1[b])
    state = np.where(both, np.where(cutbits != 0, LINK_CUT, LINK_UNCUT), LINK_BOUNDARY).astype(np.int8)
    return state, cutbits


# ----------------------------------------------------------------------------- Hermite data

def _log_links(lattice, log):
    """Map cut_events_v1 log rows to (link index, t along lower -> upper, valid); invalid rows are counted."""
    p, q = np.asarray(log["p"], np.int64), np.asarray(log["q"], np.int64)
    ok = (p >= 0) & (p < lattice.n) & (q >= 0) & (q < lattice.n)
    p, q = np.where(ok, p, 0), np.where(ok, q, 0)
    diff = lattice.lattice[q]-lattice.lattice[p]
    unit = np.abs(diff).sum(1) == 1
    ok &= unit
    axis = np.argmax(np.abs(diff), 1)
    p_is_lower = diff[np.arange(len(p)), axis] > 0
    lower = np.where(p_is_lower[:, None], lattice.shifted[p], lattice.shifted[q])
    k = lattice.key(lower)*3+axis
    position = np.minimum(np.searchsorted(lattice.link_keys, k), lattice.n_links-1)
    ok &= lattice.link_keys[position] == k
    t = np.asarray(log["t"], np.float64)
    t = np.where(p_is_lower, t, 1-t)
    return np.where(ok, position, -1), t, ok


FALLBACK_COHERENCE_MIN = .5   # a planar staircase gives |sum d| / count >= 1/sqrt(3) ~ 0.577 for any plane


def _fallback_normals(lattice, links, digits_lo, radius_spacings):
    """R4 normal estimate for the conflicting links of one cut: normalised sum of the signed link directions
    (digit 1 -> digit 2) of the same cut's conflicting links whose midpoints lie within the radius.

    The directions are 0/+-1 vectors, so the sums are exact integers (order independent). A planar link
    staircase has coherence |sum d| / count >= 1/sqrt(3) for any normal; below ``FALLBACK_COHERENCE_MIN`` the
    local pattern is not planar (one-particle fragments, chip edges, corners: opposite directions cancel) and the
    link's own axis direction (the cell-face normal, consistent with the full-cell mass) is used and counted."""
    from scipy.spatial import cKDTree
    axis = lattice.link_axis[links]
    sign = np.where(digits_lo == 1, 1., -1.)
    direction = np.zeros((len(links), 3))
    direction[np.arange(len(links)), axis] = sign
    midpoint = 2*lattice.link_lower[links]+np.eye(3, dtype=np.int64)[axis]   # doubled lattice units, integers
    total = direction.copy()
    count = np.ones(len(links))
    if len(links) > 1:
        pairs = cKDTree(midpoint).query_pairs(2*radius_spacings+1e-9, output_type="ndarray")
        if len(pairs):
            np.add.at(total, pairs[:, 0], direction[pairs[:, 1]])
            np.add.at(total, pairs[:, 1], direction[pairs[:, 0]])
            count += np.bincount(pairs.ravel(), minlength=len(links))
    length = np.linalg.norm(total, axis=1)
    incoherent = length < FALLBACK_COHERENCE_MIN*count
    total[incoherent] = direction[incoherent]
    length[incoherent] = 1.
    return total/length[:, None], int(incoherent.sum())


def boundary_hermite(lattice, profile, shape=None):
    """Static boundary-link Hermite data (t along lower -> upper, normal pointing out of the material)."""
    if profile not in BOUNDARY_PROFILES:
        raise ValueError(f"boundary_profile must be one of {BOUNDARY_PROFILES}")
    links = np.flatnonzero((lattice.link_lo < 0) ^ (lattice.link_hi < 0))
    lo_present = lattice.link_lo[links] >= 0
    axis = lattice.link_axis[links]
    normal = np.zeros((len(links), 3))
    normal[np.arange(len(links)), axis] = np.where(lo_present, 1., -1.)
    t = np.full(len(links), .5)
    source = np.full(len(links), SRC_CELL_FACE, np.int8)
    if profile == "analytic_reference_v1":
        if shape is None:
            raise ValueError("analytic_reference_v1 needs the analytic initial shape (AnalyticBox, "
                             "AnalyticHeightfield or analytic_shape_from_workpiece)")
        lo_x, hi_x = lattice.link_lo_reference[links], lattice.link_hi_reference[links]
        inside = np.where(lo_present[:, None], lo_x, hi_x)
        outside = np.where(lo_present[:, None], hi_x, lo_x)
        tt, nn, consistent = analytic_crossings(shape, inside, outside)
        t = np.where(consistent, np.where(lo_present, tt, 1-tt), t)
        normal = np.where(consistent[:, None], nn, normal)
        source = np.where(consistent, SRC_ANALYTIC, SRC_ANALYTIC_INCONSISTENT).astype(np.int8)
    return links, t, normal, source


# ----------------------------------------------------------------------------- Layer 4 helpers

def cell_regions(present, connected):
    """Min-corner labels of the connected components of present corners under the connected edges.

    present (m, 8) bool, connected (m, 12) bool -> labels (m, 8) int8 (8 for absent corners)."""
    label = np.where(present, np.arange(8, dtype=np.int8), np.int8(8)).astype(np.int8)
    while True:
        before = label.copy()
        for e, (i, j, _) in enumerate(CELL_EDGES):
            c = connected[:, e]
            m = np.minimum(label[:, i], label[:, j])
            label[c, i] = m[c]
            label[c, j] = m[c]
        if np.array_equal(before, label):
            return label


def solve_qef(points, normals, weights, truncation=.1):
    """Batched QEF min sum w (n.(x-p))^2 toward the mass point (float64 SVD, relative truncation).

    points/normals (R, K, 3), weights (R, K) -> (x (R, 3), rank (R,))."""
    w = weights[..., None]
    total = weights.sum(1)
    mass = (points*w).sum(1)/np.maximum(total, 1e-300)[:, None]
    A = normals*w
    rhs = (A*(points-mass[:, None])).sum(-1)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > truncation*S[:, :1]
    inverse = np.where(keep, 1/np.where(keep, S, 1.), 0.)
    y = np.einsum("rji,rj,rkj,rk->ri", Vt, inverse, U, rhs)
    return mass+y, keep.sum(1)


# ----------------------------------------------------------------------------- mesh utilities

def triangle_normals(vertices, faces):
    v = np.asarray(vertices, np.float64)[faces]
    return np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])


def signed_volume(vertices, faces):
    v = np.asarray(vertices, np.float64)[faces]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()/6)


def mesh_shells(n_vertices, faces):
    """Connected components of the triangle graph (labels per vertex, -1 for unused vertices)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not len(faces):
        return 0, np.full(n_vertices, -1)
    rows = np.r_[faces[:, 0], faces[:, 1], faces[:, 2]]
    cols = np.r_[faces[:, 1], faces[:, 2], faces[:, 0]]
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_vertices,)*2).tocsr()
    _, labels = connected_components(graph, directed=False)
    used = np.zeros(n_vertices, bool)
    used[faces.ravel()] = True
    unique, labels_used = np.unique(labels[used], return_inverse=True)
    out = np.full(n_vertices, -1)
    out[used] = labels_used
    return len(unique), out


def mesh_validity(vertices, faces, *, area_eps=0.):
    """Edge/vertex manifoldness, orientation, boundary edges, degenerate triangles, Euler characteristic."""
    faces = np.asarray(faces, np.int64)
    out = {"vertices": int(len(vertices)), "triangles": int(len(faces))}
    if not len(faces):
        return {**out, "edges": 0, "boundary_edges": 0, "non_manifold_edges": 0, "inconsistently_oriented_edges": 0,
                "non_manifold_vertices": 0, "degenerate_triangles": 0, "unused_vertices": int(len(vertices)),
                "euler_characteristic": 0}
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, 1)
    edges, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    area = np.linalg.norm(triangle_normals(vertices, faces), axis=1)/2
    # vertex umbrellas: corners (t, v) of triangles sharing a manifold edge (v, w) are joined
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = len(faces)
    face_of = np.tile(np.arange(F), 3)
    two = np.flatnonzero(counts[inverse] == 2)
    order = np.argsort(inverse[two], kind="stable")
    pairs = two[order].reshape(-1, 2)
    corner = lambda tri, vert: tri*3+np.argmax(faces[tri] == vert[:, None], 1)  # noqa: E731
    rows, cols = [], []
    for k in range(2):
        v = undirected[pairs[:, 0], k]
        rows.append(corner(face_of[pairs[:, 0]], v)); cols.append(corner(face_of[pairs[:, 1]], v))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(3*F, 3*F)).tocsr()
    _, fan = connected_components(graph, directed=False)
    fans = np.unique(np.stack([faces.reshape(-1), fan], 1), axis=0)[:, 0]
    fans_per_vertex = np.bincount(fans, minlength=len(vertices))
    used = np.zeros(len(vertices), bool)
    used[faces.ravel()] = True
    out.update(edges=int(len(edges)), boundary_edges=int((counts == 1).sum()),
               non_manifold_edges=int((counts > 2).sum()),
               inconsistently_oriented_edges=int((directed_counts > 1).sum()),
               non_manifold_vertices=int((fans_per_vertex > 1).sum()),
               degenerate_triangles=int((area <= area_eps).sum()), unused_vertices=int((~used).sum()),
               euler_characteristic=int(used.sum()-len(edges)+F))
    return out


def face_varying_normals(vertices, faces, crease_angle_deg=30.):
    """(3F, 3) unit normals per triangle corner: area-weighted mean of the normals of the triangles sharing the
    corner's vertex whose normal is within ``crease_angle_deg`` of this triangle's normal.

    Averaging happens only among triangles sharing a vertex index. cut_aware_surface_v1 vertices belong to one
    local material region, so normals are never averaged across regions (and never across a cut)."""
    faces = np.asarray(faces, np.int64)
    n = triangle_normals(vertices, faces)
    length = np.linalg.norm(n, axis=1)
    unit = n/np.where(length > 0, length, 1.)[:, None]
    corners_vertex = faces.reshape(-1)
    corners_face = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(corners_vertex, kind="stable")
    sorted_vertex = corners_vertex[order]
    start = np.searchsorted(sorted_vertex, corners_vertex, side="left")
    stop = np.searchsorted(sorted_vertex, corners_vertex, side="right")
    cos_limit = np.cos(np.radians(crease_angle_deg))
    out = np.zeros((len(corners_vertex), 3))
    degree = stop-start
    for k in range(int(degree.max()) if len(degree) else 0):
        has = k < degree
        other = corners_face[order[np.minimum(start+k, len(order)-1)]]
        accept = has & ((unit[corners_face]*unit[other]).sum(1) >= cos_limit)
        out += np.where(accept[:, None], n[other], 0.)
    out_length = np.linalg.norm(out, axis=1)
    bad = out_length <= 0
    out[bad] = unit[corners_face[bad]]
    out_length[bad] = 1.
    return out/out_length[:, None]


# ----------------------------------------------------------------------------- Layer 5: embedding

def lsq_affine_gradients(lattice, positions, codes, link_state, particles):
    """Per-particle least-squares affine gradient over code-compatible particles within two uncut links.

    Returns (F_est (len(particles), 3, 3), ok mask); not ok (fewer than four non-coplanar particles) means the
    caller uses a translation-only term."""
    from scipy.sparse import coo_matrix, identity
    from .cut_topology import conflicting_packed
    uncut = np.flatnonzero(link_state == LINK_UNCUT)
    a, b = lattice.link_lo[uncut], lattice.link_hi[uncut]
    adjacency = coo_matrix((np.ones(2*len(a)), (np.r_[a, b], np.r_[b, a])), shape=(lattice.n,)*2).tocsr()
    rows = (identity(lattice.n, format="csr")+adjacency)[particles]
    two = (rows@adjacency+rows).tocoo()
    p_local, q = two.row, two.col
    p = particles[p_local]
    keep = (p != q) & ~conflicting_packed(codes[p], codes[q])
    p_local, p, q = p_local[keep], p[keep], q[keep]
    dX = lattice.reference[q]-lattice.reference[p]
    dx = positions[q]-positions[p]
    m = len(particles)
    SXX = np.zeros((m, 3, 3))
    SxX = np.zeros((m, 3, 3))
    for i in range(3):
        for j in range(3):
            SXX[:, i, j] = np.bincount(p_local, weights=dX[:, i]*dX[:, j], minlength=m)
            SxX[:, i, j] = np.bincount(p_local, weights=dx[:, i]*dX[:, j], minlength=m)
    count = np.bincount(p_local, minlength=m)
    eig = np.linalg.eigvalsh(SXX)
    ok = (count >= 3) & (eig[:, 0] > 1e-6*np.maximum(eig[:, 2], 1e-300))
    F = np.tile(np.eye(3), (m, 1, 1))
    if ok.any():
        F[ok] = SxX[ok]@np.linalg.inv(SXX[ok])
    return F, ok


# ----------------------------------------------------------------------------- the surface

class CutAwareSurface:
    """``cut_aware_surface_v1`` mirroring ``cutting_surface.CuttingSurface``: ``extract(positions, codes, F)``
    returns (vertices (V, 3) float64, triangles (T, 3) int32, info). Vertices and triangles are grouped by current
    mechanical component (``info['component_ranges']`` as CuttingSurface). Per-element arrays of the last
    extraction are in ``self.last`` (oracle/debug data: keep out of policy observations).

    ``positions``, ``total_F`` and the reference lattice must share one frame, or pass ``frame_rotation`` R with
    positions = R (reference-frame motion) + t. ``cut_events_log`` is the cumulative ``cut_events_v1`` log
    (``CutEventRecorder.log()`` or the saved ``*_cut_events_log.npy``); without it the labelled midpoint fallback
    is used for every cut link.
    """

    profile = PROFILE

    def __init__(self, partition, spacing_m, lattice, reference_positions, *, boundary_profile="cell_face_v1",
                 analytic_shape=None, cut_events_log=None, fallback_normal_radius_spacings=3., qef_truncation=.1,
                 incremental=True, strict_audit=True, crease_angle_deg=30., masses_kg=None, density_kg_m3=None):
        if boundary_profile not in BOUNDARY_PROFILES:
            raise ValueError(f"boundary_profile must be one of {BOUNDARY_PROFILES}")
        self.partition = partition
        self.cuts = int(partition.cuts)
        self.encoding = partition.encoding
        self.lattice = ReferenceLattice(lattice, reference_positions, spacing_m)
        if len(partition.nodes) != self.lattice.n:
            raise ValueError("The mechanical partition and the lattice need the same particles")
        self.spacing = float(spacing_m)
        self.boundary_profile, self.analytic_shape = boundary_profile, analytic_shape
        self.cut_events_log = cut_events_log
        self.fallback_radius = float(fallback_normal_radius_spacings)
        self.truncation = float(qef_truncation)
        self.incremental, self.strict_audit = bool(incremental), bool(strict_audit)
        self.crease_angle_deg = float(crease_angle_deg)
        self.masses = None if masses_kg is None else np.asarray(masses_kg, np.float64)
        self.density = None if density_kg_m3 is None else float(density_kg_m3)
        start = time.perf_counter()
        links, t, normal, source = boundary_hermite(self.lattice, boundary_profile, analytic_shape)
        self._boundary = (links, t, normal, source)
        self.static_wall_s = time.perf_counter()-start
        self._cache = None
        self.last = None
        self.extractions = 0

    # -- Hermite signature of every link: (state, cutbits, lo point(3), lo normal(3), hi point(3), hi normal(3))
    def _hermite(self, codes, log):
        lat, s = self.lattice, self.spacing
        state, cutbits = link_states(lat, codes)
        L = lat.n_links
        t_lo, t_hi = np.zeros(L), np.zeros(L)
        n_lo, n_hi = np.zeros((L, 3)), np.zeros((L, 3))
        src_lo, src_hi = np.full(L, -1, np.int8), np.full(L, -1, np.int8)
        b_links, b_t, b_n, b_src = self._boundary
        t_lo[b_links] = t_hi[b_links] = b_t
        n_lo[b_links] = n_hi[b_links] = b_n
        src_lo[b_links] = src_hi[b_links] = b_src
        counts = {name: 0 for name in SOURCE_NAMES[SRC_RECORDER:]}
        log_info = {"given": log is not None, "entries": 0, "entries_not_links": 0, "entries_not_conflicting": 0,
                    "duplicate_entries": 0, "log_sha256": None}
        cut_links = np.flatnonzero(state == LINK_CUT)
        fallback_degenerate = 0
        lookup_keys = lookup_t = lookup_n = lookup_flags = None
        if log is not None:
            log_info.update(entries=int(len(log)), log_sha256=_sha(log))
            link, t_log, ok = _log_links(lat, log)
            cut = np.asarray(log["cut"], np.int64)
            log_info["entries_not_links"] = int((~ok).sum())
            ok &= (cut >= 0) & (cut < self.cuts)
            conflicting = np.zeros(len(link), bool)
            conflicting[ok] = ((cutbits[link[ok]] >> (2*cut[ok])) & 1) != 0
            log_info["entries_not_conflicting"] = int((ok & ~conflicting).sum())
            ok &= conflicting
            k = link[ok]*16+cut[ok]
            order = np.lexsort((np.arange(len(k)), k))
            k_sorted = k[order]
            first = np.r_[True, k_sorted[1:] != k_sorted[:-1]] if len(k_sorted) else np.zeros(0, bool)
            log_info["duplicate_entries"] = int((~first).sum())
            rows = np.flatnonzero(ok)[order[first]]
            lookup_keys = k_sorted[first]
            lookup_t = t_log[rows]
            lookup_n = np.asarray(log["normal_reference"], np.float64)[rows]
            lookup_flags = np.asarray(log["flags"], np.int64)[rows]
        multi = 0
        near_coincident = 0
        if len(cut_links):
            digits_lo = (codes[lat.link_lo[cut_links]][:, None] >> (2*np.arange(self.cuts))) & 3
            best_lo = np.full(len(cut_links), np.inf)
            best_hi = np.full(len(cut_links), -np.inf)
            crossings = np.zeros(len(cut_links), np.int64)
            t_min, t_max = np.full(len(cut_links), np.inf), np.full(len(cut_links), -np.inf)
            for k in range(self.cuts):
                has = ((cutbits[cut_links] >> (2*k)) & 1) != 0
                if not has.any():
                    continue
                rows = np.flatnonzero(has)
                links_k = cut_links[rows]
                normal, degenerate = _fallback_normals(lat, links_k, digits_lo[rows, k], self.fallback_radius)
                fallback_degenerate += degenerate
                t = np.full(len(rows), .5)
                source = np.full(len(rows), SRC_MIDPOINT, np.int8)
                if lookup_keys is not None and len(lookup_keys):
                    key = links_k*16+k
                    position = np.minimum(np.searchsorted(lookup_keys, key), len(lookup_keys)-1)
                    found = lookup_keys[position] == key
                    flags = np.where(found, lookup_flags[position], 0)
                    sign_bad = found & ((flags & FLAG_SIGN_INCONSISTENT) != 0)
                    normal_bad = found & ((flags & FLAG_DEGENERATE_REFERENCE_NORMAL) != 0)
                    t = np.where(found & ~sign_bad, lookup_t[position], t)
                    t = np.where(np.isfinite(t), t, .5)
                    normal = np.where((found & ~normal_bad)[:, None], lookup_n[position], normal)
                    source = np.select([sign_bad, normal_bad, found], [SRC_RECORDER_SIGN, SRC_RECORDER_NORMAL_FALLBACK,
                                                                       SRC_RECORDER], SRC_MIDPOINT).astype(np.int8)
                for code, name in ((SRC_RECORDER, "recorder"), (SRC_RECORDER_SIGN, "recorder_sign_inconsistent_midpoint"),
                                   (SRC_RECORDER_NORMAL_FALLBACK, "recorder_point_fallback_normal"),
                                   (SRC_MIDPOINT, "midpoint_fallback")):
                    counts[name] += int((source == code).sum())
                crossings[rows] += 1
                t_min[rows] = np.minimum(t_min[rows], t)
                t_max[rows] = np.maximum(t_max[rows], t)
                take_lo = t < best_lo[rows]          # strict: ties keep the lowest cut id
                take_hi = t > best_hi[rows]
                for take, best, tt, nn, ss in ((take_lo, best_lo, t_lo, n_lo, src_lo), (take_hi, best_hi, t_hi, n_hi, src_hi)):
                    r = rows[take]
                    best[r] = t[take]
                    tt[links_k[take]] = t[take]
                    nn[links_k[take]] = normal[take]
                    ss[links_k[take]] = source[take]
            multi = int((crossings > 1).sum())
            near_coincident = int(((crossings > 1) & (t_max-t_min < .25)).sum())
        e = np.eye(3)[lat.link_axis]
        p_lo = lat.link_lo_reference+(t_lo*s)[:, None]*e
        p_hi = lat.link_lo_reference+(t_hi*s)[:, None]*e
        disconnected = state != LINK_UNCUT
        signature = np.concatenate([p_lo, n_lo, p_hi, n_hi], 1)
        signature[~disconnected] = 0.
        stats = {"crossings_by_source": counts, "cut_links": int(len(cut_links)),
                 "boundary_links": int(len(b_links)),
                 "boundary_links_by_source": {SOURCE_NAMES[c]: int((b_src == c).sum()) for c in np.unique(b_src)},
                 "multi_crossing_links": multi, "near_coincident_crossings": near_coincident,
                 "fallback_normal_incoherent_links": int(fallback_degenerate),
                 "fallback_normal_coherence_min": FALLBACK_COHERENCE_MIN, "log": log_info}
        return state, cutbits, signature, (src_lo, src_hi), stats

    # -- Layer 4 on a set of cells
    def _cells(self, cells, state, signature):
        lat = self.lattice
        present = lat.corner_particle[cells] >= 0
        edge_link = lat.cell_edge_link[cells]
        edge_state = np.where(edge_link >= 0, state[np.maximum(edge_link, 0)], -1)
        connected = edge_state == LINK_UNCUT
        disconnected = (edge_state == LINK_BOUNDARY) | (edge_state == LINK_CUT)
        feature = disconnected.any(1)
        cells, present = cells[feature], present[feature]
        edge_link, connected, disconnected = edge_link[feature], connected[feature], disconnected[feature]
        label = cell_regions(present, connected)
        root = present & (label == np.arange(8))
        rep_row, rep_root = np.nonzero(root)
        mask = label[rep_row] == rep_root[:, None]                           # (R, 8) region corners
        R = len(rep_row)
        sig = signature[np.maximum(edge_link[rep_row], 0)]                   # (R, 12, 12)
        disc = disconnected[rep_row]
        in_i, in_j = mask[:, CELL_EDGES[:, 0]], mask[:, CELL_EDGES[:, 1]]
        same = np.all(sig[..., 0:6] == sig[..., 6:12], axis=-1)
        w_lo = disc & in_i
        w_hi = disc & in_j & ~(in_i & same)
        origin = lat.cell_origin[cells[rep_row]]
        points = np.concatenate([sig[..., 0:3], sig[..., 6:9]], 1)
        normals = np.concatenate([sig[..., 3:6], sig[..., 9:12]], 1)
        weights = np.concatenate([w_lo, w_hi], 1).astype(np.float64)
        local = (points-origin[:, None, :])/self.spacing
        x, rank = solve_qef(local, normals, weights, self.truncation)
        clamped = np.any((x < -1e-12) | (x > 1+1e-12), axis=1)
        x = np.clip(x, 0., 1.)
        return {"cell": cells[rep_row], "root": rep_root.astype(np.int8), "mask": mask,
                "X": origin+x*self.spacing, "rank": rank.astype(np.int8), "clamped": clamped,
                "hermite_rows": weights.sum(1).astype(np.int16), "feature_cells": cells}

    def _merge(self, keep, new):
        out = {k: np.concatenate([keep[k], new[k]]) for k in ("cell", "root", "mask", "X", "rank", "clamped",
                                                               "hermite_rows")}
        order = np.lexsort((out["root"], out["cell"]))
        return {k: v[order] for k, v in out.items()}

    def _topology(self, codes, log, force_full):
        state, cutbits, signature, sources, stats = self._hermite(codes, log)
        lat = self.lattice
        full = force_full or not self.incremental or self._cache is None
        if full:
            reps = self._cells(np.arange(lat.n_cells), state, signature)
            dirty = lat.n_cells
        else:
            cache = self._cache
            changed = ((state != cache["state"]) | (cutbits != cache["cutbits"])
                       | np.any(signature != cache["signature"], axis=1))
            dirty_cells = np.unique(lat.link_cells[changed].ravel())
            reps = cache["reps"]
            if len(dirty_cells):
                is_dirty = np.zeros(lat.n_cells, bool)
                is_dirty[dirty_cells] = True
                keep = {k: v[~is_dirty[reps["cell"]]] for k, v in reps.items()}
                reps = self._merge(keep, self._cells(dirty_cells, state, signature))
            dirty = int(len(dirty_cells))
        reps = {k: v for k, v in reps.items() if k != "feature_cells"}
        self._cache = {"state": state, "cutbits": cutbits, "signature": signature, "reps": reps}
        return state, cutbits, sources, stats, reps, {"mode": "full" if full else "incremental",
                                                     "dirty_cells": int(dirty), "cells": int(lat.n_cells)}

    def _quads(self, state, cutbits, reps):
        lat = self.lattice
        region_of = np.full((lat.n_cells, 8), -1, np.int64)
        r, c = np.nonzero(reps["mask"])
        region_of[reps["cell"][r], c] = r
        links = np.flatnonzero(state != LINK_UNCUT)
        cells = lat.link_cells[links]
        faces, face_link, face_side = [], [], []
        for side, corner, present in ((0, lat.link_corner_lo, lat.link_lo), (1, lat.link_corner_hi, lat.link_hi)):
            has = present[links] >= 0
            sel = links[has]
            quad = region_of[cells[has], corner[sel]]
            if np.any(quad < 0):
                raise AssertionError("A disconnected link side has no region representative")
            if side == 1:
                quad = quad[:, ::-1]
            faces.append(np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]))
            face_link.append(np.r_[sel, sel])
            face_side.append(np.r_[present[sel], present[sel]])
        # order: lower-side quads (links in key order; first triangles, then second triangles), then upper sides
        return np.concatenate(faces).astype(np.int64), np.concatenate(face_link), np.concatenate(face_side)

    def _embed(self, reps, positions, F, codes, state, rotation):
        lat = self.lattice
        corners = lat.corner_particle[reps["cell"]]                           # (R, 8)
        mask = reps["mask"]
        X = reps["X"]
        u = (X-lat.cell_origin[reps["cell"]])/self.spacing
        w = np.prod(np.where(CORNER_OFFSETS[None].astype(bool), u[:, None, :], 1-u[:, None, :]), axis=2)*mask
        total = w.sum(1)
        zero = total <= 1e-12
        w[zero] = mask[zero]
        w = w/w.sum(1, keepdims=True)
        idx = np.where(mask, corners, 0)
        d = X[:, None, :]-lat.reference[idx]                                 # (R, 8, 3)
        estimator = "total_F" if F is not None else "lsq_affine"
        counts = {"estimator": estimator, "representatives": int(len(X)), "zero_weight_uniform_fallback": int(zero.sum())}
        if F is not None:
            Fp = F[idx]
            det = np.linalg.det(Fp)
            bad = (det <= 0) | ~np.isfinite(det)
            used = mask & bad
            Fp = np.where(bad[..., None, None], np.eye(3), Fp)
            counts.update(det_F_nonpositive_particle_terms=int(used.sum()),
                          det_F_nonpositive_particles=int(len(np.unique(idx[used]))),
                          translation_only_representatives=int(np.any(used, 1).sum()))
        else:
            particles = np.unique(idx[mask])
            Fe, ok = lsq_affine_gradients(lat, positions, codes, state, particles)
            table = np.zeros(lat.n, np.int64)
            table[particles] = np.arange(len(particles))
            Fp = Fe[table[idx]]
            not_ok = ~ok[table[idx]] & mask
            counts.update(lsq_particles=int(len(particles)), lsq_translation_only_particles=int((~ok).sum()),
                          translation_only_representatives=int(np.any(not_ok, 1).sum()),
                          lsq_neighbourhood="code-compatible particles within two uncut lattice links")
            if rotation is not None:   # translation-only LSQ terms still need the frame rotation
                Fp = np.where(not_ok[..., None, None], rotation, Fp)
        if rotation is not None and F is not None:
            # reference-frame F -> positions frame; the LSQ estimate already maps reference to positions frame
            Fp = np.einsum("ij,rcjk->rcik", rotation, Fp)
        term = positions[idx]+np.einsum("rcij,rcj->rci", Fp, d)
        return (w[..., None]*term).sum(1), counts

    def extract(self, positions_m, current_codes, total_F=None, *, cut_events_log=None, frame_rotation=None,
                full_rebuild=False):
        timings = {}
        start = time.perf_counter()
        lat = self.lattice
        positions = np.asarray(positions_m, np.float64)
        if positions.shape != (lat.n, 3) or not np.isfinite(positions).all():
            raise ValueError("Expected current finite particle positions matching the lattice")
        F = None
        if total_F is not None:
            F = np.asarray(total_F, np.float64)
            if F.shape != (lat.n, 3, 3):
                raise ValueError("total_F needs one 3x3 deformation gradient per particle")
        rotation = None if frame_rotation is None else np.asarray(frame_rotation, np.float64)
        codes_in = np.asarray(current_codes)
        count, labels, provenance = self.partition.partition(codes_in)       # validates the codes
        codes = (codes_in.astype(np.int64) if self.encoding == "packed2" else
                 ternary_to_packed(codes_in, self.cuts))
        timings["mechanical_partition_s"] = time.perf_counter()-start
        log = self.cut_events_log if cut_events_log is None else cut_events_log
        t0 = time.perf_counter()
        state, cutbits, (src_lo, src_hi), hermite_stats, reps, update = self._topology(codes, log, full_rebuild)
        timings["topology_and_qef_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        faces, face_link, face_side = self._quads(state, cutbits, reps)
        timings["quads_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        current, embed_counts = self._embed(reps, positions, F, codes, state, rotation)
        timings["embedding_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        # group by mechanical component (CuttingSurface-compatible component_ranges)
        first_particle = lat.corner_particle[reps["cell"], reps["root"].astype(np.int64)]
        vertex_component = labels[first_particle]
        vorder = np.argsort(vertex_component, kind="stable")
        remap = np.empty_like(vorder)
        remap[vorder] = np.arange(len(vorder))
        faces = remap[faces]
        face_component = vertex_component[vorder][faces[:, 0]]
        forder = np.argsort(face_component, kind="stable")
        faces, face_link, face_side, face_component = faces[forder], face_link[forder], face_side[forder], face_component[forder]
        reps_sorted = {k: v[vorder] for k, v in reps.items()}
        current, reference_vertices = current[vorder], reps_sorted["X"]
        vertex_component = vertex_component[vorder]
        first_particle = first_particle[vorder]
        ranges, vfirst, ffirst = [], 0, 0
        vcount = np.bincount(vertex_component, minlength=count)
        fcount = np.bincount(face_component, minlength=count)
        for label in range(count):
            ranges.append({"mechanical_component": int(label), "particles": int((labels == label).sum()),
                           "first_vertex": int(vfirst), "vertices": int(vcount[label]),
                           "first_triangle": int(ffirst), "triangles": int(fcount[label])})
            vfirst += int(vcount[label]); ffirst += int(fcount[label])
        audit = self._audit(codes, labels, count, state, faces, vertex_component, first_particle, reference_vertices,
                            current, face_component)
        timings["audit_s"] = time.perf_counter()-t0
        face_kind = state[face_link]
        validity = mesh_validity(reference_vertices, faces)
        timings["total_s"] = time.perf_counter()-start
        self.extractions += 1
        info = {**provenance, "surface_profile": PROFILE, "profile_version": PROFILE,
                "mechanical_components": int(count), "component_ranges": ranges,
                "renderer_partition_changes_physics": False, "writes_physics_state": False,
                "codes_sha256": hashlib.sha256(codes_in.tobytes()).hexdigest(),
                **lat.sha256(), "stencil_sha256": self.partition.stencil_sha256,
                "reference_grid": {"kind": "particle lattice (linked volume)", "spacing_m": self.spacing,
                                   "sites": int(lat.n), "links": int(lat.n_links), "dual_cells": int(lat.n_cells),
                                   "lattice_residual_m": lat.lattice_residual_m},
                "link_rule": "cut iff endpoint packed codes conflict; boundary iff one endpoint site absent",
                "boundary_hermite_profile": self.boundary_profile,
                "analytic_shape": None if self.analytic_shape is None else self.analytic_shape.to_dict(),
                "hermite": {"source": "cut_events_v1" if log is not None else "midpoint_fallback_R4",
                            "fallback_normal": "sum of signed same-cut link directions within radius",
                            "fallback_normal_radius_spacings": self.fallback_radius, **hermite_stats},
                "qef": {"solver": "float64 SVD, cell-local coordinates, toward mass point",
                        "truncation_relative": self.truncation, "representatives": int(len(reps["X"])),
                        "clamped": int(reps["clamped"].sum()),
                        "rank_counts": {str(k): int((reps["rank"] == k).sum()) for k in (0, 1, 2, 3)},
                        "representatives_without_hermite": int((reps["hermite_rows"] == 0).sum())},
                "regions": {"feature_cells": int(len(np.unique(reps["cell"]))),
                            "multi_region_cells": int((np.bincount(reps["cell"]) > 1).sum()),
                            "mixed_side_regions": int(self._mixed_side_regions(codes, reps))},
                "faces": {"triangles": int(len(faces)), "cut_triangles": int((face_kind == LINK_CUT).sum()),
                          "boundary_triangles": int((face_kind == LINK_BOUNDARY).sum())},
                "embedding": embed_counts, "mesh_validity_reference": validity, "audit": audit,
                "update": update, "timings_s": timings,
                "scope": "Render geometry from solver state (one-way). Not the contact surface; contact uses particles."}
        self.last = {"vertex_reference": reference_vertices, "vertex_component": vertex_component,
                     "vertex_cell": reps_sorted["cell"], "vertex_root": reps_sorted["root"],
                     "vertex_corner_mask": reps_sorted["mask"], "vertex_clamped": reps_sorted["clamped"],
                     "vertex_rank": reps_sorted["rank"], "vertex_first_particle": first_particle,
                     "face_link": face_link, "face_kind": face_kind, "face_side_particle": face_side,
                     "face_cutbits": cutbits[face_link], "face_component": face_component,
                     "link_state": state, "link_cutbits": cutbits}
        return current, faces.astype(np.int32), info

    def region_particles(self, vertex):
        """Particle indices of the local material region of one output vertex (from ``self.last``)."""
        cell = self.last["vertex_cell"][vertex]
        return self.lattice.corner_particle[cell][self.last["vertex_corner_mask"][vertex]]

    def _mixed_side_regions(self, codes, reps):
        corners = self.lattice.corner_particle[reps["cell"]]
        a1, a2 = side_masks(np.where(reps["mask"], codes[np.maximum(corners, 0)], 0))
        any1 = np.bitwise_or.reduce(a1, axis=1)
        any2 = np.bitwise_or.reduce(a2, axis=1)
        return int(((any1 & any2) != 0).sum())

    def _audit(self, codes, labels, count, state, faces, vertex_component, first_particle, reference_vertices,
               current_vertices, face_component):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        lat = self.lattice
        uncut = np.flatnonzero(state == LINK_UNCUT)
        a, b = lat.link_lo[uncut], lat.link_hi[uncut]
        graph = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(lat.n,)*2).tocsr()
        pieces, piece = connected_components(graph, directed=False)
        pairs = np.unique(np.stack([piece, labels], 1), axis=0)
        piece_violations = int(len(pairs)-pieces)
        shells, shell = mesh_shells(len(reference_vertices), faces)
        vertex_piece = piece[first_particle]
        used = shell >= 0
        shell_pairs = np.unique(np.stack([shell[used], vertex_component[used]], 1), axis=0)
        shell_piece_pairs = np.unique(np.stack([shell[used], vertex_piece[used]], 1), axis=0)
        shell_violations = int(len(shell_pairs)-shells)
        errors = []
        if piece_violations:
            errors.append(f"{piece_violations} lattice pieces span several mechanical components")
        if shell_violations:
            errors.append(f"{shell_violations} mesh shells span several mechanical components")
        if len(shell_piece_pairs) != shells:
            errors.append("a mesh shell spans several lattice pieces (region construction error)")
        pieces_per_component = np.bincount(pairs[:, 1], minlength=count)
        shells_per_component = np.bincount(shell_pairs[:, 1], minlength=count) if shells else np.zeros(count, int)
        without_surface = np.flatnonzero(shells_per_component == 0)
        if len(without_surface):
            errors.append(f"{len(without_surface)} mechanical components have no surface (dropped fragment)")
        face_shell = shell[faces[:, 0]] if len(faces) else np.zeros(0, int)
        vol_ref = np.zeros(shells)
        vol_cur = np.zeros(shells)
        if len(faces):
            vr = reference_vertices[faces]
            vc = current_vertices[faces]
            np.add.at(vol_ref, face_shell, np.einsum("ij,ij->i", vr[:, 0], np.cross(vr[:, 1], vr[:, 2]))/6)
            np.add.at(vol_cur, face_shell, np.einsum("ij,ij->i", vc[:, 0], np.cross(vc[:, 1], vc[:, 2]))/6)
        shell_component = np.full(shells, -1)
        shell_component[shell_pairs[:, 0]] = shell_pairs[:, 1]
        component_volume_ref = np.bincount(shell_component, weights=vol_ref, minlength=count) if shells else np.zeros(count)
        component_volume_cur = np.bincount(shell_component, weights=vol_cur, minlength=count) if shells else np.zeros(count)
        particles_per_component = np.bincount(labels, minlength=count)
        cell_volume = self.spacing**3
        components = []
        for label in range(count):
            row = {"mechanical_component": int(label), "particles": int(particles_per_component[label]),
                   "lattice_pieces": int(pieces_per_component[label]), "shells": int(shells_per_component[label]),
                   "represented_volume_reference_m3": float(component_volume_ref[label]),
                   "represented_volume_current_m3": float(component_volume_cur[label]),
                   "cell_volume_m3": float(particles_per_component[label]*cell_volume)}
            if self.masses is not None and self.density is not None:
                row["mass_over_density_m3"] = float(self.masses[labels == label].sum()/self.density)
            components.append(row)
        bridged = [row for row in components if row["lattice_pieces"] > 1]
        bridge = None
        if bridged:
            from .law8_validation import bridge_report
            bridge = [bridge_report(self.partition.nodes, self.partition.weights, self.partition.gradients, codes,
                                    digit_index=k, cuts=self.cuts, packed=True, reference_positions=lat.reference)
                      for k in range(self.cuts)]
        audit = {"global_piece_count_source": "mechanical_components_packed (current compatible reference stencils)",
                 "mechanical_components": int(count), "lattice_pieces": int(pieces), "mesh_shells": int(shells),
                 "piece_component_violations": piece_violations, "shell_component_violations": shell_violations,
                 "visually_separated_mechanically_bridged": [{**row, "rule": "one mechanical component, several "
                     "lattice pieces (code-0 or diagonal stencil bridge): shown, never hidden"} for row in bridged],
                 "bridge_report": bridge, "components": components if count <= 64 else None,
                 "shell_volumes_reference_m3": vol_ref.tolist() if shells <= 64 else None,
                 "errors": errors, "ok": not errors}
        if errors and self.strict_audit:
            raise CutAwareTopologyError("; ".join(errors))
        return audit

    def face_varying_normals(self, vertices, faces):
        return face_varying_normals(vertices, faces, self.crease_angle_deg)


# ----------------------------------------------------------------------------- live adapter and selection

class CutAwareComponentSurface:
    """Live-bed adapter mirroring ``cutting_surface.ComponentSurface``: ``extract(bed, recorder=None)`` reads
    ``bed.reference``, ``bed.x``, ``bed.F``, ``bed.code`` (and ``recorder.log()`` when a cut_events_v1 recorder is
    attached). Reads only; the stencil and lattice tables are built once per bed."""

    profile = PROFILE

    def __init__(self, spacing, lattice, *, boundary_profile="cell_face_v1", analytic_shape=None, **options):
        self.spacing, self.lattice = float(spacing), np.asarray(lattice)
        self.boundary_profile, self.analytic_shape, self.options = boundary_profile, analytic_shape, options
        self.surface = None
        self.bed_identity = None

    def extract(self, bed, recorder=None):
        from .cutting_surface import CurrentMechanicalPartition
        if self.surface is None or self.bed_identity != id(bed):
            partition = CurrentMechanicalPartition(bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy(),
                                                   bed.config.cuts, encoding=getattr(bed, "code_encoding", "ternary"))
            self.surface = CutAwareSurface(partition, self.spacing, self.lattice, bed.reference.numpy(),
                                           boundary_profile=self.boundary_profile, analytic_shape=self.analytic_shape,
                                           **self.options)
            self.bed_identity = id(bed)
        log = None if recorder is None else recorder.log()
        vertices, faces, info = self.surface.extract(bed.x.numpy(), bed.code.numpy(), bed.F.numpy(), cut_events_log=log)
        info["surface_selection"] = surface_selection_record(PROFILE, boundary_profile=self.boundary_profile,
                                                             cut_events="cut_events_v1" if recorder is not None else None)
        return vertices, faces, info


def surface_selection_record(profile, **options):
    """Episode-metadata record of the selected surface profile (store it in the episode receipt)."""
    if profile not in SURFACE_PROFILES:
        raise ValueError(f"Surface profile must be one of {SURFACE_PROFILES}")
    record = {"surface_profile": profile, "baseline_profile": BASELINE_PROFILE,
              "renderer_partition_changes_physics": False}
    if profile == BASELINE_PROFILE:
        record["implementation"] = "cutting_surface.CuttingSurface (Newton ParticleSurface per mechanical component)"
    else:
        record["implementation"] = "cut_aware_surface.CutAwareSurface (lattice linked volume + local DC)"
    record["options"] = {k: v for k, v in sorted(options.items())}
    return record


def make_component_surface(profile, spacing, *, lattice=None, device="cuda:0", **options):
    """Live surface adapter by profile name: ``particle_surface_v1`` -> ComponentSurface (options: voxel_size,
    profile=legacy|current_volume_v1, unmeshable, substrate_anchors); ``cut_aware_surface_v1`` ->
    CutAwareComponentSurface (needs the reference lattice)."""
    if profile == BASELINE_PROFILE:
        from .cutting_surface import ComponentSurface
        kernel = options.pop("kernel_profile", "legacy")
        return ComponentSurface(spacing, device, profile=kernel, **options)
    if profile == PROFILE:
        if lattice is None:
            raise ValueError("cut_aware_surface_v1 needs the reference lattice; use particle_surface_v1 without one")
        return CutAwareComponentSurface(spacing, lattice, **options)
    raise ValueError(f"Surface profile must be one of {SURFACE_PROFILES}")
