"""``cut_aware_surface_v2``: ``cut_aware_surface_v1`` with the method fixes M1-M5 of the Phase 3 decision review.

WP-12 Phase 3 (``docs/reports/WP12_PHASE3_DECISION_REVIEW.md`` section 2). Opt-in and versioned:
``cut_aware_surface.py`` (v1) is not modified, stays byte-identical and stays the default for every existing record.
This module subclasses ``CutAwareSurface`` and overrides only the steps the fixes touch.

Common rules (review section 2, unchanged from v1): render geometry only; the surface reads Layer-1 state (reference
lattice, positions, total F, packed codes, the ``cut_events_v1`` Hermite log) and writes nothing back; no target plane,
no blade pose beyond the recorder's Hermite data, no particle edit, and a cut face exists only across a conflicting
lattice link. Not allowed (and not done): smoothing cut-face vertices, inward offsets, snapping to the swept plane,
choosing an estimator because it scores better. Every activation of every fix is counted in ``info['v2']``.

M1 constrained crease QEF
    v1 solves the QEF unconstrained (truncated SVD toward the mass point) and clips to the dual cell. v2 keeps the v1
    solution wherever it lies in the cell (bit-identical there) and otherwise solves the box-constrained problem
    exactly: min ||A_k (x - m) - b_k||^2 over x in [0, 1]^3 (cell-local), where A_k, b_k are the same rank-truncated
    QEF that v1 minimises. Exact active-set enumeration over the 26 faces, edges and vertices of the cell (each free
    sub-problem solved by the minimum-norm least squares toward the mass point); the feasible candidate with the lowest
    truncated QEF value wins, ties (``QEF_TIE_TOLERANCE``) go to the candidate nearest the mass point. Counted:
    representatives constrained, QEF value of the clip vs the constrained optimum, distance moved from the clip.
M2 crack-front hinge placement
    A region that holds both side digits of a cut (an open crack front, ``mixed`` in v1) and at least one corner whose
    digit is 0 for every such cut (the joined hinge material below the deepest conflicting row) has its representative
    moved along the null space of its truncated QEF (in the cut plane, so its distance to the Hermite plane is
    unchanged) toward the mean of those hinge corners, by ``HINGE_FRACTION`` of the null-space offset, limited to the
    cell. The restricted trilinear embedding weights then include the joined material. One shared tip vertex per
    region is kept (``tip_representatives_shared`` and 0 boundary edges keep their v1 meaning); no face is created on
    a non-conflicting link and no side is opened beyond the conflicting links. Counted: regions moved, regions mixed
    without a hinge corner, shift lengths.
M3 fold guard (report first)
    After embedding, every lattice triangle's current normal is compared with its reference normal mapped by cof(F)
    of the triangle's side particle (positions frame). A triangle is flipped when n_current . (cof F n_reference) <= 0.
    The representatives of flipped triangles are re-embedded with the existing ``lsq_affine`` estimator (v1
    ``lsq_affine_gradients``: code-compatible particles within two uncut links); representatives still incident to a
    flipped triangle are then embedded translation-only. Never triggered where the side particle or a region particle
    has det F <= 0: that is reported as material inversion. Counted: flipped triangles before, after LSQ, after
    translation-only; representatives per stage; inversion-skipped triangles. ``self.last['vertices_before_fold_guard']``
    keeps the pre-guard positions so self-intersections can be reported before and after.
M4 unlabelled-separation detector + labelled particle-cell fallback (``unlabelled_separation_v1``)
    An uncut link (both sites present, codes not conflicting) is flagged when its current length > ``SEPARATION_LENGTH_S``
    s AND |dx - F_mean dX| > ``SEPARATION_INCONSISTENCY_S`` s (F_mean = mean of the endpoints' F; the second test
    measures inconsistency with F, so a rubbery stretch that F records does not trigger). The endpoints of flagged links
    are drawn as per-particle closed cells (cube of volume (m/rho) det F, rotation = polar factor of F, centred on x_p;
    reference cube of side s), labelled ``particle_cell`` (``face_kind`` FACE_PARTICLE_CELL): never cut faces, never in
    cut-face metrics. For the lattice-faced surface those sites are treated as absent: links from a kept site to a
    fallback site that were uncut get a ``cell_face_v1`` boundary face; conflicting links keep their cut face on the
    kept side. Every activation is a solver-state anomaly for Phase 4 (positions moved without F or codes recording
    it). Not a fallback to particle_surface_v1. Disabled (counted) when no total F is given.
M5 position-consistent embedding (separately labelled variant ``cut_aware_surface_v2_position_consistent``)
    Only with ``embedding='position_consistent_v1'``: a particle term uses its ``lsq_affine`` gradient instead of its
    total F where |det F_lsq / det F - 1| > ``POSITION_CONSISTENT_DET_TOLERANCE`` (the positions disagree with F, review
    F3/F12). Never the default.

Frozen parameters: ``FROZEN_PARAMETERS`` (declared with the code, before any measurement on physical states; sources
in the dict). Lengths in spacings s unless marked.
"""
import itertools
import time

import numpy as np

from . import cut_aware_surface as V1
from .cut_aware_surface import (CELL_EDGES, CORNER_OFFSETS, LINK_BOUNDARY, LINK_CUT, LINK_UNCUT, CutAwareSurface,
                                cell_regions, lsq_affine_gradients, mesh_validity, solve_qef)
from .cut_topology import side_masks, ternary_to_packed

PROFILE = "cut_aware_surface_v2"
PROFILE_POSITION_CONSISTENT = "cut_aware_surface_v2_position_consistent"
FALLBACK_LABEL = "unlabelled_separation_v1"
EMBEDDINGS = ("total_F_v1", "position_consistent_v1")

LINK_ABSENT = 3            # v2 only: no effective endpoint (both sites absent or in the particle-cell fallback)
FACE_PARTICLE_CELL = 4     # face_kind of the M4 particle cells (never LINK_CUT, never LINK_BOUNDARY)

QEF_TIE_TOLERANCE = 1e-9            # absolute + relative, truncated QEF value in cell-local units^2
HINGE_FRACTION = 1.0                # fraction of the null-space offset toward the hinge-corner mean
SEPARATION_LENGTH_S = 2.0           # review section 2 M4: current length > 2 s
SEPARATION_INCONSISTENCY_S = 1.0    # review section 2 M4: |dx - F dX| > s
POSITION_CONSISTENT_DET_TOLERANCE = 0.10   # M5: |det F_lsq / det F - 1| (P4 low_ratio 0.9 as the declared scale)
REFERENCE_AREA_EPS = 1e-12          # x s^2: reference triangles below this area are not judged by the fold guard

FROZEN_PARAMETERS = {
    "profile": PROFILE,
    "M1_constrained_qef": {"tie_tolerance": QEF_TIE_TOLERANCE, "inside_tolerance": 1e-12,
                           "source": "exact box-constrained minimiser of the v1 truncated QEF (review section 2 M1)"},
    "M2_hinge": {"fraction": HINGE_FRACTION, "hinge_corners": "region corners with digit 0 for every cut mixed in the region",
                 "direction": "null space of the truncated QEF (singular values <= truncation x sigma_max)",
                 "source": "review section 2 M2 (move toward the region's code-0 hinge corners)"},
    "M3_fold_guard": {"test": "n_current . (cof(F_side) n_reference) <= 0", "reference_area_eps_s2": REFERENCE_AREA_EPS,
                      "stages": ["lsq_affine (v1 estimator)", "translation_only"],
                      "never_where": "det F <= 0 (reported as material inversion)", "source": "review section 2 M3"},
    "M4_unlabelled_separation": {"label": FALLBACK_LABEL, "length_s": SEPARATION_LENGTH_S,
                                 "inconsistency_s": SEPARATION_INCONSISTENCY_S, "F_of_link": "mean of the endpoint F",
                                 "fallback_particles": "endpoints of flagged links",
                                 "cell": "cube of volume (m/rho) det F (s^3 det F without masses), polar rotation of F",
                                 "source": "review section 2 M4 (thresholds 2 s and s as stated there; surface_protocol_v2 "
                                           "declares none, so they are frozen here with the code)"},
    "M5_position_consistent": {"det_ratio_tolerance": POSITION_CONSISTENT_DET_TOLERANCE, "default": False,
                               "label": PROFILE_POSITION_CONSISTENT, "source": "review section 2 M5 (labelled variant)"},
}


def _cube_quads():
    """Six outward quads (corner indices in CORNER_OFFSETS order) of the unit cube."""
    quads = []
    centre = np.full(3, .5)
    for axis in range(3):
        for side in (0, 1):
            corners = [c for c in range(8) if CORNER_OFFSETS[c, axis] == side]
            a1, a2 = (axis+1) % 3, (axis+2) % 3
            ring = sorted(corners, key=lambda c: np.arctan2(CORNER_OFFSETS[c, a2]-.5, CORNER_OFFSETS[c, a1]-.5))
            p = CORNER_OFFSETS[ring].astype(float)
            n = np.cross(p[1]-p[0], p[2]-p[0])
            if n@(p.mean(0)-centre) < 0:
                ring = ring[::-1]
            quads.append(ring)
    return np.asarray(quads, np.int64)


CUBE_QUADS = _cube_quads()
CUBE_TRIANGLES = np.concatenate([CUBE_QUADS[:, [0, 1, 2]], CUBE_QUADS[:, [0, 2, 3]]])


# ----------------------------------------------------------------------------- M1

def solve_qef_box(points, normals, weights, truncation=.1, tie_tolerance=QEF_TIE_TOLERANCE):
    """Box-constrained ([0, 1]^3) minimiser of the v1 truncated QEF (exact active-set enumeration).

    Same inputs as ``cut_aware_surface.solve_qef`` (cell-local). Returns (x (R, 3), truncated QEF value at x (R,),
    truncated QEF value at the v1 clip (R,), active constraints (R,))."""
    w = weights[..., None]
    total = weights.sum(1)
    mass = (points*w).sum(1)/np.maximum(total, 1e-300)[:, None]
    A = normals*w
    rhs = (A*(points-mass[:, None])).sum(-1)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > truncation*S[:, :1]
    Ak = np.einsum("rki,ri,rij->rkj", U, np.where(keep, S, 0.), Vt)
    bk = np.einsum("rki,ri,rli,rl->rk", U, keep.astype(float), U, rhs)

    def value(y):
        return ((np.einsum("rkj,rj->rk", Ak, y)-bk)**2).sum(1)
    R = len(points)
    combos = [c for c in itertools.product((-1, 0, 1), repeat=3) if any(v >= 0 for v in c)]
    X = np.zeros((R, len(combos), 3))
    Q = np.zeros((R, len(combos)))
    feasible = np.zeros((R, len(combos)), bool)
    for j, combo in enumerate(combos):
        combo = np.asarray(combo)
        fixed, free = np.flatnonzero(combo >= 0), np.flatnonzero(combo < 0)
        y = np.zeros((R, 3))
        y[:, fixed] = combo[fixed][None]-mass[:, fixed]
        if len(free):
            r = bk-np.einsum("rkj,rj->rk", Ak[:, :, fixed], y[:, fixed])
            pinv = np.linalg.pinv(Ak[:, :, free], rcond=1e-10)
            y[:, free] = np.einsum("rjk,rk->rj", pinv, r)
        x = mass+y
        X[:, j], Q[:, j] = x, value(y)
        feasible[:, j] = np.all((x >= -1e-12) & (x <= 1+1e-12), axis=1)
    qmin = np.where(feasible, Q, np.inf).min(1)
    tie = feasible & (Q <= qmin[:, None]+tie_tolerance*(1+qmin[:, None]))
    dist = np.where(tie, np.linalg.norm(X-mass[:, None], axis=2), np.inf)
    pick = np.argmin(dist, 1)
    x = np.clip(X[np.arange(R), pick], 0., 1.)
    active = (np.asarray(combos)[pick] >= 0).sum(1)
    x_v1, _ = solve_qef(points, normals, weights, truncation)
    inside = np.all((x_v1 >= -1e-12) & (x_v1 <= 1+1e-12), axis=1)       # v1 solution kept where it is in the cell
    x[inside], active[inside] = np.clip(x_v1[inside], 0., 1.), 0
    clip = np.clip(x_v1, 0., 1.)
    return x, value(x-mass), value(clip-mass), active


def qef_null_projector(normals, weights, truncation=.1):
    """(R, 3, 3) projector onto the null space of the truncated QEF (identity without Hermite rows)."""
    A = normals*weights[..., None]
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    drop = ~(S > truncation*S[:, :1])
    return np.einsum("rki,rk,rkj->rij", Vt, drop.astype(float), Vt)


def _polar_rotation(F):
    U, _, Vt = np.linalg.svd(F)
    R = U@Vt
    flip = np.linalg.det(R) < 0
    if flip.any():
        U[flip, :, -1] *= -1
        R[flip] = U[flip]@Vt[flip]
    return R


# ----------------------------------------------------------------------------- the surface

class CutAwareSurfaceV2(CutAwareSurface):
    """``cut_aware_surface_v2`` (M1-M4; M5 with ``embedding='position_consistent_v1'``). Same interface as
    ``CutAwareSurface``; ``fixes`` selects a subset of {'M1', 'M2', 'M3', 'M4'} (all by default; subsets are
    ablations for unit tests and diagnostics, never a different profile)."""

    profile = PROFILE

    def __init__(self, *args, embedding="total_F_v1", fixes=("M1", "M2", "M3", "M4"), **kwargs):
        super().__init__(*args, **kwargs)
        if embedding not in EMBEDDINGS:
            raise ValueError(f"embedding must be one of {EMBEDDINGS}")
        unknown = set(fixes)-{"M1", "M2", "M3", "M4"}
        if unknown:
            raise ValueError(f"unknown fixes {sorted(unknown)}")
        self.embedding, self.fixes = embedding, tuple(sorted(fixes))
        self.profile = PROFILE if embedding == "total_F_v1" else PROFILE_POSITION_CONSISTENT
        V = self.lattice
        self._cells_of_particle = None
        self._n = V.n

    # -- M4 detector
    def _separation(self, positions, F, codes, rotation):
        lat, s = self.lattice, self.spacing
        state, _ = V1.link_states(lat, codes)
        if F is None or "M4" not in self.fixes:
            return np.zeros(lat.n, bool), {"evaluated": False, "reason": "no total F" if F is None else "M4 disabled",
                                           "flagged_links": 0, "fallback_particles": 0}
        links = np.flatnonzero(state == LINK_UNCUT)
        a, b = lat.link_lo[links], lat.link_hi[links]
        dx = positions[b]-positions[a]
        dX = lat.reference[b]-lat.reference[a]
        Fm = .5*(F[a]+F[b])
        if rotation is not None:
            Fm = np.einsum("ij,rjk->rik", rotation, Fm)
        length = np.linalg.norm(dx, axis=1)
        inconsistency = np.linalg.norm(dx-np.einsum("rij,rj->ri", Fm, dX), axis=1)
        flag = (length > SEPARATION_LENGTH_S*s) & (inconsistency > SEPARATION_INCONSISTENCY_S*s)
        fallback = np.zeros(lat.n, bool)
        fallback[a[flag]] = True
        fallback[b[flag]] = True
        long_only = int(((length > SEPARATION_LENGTH_S*s) & ~flag).sum())
        return fallback, {"evaluated": True, "flagged_links": int(flag.sum()), "fallback_particles": int(fallback.sum()),
                          "uncut_links": int(len(links)), "long_links_consistent_with_F_not_flagged": long_only,
                          "flagged_length_s": V1_quant(length[flag]/s), "flagged_inconsistency_s": V1_quant(inconsistency[flag]/s),
                          "uncut_length_max_s": float(length.max()/s) if len(length) else None,
                          "uncut_inconsistency_max_s": float(inconsistency.max()/s) if len(length) else None}

    # -- Hermite with the fallback sites absent
    def _hermite_v2(self, codes, log, fallback):
        state, cutbits, signature, sources, stats = self._hermite(codes, log)
        lat, s = self.lattice, self.spacing
        lo_eff = np.where((lat.link_lo >= 0) & ~fallback[np.maximum(lat.link_lo, 0)], lat.link_lo, -1)
        hi_eff = np.where((lat.link_hi >= 0) & ~fallback[np.maximum(lat.link_hi, 0)], lat.link_hi, -1)
        absent = (lo_eff < 0) & (hi_eff < 0)
        new_boundary = (state == LINK_UNCUT) & ((lo_eff < 0) ^ (hi_eff < 0))
        lost_boundary = (state == LINK_BOUNDARY) & absent
        state = state.copy()
        state[new_boundary] = LINK_BOUNDARY
        state[absent] = LINK_ABSENT
        if new_boundary.any():
            k = np.flatnonzero(new_boundary)
            e = np.eye(3)[lat.link_axis[k]]
            p = lat.link_lo_reference[k]+.5*s*e
            n = e*np.where(lo_eff[k] >= 0, 1., -1.)[:, None]
            signature[k] = np.concatenate([p, n, p, n], 1)
        signature[absent] = 0.
        stats = dict(stats, particle_cell_fallback={"new_boundary_links_cell_face": int(new_boundary.sum()),
                                                    "absent_links": int(absent.sum()),
                                                    "static_boundary_links_lost": int(lost_boundary.sum()),
                                                    "cut_links_with_fallback_endpoint": int(((state == LINK_CUT) & ((lo_eff < 0) | (hi_eff < 0))).sum())})
        return state, cutbits, signature, sources, stats, lo_eff, hi_eff

    # -- Layer 4 with M1 / M2
    def _cells(self, cells, state, signature, codes=None, fallback=None):
        lat = self.lattice
        cp = lat.corner_particle[cells]
        present = cp >= 0
        if fallback is not None:
            present &= ~fallback[np.maximum(cp, 0)]
        edge_link = lat.cell_edge_link[cells]
        edge_state = np.where(edge_link >= 0, state[np.maximum(edge_link, 0)], -1)
        connected = edge_state == LINK_UNCUT
        disconnected = (edge_state == LINK_BOUNDARY) | (edge_state == LINK_CUT)
        feature = disconnected.any(1)
        cells, present, cp = cells[feature], present[feature], cp[feature]
        edge_link, connected, disconnected = edge_link[feature], connected[feature], disconnected[feature]
        label = cell_regions(present, connected)
        root = present & (label == np.arange(8))
        rep_row, rep_root = np.nonzero(root)
        mask = label[rep_row] == rep_root[:, None]
        R = len(rep_row)
        sig = signature[np.maximum(edge_link[rep_row], 0)]
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
        m1 = np.zeros(R, bool)
        m1_q = np.full(R, np.nan)
        m1_q_clip = np.full(R, np.nan)
        m1_shift = np.zeros(R)
        if "M1" in self.fixes and clamped.any():
            k = np.flatnonzero(clamped)
            xc, qv, qclip, _ = solve_qef_box(local[k], normals[k], weights[k], self.truncation)
            m1_shift[k] = np.linalg.norm(xc-x[k], axis=1)
            m1[k], m1_q[k], m1_q_clip[k] = True, qv, qclip
            x[k] = xc
        m2 = np.zeros(R, bool)
        m2_shift = np.zeros(R)
        mixed_no_hinge = np.zeros(R, bool)
        if "M2" in self.fixes and codes is not None and R:
            corner_codes = np.where(mask, codes[np.maximum(cp[rep_row], 0)], 0)
            a1, a2 = side_masks(corner_codes)
            mixed_bits = np.bitwise_or.reduce(a1, axis=1) & np.bitwise_or.reduce(a2, axis=1)   # bit 2k: cut k mixed
            mixed = mixed_bits != 0
            if mixed.any():
                k = np.flatnonzero(mixed)
                nonzero = (a1[k] | a2[k]) & mixed_bits[k][:, None]       # corner has a side digit of a mixed cut
                hinge = mask[k] & (nonzero == 0)
                has = hinge.any(1)
                mixed_no_hinge[k[~has]] = True
                k, hinge = k[has], hinge[has]
                if len(k):
                    target = (hinge[..., None]*CORNER_OFFSETS[None]).sum(1)/hinge.sum(1, keepdims=True)
                    P = qef_null_projector(normals[k], weights[k], self.truncation)
                    d = HINGE_FRACTION*np.einsum("rij,rj->ri", P, target-x[k])
                    # largest step t in [0, 1] keeping x + t d inside the cell
                    with np.errstate(divide="ignore", invalid="ignore"):
                        up = np.where(d > 1e-15, (1-x[k])/d, np.inf)
                        down = np.where(d < -1e-15, -x[k]/d, np.inf)
                    t = np.clip(np.minimum(up, down).min(1), 0., 1.)
                    step = t[:, None]*d
                    x[k] = np.clip(x[k]+step, 0., 1.)
                    m2_shift[k] = np.linalg.norm(step, axis=1)
                    m2[k] = m2_shift[k] > 0
        return {"cell": cells[rep_row], "root": rep_root.astype(np.int8), "mask": mask,
                "X": origin+x*self.spacing, "rank": rank.astype(np.int8), "clamped": clamped,
                "hermite_rows": weights.sum(1).astype(np.int16), "m1": m1, "m1_q": m1_q, "m1_q_clip": m1_q_clip,
                "m1_shift": m1_shift*self.spacing, "m2": m2, "m2_shift": m2_shift*self.spacing,
                "mixed_no_hinge": mixed_no_hinge, "feature_cells": cells}

    _REP_KEYS = ("cell", "root", "mask", "X", "rank", "clamped", "hermite_rows", "m1", "m1_q", "m1_q_clip", "m1_shift",
                 "m2", "m2_shift", "mixed_no_hinge")

    def _merge(self, keep, new):
        out = {k: np.concatenate([keep[k], new[k]]) for k in self._REP_KEYS}
        order = np.lexsort((out["root"], out["cell"]))
        return {k: v[order] for k, v in out.items()}

    def _topology_v2(self, codes, log, force_full, fallback):
        state, cutbits, signature, sources, stats, lo_eff, hi_eff = self._hermite_v2(codes, log, fallback)
        lat = self.lattice
        full = force_full or not self.incremental or self._cache is None
        if full:
            reps = self._cells(np.arange(lat.n_cells), state, signature, codes, fallback)
            dirty = lat.n_cells
        else:
            cache = self._cache
            changed = ((state != cache["state"]) | (cutbits != cache["cutbits"])
                       | np.any(signature != cache["signature"], axis=1))
            particle_changed = (codes != cache["codes"]) | (fallback != cache["fallback"])
            cp = lat.corner_particle
            cell_changed = np.any((cp >= 0) & particle_changed[np.maximum(cp, 0)], axis=1)
            dirty_cells = np.union1d(np.unique(lat.link_cells[changed].ravel()), np.flatnonzero(cell_changed))
            reps = cache["reps"]
            if len(dirty_cells):
                is_dirty = np.zeros(lat.n_cells, bool)
                is_dirty[dirty_cells] = True
                keep = {k: v[~is_dirty[reps["cell"]]] for k, v in reps.items()}
                reps = self._merge(keep, self._cells(dirty_cells, state, signature, codes, fallback))
            dirty = int(len(dirty_cells))
        reps = {k: v for k, v in reps.items() if k != "feature_cells"}
        self._cache = {"state": state, "cutbits": cutbits, "signature": signature, "reps": reps,
                       "codes": codes.copy(), "fallback": fallback.copy()}
        return state, cutbits, sources, stats, reps, lo_eff, hi_eff, {"mode": "full" if full else "incremental",
                                                                     "dirty_cells": int(dirty), "cells": int(lat.n_cells)}

    def _quads_v2(self, state, reps, lo_eff, hi_eff):
        lat = self.lattice
        region_of = np.full((lat.n_cells, 8), -1, np.int64)
        r, c = np.nonzero(reps["mask"])
        region_of[reps["cell"][r], c] = r
        links = np.flatnonzero((state == LINK_BOUNDARY) | (state == LINK_CUT))
        cells = lat.link_cells[links]
        faces, face_link, face_side = [], [], []
        for side, corner, present in ((0, lat.link_corner_lo, lo_eff), (1, lat.link_corner_hi, hi_eff)):
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
        return np.concatenate(faces).astype(np.int64), np.concatenate(face_link), np.concatenate(face_side)

    # -- Layer 5 with M5 and the M3 fold guard
    def _embed_v2(self, reps, positions, F, codes, state, rotation):
        lat = self.lattice
        corners = lat.corner_particle[reps["cell"]]
        mask = reps["mask"]
        X = reps["X"]
        u = (X-lat.cell_origin[reps["cell"]])/self.spacing
        w = np.prod(np.where(CORNER_OFFSETS[None].astype(bool), u[:, None, :], 1-u[:, None, :]), axis=2)*mask
        total = w.sum(1)
        zero = total <= 1e-12
        w[zero] = mask[zero]
        w = w/w.sum(1, keepdims=True)
        idx = np.where(mask, corners, 0)
        d = X[:, None, :]-lat.reference[idx]
        rot = np.eye(3) if rotation is None else rotation
        counts = {"estimator": "total_F" if F is not None else "lsq_affine", "embedding_profile": self.embedding,
                  "representatives": int(len(X)), "zero_weight_uniform_fallback": int(zero.sum())}
        bad_term = np.zeros(mask.shape, bool)
        if F is not None:
            Fp = F[idx]
            det = np.linalg.det(Fp)
            bad = (det <= 0) | ~np.isfinite(det)
            bad_term = mask & bad
            Fp = np.where(bad[..., None, None], np.eye(3), Fp)
            counts.update(det_F_nonpositive_particle_terms=int(bad_term.sum()),
                          det_F_nonpositive_particles=int(len(np.unique(idx[bad_term]))),
                          translation_only_representatives=int(np.any(bad_term, 1).sum()))
            Fp = np.einsum("ij,rcjk->rcik", rot, Fp)
            if self.embedding == "position_consistent_v1":
                particles = np.unique(idx[mask])
                Fe, ok = lsq_affine_gradients(lat, positions, codes, state, particles)
                ratio = np.linalg.det(Fe)/np.linalg.det(F[particles])
                use = ok & np.isfinite(ratio) & (np.abs(ratio-1) > POSITION_CONSISTENT_DET_TOLERANCE) & (np.linalg.det(F[particles]) > 0)
                table = np.full(lat.n, -1, np.int64)
                table[particles] = np.arange(len(particles))
                t = table[idx]
                swap = mask & (t >= 0) & use[np.maximum(t, 0)]
                Fp = np.where(swap[..., None, None], Fe[np.maximum(t, 0)], Fp)
                counts["M5_position_consistent"] = {"particles": int(len(particles)), "particles_lsq": int(use.sum()),
                                                    "lsq_not_ok": int((~ok).sum()), "terms_lsq": int(swap.sum()),
                                                    "representatives_with_lsq_term": int(np.any(swap, 1).sum()),
                                                    "det_ratio_tolerance": POSITION_CONSISTENT_DET_TOLERANCE}
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
            Fp = np.where(not_ok[..., None, None], rot, Fp)
        term = positions[idx]+np.einsum("rcij,rcj->rci", Fp, d)
        current = (w[..., None]*term).sum(1)
        return current, counts, {"w": w, "idx": idx, "d": d, "mask": mask, "bad_term": bad_term, "rot": rot}

    def _flipped(self, X, current, faces, face_side, Fw, judge):
        nr = np.cross(X[faces[:, 1]]-X[faces[:, 0]], X[faces[:, 2]]-X[faces[:, 0]])
        nc = np.cross(current[faces[:, 1]]-current[faces[:, 0]], current[faces[:, 2]]-current[faces[:, 0]])
        if Fw is None:
            expected = nr
        else:
            expected = np.einsum("rij,rj->ri", np.transpose(np.linalg.inv(Fw), (0, 2, 1)), nr)
        return judge & (np.einsum("ij,ij->i", nc, expected) <= 0)

    def _fold_guard(self, reps, current, faces, face_side, positions, F, codes, state, rotation, emb):
        lat, s = self.lattice, self.spacing
        X = reps["X"]
        stats = {"enabled": "M3" in self.fixes}
        stage = np.zeros(len(X), np.int8)          # 0 total F, 1 lsq_affine, 2 translation-only
        if "M3" not in self.fixes or not len(faces):
            return current, stage, {**stats, "flipped_triangles_before": None}
        nr = np.linalg.norm(np.cross(X[faces[:, 1]]-X[faces[:, 0]], X[faces[:, 2]]-X[faces[:, 0]]), axis=1)
        judge = nr > REFERENCE_AREA_EPS*s**2
        Fw = None
        inverted_rep = np.zeros(len(X), bool)
        if F is not None:
            Fs = F[face_side]
            dets = np.linalg.det(Fs)
            judge_inv = dets > 0
            Fw = np.einsum("ij,rjk->rik", emb["rot"], np.where(judge_inv[:, None, None], Fs, np.eye(3)))
            stats["inversion_skipped_triangles"] = int((judge & ~judge_inv).sum())
            judge = judge & judge_inv
            inverted_rep = np.any(emb["bad_term"], 1)
        else:
            Fw = np.broadcast_to(emb["rot"], (len(faces), 3, 3))
        stats["unjudged_degenerate_reference_triangles"] = int((nr <= REFERENCE_AREA_EPS*s**2).sum())
        flip0 = self._flipped(X, current, faces, face_side, Fw, judge)
        stats["flipped_triangles_before"] = int(flip0.sum())
        flagged = np.zeros(len(X), bool)
        flagged[np.unique(faces[flip0])] = True
        stats["representatives_inversion_skipped"] = int((flagged & inverted_rep).sum())
        flagged &= ~inverted_rep
        stats["representatives_lsq"] = int(flagged.sum())
        out = current.copy()
        w, idx, d, mask = emb["w"], emb["idx"], emb["d"], emb["mask"]
        if flagged.any():
            k = np.flatnonzero(flagged)
            particles = np.unique(idx[k][mask[k]])
            Fe, ok = lsq_affine_gradients(lat, positions, codes, state, particles)
            table = np.zeros(lat.n, np.int64)
            table[particles] = np.arange(len(particles))
            Fk = Fe[table[idx[k]]]
            not_ok = ~ok[table[idx[k]]]
            Fk = np.where(not_ok[..., None, None], emb["rot"], Fk)
            term = positions[idx[k]]+np.einsum("rcij,rcj->rci", Fk, d[k])
            out[k] = (w[k][..., None]*term).sum(1)
            stage[k] = 1
            stats["lsq_translation_only_particles"] = int((~ok).sum())
            flip1 = self._flipped(X, out, faces, face_side, Fw, judge)
            stats["flipped_triangles_after_lsq"] = int(flip1.sum())
            again = np.zeros(len(X), bool)
            again[np.unique(faces[flip1])] = True
            again &= flagged
            stats["representatives_translation_only"] = int(again.sum())
            if again.any():
                k2 = np.flatnonzero(again)
                term = positions[idx[k2]]+np.einsum("ij,rcj->rci", emb["rot"], d[k2])
                out[k2] = (w[k2][..., None]*term).sum(1)
                stage[k2] = 2
            flip2 = self._flipped(X, out, faces, face_side, Fw, judge)
            stats["flipped_triangles_after"] = int(flip2.sum())
        else:
            stats.update(flipped_triangles_after_lsq=0, representatives_translation_only=0, flipped_triangles_after=0,
                         lsq_translation_only_particles=0)
        return out, stage, stats

    # -- M4 particle cells
    def _particle_cells(self, particles, positions, F, rotation):
        s = self.spacing
        n = len(particles)
        if not n:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int64), {"cells": 0}
        offsets = CORNER_OFFSETS.astype(float)-.5
        Xc = self.lattice.reference[particles][:, None, :]+s*offsets[None]
        volume = (self.masses[particles]/self.density if self.masses is not None and self.density is not None
                  else np.full(n, s**3))
        info = {"cells": int(n)}
        if F is not None:
            J = np.linalg.det(F[particles])
            nonpos = J <= 0
            Rp = _polar_rotation(F[particles])
            Rp[nonpos] = np.eye(3)
            info["det_F_nonpositive_cells"] = int(nonpos.sum())
            side = np.cbrt(volume*np.where(nonpos, np.abs(J), J))
        else:
            Rp = np.tile(np.eye(3), (n, 1, 1))
            side = np.cbrt(volume)
        if rotation is not None:
            Rp = np.einsum("ij,rjk->rik", rotation, Rp)
        xc = positions[particles][:, None, :]+side[:, None, None]*np.einsum("rij,cj->rci", Rp, offsets)
        faces = (np.arange(n)[:, None, None]*8+CUBE_TRIANGLES[None]).reshape(-1, 3)
        info["volume_m3"] = float((side**3).sum())
        return Xc.reshape(-1, 3), xc.reshape(-1, 3), faces, info

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
        count, labels, provenance = self.partition.partition(codes_in)
        codes = (codes_in.astype(np.int64) if self.encoding == "packed2" else ternary_to_packed(codes_in, self.cuts))
        timings["mechanical_partition_s"] = time.perf_counter()-start
        log = self.cut_events_log if cut_events_log is None else cut_events_log
        t0 = time.perf_counter()
        fallback, sep_stats = self._separation(positions, F, codes, rotation)
        timings["M4_separation_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        state, cutbits, _, hermite_stats, reps, lo_eff, hi_eff, update = self._topology_v2(codes, log, full_rebuild, fallback)
        timings["topology_and_qef_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        faces, face_link, face_side = self._quads_v2(state, reps, lo_eff, hi_eff)
        timings["quads_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        current, embed_counts, emb = self._embed_v2(reps, positions, F, codes, state, rotation)
        timings["embedding_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        before_guard = current
        current, guard_stage, guard = self._fold_guard(reps, current, faces, face_side, positions, F, codes, state, rotation, emb)
        timings["M3_fold_guard_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        fb_particles = np.flatnonzero(fallback)
        Xc, xc, fc_faces, cell_info = self._particle_cells(fb_particles, positions, F, rotation)
        timings["M4_particle_cells_s"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        R = len(reps["X"])
        nc = len(fb_particles)
        # vertices: representatives then particle-cell corners; faces: lattice faces then particle cells
        X_all = np.concatenate([reps["X"], Xc])
        cur_all = np.concatenate([current, xc])
        before_all = np.concatenate([before_guard, xc])
        rep_first = lat.corner_particle[reps["cell"], reps["root"].astype(np.int64)]
        first_particle = np.concatenate([rep_first, np.repeat(fb_particles, 8)])
        is_cell_vertex = np.r_[np.zeros(R, bool), np.ones(8*nc, bool)]
        faces_all = np.concatenate([faces, fc_faces+R]) if nc else faces
        face_link_all = np.r_[face_link, np.full(len(fc_faces), -1)]
        face_side_all = np.r_[face_side, np.repeat(fb_particles, 12)]
        face_kind_all = np.r_[state[face_link].astype(np.int8), np.full(len(fc_faces), FACE_PARTICLE_CELL, np.int8)]
        face_cutbits_all = np.r_[cutbits[face_link], np.zeros(len(fc_faces), cutbits.dtype)]
        vertex_component = labels[first_particle]
        vorder = np.argsort(vertex_component, kind="stable")
        remap = np.empty_like(vorder)
        remap[vorder] = np.arange(len(vorder))
        faces_all = remap[faces_all]
        face_component = vertex_component[vorder][faces_all[:, 0]] if len(faces_all) else np.zeros(0, np.int64)
        forder = np.argsort(face_component, kind="stable")
        faces_all, face_component = faces_all[forder], face_component[forder]
        face_link_all, face_side_all = face_link_all[forder], face_side_all[forder]
        face_kind_all, face_cutbits_all = face_kind_all[forder], face_cutbits_all[forder]
        pad = lambda a, fill: np.concatenate([a, np.full((8*nc,)+a.shape[1:], fill, a.dtype)])  # noqa: E731
        rep_arrays = {"cell": pad(reps["cell"], -1), "root": pad(reps["root"], 0), "mask": pad(reps["mask"], False),
                      "clamped": pad(reps["clamped"], False), "rank": pad(reps["rank"], 0),
                      "m1": pad(reps["m1"], False), "m2": pad(reps["m2"], False), "stage": pad(guard_stage, 0)}
        rep_arrays = {k: v[vorder] for k, v in rep_arrays.items()}
        reference_vertices, current_v = X_all[vorder], cur_all[vorder]
        before_v = before_all[vorder]
        vertex_component, first_particle = vertex_component[vorder], first_particle[vorder]
        is_cell_vertex = is_cell_vertex[vorder]
        ranges, vfirst, ffirst = [], 0, 0
        vcount = np.bincount(vertex_component, minlength=count)
        fcount = np.bincount(face_component, minlength=count)
        for lab in range(count):
            ranges.append({"mechanical_component": int(lab), "particles": int((labels == lab).sum()),
                           "first_vertex": int(vfirst), "vertices": int(vcount[lab]),
                           "first_triangle": int(ffirst), "triangles": int(fcount[lab])})
            vfirst += int(vcount[lab]); ffirst += int(fcount[lab])
        # the audit uses the v1 uncut-link graph (fallback particles stay in their lattice piece)
        state_v1, _ = V1.link_states(lat, codes)
        audit = self._audit(codes, labels, count, state_v1, faces_all, vertex_component, first_particle,
                            reference_vertices, current_v, face_component)
        timings["audit_s"] = time.perf_counter()-t0
        validity = mesh_validity(reference_vertices, faces_all)
        timings["total_s"] = time.perf_counter()-start
        self.extractions += 1
        m1 = reps["m1"]
        v2 = {"profile": self.profile, "fixes": list(self.fixes), "frozen_parameters": FROZEN_PARAMETERS,
              "M1_constrained_qef": {"v1_clamped": int(reps["clamped"].sum()), "constrained": int(m1.sum()),
                                     "qef_clip_minus_constrained": V1_quant(reps["m1_q_clip"][m1]-reps["m1_q"][m1]),
                                     "qef_constrained": V1_quant(reps["m1_q"][m1]),
                                     "shift_from_clip_m": V1_quant(reps["m1_shift"][m1])},
              "M2_hinge": {"mixed_regions": int(self._mixed_side_regions(codes, reps)), "moved": int(reps["m2"].sum()),
                           "mixed_without_hinge_corner": int(reps["mixed_no_hinge"].sum()),
                           "shift_m": V1_quant(reps["m2_shift"][reps["m2"]])},
              "M3_fold_guard": guard,
              "M4_unlabelled_separation": {**sep_stats, **cell_info, "label": FALLBACK_LABEL,
                                           "particle_cell_triangles": int(len(fc_faces)),
                                           **hermite_stats.get("particle_cell_fallback", {}),
                                           "note": "solver-state anomaly for Phase 4: positions moved apart without F or codes"},
              "M5_position_consistent": embed_counts.get("M5_position_consistent")}
        info = {**provenance, "surface_profile": self.profile, "profile_version": self.profile,
                "v1_module_profile": V1.PROFILE, "mechanical_components": int(count), "component_ranges": ranges,
                "renderer_partition_changes_physics": False, "writes_physics_state": False,
                "codes_sha256": __import__("hashlib").sha256(codes_in.tobytes()).hexdigest(),
                **lat.sha256(), "stencil_sha256": self.partition.stencil_sha256,
                "reference_grid": {"kind": "particle lattice (linked volume)", "spacing_m": self.spacing,
                                   "sites": int(lat.n), "links": int(lat.n_links), "dual_cells": int(lat.n_cells),
                                   "lattice_residual_m": lat.lattice_residual_m},
                "link_rule": "cut iff endpoint packed codes conflict; boundary iff one effective endpoint absent "
                             "(lattice site absent or in the unlabelled_separation_v1 particle-cell fallback)",
                "boundary_hermite_profile": self.boundary_profile,
                "analytic_shape": None if self.analytic_shape is None else self.analytic_shape.to_dict(),
                "hermite": {"source": "cut_events_v1" if log is not None else "midpoint_fallback_R4",
                            "fallback_normal": "sum of signed same-cut link directions within radius",
                            "fallback_normal_radius_spacings": self.fallback_radius, **hermite_stats},
                "qef": {"solver": "float64 SVD toward mass point; M1 box-constrained where the v1 solution leaves the cell",
                        "truncation_relative": self.truncation, "representatives": int(R),
                        "clamped": int(reps["clamped"].sum()),
                        "clamped_after_v2": int((reps["clamped"] & ~m1).sum()),
                        "rank_counts": {str(k): int((reps["rank"] == k).sum()) for k in (0, 1, 2, 3)},
                        "representatives_without_hermite": int((reps["hermite_rows"] == 0).sum())},
                "regions": {"feature_cells": int(len(np.unique(reps["cell"]))),
                            "multi_region_cells": int((np.bincount(reps["cell"]) > 1).sum()) if R else 0,
                            "mixed_side_regions": int(self._mixed_side_regions(codes, reps))},
                "faces": {"triangles": int(len(faces_all)), "cut_triangles": int((face_kind_all == LINK_CUT).sum()),
                          "boundary_triangles": int((face_kind_all == LINK_BOUNDARY).sum()),
                          "particle_cell_triangles": int((face_kind_all == FACE_PARTICLE_CELL).sum())},
                "embedding": embed_counts, "mesh_validity_reference": validity, "audit": audit, "v2": v2,
                "update": update, "timings_s": timings,
                "scope": "Render geometry from solver state (one-way). Not the contact surface; contact uses particles."}
        self.last = {"vertex_reference": reference_vertices, "vertex_component": vertex_component,
                     "vertex_cell": rep_arrays["cell"], "vertex_root": rep_arrays["root"],
                     "vertex_corner_mask": rep_arrays["mask"], "vertex_clamped": rep_arrays["clamped"],
                     "vertex_rank": rep_arrays["rank"], "vertex_first_particle": first_particle,
                     "vertex_m1_constrained": rep_arrays["m1"], "vertex_m2_hinge": rep_arrays["m2"],
                     "vertex_fold_guard_stage": rep_arrays["stage"], "vertex_particle_cell": is_cell_vertex,
                     "vertices_before_fold_guard": before_v,
                     "face_link": face_link_all, "face_kind": face_kind_all, "face_side_particle": face_side_all,
                     "face_cutbits": face_cutbits_all, "face_component": face_component,
                     "face_particle_cell": face_kind_all == FACE_PARTICLE_CELL, "particle_cell_fallback": fallback,
                     "link_state": state, "link_cutbits": cutbits}
        return current_v, faces_all.astype(np.int32), info

    def region_particles(self, vertex):
        if self.last["vertex_particle_cell"][vertex]:
            return np.asarray([self.last["vertex_first_particle"][vertex]])
        return super().region_particles(vertex)


def V1_quant(values):
    """P50 / P95 / max / count of a 1-D array (None when empty)."""
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0}
    return {"count": int(len(a)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}
