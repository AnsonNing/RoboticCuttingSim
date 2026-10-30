"""Diagnostic extension of ``cut_events_v1``: assignment-instant positions and chunk hooks (WP-12 Phase 1).

``cut_event_snapshots_v1`` is a subclass of ``cut_events.CutEventRecorder`` used only by diagnosis drivers.
It writes nothing the solver reads: in addition to the parent's log it stores, for every particle p newly
assigned in cut k at a substep, the current position of p and of its six lattice neighbours *at that
instant* (the same ``x`` the parent's event kernel reads). With them the current-space Hermite point of an
event at its assignment instant is exact::

    h = x_p(t_a) + t (x_q(t_a) - x_p(t_a))     (t = 1/2 for sign-inconsistent records)

which lies on the solver's swept plane (o, n) of that substep by construction. The parent log keeps only the
reference point and normals; these positions are what separates the commanded sweep / assignment layer from
post-cut physical deformation in the Phase 1 attribution.

``check()`` (called by ``drive_blade`` after every 200-substep chunk and by ``LiveCuttingMaterial`` after every
control step) additionally calls an optional host hook with the recorder, e.g. to subsample the chunk's
blade poses (``bed.poses``/``bed.cuts``) or to copy a state snapshot. The hook only reads.

Activation: ``installed(...)`` temporarily rebinds ``cut_events.CutEventRecorder`` to this class, so the
existing opt-in paths (``drive_blade(cut_events='cut_events_v1')``, ``KinematicConfig(cut_events=...)``)
construct it unchanged. Oracle/debug data; never an observation.
"""
from contextlib import contextmanager

import numpy as np
import warp as wp

from . import cut_events as ce

SNAPSHOT_VERSION = "cut_event_snapshots_v1"


@wp.func
def _cs_digit(code: int, cut: int):
    return (code >> (2*cut)) & 3


@wp.kernel
def _snapshot_assignment(x: wp.array(dtype=wp.vec3), code: wp.array(dtype=int), code_prev: wp.array(dtype=int),
                         neighbours: wp.array2d(dtype=int), cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                         shift: int, max_cuts: int, n: int, substep: wp.array(dtype=int),
                         x_assign: wp.array(dtype=wp.vec3), x_neighbour: wp.array(dtype=wp.vec3),
                         assign_substep: wp.array(dtype=int)):
    p = wp.tid()
    i = index[0]-shift
    cut = int(-1)
    if i >= 0:
        cut = cuts[i]
    if cut < 0 or cut >= max_cuts:
        return
    if _cs_digit(code[p], cut) == 0 or _cs_digit(code_prev[p], cut) != 0:
        return
    slot = cut*n+p
    x_assign[slot] = x[p]
    for j in range(6):
        q = neighbours[p, j]
        if q >= 0:
            x_neighbour[slot*6+j] = x[q]
    assign_substep[slot] = substep[0]


class AssignmentSnapshotRecorder(ce.CutEventRecorder):
    """cut_events_v1 + assignment-instant positions (see module docstring)."""

    snapshot_version = SNAPSHOT_VERSION

    def __init__(self, bed, lattice, *, hook=None, **kwargs):
        super().__init__(bed, lattice, **kwargs)
        self.hook = hook
        n, k = bed.n, bed.max_cuts
        with wp.ScopedDevice(self.device):
            self.x_assign = wp.zeros(k*n, dtype=wp.vec3)
            self.x_neighbour = wp.zeros(k*n*6, dtype=wp.vec3)
            self.assign_substep = wp.full(k*n, -1, dtype=int)
        wp.load_module(module=__name__, device=self.device)

    def after_step(self, advanced=True):
        bed = self.bed
        with wp.ScopedDevice(self.device):
            # Launched before the parent's kernels so ``substep`` still equals this substep's row.
            wp.launch(_snapshot_assignment, bed.n, [bed.x, bed.code, self.code_prev, self.neighbours, bed.cuts,
                bed.index, 1 if advanced else 0, bed.max_cuts, bed.n, self.substep, self.x_assign,
                self.x_neighbour, self.assign_substep])
        super().after_step(advanced)

    def check(self):
        counters = super().check()
        if self.hook is not None:
            self.hook(self)
        return counters

    def assignment_positions(self, log=None):
        """(E, 3) current-space Hermite points at each logged event's assignment instant, plus the per-event
        assignment-time endpoint positions (x_p, x_q) and the substep at which p was assigned (must equal
        the event's substep)."""
        log = self.log() if log is None else log
        n = self.bed.n
        xa = self.x_assign.numpy().reshape(self.bed.max_cuts, n, 3)
        xn = self.x_neighbour.numpy().reshape(self.bed.max_cuts, n, 6, 3)
        sub = self.assign_substep.numpy().reshape(self.bed.max_cuts, n)
        p, q, k = log["p"].astype(np.int64), log["q"].astype(np.int64), log["cut"].astype(np.int64)
        slot = np.argmax(self.neighbours_host[p] == q[:, None], axis=1)
        if not np.all(self.neighbours_host[p, slot] == q):
            raise RuntimeError("An event's q is not a lattice neighbour of p")
        xp = xa[k, p].astype(np.float64)
        xq = xn[k, p, slot].astype(np.float64)
        t = np.where(np.isfinite(log["t"]), log["t"], .5).astype(np.float64)
        return {"hermite_current_at_assignment": xp+t[:, None]*(xq-xp), "x_p": xp, "x_q": xq,
                "assign_substep": sub[k, p], "substep_matches_log": bool(np.array_equal(sub[k, p], log["substep"]))}

    def snapshot_metadata(self):
        return {"version": SNAPSHOT_VERSION, "base": ce.CUT_EVENTS_VERSION,
                "stores": "x_p and its six lattice neighbours at p's assignment substep, per cut digit",
                "device_bytes": int(self.x_assign.capacity+self.x_neighbour.capacity+self.assign_substep.capacity),
                "scope": "Read-only diagnosis (oracle) data; never an observation"}


@contextmanager
def installed(hook=None, capacity=None):
    """Rebind ``cut_events.CutEventRecorder`` to ``AssignmentSnapshotRecorder`` for the duration; yields the list
    of recorders created (the existing opt-in paths construct the recorder by that module attribute)."""
    original = ce.CutEventRecorder
    created = []

    class _Bound(AssignmentSnapshotRecorder):
        def __init__(self, bed, lattice, **kwargs):
            if capacity is not None and kwargs.get("capacity") is None:
                kwargs["capacity"] = capacity
            super().__init__(bed, lattice, hook=hook, **kwargs)
            created.append(self)

    ce.CutEventRecorder = _Bound
    try:
        yield created
    finally:
        ce.CutEventRecorder = original


class PoseTrail:
    """A ``check()`` hook that keeps every ``stride``-th substep pose (solver-local) and its cut digit, and
    optional state snapshots on request (``want_snapshot(recorder) -> bool``).

    The pose read-back is valid for ``LiveCuttingMaterial`` (one trajectory buffer, reassigned every control
    step). ``drive_blade`` rebinds ``bed.poses`` when a chunk of a new length is captured while earlier graphs keep
    their own buffers, so probe drivers must rebuild the poses on the host from the phases instead."""

    def __init__(self, stride=10, want_snapshot=None, snapshot_F=False):
        self.stride, self.want_snapshot, self.snapshot_F = int(stride), want_snapshot, snapshot_F
        self.poses, self.cuts, self.chunk_end_substep, self.snapshots = [], [], [], []

    def __call__(self, recorder):
        bed = recorder.bed
        substeps = int(recorder.substep.numpy()[0])
        used = int(bed.index.numpy()[0])  # poses consumed in this chunk
        poses = bed.poses.numpy()[:used+1]
        cuts = bed.cuts.numpy()[:used+1]
        keep = np.r_[np.arange(0, len(poses), self.stride), len(poses)-1]
        keep = np.unique(keep)
        self.poses.append(poses[keep].copy())
        self.cuts.append(cuts[keep].copy())
        self.chunk_end_substep.append(substeps)
        if self.want_snapshot is not None and self.want_snapshot(recorder):
            snap = {"substep": substeps, "positions": bed.x.numpy().copy(), "codes": bed.code.numpy().copy()}
            if self.snapshot_F:
                snap["F"] = bed.F.numpy().copy()
            self.snapshots.append(snap)

    def chains(self):
        """List of (T, 7) pose arrays and (T,) digits, one per chunk."""
        return list(zip(self.poses, self.cuts))
