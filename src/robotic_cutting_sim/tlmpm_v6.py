"""CANDIDATE solver ``tlmpm_v6``: tlmpm_v5 with a grid-consistent position update (WP-12 Phase 4, ADR 0020 proposed).

Status: **opt-in candidate, not adopted.** Nothing selects it by default; only an explicit ``solver="tlmpm_v6"``
(``LiveCuttingMaterial``, ``KinematicConfig``) or ``solver_version=6`` (``blade_probe.drive_blade``) constructs it.
Every production preset, demo, envelope v1, benchmark v1, E2 and Phase 5 stay on ``tlmpm_v5``. Acceptance is the
staged protocol ``configs/wp12/phase4_gridpos_protocol_v1.json`` (decision ``docs/reports/
WP12_PHASE4_PROTOTYPE_DECISION.md`` P-A); building it is not adopting it.

``TLMPM`` subclasses ``tlmpm_v5.TLMPM`` and overrides only ``step()``. Its body is the v5 step launch for launch
(same kernels from ``tlmpm_v5``, same order, same arguments, ``tlmpm_v5.py:816-890``) plus two insertions, active
only with ``position_update="grid_musl_v1"`` (the default of this module):

1. before ``_start_cut`` and every contact kernel: ``x_start <- copy(x)``;
2. after ``_deformation`` (F += dt sum_I v_I^MUSL (x) grad N_I) and before the law-4/5/8 side assignment
   (``_update_stroke``, ``_assign_band``, ``_assign_lead``, which therefore read the new positions):
   ``x_p <- x_start_p + dt * sum_{c<8, slot[p,c]>=0} weights[p,c] * new_v[nodes[p,c]*S + slot[p,c]]``,
   i.e. the same MUSL grid velocity field, slots and weights that update F. ``_g2p``'s ``x += dt v_p`` is
   overwritten; the FLIP/PIC-blended particle velocity v_p (alpha = ``config.pic``) is kept unchanged.

``position_update="flip_v5"`` is the identity option (tests only): the step is then bitwise ``tlmpm_v5`` on CPU.
Declared and unchanged: ``_assign_band`` keeps v_p in its rear tolerance (``tlmpm_v5.py:634``); the mismatch is
<= dt |v_p - S v^MUSL| (about 10 um per substep) against b_back = 8 mm. Laws 4/5/8, their parameters, the contact
path, the material config and the cut-history representation are tlmpm_v5's. This is a time-integration change
(removing strain-free FLIP null-space drift of x); it is not a cutting law, not fracture, not calibration and not
"the standard TLMPM update" (Karamelo advects with v+; this rule uses v^MUSL so that x and F share one field).
It does not import the WP-12 V1-V4 COUNTERFACTUAL module; its kernel is a copy of that module's grid-position
hook, so the results equal the counterfactual ``position_update='grid_musl'`` bitwise on CPU (tested in
``tests/test_tlmpm_v6.py``, which is the only place that names both).
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from . import tlmpm_v5 as v5
from .cut_topology import PACKED_MAX_CUTS, decode_digits, packed_to_ternary  # noqa: F401  (re-exported helpers)
from .material_config import TLMPMConfig

SOLVER_NAME = "tlmpm_v6"
STATUS = "CANDIDATE_NOT_ADOPTED"
CODE_ENCODING = v5.CODE_ENCODING
DEFAULT_SLOTS = v5.DEFAULT_SLOTS
SlotOverflowError = v5.SlotOverflowError
GRID_MUSL_V1 = "grid_musl_v1"
FLIP_V5 = "flip_v5"
POSITION_UPDATES = (GRID_MUSL_V1, FLIP_V5)
DEFAULT_POSITION_UPDATE = GRID_MUSL_V1
ADR = "docs/adr/0020-tlmpm-v6-grid-consistent-position-update.md"
_HERE = Path(__file__).resolve().parent


def source_sha256():
    """sha256 of tlmpm_v5.py and tlmpm_v6.py as they are on disk (receipts)."""
    return {name: hashlib.sha256((_HERE/name).read_bytes()).hexdigest() for name in ("tlmpm_v5.py", "tlmpm_v6.py")}


@wp.kernel
def _grid_positions(x: wp.array(dtype=wp.vec3), x_start: wp.array(dtype=wp.vec3), slot: wp.array2d(dtype=int),
                    nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
                    velocity: wp.array(dtype=wp.vec3), slots: int, dt: float):
    """x_p <- x_p(start of step) + dt sum_I w_pI v_I^MUSL (same field, slots and weights as _deformation)."""
    p = wp.tid()
    vg = wp.vec3(0.0)
    for corner in range(8):
        s = slot[p, corner]
        if s >= 0:
            vg += weights[p, corner]*velocity[nodes[p, corner]*slots+s]
    x[p] = x_start[p]+dt*vg


class TLMPM(v5.TLMPM):
    """tlmpm_v5 with the grid-consistent position update (CANDIDATE; see the module docstring).

    Same constructor as ``tlmpm_v5.TLMPM`` plus ``position_update`` ("grid_musl_v1", default, or "flip_v5",
    the bitwise-v5 identity used by tests)."""
    solver_name = SOLVER_NAME

    def __init__(self, points, masses, spacing, config=TLMPMConfig(), device="cuda:0", *,
                 position_update=DEFAULT_POSITION_UPDATE, **kwargs):
        if position_update not in POSITION_UPDATES:
            raise ValueError(f"position_update must be one of {POSITION_UPDATES}")
        super().__init__(points, masses, spacing, config, device, **kwargs)
        self.position_update = position_update
        with wp.ScopedDevice(device):
            self.x_start = wp.zeros(self.n, dtype=wp.vec3)
        # step() launches tlmpm_v5 kernels and this module's kernel: load both before any CUDA graph capture
        # (runtimes load only the solver class's own module).
        wp.load_module(module=v5.__name__, device=device)
        wp.load_module(module=__name__, device=device)

    def solver_receipt(self):
        return {"solver": SOLVER_NAME, "status": STATUS, "module": __name__, "position_update": self.position_update,
                "position_rule": ("x_p <- x_p(step start) + dt sum_I w_pI v_I^MUSL after _deformation, before the "
                                  "law-4/5/8 side assignment; v_p (FLIP/PIC blend) unchanged")
                if self.position_update == GRID_MUSL_V1 else "flip_v5 identity: x += dt v_p in _g2p (tlmpm_v5)",
                "source_sha256": source_sha256(), "adr": ADR}

    def step(self, advance=True):
        # tlmpm_v5.TLMPM.step, launch for launch, plus the two grid_musl_v1 insertions marked GRID_X.
        c, n, S = self.config, self.n, self.slots_per_node
        grid_x = self.position_update == GRID_MUSL_V1
        lam = c.young*c.poisson/((1+c.poisson)*(1-2*c.poisson))
        mu = c.young/(2*(1+c.poisson))
        with wp.ScopedDevice(self.device):
            if grid_x:   # GRID_X (1)
                wp.copy(self.x_start, self.x)
            wp.launch(v5._start_cut, n, [self.progress, self.cuts, self.index, self.previous_cut])
            wp.launch(v5._remember_cut, 1, [self.cuts, self.index, self.previous_cut])
            wp.launch(v5._stress, n, [self.F, self.Fpi, self.stress, self.plastic, lam, mu, c.yield_stress])
            self.edge_flags.zero_()
            self.stroke_engaged.zero_(); self.stroke_any_motion.zero_()
            wp.launch(v5._segment_motion, self.edge_segments, [self.poses, self.twists, self.cuts, self.index,
                c.blade_half_width, self.edge_segments, self.law4_parameters["min_speed_m_s"],
                float(np.tan(np.deg2rad(self.law4_parameters["theta_max_deg"]))**2),
                self.segment_ok, self.stroke_any_motion])
            if self.cutting_law == 8:
                wp.launch(v5._segment_plow, self.edge_segments, [self.poses, self.twists, self.index,
                    c.blade_half_width, self.edge_segments,
                    float(np.tan(np.deg2rad(self.law8_parameters["theta_plow_deg"]))**2),
                    self.segment_ok, self.segment_plow])
            wp.launch(v5._rigid_contacts, n, [self.x, self.v, self.mass, self.impulse, self.progress,
                self.contact_work, self.poses, self.twists, self.cuts, self.index,
                self.reaction, self.stats, self.radius, self.volume, c.dt, c.contact_stiffness,
                c.contact_friction, c.cutting_energy, c.sharpness, self.profile,
                len(self.blade_profile_xz), c.blade_half_width, int(c.floor), int(c.fixture_walls), int(c.tool),
                4, self.slice_efficiency, float(self.spacing),  # laws 5 and 8 use the law-4 contact path
                2*c.dx, self.edge_flags,
                self.edge_segments, self.segment_ok, self.stroke_engaged])
            if c.tool:
                for box in range(len(self.obstacle_boxes_edge)):
                    wp.launch(v5._obstacle_contacts, n, [self.x, self.v, self.mass, self.impulse, self.poses,
                        self.twists, self.index, self.reaction, self.radius, c.dt, c.contact_stiffness,
                        c.contact_friction, self.box_centers, self.box_halves, box, self.box_reactions])
            if self.fixture_data is not None:
                kernel, centers, halves, plane = self.fixture_data
                wp.launch(kernel, n, [self.x, self.v, self.mass, self.impulse, centers, halves,
                    plane, self.radius, c.dt, c.contact_stiffness, c.contact_friction, self.stats])
            if c.material_contact:
                self.hash.build(self.x, 2.4*self.radius)
                wp.launch(v5._particle_contacts, n, [self.hash.id, self.x, self.reference,
                    self.v, self.mass, self.code, self.impulse, self.stats, self.radius,
                    3*c.dx, c.dt, c.particle_contact_stiffness, c.contact_friction])
            self.slot_code.fill_(-1)
            wp.launch(v5._claim_slots, (n, 8), [self.code, self.nodes, self.slot_code, self.slot, S,
                self.slot_overflow, self.slot_max_occupied])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._p2g, (n, 8), [self.v, self.mass, self.slot, self.stress, self.impulse,
                self.nodes, self.weights, self.gradients, self.gm, self.gp, self.gj,
                S, self.volume, c.dt, wp.vec3(*c.gravity)])
            wp.launch(v5._grid_update, (S, self.node_count), [self.gm, self.gp, self.gj,
                self.old, self.new_v, self.slot_code, S])
            wp.launch(v5._g2p, n, [self.x, self.v, self.slot, self.nodes, self.weights,
                self.old, self.new_v, S, c.dt, c.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._musl, (n, 8), [self.v, self.mass, self.slot, self.nodes,
                self.weights, self.gp, S])
            wp.launch(v5._grid_update, (S, self.node_count), [self.gm, self.gp, self.gj,
                self.old, self.new_v, self.slot_code, S])
            wp.launch(v5._deformation, n, [self.F, self.slot, self.nodes, self.gradients,
                self.new_v, S, c.dt])
            if grid_x:   # GRID_X (2): positions from the same grid velocity as F, before the side assignment
                wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights,
                    self.new_v, S, c.dt])
            law4 = self.law4_parameters
            wp.launch(v5._update_stroke, 1, [self.twists, self.cuts, self.index, self.stroke_engaged,
                self.stroke_any_motion, self.stroke, self.stroke_travel, c.dt, law4["persist_b_persist_m"]])
            wp.launch(v5._assign_band, n, [self.x, self.v, self.code, self.poses, self.twists, self.cuts,
                self.index, law4["band_h_m"], c.blade_half_width, law4["rear_b_back_m"], c.dt,
                self.stroke, self.segment_ok, self.edge_segments, self.max_cuts])
            if self.cutting_law in (5, 8):
                law5 = self.law5_parameters
                mask = self.segment_plow if self.cutting_law == 8 else self.segment_ok
                wp.launch(v5._assign_lead, n, [self.x, self.code, self.poses, self.twists, self.cuts, self.index,
                    law4["band_h_m"], c.blade_half_width, law5["lead_b_lead_m"], law5["core_s_core_m"],
                    self.stroke, mask, self.edge_segments, self.max_cuts])
            if advance:
                wp.launch(v5._advance, 1, [self.index])
