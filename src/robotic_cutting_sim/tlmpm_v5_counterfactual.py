"""COUNTERFACTUAL tlmpm_v5 variants for the WP-12 Phase 4 root-cause verification (V1-V4). NOT A GATE.

Label: ``COUNTERFACTUAL_NOT_A_GATE``. Results computed with this module are counterfactual measurements that
test a diagnosed mechanism. They are never acceptance results, never a cutting law, never a material or surface
profile, and never enter a verdict on law 8 or on a supported configuration. The module is deliberately not
importable through any production entry point: ``LiveCuttingMaterial``, ``blade_probe``, ``scene_config``,
``policy_env`` and every preset select ``tlmpm_v5`` (or older) only; ``tests/test_tlmpm_v5_counterfactual.py``
checks that no module under ``src/`` other than this one names it. ``tlmpm_v5.py`` is not modified.

``CounterfactualTLMPM`` subclasses ``tlmpm_v5.TLMPM`` and replaces only ``step()``. Its body is the v5 step, launch
for launch (same kernels from ``tlmpm_v5``, same order, same arguments) except for four opt-in hooks, each of
which defaults to the v5 behaviour:

* ``position_update='grid_musl'``: before the step x_prev <- x; after ``_deformation`` (and before the law-4/5/8
  side assignment, which therefore sees the new positions) x <- x_prev + dt * sum_I w_pI v_I^MUSL, i.e. particles
  are advected with the same MUSL grid velocity that updates F. The FLIP-blended particle velocity v_p is kept.
  Default ``'flip'``: x += dt v_p in ``_g2p`` (v5).
* ``pic``: replaces ``config.pic`` (the PIC fraction alpha of the G2P blend). Default None: the config value.
* ``blade_contact_stiffness``: replaces the penalty factor k of the *blade* contact in ``_rigid_contacts`` only
  (board/table fixtures, the handle obstacle and particle-particle contact keep the config values). Default None.
* ``cutting_zone_m``: replaces the cutting-zone half thickness |local z| <= zone of ``_rigid_contacts`` (v5 passes
  the particle spacing s). Default None.

With every option at its default ``step()`` is bitwise equal to ``tlmpm_v5.TLMPM.step`` on CPU (tested), so the
copy cannot silently drift from v5.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
import warp as wp

from . import tlmpm_v5 as v5
from .material_config import TLMPMConfig

COUNTERFACTUAL_LABEL = "COUNTERFACTUAL_NOT_A_GATE"
POSITION_UPDATES = ("flip", "grid_musl")


@dataclass(frozen=True)
class CounterfactualOptions:
    position_update: str = "flip"
    pic: float = None
    blade_contact_stiffness: float = None
    cutting_zone_m: float = None

    def __post_init__(self):
        if self.position_update not in POSITION_UPDATES:
            raise ValueError(f"position_update must be one of {POSITION_UPDATES}")
        if self.pic is not None and not (np.isfinite(self.pic) and 0 <= self.pic <= 1):
            raise ValueError("pic must be in [0, 1]")
        if self.blade_contact_stiffness is not None and not (np.isfinite(self.blade_contact_stiffness)
                                                             and self.blade_contact_stiffness > 0):
            raise ValueError("blade_contact_stiffness must be finite and positive")
        if self.cutting_zone_m is not None and not (np.isfinite(self.cutting_zone_m) and self.cutting_zone_m > 0):
            raise ValueError("cutting_zone_m must be finite and positive")

    def is_identity(self):
        return self == CounterfactualOptions()

    def to_dict(self):
        return asdict(self)


@wp.kernel
def _grid_positions(x: wp.array(dtype=wp.vec3), x_prev: wp.array(dtype=wp.vec3), slot: wp.array2d(dtype=int),
                    nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
                    velocity: wp.array(dtype=wp.vec3), slots: int, dt: float):
    """COUNTERFACTUAL: x_p <- x_p(start of step) + dt sum_I w_pI v_I (same field and slots as _deformation)."""
    p = wp.tid()
    vg = wp.vec3(0.0)
    for corner in range(8):
        s = slot[p, corner]
        if s >= 0:
            vg += weights[p, corner]*velocity[nodes[p, corner]*slots+s]
    x[p] = x_prev[p]+dt*vg


class CounterfactualTLMPM(v5.TLMPM):
    """tlmpm_v5 with opt-in counterfactual hooks (COUNTERFACTUAL_NOT_A_GATE; see the module docstring)."""
    counterfactual_label = COUNTERFACTUAL_LABEL

    def __init__(self, points, masses, spacing, config=TLMPMConfig(), device="cuda:0", *, counterfactual=None,
                 **kwargs):
        options = counterfactual if isinstance(counterfactual, CounterfactualOptions) else \
            CounterfactualOptions(**(counterfactual or {}))
        if options.pic is not None:
            config = replace(config, pic=float(options.pic))
        if options.blade_contact_stiffness is not None and config.floor:
            raise ValueError("blade_contact_stiffness: _rigid_contacts shares k with the legacy tray; use floor=False")
        super().__init__(points, masses, spacing, config, device, **kwargs)
        self.counterfactual = options
        with wp.ScopedDevice(device):
            self.x_prev = wp.zeros(self.n, dtype=wp.vec3)

    def blade_contact_parameters(self):
        """(blade penalty factor k, cutting-zone half thickness, zone rear) as passed to _rigid_contacts."""
        o, c = self.counterfactual, self.config
        return (float(c.contact_stiffness if o.blade_contact_stiffness is None else o.blade_contact_stiffness),
                float(self.spacing if o.cutting_zone_m is None else o.cutting_zone_m), float(2*c.dx))

    def counterfactual_receipt(self):
        return {"label": COUNTERFACTUAL_LABEL, "module": __name__, "options": self.counterfactual.to_dict(),
                "identity": self.counterfactual.is_identity(), "effective_pic": float(self.config.pic),
                "blade_contact_parameters": dict(zip(("stiffness", "zone_m", "zone_rear_m"),
                                                     self.blade_contact_parameters()))}

    def step(self, advance=True):
        # tlmpm_v5.TLMPM.step, launch for launch, with the four hooks marked COUNTERFACTUAL.
        c, n, S = self.config, self.n, self.slots_per_node
        o = self.counterfactual
        blade_k, zone, zone_rear = self.blade_contact_parameters()
        grid_x = o.position_update == "grid_musl"
        lam = c.young*c.poisson/((1+c.poisson)*(1-2*c.poisson))
        mu = c.young/(2*(1+c.poisson))
        with wp.ScopedDevice(self.device):
            if grid_x:   # COUNTERFACTUAL
                wp.copy(self.x_prev, self.x)
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
                self.reaction, self.stats, self.radius, self.volume, c.dt,
                blade_k,   # COUNTERFACTUAL hook (v5: c.contact_stiffness; also the legacy tray, off in every row)
                c.contact_friction, c.cutting_energy, c.sharpness, self.profile,
                len(self.blade_profile_xz), c.blade_half_width, int(c.floor), int(c.fixture_walls), int(c.tool),
                4, self.slice_efficiency,
                zone,      # COUNTERFACTUAL hook (v5: float(self.spacing))
                zone_rear, self.edge_flags,
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
            if grid_x:   # COUNTERFACTUAL: positions from the same grid velocity as F, before the side assignment
                wp.launch(_grid_positions, n, [self.x, self.x_prev, self.slot, self.nodes, self.weights,
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
