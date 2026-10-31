"""Opt-in WP-13 integrated cutting experiment; uncalibrated, never a default.

Contact uses velocity-level nonpenetration with a fixed penetration relaxation
 time. Both x and F integrate the MUSL field. A digit is written only when the
finite advancing blade front crosses the moving material layer during that
same substep, with directional motion and actual cutting work engagement.
No particle deletion, lead assignment, mesh smoothing or stress-driven fracture.
"""
import hashlib
from pathlib import Path
import numpy as np
import warp as wp
from . import tlmpm_v5 as v5
from .material_config import TLMPMConfig
from .tlmpm_v6 import _grid_positions

SOLVER_NAME = "tlmpm_v7"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
CONTACT_RELAXATION_S = 0.0002
EDGE_RELEASE_DEPTH_M = 0.0005
EDGE_RELEASE_HALF_THICKNESS_M = 0.00025
CODE_ENCODING = v5.CODE_ENCODING
SlotOverflowError = v5.SlotOverflowError

@wp.func
def _contact_impulse(gap: float, normal: wp.vec3, relative_v: wp.vec3,
                     mass: float, dt: float, stiffness: float, friction: float):
    # Constraint impulse in kg m/s; tau is numerical stabilization, not calibration.
    vn = wp.dot(relative_v, normal)
    jn = float(0.0)
    if gap + dt * wp.min(vn, 0.0) < 0.0:
        target = -wp.min(gap, 0.0) / 0.0002
        jn = mass * wp.max(target - vn, 0.0)
    tangent = relative_v - vn * normal
    speed = wp.length(tangent)
    jt = wp.vec3(0.0)
    if speed > 1.e-10:
        jt = -wp.min(mass * speed, friction * jn) * tangent / speed
    return jn * normal + jt

@wp.kernel
def _rigid_contacts(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                    mass: wp.array(dtype=float), impulse: wp.array(dtype=wp.vec3),
                    progress: wp.array(dtype=float), contact_work: wp.array(dtype=float),
                    poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                    cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                    reaction: wp.array(dtype=wp.spatial_vector), stats: wp.array(dtype=float),
                    radius: float, volume: float, dt: float, stiffness: float,
                    friction: float, gc: float, sharpness: float, profile: wp.array(dtype=wp.vec2),
                    profile_count: int, blade_width: float, floor: int, walls: int, tool: int,
                    law: int, slice_efficiency: float, zone: float, zone_rear: float, flags: wp.array(dtype=int),
                    segments: int, seg_ok: wp.array(dtype=int), engaged: wp.array(dtype=int),
                    totals: wp.array(dtype=wp.float64)):
    p = wp.tid()
    impulse[p] = wp.vec3(0.0)
    xp, vp, mp = x[p], v[p], mass[p]
    if floor != 0:
        # Finite tray boxes, so a detached chip can clear a wall and land on the
        # surrounding table. Half-space wall tests would spuriously accelerate
        # an already-outside chip when it falls below the wall's top edge.
        for side in range(6):
            center = wp.vec3(0.0, 0.0, -0.005)
            half = wp.vec3(0.068, 0.048, 0.005)
            if side == 1:
                center = wp.vec3(-0.064, 0.0, 0.0075)
                half = wp.vec3(0.004, 0.044, 0.0075)
            elif side == 2:
                center = wp.vec3(0.064, 0.0, 0.0075)
                half = wp.vec3(0.004, 0.044, 0.0075)
            elif side == 3:
                center = wp.vec3(0.0, -0.044, 0.0075)
                half = wp.vec3(0.060, 0.004, 0.0075)
            elif side == 4:
                center = wp.vec3(0.0, 0.044, 0.0075)
                half = wp.vec3(0.060, 0.004, 0.0075)
            hit = v5._box_sdf(xp-center, half)
            normal = wp.vec3(hit[0], hit[1], hit[2])
            gap = hit[3]-radius
            if side == 5:
                normal = wp.vec3(0.0, 0.0, 1.0)
                gap = xp[2]+0.010-radius  # support table's top z=-10mm
            if side == 0 or side == 5 or walls != 0:
                impulse[p] = impulse[p]+_contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
                wp.atomic_max(stats, 0, wp.max(-gap, 0.0))
    if tool != 0:
        i = index[0]
        pose = poses[i]
        local = wp.transform_point(wp.transform_inverse(pose), xp)
        hit = v5._profile_sdf(local, profile, profile_count, blade_width)
        normal = wp.transform_vector(pose, wp.vec3(hit[0], hit[1], hit[2]))
        gap = hit[3]-radius
        twist = twists[i]
        # This module's twist convention is [linear, angular], unlike wp.spatial.
        linear = wp.vec3(twist[0], twist[1], twist[2])
        angular = wp.vec3(twist[3], twist[4], twist[5])
        lever = xp-wp.transform_get_translation(pose)-normal*radius
        relative = vp-linear-wp.cross(angular, lever)
        ji = _contact_impulse(gap, normal, relative, mp, dt, stiffness, friction)
        advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
        uc = wp.dot(-relative, advance)
        if law == 1:
            if cuts[i] >= 0 and local[0] >= 0.0 and local[0] <= 2.0*radius and uc > 1.0e-5:
                dw = wp.max(wp.dot(ji, advance), 0.0)*uc
                threshold = gc*wp.pow(volume, 2.0/3.0)*(1.0-sharpness)/sharpness
                remaining = wp.max((1.0-progress[p])*threshold, 0.0)
                fraction = float(1.0)
                if dw > 1.0e-20:
                    fraction = wp.min(remaining/dw, 1.0)
                contact_work[p] = contact_work[p]+fraction*dw
                progress[p] = wp.min(progress[p]+fraction*dw/wp.max(threshold, 1.0e-12), 1.0)
                ji = ji-(1.0-fraction)*wp.max(wp.dot(ji, advance), 0.0)*advance
        else:
            in_zone = (local[0] >= -zone_rear and local[0] <= 2.0*radius and wp.abs(local[2]) <= zone+radius
                       and wp.abs(local[1]) <= blade_width)
            if cuts[i] >= 0 and in_zone:
                along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
                us = wp.dot(-relative, along)
                slice_dir = along*wp.sign(us)
                push = wp.max(wp.dot(ji, advance), 0.0)
                drag = wp.max(wp.dot(ji, slice_dir), 0.0)
                # Only the in-plane part of the edge's relative motion opens the
                # crack: weight by cos^2 of the angle between motion and blade plane.
                blade_normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
                un = wp.dot(-relative, blade_normal)
                in_plane = wp.max(wp.length_sq(relative)-un*un, 0.0)
                alignment = in_plane/(in_plane+un*un+1.0e-20)
                dw = (push*wp.max(uc, 0.0)+slice_efficiency*drag*wp.abs(us))*alignment
                if dw > 1.0e-20:
                    threshold = gc*wp.pow(volume, 2.0/3.0)*(1.0-sharpness)/sharpness
                    remaining = wp.max((1.0-progress[p])*threshold, 0.0)
                    fraction = wp.min(remaining/dw, 1.0)
                    contact_work[p] = contact_work[p]+fraction*dw
                    wp.atomic_add(totals, 5, wp.float64(fraction*dw))
                    progress[p] = wp.min(progress[p]+fraction*dw/wp.max(threshold, 1.0e-12), 1.0)
                    # Material already cut no longer resists in-plane edge motion.
                    if local[0] >= -0.0005 and wp.abs(local[2]) <= 0.00025+radius:
                        ji = ji-(1.0-fraction)*(push*advance+drag*slice_dir)
                if law == 4:
                    # Law 4 engagement W: directional cutting work at a segment
                    # with actual cutting motion M_k (stroke gate, not per segment).
                    if dw > 1.0e-20 and seg_ok[v5._edge_segment(local[1], blade_width, segments)] != 0:
                        wp.atomic_max(engaged, 0, 1)
                else:
                    segment_motion = linear + wp.cross(angular, along * local[1])
                    advancing = wp.dot(segment_motion, advance)
                    drawing = wp.dot(segment_motion, along)
                    actual_edge_motion = advancing > 1.0e-5 or (wp.abs(drawing) > 1.0e-5 and advancing >= -1.0e-5)
                    if progress[p] >= 1.0 and (law != 3 or (actual_edge_motion and dw > 1.0e-20)):
                        wp.atomic_max(flags, v5._edge_segment(local[1], blade_width, segments), 1)
        if wp.length_sq(ji) > 0.0:
            jn = wp.dot(ji, normal)
            wp.atomic_add(totals, 0, wp.float64(jn))
            wp.atomic_add(totals, 1, wp.float64(wp.length(ji-jn*normal)))
            partition = int(3)
            if local[0] >= -zone_rear and local[0] <= 2.0*radius:
                partition = 2
            wp.atomic_add(totals, partition, wp.float64(wp.length(ji)))
            wp.atomic_add(totals, 4, wp.float64(wp.dot(ji, linear+wp.cross(angular, lever))))
            wp.atomic_add(totals, 9, wp.float64(wp.dot(ji, vp)+0.5*wp.length_sq(ji)/mp))
            wp.atomic_max(totals, 11, wp.float64(wp.max(-gap, 0.0)))
        impulse[p] = impulse[p]+ji
        torque = wp.cross(lever, -ji)
        wp.atomic_add(reaction, 0, wp.spatial_vector(-ji[0], -ji[1], -ji[2], torque[0], torque[1], torque[2]))
        if wp.length_sq(ji) > 0.0:
            wp.atomic_add(stats, 1, 1.0)
            wp.atomic_add(stats, 2, wp.dot(ji, linear+wp.cross(angular, lever)))
            wp.atomic_max(stats, 3, wp.max(-gap, 0.0))


@wp.kernel
def fixture_contacts(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                     mass: wp.array(dtype=float), impulse: wp.array(dtype=wp.vec3),
                     centers: wp.array(dtype=wp.vec3), halves: wp.array(dtype=wp.vec3),
                     plane: wp.vec4, radius: float, dt: float,
                     stiffness: float, friction: float, stats: wp.array(dtype=float),
                     totals: wp.array(dtype=wp.float64)):
    p = wp.tid()
    xp, vp, mp = x[p], v[p], mass[p]
    for box in range(centers.shape[0]):
        hit = v5._box_sdf(xp-centers[box], halves[box])
        normal = wp.vec3(hit[0], hit[1], hit[2])
        gap = hit[3]-radius
        ji = _contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
        impulse[p] = impulse[p]+ji
        if wp.length_sq(ji) > 0.0:
            jn = wp.dot(ji, normal)
            wp.atomic_add(totals, 6, wp.float64(jn))
            wp.atomic_add(totals, 7, wp.float64(wp.length(ji-jn*normal)))
            wp.atomic_add(totals, 8, wp.float64(wp.dot(ji, vp)+0.5*wp.length_sq(ji)/mp))
            wp.atomic_max(totals, 10, wp.float64(wp.max(-gap, 0.0)))
        wp.atomic_max(stats, 0, wp.max(-gap, 0.0))
    normal = wp.vec3(plane[0], plane[1], plane[2])
    gap = wp.dot(normal, xp)+plane[3]-radius
    ji = _contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
    impulse[p] = impulse[p]+ji
    if wp.length_sq(ji) > 0.0:
        jn = wp.dot(ji, normal)
        wp.atomic_add(totals, 6, wp.float64(jn))
        wp.atomic_add(totals, 7, wp.float64(wp.length(ji-jn*normal)))
        wp.atomic_add(totals, 8, wp.float64(wp.dot(ji, vp)+0.5*wp.length_sq(ji)/mp))
        wp.atomic_max(totals, 10, wp.float64(wp.max(-gap, 0.0)))
    wp.atomic_max(stats, 0, wp.max(-gap, 0.0))

@wp.kernel
def _assign_front(x0: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
                  codes: wp.array(dtype=int), previous: wp.array(dtype=wp.transform),
                  poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                  cuts: wp.array(dtype=int), index: wp.array(dtype=int), stroke: wp.array(dtype=int),
                  seg_ok: wp.array(dtype=int), width: float, band: float, segments: int,
                  max_cuts: int, crossed: wp.array(dtype=int), reached: wp.array(dtype=int),
                  bevel_depth: float):
    p = wp.tid()
    i, cut = index[0], cuts[index[0]]
    if cut >= 0 and cut < max_cuts and v5._digit(codes[p], cut) == 0:
        a = wp.transform_point(wp.transform_inverse(previous[0]), x0[p])
        b = wp.transform_point(wp.transform_inverse(poses[i]), x[p])
        # Actual relative sweep across the leading plane; no rear/lead closure.
        if a[0] >= 0.0 and b[0] <= 0.0 and a[0] > b[0] and v5._digit(reached[p], cut) == 0:
            t = a[0] / (a[0]-b[0])
            y = a[1] + t*(b[1]-a[1])
            side = v5._swept_side(x[p], poses[i], twists[i], b)
            if wp.abs(y) <= width and wp.abs(side) <= band:
                if seg_ok[v5._edge_segment(y, width, segments)] != 0:
                    digit = int(2)
                    if side < 0.0:
                        digit = 1
                    reached[p] = reached[p] | (digit << (2*cut))
                    crossed[p] = crossed[p]+1
        # Work can first occur on the resolved bevel after the infinitesimal
        # edge passed a layer. Accept only its recorded sweep, while still at
        # that physical bevel; never label a layer the edge has not traversed.
        d = v5._digit(reached[p], cut)
        if d != 0 and stroke[0] == cut+1 and b[0] <= 0.0 and b[0] >= -bevel_depth:
            if wp.abs(b[1]) <= width and seg_ok[v5._edge_segment(b[1], width, segments)] != 0:
                codes[p] = codes[p] | (d << (2*cut))

@wp.kernel
def _remember_pose(previous: wp.array(dtype=wp.transform), poses: wp.array(dtype=wp.transform),
                   index: wp.array(dtype=int)):
    previous[0] = poses[index[0]]


class TLMPM(v5.TLMPM):
    solver_name = SOLVER_NAME

    def __init__(self, points, masses, spacing, config=TLMPMConfig(), device="cuda:0", **kwargs):
        if kwargs.get('cutting_law', 8) != 8:
            raise ValueError('tlmpm_v7 is an integrated law-8-derived experiment only')
        super().__init__(points, masses, spacing, config, device, **kwargs)
        with wp.ScopedDevice(device):
            self.x_start = wp.zeros(self.n, dtype=wp.vec3)
            self.previous_pose = wp.array(self.poses.numpy()[:1], dtype=wp.transform)
            self.front_crossings = wp.zeros(self.n, dtype=int)
            self.front_reached = wp.zeros(self.n, dtype=int)
            self.contact_totals = wp.zeros(12, dtype=wp.float64)
        behind = self.blade_profile_xz[:, 0]
        self.bevel_depth_m = float(-behind[behind < 0].max())
        self.blade_half_thickness_m = float(np.abs(self.blade_profile_xz[:, 1]).max())
        if self.fixture_data is not None:
            _, centers, halves, plane = self.fixture_data
            self.fixture_data = (fixture_contacts, centers, halves, plane)
        wp.load_module(module=v5.__name__, device=device)
        wp.load_module(module="robotic_cutting_sim.tlmpm_v6", device=device)
        wp.load_module(module=__name__, device=device)

    def set_trajectory(self, poses, twists, cuts):
        super().set_trajectory(poses, twists, cuts)
        if hasattr(self, "previous_pose"):
            self.previous_pose.assign(np.asarray(poses[:1], np.float32))

    def solver_receipt(self):
        paths = ("tlmpm_v5.py", "tlmpm_v6.py", "tlmpm_v7.py")
        return {"solver": SOLVER_NAME, "status": STATUS,
                "position_update": "grid_musl_v1", "cut_front": "finite_relative_sweep_v1",
                "diagnostic_accumulation": "float64",
                "contact": "velocity_projection_tau_v1", "contact_relaxation_s": CONTACT_RELAXATION_S,
                "work_bevel_depth_m": self.bevel_depth_m,
                "edge_release_depth_m": EDGE_RELEASE_DEPTH_M,
                "edge_release_half_thickness_m": EDGE_RELEASE_HALF_THICKNESS_M,
                "source_sha256": {p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                                  for p in paths}}

    def contact_receipt(self):
        names = ('blade_normal_impulse_Ns', 'blade_tangent_impulse_Ns', 'bevel_impulse_norm_sum_Ns',
                 'flank_impulse_norm_sum_Ns', 'tool_work_J', 'consumed_cutting_work_J',
                 'fixture_normal_impulse_Ns', 'fixture_friction_impulse_Ns',
                 'fixture_pretransfer_energy_change_J', 'blade_pretransfer_energy_change_J',
                 'fixture_penetration_max_m', 'blade_penetration_max_m')
        return {'values': dict(zip(names, self.contact_totals.numpy().astype(float).tolist())),
                'scope': 'Cumulative candidate contacts; impulse norms are not vector net impulses. Energy is '
                         'the particle impulse expression before grid transfer, not total-system energy. '
                         'Flank/edge denotes physical bevel-depth partition; material contacts excluded.'}

    def step(self, advance=True):
        # tlmpm_v5.TLMPM.step, launch for launch, plus the two grid_musl_v1 insertions marked GRID_X.
        c, n, S = self.config, self.n, self.slots_per_node
        grid_x = True
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
            wp.launch(_rigid_contacts, n, [self.x, self.v, self.mass, self.impulse, self.progress,
                self.contact_work, self.poses, self.twists, self.cuts, self.index,
                self.reaction, self.stats, self.radius, self.volume, c.dt, c.contact_stiffness,
                c.contact_friction, c.cutting_energy, c.sharpness, self.profile,
                len(self.blade_profile_xz), c.blade_half_width, int(c.floor), int(c.fixture_walls), int(c.tool),
                4, self.slice_efficiency, self.blade_half_thickness_m,
                self.bevel_depth_m, self.edge_flags,
                self.edge_segments, self.segment_ok, self.stroke_engaged, self.contact_totals])
            if c.tool:
                for box in range(len(self.obstacle_boxes_edge)):
                    wp.launch(v5._obstacle_contacts, n, [self.x, self.v, self.mass, self.impulse, self.poses,
                        self.twists, self.index, self.reaction, self.radius, c.dt, c.contact_stiffness,
                        c.contact_friction, self.box_centers, self.box_halves, box, self.box_reactions])
            if self.fixture_data is not None:
                kernel, centers, halves, plane = self.fixture_data
                wp.launch(kernel, n, [self.x, self.v, self.mass, self.impulse, centers, halves,
                    plane, self.radius, c.dt, c.contact_stiffness, c.contact_friction, self.stats, self.contact_totals])
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
            wp.launch(_assign_front, n, [self.x_start, self.x, self.code, self.previous_pose,
                self.poses, self.twists, self.cuts, self.index, self.stroke, self.segment_ok,
                c.blade_half_width, law4["band_h_m"], self.edge_segments, self.max_cuts, self.front_crossings,
                self.front_reached, self.bevel_depth_m])
            wp.launch(_remember_pose, 1, [self.previous_pose, self.poses, self.index])
            if advance:
                wp.launch(v5._advance, 1, [self.index])
