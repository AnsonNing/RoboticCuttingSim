"""Opt-in finite support-interface experiment derived from WP-14 substep traces.

The v7 position, deformation and contact updates remain shared. A missing side
digit can additionally be completed only after an actual moving support link
crosses the finite edge, while that interface remains inside the physical bevel
and its other endpoint does actual directional knife contact work. No target
plane, task family, selected bridge id, wider label band or particle deletion.
This is experimental logical-grid cutting, not a cohesive fracture model.
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from . import tlmpm_v5 as B, tlmpm_v7 as V

SOLVER_NAME = 'tlmpm_v7_1'
CODE_ENCODING = B.CODE_ENCODING
SlotOverflowError = B.SlotOverflowError


@wp.func
def _interface(p: wp.vec3, q: wp.vec3, pose: wp.transform, twist: wp.spatial_vector):
    a = wp.transform_point(wp.transform_inverse(pose), p)
    b = wp.transform_point(wp.transform_inverse(pose), q)
    sa = B._swept_side(p, pose, twist, a)
    sb = B._swept_side(q, pose, twist, b)
    result = wp.vec4(0.0)
    if sa*sb < 0.0:
        point = a+(b-a)*(sa/(sa-sb))
        result = wp.vec4(point[0], point[1], point[2], 1.0)
    return result


@wp.func
def _endpoint_work(q: int, start: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                   mass: wp.array(dtype=float), pose: wp.transform, twist: wp.spatial_vector,
                   profile: wp.array(dtype=wp.vec2), profile_count: int, width: float,
                   radius: float, dt: float, friction: float, bevel: float, half_thickness: float,
                   slice_efficiency: float):
    local = wp.transform_point(wp.transform_inverse(pose), start[q])
    work = float(0.0)
    if (local[0] >= -bevel and local[0] <= 2.0*radius and wp.abs(local[1]) <= width
            and wp.abs(local[2]) <= half_thickness+radius):
        hit = B._profile_sdf(local, profile, profile_count, width)
        normal = wp.transform_vector(pose, wp.vec3(hit[0],hit[1],hit[2]))
        linear = wp.vec3(twist[0],twist[1],twist[2])
        angular = wp.vec3(twist[3],twist[4],twist[5])
        lever = start[q]-wp.transform_get_translation(pose)-normal*radius
        relative = velocity[q]-linear-wp.cross(angular,lever)
        impulse = V._contact_impulse(hit[3]-radius,normal,relative,mass[q],dt,.1,friction)
        advance = wp.transform_vector(pose,wp.vec3(1.0,0.0,0.0))
        along = wp.transform_vector(pose,wp.vec3(0.0,1.0,0.0))
        normal_b = wp.transform_vector(pose,wp.vec3(0.0,0.0,1.0))
        uc = wp.dot(-relative,advance); us = wp.dot(-relative,along); un = wp.dot(-relative,normal_b)
        in_plane = wp.max(wp.length_sq(relative)-un*un,0.0)
        push = wp.max(wp.dot(impulse,advance),0.0)
        drag = wp.max(wp.dot(impulse,along*wp.sign(us)),0.0)
        work = (push*wp.max(uc,0.0)+slice_efficiency*drag*wp.abs(us))*in_plane/(in_plane+un*un+1.e-20)
    return work


@wp.kernel
def _complete_links(start: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
                    velocity: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                    old_codes: wp.array(dtype=int), codes: wp.array(dtype=int),
                    reached: wp.array(dtype=int), nodes: wp.array2d(dtype=int),
                    weights: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
                    offsets: wp.array(dtype=int), neighbours: wp.array(dtype=int),
                    poses: wp.array(dtype=wp.transform), previous: wp.array(dtype=wp.transform),
                    twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                    stroke: wp.array(dtype=int), segment_ok: wp.array(dtype=int), segments: int,
                    width: float, band: float, bevel: float, half_thickness: float,
                    profile: wp.array(dtype=wp.vec2), profile_count: int, radius: float,
                    dt: float, friction: float, slice_efficiency: float,
                    pending_q: wp.array(dtype=int), pending_cut: wp.array(dtype=int),
                    diagnostics: wp.array(dtype=int)):
    p = wp.tid(); i = index[0]; cut = cuts[i]
    if cut < 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    if B._digit(old_codes[p],cut) != 0:
        return
    d = B._digit(reached[p],cut)
    if d == 0:
        return
    twist = twists[i]
    # The link construction is for a translational finite sweep, not a
    # rotating swept volume. Rotating/disarmed motion retains v7 behavior.
    if wp.length_sq(wp.vec3(twist[3],twist[4],twist[5])) > 1.e-10:
        return
    local = wp.transform_point(wp.transform_inverse(poses[i]),x[p])
    side = B._swept_side(x[p],poses[i],twist,local)
    if wp.abs(side) > band or (d == 1 and side >= 0.0) or (d == 2 and side < 0.0):
        return
    q = int(-1)
    if pending_cut[p] == cut:
        q = pending_q[p]
        if q >= 0:
            here = _interface(x[p],x[q],poses[i],twist)
            if here[3] == 0.0 or here[0] < -bevel or here[0] > 0.0 or wp.abs(here[1]) > width:
                q = -1
                pending_q[p] = -1
    if q < 0:
        for node_index in range(8):
            if q < 0 and (weights[p,node_index] > 0.0 or wp.length_sq(gradients[p,node_index]) > 0.0):
                node = nodes[p,node_index]
                for j in range(offsets[node],offsets[node+1]):
                    if q < 0:
                        other = neighbours[j]
                        if other != p and not B._conflict_packed(old_codes[p],old_codes[other]):
                            old = _interface(start[p],start[other],previous[0],twist)
                            new = _interface(x[p],x[other],poses[i],twist)
                            if (old[3] != 0.0 and new[3] != 0.0 and old[0] >= 0.0 and new[0] <= 0.0
                                    and new[0] >= -bevel and wp.abs(new[1]) <= width and old[0] > new[0]):
                                t = old[0]/(old[0]-new[0])
                                y = old[1]+t*(new[1]-old[1])
                                if wp.abs(y) <= width and segment_ok[B._edge_segment(y,width,segments)] != 0:
                                    q = other
                                    pending_q[p] = q; pending_cut[p] = cut
                                    wp.atomic_add(diagnostics,0,1)
    if q >= 0 and stroke[0] == cut+1:
        here = _interface(x[p],x[q],poses[i],twist)
        if here[3] != 0.0 and here[0] <= 0.0 and here[0] >= -bevel and wp.abs(here[1]) <= width:
            if segment_ok[B._edge_segment(here[1],width,segments)] != 0:
                dw = _endpoint_work(q,start,velocity,mass,poses[i],twist,profile,profile_count,
                                    width,radius,dt,friction,bevel,half_thickness,slice_efficiency)
                if dw > 1.e-20:
                    codes[p] = codes[p] | (d << (2*cut))
                    pending_q[p] = -1
                    wp.atomic_add(diagnostics,1,1)


class TLMPM(V.TLMPM):
    solver_name = SOLVER_NAME

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        nodes = self.nodes.numpy(); weights = self.weights.numpy(); gradients = self.gradients.numpy()
        valid = (weights > 0) | (np.linalg.norm(gradients,axis=-1) > 0)
        node_ids = nodes[valid]
        particle_ids = np.broadcast_to(np.arange(self.n,dtype=np.int32)[:,None],nodes.shape)[valid]
        order = np.argsort(node_ids,kind='stable')
        counts = np.bincount(node_ids,minlength=self.node_count)
        offsets = np.r_[0,np.cumsum(counts)].astype(np.int32)
        with wp.ScopedDevice(self.device):
            self.link_offsets = wp.array(offsets,dtype=int)
            self.link_neighbours = wp.array(particle_ids[order],dtype=int)
            self.link_previous_pose = wp.zeros(1,dtype=wp.transform)
            self.link_velocity_start = wp.zeros(self.n,dtype=wp.vec3)
            self.link_codes = wp.zeros(self.n,dtype=int)
            self.pending_q = wp.full(self.n,-1,dtype=int)
            self.pending_cut = wp.full(self.n,-1,dtype=int)
            self.link_diagnostics = wp.zeros(2,dtype=int)
        wp.load_module(module=__name__,device=self.device)

    def step(self,advance=True):
        with wp.ScopedDevice(self.device):
            wp.copy(self.link_previous_pose,self.previous_pose)
            wp.copy(self.link_velocity_start,self.v)
            super().step(advance=False)
            wp.copy(self.link_codes,self.code)
            wp.launch(_complete_links,self.n,[self.x_start,self.x,self.link_velocity_start,self.mass,
                self.link_codes,self.code,self.front_reached,self.nodes,self.weights,self.gradients,
                self.link_offsets,self.link_neighbours,self.poses,self.link_previous_pose,self.twists,self.cuts,
                self.index,self.stroke,self.segment_ok,self.edge_segments,self.config.blade_half_width,
                self.law4_parameters['band_h_m'],self.bevel_depth_m,self.blade_half_thickness_m,self.profile,
                len(self.blade_profile_xz),self.radius,self.config.dt,self.config.contact_friction,self.slice_efficiency,
                self.pending_q,self.pending_cut,self.link_diagnostics])
            if advance: wp.launch(B._advance,1,[self.index])

    def solver_receipt(self):
        receipt = super().solver_receipt()
        return {**receipt,'solver':SOLVER_NAME,'status':'EXPERIMENTAL_NOT_ACCEPTED',
                'cut_front':'finite_support_interface_v1_1','shared_base_solver':'tlmpm_v7',
                'candidate_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'link_completion_counts':dict(zip(('finite_link_events','completed_side_digits'),
                                                   self.link_diagnostics.numpy().tolist())),
                'scope':'Actual reference-node support links; translational sweep only. Same v7 contact and x/F. '
                'v1_1 clears pending evidence at disarm; its new regression is CPU-verified, GPU outcome NOT_RUN.'}
