"""Versioned bounded rigid-sweep extension of the immutable v7_1 interface law.

Only its translation-only admission changes: a substep may rotate at most
1e-4 rad. Old/new measured transforms still define the same finite interface
crossing, with the same endpoint contact work, side band, bevel, segment,
stroke and history lifecycle. This is a chord approximation, not general
rotational CCD or calibrated fracture. Large rotations clear pending evidence.
Original v7 grid-consistent x/F and contacts are unchanged.
"""
import hashlib
from pathlib import Path
import warp as wp
from . import tlmpm_v5 as B, tlmpm_v7 as V, tlmpm_v7_1 as I
from .tlmpm_v7_1 import _interface, _endpoint_work

SOLVER_NAME='tlmpm_v7_2'
CODE_ENCODING=B.CODE_ENCODING
SlotOverflowError=B.SlotOverflowError
_SHARED_V7_STEP=V.TLMPM.step

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
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    d = B._digit(reached[p],cut)
    if d == 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    twist = twists[i]
    # Same old/new rigid-transform chord interpolation as the v7 front.
    # Bound angular displacement, rather than rejecting measured servo noise.
    # At 25 mm half-width this cap bounds arc/chord sagitta by 3.2e-11 m.
    if wp.length_sq(wp.vec3(twist[3],twist[4],twist[5]))*dt*dt > 1.e-8:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    local = wp.transform_point(wp.transform_inverse(poses[i]),x[p])
    side = B._swept_side(x[p],poses[i],twist,local)
    if wp.abs(side) > band or (d == 1 and side >= 0.0) or (d == 2 and side < 0.0):
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    q = int(-1)
    if pending_cut[p] != cut:
        pending_q[p] = -1
        pending_cut[p] = -1
    if pending_cut[p] == cut:
        q = pending_q[p]
        if q >= 0:
            here = _interface(x[p],x[q],poses[i],twist)
            if here[3] == 0.0 or here[0] < -bevel or here[0] > 0.0 or wp.abs(here[1]) > width:
                q = -1
                pending_q[p] = -1
                pending_cut[p] = -1
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
                    pending_cut[p] = -1
                    wp.atomic_add(diagnostics,1,1)


class TLMPM(I.TLMPM):
    solver_name=SOLVER_NAME

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        wp.load_module(module=__name__,device=self.device)

    def step(self,advance=True):
        with wp.ScopedDevice(self.device):
            wp.copy(self.link_previous_pose,self.previous_pose)
            wp.copy(self.link_velocity_start,self.v)
            _SHARED_V7_STEP(self,advance=False)
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
        receipt=super().solver_receipt()
        return {**receipt,'solver':SOLVER_NAME,'cut_front':'finite_support_interface_rigid_chord_v2',
            'candidate_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parent_candidate_source_sha256':hashlib.sha256(Path(I.__file__).read_bytes()).hexdigest(),
            'maximum_rotation_per_substep_rad':1.e-4,
            'scope':'Experimental bounded rigid-transform chord sweep; same endpoint-work and finite-interface gates as v7_1 v1_2. '
                    'No general rotational CCD or physical calibration; original v7 CUT profile unchanged.'}
