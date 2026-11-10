"""Rubber contact law ``grid_node_coulomb_v1`` and its kick-paired contact work (ADR 0023 §22; P2-4).

Opt-in. Shared by the single-env solver ``tlmpm_rubber_contact_v1.TLMPMRubberContact`` and its batched execution
(kernels index envs through ``env_of_node``; rows of env e are ``e*P + k`` for planes, ``e*F + f`` for frames and
``(e*F + f)*B + b`` for boxes). Nothing here changes a frozen kernel; the frozen per-particle helper
``tlmpm_v5._contact_impulse`` (and the ``table_plane`` / ``fixture_boxes`` path of ``tlmpm_rubber_v1``) stays as it is.

Obstacles (per env, uniform counts per batch): P static half-spaces ``n . x + d >= 0`` (unit normal into the free
side; friction ``friction_table``), and F rigid kinematic frames with B boxes each (oriented boxes given in the
frame's reference coordinates; the frame moves rigidly, x = R(s) X + b(s), with the slerp / linear interpolation of
the rubber BC groups, s = substep / substeps; frame kind ``fixture`` uses ``friction_table``, ``pad`` uses
``friction_pad``).

Law (one thread per grid node; TLMPM reference-grid nodes are material points of the Q1 discretisation, so with the
strip on the particle lattice the outer nodes lie on the body faces). Applied in the KICK: after the P2G grid update
(``new_v`` = p/m + forces dt/m, the predicted velocity of the node), before G2P. For each obstacle in fixed order
(planes, then frames, then boxes), with x_I = X_I + u_I (u_I the float64 node displacement, advanced with the drift
velocity every substep), the gap g and normal n of the obstacle at the start of the substep and the obstacle velocity
v_o at x_I (position-exact rigid motion over the substep):

    v_rel = v - v_o,  v_n = v_rel . n,  v_min = -g / dt (g >= 0: speculative, the node may close the gap)
                                         v_min = -beta g / dt (g < 0: a fraction beta of the penetration per substep)
    if v_n < v_min:   dv_n = v_min - v_n > 0
                      v_t = v_rel - v_n n;  stick if |v_t| <= mu dv_n (dv_t = -v_t) else slip (dv_t = -mu dv_n v_t/|v_t|)
                      v += dv_n n + dv_t;  impulse on the material J = m_I (dv_n n + dv_t)

i.e. a velocity-level non-penetration projection of the node with the Coulomb cone |dv_t| <= mu dv_n evaluated with
the post-force (predicted) velocity: a body at rest in static equilibrium inside the cone is an exact fixed point
(no creep), and in slip the friction impulse is exactly mu times the normal impulse at every node. Nodes held by an
active Dirichlet group are skipped (the BC prescribes them).

Work (``contact_work = kick_average_v1``; why: ADR 0023 §21.1 / §21.7). Pure FLIP + MUSL with fixed weights is
symplectic Euler in (pi, u) with KE = 1/2 pi^T B pi (the particle KE), and the contact impulse enters the kick
pi_{n+1} = pi_n + dt f + J. Hence KE_{n+1} - KE_n = (dt f + J) . (g_n + g_{n+1}) / 2 exactly, where g_n = B pi_n is
``old`` after the P2G grid update and g_{n+1} = B pi_{n+1} is ``old`` after the (final) MUSL grid update. The
material-side contact work is accumulated in two halves (J . old / 2 at each moment), split into its normal and
tangential parts; the obstacle-side power J . v_o is accumulated separately. Columns of the per-env ``work[N, 4]``:

    0  W_n   material-side normal contact work      sum J_n . (g_n + g_{n+1}) / 2
    1  W_t   material-side tangential contact work  sum J_t . (g_n + g_{n+1}) / 2
    2  P_n   obstacle-side normal work              sum J_n . v_o
    3  P_t   obstacle-side tangential work          sum J_t . v_o

so that the books B = KE + Psi + E_grav - W_ext - (W_n + W_t) [+ W_b for the Prony branches] close up to the O(dt)
symplectic error, the friction dissipation is D_t = P_t - W_t (static obstacles: -W_t) and the normal (projection)
dissipation is D_n = P_n - W_n. Coupling or contact numerics are therefore never booked as material damping.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

PENALTY_LAW = "grid_node_penalty_coulomb_v1"           # node penalty + Coulomb spring-slider (ADR 0023 §22.1)
PROJECTION_LAW = "grid_node_coulomb_v1"                 # node velocity-level projection (ADR 0023 §22.1)
PENALTY_LAW_V2 = "grid_node_penalty_coulomb_v2"        # v1 with the dashpots on the realised node velocity (§22.9)
CONTACT_LAW = PENALTY_LAW                               # default: v1 (v2 is opt-in)
CONTACT_LAWS = (PENALTY_LAW, PROJECTION_LAW, PENALTY_LAW_V2)
PENALTY_LAWS = (PENALTY_LAW, PENALTY_LAW_V2)
CONTACT_WORK = "kick_average_v1"
FRAME_FIXTURE, FRAME_PAD = 0, 1
FRAME_KINDS = {"fixture": FRAME_FIXTURE, "pad": FRAME_PAD}
DEFAULT_PENETRATION_RECOVERY = 0.1
DEFAULT_NORMAL_STIFFNESS = 0.25          # k_n = kappa_n m_I / dt^2
DEFAULT_TANGENTIAL_STIFFNESS = 0.25      # k_t = kappa_t m_I / dt^2
DEFAULT_DAMPING_RATIO = 0.5              # zeta_t of the tangential dashpot c_t = 2 zeta_t sqrt(k_t m_I) (§22.1)
DEFAULT_NORMAL_DAMPING_RATIO = 0.5       # zeta_n of the normal dashpot (§22.1: needed against friction flutter)
WORK_COLUMNS = ("normal_work_material", "tangential_work_material", "normal_work_obstacle",
                "tangential_work_obstacle", "coulomb_dissipation", "damping_dissipation")
NWORK = len(WORK_COLUMNS)
COUNT_COLUMNS = ("impulse_events", "stick_events", "slip_events")
STATE_STICK, STATE_SLIP = 1, 2
ADR = "docs/adr/0023-rubber-material-v1.md section 22"
_HERE = Path(__file__).resolve()


def source_sha256():
    return {_HERE.name: hashlib.sha256(_HERE.read_bytes()).hexdigest()}


# ----------------------------------------------------------------------------- spec

@dataclass(frozen=True)
class ContactFrame:
    """A rigid kinematic frame with boxes ((centre xyz), (half extents xyz)) in its reference coordinates (the frame's
    pose is identity at reset; motions x = R X + b are relative to the reference, as the BC groups). ``origin``: the
    point (reference coordinates) about which box torques are reported."""
    kind: str = "fixture"
    boxes: tuple = ()
    origin: tuple = (0.0, 0.0, 0.0)
    name: str = ""


@dataclass(frozen=True)
class ContactSpec:
    """Obstacles of one env and the law's numerical parameter (penetration recovery fraction beta per substep)."""
    planes: tuple = ()                 # ((nx, ny, nz, d), ...): gap = n . x + d, n unit, pointing into the free side
    frames: tuple = ()                 # (ContactFrame, ...), every frame with the same number of boxes
    law: str = CONTACT_LAW
    penetration_recovery: float = DEFAULT_PENETRATION_RECOVERY      # projection law only
    normal_stiffness: float = DEFAULT_NORMAL_STIFFNESS              # penalty law only
    tangential_stiffness: float = DEFAULT_TANGENTIAL_STIFFNESS      # penalty law only
    damping_ratio: float = DEFAULT_DAMPING_RATIO                    # penalty law only (tangential, stick branch)
    normal_damping_ratio: float = DEFAULT_NORMAL_DAMPING_RATIO      # penalty law only
    work: str = CONTACT_WORK

    def __post_init__(self):
        if self.law not in CONTACT_LAWS:
            raise ValueError(f"contact law {self.law!r} not in {CONTACT_LAWS}")
        if self.work != CONTACT_WORK:
            raise ValueError(f"contact work {self.work!r}: only {CONTACT_WORK!r} (kick-paired, ADR 0023 §22)")
        beta = float(self.penetration_recovery)
        if not (np.isfinite(beta) and 0.0 < beta <= 1.0):
            raise ValueError("penetration_recovery must be in (0, 1]")
        for name in ("normal_stiffness", "tangential_stiffness"):
            k = float(getattr(self, name))
            if not (np.isfinite(k) and 0.0 < k <= 1.0):
                raise ValueError(f"{name} must be in (0, 1] (fraction of m_I / dt^2)")
        for name, kappa in (("damping_ratio", self.tangential_stiffness),
                            ("normal_damping_ratio", self.normal_stiffness)):
            z = float(getattr(self, name))
            if not (np.isfinite(z) and z >= 0.0 and 2*z*np.sqrt(kappa) <= 1.0):
                raise ValueError(f"{name} must be >= 0 with 2 zeta sqrt(kappa) <= 1 (at most the node's whole "
                                 "relative velocity removed per substep)")
        planes = np.asarray(self.planes, np.float64).reshape(-1, 4) if len(self.planes) else np.zeros((0, 4))
        if not np.isfinite(planes).all() or (len(planes) and np.any(np.abs(np.linalg.norm(planes[:, :3], axis=1)-1)
                                                                    > 1e-9)):
            raise ValueError("planes need finite unit normals and offsets")
        counts = {len(f.boxes) for f in self.frames}
        if len(counts) > 1:
            raise ValueError("every contact frame needs the same number of boxes (uniform B)")
        for f in self.frames:
            if f.kind not in FRAME_KINDS:
                raise ValueError(f"frame kind {f.kind!r} not in {tuple(FRAME_KINDS)}")
            boxes = np.asarray(f.boxes, np.float64).reshape(-1, 2, 3)
            if not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
                raise ValueError("contact boxes need finite centres and positive half extents")
            if len(boxes) == 0:
                raise ValueError("a contact frame needs at least one box")

    @property
    def P(self):
        return len(self.planes)

    @property
    def F(self):
        return len(self.frames)

    @property
    def B(self):
        return len(self.frames[0].boxes) if self.frames else 0

    def plane_array(self):
        return np.asarray(self.planes, np.float64).reshape(-1, 4) if self.P else np.zeros((0, 4))

    def box_arrays(self):
        """([F*B, 3] centres, [F*B, 3] half extents) float64, frame-major."""
        if not self.F:
            return np.zeros((0, 3)), np.zeros((0, 3))
        boxes = np.concatenate([np.asarray(f.boxes, np.float64).reshape(-1, 2, 3) for f in self.frames])
        return boxes[:, 0].copy(), boxes[:, 1].copy()

    def receipt(self):
        rule = ("c = 2 zeta sqrt(k m_I) on the node's predicted (post-force) relative velocity; "
                "tangential inside the Coulomb cap, normal with F_n >= 0") if self.law == PENALTY_LAW else (
                "c = 2 zeta sqrt(k m_I) on the node's realised relative velocity (the drift velocity B pi_n = old after "
                "the P2G grid update, minus the obstacle velocity; ADR 0023 §22.9); tangential inside the Coulomb cap, "
                "normal with F_n >= 0")
        params = ({"normal_stiffness": float(self.normal_stiffness),
                   "tangential_stiffness": float(self.tangential_stiffness),
                   "damping_ratio": float(self.damping_ratio),
                   "normal_damping_ratio": float(self.normal_damping_ratio),
                   "damping_rule": rule,
                   "stiffness_rule": "k = kappa m_I / dt^2 per node (uniform stiffness per area on the faces)"}
                  if self.law in PENALTY_LAWS else {"penetration_recovery": float(self.penetration_recovery)})
        return {"law": self.law, "work": self.work, **params,
                "planes": self.plane_array().tolist(),
                "frames": [{"name": f.name, "kind": f.kind, "origin": list(map(float, f.origin)),
                            "boxes": np.asarray(f.boxes, float).reshape(-1, 2, 3).tolist()} for f in self.frames],
                "P": self.P, "F": self.F, "B": self.B, "adr": ADR}


# ----------------------------------------------------------------------------- Warp functions

@wp.func
def box_sdf_d(p: wp.vec3d, half: wp.vec3d):
    """float64 twin of the frozen ``tlmpm_v5._box_sdf`` (same branches): outward normal and signed distance."""
    zero = wp.float64(0.0)
    q = wp.vec3d(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2]))-half
    outside = wp.vec3d(wp.max(q[0], zero), wp.max(q[1], zero), wp.max(q[2], zero))
    distance = wp.length(outside)
    sdf = distance+wp.min(wp.max(q[0], wp.max(q[1], q[2])), zero)
    normal = wp.vec3d()
    if distance > wp.float64(1.0e-15):
        normal = wp.vec3d(wp.sign(p[0])*outside[0], wp.sign(p[1])*outside[1], wp.sign(p[2])*outside[2])/distance
    elif q[0] >= q[1] and q[0] >= q[2]:
        normal = wp.vec3d(wp.sign(p[0]), zero, zero)
    elif q[1] >= q[2]:
        normal = wp.vec3d(zero, wp.sign(p[1]), zero)
    else:
        normal = wp.vec3d(zero, zero, wp.sign(p[2]))
    return normal, sdf


@wp.func
def coulomb_projection(v: wp.vec3d, v_obs: wp.vec3d, n: wp.vec3d, gap: wp.float64, dt: wp.float64,
                       beta: wp.float64, mu: wp.float64):
    """Velocity change (dv_n >= 0 along n, dv_t) of one node against one obstacle, and its state (0 none, 1 stick,
    2 slip). See the module docstring."""
    zero = wp.float64(0.0)
    vrel = v-v_obs
    vn = wp.dot(vrel, n)
    vmin = -gap/dt
    if gap < zero:
        vmin = -beta*gap/dt
    dvn = zero
    dvt = wp.vec3d()
    state = int(0)
    if vn < vmin:
        dvn = vmin-vn
        vt = vrel-vn*n
        st = wp.length(vt)
        if st <= mu*dvn:
            dvt = -vt
            state = STATE_STICK
        else:
            dvt = -(mu*dvn/st)*vt
            state = STATE_SLIP
    return dvn, dvt, state


@wp.func
def frame_rotation(q0: wp.quatd, q1: wp.quatd, s: wp.float64):
    return wp.quat_to_matrix(wp.quat_slerp(q0, q1, s))


# ----------------------------------------------------------------------------- kernels (shared single / batched)

@wp.kernel
def node_contact(gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
                 node_X: wp.array(dtype=wp.vec3d), node_u: wp.array(dtype=wp.vec3d),
                 node_group: wp.array(dtype=int), group_active: wp.array(dtype=int),
                 env_of_node: wp.array(dtype=int), fric: wp.array2d(dtype=float), col_table: int, col_pad: int,
                 P: int, planes: wp.array(dtype=wp.vec4d), F: int, B: int, frame_kind: wp.array(dtype=int),
                 frame_active: wp.array(dtype=int), fq0: wp.array(dtype=wp.quatd), fq1: wp.array(dtype=wp.quatd),
                 fb0: wp.array(dtype=wp.vec3d), fb1: wp.array(dtype=wp.vec3d), forigin: wp.array(dtype=wp.vec3d),
                 box_c: wp.array(dtype=wp.vec3d), box_h: wp.array(dtype=wp.vec3d), index: wp.array(dtype=int),
                 substeps: int, dt: wp.float64, beta: wp.float64, jn_out: wp.array(dtype=wp.vec3d),
                 jt_out: wp.array(dtype=wp.vec3d), plane_reaction: wp.array(dtype=wp.spatial_vectord),
                 box_reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array2d(dtype=wp.float64),
                 pen_max: wp.array(dtype=wp.float64), counts: wp.array2d(dtype=int)):
    """``grid_node_coulomb_v1`` (module docstring): one thread per node; obstacles in fixed order; writes the node's
    velocity, its normal / tangential impulse (float64, zero when free), the obstacle reactions (impulse ON the
    material and its torque: planes about the world origin, boxes about the frame's current origin), the first half of
    the kick-paired work, the obstacle power, the max penetration and the event counts of its env."""
    i = wp.tid()
    zero3 = wp.vec3d()
    jn_out[i] = zero3
    jt_out[i] = zero3
    m = gm[i]
    if m <= 1.0e-15:
        return
    grp = node_group[i]
    if grp >= 0:
        if group_active[grp] != 0:
            return
    e = env_of_node[i]
    x = node_X[i]+node_u[i]
    v = wp.vec3d(new_v[i])
    md = wp.float64(m)
    Jn = wp.vec3d()
    Jt = wp.vec3d()
    hit = int(0)
    mu_table = wp.float64(fric[e, col_table])
    mu_pad = wp.float64(fric[e, col_pad])
    zero = wp.float64(0.0)
    for k in range(P):
        pl = planes[e*P+k]
        n = wp.vec3d(pl[0], pl[1], pl[2])
        gap = wp.dot(n, x)+pl[3]
        if gap < zero:
            wp.atomic_max(pen_max, e, -gap)
        dvn, dvt, state = coulomb_projection(v, zero3, n, gap, dt, beta, mu_table)
        if state != 0:
            hit = 1
            jn = (md*dvn)*n
            jt = md*dvt
            v = v+dvn*n+dvt
            Jn += jn
            Jt += jt
            jj = jn+jt
            tau = wp.cross(x, jj)
            wp.atomic_add(plane_reaction, e*P+k, wp.spatial_vectord(jj[0], jj[1], jj[2], tau[0], tau[1], tau[2]))
            wp.atomic_add(counts, e, 0, 1)
            wp.atomic_add(counts, e, state, 1)
    if F > 0:
        s0 = wp.float64(index[0])/wp.float64(substeps)
        s1 = wp.float64(index[0]+1)/wp.float64(substeps)
        for f in range(F):
            r = e*F+f
            if frame_active[r] != 0:
                R0 = frame_rotation(fq0[r], fq1[r], s0)
                c0 = fb0[r]+s0*(fb1[r]-fb0[r])
                R1 = frame_rotation(fq0[r], fq1[r], s1)
                c1 = fb0[r]+s1*(fb1[r]-fb0[r])
                Xh = wp.transpose(R0)*(x-c0)
                v_obs = (R1*Xh+c1-x)/dt
                o1 = R1*forigin[r]+c1
                mu = mu_table
                if frame_kind[r] == FRAME_PAD:
                    mu = mu_pad
                for b in range(B):
                    rb = r*B+b
                    nl, gap = box_sdf_d(Xh-box_c[rb], box_h[rb])
                    if gap < zero:
                        wp.atomic_max(pen_max, e, -gap)
                    n = R0*nl
                    dvn, dvt, state = coulomb_projection(v, v_obs, n, gap, dt, beta, mu)
                    if state != 0:
                        hit = 1
                        jn = (md*dvn)*n
                        jt = md*dvt
                        v = v+dvn*n+dvt
                        Jn += jn
                        Jt += jt
                        jj = jn+jt
                        tau = wp.cross(x-o1, jj)
                        wp.atomic_add(box_reaction, rb, wp.spatial_vectord(jj[0], jj[1], jj[2], tau[0], tau[1],
                                                                           tau[2]))
                        wp.atomic_add(work, e, 2, wp.dot(jn, v_obs))
                        wp.atomic_add(work, e, 3, wp.dot(jt, v_obs))
                        wp.atomic_add(counts, e, 0, 1)
                        wp.atomic_add(counts, e, state, 1)
    if hit != 0:
        new_v[i] = wp.vec3(v)
        jn_out[i] = Jn
        jt_out[i] = Jt
        o = wp.vec3d(old[i])
        half = wp.float64(0.5)
        wp.atomic_add(work, e, 0, half*wp.dot(Jn, o))
        wp.atomic_add(work, e, 1, half*wp.dot(Jt, o))


@wp.func
def plane_hit(pl: wp.vec4d, x: wp.vec3d):
    n = wp.vec3d(pl[0], pl[1], pl[2])
    return n, wp.dot(n, x)+pl[3]


@wp.kernel
def node_penalty_contact(gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
                         node_X: wp.array(dtype=wp.vec3d), node_u: wp.array(dtype=wp.vec3d),
                         node_group: wp.array(dtype=int), group_active: wp.array(dtype=int),
                         env_of_node: wp.array(dtype=int), fric: wp.array2d(dtype=float), col_table: int,
                         col_pad: int, P: int, planes: wp.array(dtype=wp.vec4d), F: int, B: int,
                         frame_kind: wp.array(dtype=int), frame_active: wp.array(dtype=int),
                         fq0: wp.array(dtype=wp.quatd), fq1: wp.array(dtype=wp.quatd), fb0: wp.array(dtype=wp.vec3d),
                         fb1: wp.array(dtype=wp.vec3d), forigin: wp.array(dtype=wp.vec3d),
                         box_c: wp.array(dtype=wp.vec3d), box_h: wp.array(dtype=wp.vec3d), index: wp.array(dtype=int),
                         substeps: int, dt: wp.float64, kn_frac: wp.float64, kt_frac: wp.float64,
                         zeta: wp.float64, zeta_n: wp.float64, xi: wp.array(dtype=wp.vec3d),
                         obst: wp.array(dtype=int),
                         vobs: wp.array(dtype=wp.vec3d), jn_out: wp.array(dtype=wp.vec3d),
                         jt_out: wp.array(dtype=wp.vec3d),
                         plane_reaction: wp.array(dtype=wp.spatial_vectord),
                         box_reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array2d(dtype=wp.float64),
                         pen_max: wp.array(dtype=wp.float64), counts: wp.array2d(dtype=int)):
    """``grid_node_penalty_coulomb_v1`` (one thread per node, obstacles in fixed order; ADR 0023 §22.1).

    Normal: every obstacle the node penetrates (gap < 0 at the start of the substep) pushes with the conservative
    penalty force k_n (-gap) n, k_n = kappa_n m_I / dt^2 (potential 1/2 k_n gap^2). Tangential: one Coulomb
    spring-slider per node on its deepest obstacle, elongation ``xi`` (world vector, the relative tangential
    displacement since the anchor; projected on the current tangent plane), force -k_t xi capped at mu |F_n| (slip:
    the anchor slides, xi is scaled back onto the cap; the Coulomb dissipation mu |F_n| (|xi_trial| - |xi_cap|) is
    booked). A node that leaves contact or changes obstacle releases its anchor. The impulse dt (F_n + F_t) enters the
    kick (P2G-pass grid velocity). ``vobs`` keeps the obstacle velocity at the node for the slider update after the
    drift (``slider_update``)."""
    i = wp.tid()
    zero3 = wp.vec3d()
    jn_out[i] = zero3
    jt_out[i] = zero3
    m = gm[i]
    if m <= 1.0e-15:
        xi[i] = zero3
        obst[i] = -1
        return
    grp = node_group[i]
    if grp >= 0:
        if group_active[grp] != 0:
            xi[i] = zero3
            obst[i] = -1
            return
    e = env_of_node[i]
    x = node_X[i]+node_u[i]
    md = wp.float64(m)
    kn = kn_frac*md/(dt*dt)
    kt = kt_frac*md/(dt*dt)
    two = wp.float64(2.0)
    cn = two*zeta_n*wp.sqrt(kn*md)
    ct = two*zeta*wp.sqrt(kt*md)
    vstar = wp.vec3d(new_v[i])
    zero = wp.float64(0.0)
    Jn = wp.vec3d()
    best = int(-1)
    best_pen = zero
    best_n = wp.vec3d()
    best_vo = wp.vec3d()
    best_fn = zero
    best_mu = zero
    best_lever = wp.vec3d()
    mu_table = wp.float64(fric[e, col_table])
    mu_pad = wp.float64(fric[e, col_pad])
    for k in range(P):
        n, gap = plane_hit(planes[e*P+k], x)
        if gap < zero:
            wp.atomic_max(pen_max, e, -gap)
            vn = wp.dot(vstar, n)
            fn = wp.max(kn*(-gap)-cn*vn, zero)
            if cn > zero:
                wp.atomic_add(work, e, 5, dt*wp.min(cn*vn*vn, fn*wp.abs(vn)))
            jn = (dt*fn)*n
            Jn += jn
            tau = wp.cross(x, jn)
            wp.atomic_add(plane_reaction, e*P+k, wp.spatial_vectord(jn[0], jn[1], jn[2], tau[0], tau[1], tau[2]))
            if -gap > best_pen:
                best = k
                best_pen = -gap
                best_n = n
                best_vo = zero3
                best_fn = fn
                best_mu = mu_table
                best_lever = x
    if F > 0:
        s0 = wp.float64(index[0])/wp.float64(substeps)
        s1 = wp.float64(index[0]+1)/wp.float64(substeps)
        for f in range(F):
            r = e*F+f
            if frame_active[r] != 0:
                R0 = frame_rotation(fq0[r], fq1[r], s0)
                c0 = fb0[r]+s0*(fb1[r]-fb0[r])
                R1 = frame_rotation(fq0[r], fq1[r], s1)
                c1 = fb0[r]+s1*(fb1[r]-fb0[r])
                Xh = wp.transpose(R0)*(x-c0)
                v_obs = (R1*Xh+c1-x)/dt
                o1 = R1*forigin[r]+c1
                mu = mu_table
                if frame_kind[r] == FRAME_PAD:
                    mu = mu_pad
                for b in range(B):
                    rb = r*B+b
                    nl, gap = box_sdf_d(Xh-box_c[rb], box_h[rb])
                    if gap < zero:
                        wp.atomic_max(pen_max, e, -gap)
                        n = R0*nl
                        vn = wp.dot(vstar-v_obs, n)
                        fn = wp.max(kn*(-gap)-cn*vn, zero)
                        if cn > zero:
                            wp.atomic_add(work, e, 5, dt*wp.min(cn*vn*vn, fn*wp.abs(vn)))
                        jn = (dt*fn)*n
                        Jn += jn
                        tau = wp.cross(x-o1, jn)
                        wp.atomic_add(box_reaction, rb, wp.spatial_vectord(jn[0], jn[1], jn[2], tau[0], tau[1],
                                                                           tau[2]))
                        wp.atomic_add(work, e, 2, wp.dot(jn, v_obs))
                        if -gap > best_pen:
                            best = P+rb-e*F*B
                            best_pen = -gap
                            best_n = n
                            best_vo = v_obs
                            best_fn = fn
                            best_mu = mu
                            best_lever = x-o1
    Jt = wp.vec3d()
    if best >= 0:
        xt = wp.vec3d()
        if obst[i] == best:
            xt = xi[i]
        xt = xt-wp.dot(xt, best_n)*best_n
        vr = vstar-best_vo
        vt = vr-wp.dot(vr, best_n)*best_n
        Ft = -kt*xt-ct*vt
        Lf = wp.length(Ft)
        capf = best_mu*best_fn
        state = STATE_STICK
        if Lf > capf:
            Ft_new = wp.vec3d()
            if Lf > zero:
                Ft_new = Ft*(capf/Lf)
            xt_new = Ft_new/(-kt)
            wp.atomic_add(work, e, 4, capf*wp.length(xt-xt_new))
            xt = xt_new
            Ft = Ft_new
            state = STATE_SLIP
        elif ct > zero:
            wp.atomic_add(work, e, 5, dt*ct*wp.dot(vt, vt))
        Jt = dt*Ft
        xi[i] = xt
        obst[i] = best
        vobs[i] = best_vo
        tau = wp.cross(best_lever, Jt)
        sv = wp.spatial_vectord(Jt[0], Jt[1], Jt[2], tau[0], tau[1], tau[2])
        if best < P:
            wp.atomic_add(plane_reaction, e*P+best, sv)
        else:
            wp.atomic_add(box_reaction, e*F*B+best-P, sv)
        wp.atomic_add(work, e, 3, wp.dot(Jt, best_vo))
        wp.atomic_add(counts, e, 0, 1)
        wp.atomic_add(counts, e, state, 1)
        v = wp.vec3d(new_v[i])+(Jn+Jt)/md
        new_v[i] = wp.vec3(v)
        jn_out[i] = Jn
        jt_out[i] = Jt
        o = wp.vec3d(old[i])
        half = wp.float64(0.5)
        wp.atomic_add(work, e, 0, half*wp.dot(Jn, o))
        wp.atomic_add(work, e, 1, half*wp.dot(Jt, o))
    else:
        xi[i] = zero3
        obst[i] = -1


@wp.kernel
def node_penalty_contact_v2(gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
                         node_X: wp.array(dtype=wp.vec3d), node_u: wp.array(dtype=wp.vec3d),
                         node_group: wp.array(dtype=int), group_active: wp.array(dtype=int),
                         env_of_node: wp.array(dtype=int), fric: wp.array2d(dtype=float), col_table: int,
                         col_pad: int, P: int, planes: wp.array(dtype=wp.vec4d), F: int, B: int,
                         frame_kind: wp.array(dtype=int), frame_active: wp.array(dtype=int),
                         fq0: wp.array(dtype=wp.quatd), fq1: wp.array(dtype=wp.quatd), fb0: wp.array(dtype=wp.vec3d),
                         fb1: wp.array(dtype=wp.vec3d), forigin: wp.array(dtype=wp.vec3d),
                         box_c: wp.array(dtype=wp.vec3d), box_h: wp.array(dtype=wp.vec3d), index: wp.array(dtype=int),
                         substeps: int, dt: wp.float64, kn_frac: wp.float64, kt_frac: wp.float64,
                         zeta: wp.float64, zeta_n: wp.float64, xi: wp.array(dtype=wp.vec3d),
                         obst: wp.array(dtype=int),
                         vobs: wp.array(dtype=wp.vec3d), jn_out: wp.array(dtype=wp.vec3d),
                         jt_out: wp.array(dtype=wp.vec3d),
                         plane_reaction: wp.array(dtype=wp.spatial_vectord),
                         box_reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array2d(dtype=wp.float64),
                         pen_max: wp.array(dtype=wp.float64), counts: wp.array2d(dtype=int)):
    """``grid_node_penalty_coulomb_v2`` (ADR 0023 §22.9): ``node_penalty_contact`` (v1, unchanged above) with ONE change:
    both dashpots and the tangential trial act on the node's REALISED velocity g_n = ``old`` after the P2G grid update
    (= B pi_n, the drift velocity with which the node moved in the previous substep, i.e. the realised rate of its
    gap), relative to the obstacle velocity, instead of the predicted lumped velocity v* = ``new_v`` (= M_l^-1 pi_n +
    dt f / m_I). The impulse still enters the kick (``new_v`` += J / m_I) and the work is the same kick-paired
    ``kick_average_v1``. Why (X1 diagnostics, ``scripts/rubber/p2_4/diag_contact_x1.py``): under the TLMPM drift
    metric B the v1 dashpot is not dissipative for the face's grid-scale modes -- a frictionless elastic block at rest
    keeps a self-excited contact rattle (27 % of the bottom nodes in contact per substep, vibration energy ~100x the
    static elastic energy, net POSITIVE normal contact work) -- while with g_n it settles (all nodes in contact).
    At a static fixed point g_n = 0, so the spring alone carries the load: delta_I = F_I / k_n.

    Normal: every obstacle the node penetrates (gap < 0 at the start of the substep) pushes with the conservative
    penalty force k_n (-gap) n, k_n = kappa_n m_I / dt^2 (potential 1/2 k_n gap^2). Tangential: one Coulomb
    spring-slider per node on its deepest obstacle, elongation ``xi`` (world vector, the relative tangential
    displacement since the anchor; projected on the current tangent plane), force -k_t xi capped at mu |F_n| (slip:
    the anchor slides, xi is scaled back onto the cap; the Coulomb dissipation mu |F_n| (|xi_trial| - |xi_cap|) is
    booked). A node that leaves contact or changes obstacle releases its anchor. The impulse dt (F_n + F_t) enters the
    kick (P2G-pass grid velocity). ``vobs`` keeps the obstacle velocity at the node for the slider update after the
    drift (``slider_update``)."""
    i = wp.tid()
    zero3 = wp.vec3d()
    jn_out[i] = zero3
    jt_out[i] = zero3
    m = gm[i]
    if m <= 1.0e-15:
        xi[i] = zero3
        obst[i] = -1
        return
    grp = node_group[i]
    if grp >= 0:
        if group_active[grp] != 0:
            xi[i] = zero3
            obst[i] = -1
            return
    e = env_of_node[i]
    x = node_X[i]+node_u[i]
    md = wp.float64(m)
    kn = kn_frac*md/(dt*dt)
    kt = kt_frac*md/(dt*dt)
    two = wp.float64(2.0)
    cn = two*zeta_n*wp.sqrt(kn*md)
    ct = two*zeta*wp.sqrt(kt*md)
    vstar = wp.vec3d(old[i])                     # v2: the realised (drift) velocity, not the predicted new_v
    zero = wp.float64(0.0)
    Jn = wp.vec3d()
    best = int(-1)
    best_pen = zero
    best_n = wp.vec3d()
    best_vo = wp.vec3d()
    best_fn = zero
    best_mu = zero
    best_lever = wp.vec3d()
    mu_table = wp.float64(fric[e, col_table])
    mu_pad = wp.float64(fric[e, col_pad])
    for k in range(P):
        n, gap = plane_hit(planes[e*P+k], x)
        if gap < zero:
            wp.atomic_max(pen_max, e, -gap)
            vn = wp.dot(vstar, n)
            fn = wp.max(kn*(-gap)-cn*vn, zero)
            if cn > zero:
                wp.atomic_add(work, e, 5, dt*wp.min(cn*vn*vn, fn*wp.abs(vn)))
            jn = (dt*fn)*n
            Jn += jn
            tau = wp.cross(x, jn)
            wp.atomic_add(plane_reaction, e*P+k, wp.spatial_vectord(jn[0], jn[1], jn[2], tau[0], tau[1], tau[2]))
            if -gap > best_pen:
                best = k
                best_pen = -gap
                best_n = n
                best_vo = zero3
                best_fn = fn
                best_mu = mu_table
                best_lever = x
    if F > 0:
        s0 = wp.float64(index[0])/wp.float64(substeps)
        s1 = wp.float64(index[0]+1)/wp.float64(substeps)
        for f in range(F):
            r = e*F+f
            if frame_active[r] != 0:
                R0 = frame_rotation(fq0[r], fq1[r], s0)
                c0 = fb0[r]+s0*(fb1[r]-fb0[r])
                R1 = frame_rotation(fq0[r], fq1[r], s1)
                c1 = fb0[r]+s1*(fb1[r]-fb0[r])
                Xh = wp.transpose(R0)*(x-c0)
                v_obs = (R1*Xh+c1-x)/dt
                o1 = R1*forigin[r]+c1
                mu = mu_table
                if frame_kind[r] == FRAME_PAD:
                    mu = mu_pad
                for b in range(B):
                    rb = r*B+b
                    nl, gap = box_sdf_d(Xh-box_c[rb], box_h[rb])
                    if gap < zero:
                        wp.atomic_max(pen_max, e, -gap)
                        n = R0*nl
                        vn = wp.dot(vstar-v_obs, n)
                        fn = wp.max(kn*(-gap)-cn*vn, zero)
                        if cn > zero:
                            wp.atomic_add(work, e, 5, dt*wp.min(cn*vn*vn, fn*wp.abs(vn)))
                        jn = (dt*fn)*n
                        Jn += jn
                        tau = wp.cross(x-o1, jn)
                        wp.atomic_add(box_reaction, rb, wp.spatial_vectord(jn[0], jn[1], jn[2], tau[0], tau[1],
                                                                           tau[2]))
                        wp.atomic_add(work, e, 2, wp.dot(jn, v_obs))
                        if -gap > best_pen:
                            best = P+rb-e*F*B
                            best_pen = -gap
                            best_n = n
                            best_vo = v_obs
                            best_fn = fn
                            best_mu = mu
                            best_lever = x-o1
    Jt = wp.vec3d()
    if best >= 0:
        xt = wp.vec3d()
        if obst[i] == best:
            xt = xi[i]
        xt = xt-wp.dot(xt, best_n)*best_n
        vr = vstar-best_vo
        vt = vr-wp.dot(vr, best_n)*best_n
        Ft = -kt*xt-ct*vt
        Lf = wp.length(Ft)
        capf = best_mu*best_fn
        state = STATE_STICK
        if Lf > capf:
            Ft_new = wp.vec3d()
            if Lf > zero:
                Ft_new = Ft*(capf/Lf)
            xt_new = Ft_new/(-kt)
            wp.atomic_add(work, e, 4, capf*wp.length(xt-xt_new))
            xt = xt_new
            Ft = Ft_new
            state = STATE_SLIP
        elif ct > zero:
            wp.atomic_add(work, e, 5, dt*ct*wp.dot(vt, vt))
        Jt = dt*Ft
        xi[i] = xt
        obst[i] = best
        vobs[i] = best_vo
        tau = wp.cross(best_lever, Jt)
        sv = wp.spatial_vectord(Jt[0], Jt[1], Jt[2], tau[0], tau[1], tau[2])
        if best < P:
            wp.atomic_add(plane_reaction, e*P+best, sv)
        else:
            wp.atomic_add(box_reaction, e*F*B+best-P, sv)
        wp.atomic_add(work, e, 3, wp.dot(Jt, best_vo))
        wp.atomic_add(counts, e, 0, 1)
        wp.atomic_add(counts, e, state, 1)
        v = wp.vec3d(new_v[i])+(Jn+Jt)/md
        new_v[i] = wp.vec3(v)
        jn_out[i] = Jn
        jt_out[i] = Jt
        o = wp.vec3d(old[i])
        half = wp.float64(0.5)
        wp.atomic_add(work, e, 0, half*wp.dot(Jn, o))
        wp.atomic_add(work, e, 1, half*wp.dot(Jt, o))
    else:
        xi[i] = zero3
        obst[i] = -1


@wp.kernel
def slider_update(xi: wp.array(dtype=wp.vec3d), obst: wp.array(dtype=int), vobs: wp.array(dtype=wp.vec3d),
                  new_v: wp.array(dtype=wp.vec3), dt: wp.float64):
    """Spring-slider elongation after the drift: xi += dt (v_I - v_obs) for nodes in contact (projected onto the
    tangent plane at the next kick)."""
    i = wp.tid()
    if obst[i] >= 0:
        xi[i] = xi[i]+dt*(wp.vec3d(new_v[i])-vobs[i])


@wp.kernel
def contact_energy(node_X: wp.array(dtype=wp.vec3d), node_u: wp.array(dtype=wp.vec3d), gm: wp.array(dtype=float),
                   env_of_node: wp.array(dtype=int), fric: wp.array2d(dtype=float), col_table: int, col_pad: int,
                   P: int, planes: wp.array(dtype=wp.vec4d), F: int, B: int, frame_kind: wp.array(dtype=int),
                   frame_active: wp.array(dtype=int), fq1: wp.array(dtype=wp.quatd), fb1: wp.array(dtype=wp.vec3d),
                   box_c: wp.array(dtype=wp.vec3d), box_h: wp.array(dtype=wp.vec3d), dt: wp.float64,
                   kn_frac: wp.float64, kt_frac: wp.float64, xi: wp.array(dtype=wp.vec3d), obst: wp.array(dtype=int),
                   node_offset: wp.array(dtype=int), out: wp.array2d(dtype=wp.float64)):
    """Penalty-law stored contact energy of env e at the current state (frames at their end-of-step pose; one thread
    per env, fixed order): Phi_n = sum 1/2 k_n max(-gap, 0)^2 over penetrated obstacles, Phi_t = sum 1/2 k_t |xi_t|^2
    with xi_t the slider elongation projected on the deepest obstacle's tangent plane and capped at mu F_n / k_t
    (the value the next kick would use)."""
    e = wp.tid()
    zero = wp.float64(0.0)
    phin = zero
    phit = zero
    mu_table = wp.float64(fric[e, col_table])
    mu_pad = wp.float64(fric[e, col_pad])
    for i in range(node_offset[e], node_offset[e+1]):
        m = gm[i]
        if m > 1.0e-15:
            md = wp.float64(m)
            kn = kn_frac*md/(dt*dt)
            kt = kt_frac*md/(dt*dt)
            x = node_X[i]+node_u[i]
            best = int(-1)
            best_pen = zero
            best_n = wp.vec3d()
            best_cap = zero
            for k in range(P):
                n, gap = plane_hit(planes[e*P+k], x)
                if gap < zero:
                    phin += wp.float64(0.5)*kn*gap*gap
                    if -gap > best_pen:
                        best = k
                        best_pen = -gap
                        best_n = n
                        best_cap = mu_table*kn*(-gap)/kt
            for f in range(F):
                r = e*F+f
                if frame_active[r] != 0:
                    R1 = wp.quat_to_matrix(fq1[r])
                    Xh = wp.transpose(R1)*(x-fb1[r])
                    mu = mu_table
                    if frame_kind[r] == FRAME_PAD:
                        mu = mu_pad
                    for b in range(B):
                        rb = r*B+b
                        nl, gap = box_sdf_d(Xh-box_c[rb], box_h[rb])
                        if gap < zero:
                            phin += wp.float64(0.5)*kn*gap*gap
                            if -gap > best_pen:
                                best = P+rb-e*F*B
                                best_pen = -gap
                                best_n = R1*nl
                                best_cap = mu*kn*(-gap)/kt
            if best >= 0 and obst[i] == best:
                xt = xi[i]
                xt = xt-wp.dot(xt, best_n)*best_n
                L = wp.min(wp.length(xt), best_cap)
                phit += wp.float64(0.5)*kt*L*L
    out[e, 0] = phin
    out[e, 1] = phit


@wp.kernel
def contact_work_end(jn: wp.array(dtype=wp.vec3d), jt: wp.array(dtype=wp.vec3d), old: wp.array(dtype=wp.vec3),
                     env_of_node: wp.array(dtype=int), work: wp.array2d(dtype=wp.float64)):
    """Second half of the kick-paired contact work: J . g_{n+1} / 2 with g_{n+1} = ``old`` after the (final) MUSL grid
    update (= B pi_{n+1}); launched before the MUSL-pass BC."""
    i = wp.tid()
    a = jn[i]
    b = jt[i]
    zero = wp.float64(0.0)
    if a[0] != zero or a[1] != zero or a[2] != zero or b[0] != zero or b[1] != zero or b[2] != zero:
        o = wp.vec3d(old[i])
        half = wp.float64(0.5)
        e = env_of_node[i]
        wp.atomic_add(work, e, 0, half*wp.dot(a, o))
        wp.atomic_add(work, e, 1, half*wp.dot(b, o))


@wp.kernel
def node_displacement(node_u: wp.array(dtype=wp.vec3d), new_v: wp.array(dtype=wp.vec3), dt: wp.float64):
    """u_I += dt v_I with the drift velocity (the velocity that moves F and x this substep), float64."""
    i = wp.tid()
    node_u[i] = node_u[i]+dt*wp.vec3d(new_v[i])


# ----------------------------------------------------------------------------- host reference (tests)

def host_box_sdf(p, half):
    p, half = np.asarray(p, np.float64), np.asarray(half, np.float64)
    q = np.abs(p)-half
    outside = np.maximum(q, 0.0)
    distance = float(np.linalg.norm(outside))
    sdf = distance+min(float(q.max()), 0.0)
    sign = np.where(p < 0, -1.0, 1.0)
    if distance > 1e-15:
        return sign*outside/distance, sdf
    k = 0 if (q[0] >= q[1] and q[0] >= q[2]) else (1 if q[1] >= q[2] else 2)
    n = np.zeros(3)
    n[k] = sign[k]
    return n, sdf


def host_coulomb_projection(v, v_obs, n, gap, dt, beta, mu):
    """float64 reference of ``coulomb_projection``: (dv_n, dv_t, state)."""
    v, v_obs, n = (np.asarray(a, np.float64) for a in (v, v_obs, n))
    vrel = v-v_obs
    vn = float(vrel @ n)
    vmin = -gap/dt if gap >= 0 else -beta*gap/dt
    if not vn < vmin:
        return 0.0, np.zeros(3), 0
    dvn = vmin-vn
    vt = vrel-vn*n
    st = float(np.linalg.norm(vt))
    if st <= mu*dvn:
        return dvn, -vt, STATE_STICK
    return dvn, -(mu*dvn/st)*vt, STATE_SLIP


__all__ = ["ContactSpec", "ContactFrame", "node_contact", "node_penalty_contact", "node_penalty_contact_v2",
           "slider_update", "contact_energy", "contact_work_end", "node_displacement", "box_sdf_d",
           "coulomb_projection", "CONTACT_LAW", "CONTACT_LAWS", "PENALTY_LAW", "PENALTY_LAW_V2", "PENALTY_LAWS",
           "CONTACT_WORK", "WORK_COLUMNS", "COUNT_COLUMNS", "FRAME_KINDS",
           "host_box_sdf", "host_coulomb_projection"]
