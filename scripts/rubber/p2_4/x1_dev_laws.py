"""DIAGNOSTIC development variants of the node penalty contact law (task X1, ADR 0023 §22.8; NOT a declared law).

Planes only (the INC / rest scenes of ``diag_contact_x1.py``). ``dev_penalty_contact`` is the committed
``grid_node_penalty_coulomb_v1`` plane branch with switches:

- ``vmode``: velocity used by BOTH dashpots and by the tangential trial: 0 = the predicted lumped velocity v* (= v1),
  1 = g_n = ``old`` after the P2G grid update (the drift velocity B pi_n with which the node moved in the previous
  substep, i.e. the realised rate of the gap), 2 = (v* + g_n) / 2;
- ``capmode``: 0 = cap mu F_n (v1), 1 = cap mu F_spring (k_n (-gap) only; the dashpot excluded from the cone);
- ``dirmode``: 0 = slip direction of the trial force (v1), 1 = slip direction -v_t (velocity of ``vmode``).

``install(live, **modes)`` replaces the solver's ``_launch_contact`` (the rest of the substep is unchanged).
"""
import warp as wp

from robotic_cutting_sim import rubber_contact as rct

wp.set_module_options({"enable_backward": False})


@wp.kernel
def dev_penalty_contact(gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
                        node_X: wp.array(dtype=wp.vec3d), node_u: wp.array(dtype=wp.vec3d),
                        node_group: wp.array(dtype=int), group_active: wp.array(dtype=int),
                        fric: wp.array2d(dtype=float), P: int, planes: wp.array(dtype=wp.vec4d), dt: wp.float64,
                        kn_frac: wp.float64, kt_frac: wp.float64, zeta: wp.float64, zeta_n: wp.float64,
                        vmode: int, capmode: int, dirmode: int, alpha: wp.float64,
                        nf: wp.array(dtype=wp.float64), vf: wp.array(dtype=wp.vec3d),
                        xi: wp.array(dtype=wp.vec3d), obst: wp.array(dtype=int), vobs: wp.array(dtype=wp.vec3d),
                        jn_out: wp.array(dtype=wp.vec3d), jt_out: wp.array(dtype=wp.vec3d),
                        plane_reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array2d(dtype=wp.float64),
                        pen_max: wp.array(dtype=wp.float64), counts: wp.array2d(dtype=int)):
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
    e = 0
    x = node_X[i]+node_u[i]
    md = wp.float64(m)
    kn = kn_frac*md/(dt*dt)
    kt = kt_frac*md/(dt*dt)
    two = wp.float64(2.0)
    cn = two*zeta_n*wp.sqrt(kn*md)
    ct = two*zeta*wp.sqrt(kt*md)
    vpred = wp.vec3d(new_v[i])
    vstar = vpred
    if vmode == 1:
        vstar = wp.vec3d(old[i])
    elif vmode == 2:
        vstar = wp.float64(0.5)*(vpred+wp.vec3d(old[i]))
    zero = wp.float64(0.0)
    Jn = wp.vec3d()
    best = int(-1)
    best_pen = zero
    best_n = wp.vec3d()
    best_fn = zero
    best_lever = wp.vec3d()
    mu_table = wp.float64(fric[e, 0])
    for k in range(P):
        pl = planes[k]
        n = wp.vec3d(pl[0], pl[1], pl[2])
        gap = wp.dot(n, x)+pl[3]
        if gap < zero:
            wp.atomic_max(pen_max, e, -gap)
            vn = wp.dot(vstar, n)
            fn = wp.max(kn*(-gap)-cn*vn, zero)
            if cn > zero:
                wp.atomic_add(work, e, 5, dt*wp.min(cn*vn*vn, fn*wp.abs(vn)))
            jn = (dt*fn)*n
            Jn += jn
            tau = wp.cross(x, jn)
            wp.atomic_add(plane_reaction, k, wp.spatial_vectord(jn[0], jn[1], jn[2], tau[0], tau[1], tau[2]))
            if -gap > best_pen:
                best = k
                best_pen = -gap
                best_n = n
                best_fn = fn
                if capmode == 1:
                    best_fn = kn*(-gap)
                best_lever = x
    # filtered normal force (capmode 2: every substep incl. zeros; capmode 3: contact substeps only, initialised on
    # first contact) and filtered velocity (dirmode 2)
    fn_now = zero
    if best >= 0:
        fn_now = best_fn
    nfo = nf[i]
    if capmode == 2 or (capmode >= 4 and capmode <= 6):
        nfo = nfo+alpha*(fn_now-nfo)
        nf[i] = nfo
    if capmode == 7:
        nfo = nfo+alpha*(fn_now-nfo)
    elif capmode == 3:
        if best >= 0:
            if obst[i] != best:
                nfo = fn_now
            else:
                nfo = nfo+alpha*(fn_now-nfo)
            nf[i] = nfo
    vfo = vf[i]
    if dirmode == 2:
        vfo = vfo+alpha*(vstar-vfo)
        vf[i] = vfo
    fn_inst = best_fn
    if capmode == 2 or capmode == 3:
        best_fn = nfo
    if capmode == 6:
        best_fn = wp.min(fn_inst, nfo)
    Jt = wp.vec3d()
    if best >= 0:
        xt = wp.vec3d()
        if obst[i] == best:
            xt = xi[i]
        xt = xt-wp.dot(xt, best_n)*best_n
        vt = vstar-wp.dot(vstar, best_n)*best_n
        Ft = -kt*xt-ct*vt
        Lf = wp.length(Ft)
        capf = mu_table*best_fn
        state = rct.STATE_STICK
        if Lf > capf:
            if capmode == 4 or capmode == 7:
                capf = wp.min(mu_table*nfo, Lf)
            elif capmode == 5:
                capf = mu_table*wp.min(fn_inst, nfo)
            Ft_new = wp.vec3d()
            vtf = vfo-wp.dot(vfo, best_n)*best_n
            if dirmode == 1 and wp.length(vt) > zero:
                Ft_new = -vt*(capf/wp.length(vt))
            elif dirmode == 2 and wp.length(vtf) > zero:
                Ft_new = -vtf*(capf/wp.length(vtf))
            elif Lf > zero:
                Ft_new = Ft*(capf/Lf)
            xt_new = Ft_new/(-kt)
            wp.atomic_add(work, e, 4, capf*wp.length(xt-xt_new))
            xt = xt_new
            Ft = Ft_new
            state = rct.STATE_SLIP
        elif ct > zero:
            wp.atomic_add(work, e, 5, dt*ct*wp.dot(vt, vt))
        if capmode == 7:
            if state == rct.STATE_STICK:
                nfo = fn_inst
            nf[i] = nfo
        Jt = dt*Ft
        xi[i] = xt
        obst[i] = best
        vobs[i] = zero3
        tau = wp.cross(best_lever, Jt)
        wp.atomic_add(plane_reaction, best, wp.spatial_vectord(Jt[0], Jt[1], Jt[2], tau[0], tau[1], tau[2]))
        wp.atomic_add(counts, e, 0, 1)
        wp.atomic_add(counts, e, state, 1)
        v = vpred+(Jn+Jt)/md
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
        if capmode == 7:
            nf[i] = nfo


def install(live, vmode=0, capmode=0, dirmode=0, ntau=50.0):
    s = live.solver
    if s.ct_F:
        raise ValueError("dev law: planes only")
    s.dev_nf = wp.zeros(s.node_count, dtype=wp.float64, device=s.device)
    s.dev_vf = wp.zeros(s.node_count, dtype=wp.vec3d, device=s.device)

    def launch():
        wp.launch(dev_penalty_contact, s.node_count,
                  [s.gm, s.new_v, s.old, s.node_X, s.ct_node_u, s.node_group, s.group_active, s.ct_fric, s.ct_P,
                   s.ct_planes, wp.float64(s.dt), wp.float64(s.ct_kn), wp.float64(s.ct_kt), wp.float64(s.ct_zeta),
                   wp.float64(s.ct_zeta_n), int(vmode), int(capmode), int(dirmode),
                   wp.float64(1.0/float(ntau)), s.dev_nf, s.dev_vf, s.ct_xi, s.ct_obst, s.ct_vobs,
                   s.ct_jn, s.ct_jt, s.ct_plane_reaction, s.ct_work, s.ct_pen_max, s.ct_counts], device=s.device)
    s._launch_contact = launch
