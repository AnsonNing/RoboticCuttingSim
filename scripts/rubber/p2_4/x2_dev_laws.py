"""DIAGNOSTIC development variants for task X2 (ADR 0023 §22.10, defect 2 of §22.9: friction-normal flutter); NOT a
declared law and not importable from ``src``. Raw outputs: ``artifacts/rl_ready/p2_4/x2_diag_001`` (README).

Planes only, single env (the INC scenes of ``diag_contact_x2.py``). The launches replace the solver's
``_launch_contact``: ``x2_normal`` = the ``grid_node_penalty_coulomb_v2`` normal branch (every penetrated plane,
dashpot on the realised velocity g_n), storing per node the normal force on its deepest plane ``fnb`` and that plane's
id ``bst``; optional ``x2_share`` / ``x2_patch_sums`` / ``x2_patch_lag`` / ``x2_gate``; ``x2_tangential`` = the v2
spring-dashpot-slider (trial on g_n) with the Coulomb cap mu * capn of mode ``smode``:

    smode 0: F_n,i                                     (= v2; reproduces the src v2 kernel to the last digit)
    smode 1: sum_j w_ij F_n,j / Z_j, Z_j = sum_i w_ij  (nonlocal cap, Oden-Pires type: reference-lattice neighbours
             |i - j|_inf <= radius in contact with the SAME plane, tensor hat weights; conservative: sum_i cap_i =
             mu sum_j F_n,j)
    smode 2: max(F_n,i, F_lag,i), F_lag relaxing to F_n (0 out of contact) with time constant tau (cap never below the
             instantaneous Coulomb cap)
    smode 3: as 2 with F_lag only in contact (initialised to F_n at first contact, cleared on release)
    smode 4: F_lag (symmetric first-order lag, time constant tau, initialised to F_n at (re)contact; X1 capmode 3
             analogue)
    smode 5: F_n,i until the node has slipped a distance ds without a stick substep, then F_lag (gate per node)
    smode 6: (sum_j F_n,j) m_i / sum_j m_j over every node in contact with the same plane (patch-level cap)
    smode 7: smode 4 only while the node's patch macro-slides (gate from the previous substep's patch slip speed
             |sum m v_t| / sum m: open >= vg, close < vg / 2), F_n,i otherwise
    smode 8: F_lag of the node's normal force INCLUDING zeros while out of contact (memory kept across bounces)
    smode 9: smode 8 gated by the patch gate of smode 7 (closed: F_lag = F_n)
    smode 10: L_tau(sum_j F_n,j) m_i / sum_j m_j (patch-level cap on the lagged patch normal force)
    smode 11: F_n,i (= v2 cap) plus, in slip only and outside the cap, a slip-fluctuation damper
             -2 zd sqrt(k_t m_i) (v_t,i - vbar_t) about the patch mean realised tangential velocity of the previous
             substep (zero net force for a uniformly sliding patch; booked as contact damping)
    smode 13: smode 12 with the mean over the nodes whose v2 trial slips in this substep (zero net damper force over
             the slipping set, so the patch-total friction never exceeds mu sum F_n)
    smode 12: smode 11 with the patch mean of the SAME substep (sum m v_t / sum m over the nodes in contact with the
             plane, realised velocities, computed before the tangential pass)

``switch_t``: the law is smode 0 (= v2) until t >= switch_t, then ``smode`` (recovery from a developed flutter).
``install(live, smode, radius, tau, ds, vg, switch_t, zd)`` replaces the solver's ``_launch_contact`` (the rest of the
substep unchanged). Every later mode was added without changing the code path of an earlier one.
"""
import warp as wp

from robotic_cutting_sim import rubber_contact as rct

wp.set_module_options({"enable_backward": False})


@wp.func
def hat_w(d: int, radius: int):
    return wp.float64(radius+1-abs(d))


@wp.kernel
def x2_normal(gm: wp.array(dtype=float), old: wp.array(dtype=wp.vec3), node_X: wp.array(dtype=wp.vec3d),
              node_u: wp.array(dtype=wp.vec3d), node_group: wp.array(dtype=int), group_active: wp.array(dtype=int),
              P: int, planes: wp.array(dtype=wp.vec4d), dt: wp.float64, kn_frac: wp.float64, zeta_n: wp.float64,
              fnb: wp.array(dtype=wp.float64), bst: wp.array(dtype=int), jn_out: wp.array(dtype=wp.vec3d),
              plane_reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array2d(dtype=wp.float64),
              pen_max: wp.array(dtype=wp.float64)):
    i = wp.tid()
    zero3 = wp.vec3d()
    jn_out[i] = zero3
    fnb[i] = wp.float64(0.0)
    bst[i] = -1
    m = gm[i]
    if m <= 1.0e-15:
        return
    grp = node_group[i]
    if grp >= 0:
        if group_active[grp] != 0:
            return
    e = 0
    x = node_X[i]+node_u[i]
    md = wp.float64(m)
    kn = kn_frac*md/(dt*dt)
    cn = wp.float64(2.0)*zeta_n*wp.sqrt(kn*md)
    vstar = wp.vec3d(old[i])
    zero = wp.float64(0.0)
    Jn = wp.vec3d()
    best = int(-1)
    best_pen = zero
    best_fn = zero
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
                best_fn = fn
    jn_out[i] = Jn
    fnb[i] = best_fn
    bst[i] = best


@wp.kernel
def x2_share(bst: wp.array(dtype=int), fnb: wp.array(dtype=wp.float64), d0: int, d1: int, d2: int, radius: int,
             share: wp.array(dtype=wp.float64)):
    j = wp.tid()
    share[j] = wp.float64(0.0)
    b = bst[j]
    if b < 0:
        return
    ix = j//(d1*d2)
    iy = (j//d2) % d1
    iz = j % d2
    Z = wp.float64(0.0)
    for a in range(-radius, radius+1):
        for c in range(-radius, radius+1):
            for d in range(-radius, radius+1):
                jx = ix+a
                jy = iy+c
                jz = iz+d
                if jx >= 0 and jx < d0 and jy >= 0 and jy < d1 and jz >= 0 and jz < d2:
                    k = (jx*d1+jy)*d2+jz
                    if bst[k] == b:
                        Z += hat_w(a, radius)*hat_w(c, radius)*hat_w(d, radius)
    share[j] = fnb[j]/Z


@wp.kernel
def x2_patch_sums(gm: wp.array(dtype=float), bst: wp.array(dtype=int), fnb: wp.array(dtype=wp.float64),
                  psum: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    b = bst[i]
    if b >= 0:
        wp.atomic_add(psum, b, 0, fnb[i])
        wp.atomic_add(psum, b, 1, wp.float64(gm[i]))


@wp.kernel
def x2_patch_lag(P: int, psum: wp.array2d(dtype=wp.float64), plag: wp.array(dtype=wp.float64), alpha: wp.float64,
                 started: wp.array(dtype=int)):
    """smode 10: lagged patch normal force (time constant tau; initialised to the first nonzero patch total)."""
    k = wp.tid()
    if started[k] == 0:
        if psum[k, 0] > wp.float64(0.0):
            plag[k] = psum[k, 0]
            started[k] = 1
    else:
        plag[k] = plag[k]+alpha*(psum[k, 0]-plag[k])


@wp.kernel
def x2_patch_vt(gm: wp.array(dtype=float), old: wp.array(dtype=wp.vec3), bst: wp.array(dtype=int),
                planes: wp.array(dtype=wp.vec4d), pacc: wp.array2d(dtype=wp.float64)):
    """smode 12: this substep's patch sums of m v_t (realised tangential velocity) over the nodes in contact."""
    i = wp.tid()
    b = bst[i]
    if b >= 0:
        pl = planes[b]
        n = wp.vec3d(pl[0], pl[1], pl[2])
        vr = wp.vec3d(old[i])
        vt = vr-wp.dot(vr, n)*n
        md = wp.float64(gm[i])
        wp.atomic_add(pacc, b, 0, md*vt[0])
        wp.atomic_add(pacc, b, 1, md*vt[1])
        wp.atomic_add(pacc, b, 2, md*vt[2])
        wp.atomic_add(pacc, b, 3, md)


@wp.kernel
def x2_patch_vt_slip(gm: wp.array(dtype=float), old: wp.array(dtype=wp.vec3), bst: wp.array(dtype=int),
                     planes: wp.array(dtype=wp.vec4d), fnb: wp.array(dtype=wp.float64), fric: wp.array2d(dtype=float),
                     xi: wp.array(dtype=wp.vec3d), obst: wp.array(dtype=int), dt: wp.float64, kt_frac: wp.float64,
                     zeta: wp.float64, pacc: wp.array2d(dtype=wp.float64)):
    """smode 13: patch sums of m v_t over the nodes whose v2 trial SLIPS this substep (same trial as x2_tangential)."""
    i = wp.tid()
    b = bst[i]
    if b >= 0:
        pl = planes[b]
        n = wp.vec3d(pl[0], pl[1], pl[2])
        md = wp.float64(gm[i])
        kt = kt_frac*md/(dt*dt)
        ct = wp.float64(2.0)*zeta*wp.sqrt(kt*md)
        xt = wp.vec3d()
        if obst[i] == b:
            xt = xi[i]
        xt = xt-wp.dot(xt, n)*n
        vr = wp.vec3d(old[i])
        vt = vr-wp.dot(vr, n)*n
        Ft = -kt*xt-ct*vt
        if wp.length(Ft) > wp.float64(fric[0, 0])*fnb[i]:
            wp.atomic_add(pacc, b, 0, md*vt[0])
            wp.atomic_add(pacc, b, 1, md*vt[1])
            wp.atomic_add(pacc, b, 2, md*vt[2])
            wp.atomic_add(pacc, b, 3, md)


@wp.kernel
def x2_patch_mean(pacc: wp.array2d(dtype=wp.float64), pvm: wp.array(dtype=wp.vec3d)):
    """smode 11: this substep's patch mean realised tangential velocity sum m v_t / sum m (used in the next substep);
    accumulators cleared."""
    k = wp.tid()
    m = pacc[k, 3]
    pvm[k] = wp.vec3d()
    if m > wp.float64(0.0):
        pvm[k] = wp.vec3d(pacc[k, 0], pacc[k, 1], pacc[k, 2])/m
    for c in range(4):
        pacc[k, c] = wp.float64(0.0)


@wp.kernel
def x2_gate(P: int, pacc: wp.array2d(dtype=wp.float64), vg: wp.float64, gate: wp.array(dtype=int)):
    """Patch gate for the next substep from this substep's patch slip velocity |sum m v_t| / sum m (hysteresis
    open >= vg, close < vg / 2); accumulators cleared."""
    k = wp.tid()
    m = pacc[k, 3]
    sp = wp.float64(0.0)
    if m > wp.float64(0.0):
        sp = wp.length(wp.vec3d(pacc[k, 0], pacc[k, 1], pacc[k, 2]))/m
    if gate[k] == 0:
        if sp >= vg:
            gate[k] = 1
    else:
        if sp < wp.float64(0.5)*vg:
            gate[k] = 0
    for c in range(4):
        pacc[k, c] = wp.float64(0.0)


@wp.kernel
def x2_tangential(gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
                  node_X: wp.array(dtype=wp.vec3d), node_u: wp.array(dtype=wp.vec3d), fric: wp.array2d(dtype=float),
                  P: int, planes: wp.array(dtype=wp.vec4d), dt: wp.float64, kt_frac: wp.float64, zeta: wp.float64,
                  smode: int, radius: int, d0: int, d1: int, d2: int, fnb: wp.array(dtype=wp.float64),
                  bst: wp.array(dtype=int), share: wp.array(dtype=wp.float64), xi: wp.array(dtype=wp.vec3d),
                  obst: wp.array(dtype=int), vobs: wp.array(dtype=wp.vec3d), jn_out: wp.array(dtype=wp.vec3d),
                  jt_out: wp.array(dtype=wp.vec3d), plane_reaction: wp.array(dtype=wp.spatial_vectord),
                  work: wp.array2d(dtype=wp.float64), counts: wp.array2d(dtype=int), capout: wp.array(dtype=wp.float64),
                  nf: wp.array(dtype=wp.float64), alpha: wp.float64, sacc: wp.array(dtype=wp.float64),
                  ds: wp.float64, psum: wp.array2d(dtype=wp.float64), pacc: wp.array2d(dtype=wp.float64),
                  gate: wp.array(dtype=int), plag: wp.array(dtype=wp.float64), pvm: wp.array(dtype=wp.vec3d),
                  zd: wp.float64):
    i = wp.tid()
    zero3 = wp.vec3d()
    jt_out[i] = zero3
    capout[i] = wp.float64(0.0)
    best = bst[i]
    if smode == 0:
        nf[i] = fnb[i]
    if smode == 2 or smode == 8:
        nf[i] = nf[i]+alpha*(fnb[i]-nf[i])          # every substep, zero normal force when out of contact
    if smode == 9:
        if gate[0] != 0:
            nf[i] = nf[i]+alpha*(fnb[i]-nf[i])
        else:
            nf[i] = fnb[i]
    if best < 0:
        if smode == 3 or smode == 5:
            nf[i] = wp.float64(0.0)
        sacc[i] = wp.float64(0.0)
        xi[i] = zero3
        obst[i] = -1
        return
    e = 0
    md = wp.float64(gm[i])
    kt = kt_frac*md/(dt*dt)
    ct = wp.float64(2.0)*zeta*wp.sqrt(kt*md)
    zero = wp.float64(0.0)
    x = node_X[i]+node_u[i]
    pl = planes[best]
    best_n = wp.vec3d(pl[0], pl[1], pl[2])
    mu = wp.float64(fric[e, 0])
    capn = fnb[i]
    if smode == 1:
        ix = i//(d1*d2)
        iy = (i//d2) % d1
        iz = i % d2
        capn = zero
        for a in range(-radius, radius+1):
            for c in range(-radius, radius+1):
                for d in range(-radius, radius+1):
                    jx = ix+a
                    jy = iy+c
                    jz = iz+d
                    if jx >= 0 and jx < d0 and jy >= 0 and jy < d1 and jz >= 0 and jz < d2:
                        k = (jx*d1+jy)*d2+jz
                        if bst[k] == best:
                            capn += hat_w(a, radius)*hat_w(c, radius)*hat_w(d, radius)*share[k]
    if smode == 3:
        if obst[i] != best:
            nf[i] = fnb[i]
        else:
            nf[i] = nf[i]+alpha*(fnb[i]-nf[i])
    if smode == 2 or smode == 3:
        capn = wp.max(fnb[i], nf[i])
    if smode == 4:                                  # symmetric lag (X1 capmode 3 analogue) for reference
        if obst[i] != best:
            nf[i] = fnb[i]
        else:
            nf[i] = nf[i]+alpha*(fnb[i]-nf[i])
        capn = nf[i]
    if smode == 6:                                  # patch-level cap: mu (sum F_n) m_i / sum m over the obstacle's nodes
        capn = psum[best, 0]*md/psum[best, 1]
    if smode == 10:                                 # patch-level cap on the LAGGED patch normal force
        capn = plag[best]*md/psum[best, 1]
    if smode == 8 or smode == 9:
        capn = nf[i]
    if smode == 7:                                  # patch-gated lag: memory only while the patch macro-slides
        if obst[i] != best:
            nf[i] = fnb[i]
        if gate[best] != 0:
            nf[i] = nf[i]+alpha*(fnb[i]-nf[i])
            capn = nf[i]
        else:
            nf[i] = fnb[i]
    lagged = int(0)
    if smode == 5:                                  # gated lag: memory only after a slip distance ds without stick
        if obst[i] != best:
            nf[i] = fnb[i]
            sacc[i] = zero
        if sacc[i] >= ds:
            nf[i] = nf[i]+alpha*(fnb[i]-nf[i])
            capn = nf[i]
            lagged = 1
        else:
            nf[i] = fnb[i]
    capf = mu*capn
    capout[i] = capf
    xt = wp.vec3d()
    if obst[i] == best:
        xt = xi[i]
    xt = xt-wp.dot(xt, best_n)*best_n
    vr = wp.vec3d(old[i])
    vt = vr-wp.dot(vr, best_n)*best_n
    Ft = -kt*xt-ct*vt
    Lf = wp.length(Ft)
    if smode == 7 or smode == 9 or smode == 11:
        wp.atomic_add(pacc, best, 0, md*vt[0])
        wp.atomic_add(pacc, best, 1, md*vt[1])
        wp.atomic_add(pacc, best, 2, md*vt[2])
        wp.atomic_add(pacc, best, 3, md)
    state = rct.STATE_STICK
    if Lf > capf:
        Ft_new = wp.vec3d()
        if Lf > zero:
            Ft_new = Ft*(capf/Lf)
        xt_new = Ft_new/(-kt)
        wp.atomic_add(work, e, 4, capf*wp.length(xt-xt_new))
        if smode == 5:
            sacc[i] = sacc[i]+wp.length(xt-xt_new)
        xt = xt_new
        Ft = Ft_new
        state = rct.STATE_SLIP
        if smode == 11 or smode == 12 or smode == 13:   # slip-fluctuation damper about the patch mean (outside cap)
            dvt = vt-pvm[best]
            cd = wp.float64(2.0)*zd*wp.sqrt(kt*md)
            Ft = Ft-cd*dvt
            wp.atomic_add(work, e, 5, dt*cd*wp.dot(dvt, dvt))
    else:
        if smode == 5:
            sacc[i] = zero
            nf[i] = fnb[i]
        if ct > zero:
            wp.atomic_add(work, e, 5, dt*ct*wp.dot(vt, vt))
    Jt = dt*Ft
    xi[i] = xt
    obst[i] = best
    vobs[i] = zero3
    tau = wp.cross(x, Jt)
    wp.atomic_add(plane_reaction, best, wp.spatial_vectord(Jt[0], Jt[1], Jt[2], tau[0], tau[1], tau[2]))
    wp.atomic_add(counts, e, 0, 1)
    wp.atomic_add(counts, e, state, 1)
    Jn = jn_out[i]
    v = wp.vec3d(new_v[i])+(Jn+Jt)/md
    new_v[i] = wp.vec3(v)
    jt_out[i] = Jt
    o = wp.vec3d(old[i])
    half = wp.float64(0.5)
    wp.atomic_add(work, e, 0, half*wp.dot(Jn, o))
    wp.atomic_add(work, e, 1, half*wp.dot(Jt, o))


def install(live, smode=1, radius=1, tau=1e-3, ds=5e-6, vg=1e-3, switch_t=0.0, zd=0.5):
    s = live.solver
    if s.ct_F:
        raise ValueError("dev law: planes only")
    dims = [int(v) for v in live.scene.grid_dims]
    dev = s.device
    s.x2_fnb = wp.zeros(s.node_count, dtype=wp.float64, device=dev)
    s.x2_bst = wp.full(s.node_count, -1, dtype=int, device=dev)
    s.x2_share = wp.zeros(s.node_count, dtype=wp.float64, device=dev)
    s.x2_cap = wp.zeros(s.node_count, dtype=wp.float64, device=dev)
    s.x2_nf = wp.zeros(s.node_count, dtype=wp.float64, device=dev)
    s.x2_sacc = wp.zeros(s.node_count, dtype=wp.float64, device=dev)
    s.x2_psum = wp.zeros((max(s.ct_P, 1), 2), dtype=wp.float64, device=dev)
    s.x2_pacc = wp.zeros((max(s.ct_P, 1), 4), dtype=wp.float64, device=dev)
    s.x2_gate = wp.zeros(max(s.ct_P, 1), dtype=int, device=dev)
    s.x2_plag = wp.zeros(max(s.ct_P, 1), dtype=wp.float64, device=dev)
    s.x2_pvm = wp.zeros(max(s.ct_P, 1), dtype=wp.vec3d, device=dev)
    s.x2_pstart = wp.zeros(max(s.ct_P, 1), dtype=int, device=dev)
    alpha = min(1.0, float(s.dt)/float(tau))

    count = [0]
    n_switch = int(round(float(switch_t)/float(s.dt)))
    mode_final = smode

    def launch():
        smode = mode_final if count[0] >= n_switch else 0
        count[0] += 1
        wp.launch(x2_normal, s.node_count,
                  [s.gm, s.old, s.node_X, s.ct_node_u, s.node_group, s.group_active, s.ct_P, s.ct_planes,
                   wp.float64(s.dt), wp.float64(s.ct_kn), wp.float64(s.ct_zeta_n), s.x2_fnb, s.x2_bst, s.ct_jn,
                   s.ct_plane_reaction, s.ct_work, s.ct_pen_max], device=dev)
        if smode == 1:
            wp.launch(x2_share, s.node_count, [s.x2_bst, s.x2_fnb, dims[0], dims[1], dims[2], int(radius),
                                               s.x2_share], device=dev)
        if smode == 6 or smode == 10:
            s.x2_psum.zero_()
            wp.launch(x2_patch_sums, s.node_count, [s.gm, s.x2_bst, s.x2_fnb, s.x2_psum], device=dev)
        if smode == 12:
            wp.launch(x2_patch_vt, s.node_count, [s.gm, s.old, s.x2_bst, s.ct_planes, s.x2_pacc], device=dev)
            wp.launch(x2_patch_mean, max(s.ct_P, 1), [s.x2_pacc, s.x2_pvm], device=dev)
        if smode == 13:
            wp.launch(x2_patch_vt_slip, s.node_count, [s.gm, s.old, s.x2_bst, s.ct_planes, s.x2_fnb, s.ct_fric,
                                                       s.ct_xi, s.ct_obst, wp.float64(s.dt), wp.float64(s.ct_kt),
                                                       wp.float64(s.ct_zeta), s.x2_pacc], device=dev)
            wp.launch(x2_patch_mean, max(s.ct_P, 1), [s.x2_pacc, s.x2_pvm], device=dev)
        if smode == 10:
            wp.launch(x2_patch_lag, max(s.ct_P, 1), [s.ct_P, s.x2_psum, s.x2_plag, wp.float64(alpha), s.x2_pstart],
                      device=dev)
        wp.launch(x2_tangential, s.node_count,
                  [s.gm, s.new_v, s.old, s.node_X, s.ct_node_u, s.ct_fric, s.ct_P, s.ct_planes, wp.float64(s.dt),
                   wp.float64(s.ct_kt), wp.float64(s.ct_zeta), int(smode), int(radius), dims[0], dims[1], dims[2],
                   s.x2_fnb, s.x2_bst, s.x2_share, s.ct_xi, s.ct_obst, s.ct_vobs, s.ct_jn, s.ct_jt,
                   s.ct_plane_reaction, s.ct_work, s.ct_counts, s.x2_cap, s.x2_nf, wp.float64(alpha),
                   s.x2_sacc, wp.float64(ds), s.x2_psum, s.x2_pacc, s.x2_gate, s.x2_plag, s.x2_pvm,
                   wp.float64(zd)], device=dev)
        if smode == 11:
            wp.launch(x2_patch_mean, max(s.ct_P, 1), [s.x2_pacc, s.x2_pvm], device=dev)
        if smode == 7 or smode == 9:
            wp.launch(x2_gate, max(s.ct_P, 1), [s.ct_P, s.x2_pacc, wp.float64(vg), s.x2_gate], device=dev)
    s._launch_contact = launch
    return s
