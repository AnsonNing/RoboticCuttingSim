"""CPU tests for the WP-12 resolution study runner helpers and protocol invariants (no GPU)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('run_wp12_resolution', ROOT/'scripts/run_wp12_resolution.py')
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
SPACINGS = (2e-3, 1.5e-3, 1e-3, .75e-3, .5e-3)


@pytest.mark.parametrize('s', SPACINGS)
def test_lattice_tiles_the_same_box(s):
    pts, lat = R.lattice_box(R.BAR, s)
    assert len(pts) == round(R.BAR[0]/s)*round(R.BAR[1]/s)*round(R.BAR[2]/s)
    assert np.allclose(pts.min(0), [-R.BAR[0]/2+s/2, -R.BAR[1]/2+s/2, s/2])
    assert np.allclose(pts.max(0), [R.BAR[0]/2-s/2, R.BAR[1]/2-s/2, R.BAR[2]-s/2])
    assert np.allclose(pts, (lat+.5)*s-np.array([R.BAR[0]/2, R.BAR[1]/2, 0.]))
    # the CUT plane x = 0 lies between two particle columns, and so do the MULTI2 planes x = -/+12 mm
    for x0 in (0., -.012, .012):
        assert np.min(np.abs(pts[:, 0]-x0)) == pytest.approx(s/2)


def test_spacing_not_dividing_box_raises():
    with pytest.raises(ValueError):
        R.lattice_box((.072, .024, .018), .0007)


@pytest.mark.parametrize('s', SPACINGS)
def test_dome_on_base_lattice_and_volume(s):
    pts, lat, masses, geo = R.workpiece('TRIM', s)
    assert len(np.unique(lat, axis=0)) == len(lat)
    assert np.allclose(pts, (lat+.5)*s-np.array([R.TRIM_BASE[0]/2, R.TRIM_BASE[1]/2, 0.]))
    assert abs(geo['dome_volume_lattice_m3']/geo['dome_volume_analytic_m3']-1) < .01
    assert np.allclose(masses, 1500*s**3)


@pytest.mark.parametrize('case', ['CUT', 'OBL20', 'TRIM', 'MULTI2'])
def test_blade_path_is_fixed_in_metres(case):
    plans = [R.plan(case, s, 'formal')[0] for s in SPACINGS]
    ref = np.array([p['pos'] for p in plans[0]])
    for poses in plans[1:]:
        assert np.array_equal(np.array([p['pos'] for p in poses]), ref)
        assert [p['armed'] for p in poses] == [p['armed'] for p in plans[0]]


def test_trim_gap_variant_fixes_gap_in_units_of_s():
    for s in SPACINGS:
        _, geo = R.plan('TRIM', s, 'gap_s')
        assert geo['edge_gap_to_lowest_dome_layer_m'] == pytest.approx(.3*s)
        _, geo = R.plan('TRIM', s, 'formal')
        assert geo['edge_z_m'] == pytest.approx(.0182)


def test_edge_never_below_board_and_control_speeds():
    for case in ('CUT', 'OBL20', 'MULTI2'):
        poses, geo = R.plan(case, 1e-3, 'formal')
        z = np.array([p['pos'][2] for p in poses])
        assert z.min() == pytest.approx(0., abs=1e-12)
        step = np.linalg.norm(np.diff(np.array([p['pos'] for p in poses]), axis=0), axis=1)
        armed = np.array([p['armed'] for p in poses])[1:]
        assert step[armed].max() <= R.CUT_SPEED*R.CONTROL_DT+1e-12
    poses, _ = R.plan('MULTI2', 1e-3, 'formal')
    assert sorted({p['cut'] for p in poses if p['armed']}) == [0, 1]


def test_contact_window_is_planned_geometry():
    poses, geo = R.plan('CUT', 1e-3, 'formal')
    win, steady = R.contact_window('CUT', poses, geo)
    z = np.array([p['pos'][2] for p in poses])
    assert np.all(z[win] <= R.BAR[2]+1e-9) and np.all(z[win] >= 0)
    assert np.all(win[steady])
    assert win.sum() == 91   # 18 mm at 0.2 mm per control step, ends included


def test_crossing_links_and_coverage():
    pts, lat = R.lattice_box((.012, .006, .006), 1e-3)
    links = R.crossing_links(pts, lat, [0., 0., 0.], [1., 0., 0.])
    assert len(links) == 6*6
    digits = np.where(pts[:, 0] < 0, 1, 2)
    assert R.link_coverage(links, digits)['coverage'] == 1.
    digits[pts[:, 2] < 1e-3] = 0
    cov = R.link_coverage(links, digits, lat, 0)
    assert cov['links'] == 6 and cov['conflicting'] == 0 and cov['both_unassigned'] == 6


def test_protocol_rows_valid():
    protocol = json.loads((ROOT/'configs/wp12/resolution_protocol_v1.json').read_text())
    rows = protocol['rows']
    assert len(rows) == len(set(rows))
    for rid, row in rows.items():
        dt = row['dt_s']
        assert abs(round(.02/dt)*dt-.02) < 1e-12, rid
        assert row['case'] in ('CUT', 'OBL20', 'TRIM', 'MULTI2')
        for size in (R.BAR, R.TRIM_BASE):
            R.lattice_box(size, row['spacing_m'])
        cfl = R.cfl(row['grid_dx_m'], dt)['grid_cfl']
        assert cfl == pytest.approx(row['grid_cfl'], abs=1e-4)
        if row['family'] == 'F7_stability_probe':
            assert cfl > .5
        else:
            assert cfl <= .5, rid
    # the formal baseline and the 0.15 ms dt that does not divide 20 ms
    assert rows['CUT_s100_g4s_dt0100']['grid_dx_m'] == pytest.approx(.004)
    assert not any(abs(r['dt_s']-1.5e-4) < 1e-12 for r in rows.values())


def test_severed_lines_tolerates_offset_cut():
    pts, lat = R.lattice_box((.012, .006, .006), 1e-3)
    digits = np.where(pts[:, 0] < -1e-3, 1, 2)      # realised one column off the declared plane x = 0
    exact = R.link_coverage(R.crossing_links(pts, lat, [0., 0., 0.], [1., 0., 0.]), digits)
    assert exact['coverage'] == 0.
    sev = R.severed_lines(pts, lat, digits, [0., 0., 0.], [1., 0., 0.])
    assert sev['lines'] == 36 and sev['fraction'] == 1.
    digits[pts[:, 2] < 1e-3] = 0
    assert R.severed_lines(pts, lat, digits, [0., 0., 0.], [1., 0., 0.], layer=0)['fraction'] == 0.
    far = np.where(pts[:, 0] < -5e-3, 1, 2)          # 5 mm off: outside the 4 mm band
    assert R.severed_lines(pts, lat, far, [0., 0., 0.], [1., 0., 0.])['fraction'] == 0.


_sspec = importlib.util.spec_from_file_location('summarize_wp12_resolution', ROOT/'scripts/summarize_wp12_resolution.py')
S = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(S)


def test_observed_order_recovers_power_law_on_nonuniform_ratios():
    h = [1e-3, .75e-3, .5e-3]
    for p in (1., 2.):
        q = [1.+50.*x**p/1e-3**p for x in h]
        est, how = S.observed_order(h, q)
        assert how == 'solved' and est == pytest.approx(p, rel=1e-6)


def test_convergence_rules():
    h = [1e-3, .75e-3, .5e-3]
    q = [1.+.4*(x/1e-3)**2 for x in h]     # second order; GCI_fine = 1.25 * 0.125 / 1.25 / 1.1 = 0.114
    v = S.continuous_verdict(h, q, .25, 0.)
    assert v['verdict'] == 'CONVERGED' and v['p'] == pytest.approx(2.)
    assert S.continuous_verdict(h, q, .10, 0.)['verdict'] == 'NOT_CONVERGED'   # asymptotic, error above tol
    assert S.continuous_verdict(h, [1., 2., 1.], .1, 0.)['verdict'] == 'NOT_CONVERGED'     # oscillation
    assert S.continuous_verdict(h, [1., 1.02, 1.01], .1, 0.)['verdict'] == 'CONVERGED'     # insensitive
    assert S.continuous_verdict(h, [1., 2., 3.], .1, 0.)['verdict'] == 'NOT_CONVERGED'     # p = 0 (linear in index)
    assert S.continuous_verdict(h[:2], [1., 2.], .1, 0.)['verdict'] == 'INCONCLUSIVE'
    assert S.zero_limit_verdict(h, [1e-3, .75e-3, .5e-3], 1e-6, 0.)['verdict'] == 'CONVERGED'
    assert S.zero_limit_verdict(h, [1e-7, 2e-7, 1e-7], 1e-6, 0.)['verdict'] == 'CONVERGED'
    assert S.zero_limit_verdict(h, [1e-3, 1e-3, 1.1e-3], 1e-6, 1e-6)['verdict'] == 'NOT_CONVERGED'
    assert S.discrete_verdict([2, 1.5, 1, .75], [1, 2, 2, 2])['verdict'] == 'CONVERGED'
    assert S.discrete_verdict([2, 1.5, 1, .75], [2, 2, 2, 1])['verdict'] == 'NOT_CONVERGED'


def test_summary_row_ids_match_protocol():
    rows = json.loads((ROOT/'configs/wp12/resolution_protocol_v1.json').read_text())['rows']
    for case in ('CUT', 'OBL20', 'TRIM', 'MULTI2'):
        for kind in ('P_s_primary', 'P_s_formal_dt', 'P_ratio'):
            assert all(r in rows for _, r in S.path_rows(rows, case, kind))
        for s in SPACINGS:
            ids = [r for _, r in S.path_rows(rows, case, 'P_dt', s)]
            assert all(r in rows for r in ids if not r.startswith(f'{case}_s050_g4s_dt0200'))
