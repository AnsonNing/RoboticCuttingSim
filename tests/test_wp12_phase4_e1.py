"""E1 protocol and runner helpers (configs/wp12/phase4_e1_protocol_v1.json, scripts/run_wp12_phase4_e1.py); pure CPU."""
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

import run_wp12_phase4_e1 as e1  # noqa: E402


def test_protocol_rows_are_unique_and_complete():
    protocol = json.loads(e1.PROTOCOL.read_text())
    ids = [r['id'] for r in protocol['rows']]
    assert len(ids) == len(set(ids)) == protocol['budget']['rows']
    taus = {r['tau_deg'] for r in protocol['rows'] if r['family'] == 'E1.1_option_A'}
    assert taus == {float(t) for t in (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, -1, -2, -3, -5, -10, -15, -20, -25, -30, -35)}
    raised = {(r['tau_deg'], r['edge_end_z_mm']) for r in protocol['rows'] if r['family'] == 'E1.1_option_A_raised'}
    assert raised == {(float(t), h) for t in (5, 20, 35) for h in (.25, .375, .5, .75, 1.)}
    slotted = [r for r in protocol['rows'] if r['family'] == 'E1.2_option_D_slotted']
    assert {r['tau_deg'] for r in slotted} == {0., 20., 25., 30., 35.}
    for r in slotted:
        assert math.isclose(r['edge_end_z_mm'], -r['overtravel_mm']*math.cos(math.radians(r['tau_deg'])), abs_tol=1e-4)
        assert r['edge_end_z_mm'] >= -8.0   # action-workspace floor of the slotted board


def test_frame_matches_the_vertical_planner_rotation_and_mirrors():
    from robotic_cutting_sim.goal_planner import through_cut_rotation
    assert np.allclose(e1.edge_rotation(0.).as_matrix(), through_cut_rotation([1., 0., 0.]).as_matrix(), atol=1e-12)
    for tau in (5., 20., 35.):
        d, e, n = e1.frame(tau)
        dm, em, nm = e1.frame(-tau)
        assert np.allclose(np.column_stack([d, e, n]).T @ np.column_stack([d, e, n]), np.eye(3), atol=1e-12)
        assert np.allclose(dm*[-1, 1, 1], d) and np.allclose(nm*[1, 1, -1], n)


def test_schedule_reaches_the_declared_end_exactly():
    rows = e1.protocol_rows()
    for name in ('A_p00', 'A_p35', 'R_p20_h0375', 'D_p35_reach'):
        row = rows[name]
        schedule = e1.control_schedule(row)
        cut = [p for p, armed, phase in schedule if phase == 'CUT']
        assert all(armed for _, armed, phase in schedule if phase == 'CUT')
        assert not any(armed for _, armed, phase in schedule if phase != 'CUT')
        assert np.isclose(cut[-1][2], row['edge_end_z_mm']*1e-3, atol=1e-12)
        path = e1.edge_path(row)
        assert np.isclose((cut[-1]-path['o'])@path['n'], 0., atol=1e-12)     # the edge stays on the cut plane
        steps = np.linalg.norm(np.diff(np.array([p for p, _, ph in schedule if ph in ('SETTLE', 'CUT')]), axis=0), axis=1)
        assert steps.max() <= e1.CUT_SPEED*e1.CONTROL_DT+1e-12


def test_option_a_blade_touches_but_never_enters_the_board_and_slots_keep_one_mm():
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y
    profile = np.asarray(RULER_KNIFE_V2_Y.tlmpm_profile()[0], float)
    solid = [(np.array([0., .03, -.01]), np.array([.18, .095, .01]))]
    slotted = [(np.array([0., .03, -.014]), np.array([.18, .095, .006])),
               (np.array([-.116, .03, -.004]), np.array([.064, .095, .004])),
               (np.array([-.032, .03, -.004]), np.array([.012, .095, .004])),
               (np.array([0., .03, -.004]), np.array([.012, .095, .004])),
               (np.array([.032, .03, -.004]), np.array([.012, .095, .004])),
               (np.array([.116, .03, -.004]), np.array([.064, .095, .004]))]
    for row in e1.protocol_rows().values():
        rotation = e1.edge_rotation(row['tau_deg'])
        lowest, vertex = e1.lowest_blade_vertex(profile, rotation)
        assert vertex == 0 and lowest == 0.
        path = e1.edge_path(row)
        if row['scene'] == 'bar_flat_board':
            c = e1.blade_board_clearance(profile, rotation, path['start'], path['end'], solid)
            assert np.isclose(c, row['edge_end_z_mm']*1e-3, atol=1e-9)
        else:
            assert e1.blade_board_clearance(profile, rotation, path['start'], path['end'], slotted) >= 1e-3
