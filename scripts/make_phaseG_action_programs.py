#!/usr/bin/env python3
"""Write the hand-authored WP-11 Phase G (Test E) open-loop action programs (``rcs_action_program_v1``).

Independent of the goal planner: no ``goal_planner``/``policies`` import. Each
program is written from the declared scene numbers (workpiece pose and size,
protrusion position, declared HOME hand pose), the tool profile's public mount
(``hand_from_edge``) and the goal's declared clearance and rake, with its own
edge-frame formula and its own speeds, start/end points and phase timing. The
programs are open loop: no observation is read while they run.

  .venv/bin/python scripts/make_phaseG_action_programs.py --out-dir configs/wp11/action_programs
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim.scene_config import load_json, scene_world_pose, workcell_home  # noqa: E402
from robotic_cutting_sim.tool_profile import PROFILES  # noqa: E402


def edge_quaternion(advance, edge_line):
    """Edge frame: x = advance, y = edge line, z = x cross y (blade normal); returns XYZW."""
    x = np.asarray(advance, float)/np.linalg.norm(advance)
    y = np.asarray(edge_line, float)
    y = y-(y@x)*x
    y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.column_stack([x, y, np.cross(x, y)])).as_quat()


def pose(xyz, q):
    return [float(v) for v in xyz]+[float(v) for v in q]


def trim_program():
    scene = load_json('trim01_wp11_workcell', ROOT/'configs/wp11/scenes')
    goal = load_json('trim_bump_rho10', ROOT/'configs/wp11/goals')
    base = scene['workpiece']['base_box_m']
    world = scene_world_pose(scene)
    dome = scene['workpiece']['protrusions'][0]
    top = world[2]+base['height_m']                       # bar top, world z (identity yaw scene)
    edge_z = top+goal['edge_clearance_m']                 # the goal's declared edge clearance above the plane
    rho = np.deg2rad(goal['planner']['rake_deg'])         # the goal's declared rake
    # Near-horizontal skim along world +x, edge line along world +y (tip toward +y),
    # blade body behind the edge rising by rho: x = (cos rho, 0, -sin rho).
    q_skim = edge_quaternion([np.cos(rho), 0., -np.sin(rho)], [0., 1., 0.])
    home = workcell_home(scene)
    profile = PROFILES[scene['tool']['profile_id']]
    hand = np.asarray(home['hand_pose_world_xyzw'], float)
    mount = np.asarray(profile.hand_from_edge(), float)
    R_hand = Rotation.from_quat(hand[3:])
    home_edge = np.r_[hand[:3]+R_hand.apply(mount[:3]), (R_hand*Rotation.from_quat(mount[3:])).as_quat()]
    x0, x1 = dome['x_m']-dome['radius_x_m']-.014, dome['x_m']+dome['radius_x_m']+.014   # +/- 26 mm
    y = world[1]+dome['y_m']
    above = top+.040
    segments = [
        dict(label='REORIENT', kind='edge_line', **{'from': pose(home_edge[:3], home_edge[3:])},
             to=pose([x0, y, above], q_skim), speed_m_s=.04, cut_enabled=False),
        dict(label='PRE_CUT', kind='edge_line', **{'from': pose([x0, y, above], q_skim)},
             to=pose([x0, y, edge_z+.010], q_skim), speed_m_s=.02, cut_enabled=False),
        dict(label='PRE_CUT', kind='edge_line', **{'from': pose([x0, y, edge_z+.010], q_skim)},
             to=pose([x0, y, edge_z], q_skim), speed_m_s=.01, cut_enabled=False),
        dict(label='PRE_CUT', kind='edge_pose', target=pose([x0, y, edge_z], q_skim), steps=15, cut_enabled=False),
        dict(label='SKIM', kind='edge_line', **{'from': pose([x0, y, edge_z], q_skim)},
             to=pose([x1, y, edge_z], q_skim), speed_m_s=.02, cut_enabled=True, cut_index=0),
        dict(label='WITHDRAW', kind='edge_line', **{'from': pose([x1, y, edge_z], q_skim)},
             to=pose([x1, y, above], q_skim), speed_m_s=.02, cut_enabled=False),
        dict(label='HOLD', kind='hold_until_time_limit'),
    ]
    return dict(schema='rcs_action_program_v1', program_id='trim_rho10_handwritten_v1',
                task=dict(scene='trim01_wp11_workcell', goal='trim_bump_rho10'),
                description=('Hand-authored open-loop TRIM (Test E): synchronized straight-line REORIENT from the '
                             'declared HOME edge pose at 40 mm/s, descent to the goal clearance (0.2 mm), 0.3 s '
                             'settle, armed skim at 20 mm/s from x = -26 mm to +26 mm (planner: 25 mm/s, +/-22 mm, '
                             'closed loop), lift 40 mm, hold to the time limit. Written from scene/tool/goal numbers '
                             'only; no planner code.'),
                derivation=dict(bar_top_world_z_m=top, edge_z_world_m=edge_z, rake_deg=goal['planner']['rake_deg'],
                                home_edge_world_xyzw=home_edge.tolist(), skim_x_m=[x0, x1]),
                segments=segments)


def sever_program():
    scene = load_json('bar_160x30x20', ROOT/'configs/wp11/scenes')
    world = scene_world_pose(scene)
    top = world[2]+scene['workpiece']['base_box_m']['height_m']
    # Vertical through-cut at x = 0: advance -z, edge line along +y.
    q = edge_quaternion([0., 0., -1.], [0., 1., 0.])
    segments = [
        dict(label='APPROACH', kind='edge_pose', target=pose([0., 0., top+.010], q), steps=40, cut_enabled=False),
        dict(label='CUT', kind='edge_line', **{'from': pose([0., 0., top+.010], q)},
             to=pose([0., 0., world[2]-.004], q), speed_m_s=.008, cut_enabled=True, cut_index=0),
        dict(label='HOLD', kind='edge_pose', target=pose([0., 0., world[2]-.004], q), steps=15, cut_enabled=False),
        dict(label='WITHDRAW', kind='edge_line', **{'from': pose([0., 0., world[2]-.004], q)},
             to=pose([0., 0., top+.010], q), speed_m_s=.02, cut_enabled=False),
        dict(label='HOLD', kind='hold_until_time_limit'),
    ]
    return dict(schema='rcs_action_program_v1', program_id='sever_mid_handwritten_v1',
                task=dict(scene='bar_160x30x20', goal='sever_mid'),
                description=('Hand-authored open-loop through-cut (Test E): clamped approach to 10 mm above the bar '
                             'top, armed vertical cut at 8 mm/s to 4 mm below the bar bottom (planner: 10 mm/s), '
                             '0.3 s hold, withdraw, hold to the time limit. Scene numbers only; no planner code.'),
                derivation=dict(bar_top_world_z_m=top, cut_end_world_z_m=world[2]-.004), segments=segments)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for program in (trim_program(), sever_program()):
        path = a.out_dir/f"{program['program_id']}.json"
        if path.exists():
            raise SystemExit(f'{path} exists; never overwrite (bump the program version)')
        path.write_text(json.dumps(program, indent=2)+'\n')
        print(path)


if __name__ == '__main__':
    sys.exit(main())
