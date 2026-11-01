"""Compiled command replay must detect altered physical execution receipts."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from summarize_wp14 import replay
from run_wp13 import build


@pytest.mark.parametrize('corruption', [None, 'position', 'angle', 'cut', 'phase'])
def test_replay_audits_actual_recorded_command_prefix(corruption):
    task, _ = build({'id': 'C00_cut', 'episode': 'CUT', 'seed': 42}, 'tlmpm_v7')
    actions = deepcopy(json.loads((ROOT/'tests/fixtures/wp14/c00_recorded_prefix.json').read_text())['rows'])
    row = actions[-1]
    if corruption == 'position':
        row['edge_pose_world'][0] += .001
    elif corruption == 'angle':
        row['edge_pose_world'][3:] = (Rotation.from_rotvec([.01, 0., 0.])*
                                     Rotation.from_quat(row['edge_pose_world'][3:])).as_quat().tolist()
    elif corruption == 'cut':
        row['cut_enabled'] = True
    elif corruption == 'phase':
        row['phase'] = 'WRONG'
    audit = replay(task, actions)
    assert audit['status'] == ('PASS' if corruption is None else 'FAIL')
    assert audit['controls'] == 3
