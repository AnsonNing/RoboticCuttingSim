"""WP-12 Phase 5 FROZEN benchmark: pinned file hashes, freeze_problems == [], the phase5_id_rules population split,
runner provenance (frozen envelope, DRAFT refusal), the Demo 7 selector and the PREDECLARED Phase 7 demo file.
CPU only (no physics, no preflight)."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim import generalization_wp12 as W
from robotic_cutting_sim.goals import _TYPE_FIELDS

ROOT = W.ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(f'test_p5f_{name}', ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def P5():
    return _load('run_wp12_phase5')


@pytest.fixture(scope='module')
def config():
    return W.load_benchmark(W.BENCHMARK_V1)


@pytest.fixture(scope='module')
def envelope():
    return W.load_envelope(W.ENVELOPE_V2)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- frozen files

def test_frozen_file_hashes():
    """The committed frozen files (722468c); an edit is benchmark v2 / envelope v3, never a change to these."""
    assert W.FROZEN_SHA256['configs/wp12/benchmark_v1.json'].startswith('915a2a8f')
    assert W.FROZEN_SHA256['configs/wp12/supported_envelope_v2.json'].startswith('040e7b96')
    for rel, digest in W.FROZEN_SHA256.items():
        assert _sha(ROOT/rel) == digest, rel


def test_frozen_state_and_declared_envelope(config, envelope):
    assert config['status'] == 'FROZEN' and config['freeze']['state'] == 'FROZEN_ON_COMMIT'
    assert 'DRAFT' not in config['benchmark_id']
    assert config['envelope'] == 'configs/wp12/supported_envelope_v2.json'
    assert envelope['status'] == 'DECLARED'


def test_freeze_problems_empty(config, envelope):
    assert W.freeze_problems(config, envelope) == []


def test_thresholds_are_the_goal_defaults(config):
    snapshot = config['thresholds']['snapshot']
    current = {k: {kk: vv for kk, vv in v.items() if vv is not None and not isinstance(vv, (list, dict))}
               for k, v in _TYPE_FIELDS.items()}
    assert snapshot == current


def test_declared_sizes(config):
    assert [s['id'] for s in W.strata(config)] == ['A_ID', 'A_OOD', 'B_ID', 'B_OOD', 'BF_OOD', 'C_ID', 'C_OOD', 'D_ID',
                                                    'D_OOD', 'E_ID', 'E_OOD', 'F_ID', 'F_OOD', 'G_ID', 'G_OOD', 'H_ID', 'H_OOD']
    assert {sp: len(W.sample_ids(config, sp)) for sp in W.SPLITS} == {'benchmark': 170, 'held_out': 85, 'dev': 34}
    ids = [W.make_sample_id(config, f, rs, sp, s) for sp in W.SPLITS for f, rs, s, _ in W.sample_ids(config, sp)]
    assert len(ids) == len(set(ids))


def test_id_rules_split_the_id_population(config, envelope):
    """Every ID_validated sample satisfies envelope v2 phase5_id_rules; every ID_unconfirmed sample fails at least one
    (decision E-2); counts 64/16 benchmark, 32/8 held-out, 13/3 dev. BF has no ID sample."""
    id_env = {'schema': W.ENVELOPE_SCHEMA, 'envelope_id': 'phase5_id_rules', 'status': 'DECLARED',
              'rules': envelope['phase5_id_rules']['rules']}
    counts = {}
    for split in W.SPLITS:
        for family, rs, seed, _ in W.sample_ids(config, split):
            if rs != 'ID':
                continue
            assert family != 'BF'
            sample = W.build_sample(config, family, rs, seed, split=split)
            population = W.id_population(config, family, sample['variant'])
            status = W.evaluate_envelope(id_env, sample['envelope_parameters'])['status']
            assert status == ('SUPPORTED' if population == 'ID_validated' else 'UNSUPPORTED'), (sample['sample_id'], status)
            assert W.evaluate_envelope(envelope, sample['envelope_parameters'])['status'] == 'SUPPORTED'
            counts[(split, population)] = counts.get((split, population), 0)+1
    assert counts == {('benchmark', 'ID_validated'): 64, ('benchmark', 'ID_unconfirmed'): 16,
                      ('held_out', 'ID_validated'): 32, ('held_out', 'ID_unconfirmed'): 8,
                      ('dev', 'ID_validated'): 13, ('dev', 'ID_unconfirmed'): 3}


def test_targets_are_over_id_validated(config):
    for name in ('kinematic_ID', 'native_ID'):
        target = config['targets'][name]
        assert target['population'] == 'ID_validated' and target['metric'] == 'success_among_feasible'
    assert config['targets']['kinematic_ID']['value'] == 0.9 and config['targets']['native_ID']['value'] == 0.8


# ----------------------------------------------------------------------------- runner provenance (finding 3)

def test_runner_uses_the_declared_envelope_of_a_frozen_config(P5):
    cfg, env, record = P5.resolve_provenance(W.BENCHMARK_V1, None)
    assert env == W.ENVELOPE_V2.resolve() and record['frozen_hashes_verified'] and not record['dev']
    cfg, env, record = P5.resolve_provenance(W.BENCHMARK_V1, W.ENVELOPE_V2)
    assert env == W.ENVELOPE_V2.resolve()
    with pytest.raises(SystemExit, match='does not match'):
        P5.resolve_provenance(W.BENCHMARK_V1, W.ENVELOPE)
    with pytest.raises(SystemExit, match='does not match'):   # --dev never lifts the envelope mismatch
        P5.resolve_provenance(W.BENCHMARK_V1, W.ENVELOPE, dev=True)


def test_runner_refuses_the_draft_unless_dev(P5):
    with pytest.raises(SystemExit, match='DRAFT'):
        P5.resolve_provenance(W.DRAFT, None)
    _, env, record = P5.resolve_provenance(W.DRAFT, None, dev=True)
    assert env == W.ENVELOPE.resolve() and record['dev'] and record['dev_reasons']
    with pytest.raises(SystemExit):
        P5.main(['--config', str(W.DRAFT), 'samples', '--split', 'dev', '--out', '/nonexistent/x.json'])


def test_runner_refuses_an_edited_frozen_copy(P5, tmp_path):
    doc = json.loads(W.BENCHMARK_V1.read_text())
    doc['envelope'] = str(W.ENVELOPE_V2.relative_to(ROOT))
    copy = tmp_path/'benchmark_v1.json'
    copy.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match='pinned'):
        P5.resolve_provenance(copy, None)
    _, _, record = P5.resolve_provenance(copy, None, dev=True)
    assert record['dev'] and not record['frozen_hashes_verified']


def test_default_config_is_frozen_and_queue_forwards_envelope(P5):
    parser_default = P5.W.BENCHMARK_V1
    assert parser_default == ROOT/'configs/wp12/benchmark_v1.json'
    args = SimpleNamespace(config=W.BENCHMARK_V1, envelope=W.ENVELOPE_V2, dev=False)
    prefix = P5.child_prefix(args)
    assert prefix[prefix.index('--config')+1] == str(W.BENCHMARK_V1)
    assert prefix[prefix.index('--envelope')+1] == str(W.ENVELOPE_V2) and '--dev' not in prefix
    assert '--dev' in P5.child_prefix(SimpleNamespace(**{**vars(args), 'dev': True}))


# ----------------------------------------------------------------------------- Demo 7 selector

def _feasible_rows(config, split, cls=lambda sid: 'valid_feasible'):
    rows = []
    for family, rs, seed, index in W.sample_ids(config, split):
        sid = W.make_sample_id(config, family, rs, split, seed)
        rows.append({'sample_id': sid, 'family': family, 'range_set': rs, 'variant': W.variant_of(config, family, index),
                     'feasibility_class': cls(sid), 'reason': 'synthetic'})
    return rows


def test_demo7_selection_rule(config):
    rows = _feasible_rows(config, 'held_out')
    sel = W.demo7_selection(config, rows)
    assert sel['pool_size'] == 32 and sel['rng_seed'] == [1205, 7]
    pool = [p['sample_id'] for p in sel['pool']]
    declared = [W.make_sample_id(config, f, rs, 'held_out', s) for f, rs, s, _ in W.sample_ids(config, 'held_out')]
    assert pool == [sid for sid in declared if sid in set(pool)]            # declared order
    assert all('_ID_held_out_' in sid for sid in pool)
    assert not any(sid.startswith(('P5_F_',)) for sid in pool)              # F is ID_unconfirmed
    perm = np.random.default_rng([1205, 7]).permutation(len(pool))
    assert sel['demo_7_sample_id'] == pool[perm[0]]
    assert len(sel['excluded_id_samples']) == 8                             # the ID_unconfirmed held-out samples
    # outcomes are not an input; OOD classes and unconfirmed classes never change the draw
    other = _feasible_rows(config, 'held_out', lambda sid: 'valid_feasible' if '_ID_' in sid else 'ik_unreachable')
    assert W.demo7_selection(config, other)['demo_7_sample_id'] == sel['demo_7_sample_id']


def test_demo7_selection_excludes_infeasible_and_needs_complete_feasibility(config):
    rows = _feasible_rows(config, 'held_out')
    first = W.demo7_selection(config, rows)['demo_7_sample_id']
    changed = [{**r, 'feasibility_class': 'collision_constrained'} if r['sample_id'] == first else r for r in rows]
    sel = W.demo7_selection(config, changed)
    assert first not in [p['sample_id'] for p in sel['pool']] and sel['pool_size'] == 31
    assert 'collision_constrained' in sel['excluded_id_samples'][first]
    with pytest.raises(ValueError, match='incomplete'):
        W.demo7_selection(config, [r for r in rows if r['sample_id'] != first])
    assert 'default_rng([1205, 7])' in config['phase7_demos']['demo_7']['order']


def test_demo7_select_refuses_a_second_draw_and_after_held_out_episodes(P5, config, tmp_path, monkeypatch):
    feas = tmp_path/'feas'
    feas.mkdir()
    for r in _feasible_rows(config, 'held_out'):
        (feas/f"{r['sample_id']}.json").write_text(json.dumps(r))
    fake = tmp_path/'root'
    (fake/'artifacts/wp12').mkdir(parents=True)
    monkeypatch.setattr(P5, 'ROOT', fake)
    args = SimpleNamespace(config=W.BENCHMARK_V1, envelope=W.ENVELOPE_V2, dev=False, cmd='demo7-select', feasibility=str(feas),
                           out=str(fake/'artifacts/wp12/phase7_demo7_selection_001.json'), evidence_copy=None,
                           provenance={'dev': False})
    assert P5.cmd_demo7_select(args) == 0
    record = json.loads((fake/'artifacts/wp12/phase7_demo7_selection_001.json').read_text())
    assert record['demo_7_sample_id'] == W.demo7_selection(config, _feasible_rows(config, 'held_out'))['demo_7_sample_id']
    with pytest.raises(SystemExit, match='never redrawn'):
        P5.cmd_demo7_select(SimpleNamespace(**{**vars(args), 'out': str(tmp_path/'second.json')}))
    (fake/'artifacts/wp12/phase7_demo7_selection_001.json').unlink()
    episode = fake/'artifacts/wp12/phase5_x/kinematic_held_out'/record['demo_7_sample_id']
    episode.mkdir(parents=True)
    (episode/'manifest.json').write_text('{}')
    with pytest.raises(SystemExit, match='held-out episode outputs'):
        P5.cmd_demo7_select(args)


def test_repeat_selection_is_first_feasible_id_per_family(P5, config):
    rows = _feasible_rows(config, 'benchmark',
                          lambda sid: 'ik_unreachable' if sid == W.make_sample_id(config, 'A', 'ID', 'benchmark',
                                                                                     W.seed_list(config, 'A', 'benchmark', 'ID')[0])
                          else 'valid_feasible')
    sel = P5.repeat_selection(config, rows)
    assert sorted(sel) == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    assert sel['A'] == W.make_sample_id(config, 'A', 'ID', 'benchmark', W.seed_list(config, 'A', 'benchmark', 'ID')[1])
    assert sel['C'] == W.make_sample_id(config, 'C', 'ID', 'benchmark', W.seed_list(config, 'C', 'benchmark', 'ID')[0])


# ----------------------------------------------------------------------------- Phase 7 demos (PREDECLARED)

def test_phase7_demos_file(config):
    doc = json.loads((ROOT/'configs/wp12/phase7_demos_v1.json').read_text())
    assert doc['status'] == 'PREDECLARED' and [d['id'] for d in doc['demos']] == [f'DEMO{i}' for i in range(1, 8)]
    live = json.loads((ROOT/'configs/wp12/workcell_live_v1.json').read_text())
    episodes = {e['id']: e for e in live['episodes']}
    common = doc['common']
    assert common['runtime']['flags'] == live['runtime']['flags']
    assert {'--record-cameras', '--overview-camera', '--return-home', '--hold-until-time-limit'} <= set(common['runtime']['flags'])
    assert common['planner_overrides'] == {'max_rotation_deg': 1.0, 'max_translation_m': 0.001, 'controller_profile': 'path_v2'}
    assert common['planner_overrides'] == config['native']['planner_overrides']
    assert common['surface_profiles']['order'] == ['particle_surface_v1', 'cut_aware_surface_v1']
    assert 'cut_aware_surface_v2' not in common['surface_profiles']['order']
    assert doc['benchmark']['sha256'] == W.FROZEN_SHA256['configs/wp12/benchmark_v1.json']
    for demo in doc['demos'][:6]:
        scene = json.loads((ROOT/demo['scene']).read_text())
        assert _sha(ROOT/demo['scene']) == demo['scene_sha256'] and _sha(ROOT/demo['goal']) == demo['goal_sha256'], demo['id']
        solver = scene['solver']
        assert (solver['module'], solver['cutting_law'], solver['particle_spacing_m'], solver['grid_dx_m']) == \
            ('tlmpm_v5', 8, 0.001, 0.004), demo['id']
        assert scene['workpiece']['material_profile'] == 'cutting_tasks_v1'
        assert scene['supports']['kind'] == 'flat_board'
        if 'workcell_live_episode' in demo:
            spec = episodes[demo['workcell_live_episode']]
            assert (demo['scene'], demo['goal']) == (spec['scene'], spec['goal']), demo['id']
    assert [d.get('workcell_live_episode') for d in doc['demos']] == ['CUT', None, 'TRIM', 'TESTB', 'TESTC', None, None]
    slot = json.loads((ROOT/doc['demos'][1]['scene']).read_text())
    board = slot['supports']['workcell']['overrides']['cutting_board']
    assert board['kind'] == 'slotted' and board['slot_depth_m'] == 0.008 and board['slots'] == [{'x_board_m': 0.0, 'width_m': 0.008}]
    tilted = json.loads((ROOT/doc['demos'][1]['goal']).read_text())['planes'][0]['normal_local']
    assert abs(np.degrees(np.arctan2(-tilted[2], tilted[0])) - 20.0) < 1e-9
    seq = json.loads((ROOT/doc['demos'][5]['goal']).read_text())
    assert [s['type'] for s in seq['steps']] == ['sever', 'move_piece', 'remove_protrusions']
    assert seq['steps'][1]['distance_m'] == 0.02
    demo7 = doc['demos'][6]['selection']
    for key in ('pool', 'order', 'when', 'never'):
        assert demo7[key] == config['phase7_demos']['demo_7'][key]


def test_demo2_and_demo6_inputs_regenerate_from_e2():
    """configs/wp12/phase7/*.json are exactly the E2 SLOT_p20 / CUT_PUSH_TRIM inputs (rebuilt from the protocol)."""
    M = _load('make_wp12_phase7_demo_inputs')
    built = M.build_inputs()
    assert built['demo2_slot_p20']['task_id'] == 'CFG-E2_SLOT_p20-E2_SLOT_p20_sever_tilt_slotted-c2423454'
    assert built['demo6_cut_push_trim']['task_id'] == 'CFG-E2_CUT_PUSH_TRIM-E2_CUT_PUSH_TRIM_cut_push_trim-f8bb915d'
    for name, item in built.items():
        for kind in ('scene', 'goal'):
            assert (ROOT/f'configs/wp12/phase7/{name}_{kind}.json').read_text() == M.text(item[kind]), (name, kind)
    e2 = ROOT/'artifacts/wp12/phase4_e2_001'
    if e2.exists():   # the recorded E2 inputs (artifacts are not tracked)
        for name, row in M.DEMO_ROWS.items():
            recorded = json.loads((e2/row/'manifest.json').read_text())['config']
            assert recorded['scene'] == deepcopy(built[name]['scene']) and recorded['goal'] == built[name]['goal']
