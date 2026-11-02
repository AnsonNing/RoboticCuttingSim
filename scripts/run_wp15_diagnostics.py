#!/usr/bin/env python3
"""Bounded scripted environment diagnostics for agentic control robot research.

``accept(open_env, ctx)`` runs inside the existing guarded Native worker. This
program is evaluator-private environment validation, never a candidate skill.
The CLI only checks the predeclared scene seed; it does not launch a GPU job.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from robotic_cutting_sim.agentic_trim_env import write_json
from robotic_cutting_sim.seeded_trim import compile_seed

SEEDS = {0: 2, 1: 1}  # Count only, declared before any diagnostic outcome.


def preflight(seed):
    if seed not in SEEDS:
        raise ValueError('The bounded diagnostic protocol declares scene seeds 0 and 1 only')
    task, manifest = compile_seed(seed)
    count = len(task.geometry.bumps)
    if count != SEEDS[seed]:
        raise ValueError('Predeclared diagnostic bump count changed; preserve and investigate')
    return dict(scene_seed=seed, protrusion_count=count, scene_hash=manifest['scene_hash'],
                initial_particles_sha256=manifest['initial_particles_sha256'],
                target_particle_counts=manifest['target_particle_counts'],
                controller_provenance='scripted_reference_diagnostic', agent_success_eligible=False,
                status='PASS', scope='Geometry preflight only; Native cutting NOT_RUN')


def run_diagnostic(adapter, seed):
    """Use the existing reference schedule unchanged, preserving the full recorder."""
    plan = preflight(seed)
    if adapter.seed != seed or len(adapter.task.geometry.bumps) != SEEDS[seed]:
        raise ValueError('Opened Native scene differs from the predeclared diagnostic')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    adapter.provenance.update(controller_provenance='scripted_reference_diagnostic',
        split='scripted_diagnostic', candidate_id='built_in_reference_diagnostic',
        controller_sha256=source_hash, agent_success_eligible=False)
    wall = time.perf_counter()
    adapter.reset(seed=seed)
    rec = adapter.current_recording
    write_json(rec / 'diagnostic_protocol.json', {**plan, 'script_sha256': source_hash,
        'reference_action_sha256': hashlib.sha256(
            inspect.getsource(type(adapter.task).reference_action).encode()).hexdigest(),
        'controller_source': 'existing compiled task reference action',
        'maximum_control_steps': 1200, 'seed_replacement_allowed': False})
    rejected = clamped = 0
    termination = None
    try:
        for _ in range(1200):
            next_time = min(adapter.task.duration_s, adapter.time_s + adapter.task.control_dt_s)
            action = adapter.task.reference_action(next_time)
            _, _, terminated, truncated, info = adapter.env.step(action)
            adapter.total_steps += 1
            rejected += str(info.get('action_status', '')).startswith('REJECTED')
            clamped += info.get('action_status') == 'CLAMPED'
            termination = info.get('termination_reason')
            if terminated or truncated or adapter.time_s >= adapter.task.duration_s - 1e-9:
                break
    except BaseException as error:
        write_json(rec / 'diagnostic_failure.json', dict(status='FAIL_EXECUTION',
            error_type=type(error).__name__, error=str(error), scene_seed=seed,
            controller_provenance='scripted_reference_diagnostic'))
        raise
    finally:
        # Native shutdown can terminate Python. Finalize physical data and all
        # automatic success media while the running environment still exists.
        adapter.finalize('SCRIPTED_DIAGNOSTIC_DONE')
    result = json.loads((rec / 'independent_evaluation.json').read_text())
    report = dict(controller_provenance='scripted_reference_diagnostic', agent_success_eligible=False,
        scene_seed=seed, protrusion_count=SEEDS[seed], scene_hash=plan['scene_hash'],
        status=result['status'], task_success=bool(result.get('success')),
        integrity_status=result.get('integrity', {}).get('status'),
        outcome_status=result.get('outcome', {}).get('status'),
        quality_status=result.get('surface_quality_status'), home_status=result.get('home_status'),
        metrics=result.get('outcome', {}).get('metrics', {}), steps=adapter.total_steps,
        action_rejections=rejected, clamped_actions=clamped, termination_reason=termination,
        simulation_duration_s=adapter.time_s, wall_time_s=time.perf_counter() - wall)
    write_json(rec / 'diagnostic_result.json', report)
    return report


def accept(open_env, ctx):
    """Existing guarded-worker acceptance entry: exactly one declared seed/job."""
    seed = int(ctx['seed'])
    preflight(seed)
    with open_env(seed, name='scripted_diagnostic') as adapter:
        report = run_diagnostic(adapter, seed)
    return dict(layers={'scripted_reference_task': 'PASS' if report['task_success'] else 'FAIL'},
                diagnostic=report, agent_success_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=tuple(SEEDS))
    args = parser.parse_args()
    print(json.dumps(preflight(args.seed), indent=2))


if __name__ == '__main__':
    main()
