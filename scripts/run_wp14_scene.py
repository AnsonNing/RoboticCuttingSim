#!/usr/bin/env python3
"""Prepare or run a source-bound WP-14 Native scene, with real camera recording."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
CATALOG = ROOT/'configs/wp14/scene_catalog_v1.json'


def prepare(name, out, gpu=0):
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.research_profiles import validate_runtime
    from robotic_cutting_sim.recording import write_json
    catalog=json.loads(CATALOG.read_text());entry=catalog['entries'][name]
    inputs={}
    for kind in ('scene','goal'):
        source=ROOT/entry[kind]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['input_sha256'][kind]:
            raise ValueError(f'{kind} differs from recorded Native inputs')
        inputs[kind]=json.loads(source.read_text())
    task=compile_task(inputs['scene'],inputs['goal'],planner_options=entry['planner_options'],seed=entry['seed'])
    receipt=validate_runtime(task,'tlmpm_v7',8,.0001,4,8,1.)
    if receipt is None or receipt['id']!=catalog['profile']:
        raise ValueError('Scene did not select the explicit research profile')
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    folder=out/'inputs';folder.mkdir()
    for kind in ('scene','goal'):write_json(folder/f'{kind}.json',inputs[kind])
    command=[str(ROOT/'.deps/isaacsim-venv/bin/python'),str(ROOT/'scripts/run_wp10_live.py'),
        '--runtime','isaac_sim','--scene',str(folder/'scene.json'),'--goal',str(folder/'goal.json'),
        '--planner-options',json.dumps(entry['planner_options']),'--controller','scripted',
        '--edge-feedback','measured','--hold-until-time-limit','--return-home','--record-cameras',
        '--overview-camera','--record-every','10','--oracle-stride','20',
        '--steps',str(round(task.duration_s/task.control_dt_s)),'--gpu',str(gpu),
        '--seed',str(entry['seed']),'--out',str(out/'run'),
        '--surface-profile',catalog['surface_profile'],'--surface-timing-log']
    record={'scene':name,'catalog':str(CATALOG.relative_to(ROOT)),'task_id':task.task_id,
        'profile':receipt,'source_inputs':entry,'command':command,
        'camera_rig':catalog['camera_rig'],'native_execution':'NOT_RUN',
        'scope':'Explicit CUT research scene; full Native plan preflight remains enabled. '
                'A new run is an engineering replay, not an independent adoption sample.'}
    write_json(out/'prepared.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','launch'))
    p.add_argument('--scene',choices=tuple(json.loads(CATALOG.read_text())['entries']),default='standard_cut')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();record=prepare(a.scene,a.out,a.gpu)
    if a.mode=='prepare':
        print(json.dumps(record,indent=2));return 0
    from robotic_cutting_sim.recording import write_json
    start=time.monotonic()
    with (a.out/'native.log').open('x') as log:
        code=subprocess.call(record['command'],cwd=ROOT,env={**os.environ,
            'PYTHONPATH':str(ROOT/'src'),'PYTHONUNBUFFERED':'1','OMNI_KIT_ACCEPT_EULA':'YES'},
            stdout=log,stderr=subprocess.STDOUT)
    write_json(a.out/'attempt.json',{'command':record['command'],'returncode':code,
        'wall_s':time.monotonic()-start,'result_path':str(a.out/'run/result.json'),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'ENGINEERING_SCENE_VIDEO_REFRESH; evaluate actual result separately'})
    return code


if __name__=='__main__':
    raise SystemExit(main())
