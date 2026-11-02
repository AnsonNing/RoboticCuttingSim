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


def prepare(name, out, gpu=0, camera_rig_profile=None, global_view=None):
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.research_profiles import validate_runtime
    from robotic_cutting_sim.recording import write_json
    catalog=json.loads(CATALOG.read_text());entry=catalog['entries'][name]
    display_views=[]
    if global_view is not None:
        from robotic_cutting_sim.camera_rig import GLOBAL_VIEW_PROFILES
        if global_view!='all' and global_view not in GLOBAL_VIEW_PROFILES:raise ValueError('Unknown named Global view')
        display_views=list(GLOBAL_VIEW_PROFILES) if global_view=='all' else [global_view]
        camera_rig_profile=camera_rig_profile or 'wrist_camera_rig_v5'
    if camera_rig_profile is not None:
        from robotic_cutting_sim.camera_rig import RIG_PROFILES
        if camera_rig_profile not in RIG_PROFILES:raise ValueError('Unknown camera rig profile')
        if camera_rig_profile in ('wrist_camera_rig_v10','wrist_camera_rig_v11','wrist_camera_rig_v13','wrist_camera_rig_v14'):
            raise NotImplementedError('Use controlled recording optics with --global-view for subsequent Global RGB replay; narrow Native stereo controls remain BLOCKED')
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
    if camera_rig_profile is not None:
        command+=['--camera-rig-profile',camera_rig_profile]
    if display_views:command+=['--save-render-surfaces']
    record={'scene':name,'catalog':str(CATALOG.relative_to(ROOT)),'task_id':task.task_id,
        'profile':receipt,'source_inputs':entry,'command':command,
        'camera_rig':camera_rig_profile or catalog['camera_rig'],'native_execution':'NOT_RUN',
        'scope':'Explicit CUT research scene; full Native plan preflight remains enabled. '
                'A new run is an engineering replay, not an independent adoption sample.'}
    if display_views:
        record['display_global_views']=display_views
        record['display_global_profiles']={name:GLOBAL_VIEW_PROFILES[name] for name in display_views}
        record['display_command']=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/render_native_global_views.py'),
            str(out/'run/episode_000'),'--out',str(out/'display'),'--gpu',str(gpu),'--views',*display_views]
        record['display_scope']='Record physics once with controlled Global sensor; replay exact meshes/poses for Global RGB only; original wrist RGB-D unchanged'
    write_json(out/'prepared.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','launch'))
    p.add_argument('--scene',choices=tuple(json.loads(CATALOG.read_text())['entries']),default='standard_cut')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    from robotic_cutting_sim.camera_rig import RIG_PROFILES
    p.add_argument('--camera-rig-profile',choices=RIG_PROFILES)
    from robotic_cutting_sim.camera_rig import GLOBAL_VIEW_PROFILES
    p.add_argument('--global-view',choices=(*GLOBAL_VIEW_PROFILES,'all'),help='Display RGB viewpoint(s) after one controlled-optics recording')
    a=p.parse_args();record=prepare(a.scene,a.out,a.gpu,a.camera_rig_profile,a.global_view)
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
    if code==0 and record.get('display_command'):
        if not (a.out/'run/episode_000/frame_manifest.json').exists():
            raise RuntimeError('Native process exited without a completed recording; display refused')
        with (a.out/'display.log').open('x') as log:
            code=subprocess.call(record['display_command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    return code


if __name__=='__main__':
    raise SystemExit(main())
