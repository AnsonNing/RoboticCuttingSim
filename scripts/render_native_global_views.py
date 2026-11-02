#!/usr/bin/env python3
"""Replay named fixed Global RGB views of one exact Native recording.

Record physics once with controlled v5/v8/v9 optics and --save-render-surfaces.
This command never steps physics and retains the original wrist RGB-D streams.
Narrow-lens native stereo depth remains unsupported by its existing guard.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from robotic_cutting_sim.camera_rig import GLOBAL_VIEW_PROFILES
from robotic_cutting_sim.demo_media import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--views',nargs='+',choices=tuple(GLOBAL_VIEW_PROFILES),default=list(GLOBAL_VIEW_PROFILES))
    parser.add_argument('--no-encode',action='store_true')
    parser.add_argument('--archived-native',action='store_true')
    parser.add_argument('--title',default='Three Global views and two wrist RGB-D cameras')
    args=parser.parse_args()
    if len(set(args.views))!=len(args.views):parser.error('Duplicate views')
    if not args.no_encode and len(args.views) not in (1,3):parser.error('Encode one view or all three')
    args.episode=args.episode.resolve();args.out=args.out.resolve()
    frames=json.loads((args.episode/'frame_manifest.json').read_text())
    for row in frames:
        path=args.episode/'surface_debug'/f"surface_{row['frame']:04d}.npz"
        if not path.is_file():raise ValueError(f'Missing exact surface {path}; position-only substitution refused')
    args.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();jobs=[]
    try:
        for view in args.views:
            output=args.out/view
            command=[str(ROOT/'.deps/isaacsim-venv/bin/python'),str(ROOT/'scripts/rerender_native_global.py'),
                str(args.episode),'--out',str(output),'--gpu',str(args.gpu),'--global-view',view]
            with (args.out/(view+'.log')).open('x') as log:
                subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMNI_KIT_ACCEPT_EULA':'YES'},stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt=output/'replay_receipt.json'
            if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='PASS_RENDER':
                raise RuntimeError(f'{view}: Native replay did not complete')
            jobs.append({'view':view,'command':command,'receipt':str(receipt),'sha256':sha256_file(receipt)})
        if not args.no_encode:
            command=[sys.executable,str(ROOT/'scripts/render_demo_episode.py'),str(args.episode),
                '--out',str(args.out/'movie'),'--layout','native_multiview_v2','--title',args.title,'--gif-speed','1']
            if len(args.views)==3:
                command+=['--global-replays',*[str(args.out/name) for name in GLOBAL_VIEW_PROFILES]]
            else:command+=['--global-replay',str(args.out/args.views[0])]
            if args.archived_native:command.append('--archived-native')
            with (args.out/'encode.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            jobs.append({'command':command,'receipt':str(args.out/'movie/render_receipt.json')})
        result={'status':'PASS','physics_runs':0,'views':args.views,'source_episode':str(args.episode),
            'jobs':jobs,'wall_s':time.perf_counter()-started,'scope':'Global RGB replay only; original real wrist RGB-D retained. No narrow native stereo-depth promotion.'}
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    except Exception:
        (args.out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
