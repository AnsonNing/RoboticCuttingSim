#!/usr/bin/env python3
"""Publish a genuine WP-14 scene replay and an exact captured workcell image."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from encode_wp13_media import encode_frames, preview
from run_wp13 import G


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--publish',type=Path,default=Path('docs/media/wp14'))
    p.add_argument('--out',type=Path,default=Path('docs/evidence/wp14/scene_refresh_001.json'))
    a=p.parse_args();episode=a.source/'run/episode_000'
    result=json.loads((episode/'result.json').read_text())
    material=json.loads((episode/'material.json').read_text())
    prepared=json.loads((a.source/'prepared.json').read_text())
    if result['runtime_source']!='live_isaac_sim' or result['task_status']!='PASS':
        raise ValueError('Expected completed actual Native task; preserve failed sources')
    if (material['solver_module']!='robotic_cutting_sim.tlmpm_v7' or
            material['research_profile']['configuration_sha256']!=prepared['profile']['configuration_sha256']):
        raise ValueError('Executed profile differs from prepared scene')
    a.publish.mkdir(exist_ok=True,parents=True)
    mp4=a.publish/'scene_walkthrough_native.mp4';gif=a.publish/'scene_walkthrough_preview.gif'
    image=a.publish/'scene_overview.png'
    if a.out.exists() or any(p.exists() for p in (mp4,gif,image)):
        raise FileExistsError('Use fresh output names; never overwrite media')
    files=sorted((episode/'observations').glob('overview_rgb_*.npz'))
    capture=encode_frames(files,mp4,'Live Native | WP14 CUT research scene | original v7 | 1x')
    command=preview(mp4,gif,start=3.,duration=7.,width=640,speed=1.)
    selected=min(files,key=lambda p:abs(float(np.load(p)['physics_timestamp_s'])-7.2))
    with np.load(selected) as data:
        timestamp=float(data['physics_timestamp_s'])
        if abs(timestamp-7.2)>1.e-8:
            raise ValueError('Declared 7.2 s scene frame was not captured')
        Image.fromarray(data['rgb']).save(image,optimize=True)
    record={'scope':'ENGINEERING_SCENE_VIDEO_REFRESH; not an additional independent adoption sample',
        'runtime_source':'live_isaac_sim','source':str(a.source),'prepared':prepared,
        'attempt':json.loads((a.source/'attempt.json').read_text()),'task_status':result['task_status'],
        'home_outcome':result['home_outcome'],'profile':material['research_profile'],
        'source_sha256':{name:G.sha(episode/name) for name in ('result.json','material.json','manifest.json','final_live_scene.usda')},
        'capture':capture,'image':{'path':str(image),'sha256':G.sha(image),'source_frame':str(selected),
            'source_frame_sha256':G.sha(selected),'physics_timestamp_s':timestamp,
            'scope':'Unmodified actual overview RGB capture; presentation camera is separate from policy observations'},
        'media':[{'gif':str(gif),'mp4':str(mp4),'runtime_source':'live_isaac_sim',
            'gif_sha256':G.sha(gif),'mp4_sha256':G.sha(mp4),'preview_speed':1.,'preview_interval_s':[3.,10.],
            'preview_command':command}],'full_stage':str(episode/'final_live_scene.usda')}
    G.write_new(a.out,record)
    print(json.dumps({'status':'ENCODED','task_status':result['task_status'],'image':str(image)}))


if __name__=='__main__':
    main()
