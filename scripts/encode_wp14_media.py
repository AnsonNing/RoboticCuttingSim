#!/usr/bin/env python3
"""Real Native frames only; matched capture times and explicit profile provenance."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from encode_wp13_media import encode_frames,preview
from run_wp13 import G


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--publish',type=Path,default=Path('docs/media/wp14'))
    p.add_argument('--out',type=Path,default=Path('docs/evidence/wp14/media_001.json'))
    a=p.parse_args();a.stage.mkdir(parents=True,exist_ok=False)
    records=[];comparisons=[]
    for case,label in (('C00_cut','standard_cut'),('H01_cut','firm_offset_cut')):
        close=[]
        for solver in ('tlmpm_v5','tlmpm_v7'):
            episode=a.native_root/solver/case/'run/episode_000'
            result=json.loads((episode/'result.json').read_text())
            material=json.loads((episode/'material.json').read_text())
            if result['runtime_source']!='live_isaac_sim' or material['solver_module']!='robotic_cutting_sim.'+solver:
                raise ValueError(f'Actual Native solver provenance mismatch: {episode}')
            profile=material.get('research_profile')
            if solver=='tlmpm_v7' and (profile or {}).get('id')!='v7_cut_research_v1':
                raise ValueError(f'Candidate Native recording did not execute the explicit profile: {episode}')
            short=solver.replace('tlmpm_','')
            title=f'Live Native | {short} | {label} | 1x | uncalibrated research'
            overview=sorted((episode/'observations').glob('overview_rgb_*.npz'))
            files=sorted((episode/'observations').glob('closeup_rgb_*.npz'))
            full_name=f'{short}_{label}_native.mp4'
            full=encode_frames(overview,a.stage/full_name,title)
            close_name=f'{short}_{label}_closeup.mp4'
            detail=encode_frames(files,a.stage/close_name,title,view='closeup_rgb')
            close.append((close_name,detail))
            records.append({'path':str(a.publish/full_name),'source_episode':str(episode),
                'runtime_source':result['runtime_source'],'task_id':result['task_id'],'task_status':result['task_status'],
                'solver':solver,'profile':profile,'material_seed':42,
                'surface_profile':'particle_surface_v1','capture':full,
                'source_sha256':{name:G.sha(episode/name) for name in ('result.json','material.json','manifest.json')},
                'sha256':G.sha(a.stage/full_name)})
        left,right=close
        ta,tb=left[1]['capture_time_s'],right[1]['capture_time_s']
        if len(ta)!=len(tb) or not np.allclose(ta,tb,atol=1.e-8,rtol=0):
            raise ValueError(f'Actual capture timestamps do not pair: {case}')
        name=f'{label}_v5_v7_comparison.mp4'
        command=['ffmpeg','-v','error','-n','-i',str(a.stage/left[0]),'-i',str(a.stage/right[0]),
                 '-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-crf','24',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(a.stage/name)]
        subprocess.run(command,check=True)
        gif_name=f'{label}_v5_v7_preview.gif'
        gif_command=preview(a.stage/name,a.stage/gif_name,start=3.,duration=7.,width=960,speed=1.)
        comparisons.append({'case':case,'mp4':str(a.publish/name),'gif':str(a.publish/gif_name),
            'left':'v5 default baseline','right':'v7_cut_research_v1 (original tlmpm_v7, not v7_1)',
            'matched_physics_timestamps_s':ta,'sources':[left[1],right[1]],
            'comparison_command':command,'preview_command':gif_command,'preview_speed':1.,
            'preview_interval_s':[3.,10.],'mp4_sha256':G.sha(a.stage/name),'gif_sha256':G.sha(a.stage/gif_name)})
    published=[Path(r['path']).name for r in records]
    published += [Path(c[k]).name for c in comparisons for k in ('mp4','gif')]
    a.publish.mkdir(parents=True,exist_ok=False)
    for name in published:
        source=a.stage/name
        if source.suffix=='.mp4':
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(source),'-f','null','-'],check=True)
        shutil.copyfile(source,a.publish/name)
    G.write_new(a.out,{'records':records,'comparisons':comparisons,'full_decode':'PASS',
        'scope':'Every full clip uses all genuine captured Native RGB frames at nominal 5 fps. '
        'Comparisons require identical capture timestamps. Presentation close-up cameras are not policy observations. '
        'No generated images, geometry animation, smoothing, frame synthesis, temporal shift or solver substitution. '
        'Task integrity/outcome/quality and HOME are assessed independently in native_001.json.'})


if __name__=='__main__':
    main()
