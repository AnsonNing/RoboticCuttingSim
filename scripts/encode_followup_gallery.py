#!/usr/bin/env python3
"""Remaster an explicit inventory through the existing Native multiview renderer.

No simulator launches. Every source, refusal, decode and camera audit is retained.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from robotic_cutting_sim.demo_media import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--publish',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--reuse-evidence',type=Path)
    a=p.parse_args()
    a.stage.mkdir(parents=True,exist_ok=False)
    a.publish.mkdir(parents=True,exist_ok=False)
    catalog=json.loads(a.catalog.read_text());receipts=[]
    previous={r['id']:r for r in json.loads(a.reuse_evidence.read_text())['episodes']} if a.reuse_evidence else {}
    for row in catalog['episodes']:
        source=Path(row['source_episode']);out=a.stage/row['id']
        old=previous.get(row['id'])
        if old and old['status']=='PASS' and old['source_episode']==row['source_episode']:
            outputs={}
            for path,expected in old['published'].items():
                if sha256_file(Path(path))!=expected:raise ValueError('Published reuse source changed')
                target=a.publish/Path(path).name
                with target.open('xb') as dest,Path(path).open('rb') as src:shutil.copyfileobj(src,dest)
                outputs[str(target)]=expected
            receipts.append({**old,'published':outputs,'reused_verified_evidence':str(a.reuse_evidence)})
            continue
        command=[sys.executable,'scripts/render_demo_episode.py',str(source),'--out',str(out),
            '--layout','native_multiview_v2','--title',row['title'],'--gif-speed',str(row.get('preview_speed',1.)),
            '--gif-width',str(row.get('preview_width',960))]
        if row.get('archived'):command.append('--archived-native')
        if row.get('global_pixel_crop'):
            command += ['--global-pixel-crop',*[str(v) for v in row['global_pixel_crop']]]
        with (a.stage/(row['id']+'.log')).open('x') as log:
            code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record={**row,'command':command,'returncode':code,'status':'PASS' if code==0 else 'BLOCKED'}
        if code==0:
            receipt=json.loads((out/'render_receipt.json').read_text())
            outputs={}
            names=['demo.mp4']+(['demo.gif'] if not row.get('archived') else [])
            for name in names:
                target=a.publish/(row['id']+Path(name).suffix)
                with target.open('xb') as dest,(out/name).open('rb') as src:shutil.copyfileobj(src,dest)
                outputs[str(target)]=sha256_file(target)
            record.update(receipt=receipt,published=outputs)
        else:record['failure_log']=str(a.stage/(row['id']+'.log'))
        receipts.append(record)
        (a.stage/'progress.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print(row['id'],record['status'],flush=True)
    # Paired full clips use all camera panes and identical captured times.
    comparisons=[]
    by_id={r['id']:r for r in receipts}
    for row in catalog.get('comparisons',[]):
        left,right=[by_id[k] for k in row['sources']]
        if left['status']!='PASS' or right['status']!='PASS':
            comparisons.append({**row,'status':'BLOCKED','reason':'source remaster failed'});continue
        import numpy as np
        ta,tb=left['receipt']['capture_time_s'],right['receipt']['capture_time_s']
        if len(ta)!=len(tb) or not np.allclose(ta,tb,atol=1e-8,rtol=0):
            raise ValueError('Comparison source physics timestamps differ; pairing refused')
        target=a.publish/(row['id']+'.mp4')
        command=['ffmpeg','-v','error','-n','-i',str(a.publish/(left['id']+'.mp4')),
            '-i',str(a.publish/(right['id']+'.mp4')),'-filter_complex',
            '[0:v]scale=1280:720[l];[1:v]scale=1280:720[r];[l][r]hstack=inputs=2[v]',
            '-map','[v]','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        subprocess.run(command,check=True)
        from encode_wp13_media import preview
        gif=a.publish/(row['id']+'.gif')
        preview_command=preview(target,gif,start=0,duration=len(ta)/5,width=1280,speed=1.)
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(target),'-f','null','-'],check=True)
        comparisons.append({**row,'status':'PASS','capture_time_s':ta,'command':command,
                            'preview_command':preview_command,'mp4_sha256':sha256_file(target),'gif_sha256':sha256_file(gif)})
    payload={'schema':'wp14_followup_gallery_v1','catalog_sha256':sha256_file(a.catalog),
        'simulator_runs':0,'scope':'Remastered actual saved Native cameras. Prior task evaluations and failures unchanged.',
        'episodes':receipts,'comparisons':comparisons,'archive_inventory':catalog.get('archive_inventory',[])}
    with a.evidence.open('x') as f:f.write(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':main()
