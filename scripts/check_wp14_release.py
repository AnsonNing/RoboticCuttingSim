#!/usr/bin/env python3
"""Read-only tracked links, real animation/full decode and protected evidence."""
import argparse
import hashlib
import json
import posixpath
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote,urlsplit

from check_wp12_release import ROOT,git,headings,animated_gif_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protected',type=Path,required=True)
    p.add_argument('--followup',action='store_true')
    a=p.parse_args();tracked=set(git('ls-files','-z').decode().rstrip('\0').split('\0'))
    pages=['README.md','docs/README.md','docs/WP14_PLAN.md','docs/reports/WP14_RESULTS.md','docs/WP14_SCENES.md']
    if a.followup:
        pages=['README.md','docs/README.md','docs/V7_RESEARCH_WORKFLOW.md',
               'docs/NATIVE_MULTIVIEW_GALLERY.md','docs/reports/WP14_FOLLOWUP_RESULTS.md','docs/WP14_SCENES.md']
    errors=[];links=[];media={}
    for page in pages:
        text=(ROOT/page).read_text();text=re.sub(r'^```.*?^```\s*$','',text,flags=re.M|re.S)
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text):
            url=urlsplit(target.strip('<>'))
            if url.scheme or url.netloc:continue
            path=posixpath.normpath(posixpath.join(posixpath.dirname(page),unquote(url.path))) if url.path else page
            links.append({'source':page,'target':target,'resolved':path})
            if path not in tracked and not any(t.startswith(path.rstrip('/')+'/') for t in tracked):
                errors.append({'source':page,'target':target,'error':'not tracked'});continue
            if path not in tracked:continue
            if url.fragment and path.endswith('.md') and unquote(url.fragment) not in headings((ROOT/path).read_text()):
                errors.append({'source':page,'target':target,'error':'missing heading'})
            if path in media:continue
            if path.endswith('.gif'):
                try:media[path]=animated_gif_metadata((ROOT/path).read_bytes())
                except Exception as error:errors.append({'path':path,'error':str(error)})
            elif path.endswith('.mp4'):
                result=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(ROOT/path),'-f','null','-'],capture_output=True,text=True)
                media[path]={'full_decode':'PASS' if result.returncode==0 else 'FAIL',
                    'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
                if result.returncode:errors.append({'path':path,'error':result.stderr[-500:]})
            elif path.endswith('.png'):
                from PIL import Image
                try:
                    with Image.open(ROOT/path) as picture:
                        size=picture.size;picture.verify()
                    media[path]={'decode':'PASS','size':size,
                        'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
                except Exception as error:errors.append({'path':path,'error':str(error)})
    provenance={}
    for file,key in (('docs/evidence/wp13/media_001.json','media'),('docs/evidence/wp14/media_001.json','comparisons'),
                     ('docs/evidence/wp14/scene_refresh_001.json','media')):
        for row in json.loads((ROOT/file).read_text())[key]:provenance[row['gif']]=row
    if a.followup:
        for filename in ('current_gallery_003.json','archive_gallery_003.json','two_bumps_media_002.json'):
            document=ROOT/'docs/evidence/wp14_followup'/filename
            data=json.loads(document.read_text())
            for row in data['episodes']:
                if row['status']!='PASS':
                    errors.append({'path':str(document),'error':'remaster refused','episode':row['id']})
                    continue
                receipt=row['receipt']
                for filename,expected in row['published'].items():
                    if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=expected:
                        errors.append({'path':filename,'error':'published bytes differ from camera receipt'})
                    if filename.endswith('.gif'):
                        mp4=next(p for p in row['published'] if p.endswith('.mp4'))
                        provenance[filename]={'runtime_source':receipt['runtime_source'],'gif_sha256':expected,
                                              'mp4':mp4,'mp4_sha256':row['published'][mp4]}
                for filename,expected in receipt.get('implementation_sha256',{}).items():
                    relative=str(Path(filename).relative_to(ROOT))
                    source=git('show',receipt['source_commit']+':'+relative)
                    if hashlib.sha256(source).hexdigest()!=expected:
                        errors.append({'path':relative,'error':'renderer source differs from recorded commit'})
                if receipt['sync_status']!='PASS' or receipt['full_decode']!='PASS' or receipt.get('renderer_source_status'):
                    errors.append({'path':str(document),'error':'camera sync/decode/source gate failed','episode':row['id']})
            for row in data['comparisons']:
                prefix='docs/media/wp14_followup/current_v3/'+row['id']
                provenance[prefix+'.gif']={'runtime_source':'live_isaac_sim','gif_sha256':row['gif_sha256'],
                                          'mp4':prefix+'.mp4','mp4_sha256':row['mp4_sha256']}
        physics=json.loads((ROOT/'docs/evidence/wp14_followup/native_detachment_001.json').read_text())
        native=json.loads((ROOT/'docs/evidence/wp14_followup/native_001.json').read_text())
        candidate=next(r for r in native['rows'] if r['label']=='bounded_rigid_v2')
        if (physics['status']!='PASS' or candidate['task_status']!='PASS' or
                candidate['home_outcome']['status']!='PASS' or
                candidate['solver_module']!='robotic_cutting_sim.tlmpm_v7_2'):
            errors.append({'error':'requested Native two-bump physical gate failed'})
        camera=json.loads((ROOT/'docs/evidence/wp14_followup/visibility_review_001.json').read_text())
        if camera['status']!='PASS_VISUAL_REVIEW':errors.append({'error':'actual eight-phase view review incomplete'})
    for path,item in media.items():
        if path.endswith('.gif'):
            row=provenance.get(path)
            if row is None or row['runtime_source']!='live_isaac_sim':errors.append({'path':path,'error':'no Native provenance'})
            elif (item['sha256']!=row['gif_sha256'] or hashlib.sha256((ROOT/row['mp4']).read_bytes()).hexdigest()!=row['mp4_sha256']):
                errors.append({'path':path,'error':'media hash differs from provenance'})
    scene_image=json.loads((ROOT/'docs/evidence/wp14/scene_refresh_001.json').read_text())['image']
    if hashlib.sha256((ROOT/scene_image['path']).read_bytes()).hexdigest()!=scene_image['sha256']:
        errors.append({'path':scene_image['path'],'error':'scene image differs from actual capture provenance'})
    baseline=json.loads(a.protected.read_text());changed=[]
    for path,expected in baseline['blobs'].items():
        if git('hash-object','--',path).decode().strip()!=expected:changed.append(path)
    if changed:errors.append({'error':'protected blobs changed','paths':changed})
    report={'status':'FAIL' if errors else 'PASS','head':git('rev-parse','HEAD').decode().strip(),
        'scope':'ENGINEERING_PUBLICATION_ONLY','links':links,'media':media,
        'protected':{'baseline':baseline['baseline'],'checked':len(baseline['blobs']),'changed':changed},'errors':errors}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'links':len(links),'media':len(media),'protected':len(baseline['blobs']),'errors':errors}))
    return bool(errors)


if __name__=='__main__':
    raise SystemExit(main())
