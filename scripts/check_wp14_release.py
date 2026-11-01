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
    a=p.parse_args();tracked=set(git('ls-files','-z').decode().rstrip('\0').split('\0'))
    pages=['README.md','docs/README.md','docs/WP14_PLAN.md','docs/reports/WP14_RESULTS.md','docs/WP14_SCENES.md']
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
    provenance={}
    for file,key in (('docs/evidence/wp13/media_001.json','media'),('docs/evidence/wp14/media_001.json','comparisons'),
                     ('docs/evidence/wp14/scene_refresh_001.json','media')):
        for row in json.loads((ROOT/file).read_text())[key]:provenance[row['gif']]=row
    for path,item in media.items():
        if path.endswith('.gif'):
            row=provenance.get(path)
            if row is None or row['runtime_source']!='live_isaac_sim':errors.append({'path':path,'error':'no Native provenance'})
            elif (item['sha256']!=row['gif_sha256'] or hashlib.sha256((ROOT/row['mp4']).read_bytes()).hexdigest()!=row['mp4_sha256']):
                errors.append({'path':path,'error':'media hash differs from provenance'})
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
