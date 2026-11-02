"""Read-only publication checks for the physical 0.60 m camera refresh."""
import hashlib
import json
import posixpath
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from check_wp12_release import ROOT, git, headings, animated_gif_metadata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_global60(out, protected):
    errors, links, media = [], [], {}
    pages=['README.md','docs/README.md','docs/V7_RESEARCH_WORKFLOW.md',
           'docs/NATIVE_MULTIVIEW_GALLERY.md','docs/reports/GLOBAL60_CAMERA_REFRESH.md','docs/WP14_SCENES.md']
    tracked=set(git('ls-files','-z').decode().rstrip('\0').split('\0'))
    for page in pages:
        source=re.sub(r'^```.*?^```\s*$','',(ROOT/page).read_text(),flags=re.M|re.S)
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)',source):
            url=urlsplit(target.strip('<>'))
            if url.scheme or url.netloc:continue
            path=posixpath.normpath(posixpath.join(posixpath.dirname(page),unquote(url.path))) if url.path else page
            links.append({'source':page,'target':target,'resolved':path})
            if path not in tracked and not any(n.startswith(path.rstrip('/')+'/') for n in tracked):
                errors.append({'path':path,'error':'not tracked'});continue
            if path not in tracked:continue
            if url.fragment and path.endswith('.md') and unquote(url.fragment) not in headings((ROOT/path).read_text()):
                errors.append({'path':path,'error':'missing heading'})
            if not path.startswith('docs/media/global60_v1/') or path in media:continue
            if path.endswith('.gif'):
                media[path]=animated_gif_metadata((ROOT/path).read_bytes())
            elif path.endswith('.mp4'):
                decoded=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(ROOT/path),'-f','null','-'],capture_output=True,text=True)
                media[path]={'sha256':digest(ROOT/path),'full_decode':'PASS' if decoded.returncode==0 else 'FAIL'}
                if decoded.returncode:errors.append({'path':path,'error':'full decode failed'})
            elif path.endswith('.png'):
                from PIL import Image
                with Image.open(ROOT/path) as im:im.verify()
                media[path]={'sha256':digest(ROOT/path),'decode':'PASS'}
    catalog=json.loads((ROOT/'docs/evidence/global60/media_001.json').read_text())
    if len(catalog['episodes'])!=44 or len(catalog['comparisons'])!=3:
        errors.append({'error':'incomplete 44 individual / 3 comparison movie catalog'})
    if sum(p.endswith('.mp4') for p in media)!=47 or sum(p.endswith('.gif') for p in media)!=8:
        errors.append({'error':'README/gallery do not expose all 47 MP4s and eight animations'})
    published={}
    origins={}
    for row in catalog['episodes']:
        receipt=row.get('receipt',{})
        if row['status']!='PASS' or receipt.get('full_decode')!='PASS' or receipt.get('sync_status')!='PASS':
            errors.append({'episode':row['id'],'error':'camera encoding/sync/decode failed'});continue
        mode=row['origin']['mode'];origins.setdefault(mode,set()).add(str(row['source_episode']))
        if abs(receipt.get('global_working_distance_m',0)-.60)>1e-6:
            errors.append({'episode':row['id'],'error':'Global working distance is not 0.60 m'})
        if receipt.get('renderer_source_status'):
            errors.append({'episode':row['id'],'error':'uncommitted renderer implementation'})
        for filename,expected in receipt['implementation_sha256'].items():
            relative=str(Path(filename).relative_to(ROOT))
            if hashlib.sha256(git('show',receipt['source_commit']+':'+relative)).hexdigest()!=expected:
                errors.append({'episode':row['id'],'error':'renderer source commit mismatch','path':relative})
        for filename,expected in row['published'].items():
            published[filename]=expected
            if digest(ROOT/filename)!=expected:errors.append({'path':filename,'error':'published bytes changed'})
        for filename,expected in receipt['source_sha256'].items():
            if digest(filename)!=expected:errors.append({'path':filename,'error':'original camera/recording input changed'})
        replay=receipt.get('global_camera_replay')
        if mode=='exact_surface_global_replay':
            if not replay or replay['physics_runs']!=0 or replay['runtime_source']!='native_camera_replay':
                errors.append({'episode':row['id'],'error':'incorrect replay provenance'});continue
            original=json.loads((Path(replay['source_episode'])/'frame_manifest.json').read_text())
            originals={r['frame']:r for r in original}
            for frame in replay['frames']:
                expected=originals[frame['frame']]['surface']['surface_sha256']
                if frame['staged']['surface_sha256']!=expected or frame['maximum_rendered_body_matrix_error']>1e-5:
                    errors.append({'episode':row['id'],'frame':frame['frame'],'error':'surface/body differs from original Native capture'})
            if receipt.get('sync_kind')!='same_measured_state_time_with_global_camera_replay':
                errors.append({'episode':row['id'],'error':'camera replay mislabeled as live simultaneous capture'})
        elif mode=='new_native_recording':
            if replay or receipt['runtime_source']!='live_isaac_sim':
                errors.append({'episode':row['id'],'error':'new recording is not genuine Native capture'})
        else:errors.append({'episode':row['id'],'error':'unsupported camera source'})
    if len(origins.get('exact_surface_global_replay',set()))!=13 or len(origins.get('new_native_recording',set()))!=26:
        errors.append({'error':'source inventory does not retain 13 exact-surface replays / 26 unique new recordings'})
    for row in catalog['comparisons']:
        if row['status']!='PASS':errors.append({'episode':row['id'],'error':'comparison encoding failed'})
        prefix='docs/media/global60_v1/'+row['id']
        for extension,key in [('.mp4','mp4_sha256'),('.gif','gif_sha256')]:
            published[prefix+extension]=row[key]
            if digest(ROOT/(prefix+extension))!=row[key]:errors.append({'path':prefix+extension,'error':'comparison bytes changed'})
    for path,item in media.items():
        if path.endswith(('.gif','.mp4')) and item['sha256']!=published.get(path):
            errors.append({'path':path,'error':'no matching exact camera publication receipt'})
    detachment=json.loads((ROOT/'docs/evidence/global60/native_two_bumps_detachment_001.json').read_text())
    if detachment['status']!='PASS' or not all(detachment['checks'].values()):
        errors.append({'error':'new two-bump demonstration lacks unchanged physical detachment proof'})
    hero=next(r for r in catalog['episodes'] if r['id']=='two_bumps_v7_2_native')
    hero_result=json.loads((Path(hero['source_episode'])/'result.json').read_text())
    if hero_result.get('task_status')!='PASS' or hero_result.get('home_outcome',{}).get('status')!='PASS':
        errors.append({'error':'new two-bump Native task or HOME failed'})
    visibility=json.loads((ROOT/'docs/evidence/global60/visibility_review_001.json').read_text())
    if visibility['status']!='PASS_VISUAL_REVIEW':errors.append({'error':'Global working-edge visibility review incomplete'})
    scene=json.loads((ROOT/'docs/evidence/global60/scene_001.json').read_text())
    if digest(ROOT/scene['path'])!=scene['sha256']:errors.append({'error':'scene image differs from Native source'})
    baseline=json.loads(Path(protected).read_text());changed=[]
    for path,expected in baseline['blobs'].items():
        if git('hash-object','--',path).decode().strip()!=expected:changed.append(path)
    if changed:errors.append({'error':'protected history/source/media changed','paths':changed})
    report={'status':'FAIL' if errors else 'PASS','head':git('rev-parse','HEAD').decode().strip(),
        'scope':'ENGINEERING_CAMERA_PUBLICATION_ONLY','links':links,'media':media,
        'origins':{k:len(v) for k,v in origins.items()},'protected':{'baseline':baseline['baseline'],
        'checked':len(baseline['blobs']),'changed':changed},'errors':errors}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with Path(out).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'links':len(links),'media':len(media),'protected':len(baseline['blobs']),'errors':errors}))
    return bool(errors)
