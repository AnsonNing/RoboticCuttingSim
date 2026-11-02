"""Read-only three-Global RGB / original wrist RGB-D publication checks."""
import hashlib
import json
import posixpath
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote,urlsplit

import numpy as np
from check_wp12_release import ROOT,git,headings,animated_gif_metadata


def check_three_global(out,protected):
    prefix='docs/media/three_global_v1/'
    evidence=ROOT/'docs/evidence/three_global'
    errors=[];links=[];media={};verified=set();hashes={}
    def digest(path):
        path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        if key not in hashes:hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[key]
    def require(condition,message,**context):
        if not condition:errors.append({'error':message,**context})
    tracked=set(git('ls-files','-z').decode().rstrip('\0').split('\0'))
    pages=['README.md','docs/README.md','docs/GLOBAL_CAMERA_VIEWS.md','docs/V7_RESEARCH_WORKFLOW.md',
           'docs/NATIVE_MULTIVIEW_GALLERY.md','docs/reports/THREE_GLOBAL_CAMERA_REFRESH.md','docs/WP14_SCENES.md']
    for page in pages:
        source=re.sub(r'^```.*?^```\s*$','',(ROOT/page).read_text(),flags=re.M|re.S)
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^\"]*")?\)',source):
            url=urlsplit(target.strip('<>'))
            if url.scheme or url.netloc:continue
            name=posixpath.normpath(posixpath.join(posixpath.dirname(page),unquote(url.path))) if url.path else page
            links.append({'source':page,'target':target,'resolved':name})
            require(name in tracked or any(p.startswith(name.rstrip('/')+'/') for p in tracked),'untracked link',path=name)
            if name not in tracked:continue
            if url.fragment and name.endswith('.md'):
                require(unquote(url.fragment) in headings((ROOT/name).read_text()),'missing heading',path=name)
            if not name.startswith(prefix) or name in media:continue
            path=ROOT/name
            if name.endswith('.gif'):media[name]=animated_gif_metadata(path.read_bytes())
            elif name.endswith('.mp4'):
                decoded=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True,text=True)
                require(decoded.returncode==0,'full decode failed',path=name)
                media[name]={'sha256':digest(path),'full_decode':'PASS' if decoded.returncode==0 else 'FAIL'}
            elif name.endswith('.png'):
                from PIL import Image
                with Image.open(path) as im:im.verify()
                media[name]={'sha256':digest(path),'decode':'PASS'}
    require(sum(p.endswith('.mp4') for p in media)==47,'not all47 full movies linked')
    require(sum(p.endswith('.gif') for p in media)==8,'not all8 README animations linked')
    catalog=json.loads((evidence/'media_001.json').read_text());published={};sources=set()
    require(len(catalog['episodes'])==44 and len(catalog['comparisons'])==3,'incomplete movie catalog')
    profiles={'left':'wrist_camera_rig_v10','right':'wrist_camera_rig_v11','opposite_tool':'wrist_camera_rig_v14'}
    for row in catalog['episodes']:
        receipt=row['receipt'];sources.add(str(Path(row['source_episode']).resolve()))
        require(row['status']=='PASS' and receipt['sync_status']=='PASS' and receipt['full_decode']=='PASS','media sync/decode failed',episode=row['id'])
        require(receipt['layout']=='native_multiview_three_global_v1' and len(receipt['panes'])==7,'missing three Global RGB / four wrist panes',episode=row['id'])
        require(receipt['encoder_simulator_runs']==0 and receipt['sync_kind']=='same_measured_state_time_with_global_camera_replay','replay source mislabelled',episode=row['id'])
        require(not receipt.get('renderer_source_status'),'uncommitted renderer source',episode=row['id'])
        require(set(receipt.get('three_global_views',{}))==set(profiles),'missing named external views',episode=row['id'])
        for filename,expected in receipt['implementation_sha256'].items():
            relative=str(Path(filename).relative_to(ROOT))
            require(hashlib.sha256(git('show',receipt['source_commit']+':'+relative)).hexdigest()==expected,'renderer commit mismatch',path=relative)
        for filename,expected in receipt['source_sha256'].items():require(digest(filename)==expected,'changed recording input',path=filename)
        for filename,expected in row['published'].items():
            published[filename]=expected;require(digest(ROOT/filename)==expected,'changed published media',path=filename)
        for name,summary in receipt['three_global_views'].items():
            path=Path(summary['receipt_path'])
            require(digest(path)==summary['receipt_sha256'],'changed camera receipt',view=name)
            if str(path) in verified:continue
            verified.add(str(path));replay=json.loads(path.read_text());profile=replay['profile']
            require(hashlib.sha256(git('show',replay['source_commit']+':scripts/rerender_native_global.py')).hexdigest()==replay['implementation_sha256'],
                    'replay source differs from producer commit',view=name)
            require(digest(ROOT/'configs'/(profiles[name]+'.json'))==replay['profile_sha256'],'replay profile changed',view=name)
            require(replay['status']=='PASS_RENDER' and replay['physics_runs']==0 and not replay['wrist_captures_modified'],'incorrect replay provenance',view=name)
            require(replay.get('published_modalities')==['rgb'],'Global depth falsely promoted',view=name)
            require(profile['rig_profile']==profiles[name],'wrong named profile',view=name)
            transform=np.asarray(replay['calibration']['T_world_camera']);target=np.asarray(profile['look_at_target_world_m'])
            direction=transform[:3,3]-target;distance=float(np.linalg.norm(direction))
            elevation=float(np.degrees(np.arcsin(direction[2]/distance)))
            require(abs(distance-.6)<1e-6 and abs(elevation-17.123944951922702)<.001,'wrong camera distance/elevation',view=name)
            require(np.allclose(replay['calibration']['K'],[[1505.4816,0,320],[0,1505.4816,240],[0,0,1]],atol=.002,rtol=0),'wrong actual lens',view=name)
            source=Path(replay['source_episode']);frames=json.loads((source/'frame_manifest.json').read_text());originals={f['frame']:f for f in frames}
            require([f['frame'] for f in replay['frames']]==[f['frame'] for f in frames],'truncated replay timeline',view=name)
            for filename,expected in replay['source_sha256'].items():require(digest(filename)==expected,'changed exact replay input',path=filename)
            for frame in replay['frames']:
                index=frame['frame'];mesh=source/'surface_debug'/f'surface_{index:04d}.npz'
                with np.load(mesh) as data:
                    expected=str(data['source_surface_sha256'].item()) if 'source_surface_sha256' in data else originals[index]['surface']['surface_sha256']
                require(frame['staged']['surface_sha256']==expected and frame['maximum_rendered_body_matrix_error']<=1e-5,'mesh/body differs from measured Native state',view=name,frame=index)
                require(abs(frame['time_s']-originals[index]['time_s'])<1e-7,'changed capture time',view=name,frame=index)
                require(frame['render_clipping_range_m'][0]<=.00101,'foreground near-clipped',view=name,frame=index)
                require(frame['valid_depth_fraction']==0.,'unsupported Global depth reported valid',view=name,frame=index)
                if name=='opposite_tool':
                    readback=frame['tool_opposite_readback']
                    require(readback['horizontal_normal_incidence_error_deg']<.001 and abs(readback['normal_incidence_error_deg']-17.123944951922702)<.001,
                            'incorrect horizontal/3D opposite-tool incidence',frame=index)
    require(len(sources)==37,'wrong unique source count; aliases or missing sources not reconciled')
    require(len(verified)==111,'not all37 source streams have three replay receipts')
    for row in catalog['comparisons']:
        require(row['status']=='PASS' and row['pts_check']=='PASS' and row['full_decode']=='PASS','comparison decode/PTS failed',episode=row['id'])
        for extension,key in [('.mp4','mp4_sha256'),('.gif','gif_sha256')]:published[prefix+row['id']+extension]=row[key]
    for name,item in media.items():
        if name.endswith(('.gif','.mp4')):require(item['sha256']==published.get(name),'unbound movie/animation',path=name)
    physics=json.loads((evidence/'native_two_bumps_detachment_001.json').read_text())
    require(physics['status']=='PASS' and all(physics['checks'].values()),'two-bump physical proof failed')
    hero=next(r for r in catalog['episodes'] if r['id']=='two_bumps_v7_2_native')
    actual=json.loads((Path(hero['source_episode'])/'result.json').read_text())
    require(actual['task_status']=='PASS' and actual['home_outcome']['status']=='PASS','hero task/HOME failed')
    control=json.loads((evidence/'narrow_controls_001.json').read_text())
    require(control['status']=='BLOCKED' and not control['source_correspondence_candidate_promoted'],'narrow stereo failure relabelled')
    visibility=json.loads((evidence/'visibility_review_001.json').read_text())
    require(visibility['status']=='PASS_VISUAL_REVIEW','selected Native phase visibility review missing')
    require(len(visibility['frames'])==8,'missing selected Native cutting/post-cut review frames')
    for frame in visibility['frames']:
        require(digest(ROOT/frame['path'])==frame['sha256'],'changed selected Native image',frame=frame['frame'])
        for path,expected in frame['source_sha256'].items():require(digest(path)==expected,'changed selected Native preview input',path=path)
    scene=json.loads((evidence/'scene_001.json').read_text());require(digest(ROOT/scene['path'])==scene['sha256'],'scene provenance changed')
    baseline=json.loads(Path(protected).read_text());changed=[p for p,h in baseline['blobs'].items() if git('hash-object','--',p).decode().strip()!=h]
    require(not changed,'protected evidence changed',paths=changed)
    unstaged=git('diff','--name-only').decode().splitlines()
    require(not unstaged,'audit requires working files to match the staged payload',paths=unstaged)
    report={'status':'FAIL' if errors else 'PASS','head':git('rev-parse','HEAD').decode().strip(),
        'index_tree':git('write-tree').decode().strip(),'unstaged_tracked_changes':unstaged,
        'scope':'ENGINEERING_CAMERA_PUBLICATION_ONLY','links':links,'media':media,'unique_sources':len(sources),
        'replay_receipts':len(verified),'protected':{'baseline':baseline['baseline'],'checked':len(baseline['blobs']),'changed':changed},'errors':errors}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with Path(out).open('x') as file:json.dump(report,file,indent=2);file.write('\n')
    print(json.dumps({k:report[k] for k in ['status','unique_sources','replay_receipts','errors']}))
    return bool(errors)
