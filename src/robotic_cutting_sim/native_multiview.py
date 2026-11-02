"""Camera-only v2 layout of the existing recorded-episode renderer/encoder.

Every pane is a measured camera of one Native episode. Missing streams or
mismatched clocks/extrinsics refuse publication; no camera substitution occurs.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import demo_media as dm

CAMERAS=('global_d455','wrist_d405_0','wrist_d405_1')
CLOCKS=('physics_timestamp_s','surface_timestamp_s','camera_extrinsics_timestamp_s')
LAYOUT='native_multiview_v2'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
EXTERNAL_VIEWS=('left','right','opposite_tool')


def three_global_canvas(frames, external, solver, phase, status, title, surface, archived):
    """Three uncut full Global frames above the two actual wrist RGB/depth pairs."""
    im=Image.new('RGB',(1920,1080),(17,19,24));draw=ImageDraw.Draw(im)
    font=ImageFont.truetype(FONT,18);large=ImageFont.truetype(FONT,24)
    t=float(frames['global_d455']['physics_timestamp_s'])
    draw.text((15,8),f'{title} | Native Isaac Sim | {solver} | {surface}',font=large,fill='white')
    draw.text((15,40),f'sim {t:.2f}s | {phase} | task {status} | 1x | exact saved-state Global replay + original Native wrists',font=font,fill='#e5dbab')
    def pane(frame,box,label,depth=False):
        x,y,w,h=box
        draw.text((x+8,y+4),label,font=font,fill='white')
        draw.text((x+8,y+27),f'{t:.2f}s | {phase}',font=font,fill='#bbc3d0')
        if frame is None:
            draw.text((x+8,y+65),'NO RAW CAPTURE (historical gap)',font=font,fill='#e5dbab');return
        pixels=depth_rgb(frame) if depth else frame['rgb'][...,:3]
        pic=ImageOps.contain(Image.fromarray(np.asarray(pixels,np.uint8)),(w,h-54),Image.Resampling.LANCZOS)
        im.paste(pic,(x+(w-pic.width)//2,y+54+(h-54-pic.height)//2))
    for i,name in enumerate(EXTERNAL_VIEWS):
        pane(external[name],(i*640,75,640,570),f'Global {name.upper()} RGB | '+('rig14 / normal bearing +17deg' if i==2 else f'rig{10+i} / low side view'))
    for i,(name,depth) in enumerate([('wrist_d405_0',False),('wrist_d405_1',False),('wrist_d405_0',True),('wrist_d405_1',True)]):
        pane(frames[name],(i*480,655,480,365),name+(' Depth 70-500mm' if depth else ' RGB'),depth)
    scope='Historical wrist calibration retained.' if archived else 'Fixed hand-local v5 wrist mounts.'
    draw.text((15,1048),'All external views and real occluders retained; no crop. '+scope+' Sensor models provisional.',font=font,fill='#bbc3d0')
    return np.asarray(im)


def require_external_replays(paths,episode,indices):
    from .camera_rig import GLOBAL_VIEW_PROFILES
    if len(paths)!=3:raise ValueError('All three named external views are required')
    result={}
    for name,path in zip(EXTERNAL_VIEWS,paths):
        path=Path(path).resolve();receipt=json.loads((path/'replay_receipt.json').read_text())
        if receipt.get('status')!='PASS_RENDER' or Path(receipt['source_episode']).resolve()!=episode.resolve():
            raise ValueError(f'{name}: incomplete replay or wrong Native episode')
        if receipt['profile']['rig_profile']!=GLOBAL_VIEW_PROFILES[name]:
            raise ValueError(f'{name}: wrong camera profile')
        if [r['frame'] for r in receipt['frames']]!=indices:
            raise ValueError(f'{name}: replay does not cover complete capture timeline')
        if receipt.get('physics_runs')!=0 or receipt.get('wrist_captures_modified'):
            raise ValueError(f'{name}: replay changed physics or wrist captures')
        for row in receipt['frames']:
            if row.get('render_clipping_range_m',[1])[0]>.00101:
                raise ValueError(f'{name}: missing actual small render-near readback')
            capture=path/'observations'/f"global_d455_{row['frame']:04d}.npz"
            if dm.sha256_file(capture)!=row['capture_sha256']:raise ValueError(f'{name}: changed capture bytes')
        for source,expected in receipt['source_sha256'].items():
            if dm.sha256_file(Path(source))!=expected:raise ValueError(f'{name}: changed measured replay inputs')
        result[name]=(path,receipt)
    return result


def read_set(observations, index, *, allow_missing=False):
    result={}
    for name in CAMERAS:
        path=observations/f'{name}_{index:04d}.npz'
        if allow_missing and not path.exists():
            result[name]=None
            continue
        with np.load(path) as source:
            result[name]={k:source[k] for k in source.files}
    return result


def check_set(frames, initial=None, *, archived=False):
    """Check clock, actual camera identity, fixed Global and measured hand mount."""
    t=float(frames[CAMERAS[0]]['physics_timestamp_s'])
    native=float(frames[CAMERAS[0]]['native_physics_timestamp_s'])
    fractions={}
    for name in CAMERAS:
        f=frames[name]
        if f is None:
            if not archived:raise ValueError(f'{name}: missing current camera capture')
            fractions[name]=None
            continue
        if str(f['camera_id'].item())!=name or 'Isaac Sim' not in str(f['live_source'].item()):
            raise ValueError(f'{name}: incorrect camera identity or Native provenance')
        if any(abs(float(f[key])-t)>1e-7 for key in CLOCKS):
            raise ValueError(f'{name}: physics/surface/extrinsics clock mismatch')
        if abs(float(f['native_physics_timestamp_s'])-native)>1e-7:
            raise ValueError(f'{name}: Native clocks differ across cameras')
        if initial is not None:
            if not np.allclose(f['K'],initial[name]['K'],atol=1e-6,rtol=0):
                raise ValueError(f'{name}: changing intrinsics')
            if name=='global_d455' and not np.allclose(f['T_world_camera'],initial[name]['T_world_camera'],atol=1e-5,rtol=0):
                raise ValueError('Global camera moved')
        if name.startswith('wrist'):
            mount=np.asarray(f['T_hand_camera'])
            if not archived and (not np.allclose(mount[:3,:3],np.eye(3),atol=1e-8,rtol=0) or not np.allclose(mount[:3,3],[-.085,.045,.020],atol=1e-8,rtol=0)):
                raise ValueError(f'{name}: v5 hand-local calibration changed')
            if initial is not None and not np.allclose(mount,initial[name]['T_hand_camera'],atol=1e-8,rtol=0):
                raise ValueError(f'{name}: hand-local mount changed during recording')
            if not np.allclose(f['T_world_hand']@mount,f['T_world_camera'],atol=1e-5,rtol=0):
                raise ValueError(f'{name}: camera does not follow measured hand')
        depth=np.asarray(f['depth_m']).squeeze();valid=np.asarray(f['valid_mask']).squeeze().astype(bool)
        if f['rgb'].shape[:2]!=depth.shape or valid.shape!=depth.shape:
            raise ValueError(f'{name}: RGB/depth dimensions differ')
        if not np.isfinite(depth[valid]).all():raise ValueError(f'{name}: invalid finite-depth mask')
        fractions[name]=float(valid.mean())
    return {'time_s':t,'native_time_s':native,'valid_fraction':fractions,
            'missing_cameras':[n for n in CAMERAS if frames[n] is None]}


def read_closeup(observations, index, time_s):
    """A real, optional presentation stream; never an alias of a wrist camera."""
    with np.load(observations/f'closeup_rgb_{index:04d}.npz') as data:
        if abs(float(data['physics_timestamp_s'])-time_s)>1e-7:
            raise ValueError('Presentation close-up timestamp differs from actual Global capture')
        return {'rgb':data['rgb'].copy()}


def depth_rgb(frame):
    """Fixed D405 70–500 mm optical-Z colour scale; invalid pixels stay black."""
    depth=np.asarray(frame['depth_m']).squeeze()
    valid=np.asarray(frame['valid_mask']).squeeze().astype(bool)
    v=np.clip((depth-.07)/(.50-.07),0,1)
    rgb=np.stack([v,1-np.abs(2*v-1),1-v],-1)
    return np.where(valid[...,None],np.uint8(rgb*255),0)


def canvas(frames, solver, phase, task_status, title, surface='particle_surface_v1', archived=False, global_crop=None, global_replay=False, global_distance_m=None):
    im=Image.new('RGB',(1920,1080),(17,19,24));draw=ImageDraw.Draw(im)
    small=ImageFont.truetype(FONT,18);large=ImageFont.truetype(FONT,25)
    t=float(frames['global_d455']['physics_timestamp_s'])
    draw.text((18,10),f'{title} | Native Isaac Sim | {solver} | {surface}',font=large,fill='white')
    draw.text((18,43),f'sim {t:.2f} s | phase {phase} | recorded task {task_status} | playback 1x',font=small,fill='#e5dbab')
    if global_replay:
        draw.text((18,65),'Global: physical 0.60 m camera replay | wrists: original Native captures',font=small,fill='#e5dbab')
    elif global_distance_m is not None:
        draw.text((18,65),f'New Native rig9 recording | fixed Global working distance {global_distance_m:.2f} m',font=small,fill='#e5dbab')
    def pane(name,box,depth=False):
        x,y,w,h=box
        frame=frames[name]
        if frame is None:
            draw.text((x+8,y+3),name,font=small,fill='white')
            draw.text((x+8,y+35),f'{t:.2f}s | NO RAW CAPTURE',font=small,fill='#e5dbab')
            draw.text((x+8,y+65),'Archived recording gap',font=small,fill='#e5dbab')
            return
        pixels=depth_rgb(frame) if depth else frame['rgb'][...,:3]
        if name=='global_d455' and global_crop is not None:
            cx,cy,cw,ch=global_crop;pixels=pixels[cy:cy+ch,cx:cx+cw]
        pic=ImageOps.contain(Image.fromarray(np.asarray(pixels,np.uint8)),(w,h-52),Image.Resampling.LANCZOS)
        im.paste(pic,(x+(w-pic.width)//2,y+52+(h-52-pic.height)//2))
        font=ImageFont.truetype(FONT,11) if depth else small
        typ='Depth 70–500 mm; invalid black' if depth else 'RGB'
        if name=='global_d455' and global_crop is not None:typ+=f' fixed pixel crop {tuple(global_crop)}; full view inset'
        draw.text((x+8,y+3),f'{name} {typ}',font=font,fill='white')
        short_solver=solver.replace('tlmpm_','') if ' ' not in solver else 'solver unrecorded'
        text=f'{t:.2f}s | {short_solver} | {phase}'
        draw.text((x+8,y+21 if depth else y+27),text,font=font,fill='#bbc3d0')
        if depth:draw.text((x+8,y+36),f"valid {np.mean(frame['valid_mask']):.1%}",font=font,fill='#bbc3d0')
    has_closeup='closeup_rgb' in frames
    pane('global_d455',(0,80,1280,640 if has_closeup else 960))
    pane('wrist_d405_0',(1280,80,640,340))
    pane('wrist_d405_1',(1280,420,640,340))
    pane('wrist_d405_0',(1280,760,320,280),True)
    pane('wrist_d405_1',(1600,760,320,280),True)
    if global_crop is not None:
        box=(220,165) if has_closeup else (320,240);origin=(8,160) if has_closeup else (952,145)
        full=ImageOps.contain(Image.fromarray(frames['global_d455']['rgb'][...,:3]),box,Image.Resampling.LANCZOS)
        im.paste(full,origin);draw.rectangle((*origin,origin[0]+full.width,origin[1]+full.height),outline='white')
        draw.text((origin[0]+4,origin[1]-18),'Global full RGB; all occluders retained',font=ImageFont.truetype(FONT,11),fill='white')
    if has_closeup:
        picture=ImageOps.contain(Image.fromarray(frames['closeup_rgb']['rgb'][...,:3]),(1280,268),Image.Resampling.LANCZOS)
        im.paste(picture,((1280-picture.width)//2,772))
        draw.text((8,722),'closeup_rgb: actual Native presentation camera (not an observation)',font=small,fill='white')
        draw.text((8,746),f'{t:.2f}s | {solver} | {phase}',font=small,fill='#bbc3d0')
    footer='Archived camera calibration retained; no retrofit to the current rig.' if archived else 'Fixed hand-local v5 D405 mounts.'
    if archived and global_replay:footer='Archived wrist calibration retained; fixed Global rig9.'
    source='Global re-render of measured states; original wrists. ' if global_replay else 'Actual recorded cameras. '
    draw.text((18,1052),source+footer+' Physical sensor calibration unvalidated.',font=small,fill='#bbc3d0')
    return np.asarray(im)


def render_episode(args, episode, out, started):
    from encode_wp13_media import encode_frames, encode_timed_frames, preview
    import subprocess
    result=json.loads((episode/'result.json').read_text())
    archived=args.archived_native
    if result.get('runtime_source')!='live_isaac_sim' and not (archived and result.get('robot_dynamics_source')=='Isaac PhysX' and result.get('replay_inputs') is False):
        raise ValueError('Requires genuine live Native episode')
    setup=json.loads((episode/'setup.json').read_text()) if (episode/'setup.json').exists() else {}
    material=json.loads((episode/'material.json').read_text()) if (episode/'material.json').exists() else {}
    solver=material.get('solver_module','legacy solver unrecorded').split('.')[-1]
    surface=setup.get('config',{}).get('surface_profile') or ('recorded archive surface' if archived else 'particle_surface_v1')
    manifest=json.loads((episode/'manifest.json').read_text())
    runtime_config=manifest.get('config',{})
    runtime_config=runtime_config.get('config',runtime_config)
    # Policy episode setup contains body labels only. Read its actual Native
    # configuration before labelling an experimental cut-aware recording.
    if runtime_config.get('surface_profile')=='cut_aware_surface_v1':
        surface='cut_aware_surface_v1'
    obs=episode/'observations'
    streams=dm.discover_streams(p.name for p in obs.glob('*.npz'))
    if not all(name in streams for name in CAMERAS):raise ValueError('Missing actual Global or either wrist stream')
    indices=streams[CAMERAS[0]]
    if not indices or (not archived and any(streams[name]!=indices for name in CAMERAS)):
        raise ValueError('Camera frame indices are not paired')
    triple_paths=getattr(args,'global_replays',None)
    if triple_paths and (getattr(args,'global_replay',None) is not None or args.global_pixel_crop is not None or args.presentation_closeup):
        raise ValueError('Three Global layout uses complete external frames; single replay/crop/closeup options conflict')
    triples=require_external_replays(triple_paths,episode,indices) if triple_paths else {}
    replay_dir=triples['left'][0] if triples else getattr(args,'global_replay',None)
    replay=None
    global_distance=None
    if replay_dir is not None:
        replay_dir=replay_dir.resolve()
        replay=json.loads((replay_dir/'replay_receipt.json').read_text())
        if replay['status']!='PASS_RENDER' or Path(replay['source_episode']).resolve()!=episode.resolve():
            raise ValueError('Global replay is incomplete or belongs to another Native episode')
        if [r['frame'] for r in replay['frames']]!=indices:
            raise ValueError('Global replay does not cover the complete original timeline')
        for row in replay['frames']:
            if dm.sha256_file(replay_dir/'observations'/f"global_d455_{row['frame']:04d}.npz")!=row['capture_sha256']:
                raise ValueError('Replayed Global source bytes changed')
        for path,expected in replay['source_sha256'].items():
            if dm.sha256_file(Path(path))!=expected:raise ValueError('Measured replay input changed')
        global_distance=replay['profile']['working_distance_to_target_m']
    elif (episode/'camera_rig.json').exists():
        rig=json.loads((episode/'camera_rig.json').read_text())['specs']['global_d455']
        if rig.get('rig_profile')=='wrist_camera_rig_v9':
            target=np.asarray(rig['look_at_target_world_m'])
            eye=np.asarray(rig['T_world_camera'])[:3,3]
            global_distance=float(np.linalg.norm(eye-target))
            if abs(global_distance-.6)>1e-6:raise ValueError('Rig9 actual physical distance is not 0.60 m')
    def load(index):
        frames=read_set(obs,index,allow_missing=archived)
        if replay_dir is not None:
            with np.load(replay_dir/'observations'/f'global_d455_{index:04d}.npz') as data:
                frames['global_d455']={k:data[k] for k in data.files}
        if args.presentation_closeup:
            frames['closeup_rgb']=read_closeup(obs,index,float(frames['global_d455']['physics_timestamp_s']))
        return frames
    def external_load(index):
        result={}
        for name,(path,_) in triples.items():
            with np.load(path/'observations'/f'global_d455_{index:04d}.npz') as data:
                result[name]={k:data[k] for k in data.files}
        return result
    initial={name:load(streams[name][0])[name] for name in CAMERAS}
    external_initial=external_load(indices[0]) if triples else {}
    if global_distance is not None and replay is None:
        if not np.allclose(initial['global_d455']['T_world_camera'],rig['T_world_camera'],atol=1e-5,rtol=0):
            raise ValueError('Recorded Native Global pose differs from declared rig9 physical pose')
    crop=args.global_pixel_crop
    if crop is not None:
        x,y,w,h=crop;rgb=initial['global_d455']['rgb']
        if min(x,y)<0 or min(w,h)<1 or x+w>rgb.shape[1] or y+h>rgb.shape[0]:raise ValueError('Global pixel crop outside actual image')
    audit=[];inputs={}
    for index in indices:
        current=load(index);checked=check_set(current,initial,archived=archived)
        if triples:
            external=external_load(index);checked['external_views']={}
            for name,frame in external.items():
                checked['external_views'][name]=check_set({**current,'global_d455':frame},
                    {**initial,'global_d455':external_initial[name]},archived=archived)
                path=triples[name][0]/'observations'/f'global_d455_{index:04d}.npz'
                inputs[str(path)]=dm.sha256_file(path)
        audit.append(checked)
        for name in (*CAMERAS,*(('closeup_rgb',) if args.presentation_closeup else ())):
            path=obs/f'{name}_{index:04d}.npz'
            if path.exists():inputs[str(path)]=dm.sha256_file(path)
        if replay_dir is not None:
            path=replay_dir/'observations'/f'global_d455_{index:04d}.npz'
            inputs[str(path)]=dm.sha256_file(path)
    times=np.array([r['time_s'] for r in audit]);dt=np.diff(times)
    offsets=np.array([r['native_time_s']-r['time_s'] for r in audit])
    if not np.allclose(offsets,offsets[0],atol=1e-6,rtol=0):raise ValueError('Native/episode clock origin changed during recording')
    if not len(dt) or np.any(dt<=0):raise ValueError('Capture timestamps must increase')
    constant=np.allclose(dt,np.median(dt),atol=1e-7,rtol=0)
    fps=1/float(np.median(dt))
    rows=dm.load_jsonl(episode/'actions.jsonl') if (episode/'actions.jsonl').exists() else []
    chosen=dm.nearest_rows([r['time_s'] for r in rows],times)
    phases={index:(('ARCHIVED_REFERENCE' if archived and not rows else 'HOME') if row<0 else rows[row].get('policy_phase','UNKNOWN')) for index,row in zip(indices,chosen)}
    status=result.get('task_status','NOT_REPORTED');title=args.title or result['task_id']
    def transform(path):
        index=int(path.stem.rsplit('_',1)[1])
        if triples:return three_global_canvas(load(index),external_load(index),solver,phases[index],status,title,surface,archived)
        return canvas(load(index),solver,phases[index],status,title,surface,archived,crop,replay is not None,global_distance)
    files=[obs/f'global_d455_{i:04d}.npz' for i in indices]
    encoded=(encode_frames(files,out/'demo.mp4',None,view=LAYOUT,frame_transform=transform,fps=fps) if constant
             else encode_timed_frames(files,out/'demo.mp4',transform,times))
    poster=transform(files[-1]);Image.fromarray(poster).save(out/'poster.png')
    speed=args.gif_speed
    command=preview(out/'demo.mp4',out/'demo.gif',start=0,duration=float(times[-1]-times[0]+1/fps),width=args.gif_width,speed=speed)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out/'demo.mp4'),'-f','null','-'],check=True)
    replay_summary=({'receipt_path':str(replay_dir/'replay_receipt.json'),
        'receipt_sha256':dm.sha256_file(replay_dir/'replay_receipt.json'),
        'source_episode':replay['source_episode'],'profile':replay['profile'],'physics_runs':0} if triples else replay)
    receipt={'schema':'native_multiview_render_receipt_v2','layout':'native_multiview_three_global_v1' if triples else LAYOUT,'episode_dir':str(episode),
        'runtime_source':'native_camera_replay_and_original_wrist_captures' if replay else 'live_isaac_sim','solver':solver,'surface_profile':surface,'archived_calibration':archived,
        'global_camera_replay':replay_summary,'encoder_simulator_runs':0,
        'global_working_distance_m':global_distance,
        'task_id':result['task_id'],'task_status':status,'cameras':list(CAMERAS),
        'panes':['global_d455 RGB','wrist_d405_0 RGB','wrist_d405_1 RGB','wrist_d405_0 Depth','wrist_d405_1 Depth'],
        'frame_count':len(indices),'capture_time_s':times.tolist(),'frame_checks':audit,
        'sync_status':'PASS','depth_colour_range_m':[.07,.50],'invalid_depth':'black',
        'sync_kind':'same_measured_state_time_with_global_camera_replay' if replay else 'simultaneous_native_camera_capture',
        'global_pixel_crop_xywh':crop,'native_clock_origin_offset_s':float(offsets[0]),
        'presentation_closeup':{'included':bool(args.presentation_closeup),'policy_observation':False,
                               'recorded_spec':setup.get('presentation_closeup_camera')},
        'initial_calibration':{name:{key:initial[name][key].tolist() for key in ('K','T_world_camera','T_hand_camera') if key in initial[name]} for name in CAMERAS},
        'missing_raw_captures':[{k:r[k] for k in ('time_s','missing_cameras')} for r in audit if r['missing_cameras']],
        'full_clip_speed':1.,'gif_speed':speed,'gif_command':command,'encoder':encoded,'full_decode':'PASS',
        'source_sha256':{**inputs,**{str(episode/n):dm.sha256_file(episode/n) for n in ('result.json','material.json','actions.jsonl','setup.json') if (episode/n).exists()}},
        'recording_modified':False,'physical_sensor_calibration_validated':False,
        'command':sys.argv,'versions':{'python':platform.python_version(),'numpy':np.__version__,
            'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},
        'implementation_sha256':{str(path):dm.sha256_file(path) for path in (
            Path(__file__),Path(dm.__file__),Path(__file__).resolve().parents[2]/'scripts/encode_wp13_media.py',
            Path(__file__).resolve().parents[2]/'scripts/render_demo_episode.py')},
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'renderer_source_status':subprocess.check_output(['git','status','--porcelain','--',
            str(Path(__file__)),str(Path(dm.__file__)),
            str(Path(__file__).resolve().parents[2]/'scripts/encode_wp13_media.py'),
            str(Path(__file__).resolve().parents[2]/'scripts/render_demo_episode.py')],text=True).strip(),
        'outputs':{n:dm.sha256_file(out/n) for n in ('demo.mp4','demo.gif','poster.png')},
        'wall_s':time.perf_counter()-started}
    if triples:
        receipt['three_global_views']={name:{'receipt_path':str(path/'replay_receipt.json'),
            'receipt_sha256':dm.sha256_file(path/'replay_receipt.json'),'profile':r['profile'],
            'source_commit':r['source_commit'],'physics_runs':r['physics_runs'],
            'frame_count':len(r['frames']),'maximum_body_matrix_error':max(x['maximum_rendered_body_matrix_error'] for x in r['frames']),
            'maximum_render_near_m':max(x['render_clipping_range_m'][0] for x in r['frames']),
            'initial_calibration':{key:external_initial[name][key].tolist() for key in ('K','T_world_camera')}}
            for name,(path,r) in triples.items()}
        receipt['cameras']=['global_'+name for name in EXTERNAL_VIEWS]+list(CAMERAS[1:])
        receipt['panes']=['global_'+name+' RGB' for name in EXTERNAL_VIEWS]+receipt['panes'][1:]
        for path,_ in triples.values():inputs[str(path/'replay_receipt.json')]=dm.sha256_file(path/'replay_receipt.json')
        receipt['source_sha256'].update(inputs)
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
