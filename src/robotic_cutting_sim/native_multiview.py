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


def canvas(frames, solver, phase, task_status, title, surface='particle_surface_v1', archived=False, global_crop=None):
    im=Image.new('RGB',(1920,1080),(17,19,24));draw=ImageDraw.Draw(im)
    small=ImageFont.truetype(FONT,18);large=ImageFont.truetype(FONT,25)
    t=float(frames['global_d455']['physics_timestamp_s'])
    draw.text((18,10),f'{title} | Native Isaac Sim | {solver} | {surface}',font=large,fill='white')
    draw.text((18,43),f'sim {t:.2f} s | phase {phase} | recorded task {task_status} | playback 1x',font=small,fill='#e5dbab')
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
    draw.text((18,1052),'Actual recorded cameras. '+footer+' Physical sensor calibration unvalidated.',font=small,fill='#bbc3d0')
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
    obs=episode/'observations'
    streams=dm.discover_streams(p.name for p in obs.glob('*.npz'))
    if not all(name in streams for name in CAMERAS):raise ValueError('Missing actual Global or either wrist stream')
    indices=streams[CAMERAS[0]]
    if not indices or (not archived and any(streams[name]!=indices for name in CAMERAS)):
        raise ValueError('Camera frame indices are not paired')
    initial={name:read_set(obs,streams[name][0],allow_missing=archived)[name] for name in CAMERAS}
    def load(index):
        frames=read_set(obs,index,allow_missing=archived)
        if args.presentation_closeup:
            frames['closeup_rgb']=read_closeup(obs,index,float(frames['global_d455']['physics_timestamp_s']))
        return frames
    crop=args.global_pixel_crop
    if crop is not None:
        x,y,w,h=crop;rgb=initial['global_d455']['rgb']
        if min(x,y)<0 or min(w,h)<1 or x+w>rgb.shape[1] or y+h>rgb.shape[0]:raise ValueError('Global pixel crop outside actual image')
    audit=[];inputs={}
    for index in indices:
        audit.append(check_set(load(index),initial,archived=archived))
        for name in (*CAMERAS,*(('closeup_rgb',) if args.presentation_closeup else ())):
            path=obs/f'{name}_{index:04d}.npz'
            if path.exists():inputs[str(path)]=dm.sha256_file(path)
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
        index=int(path.stem.rsplit('_',1)[1]);return canvas(load(index),solver,phases[index],status,title,surface,archived,crop)
    files=[obs/f'global_d455_{i:04d}.npz' for i in indices]
    encoded=(encode_frames(files,out/'demo.mp4',None,view=LAYOUT,frame_transform=transform,fps=fps) if constant
             else encode_timed_frames(files,out/'demo.mp4',transform,times))
    poster=transform(files[-1]);Image.fromarray(poster).save(out/'poster.png')
    speed=args.gif_speed
    command=preview(out/'demo.mp4',out/'demo.gif',start=0,duration=float(times[-1]-times[0]+1/fps),width=args.gif_width,speed=speed)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out/'demo.mp4'),'-f','null','-'],check=True)
    receipt={'schema':'native_multiview_render_receipt_v2','layout':LAYOUT,'episode_dir':str(episode),
        'runtime_source':'live_isaac_sim','solver':solver,'surface_profile':surface,'archived_calibration':archived,
        'task_id':result['task_id'],'task_status':status,'cameras':list(CAMERAS),
        'panes':['global_d455 RGB','wrist_d405_0 RGB','wrist_d405_1 RGB','wrist_d405_0 Depth','wrist_d405_1 Depth'],
        'frame_count':len(indices),'capture_time_s':times.tolist(),'frame_checks':audit,
        'sync_status':'PASS','depth_colour_range_m':[.07,.50],'invalid_depth':'black',
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
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
