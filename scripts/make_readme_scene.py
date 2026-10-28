#!/usr/bin/env python3
"""Lay out three actual synchronized Native RGB captures; no new rendering."""
import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robotic_cutting_sim.recording import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--time",type=float,default=4.)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists() or args.output.with_suffix('.encoder.py').exists():
        raise FileExistsError("Use a fresh image/provenance name")
    result_path=args.episode/'result.json'
    result=json.loads(result_path.read_text())
    task_id=result['task_id']
    frames=json.loads((args.episode/'frame_manifest.json').read_text())
    row=min(frames,key=lambda r:abs(r['time_s']-args.time))
    if abs(row['time_s']-args.time)>1e-9:
        raise ValueError("Requested actual frame timestamp was not captured")
    rig=json.loads((args.episode/'camera_rig.json').read_text())
    profile=rig['specs']['wrist_d405_1']['rig_profile']
    if profile not in ('wrist_camera_rig_v4','wrist_camera_rig_v5'):
        raise ValueError("Expected an actual parked-wrist v4/v5 recording")
    canvas=Image.new('RGB',(1600,850),'#f2f4f6')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    panels=[('global_d455','Close Global D455',(0,40,1000,750)),
        ('wrist_d405_0','Wrist D405 0 | tool hand',(1000,40,600,340)),
        ('wrist_d405_1','Wrist D405 1 | normal parked arm',(1000,440,600,340))]
    inputs=[]
    for name,label,(x,y,w,h) in panels:
        path=args.episode/'observations'/f"{name}_{row['frame']:04d}.npz"
        with np.load(path) as data:
            timestamp=float(data['physics_timestamp_s'])
            if abs(timestamp-row['time_s'])>1e-9:
                raise ValueError("Captured camera timestamp mismatch")
            picture=Image.fromarray(data['rgb'])
        scale=min(w/picture.width,h/picture.height)
        picture=picture.resize((round(picture.width*scale),round(picture.height*scale)),Image.Resampling.LANCZOS)
        canvas.paste(picture,(x+(w-picture.width)//2,y+(h-picture.height)//2))
        draw.text((x+14,y-28),label,fill='#22303c',font=font)
        inputs.append({'camera_id':name,'file':str(path.resolve()),'sha256':digest(path),'physics_timestamp_s':timestamp})
    attachment=result['tool_attachment_mode'].replace('_',' ')
    draw.text((14,808),f"Actual Native {task_id} | t={row['time_s']:.1f}s | {attachment} | {profile} | Franka 1 stays parked",fill='#22303c',font=font)
    canvas.save(args.output,optimize=True)
    encoder_path=args.output.with_suffix('.encoder.py')
    encoder_path.write_bytes(Path(__file__).read_bytes())
    write_json(args.output.with_suffix('.json'),{'scope':__doc__,'source_episode':str(args.episode.resolve()),
        'task_id':task_id,'source_result_sha256':digest(result_path),
        'source_execution_status':result['status'],'source_task_goal_status':result['task_goal_status'],
        'source_manifest_sha256':digest(args.episode/'manifest.json'),'camera_rig_sha256':digest(args.episode/'camera_rig.json'),
        'frame':row['frame'],'physics_timestamp_s':row['time_s'],'camera_rig_version':profile,
        'inputs':inputs,'output_sha256':digest(args.output),'encoder_sha256':digest(Path(__file__)),
        'encoder_source_file':str(encoder_path),
        'display':'Aspect-preserving resizing and labels only; original RGB-D arrays and historical README image remain unchanged'})


if __name__=='__main__':
    main()
