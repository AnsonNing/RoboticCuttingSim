#!/usr/bin/env python3
"""Eight actual Native Global views, projected material range and camera provenance.

RGB colour coverage is a diagnostic, not cutting evidence or an occlusion model.
The unmodified PNGs and contact sheet require visual review of the working edge.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from robotic_cutting_sim.demo_media import sha256_file,load_jsonl,nearest_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=load_jsonl(a.episode/'actions.jsonl');ph={}
    for r in rows:ph.setdefault(r['policy_phase'],[]).append(r['time_s'])
    files=sorted((a.episode/'observations').glob('global_d455_*.npz'))
    ts=np.array([float(np.load(f)['physics_timestamp_s']) for f in files])
    wanted={'initial':0.,'first_approach':ph['PRE_CUT:A_dome'][0],
        'middle_A':float(np.mean(ph['SKIM:A_dome'])),'A_complete':ph['WITHDRAW:A_dome'][0],
        'second_approach':ph['PRE_CUT:B_irregular'][0],'middle_B':float(np.mean(ph['SKIM:B_irregular'])),
        'final_surface':ph['HOLD_AFTER_STOP'][0],'HOME':float(ts[-1])}
    z=np.load(a.episode/'physics_state.npz');X=z['reference_positions_local'];states=z['positions'];pt=z['time_s']
    base_top=(X[:,2]<.02)&(X[:,2]>.019)
    sheet=Image.new('RGB',(1280,2040));draw=ImageDraw.Draw(sheet);frames=[]
    for j,(label,t) in enumerate(wanted.items()):
        i=int(np.argmin(abs(ts-t)));source=files[i]
        with np.load(source) as d:
            rgb=d['rgb'];camera=np.asarray(d['T_world_camera']);K=np.asarray(d['K']);height,width=rgb.shape[:2]
            state=states[int(np.argmin(abs(pt-ts[i])))];q=(state-camera[:3,3])@camera[:3,:3]
            proj=q@K.T;uv=proj[:,:2]/proj[:,2:3]
            top=q[base_top];uv_top=np.floor(uv[base_top]).astype(int)
            inside=(uv_top[:,0]>=0)&(uv_top[:,0]<width)&(uv_top[:,1]>=0)&(uv_top[:,1]<height)&(top[:,2]>0)
            unique=np.unique(uv_top[inside],axis=0);pixels=rgb[unique[:,1],unique[:,0]].astype(float)
            orange=(pixels[:,0]>1.15*pixels[:,1])&(pixels[:,1]>1.15*pixels[:,2])
            Image.fromarray(rgb).save(a.out/(label+'.png'))
            sheet.paste(Image.fromarray(rgb),((j%2)*640,(j//2)*510+30))
            draw.text(((j%2)*640+8,(j//2)*510+8),f'{label}: actual sim {ts[i]:.2f}s',fill='white')
            frames.append({'phase':label,'requested_time_s':t,'captured_time_s':float(ts[i]),
                'source':str(source),'source_sha256':sha256_file(source),'png':str(a.out/(label+'.png')),
                'K':K.tolist(),'T_world_camera':camera.tolist(),'valid_depth_fraction_full_image':float(d['valid_mask'].mean()),
                'material_optical_Z_range_m':[float(q[:,2].min()),float(q[:,2].max())],
                'projected_base_top_unique_pixels':len(unique),'base_top_material_colour_fraction_diagnostic':float(orange.mean()),
                'range_of_colour_rule':'RGB R>1.15G and G>1.15B; diagnostic only, never task outcome'})
    sheet.save(a.out/'contact_sheet.png')
    q1=np.array([r['observation_proprio']['q'][1] for r in rows])
    receipt={'schema':'native_eight_phase_camera_audit_v1','frames':frames,
        'secondary_arm_joint_max_change_from_first_rad':float(np.abs(q1-q1[0]).max()),
        'projection_only_status':'MEASURED','visual_visibility_status':'PENDING_VISUAL_REVIEW',
        'scope':'Actual Native RGB with robot/tool/gripper rendered. No image alteration except contact-sheet labels. '
                'Pixel colour coverage is not a visibility or physical-cutting acceptance gate. '
                'Wrist extrinsics, fixed Global and all timestamps are checked independently by the multiview renderer.'}
    (a.out/'camera_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
