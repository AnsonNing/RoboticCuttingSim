#!/usr/bin/env python3
"""Uncut 2mm reconstruction and known missing-ray/geometry negative controls."""
import argparse
from pathlib import Path
import hashlib
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from robotic_cutting_sim.recording import new_output,write_json
from robotic_cutting_sim.surface_evaluation import TargetGeometry,RegionSpec,evaluate_height_surface
from robotic_cutting_sim.evaluation import PROTOCOL_PATH


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    out=new_output(args.output_dir)
    target_path=args.target/'fixed_target.npz'
    with np.load(target_path) as a:
        vertices,faces=a['vertices_local_m'],a['triangles']
    plane=TargetGeometry('plane','original analytic flatbar','analytic_v1',parameters={'origin_m':[0,0,.02]})
    represented=TargetGeometry('mesh',str(target_path.resolve()),'independent_uncut_flatbar_v1',
        parameters={'vertices_m':vertices,'triangles':faces,'independent_uncut_target':True})
    boundary=RegionSpec('boundary','boundary',{'bounds_xy_m':[[-.08,-.015],[.08,.015]],'width_m':.003},'evaluation_v2_predeclared_band')
    cases={}
    for name,v,f in [('uncut_flatbar_2mm',vertices,faces),('empty_surface',vertices,faces[:0]),
            ('known_4mm_vertical_offset',vertices+[0,0,-.004],faces)]:
        r,raw=evaluate_height_surface(v,f,plane,world_from_target_xyzw=(0,0,0,0,0,0,1),actual_frame_id='workpiece_local',
            bounds_xy_m=[[-.08,-.015],[.08,.015]],regions=(boundary,),represented_target=represented)
        np.savez_compressed(out/(name+'_raw.npz'),**raw)
        cases[name]=r
    m=cases['uncut_flatbar_2mm']['analytic_target']['entire_region']
    empty=cases['empty_surface']['analytic_target']['entire_region']
    checks={'uncut_reconstruction_has_analytic_error':m['paired_p95_absolute_error_m']>.001,
        'missing_rays_remain_missing':m['missing_actual_columns']>0,
        'missing_not_reported_as_measured_20mm_gouge':m['measured_maximum_surface_deficit_m']<.02,
        'represented_self_paired_error_zero':cases['uncut_flatbar_2mm']['represented_target']['regions']['entire_region']['paired_p95_absolute_error_m']==0,
        'represented_self_gaps_still_reported':cases['uncut_flatbar_2mm']['represented_target']['regions']['entire_region']['unknown_target_columns']>0,
        'empty_surface_not_measurable':empty['status']=='NOT_MEASURABLE' and empty['measured_maximum_surface_deficit_m'] is None,
        'empty_surface_no_whole_region_pass':empty['whole_region_deficit_volume_m3'] is None}
    paths=[target_path,args.target/'protocol.json',PROTOCOL_PATH,Path(__file__),ROOT/'src/robotic_cutting_sim/surface_evaluation.py']
    write_json(out/'result.json',{'status':'PASS_CONTROLS' if all(checks.values()) else 'FAIL_CONTROLS','checks':checks,
        'cases':cases,'input_source_sha256':{str(p):digest(p) for p in paths},
        'scope':'Actual independently built uncut target and explicit geometric offsets; no material advance or cutting success claim'})
    print('PASS_CONTROLS' if all(checks.values()) else 'FAIL_CONTROLS')
    return int(not all(checks.values()))

if __name__=='__main__':
    raise SystemExit(main())
