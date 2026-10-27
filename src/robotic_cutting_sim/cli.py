import argparse
import json
import traceback
from pathlib import Path

from .recording import ROOT, new_output, write_json


def positive(value):
    value=float(value)
    if not __import__("math").isfinite(value) or value<=0:
        raise argparse.ArgumentTypeError("Value must be finite and positive")
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description="RoboticCuttingSim: blade-only MPM research testbed")
    sub=parser.add_subparsers(dest="command",required=True)
    doctor=sub.add_parser("doctor",help="Read-only hardware and version audit")
    doctor.add_argument("--output-dir",type=Path,required=True)
    scene=sub.add_parser("scene",help="Create a static USD preview and continuous particle sample")
    scene.add_argument("--preset",choices=["physics_only"],default="physics_only")
    scene.add_argument("--variant",choices=["D0","D1"],default="D0")
    scene.add_argument("--family",choices=["dome","gaussian","ellipsoid_cap","ridge"],default="dome")
    scene.add_argument("--seed",type=int,default=42); scene.add_argument("--spacing",type=positive,default=.002)
    scene.add_argument("--output-dir",type=Path,required=True)
    probe=sub.add_parser("physics-probe",help="Actual GPU MPM deformation/contact experiment")
    probe.add_argument("--solver",choices=["newton","cressim"],default="newton")
    probe.add_argument("--case",choices=["settle","indent_unload","shallow_scrape","through_bump_cut"],required=True)
    probe.add_argument("--seed",type=int,default=42); probe.add_argument("--variant",choices=["D0","D1"],default="D0")
    probe.add_argument("--voxel-size",type=positive,default=.004); probe.add_argument("--dt",type=positive,default=1/60)
    probe.add_argument("--substeps",type=int,default=4); probe.add_argument("--frames",type=int,default=120)
    probe.add_argument("--device",default="cuda:0"); probe.add_argument("--trajectory",type=Path)
    probe.add_argument("--material",type=Path,default=ROOT/"configs/materials/plasticine_provisional.yaml")
    probe.add_argument("--output-dir",type=Path,required=True)
    camera=sub.add_parser("camera-check",help="Actual ParticleSurface extraction to ideal Warp RGB-D arrays")
    camera.add_argument("--input",type=Path,required=True); camera.add_argument("--surface-voxel-size",type=positive,default=.0015)
    camera.add_argument("--samples",type=int,default=5); camera.add_argument("--device",default="cuda:0")
    camera.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="physics-probe" and (args.frames<1 or args.substeps<1):
        parser.error("--frames and --substeps must be positive integers")
    if args.command=="camera-check" and args.samples<2:
        parser.error("--samples must be at least two")
    try:
        if args.command=="doctor":
            from .doctor import run
            result=run(args.output_dir)
            print(json.dumps(result,indent=2)); return 2 if result["cuda13_driver_gate"]=="BLOCKED" or result["isaacsim_runtime"]=="BLOCKED" else 0
        output=new_output(args.output_dir)
        if args.command=="scene":
            from .scene import export
            result=export(args,output)
        elif args.command=="physics-probe":
            if args.solver=="newton":
                from .newton_probe import run
            else:
                from .cressim_probe import run
            result=run(args,output)
        else:
            from .camera import run
            result=run(args,output)
        print(json.dumps(result,indent=2))
        return 0 if result["status"]=="PASS" else 1
    except FileExistsError:
        print(json.dumps({"status":"FAIL","error":"Output directory exists; choose a fresh run path"}))
        return 1
    except Exception as error:
        detail=traceback.format_exc()
        if args.output_dir.is_dir():
            (args.output_dir/"failure.log").write_text(detail)
            write_json(args.output_dir/"failure.json",{"status":"FAIL","error":str(error),"type":type(error).__name__})
        print(detail)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
