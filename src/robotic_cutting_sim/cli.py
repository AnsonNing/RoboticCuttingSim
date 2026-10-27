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
    scene.add_argument("--preset",choices=["physics_only","workcell"],default="physics_only")
    scene.add_argument("--asset-root",help="Complete local or official Isaac asset root for workcell composition")
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
    demo=sub.add_parser("env-demo",help="Live MPM environment with reset/step and two ideal RGB-D observations")
    demo.add_argument("--steps",type=int,default=120); demo.add_argument("--seed",type=int,default=42)
    demo.add_argument("--case",choices=["indent_unload","through_bump_cut","shallow_scrape","settle"],default="indent_unload")
    demo.add_argument("--voxel-size",type=positive,default=.004); demo.add_argument("--device",default="cuda:0")
    demo.add_argument("--output-dir",type=Path,required=True)
    cutting=sub.add_parser("cutting-probe",help="Experimental persistent TLMPM cutting, withdrawal and recontact")
    cutting.add_argument("--case",choices=["settle","no_contact","indent","uncut_scrape","cut","cut_recontact","two_cuts","cross_cuts","supported_cross_cuts"],default="cut")
    cutting.add_argument("--dx",type=positive,default=.004)
    cutting.add_argument("--dt",type=positive,default=.0001)
    cutting.add_argument("--duration",type=positive,default=2.5)
    cutting.add_argument("--variant",choices=["D0","D1"],default="D0")
    cutting.add_argument("--seed",type=int,default=42)
    cutting.add_argument("--device",default="cuda:0")
    cutting.add_argument("--output-dir",type=Path,required=True)
    research=sub.add_parser("research-demo",help="Live dual-Panda cutting with measured proprioception and full-scene ideal RGB-D")
    research.add_argument("--steps",type=int,default=400)
    research.add_argument("--record-every",type=int,default=10)
    research.add_argument("--device",default="cuda:0")
    research.add_argument("--scene",type=Path)
    research.add_argument("--output-dir",type=Path,required=True)
    benchmark=sub.add_parser("benchmark",help="Execute the standalone D0–D3 simulation benchmark; no training")
    benchmark.add_argument("--variants",nargs="+",choices=["D0","D1","D2","D3"],default=["D0","D1","D2","D3"])
    benchmark.add_argument("--seeds",nargs="+",type=int,default=[42])
    benchmark.add_argument("--duration",type=positive,default=8.)
    benchmark.add_argument("--device",default="cuda:0")
    benchmark.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command in ("cutting-probe","research-demo","benchmark"):
        import subprocess
        import sys
        scripts={"cutting-probe":"run_cutting_probe.py", "research-demo":"check_research_env.py",
                 "benchmark":"run_simulation_benchmark.py"}
        command=[sys.executable,str(ROOT/"scripts"/scripts[args.command])]
        for name,value in vars(args).items():
            if name=="command" or value is None:
                continue
            command.append("--"+name.replace("_","-"))
            command.extend(map(str,value)) if isinstance(value,list) else command.append(str(value))
        return subprocess.run(command,cwd=ROOT).returncode
    if args.command=="physics-probe" and (args.frames<1 or args.substeps<1):
        parser.error("--frames and --substeps must be positive integers")
    if args.command=="camera-check" and args.samples<2:
        parser.error("--samples must be at least two")
    if args.command=="env-demo" and args.steps<1:
        parser.error("--steps must be positive")
    try:
        if args.command=="doctor":
            from .doctor import run
            result=run(args.output_dir)
            print(json.dumps(result,indent=2)); return 2 if result["cuda13_driver_gate"]=="BLOCKED" or result["isaacsim_runtime"]=="BLOCKED" else 0
        output=new_output(args.output_dir)
        if args.command=="scene":
            if args.preset=="workcell":
                from .workcell import export
            else:
                from .scene import export
            result=export(args,output)
        elif args.command=="env-demo":
            from .env_demo import run
            result=run(args,output)
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
