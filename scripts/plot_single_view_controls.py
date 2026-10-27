#!/usr/bin/env python3
"""Plot frozen native-control errors, coverage, actual RGB and retained noise."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit_dir",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    result=json.loads((args.audit_dir/"result.json").read_text())
    rows=result["rows"]
    case_ids=list(dict.fromkeys(r["case_id"] for r in rows))
    figure,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    x=np.arange(len(case_ids))
    for h,color in (("H0","tab:orange"),("H1","tab:blue")):
        values=[max(r["plane_error"]["p95_absolute_m"] for r in rows if r["label"]=="off" and r["hypothesis"]==h and r["case_id"]==name)*1e3 for name in case_ids]
        axes[0,0].plot(x,values,"o-",label=h,color=color)
    axes[0,0].axhline(1.,color="black",linestyle="--",label="frozen OFF gate")
    axes[0,0].set(yscale="log",ylabel="Plane p95 absolute error (mm)",title="Actual native OFF, worst of two cameras")
    axes[0,0].set_xticks(x,case_ids,rotation=70,fontsize=7)
    axes[0,0].legend()
    for label,color in (("off","tab:blue"),("on","tab:red")):
        values=[min(r["known_geometry_valid_coverage"] for r in rows if r["label"]==label and r["hypothesis"]=="H1" and r["case_id"]==name)*100 for name in case_ids]
        axes[0,1].plot(x,values,"o-",label=f"noise {label}",color=color)
    axes[0,1].axhline(90.,color="black",linestyle="--",label="OFF-only coverage gate")
    axes[0,1].set(ylabel="Valid registered geometry (%)",ylim=(0,101),title="Native invalid pixels remain holes")
    axes[0,1].set_xticks(x,case_ids,rotation=70,fontsize=7)
    axes[0,1].legend()
    heldout=next(r for r in rows if r["label"]=="off" and r["hypothesis"]=="H1" and r["camera"]=="d455_0" and r["case_id"]=="step_held_out")
    example=np.load(args.audit_dir/"off"/f"d455_0_{heldout['frame_index']:04d}.npz")
    box=example["analytic_box_mask"]
    ys,xs=np.where(box)
    a,b=max(0,ys.min()-10),min(box.shape[0],ys.max()+11)
    c,d=max(0,xs.min()-10),min(box.shape[1],xs.max()+11)
    axes[1,0].imshow(example["rgb"][a:b,c:d])
    axes[1,0].contour(box[a:b,c:d],levels=[.5],colors=["cyan"],linewidths=.8)
    axes[1,0].set(title="Held-out native camera RGB + analytic box edge\nH1 depth/RGB boundary p95=0 pixels",xlabel="Original RGB pixels",ylabel="Original RGB pixels")
    onrow=next(r for r in rows if r["label"]=="on" and r["hypothesis"]=="H1" and r["camera"]=="d455_0" and r["case_id"]=="mixed_negative")
    noisy=np.load(args.audit_dir/"on"/f"d455_0_{onrow['frame_index']:04d}.npz")
    delta=np.ma.masked_where(~noisy["valid_mask"],(noisy["depth_m"]-noisy["analytic_scene_depth_m"])*1e3)
    cmap=plt.get_cmap("coolwarm").copy();cmap.set_bad(".65")
    plot=axes[1,1].imshow(delta,cmap=cmap,vmin=-3,vmax=3)
    axes[1,1].set(title="Native noise ON, held-out slanted plane\nRaw measured Z preserved; gray=invalid",xlabel="RGB pixel X",ylabel="RGB pixel Y")
    figure.colorbar(plot,ax=axes[1,1],label="Native depth minus analytic plane (mm)")
    runtime = "real Isaac Sim6.2/Kit" if "native_kit_" in result["protocol_id"] else "standalone OVRTX"
    figure.suptitle(f"NVIDIA SingleView source correspondence controls — {runtime}, not hardware calibration")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output,dpi=150)


if __name__=="__main__":
    main()
