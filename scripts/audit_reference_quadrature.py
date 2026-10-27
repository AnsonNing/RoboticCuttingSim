#!/usr/bin/env python3
"""Measure D0 quadrature volume separately from nominal boundary and rendering."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.geometry import Specimen, sample_particles
from robotic_cutting_sim.recording import new_output, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir",type=Path)
    args=p.parse_args()
    output=new_output(args.output_dir)
    spec=Specimen()
    density=1500.
    base_volume=spec.length*spec.width*spec.base_height
    bump_volume=2*np.pi*spec.bump_height*spec.bump_radius**2*(.25-1/np.pi**2)
    rows=[]
    for spacing in (.001,.0015,.002):
        points,masses,_=sample_particles(spec,spacing,density)
        base=points[points[:,2] <= spec.base_height+1e-9]
        lower,upper=base.min(0).astype(float)-spacing/2,base.max(0).astype(float)+spacing/2
        rows.append({"spacing_m":spacing,"particles":len(points),"base_quadrature_particles":len(base),
            "reference_volume_m3":float(masses.sum()/density),"analytic_total_volume_m3":base_volume+bump_volume,
            "relative_quadrature_volume_error":float((masses.sum()/density)/(base_volume+bump_volume)-1),
            "base_reference_volume_m3":len(base)*spacing**3,"analytic_base_volume_m3":base_volume,
            "base_center_z_range_m":[float(base[:,2].min()),float(base[:,2].max())],
            "base_full_cell_union_bounds_m":[lower.tolist(),upper.tolist()],
            "base_full_cell_top_minus_nominal_m":float(upper[2]-spec.base_height),
            "full_cell_interpretation":"Newton density kernel uses volume8r^3=spacing^3. These are quadrature-cell extents, not a physical boundary oracle or cut score."})
    source=Path("src/robotic_cutting_sim/geometry.py")
    (output/"geometry_source.py").write_bytes(source.read_bytes())
    result={"scope":"CPU audit of actual default D0 sampler, not dynamics or cutting acceptance",
        "specimen":asdict(spec),"density_kg_m3":density,"rows":rows,
        "analytic_dome_extra_volume_formula":"2*pi*height*radius^2*(1/4-1/pi^2)",
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "physics_or_rendering_modified":False}
    write_json(output/"result.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
