#!/usr/bin/env python3
"""Execute the explicitly named finite-interface candidate through compiled control."""
import argparse
from pathlib import Path

from run_wp13 import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--solver',choices=('tlmpm_v7_1','tlmpm_v7_2'),default='tlmpm_v7_1')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dt',type=float)
    a=p.parse_args()
    a.episode='TRIM';a.surface='particle_surface_v1'
    a.label='WP14_FINITE_SUPPORT_CANDIDATE_NOT_ADOPTED'
    a.planner_source='compiled'
    return run(a)


if __name__=='__main__':
    raise SystemExit(main())
