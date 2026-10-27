#!/usr/bin/env python3
"""Regenerate the small report figure from corrected physical experiment frames."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root=Path(__file__).resolve().parents[1]
run=root/"artifacts/newton_indent_002/preview_frames"
files=sorted(run.glob("*.png"))
fig,axes=plt.subplots(3,1,figsize=(9,12))
for ax,i,label in zip(axes,[0,36,len(files)-1],["Before indentation","During contact","After unloading"]):
    ax.imshow(Image.open(files[i])); ax.set_axis_off(); ax.set_title(label)
fig.suptitle("Recorded Newton MPM deformation | provisional uncalibrated material\nDiagnostic cross-section; persistent separation NOT proven",fontsize=12)
fig.tight_layout(rect=[0,0,1,.95])
fig.savefig(root/"docs/evidence/newton_indentation.png",dpi=100)
