#!/usr/bin/env python3
"""Plot immutable initial-flatbar control arrays; no extraction or task scoring."""
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from robotic_cutting_sim.recording import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    if args.output_png.exists():
        raise FileExistsError(args.output_png)
    figure, axes = plt.subplots(2, 1, figsize=(11, 4.8), constrained_layout=True)
    inputs = {}
    for ax, name, label in zip(axes, ("analytic_exact_flatbar", "h2_initial_flatbar"),
                               ("Exact analytic flatbar", "Uncut H2 particle reconstruction")):
        source = args.control/(name+".npz")
        inputs[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source) as data:
            deficit = data["conservative_gouge_m"]*1000
            plotted = ax.imshow(deficit, origin="lower", extent=[-80, 80, -15, 15],
                                vmin=0., vmax=1.5, cmap="magma", aspect="auto")
            ax.add_patch(Rectangle((-77, -12), 154, 24, fill=False, linestyle="--",
                                   edgecolor="cyan", linewidth=.9))
            ax.set_title(f"{label}: max deficit {deficit.max():.6f} mm; full 160 × 30 mm footprint")
        ax.set_ylabel("Local Y (mm)")
        ax.set_xlabel("Local X (mm)")
    figure.colorbar(plotted, ax=axes, label="Conservative target-plane deficit (mm)")
    figure.suptitle("No knife or dynamics; cyan interior is descriptive only; whole-footprint v1 unchanged")
    figure.savefig(args.output_png, dpi=180)
    plt.close(figure)
    write_json(args.output_png.with_suffix(".json"), {"source_sha256": inputs,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(args.output_png.read_bytes()).hexdigest(),
        "display_range_mm": [0., 1.5], "full_numeric_arrays_unmodified": True})


if __name__ == "__main__":
    main()
