# Third-party sources

External checkouts are excluded from Git. Exact upstream revisions are pinned in
[configs/source_refs.json](configs/source_refs.json); build/runtime manifests also
record their tracked diff hashes. No third-party asset is redistributed here.

| Source | License | Use |
|---|---|---|
| [Newton](https://github.com/newton-physics/newton) 1.6.1 | Apache-2.0 | Installed official wheel, public MPM and ParticleSurface APIs; official example tested separately |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 3.0.0 | BSD-3-Clause | Pinned source and official CLI/dependency/asset reference; full runtime blocked |
| [CRESSim-MPM](https://github.com/yafei-ou/CRESSim-MPM) | BSD-3-Clause | Isolated shared-library build and headless API harness; license retained at [licenses/CRESSim-MPM.txt](licenses/CRESSim-MPM.txt) |
| NVIDIA Warp 1.17.0 | Apache-2.0 | Official runtime and programmable mesh ray camera |
| OpenUSD core | Apache-2.0 | `usd-core` standalone scene export; not coinstalled with Isaac's `usd-exchange` |

The CRESSim patch in `scripts/patches/cressim_si_mls.patch` changes only its isolated
checkout: SI-scale nodal mass cutoff/empty-node velocity, missing constitutive switch
break, quadratic APIC inverse-spacing factor, and world-frame slicer normal. It is
an experimental local correction, not an upstream release or validated cutting law.
Original copyright/license headers remain in the source; the patch is shipped with
the upstream BSD notice. `build_cressim.sh` applies it idempotently and refuses
incompatible patch context. Moving-normal slicer relative-velocity semantics remain
unresolved, and no persistent fracture labels are added by this patch.

Isaac Sim and official Franka/D455 assets have separate NVIDIA/asset terms. Assets
are to be resolved from their official root and used under those terms; see
[ASSET_SOURCES](assets/ASSET_SOURCES.md). DiSECt, CPIC, BladeMaster and research
datasets have not been copied, installed or redistributed.
