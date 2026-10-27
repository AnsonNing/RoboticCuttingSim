# Asset sources and acquisition

No Franka, RealSense or other vendor binary asset is redistributed in this repo.
The workcell exporter now fetches the complete USD composition subset (13files,
21.4MB) and successfully composes both robots and native camera hierarchies.
Per-layer hashes/URLs are recorded in each workcell's asset_fetch.json. External
MDL shader validation remains pending. See [WORKCELL](../docs/WORKCELL.md).
`scripts/audit_assets.py` reads the official asset root from the pinned Lab 3.0
checkout and fetches only the two small root USD files (16 MiB cap per file).
Nested composition, joint drives, sensor annotators and RTX rendering are NOT_RUN.

| Root asset | Download verification on 2026-10-28 |
|---|---|
| `Isaac/Sensors/RealSense/D455/rsd455.usd` | HTTP 200, 14,070,691 bytes; SHA256 `441c2bf503f7007f5ff284283459bf5c21507ae562074f37f54f9223ea3956a7` |
| `Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda` | HTTP 200, 2,096 bytes; SHA256 `eaf27723ba1934d792c6876679eb75491bdff944ad108d4e59734fa07dc65809` |

Asset root:
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2`.
Franka path comes from `source/isaaclab_assets/isaaclab_assets/robots/franka.py`
in pinned Lab source. D455 reference:
[official depth sensor catalog](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_depth_sensors.html).
These assets remain subject to their official NVIDIA/vendor licensing; HTTP
availability does not grant redistribution permission. Full original requests,
ETags and hashes are local in `artifacts/assets_audit_002/results.json`.

The first D455 audit reached HTTP 200 but deliberately stopped at a 2 MiB cap;
that receipt is retained in `artifacts/assets_audit_001`. A second audit used a
16 MiB cap and completed the root-file download. This was an audit cap, not an
asset access denial.

Table, tray and box blade are self-generated metric primitives in `scene.py`.
The initial continuous specimen mesh is a height-function CAD preview; recorded
MPM replay replaces it with ParticleSurface extraction from measured states.
The physics-only preview/live ideal renderer uses own pinhole-camera specs;
the complete workcell additionally loads the actual D455 digital twins. The box
blade is a collision proxy; it does not
model a calibrated knife edge or force feedback into a robot.
