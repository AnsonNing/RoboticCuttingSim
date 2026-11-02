# Three Global RGB views and two wrist RGB-D cameras

The current display keeps **LEFT (rig10), RIGHT (rig11), OPPOSITE TOOL (rig14)**
and both actual hand-local Wrist D405s. Full movies show all three Global RGB
frames above arm_0/arm_1 wrist RGB and depth. No camera pane is a presentation
close-up, a replacement wrist view or a crop hiding the robot.

| External view | Direction / target | Distance | Elevation |
| --- | --- | --- | --- |
| LEFT / rig10 | Retained front-right low composition; initial substrate top centre | 0.60 m | 17.12394° |
| RIGHT / rig11 | Retained low side composition; initial substrate top centre | 0.60 m | 17.12394° |
| OPPOSITE TOOL / rig14 | Blade-tip side, material local +Y; horizontal optical projection is −Y | 0.60 m | 17.12394° |

The ruler knife edge frame has +Y along the blade from grip toward tip. The
material's long axis is local +X. OPPOSITE TOOL faces the vertical long side
at local `y=width/2`, aiming at its mid-height. Its horizontal bearing is normal
to that side, with **zero sideways obliquity**. Full 3D incidence relative to the
side normal is **17.12394°**, because the camera is slightly elevated. It does
not use the horizontal TRIM scoring plane's +Z normal. The initial workpiece
transform establishes the camera once; it never tracks the moving knife.

All three use an explicitly versioned **simulated 24° horizontal FOV, 640×480**
with actual render-product `K/T` readback. The nominal square-pixel focal length
is about 1505.482 px. The Global housing and optical prim actually move; no
digital enlargement establishes the viewpoint. Rig13's level candidate, rig12's
base-derived candidate and the earlier rig9 high-angle captures remain historical.
The no-argument camera default remains v5.

## RGB-D and synchronization

Both wrist cameras retain the v5 hand-local mount, identity optical rotation,
translation `[-0.085, 0.045, 0.020] m`, and their measured-hand world calibration.
Franka **arm_0** cuts; **arm_1** remains at its normal safe park. The parked wrist
can see little or no material; its actual RGB and depth are retained.

Wrist depth is **optical Z in meters**, valid from **0.07 to 0.50 m**. The fixed
blue→green→red scale has labels 0.07 / 0.285 / 0.50 m. Black means invalid/out of
range, never a filled estimate. These are provisional ideal D405 prototypes,
not calibrated physical devices.

Global panels publish **RGB only**. Independent native narrow-stereo controls
remain **BLOCKED**: plane error passes, but full-image coverage and feature gates
fail. Limited stereo overlap is a plausible explanation, not a calibration
conclusion. Original criteria and the source-correspondence guard remain intact.
No narrow Global registered depth or physical D455 calibration is claimed.
Debug native/ideal AOVs are retained separately; normalized registered depth is
invalid rather than silently substituting another signal.

Actual RGB render clipping is **0.001–20 m**, read back from the camera. The
native stereo valid-depth minimum **0.52 m is not a render near plane**. Nearby
arms, fingers, blade and housings remain visible as real occluders.

Each Global replay uses the original exact saved live mesh and the exact measured
robot/tool pose at each capture time, with zero physics steps. Original wrist
RGB-D is reused at that same measured-state timestamp. This is explicitly a
camera replay, not a simultaneous new five-camera physics recording. Capture
indices, physics/native clock offsets, fixed `K/T`, mesh hashes and body-matrix
readback are checked. No particles are removed or smoothed.

## Record once, display all views

Use a fresh output path and a free GPU. This maintained entry records with
supported original Global sensor optics and then replays all three RGB views:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/run_wp14_scene.py launch \
  --scene standard_cut --gpu 0 --camera-rig-profile wrist_camera_rig_v9 \
  --global-view all --out artifacts/my_three_view_scene_NEW
```

`prepare` performs only configuration preparation. `--global-view left`, `right`
or `opposite_tool` selects one display view without changing the recording
sensor or physical task. `--global-view all` keeps all three in the movie.

For an existing exact recording, avoid another material run:

```bash
.venv/bin/python scripts/render_native_global_views.py /path/to/episode_000 \
  --gpu 0 --out artifacts/my_three_view_replay_NEW
# Optional: --views right, or --no-encode for calibrated RGB captures only.
```

The underlying renderer also accepts `--global-view left/right/opposite_tool`.
Python geometry clients can use `camera_rig.GLOBAL_VIEW_PROFILES` and `make_rig`.
Normal live `open_cutting_env` recording should use supported v5/v8/v9 optics;
direct narrow Native stereo requests fail explicitly. Recording/display are
separate supported entry points, not a claim that narrow live RGB-D passed.

[Complete movies and honest historical-source handling](NATIVE_MULTIVIEW_GALLERY.md) ·
[Publication measurements and limitations](reports/THREE_GLOBAL_CAMERA_REFRESH.md) ·
[Media calibration, synchronization and source receipts](evidence/three_global/media_001.json) ·
[Native stereo control failure](evidence/three_global/narrow_controls_001.json).
