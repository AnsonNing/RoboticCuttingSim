# Coordinates and data contract

All geometry and dynamics use metres, kilograms, seconds and Pascals. World is
right handed, +Z up, tray floor at Z=0, specimen centered in XY. Quaternions are
XYZW at every public CSV/NumPy boundary. Newton spatial velocity is linear XYZ
then angular XYZ in radians/s. Tool poses locate the box center; a scrape box has
half extents (1,25,10) mm, with lower cutting edge 10 mm below that center. The
indentation punch has half extents (10,10,5) mm. CRESSim's slicer plane is at the
Newton box lower edge; this is a different collision representation.

CSV trajectory columns are `time_s,x_m,y_m,z_m,qx,qy,qz,qw`. Positions interpolate
linearly, rotations with SLERP. Poses are clamped beyond the endpoints. Fresh
experiments construct a fresh model, state buffers and solver, clearing numerical
history rather than just moving particles back.

`physics_state.npz`: positions/velocities `(F+1,N,3)` float32, times `(F+1,)` s,
constant particle masses `(N,)` kg, tool poses `(F+1,7)` XYZW, tool half extents m.
Newton final stress, elastic strain and plastic-volume Jp are oracle diagnostics;
Jp=1 does not exclude isochoric plastic shear. Older pre-fix artifacts with F tool
poses are identified in `artifacts/RUN_VALIDITY.json` and are not acceptance data.

`trace.csv` includes pre- and post-projection **particle-center** blade penetration,
maximum positional projection correction, and solver contact impulse / frame dt.
Projection correction is not included in the force estimate, and neither metric
is a full material-surface penetration measurement. Forces are debug/calibration
data only. No future vision observation exposes them.

The camera convention is OpenCV (+X right, +Y down, +Z forward).
`T_world_camera` maps camera coordinates into world. `K` uses pixel centers at
(u+0.5,v+0.5). RGB is uint8 H×W×3; depth is float32 H×W optical Z in **metres**;
invalid depth is 0 and `valid_mask=depth>0`. Surface extraction preserves physical
timestamp. RGB-D is stored separately from oracle state/surface files.

USD cameras use -Z forward/+Y up. The explicit conversion is
`R_world_usd = R_world_cv @ diag(1,-1,-1)`; no implicit quaternion reordering.
