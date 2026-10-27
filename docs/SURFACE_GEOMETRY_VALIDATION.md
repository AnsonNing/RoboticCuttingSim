# Independent ParticleSurface geometry controls

The fixed half-density profile passes twelve independent known-volume controls.
The current default surface overestimates their volume by 8.25–12.06%. This
control changes neither cutting physics nor the default renderer, and cannot
promote TASK-02. A separate D0 quadrature audit identifies a larger source of
nominal-plane error at 2 mm sampling.

## Frozen recipe and actual controls

Commit `6b02ba4` freezes
[`particle_surface_geometry_v1.json`](../configs/particle_surface_geometry_v1.json)
before extraction, SHA256
`9c0f77dcf661b8a0ef3ff03fa274483e37f4032ef4a390599bb70e8aa966e4d1`.
Four closed boxes include two held-out rotations/aspect ratios, each at
1/1.5/2 mm particle spacing. Particle mass divided by density exactly equals
the analytic volume; sampling uses complete cell centers. All six box faces
provide known planes. Rotations test the extraction voxel grid and translations
test its phase. They do not test arbitrary plane orientation relative to the
particle lattice, compressed/sheared materials or curved bumps.

Pinned Newton 1.6.1 source uses a normalized cubic B-spline density field with
weight `8 * particle_radius**3`. A symmetric kernel over a continuum half-space
has density 0.5 on its physical boundary. H0 is the existing 1 mm voxel/3 mm
kernel-radius/0.25-threshold profile. H1 is fixed before results:

```python
ParticleSurface(voxel_size=spacing/2, kernel_radius=3*spacing,
                threshold=0.5, kernel_scale=0.5, smooth_lambda=0,
                anisotropic=False, field_smooth_iterations=0,
                mesh_smooth_iterations=0)
# Input radius = spacing/2, so the kernel weight volume is spacing**3.
```

All sharp-edge/corner triangles remain in full-mesh SDF and volume statistics.
The separately reported face-center bias has a predefined three-spacing edge
margin. Promotion requires a closed oriented manifold, one component, Euler
characteristic 2, volume error ≤5%, full area-weighted p95 boundary error≤half a
particle spacing, and bounded face/vertex errors. No candidate search, global
volume fit, target-plane fit or cut-result selection is used.

| Spacing | H0 worst volume inflation | H1 worst volume deficit | H1 whole-surface p95 | H1 worst face-center mean bias |
|---|---:|---:|---:|---:|
| 1 mm | 12.06% | 0.767% | 0.234 mm | 0.35 μm |
| 1.5 mm | 11.51% | 1.685% | 0.467 mm | 2.95 μm |
| 2 mm | 10.41% | 2.976% | 0.689 mm | 0.80 μm |

H1 passes all twelve controls, including all held-out cases. H0 fails all twelve;
the failed results remain in the receipt. H1 rounds sharp corners because kernel
density is a convolution, even with explicit mesh/field smoothing disabled.
Its remaining volume deficit is reported rather than scaled away. Warm H1 GPU0
extraction takes 3.4–5.5 ms for these small fixtures; the initial H0 extraction
takes 96 ms with cached kernels. These times exclude mesh QA and storage and are
not live-cut performance measurements.

Actual environment: Newton 1.6.1 at commit
`713fecdc41caf0c9d726f5c016939f36e66e3dff`, Warp 1.17.0, RTX 5090 GPU0,
driver 570.211.01. Raw NPZ meshes, masses, protocol, manifest and exact source
byte snapshots are in `artifacts/particle_surface_geometry_001`.
[Complete control receipt](evidence/particle_surface_geometry_audit.json).

## Quadrature error is distinct from rendering bias

The actual default D0 sampler represents each retained point with a full
`spacing**3` reference volume. Its analytic cosine-dome extra volume is
`2*pi*bump_height*bump_radius**2*(1/4-1/pi**2)`.

| D0 spacing | Actual particles | Reference volume error vs analytic D0 | Flat-base full-cell top |
|---|---:|---:|---:|
| 1 mm | 145,656 | −0.0175% | 15 mm |
| 1.5 mm | 42,890 | −0.6368% | 15 mm; width 79.5 mm from floor division |
| 2 mm | 19,340 | +6.2043% | 16 mm |

At 2 mm, base centers are z=1,3,…,15 mm and there are 19,200 base points. Their
full-cell quadrature support extends to 16 mm, although nominal base height is
15 mm. This is a volume interpretation consistent with the density kernel,
not a physical-boundary oracle. An honest half-density reconstruction must not
be shifted down to 15 mm just to pass flatness. Correcting the renderer is not
equivalent to correcting this sampling error or cutting the entire bump.

CPU evidence and exact sampler snapshot are in
`artifacts/reference_quadrature_001`;
[quadrature receipt](evidence/reference_quadrature_audit.json).

Dynamic deformation needs additional controls before adopting H1 globally:
constant input radii represent reference volume and ignore `det(F)`. Current
particle volume, anisotropic deformation, thin chips and near-contact separate
surfaces are not validated here. Existing live baselines remain immutable.

```bash
.venv/bin/python -m pytest tests/test_surface_controls.py -q
.venv/bin/python scripts/audit_particle_surface_geometry.py artifacts/surface_controls_fresh --device cuda:0
.venv/bin/python scripts/audit_reference_quadrature.py artifacts/quadrature_fresh
```

Two meaningful CPU geometry tests pass, covering full-cell mass, analytic SDF,
signed mesh volume/manifold validity and a missing-triangle failure.
