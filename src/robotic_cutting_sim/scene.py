"""Export a self-contained blade-only USD scene for inspection, without robot integration."""
import numpy as np

from .geometry import Specimen, sample_particles, specimen_dict
from .recording import write_json


def export(args, output):
    from pxr import Gf, Usd, UsdGeom, UsdLux, Vt
    from scipy.spatial.transform import Rotation
    from .camera import calibration
    from .trajectory import reference_trajectory

    spec = Specimen.generate(args.seed,args.variant,args.family)
    points,mass,_ = sample_particles(spec,args.spacing,1500)
    np.savez_compressed(output/"initial_particles.npz",positions_m=points,mass_kg=mass)
    info = specimen_dict(spec,points,mass,args.spacing)
    write_json(output/"specimen.json", info)
    stage = Usd.Stage.CreateNew(str(output/"physics_testbed.usda"))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
    root = UsdGeom.Xform.Define(stage,"/World"); stage.SetDefaultPrim(root.GetPrim())

    def box(name, center, half, color):
        cube = UsdGeom.Cube.Define(stage,f"/World/{name}"); cube.CreateSizeAttr(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center)); cube.AddScaleOp().Set(Gf.Vec3f(*half))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return cube
    box("Table",[0,0,-.03],[.4,.3,.02],[.28,.31,.35])
    box("TrayFloor",[0,0,-.005],[.068,.048,.005],[.35,.38,.42])
    for name,center,half in [("Left",[-.064,0,.0075],[.004,.044,.0075]),
                              ("Right",[.064,0,.0075],[.004,.044,.0075]),
                              ("Front",[0,-.044,.0075],[.06,.004,.0075]),
                              ("Back",[0,.044,.0075],[.06,.004,.0075])]:
        box("Tray"+name,center,half,[.35,.38,.42])
    pose,_ = reference_trajectory("through_bump_cut").at(0)
    box("KinematicBlade",pose[:3],[.001,.025,.010],[.65,.70,.76])
    # One closed mesh: top, bottom, and boundary walls.
    nx,ny = 61,41
    x,y = np.meshgrid(np.linspace(-spec.length/2,spec.length/2,nx),
                      np.linspace(-spec.width/2,spec.width/2,ny),indexing="ij")
    top = np.c_[x.ravel(),y.ravel(),spec.height(x,y).ravel()]
    bottom = top.copy(); bottom[:,2] = 0
    verts = np.vstack([top,bottom]); n = len(top); faces = []
    for i in range(nx-1):
        for j in range(ny-1):
            a=i*ny+j; b=a+ny; c=b+1; d=a+1
            faces += [(a,b,c),(a,c,d),(a+n,c+n,b+n),(a+n,d+n,c+n)]
    perimeter = ([i*ny for i in range(nx)] + [(nx-1)*ny+j for j in range(1,ny)]
                 + [i*ny+ny-1 for i in range(nx-2,-1,-1)] + [j for j in range(ny-2,0,-1)])
    for a,b in zip(perimeter,perimeter[1:]+perimeter[:1]):
        faces += [(a,b+n,b),(a,a+n,b+n)]
    mesh=UsdGeom.Mesh.Define(stage,"/World/ContinuousSpecimen")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces)); mesh.CreateFaceVertexIndicesAttr(np.array(faces).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none"); mesh.CreateDisplayColorAttr([Gf.Vec3f(.72,.32,.13)])
    samples=UsdGeom.Points.Define(stage,"/World/MPMSamples_debug")
    samples.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points)); samples.CreateWidthsAttr([args.spacing]*len(points))
    samples.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    cameras = {}
    for name,eye in [("IdealCamera0",[0,-.5,.55]),("IdealCamera1",[.5,-.4,.25]),("GlobalCamera",[.8,-.8,.8])]:
        calib=calibration(eye); cameras[name]=calib
        cv=np.array(calib["T_world_camera"]); rotation=cv[:3,:3]@np.diag([1,-1,-1])
        q=Rotation.from_matrix(rotation).as_quat()
        camera=UsdGeom.Camera.Define(stage,f"/World/{name}")
        camera.AddTranslateOp().Set(Gf.Vec3d(*eye)); camera.AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
        camera.CreateHorizontalApertureAttr(20.955); camera.CreateVerticalApertureAttr(20.955*480/640)
        camera.CreateFocalLengthAttr(float(20.955/(2*np.tan(np.deg2rad(87)/2))))
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10))
    light=UsdLux.DomeLight.Define(stage,"/World/EnvironmentLight"); light.CreateIntensityAttr(700)
    stage.GetRootLayer().customLayerData={"status":"STATIC_PHYSICS_TESTBED_PREVIEW",
        "simulation":"Run physics-probe CLI; opening this USD does not step MPM",
        "robot_integration":"Deferred until G2 and G3", "material":"uncalibrated"}
    stage.GetRootLayer().Save(); write_json(output/"cameras.json",cameras)
    result={"status":"PASS","scope":"static blade-only USD and continuous particle initialization",
            "scene":str(output/"physics_testbed.usda"),**info,"workcell_proven":False}
    write_json(output/"result.json",result)
    return result
