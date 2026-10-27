"""USD render-purpose filtering including instanced asset collision proxies."""


def non_render_purpose_paths(stage):
    from pxr import Usd, UsdGeom
    return [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
            if UsdGeom.Imageable(prim) and UsdGeom.Imageable(prim).ComputePurpose() not in ("default", "render")]
