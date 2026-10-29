"""Native USD task-fixture authoring plan (ADR 0007 section 1).

The pure part (``fixture_authoring_plan``) decides prim paths, display
colours and PhysX contact materials from the task and scenario, so it is
testable without Isaac. Boxes without a declared material (v1/v2 scenarios)
keep the historical colour and no physics material; the runtime then takes
exactly the historical authoring path. ``bind_physics_material`` needs only
``pxr`` (usd-core or Kit).
"""
LEGACY_FIXTURE_COLOR = (.22, .27, .31)
FIXTURE_ROOT = "/World/TaskFixture"
MATERIAL_ROOT = "/World/TaskFixtureMaterials"
TABLE_PATH = "/World/Table"


def fixture_authoring_plan(task, scenario=None):
    """Per-box USD authoring rows plus the declared table material (or None).

    Each row: index, prim_path, name, role, center_m, half_extents_m,
    display_color_rgb, material_declared and physics_material (None or
    {prim_path, static_friction, dynamic_friction, restitution}).
    """
    from .task_scenarios import fixture_materials, table_material
    boxes = tuple(task.fixture_boxes_local)
    materials = fixture_materials(task)
    if len(materials) != len(boxes):
        raise ValueError("Fixture materials must align with the task fixture boxes")
    rows = []
    for index, (box, material) in enumerate(zip(boxes, materials)):
        if material["name"] != box.name:
            raise ValueError("Fixture material rows must follow the fixture box order")
        declared = bool(material["material_declared"])
        physics = None
        if declared:
            physics = {"prim_path": f"{MATERIAL_ROOT}/Box_{index}",
                       **{key: float(material[key]) for key in ("static_friction", "dynamic_friction", "restitution")}}
        rows.append({"index": index, "prim_path": f"{FIXTURE_ROOT}/Box_{index}", "name": box.name,
                     "role": material["role"], "support_name": material["support_name"],
                     "center_m": [float(v) for v in box.center_m],
                     "half_extents_m": [float(v) for v in box.half_extents_m],
                     "display_color_rgb": [float(v) for v in (material["display_color_rgb"] if declared
                                                              else LEGACY_FIXTURE_COLOR)],
                     "material_declared": declared, "physics_material": physics})
    table = table_material(task, scenario)
    if table is None and scenario is None and getattr(task, "table_contact_material", None) is not None:
        # Configured scene (scene_config.ConfiguredTaskSpec): its workcell declares the table material.
        table = task.table_contact_material
    table_row = None if table is None else {"prim_path": TABLE_PATH, "physics_material": {
        "prim_path": f"{MATERIAL_ROOT}/Table", **{key: float(value) for key, value in table.items()}}}
    return {"boxes": rows, "table": table_row, "scenario": scenario,
            "any_material_declared": any(row["material_declared"] for row in rows) or table_row is not None,
            "scope": "PhysX static/dynamic friction and restitution bound with the physics material purpose; "
                     "visual box equals the collider box; TLMPM fixture contacts use the task material friction"}


def bind_physics_material(stage, prim, material_row):
    """Define a UsdPhysics material and bind it to ``prim`` for the physics purpose."""
    from pxr import UsdPhysics, UsdShade
    material = UsdShade.Material.Define(stage, material_row["prim_path"])
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(material_row["static_friction"]))
    physics.CreateDynamicFrictionAttr(float(material_row["dynamic_friction"]))
    physics.CreateRestitutionAttr(float(material_row["restitution"]))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    return material


def authored_material_receipt(stage, plan):
    """Read back the authored USD physics materials and bindings (for setup.json)."""
    from pxr import UsdPhysics, UsdShade

    def readback(path):
        prim = stage.GetPrimAtPath(path)
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        if not bound:
            return {"prim_path": path, "physics_material": None}
        api = UsdPhysics.MaterialAPI(bound.GetPrim())
        return {"prim_path": path, "physics_material": str(bound.GetPath()),
                "static_friction": float(api.GetStaticFrictionAttr().Get()),
                "dynamic_friction": float(api.GetDynamicFrictionAttr().Get()),
                "restitution": float(api.GetRestitutionAttr().Get()),
                "collision_api": prim.HasAPI(UsdPhysics.CollisionAPI)}

    rows = []
    for row in plan["boxes"]:
        rows.append({"name": row["name"], "role": row["role"], "display_color_rgb": row["display_color_rgb"],
                     **readback(row["prim_path"])})
    return {"boxes": rows, "table": readback(TABLE_PATH) if plan["table"] else None,
            "source": "USD readback after authoring", "scope": plan["scope"]}
