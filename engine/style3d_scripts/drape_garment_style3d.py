"""
Style3D Studio script: import an avatar + garment, apply fabric physics,
run the cloth simulation to convergence, and export the draped result.

This is the Style3D counterpart to engine/blender_scripts/drape_garment.py +
stitch_garment.py, using Style3D's own `spy` Python API (Style3D Studio 5.x,
spy 1.3.9 -- see C:\\Program Files\\Style3D\\spy_documents\\html) instead of
Blender's bpy/Cloth modifier.

IMPORTANT -- hard API limitation (verified against the local spy docs and
C:\\Program Files\\Style3D\\ApplicationData\\McpServer\\PythonApi\\spy_api_detail.json):
`spy` has NO function to create a sew/connect-pair between two piece edges at
script time. `spy.GetClothPieceAllSewlineEdgeIds` / `spy.GetAllConnectPairIds`
can only *query* sewlines that already exist on imported pieces. Sewlines are
an attribute baked into the pattern at authoring time (a .pxf pattern, or a
full .sgar/.sproj project built in Style3D Studio's pattern editor).

This script therefore supports two garment sources:
  1. "sewn" sources (.sproj / .sgar / .pxf) that already carry sewlines --
     the normal, correct path. Sewing is NOT redone here; it's whatever the
     pattern file already has.
  2. Raw flat panel OBJ (the same panels_glb this pipeline feeds to Blender,
     converted to OBJ by Style3DDraper) as a fallback -- these import as
     UNSTITCHED loose cloth pieces. The script detects this (no sewline
     edges on any imported piece) and writes a loud warning into the status
     file rather than silently simulating a pile of unsewn cloth.

Run mode: Style3D Studio has no confirmed CLI flag for headless script
execution (unlike `blender --background --python`). This script is meant to
be run manually from Style3D Studio's built-in Python/Script console after
opening the app. It reads its config from a sidecar JSON file next to this
script (_drape_config.json), written ahead of time by
engine/pipeline/style3d_draper.py, and writes progress/result to
_drape_status.json next to it so external tooling can poll for completion.

config JSON keys (see engine/pipeline/style3d_draper.py for how it's built):
  avatar_obj                 - path to avatar mesh (obj/fbx)
  avatar_move_to_origin       - bool, default True
  avatar_skin_offset_mm       - float [0-10], default 2.0
  avatar_static_friction      - float [0-1], default 0.3
  avatar_dynamic_friction     - float [0-1], default 0.3
  garment_source               - path to a pre-sewn .sproj/.sgar/.pxf (preferred)
  garment_source_type          - "sproj" | "sgar" | "pxf" | "panels_obj"
  panels_obj                   - fallback flat-panel OBJ (UNSTITCHED)
  fabric                       - dict of spy.FabricPhysicalProperty fields
  simulation_mode              - "Normal" | "XGpu" | "Cpu", default "Normal"
  velocity_converged_bound     - float >= 0.01, default 0.5
  gravity_factor                - float >= 0.01, default 1.0
  simulation_timeout            - seconds, default 60.0
  out_glb                       - output GLB path
  out_dxf                       - optional output DXF path
"""
import json
import os

import spy

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "_drape_config.json")
_STATUS_PATH = os.path.join(_HERE, "_drape_status.json")

_DEFAULT_FABRIC = {
    "weight": 180.0,
    "thickness": 0.4,
    "stretchWarpRatio": 3000.0,
    "stretchWeftRatio": 3000.0,
    "stretchBiasRatio": 1500.0,
    "stretchWarpLinearity": 0.4,
    "stretchWeftLinearity": 0.4,
    "stretchBiasLinearity": 0.6,
    "bendingWarpRatio": 20.0,
    "bendingWeftRatio": 20.0,
    "bendingBiasRatio": 10.0,
    "staticFriction": 0.3,
    "dynamicFriction": 0.3,
}


def _load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        raise RuntimeError(
            f"No config found at {_CONFIG_PATH}. Run "
            f"engine/pipeline/style3d_draper.py's Style3DDraper.run() first "
            f"to write it."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_status(status: str, message: str, **extra):
    payload = {"status": status, "message": message, **extra}
    with open(_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[{status}] {message}")


def _import_avatar(cfg: dict) -> int:
    before = set(spy.GetAllAvatarIds())

    load_cfg = spy.MeshLoadConfig()
    load_cfg.type = spy.MeshLoadConfig.LoadType.LoadAsAvatar
    load_cfg.isMoveToOrigin = bool(cfg.get("avatar_move_to_origin", True))
    load_cfg.openFile = False

    ok = spy.ImportFbxOrObjFile(cfg["avatar_obj"], load_cfg)
    if not ok:
        raise RuntimeError(f"spy.ImportFbxOrObjFile failed for avatar: {cfg['avatar_obj']}")

    after = set(spy.GetAllAvatarIds())
    new_ids = list(after - before)
    if not new_ids:
        raise RuntimeError("Avatar import reported success but no new avatar id appeared")
    avatar_id = new_ids[0]

    spy.ChangeAvatarProperty(avatar_id, spy.AvatarPropertyType.BodyAttendSimulation, True)
    spy.ChangeAvatarProperty(
        avatar_id, spy.AvatarPropertyType.SkinOffset, float(cfg.get("avatar_skin_offset_mm", 2.0))
    )
    spy.ChangeAvatarProperty(
        avatar_id, spy.AvatarPropertyType.StaticFriction, float(cfg.get("avatar_static_friction", 0.3))
    )
    spy.ChangeAvatarProperty(
        avatar_id, spy.AvatarPropertyType.DynamicFriction, float(cfg.get("avatar_dynamic_friction", 0.3))
    )
    print(f"  Imported avatar id={avatar_id} from {cfg['avatar_obj']}")
    return avatar_id


def _import_garment(cfg: dict) -> tuple:
    """Returns (clothPieceIds, is_unstitched_fallback)."""
    before = set(spy.GetAllClothPieceIds())

    source = cfg.get("garment_source")
    source_type = cfg.get("garment_source_type")

    if source and source_type in ("sproj", "sgar", "pxf"):
        if source_type == "sproj":
            ok = spy.OpenProject(source)
        elif source_type == "sgar":
            ok = spy.OpenGarment(source)
        else:
            ok = spy.ImportPxfFile(source)
        if not ok:
            raise RuntimeError(f"Failed to load garment source ({source_type}): {source}")
        after = set(spy.GetAllClothPieceIds())
        piece_ids = list(after - before) or list(after)
        print(f"  Loaded pre-sewn garment source ({source_type}): {source} "
              f"-> {len(piece_ids)} piece(s)")
        return piece_ids, False

    panels_obj = cfg.get("panels_obj")
    if not panels_obj:
        raise RuntimeError(
            "config has neither a sewn garment_source (sproj/sgar/pxf) nor a "
            "panels_obj fallback -- nothing to import"
        )

    load_cfg = spy.MeshLoadConfig()
    load_cfg.type = spy.MeshLoadConfig.LoadType.LoadAsGarment
    load_cfg.openFile = False
    ok = spy.ImportFbxOrObjFile(panels_obj, load_cfg)
    if not ok:
        raise RuntimeError(f"spy.ImportFbxOrObjFile failed for panels: {panels_obj}")

    after = set(spy.GetAllClothPieceIds())
    piece_ids = list(after - before)
    print(f"  Imported UNSTITCHED flat panels: {panels_obj} -> {len(piece_ids)} piece(s)")
    return piece_ids, True


def _has_any_sewlines(piece_ids) -> bool:
    for pid in piece_ids:
        try:
            edges = spy.GetClothPieceAllSewlineEdgeIds(pid)
        except Exception:
            edges = []
        if edges:
            return True
    return False


def _apply_fabric(piece_ids, fabric_cfg: dict) -> int:
    fabric_id = spy.CreateFabric()
    prop = spy.FabricPhysicalProperty()
    merged = {**_DEFAULT_FABRIC, **(fabric_cfg or {})}
    for key, value in merged.items():
        if hasattr(prop, key):
            setattr(prop, key, value)
        else:
            print(f"  [WARN] Unknown FabricPhysicalProperty field '{key}', skipping")
    spy.ChangeFabricPhysicalProperty([fabric_id], prop)
    spy.ChangeClothPiecesFabric(piece_ids, fabric_id)
    print(f"  Applied fabric id={fabric_id} to {len(piece_ids)} piece(s)")
    return fabric_id


def _run_simulation(cfg: dict):
    mode_name = cfg.get("simulation_mode", "Normal")
    mode = getattr(spy.SimulationMode, mode_name, spy.SimulationMode.Normal)
    spy.SwitchSimulationMode(mode)

    velocity_bound = float(cfg.get("velocity_converged_bound", 0.5))
    gravity_factor = float(cfg.get("gravity_factor", 1.0))
    timeout = float(cfg.get("simulation_timeout", 60.0))

    print(f"  Simulating (mode={mode_name}, velocity_bound={velocity_bound}, "
          f"gravity_factor={gravity_factor}, timeout={timeout}s)...")
    result = spy.SimulationToConverge(velocity_bound, gravity_factor, timeout)
    spy.StopSimulation()

    result_name = getattr(result, "name", str(result))
    print(f"  Simulation result: {result_name}")
    return result_name


def _export(cfg: dict):
    out_glb = cfg["out_glb"]
    out_dir = os.path.dirname(out_glb)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    export_cfg = spy.ExportGltfConfig()
    export_cfg.targetObjectConfig.exportClothPieces = True
    export_cfg.targetObjectConfig.exportAvatars = True
    export_cfg.targetObjectConfig.exportProps = False
    export_cfg.targetObjectConfig.exportAssets = False
    export_cfg.transformConfig.unitType = spy.CommonLengthUnit.mm

    ok = spy.ExportGltfOrGlbFile(out_glb, export_cfg)
    if not ok:
        raise RuntimeError(f"spy.ExportGltfOrGlbFile failed: {out_glb}")
    print(f"  Exported draped garment + avatar to: {out_glb}")

    out_dxf = cfg.get("out_dxf")
    if out_dxf:
        out_dxf_dir = os.path.dirname(out_dxf)
        if out_dxf_dir and not os.path.exists(out_dxf_dir):
            os.makedirs(out_dxf_dir, exist_ok=True)
        if not spy.ExportDxfFile(out_dxf):
            print(f"  [WARN] spy.ExportDxfFile failed: {out_dxf}")
        else:
            print(f"  Exported DXF to: {out_dxf}")


def main():
    cfg = _load_config()

    avatar_id = _import_avatar(cfg)
    piece_ids, is_unstitched = _import_garment(cfg)
    if not piece_ids:
        raise RuntimeError("No cloth pieces were imported")

    sewn = _has_any_sewlines(piece_ids)
    if is_unstitched or not sewn:
        print("  [WARN] No sewline edges found on any imported piece -- the "
              "garment will simulate as separate UNSTITCHED cloth pieces. "
              "Provide a pre-sewn garment_source (.sproj/.sgar/.pxf) for a "
              "real stitched drape.")

    _apply_fabric(piece_ids, cfg.get("fabric", {}))
    result_name = _run_simulation(cfg)
    _export(cfg)

    _write_status(
        "OK",
        f"Drape complete (simulation={result_name})",
        avatar_id=avatar_id,
        cloth_piece_ids=piece_ids,
        stitched=bool(sewn and not is_unstitched),
        simulation_result=result_name,
        out_glb=cfg["out_glb"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _write_status("ERROR", str(exc))
