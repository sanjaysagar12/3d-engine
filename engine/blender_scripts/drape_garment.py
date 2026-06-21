"""
Blender background script: run Blender's Cloth simulation on an already-
stitched garment GLB, draping it onto the SMPL-X avatar.

Stitching happens in a separate prior Blender call (stitch_garment.py).
This script only handles physics: import, scale, pin, simulate, export.

Run via: blender --background --python drape_garment.py -- <config.json>

config.json keys:
  stitched_glb      - path to the stitched garment GLB (from stitch_garment.py)
  avatar_glb        - path to the avatar GLB (converted from avatar.obj by
                      ClothDraper so axis conversion matches garment panels)
  out_glb           - output path for the draped result (garment + avatar)
  blend_out         - optional .blend output with baked physics cache
  sim_frames        - number of simulation frames (default 150)
  sewing_force_max  - max sewing spring force in N (default 3.0)
  quality           - cloth solver substeps per frame (default 10)
  mass              - cloth vertex mass in kg (default 0.3)

Pinning: top-edge vertices on the FRONT side of the garment (negative Blender
Y after Y-up->Z-up import = positive SMPL-X Z = body front) are pinned.
This is determined by vertex position rather than vertex groups (vertex groups
are not preserved across GLB export/import). Pinning only the front side lets
the sewing springs close the shoulder seam — pinning both sides would freeze
each panel's top at its starting position and prevent seam closure.
"""
import json
import os
import sys

import bpy

MM_TO_M = 0.001
M_TO_MM = 1000.0


def _load_config() -> dict:
    argv = sys.argv
    if "--" not in argv or argv.index("--") + 1 >= len(argv):
        raise RuntimeError("Missing config JSON path after '--'")
    with open(argv[argv.index("--") + 1], "r", encoding="utf-8") as f:
        return json.load(f)


def _clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)


def _import_glb(path: str, name: str):
    bpy.ops.import_scene.gltf(filepath=path)
    # Filter to mesh objects only — GLBs can also import empties/armatures.
    # Use the filtered list directly (not bpy.context.view_layer.objects.active
    # which may point to a non-mesh node after import).
    objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not objs:
        raise RuntimeError(f"No mesh imported from {path}")
    if len(objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = objs[0]
    obj.name = name
    return obj


def _reconnect_sewing_edges(obj, threshold=0.01):
    """glTF exports wire edges as a separate LINES primitive with its own
    vertex buffer, so importing back creates duplicate vertices at the same
    positions that are not connected to the face mesh. Merge by distance
    welds them so the cloth simulator sees the sewing edges as actually
    attached to the garment boundary and can pull the panels together."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    result = bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  Merged duplicate vertices (threshold={threshold}): {result}")


def _scale_and_apply(objects, factor: float):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
        obj.scale = (factor, factor, factor)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _setup_collision(avatar_obj):
    mod = avatar_obj.modifiers.new(name="Collision", type='COLLISION')
    mod.settings.thickness_outer = 0.01
    mod.settings.thickness_inner = 0.05
    mod.settings.cloth_friction = 6.0
    mod.settings.damping = 0.3
    return mod


def _pin_back_top_edge(garment_obj, pin_depth_m: float = 0.02):
    """Pin top-edge vertices on the BACK (positive Blender Y) side only.

    After Y-up→Z-up import: positive Blender Y = negative SMPL-X Z = body back.
    Back panels rest against the body through collision, so their top edge is
    already near the correct shoulder position — pin it as the stable anchor.
    This lets sewing springs freely pull the FRONT shoulder toward the back,
    closing the shoulder seams.  Pinning the FRONT top was the old behaviour;
    it froze the front shoulder in its flat starting position and prevented the
    shoulder seam from closing.

    Called AFTER _scale_and_apply(MM_TO_M) so coordinates are in metres.
    """
    z_max = max(v.co.z for v in garment_obj.data.vertices)
    threshold_z = z_max - pin_depth_m

    vg = garment_obj.vertex_groups.new(name="Pin")
    pinned = [
        v.index for v in garment_obj.data.vertices
        if v.co.z >= threshold_z and v.co.y > 0  # y > 0 = back in Blender Z-up
    ]
    vg.add(pinned, 1.0, 'REPLACE')
    print(f"  Pinned {len(pinned)} back top-edge vertices "
          f"(z >= {threshold_z:.4f} m, y > 0)")
    return vg


def _setup_cloth(garment_obj, sewing_force_max, quality, mass):
    mod = garment_obj.modifiers.new(name="Cloth", type='CLOTH')
    cs = mod.settings
    cs.quality = quality
    cs.mass = mass
    cs.use_sewing_springs = True
    cs.sewing_force_max = sewing_force_max
    # No vertex-group pinning: both front and back shoulder straps must be free
    # to fall onto the avatar's physical shoulder surface (via collision) and then
    # be pulled together by sewing springs. Pinning either side froze that panel's
    # strap at its flat starting position, preventing the shoulder seam from closing.
    cs.effector_weights.gravity = 0.4  # reduced so springs close seams before garment slides far
    mod.collision_settings.use_self_collision = False
    mod.collision_settings.use_collision = True
    mod.collision_settings.friction = 8.0
    mod.collision_settings.distance_min = 0.01
    mod.collision_settings.impulse_clamp = 5.0
    mod.collision_settings.collision_quality = 5
    return mod


def _run_simulation(scene, cloth_mod, frame_end: int, use_disk_cache: bool = False):
    scene.frame_start = 1
    scene.frame_end = frame_end
    cloth_mod.point_cache.frame_start = 1
    cloth_mod.point_cache.frame_end = frame_end
    if use_disk_cache:
        cloth_mod.point_cache.use_disk_cache = True
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(frame_end)
    bpy.context.view_layer.update()


def _bake_static_mesh(obj, name: str):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(eval_obj)
    new_obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(new_obj)
    new_obj.scale = (1.0, 1.0, 1.0)
    new_obj.location = (0.0, 0.0, 0.0)
    new_obj.rotation_euler = (0.0, 0.0, 0.0)
    return new_obj


def _save_blend(blend_path: str, scene, reset_to_frame: int = 1):
    out_dir = os.path.dirname(blend_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    scene.frame_set(reset_to_frame)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[OK] Saved Blender scene (with baked cache) to: {blend_path}")


def main():
    cfg = _load_config()
    stitched_glb = cfg["stitched_glb"]
    avatar_glb = cfg["avatar_glb"]
    out_glb = cfg["out_glb"]
    blend_out = cfg.get("blend_out")
    sim_frames = int(cfg.get("sim_frames", 150))
    sewing_force_max = float(cfg.get("sewing_force_max", 3.0))
    quality = int(cfg.get("quality", 10))
    mass = float(cfg.get("mass", 0.3))

    _clear_scene()

    garment_obj = _import_glb(stitched_glb, "garment")
    # Wire edges (sewing seams) come from a separate glTF LINES primitive with
    # its own vertex buffer. After import their endpoints are floating copies
    # that share positions with face mesh boundary vertices but are not
    # connected to them. Merging by distance welds them so cloth sewing springs
    # can actually pull the garment panels together.
    _reconnect_sewing_edges(garment_obj, threshold=0.01)
    avatar_obj = _import_glb(avatar_glb, "avatar")

    # Garment is in mm (from stitch step); scale to metres for physics.
    # Avatar GLB is already in metres — do NOT rescale it.
    _scale_and_apply([garment_obj], MM_TO_M)

    _setup_collision(avatar_obj)
    cloth_mod = _setup_cloth(garment_obj, sewing_force_max, quality, mass)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = sim_frames

    if blend_out:
        out_dir = os.path.dirname(blend_out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=blend_out)

    print(f"  Running cloth simulation for {sim_frames} frames "
          f"(sewing_force_max={sewing_force_max}, quality={quality})...")
    _run_simulation(scene, cloth_mod, sim_frames, use_disk_cache=bool(blend_out))
    print("  Simulation complete.")

    if blend_out:
        _save_blend(blend_out, scene, reset_to_frame=1)
        scene.frame_set(sim_frames)

    draped = _bake_static_mesh(garment_obj, "garment_draped")
    avatar_static = _bake_static_mesh(avatar_obj, "avatar_static")

    _scale_and_apply([draped, avatar_static], M_TO_MM)

    bpy.ops.object.select_all(action='DESELECT')
    garment_obj.select_set(True)
    avatar_obj.select_set(True)
    bpy.ops.object.delete(use_global=False)

    out_dir = os.path.dirname(out_glb)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    draped.select_set(True)
    avatar_static.select_set(True)
    bpy.context.view_layer.objects.active = draped
    bpy.ops.export_scene.gltf(
        filepath=out_glb, use_selection=True, export_yup=True, use_mesh_edges=True
    )
    print(f"[OK] Exported draped garment + avatar to: {out_glb}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
