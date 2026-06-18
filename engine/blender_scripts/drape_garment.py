"""
Blender background script: stitch garment panels (boundary subdivision +
real sewing edges), then run Blender's Cloth simulation -- with sewing
springs enabled -- to drape the garment onto the SMPL-X avatar.

Run via: blender --background --python drape_garment.py -- <config.json>

config.json keys:
  panels_glb        - flat, open-boundary panel meshes (one object per piece)
  seam_points_json  - {piece: {point_id: [x, y, z]}} world coordinates
  stitching_json    - seam definitions (same format as stitching_instructions.json)
  avatar_glb        - path to the avatar GLB (collision target; converted
                      from avatar.obj by ClothDraper so axis conversion on
                      import matches the garment panels exactly)
  out_glb           - output path for the draped result (garment + avatar)
  blend_out         - optional .blend output preserving the live Cloth/
                      Collision setup (not a baked static mesh), with the
                      simulation cache baked to disk so opening the file
                      and scrubbing/playing the timeline shows the result
                      instantly, no recompute needed
  wrinkle_subdivide_cuts - whole-mesh subdivision after stitching, giving
                      the cloth solver enough geometry for smooth wrinkles
                      instead of sharp low-poly creases (default 3)
  sim_frames        - number of simulation frames to step (default 180;
                      side seams take a long, collision-obstructed path to
                      close so they need more frames than a simple drop)
  sewing_force_max  - max force pulling sewn edges together (default 3.0;
                      values much above ~3 destabilize the solver and the
                      cloth explodes within a few frames -- verified by
                      bisection)
  quality           - cloth solver substeps per frame (default 20)
  mass              - cloth vertex mass in kg (default 0.3)

Only the top edge of the FRONT panels is pinned (not back) -- pinning both
sides of a shoulder seam freezes them at their separate starting depths and
the seam can never close. Avatar collision stays on for the whole
simulation (tried toggling it off during seam-closing and back on after;
that collapses the garment flat against the pinned side instead of
wrapping around the body, since nothing forces the free side to go AROUND
rather than straight through). With collision on throughout, the seams
take longer to close (some residual gap remains even at 180 frames) but
the wrap-around shape is preserved.
"""
import json
import os
import sys

import bpy
import bmesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_garment import build_stitched_garment, _clear_scene, _piece_vert_indices  # noqa: E402


# Garment data is authored in millimetres; Blender's physics (gravity,
# collision thickness, default stiffness) assume metres. Scale down before
# simulating, then scale the final static result back up for export.
MM_TO_M = 0.001
M_TO_MM = 1000.0


def _load_config() -> dict:
    argv = sys.argv
    if "--" not in argv or argv.index("--") + 1 >= len(argv):
        raise RuntimeError("Missing config JSON path after '--'")
    with open(argv[argv.index("--") + 1], "r", encoding="utf-8") as f:
        return json.load(f)


def _import_avatar(avatar_glb_path: str):
    # Import via glTF (not OBJ) so the Y-up -> Z-up axis conversion is
    # applied reliably and matches the garment panels' own import path.
    # Blender's OBJ importer's up_axis/forward_axis args do NOT reliably
    # rotate the data the same way -- verified empirically.
    bpy.ops.import_scene.gltf(filepath=avatar_glb_path)
    objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not objs:
        raise RuntimeError(f"No mesh imported from {avatar_glb_path}")
    avatar = objs[0]
    avatar.name = "avatar"
    return avatar


def _scale_and_apply(objects, factor: float):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
        obj.scale = (factor, factor, factor)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _setup_collision(avatar_obj):
    mod = avatar_obj.modifiers.new(name="Collision", type='COLLISION')
    # A too-thin buffer (was 0.001 = 1mm) lets the fast-falling cloth
    # tunnel past the surface before the response triggers, so the
    # eventual correction is large and sudden -- producing sharp
    # accordion-style creases instead of a smooth drape.
    mod.settings.thickness_outer = 0.01
    mod.settings.thickness_inner = 0.05
    # Moderate friction: enough to damp jittering/creasing on contact, but
    # not so much the cloth sticks in place instead of sliding down into
    # a natural drape (15 was too sticky -- the garment sat too high).
    mod.settings.cloth_friction = 6.0
    mod.settings.damping = 0.3
    return mod


def _pin_top_edge(garment_obj, anchor_pieces=("front", "front_mirror"), pin_depth_m: float = 0.02):
    """Pin vertices near the garment's top edge (shoulder/neck line) so
    gravity doesn't slide the whole tube down to the avatar's feet --
    there's nothing else (no bent arms, no friction tuning) holding it up.

    Only pins vertices belonging to `anchor_pieces` (front by default).
    Pinning BOTH sides of a shoulder seam (front's top AND back's top)
    would freeze them at their separate starting positions and the seam
    could never close -- pinning only one side lets the sewing spring
    still pull the other side's top edge to meet it.
    """
    anchor_verts = set()
    for piece in anchor_pieces:
        anchor_verts |= _piece_vert_indices(garment_obj, piece)

    z_max = max(v.co.z for v in garment_obj.data.vertices)
    vg = garment_obj.vertex_groups.new(name="Pin")
    pinned = [
        v.index for v in garment_obj.data.vertices
        if v.co.z >= z_max - pin_depth_m and v.index in anchor_verts
    ]
    vg.add(pinned, 1.0, 'REPLACE')
    print(f"  Pinned {len(pinned)} top-edge vertices on {anchor_pieces} (z >= {z_max - pin_depth_m:.4f})")
    return vg


def _setup_cloth(garment_obj, sewing_force_max, quality, mass):
    mod = garment_obj.modifiers.new(name="Cloth", type='CLOTH')
    cs = mod.settings
    cs.quality = quality
    cs.mass = mass
    cs.use_sewing_springs = True
    cs.sewing_force_max = sewing_force_max
    cs.vertex_group_mass = "Pin"
    cs.pin_stiffness = 1.0
    # Full gravity (1.0 = -9.81 m/s^2) builds up too much downward velocity
    # by the time the cloth reaches the avatar, producing sharp accordion
    # creases on impact -- but cutting it all the way to 0.3 left the
    # garment sitting too high, not draping down naturally. 0.6 is the
    # middle ground: soft enough to avoid the violent impact, strong
    # enough to actually pull the hem down over the body.
    cs.effector_weights.gravity = 0.6
    # Self-collision destabilizes the solver here (verified empirically --
    # the cloth explodes within ~15 frames with it on). Leave it off;
    # avatar collision is what actually matters for draping.
    mod.collision_settings.use_self_collision = False
    # Collision stays ON from frame 1. Turning it off to let the seams
    # close unobstructed (then back on) was tried and rejected: with no
    # collision, the unanchored side of each seam takes the straight-line
    # path through where the avatar would be and the whole garment
    # collapses flat against the pinned (front) side, instead of wrapping
    # around the body -- collision afterwards then just pushes that flat
    # collapsed sheet further away rather than spreading it back out.
    # Keeping collision on the whole time forces the seam to close by
    # going AROUND the body, preserving the wrap-around shape.
    mod.collision_settings.use_collision = True
    mod.collision_settings.friction = 6.0
    mod.collision_settings.distance_min = 0.01
    mod.collision_settings.impulse_clamp = 5.0    # cap any single correction
    mod.collision_settings.collision_quality = 5  # default of 2 is too coarse
    return mod


def _run_simulation(scene, cloth_mod, frame_end: int, use_disk_cache: bool = False):
    """Run the cloth sim via the bake operator (not manual frame_set
    stepping) so the result is left as a populated point cache -- on disk
    if use_disk_cache, so a saved .blend can play it back instantly."""
    scene.frame_start = 1
    scene.frame_end = frame_end
    # The point cache has its own frame_start/frame_end -- it does NOT
    # auto-sync with the scene's, so bake_all silently uses its default
    # (250) unless these are set explicitly (verified empirically).
    cloth_mod.point_cache.frame_start = 1
    cloth_mod.point_cache.frame_end = frame_end
    if use_disk_cache:
        cloth_mod.point_cache.use_disk_cache = True
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(frame_end)
    bpy.context.view_layer.update()


def _bake_static_mesh(obj, name: str):
    """Evaluate modifiers at the current frame and return a new, plain
    static object (no modifiers, no physics cache) holding the result."""
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
    """Save a .blend with the live Cloth/Collision setup (not a baked
    static mesh). The point cache (already baked to disk by
    _run_simulation) lets the saved file play back instantly when the
    timeline is scrubbed or Play is pressed -- no recompute needed."""
    out_dir = os.path.dirname(blend_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    scene.frame_set(reset_to_frame)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[OK] Saved Blender scene (with baked cache) to: {blend_path}")


def main():
    cfg = _load_config()
    avatar_glb_path = cfg["avatar_glb"]
    out_glb = cfg["out_glb"]
    blend_out = cfg.get("blend_out")
    boundary_subdivide_cuts = int(cfg.get("boundary_subdivide_cuts", 8))
    wrinkle_subdivide_cuts = int(cfg.get("wrinkle_subdivide_cuts", 2))
    sim_frames = int(cfg.get("sim_frames", 180))
    sewing_force_max = float(cfg.get("sewing_force_max", 3.0))
    quality = int(cfg.get("quality", 20))
    mass = float(cfg.get("mass", 0.3))

    _clear_scene()

    garment_obj = build_stitched_garment(
        cfg["panels_glb"], cfg["seam_points_json"], cfg["stitching_json"],
        boundary_subdivide_cuts=boundary_subdivide_cuts,
        wrinkle_subdivide_cuts=wrinkle_subdivide_cuts,
    )
    avatar_obj = _import_avatar(avatar_glb_path)

    # The garment panels are authored in millimetres; rescale into metres so
    # cloth physics (gravity, mass, collision thickness) behave sensibly.
    # avatar_glb is already in metres natively (SMPL-X's own units, and the
    # glTF importer doesn't rescale) -- do NOT rescale it, or collision
    # becomes meaningless (avatar would shrink to ~1mm).
    _scale_and_apply([garment_obj], MM_TO_M)

    _pin_top_edge(garment_obj)
    _setup_collision(avatar_obj)
    cloth_mod = _setup_cloth(garment_obj, sewing_force_max, quality, mass)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = sim_frames

    if blend_out:
        # bpy.data.filepath must be set BEFORE the bake runs, or the disk
        # cache writes to a temp location instead of next to our file.
        out_dir = os.path.dirname(blend_out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=blend_out)

    print(f"  Running cloth simulation for {sim_frames} frames "
          f"(sewing_force_max={sewing_force_max}, quality={quality})...")
    _run_simulation(scene, cloth_mod, sim_frames, use_disk_cache=bool(blend_out))
    print("  Simulation complete.")

    if blend_out:
        # Save now, before any of the static-mesh-baking below adds extra
        # objects to the scene -- the .blend should contain just the
        # garment + avatar + their modifiers, ready to scrub/play.
        _save_blend(blend_out, scene, reset_to_frame=1)
        scene.frame_set(sim_frames)

    draped = _bake_static_mesh(garment_obj, "garment_draped")
    avatar_static = _bake_static_mesh(avatar_obj, "avatar_static")

    # Back to millimetres for consistency with the rest of the pipeline.
    _scale_and_apply([draped, avatar_static], M_TO_MM)

    # Remove the simulated/source objects, keep only the static results.
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
    main()
