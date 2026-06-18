"""
Blender background script: stitch garment panels together with real sewing
edges (Bridge Edge Loops -> delete faces, keep edges) and subdivide the
whole mesh afterward so the cloth solver has enough geometry to form
smooth wrinkles instead of sharp, low-poly creases.

This mirrors a manually-verified Blender workflow, with one addition: a
short straight-line boundary (e.g. a single edge from corner to corner)
bridges into only 2 connector rungs -- the endpoints -- with nothing
pulling the *middle* of that seam together, so it bows open into a visible
V-gap. Boundary edges are pre-subdivided into many segments before
bridging so every seam gets dozens of pull points along its whole length,
not just its two ends.

  1. Join front/back/front_mirror/back_mirror into one "garment" mesh.
  2. Subdivide every open boundary edge (not the interior faces) into N
     segments, so each seam has many points to bridge, not just 2.
  3. For each seam, select piece_a's boundary edges + piece_b's boundary
     edges, Bridge Edge Loops, then delete-only-faces -- leaving just the
     bridging edges as sewing lines.
  4. Select the whole mesh and Subdivide again (a couple cuts) for
     wrinkle detail on the interior faces.
  5. Cloth modifier with "Sewing" (use_sewing_springs) enabled; Collision
     on the avatar.

Run via: blender --background --python stitch_garment.py -- <config.json>

config.json keys:
  panels_glb              - flat, open-boundary panel meshes (one object per piece)
  seam_points_json        - {piece: {point_id: [x, y, z]}} world coordinates
  stitching_json          - seam definitions (same format as stitching_instructions.json)
  out_glb                 - output path for the stitched mesh
  boundary_subdivide_cuts  - per-edge boundary subdivision before bridging,
                             so each seam gets many pull points (default 8)
  wrinkle_subdivide_cuts   - whole-mesh subdivision cuts after stitching,
                             for interior wrinkle detail (default 2)
"""
import json
import os
import sys

import bpy
import bmesh
from mathutils import Vector


# ----------------------------------------------------------------------
# Pipeline helpers
# ----------------------------------------------------------------------
def _load_config() -> dict:
    argv = sys.argv
    if "--" not in argv or argv.index("--") + 1 >= len(argv):
        raise RuntimeError("Missing config JSON path after '--'")
    config_path = argv[argv.index("--") + 1]
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)


def _import_panels(panels_glb: str):
    bpy.ops.import_scene.gltf(filepath=panels_glb)
    return [o for o in bpy.context.selected_objects if o.type == 'MESH']


def _tag_vertex_groups(objects):
    """Tag each object's vertices with a group named after the piece, so
    piece identity survives the later join() into a single mesh."""
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        vg = obj.vertex_groups.get(obj.name) or obj.vertex_groups.new(name=obj.name)
        vg.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')


def _apply_transforms(objects):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _join_objects(objects, target_name="garment"):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = target_name
    return merged


def _subdivide_boundaries(obj, number_of_cuts=8):
    """Subdivide only the open boundary edges (not interior faces) into
    evenly spaced segments, so a seam that was originally a single
    corner-to-corner edge gets many pull points instead of just the 2
    endpoints once bridged."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        bmesh.ops.subdivide_edges(bm, edges=boundary_edges, cuts=number_of_cuts, use_grid_fill=False)
        bmesh.update_edit_mesh(obj.data)
        print(f"  Subdivided {len(boundary_edges)} boundary edges ({number_of_cuts} cuts each)")
    bpy.ops.object.mode_set(mode='OBJECT')


def _piece_vert_indices(obj, piece_name) -> set:
    vg = obj.vertex_groups.get(piece_name)
    if vg is None:
        return set()
    idx = vg.index
    return {
        v.index for v in obj.data.vertices
        if any(g.group == idx and g.weight > 0.5 for g in v.groups)
    }


def _to_blender_space(xyz):
    """Convert a glTF Y-up coordinate to Blender's native Z-up space.
    Blender's gltf importer bakes this conversion into vertex data on
    import: (x, y, z)_gltf -> (x, -z, y)_blender."""
    x, y, z = xyz
    return Vector((x, -z, y))


def _nearest_vert(bm, verts_subset, target_xyz):
    best, best_d = None, None
    for vi in verts_subset:
        bv = bm.verts[vi]
        d = (bv.co - target_xyz).length
        if best_d is None or d < best_d:
            best, best_d = bv, d
    return best


def _walk_boundary(bm, piece_verts, start_v, end_v):
    """Walk a piece's boundary loop from start_v to end_v in both possible
    directions; return the shorter ordered vertex list (incl. endpoints)."""
    adj: dict = {}
    for e in bm.edges:
        if not e.is_boundary:
            continue
        v0, v1 = e.verts[0], e.verts[1]
        if v0.index in piece_verts and v1.index in piece_verts:
            adj.setdefault(v0.index, []).append(v1.index)
            adj.setdefault(v1.index, []).append(v0.index)

    def walk(branch):
        neighbors = adj.get(start_v.index, [])
        if branch >= len(neighbors):
            return None
        path = [start_v.index]
        # prev starts at start_v itself so the first hop never doubles back.
        prev = start_v.index
        cur = neighbors[branch]
        steps = 0
        while steps < 100000:
            path.append(cur)
            if cur == end_v.index:
                return path
            following = [n for n in adj.get(cur, []) if n != prev]
            if not following:
                return None
            prev, cur = cur, following[0]
            steps += 1
        return None

    candidates = [p for p in (walk(0), walk(1)) if p]
    if not candidates:
        return None
    best = min(candidates, key=len)
    return [bm.verts[i] for i in best]


def _edge_between(v0, v1):
    for e in v0.link_edges:
        if e.other_vert(v0) is v1:
            return e
    return None


def _path_to_edges(path_verts):
    """Convert an ordered boundary vertex path into the chain of edges
    connecting consecutive vertices."""
    edges = []
    for i in range(len(path_verts) - 1):
        e = _edge_between(path_verts[i], path_verts[i + 1])
        if e is None:
            return None
        edges.append(e)
    return edges


def _select_only(bm, edges):
    """Clear all selection, then select exactly these edges (and their
    vertices) -- the state bpy.ops.mesh.bridge_edge_loops() reads from."""
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    for e in edges:
        e.select = True
        for v in e.verts:
            v.select = True
    bm.select_flush(True)


def build_stitched_garment(
    panels_glb: str, seam_points_path: str, stitching_path: str,
    boundary_subdivide_cuts: int = 8,
    wrinkle_subdivide_cuts: int = 2,
):
    """Import flat panels, join into one mesh, pre-subdivide boundary edges
    so each seam has many pull points, bridge each seam's boundary edges
    into real sewing lines, then subdivide the whole mesh once more for
    interior wrinkle resolution. Returns the merged 'garment' object (not
    exported). Caller owns the scene."""
    with open(seam_points_path, "r", encoding="utf-8") as f:
        seam_points = json.load(f)
    with open(stitching_path, "r", encoding="utf-8") as f:
        stitching = json.load(f)

    objs = _import_panels(panels_glb)
    if not objs:
        raise RuntimeError(f"No mesh objects imported from {panels_glb}")

    for o in objs:
        o.name = o.name.split('.')[0]

    _tag_vertex_groups(objs)
    _apply_transforms(objs)
    merged = _join_objects(objs, target_name="garment")

    if boundary_subdivide_cuts > 0:
        _subdivide_boundaries(merged, boundary_subdivide_cuts)

    # Vertex indices are stable across the bridge/delete operators used
    # below (they only add/remove faces, never verts), so this cache,
    # computed once before entering edit mode (and after the boundary
    # subdivide above, so it includes the new vertices), stays valid for
    # every seam.
    piece_vert_cache = {name: _piece_vert_indices(merged, name) for name in seam_points}

    bpy.context.view_layer.objects.active = merged
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)  # edge select
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(merged.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    stitched = 0
    total_seams = len(stitching.get("seams", []))
    for seam in stitching.get("seams", []):
        seam_id = seam.get("seam_id", "?")
        pa, pb = seam["from"]["piece_a"], seam["from"]["piece_b"]
        pa_from_id, pa_to_id = seam["from"]["point_a"], seam["to"]["point_a"]
        pb_from_id, pb_to_id = seam["from"]["point_b"], seam["to"]["point_b"]

        pa_pts, pb_pts = seam_points.get(pa, {}), seam_points.get(pb, {})
        if pa_from_id not in pa_pts or pa_to_id not in pa_pts:
            print(f"  [WARN] Seam {seam_id}: missing point ids on {pa}, skipping")
            continue
        if pb_from_id not in pb_pts or pb_to_id not in pb_pts:
            print(f"  [WARN] Seam {seam_id}: missing point ids on {pb}, skipping")
            continue

        verts_a = piece_vert_cache.get(pa, set())
        verts_b = piece_vert_cache.get(pb, set())

        v_a_start = _nearest_vert(bm, verts_a, _to_blender_space(pa_pts[pa_from_id]))
        v_a_end   = _nearest_vert(bm, verts_a, _to_blender_space(pa_pts[pa_to_id]))
        v_b_start = _nearest_vert(bm, verts_b, _to_blender_space(pb_pts[pb_from_id]))
        v_b_end   = _nearest_vert(bm, verts_b, _to_blender_space(pb_pts[pb_to_id]))

        if not all([v_a_start, v_a_end, v_b_start, v_b_end]):
            print(f"  [WARN] Seam {seam_id}: could not locate endpoint vertices, skipping")
            continue

        path_a = _walk_boundary(bm, verts_a, v_a_start, v_a_end)
        path_b = _walk_boundary(bm, verts_b, v_b_start, v_b_end)
        if not path_a or not path_b:
            print(f"  [WARN] Seam {seam_id}: boundary walk failed, skipping")
            continue

        edges_a = _path_to_edges(path_a)
        edges_b = _path_to_edges(path_b)
        if not edges_a or not edges_b:
            print(f"  [WARN] Seam {seam_id}: could not resolve boundary edges, skipping")
            continue

        # Select piece_a's boundary chain + piece_b's boundary chain, then
        # Bridge Edge Loops (creates a face strip connecting them) and
        # immediately delete only the faces -- leaving the bridging edges
        # as real sewing lines, exactly like the manually-verified workflow.
        _select_only(bm, edges_a + edges_b)
        bmesh.update_edit_mesh(merged.data)

        bpy.ops.mesh.bridge_edge_loops()
        bpy.ops.mesh.delete(type='ONLY_FACE')

        # The bridge/delete operators rebuild the edit mesh internally, so
        # any previously-held bmesh handle is stale -- refresh it.
        bm = bmesh.from_edit_mesh(merged.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        print(f"  Seam {seam_id}: bridged {pa}<->{pb} "
              f"({len(edges_a)} / {len(edges_b)} boundary edges)")
        stitched += 1

    print(f"Stitched {stitched}/{total_seams} seams")

    # Give the cloth solver enough geometry to form smooth wrinkles/folds
    # instead of sharp, low-poly creases -- subdivides the panel faces AND
    # the new loose sewing edges together in one pass.
    if wrinkle_subdivide_cuts > 0:
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=wrinkle_subdivide_cuts)
        print(f"  Subdivided whole mesh ({wrinkle_subdivide_cuts} cuts) for wrinkle resolution")

    bpy.ops.object.mode_set(mode='OBJECT')
    return merged


def main():
    cfg = _load_config()
    out_glb = cfg["out_glb"]

    _clear_scene()
    merged = build_stitched_garment(
        cfg["panels_glb"], cfg["seam_points_json"], cfg["stitching_json"],
        boundary_subdivide_cuts=int(cfg.get("boundary_subdivide_cuts", 8)),
        wrinkle_subdivide_cuts=int(cfg.get("wrinkle_subdivide_cuts", 2)),
    )

    out_dir = os.path.dirname(out_glb)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    bpy.ops.export_scene.gltf(
        filepath=out_glb, use_selection=True, export_yup=True, use_mesh_edges=True
    )
    print(f"[OK] Exported stitched garment (Blender) to: {out_glb}")


if __name__ == "__main__":
    main()
