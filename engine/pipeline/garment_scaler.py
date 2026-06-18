"""
Pipeline step: Scale garment pieces and place them around the SMPL-X avatar
using CLO-style arrangement points.

Each panel is positioned so that:
  - Z  -> at the body's front or back surface (from arrangement_points.json)
  - X  -> center seam (CF/CB) aligned to body centre-line (x = 0)
  - Y  -> panel top aligned near the shoulder/neck line

`run()` produces the solid (extruded) visualization GLB with the avatar.
`export_panels_for_stitching()` produces FLAT (single-layer, open-boundary)
panel meshes plus a seam_points.json — consumed by the Blender stitching
step, which subdivides the open boundaries and adds real sewing edges
(see engine/pipeline/blender_stitcher.py).
"""
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import yaml
from shapely.geometry import Polygon

from .pieces_to_glb import _vertices_to_coords, _random_color


class GarmentScaler:
    """Compute scale factors and produce a scaled, avatar-placed GLB assembly."""

    def __init__(
        self,
        metadata_path: str,
        stitching_path: str,
        measurements_path: str,
        avatar_obj_path: str,
        arrangement_points_json: Optional[str] = None,
        extrusion_height: float = 2.0,
        gap: float = 300.0,
    ):
        self.metadata_path = metadata_path
        self.stitching_path = stitching_path
        self.measurements_path = measurements_path
        self.avatar_obj_path = avatar_obj_path
        self.arrangement_points_json = arrangement_points_json
        self.extrusion_height = extrusion_height
        self.gap = gap

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(stitching_path, "r", encoding="utf-8") as f:
            self.stitching = json.load(f)
        self.measurements = self._load_measurements(measurements_path)

    @staticmethod
    def _load_measurements(path: str) -> dict:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        return data.get("body", data)

    # ------------------------------------------------------------------
    # Scale factor computation
    # ------------------------------------------------------------------
    def compute_scale_factors(self) -> Dict[str, Tuple[float, float]]:
        """Return {piece_name: (sx, sy)} scale factors."""
        ease = 1.05
        cm_to_mm = 10.0

        rules = {
            "front":       ("bust",        4.0, "height", 0.30),
            "back":        ("bust",        4.0, "height", 0.30),
            "armBinding":  ("armscye_depth", 1.0, "bust",  0.50),
            "neckBinding": ("neck_w",      1.0, "neck_w", 3.14159),
        }

        factors: Dict[str, Tuple[float, float]] = {}
        for piece_name, info in self.metadata.items():
            rule = None
            for key, val in rules.items():
                if key in piece_name:
                    rule = val
                    break
            if rule is None:
                factors[piece_name] = (1.0, 1.0)
                continue

            w_meas, w_div, h_meas, h_ratio = rule
            bounds = info.get("bounds", {})
            pat_w = bounds.get("width", 1.0)
            pat_h = bounds.get("height", 1.0)

            if w_meas in self.measurements:
                sx = (self.measurements[w_meas] * cm_to_mm / w_div) / pat_w * ease
            else:
                sx = 1.0

            if h_meas in self.measurements:
                sy = (self.measurements[h_meas] * cm_to_mm * h_ratio) / pat_h
            else:
                sy = 1.0

            factors[piece_name] = (sx, sy)
        return factors

    # ------------------------------------------------------------------
    # Mesh helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_2d_scale(mesh: trimesh.Trimesh, sx: float, sy: float):
        """Scale mesh in XY about its centroid without touching Z."""
        c = mesh.centroid.copy()
        mesh.apply_translation(-c)
        S = np.eye(4)
        S[0, 0] = sx
        S[1, 1] = sy
        mesh.apply_transform(S)
        mesh.apply_translation(c)

    def _build_panel_mesh(
        self, name: str, scale_factors: Dict[str, Tuple[float, float]], flat: bool = False
    ) -> Optional[trimesh.Trimesh]:
        """Build a single panel mesh (Y-flipped, scaled, NOT translated).

        flat=False -> solid extruded slab (for visualization GLBs).
        flat=True  -> single-layer triangulated sheet with real open
                      boundary edges (for the Blender stitching step).
        """
        info = self.metadata.get(name)
        if not info:
            return None
        coords = _vertices_to_coords(info.get("vertices", []))
        if len(coords) < 3:
            return None
        poly = Polygon(coords)

        try:
            if flat:
                verts2d, faces = trimesh.creation.triangulate_polygon(poly)
                verts3d = np.column_stack([verts2d, np.zeros(len(verts2d))])
                mesh = trimesh.Trimesh(vertices=verts3d, faces=faces, process=False)
            else:
                mesh = trimesh.creation.extrude_polygon(poly, height=self.extrusion_height)
        except Exception as e:
            print(f"  Error building mesh for '{name}': {e}")
            return None

        # SVG Y-down -> 3D Y-up
        rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        mesh.apply_transform(rot)
        sx, sy = scale_factors.get(name, (1.0, 1.0))
        self._apply_2d_scale(mesh, sx, sy)
        return mesh

    def _load_arrangement_points(self) -> Tuple[dict, dict]:
        ap_points: dict = {}
        body_dims: dict = {}
        if self.arrangement_points_json and os.path.exists(self.arrangement_points_json):
            with open(self.arrangement_points_json, "r", encoding="utf-8") as f:
                ap_json = json.load(f)
            ap_points = ap_json.get("points", {})
            body_dims = ap_json.get("body_dimensions", {})
        return ap_points, body_dims

    def _collect_panels(self) -> set:
        panels: set = set()
        for seam in self.stitching.get("seams", []):
            panels.add(seam["from"]["piece_a"])
            panels.add(seam["from"]["piece_b"])
            panels.add(seam["to"]["piece_a"])
            panels.add(seam["to"]["piece_b"])
        return panels

    def _compute_offset(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        ap_name: str,
        ap_points: dict,
        body_dims: dict,
        scale_factors: Dict[str, Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        ap = ap_points.get(ap_name, {})
        body_cx = body_dims.get("body_center_x", 0.0)
        # Neck joint sits above the shoulder ball-joint -- a better
        # "garment top" reference than the joint itself. Joints sit inside
        # the body, not at the visible skin surface, so add a margin to
        # bring the strap/neckline up to where the shoulder actually is.
        shoulder_y = body_dims.get("neck_y", body_dims.get("shoulder_y"))
        SHOULDER_SURFACE_MARGIN_MM = 40.0

        if ap:
            z_off = float(ap["position"][2])
            # RIGHT panels: CF/CB at mesh min-X. LEFT/mirror panels: CF/CB at mesh max-X.
            seam_x = mesh.bounds[1][0] if "LEFT" in ap_name else mesh.bounds[0][0]
            x_off = body_cx - seam_x
            y_off = (
                float(shoulder_y) + SHOULDER_SURFACE_MARGIN_MM - mesh.bounds[1][1]
                if shoulder_y is not None else 0.0
            )
        else:
            z_off = self.gap if "BACK" in ap_name else 0.0
            info_meta = self.metadata.get(name, {})
            sw = info_meta.get("bounds", {}).get("width", 240.0) * scale_factors.get(name, (1.0, 1.0))[0]
            if "LEFT" in ap_name:
                x_off = -sw - self.gap / 2.0
            elif "RIGHT" in ap_name:
                x_off = self.gap / 2.0
            else:
                x_off = 0.0
            y_off = 0.0

        return (x_off, y_off, z_off)

    # ------------------------------------------------------------------
    # Avatar loading
    # ------------------------------------------------------------------
    def _load_avatar(self) -> Optional[trimesh.Trimesh]:
        if not os.path.exists(self.avatar_obj_path):
            print(f"  [WARN] Avatar not found: {self.avatar_obj_path}")
            return None
        scene_or_mesh = trimesh.load(self.avatar_obj_path, force="mesh")
        if isinstance(scene_or_mesh, trimesh.Scene):
            scene_or_mesh = trimesh.util.concatenate(
                list(scene_or_mesh.geometry.values())
            )
        return scene_or_mesh

    # ------------------------------------------------------------------
    # Main entry — solid visualization GLB (panels + avatar, no seam fill)
    # ------------------------------------------------------------------
    def run(self, out_glb: str) -> str:
        scale_factors = self.compute_scale_factors()
        print("--- Scale Factors ---")
        for name, (sx, sy) in scale_factors.items():
            print(f"  {name}: sx={sx:.4f}  sy={sy:.4f}")

        ap_points, body_dims = self._load_arrangement_points()
        if ap_points:
            print(f"  Arrangement points loaded: {', '.join(ap_points.keys())}")

        panels = self._collect_panels()
        alignment = self.stitching.get("alignment", {})

        built_meshes: Dict[str, trimesh.Trimesh] = {}
        for name in panels:
            mesh = self._build_panel_mesh(name, scale_factors, flat=False)
            if mesh is not None:
                built_meshes[name] = mesh

        offsets: Dict[str, Tuple[float, float, float]] = {}
        for name, mesh in built_meshes.items():
            ap_name = alignment.get(name, "")
            offsets[name] = self._compute_offset(name, mesh, ap_name, ap_points, body_dims, scale_factors)

        scene = trimesh.Scene()
        for name, mesh in built_meshes.items():
            mesh.apply_translation(offsets[name])
            color = _random_color()
            mesh.visual.vertex_colors = np.tile(
                np.array(color, dtype=np.uint8), (len(mesh.vertices), 1)
            )
            scene.add_geometry(mesh, node_name=name)

        avatar = self._load_avatar()
        if avatar is not None:
            avatar.apply_scale(1000.0)
            if ap_points:
                print("  Avatar placed at SMPL-X origin (matches arrangement points)")
            else:
                all_max_y = max(m.bounds[1][1] for m in built_meshes.values()) if built_meshes else 0.0
                h_mm = self.measurements.get("height", 172.0) * 10.0
                hl_mm = self.measurements.get("head_l", 26.0) * 10.0
                y_shift = all_max_y - (h_mm - hl_mm)
                all_xs = []
                for m in built_meshes.values():
                    all_xs.extend([m.bounds[0][0], m.bounds[1][0]])
                garment_cx = (min(all_xs) + max(all_xs)) / 2.0 if all_xs else 0.0
                avatar_cx = (avatar.bounds[0][0] + avatar.bounds[1][0]) / 2.0
                avatar.apply_translation([garment_cx - avatar_cx, y_shift, self.gap / 2.0])
                print(f"  Avatar legacy-shifted: y={y_shift:.1f}  z={self.gap/2.0:.1f}")

            avatar.visual.vertex_colors = np.tile(
                np.array([200, 200, 200, 180], dtype=np.uint8),
                (len(avatar.vertices), 1),
            )
            scene.add_geometry(avatar, node_name="avatar")

        out_dir = os.path.dirname(out_glb)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_glb, "wb") as f:
            f.write(scene.export(file_type="glb"))
        print(f"[OK] Exported scaled garment with avatar to: {out_glb}")
        return out_glb

    # ------------------------------------------------------------------
    # Flat panels + seam point coordinates, for the Blender stitching step
    # ------------------------------------------------------------------
    def export_panels_for_stitching(
        self, out_panels_glb: str, out_seam_points_json: str
    ) -> Tuple[str, str]:
        """
        Export each panel as a flat, single-layer mesh (real open boundary
        edges) at its final scaled+placed world position, plus a JSON map
        of {piece: {point_id: [x, y, z]}} so the Blender script can locate
        seam endpoints without redoing any scale/offset math.
        """
        scale_factors = self.compute_scale_factors()
        ap_points, body_dims = self._load_arrangement_points()
        panels = self._collect_panels()
        alignment = self.stitching.get("alignment", {})

        built_meshes: Dict[str, trimesh.Trimesh] = {}
        for name in panels:
            mesh = self._build_panel_mesh(name, scale_factors, flat=True)
            if mesh is not None:
                built_meshes[name] = mesh

        offsets: Dict[str, Tuple[float, float, float]] = {}
        for name, mesh in built_meshes.items():
            ap_name = alignment.get(name, "")
            offsets[name] = self._compute_offset(name, mesh, ap_name, ap_points, body_dims, scale_factors)

        seam_points: Dict[str, Dict[str, list]] = {}
        scene = trimesh.Scene()
        for name, mesh in built_meshes.items():
            centroid = mesh.centroid.copy()
            sx, sy = scale_factors.get(name, (1.0, 1.0))
            off = np.array(offsets[name])

            # Polygon ring actually used to build the mesh (bezier-flattened).
            # Some "id" vertices are raw bezier control points that don't lie
            # on the curve itself -- snap each to its nearest ring coordinate
            # so it maps to a real mesh vertex instead of empty space.
            ring = np.array(_vertices_to_coords(self.metadata.get(name, {}).get("vertices", [])))

            pts: Dict[str, list] = {}
            for v in self.metadata.get(name, {}).get("vertices", []):
                if "id" not in v or "x" not in v or "y" not in v:
                    continue
                vx, vy = float(v["x"]), float(v["y"])
                if len(ring) > 0:
                    d = np.hypot(ring[:, 0] - vx, ring[:, 1] - vy)
                    vx, vy = ring[int(d.argmin())]
                flipped = np.array([vx, -vy, 0.0])
                scaled = centroid + (flipped - centroid) * np.array([sx, sy, 1.0])
                world = scaled + off
                pts[v["id"]] = [round(float(c), 3) for c in world]
            seam_points[name] = pts

            mesh.apply_translation(off)
            scene.add_geometry(mesh, node_name=name)

        out_dir = os.path.dirname(out_panels_glb)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_panels_glb, "wb") as f:
            f.write(scene.export(file_type="glb"))

        with open(out_seam_points_json, "w", encoding="utf-8") as f:
            json.dump(seam_points, f, indent=2)

        print(f"[OK] Exported flat panels for stitching: {out_panels_glb}")
        print(f"[OK] Exported seam points: {out_seam_points_json}")
        return out_panels_glb, out_seam_points_json
