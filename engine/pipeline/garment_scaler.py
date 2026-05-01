"""
Pipeline step: Scale garment pieces based on avatar body measurements.

Reads scaling_instructions.json + measurements.yaml to compute per-piece
scale factors, then applies them during the stitching/assembly export.
"""
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import yaml

from .pieces_to_glb import _vertices_to_coords, _random_color


class GarmentScaler:
    """Compute scale factors and produce a scaled, avatar-centred GLB assembly."""

    def __init__(
        self,
        metadata_path: str,
        stitching_path: str,
        measurements_path: str,
        avatar_obj_path: str,
        extrusion_height: float = 2.0,
        gap: float = 300.0,
    ):
        self.metadata_path = metadata_path
        self.stitching_path = stitching_path
        self.measurements_path = measurements_path
        self.avatar_obj_path = avatar_obj_path
        self.extrusion_height = extrusion_height
        self.gap = gap

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(stitching_path, "r", encoding="utf-8") as f:
            self.stitching = json.load(f)
        with open(measurements_path, "r", encoding="utf-8") as f:
            self.measurements = yaml.safe_load(f).get("body", {})

    def compute_scale_factors(self) -> Dict[str, Tuple[float, float]]:
        """
        Return {piece_name: (sx, sy)} scale factors.
        Hardcoded mapping from piece name to body measurement rules.
        """
        ease = 1.05  # 5% ease for width
        cm_to_mm = 10.0

        # Define rules: {piece_substring: (width_meas, width_div, height_meas, height_ratio)}
        rules = {
            "front": ("bust", 4.0, "height", 0.30),
            "back": ("bust", 4.0, "height", 0.30),
            "armBinding": ("armscye_depth", 1.0, "bust", 0.50),
            "neckBinding": ("neck_w", 1.0, "neck_w", 3.14159),
        }

        factors: Dict[str, Tuple[float, float]] = {}

        for piece_name, info in self.metadata.items():
            # Find applicable rule (check if piece_name contains the key)
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

            # --- width ---
            if w_meas in self.measurements:
                target_w_mm = (self.measurements[w_meas] * cm_to_mm) / w_div
                sx = (target_w_mm / pat_w) * ease
            else:
                sx = 1.0

            # --- height ---
            if h_meas in self.measurements:
                target_h_mm = self.measurements[h_meas] * cm_to_mm * h_ratio
                sy = target_h_mm / pat_h
            else:
                sy = 1.0

            factors[piece_name] = (sx, sy)

        return factors

    # ------------------------------------------------------------------
    # Mesh helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_2d_scale(mesh: trimesh.Trimesh, sx: float, sy: float):
        """Scale a mesh in the XY plane (about its centroid) without touching Z."""
        c = mesh.centroid.copy()
        mesh.apply_translation(-c)
        S = np.eye(4)
        S[0, 0] = sx
        S[1, 1] = sy
        mesh.apply_transform(S)
        mesh.apply_translation(c)

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
                [g for g in scene_or_mesh.geometry.values()]
            )
        return scene_or_mesh

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self, out_glb: str) -> str:
        """Build the scaled, stitched garment with avatar and export to GLB."""
        scale_factors = self.compute_scale_factors()

        print("--- Scale Factors ---")
        for name, (sx, sy) in scale_factors.items():
            print(f"  {name}: sx={sx:.4f}  sy={sy:.4f}")

        # --- Panels ---
        panels = set()
        for seam in self.stitching.get("seams", []):
            panels.add(seam["from"]["piece_a"])
            panels.add(seam["from"]["piece_b"])
            panels.add(seam["to"]["piece_a"])
            panels.add(seam["to"]["piece_b"])

        alignment = self.stitching.get("alignment", {})
        offsets: Dict[str, Tuple[float, float, float]] = {}

        for name in panels:
            align = alignment.get(name, "")
            z_off = self.gap if "BACK" in align else 0.0
            x_off = 0.0
            info = self.metadata.get(name, {})
            width = info.get("bounds", {}).get("width", 240.0)
            sx, _ = scale_factors.get(name, (1.0, 1.0))
            scaled_width = width * sx

            if "LEFT" in align:
                x_off = -scaled_width - (self.gap / 2.0)
            elif "RIGHT" in align:
                x_off = self.gap / 2.0

            offsets[name] = (x_off, 0.0, z_off)

        scene = trimesh.Scene()

        for name in panels:
            info = self.metadata.get(name)
            if not info:
                continue
            from shapely.geometry import Polygon

            coords = _vertices_to_coords(info.get("vertices", []))
            if len(coords) < 3:
                continue
            poly = Polygon(coords)
            try:
                mesh = trimesh.creation.extrude_polygon(poly, height=self.extrusion_height)
            except Exception as e:
                print(f"  Error extruding '{name}': {e}")
                continue

            # Flip Y (SVG Y-down → 3D Y-up)
            rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            mesh.apply_transform(rot)

            # Apply per-piece scale
            sx, sy = scale_factors.get(name, (1.0, 1.0))
            self._apply_2d_scale(mesh, sx, sy)

            # Offset
            mesh.apply_translation(offsets.get(name, (0.0, 0.0, 0.0)))

            color = _random_color()
            mesh.visual.vertex_colors = np.tile(
                np.array(color, dtype=np.uint8), (len(mesh.vertices), 1)
            )
            scene.add_geometry(mesh, node_name=name)

        # --- Stitching seams (reuse intermediate-point logic) ---
        self._build_seam_meshes(scene, panels, offsets, scale_factors)

        # --- Avatar ---
        avatar = self._load_avatar()
        if avatar is not None:
            # Convert avatar from metres to mm (pattern units)
            # Pattern units are mm, avatar units are m
            m_to_mm = 1000.0
            avatar.apply_scale(m_to_mm)

            # Centre avatar between front (z=0) and back (z=gap) panels
            avatar_z = self.gap / 2.0

            # Align garment neck area with avatar neck
            # Find the max Y (top) across all panel meshes already in scene
            all_max_y = -1e9
            for geom_name in scene.geometry:
                g = scene.geometry[geom_name]
                all_max_y = max(all_max_y, g.bounds[1][1])

            # Use measurements to find avatar neck height (approx: height - head_l)
            h_mm = self.measurements.get("height", 172.0) * 10.0
            hl_mm = self.measurements.get("head_l", 26.0) * 10.0
            neck_y_mm = h_mm - hl_mm

            # We want avatar_neck + y_shift = garment_top
            # So y_shift = garment_top - avatar_neck
            y_shift = all_max_y - neck_y_mm

            # Centre avatar X at the midpoint of the garment
            all_xs = []
            for geom_name in scene.geometry:
                g = scene.geometry[geom_name]
                all_xs.extend([g.bounds[0][0], g.bounds[1][0]])
            garment_cx = (min(all_xs) + max(all_xs)) / 2.0 if all_xs else 0.0
            avatar_cx = (avatar.bounds[0][0] + avatar.bounds[1][0]) / 2.0
            x_shift = garment_cx - avatar_cx

            avatar.apply_translation([x_shift, y_shift, avatar_z])

            # Use a default grey color with some transparency
            acol = [200, 200, 200, 180]
            avatar.visual.vertex_colors = np.tile(
                np.array(acol, dtype=np.uint8), (len(avatar.vertices), 1)
            )
            scene.add_geometry(avatar, node_name="avatar")
            print(f"  Avatar placed at z={avatar_z:.1f}, y_shift={y_shift:.1f}, x_shift={x_shift:.1f}")

        # --- Export ---
        out_dir = os.path.dirname(out_glb)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        glb_data = scene.export(file_type="glb")
        with open(out_glb, "wb") as f:
            f.write(glb_data)
        print(f"[OK] Exported scaled garment with avatar to: {out_glb}")
        return out_glb

    # ------------------------------------------------------------------
    # Seam helpers (mirrors GarmentStitcher logic with scaling)
    # ------------------------------------------------------------------
    def _get_point_raw(self, piece: str, pt_id: str) -> Optional[np.ndarray]:
        info = self.metadata.get(piece)
        if not info:
            return None
        for v in info.get("vertices", []):
            if v.get("id") == pt_id:
                return np.array([float(v["x"]), -float(v["y"]), 0.0])
        return None

    def _get_edge_points(self, piece: str, from_id: str, to_id: str):
        """Walk the vertex ring and return all ID'd vertices between from and to."""
        info = self.metadata.get(piece)
        if not info:
            return []
        id_verts = [v for v in info.get("vertices", []) if "id" in v and "x" in v and "y" in v]
        if not id_verts:
            return []
        from_idx = to_idx = None
        for i, v in enumerate(id_verts):
            if v["id"] == from_id:
                from_idx = i
            if v["id"] == to_id:
                to_idx = i
        if from_idx is None or to_idx is None:
            return []
        n = len(id_verts)
        fwd, i = [], from_idx
        while True:
            fwd.append(i)
            if i == to_idx:
                break
            i = (i + 1) % n
        bwd, i = [], from_idx
        while True:
            bwd.append(i)
            if i == to_idx:
                break
            i = (i - 1) % n
        path = fwd if len(fwd) <= len(bwd) else bwd
        return [np.array([float(id_verts[j]["x"]), -float(id_verts[j]["y"]), 0.0]) for j in path]

    def _build_seam_meshes(self, scene, panels, offsets, scale_factors):
        """Create triangle-strip seam meshes with scaling applied."""
        for seam in self.stitching.get("seams", []):
            p_a = seam["from"]["piece_a"]
            p_b = seam["from"]["piece_b"]

            off_a = np.array(offsets.get(p_a, (0, 0, 0)), dtype=float)
            off_b = np.array(offsets.get(p_b, (0, 0, 0)), dtype=float)
            sx_a, sy_a = scale_factors.get(p_a, (1, 1))
            sx_b, sy_b = scale_factors.get(p_b, (1, 1))

            edge_a = self._get_edge_points(p_a, seam["from"]["point_a"], seam["to"]["point_a"])
            edge_b = self._get_edge_points(p_b, seam["from"]["point_b"], seam["to"]["point_b"])
            if len(edge_a) < 2 or len(edge_b) < 2:
                continue

            # Scale then offset each edge
            edge_a = [np.array([p[0] * sx_a, p[1] * sy_a, p[2]]) + off_a for p in edge_a]
            edge_b = [np.array([p[0] * sx_b, p[1] * sy_b, p[2]]) + off_b for p in edge_b]

            # Arc-length parametric resampling
            def arc_t(pts):
                d = [0.0]
                for i in range(1, len(pts)):
                    d.append(d[-1] + np.linalg.norm(pts[i] - pts[i - 1]))
                tot = d[-1] if d[-1] > 0 else 1.0
                return [x / tot for x in d]

            t_a, t_b = arc_t(edge_a), arc_t(edge_b)
            all_t = sorted(set(t_a + t_b))

            def interp(pts, tp, tq):
                out = []
                for t in tq:
                    for j in range(len(tp) - 1):
                        if tp[j] <= t <= tp[j + 1]:
                            seg = tp[j + 1] - tp[j]
                            a = (t - tp[j]) / seg if seg > 0 else 0.0
                            out.append(pts[j] * (1 - a) + pts[j + 1] * a)
                            break
                    else:
                        out.append(pts[-1].copy())
                return out

            sa = interp(edge_a, t_a, all_t)
            sb = interp(edge_b, t_b, all_t)
            n = len(all_t)
            verts, faces = [], []
            for i in range(n):
                verts.append(sa[i])
                verts.append(sb[i])
            for i in range(n - 1):
                ai, bi, an, bn = 2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1
                faces.append([ai, an, bn])
                faces.append([ai, bn, bi])
            if verts and faces:
                sm = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
                sm.visual.vertex_colors = [200, 50, 50, 255]
                scene.add_geometry(sm, node_name=f"seam_{seam['seam_id']}")
