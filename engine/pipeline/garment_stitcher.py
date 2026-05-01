import json
import os
from typing import Dict, List, Set, Tuple
import numpy as np
import trimesh
from shapely.geometry import Polygon

from .pieces_to_glb import _vertices_to_coords, _random_color

class GarmentStitcher:
    """Pipeline step to assemble pattern pieces into a stitched 3D garment."""
    
    def __init__(self, metadata_path: str, stitching_path: str, extrusion_height: float = 2.0, gap: float = 50.0):
        self.metadata_path = metadata_path
        self.stitching_path = stitching_path
        self.extrusion_height = extrusion_height
        self.gap = gap
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        with open(stitching_path, 'r', encoding='utf-8') as f:
            self.stitching = json.load(f)

    def get_point_coords_raw(self, piece_name: str, point_id: str) -> np.ndarray:
        """Find the 3D coordinates of a point ID in a piece without global offsets."""
        piece_info = self.metadata.get(piece_name)
        if not piece_info:
            return None
            
        for v in piece_info.get('vertices', []):
            if v.get('id') == point_id:
                # Flip Y (SVG Y-down to 3D Y-up)
                return np.array([float(v['x']), -float(v['y']), 0.0])
        return None

    def get_edge_points(self, piece_name: str, from_pt_id: str, to_pt_id: str) -> List[np.ndarray]:
        """
        Collect all vertices (with IDs) between from_pt_id and to_pt_id on a piece,
        walking through the vertex list in index order (wrapping around if needed).
        Returns a list of 3D coords (Y-flipped) including both endpoints.
        """
        piece_info = self.metadata.get(piece_name)
        if not piece_info:
            return []

        # Build ordered list of only vertices that have an 'id' and x/y coords
        id_verts = []
        for v in piece_info.get('vertices', []):
            if 'id' in v and 'x' in v and 'y' in v:
                id_verts.append(v)

        if not id_verts:
            return []

        # Find indices of from and to points
        from_idx = None
        to_idx = None
        for i, v in enumerate(id_verts):
            if v['id'] == from_pt_id:
                from_idx = i
            if v['id'] == to_pt_id:
                to_idx = i

        if from_idx is None or to_idx is None:
            return []

        n = len(id_verts)

        # Walk forward from from_idx to to_idx (wrapping), collect points
        forward = []
        i = from_idx
        while True:
            forward.append(i)
            if i == to_idx:
                break
            i = (i + 1) % n

        # Walk backward (i.e. the other direction around the ring)
        backward = []
        i = from_idx
        while True:
            backward.append(i)
            if i == to_idx:
                break
            i = (i - 1) % n

        # Choose the shorter path
        path = forward if len(forward) <= len(backward) else backward

        # Convert to 3D coords (Y-flipped)
        coords = []
        for idx in path:
            v = id_verts[idx]
            coords.append(np.array([float(v['x']), -float(v['y']), 0.0]))
        return coords

    def run(self, out_glb: str):
        # Identify panels mentioned in stitching JSON
        panels = set()
        for seam in self.stitching.get('seams', []):
            panels.add(seam['from']['piece_a'])
            panels.add(seam['from']['piece_b'])
            panels.add(seam['to']['piece_a'])
            panels.add(seam['to']['piece_b'])
        
        print(f"--- Assembling Garment ---")
        print(f"Panels: {panels}")
        
        # Determine alignment offsets
        alignment = self.stitching.get('alignment', {})
        self.offsets = {}
        
        for name in panels:
            align = alignment.get(name, "")
            z_off = 200.0 if "BACK" in align else 0.0
            x_off = 0.0
            
            if "LEFT" in align:
                info = self.metadata.get(name, {})
                width = info.get('bounds', {}).get('width', 240.0)
                x_off = -width - (self.gap / 2.0)
            elif "RIGHT" in align:
                x_off = self.gap / 2.0
                
            self.offsets[name] = (x_off, 0.0, z_off)

        scene = trimesh.Scene()
        
        for name in panels:
            info = self.metadata.get(name)
            if not info:
                continue
                
            coords = _vertices_to_coords(info.get('vertices', []))
            if len(coords) < 3:
                continue
                
            poly = Polygon(coords)
            try:
                mesh = trimesh.creation.extrude_polygon(poly, height=self.extrusion_height)
            except Exception as e:
                print(f"Error extruding '{name}': {e}")
                continue
                
            # Transformations: Flip Y and Apply Offset
            rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            mesh.apply_transform(rotation)
            mesh.apply_translation(self.offsets.get(name, (0.0, 0.0, 0.0)))
            
            # Color
            color = _random_color()
            mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
            
            scene.add_geometry(mesh, node_name=name)

        # Create stitching meshes using intermediate edge points
        for seam in self.stitching.get('seams', []):
            p_a = seam['from']['piece_a']
            p_b = seam['from']['piece_b']
            
            off_a = np.array(self.offsets.get(p_a, (0.0, 0.0, 0.0)))
            off_b = np.array(self.offsets.get(p_b, (0.0, 0.0, 0.0)))
            
            # Collect all edge points between from and to on each piece
            edge_a = self.get_edge_points(p_a, seam['from']['point_a'], seam['to']['point_a'])
            edge_b = self.get_edge_points(p_b, seam['from']['point_b'], seam['to']['point_b'])
            
            if len(edge_a) < 2 or len(edge_b) < 2:
                print(f"  [WARN] Seam {seam['seam_id']}: insufficient edge points "
                      f"(piece_a={len(edge_a)}, piece_b={len(edge_b)}), skipping.")
                continue

            # Apply offsets
            edge_a = [pt + off_a for pt in edge_a]
            edge_b = [pt + off_b for pt in edge_b]

            # Resample: pair points from the longer edge to the shorter edge
            # using parametric (arc-length) interpolation so the strip is smooth.
            def arc_lengths(pts):
                """Compute cumulative arc-length parameter [0..1] for a polyline."""
                dists = [0.0]
                for i in range(1, len(pts)):
                    dists.append(dists[-1] + np.linalg.norm(pts[i] - pts[i - 1]))
                total = dists[-1] if dists[-1] > 0 else 1.0
                return [d / total for d in dists]

            t_a = arc_lengths(edge_a)
            t_b = arc_lengths(edge_b)

            # Merge both parameter sets into a unified sequence
            all_t = sorted(set(t_a + t_b))

            def interp_polyline(pts, t_params, t_query):
                """Interpolate a polyline at given parameter values."""
                result = []
                for tq in t_query:
                    # Find the segment
                    for j in range(len(t_params) - 1):
                        if t_params[j] <= tq <= t_params[j + 1]:
                            seg_len = t_params[j + 1] - t_params[j]
                            alpha = (tq - t_params[j]) / seg_len if seg_len > 0 else 0.0
                            result.append(pts[j] * (1 - alpha) + pts[j + 1] * alpha)
                            break
                    else:
                        # tq is at the very end
                        result.append(pts[-1].copy())
                return result

            strip_a = interp_polyline(edge_a, t_a, all_t)
            strip_b = interp_polyline(edge_b, t_b, all_t)

            n_pts = len(all_t)
            print(f"  Seam {seam['seam_id']}: {len(edge_a)} pts on {p_a}, "
                  f"{len(edge_b)} pts on {p_b} -> {n_pts} paired strip points")

            # Build triangle strip: for each consecutive pair of stations,
            # create a quad (2 triangles) between the two edges.
            verts = []
            faces = []
            for i in range(n_pts):
                verts.append(strip_a[i])
                verts.append(strip_b[i])

            for i in range(n_pts - 1):
                # Indices: a_i = 2*i, b_i = 2*i+1, a_next = 2*(i+1), b_next = 2*(i+1)+1
                ai = 2 * i
                bi = 2 * i + 1
                an = 2 * (i + 1)
                bn = 2 * (i + 1) + 1
                faces.append([ai, an, bn])
                faces.append([ai, bn, bi])

            if verts and faces:
                seam_mesh = trimesh.Trimesh(
                    vertices=np.array(verts), faces=np.array(faces))
                seam_mesh.visual.vertex_colors = [200, 50, 50, 255]
                scene.add_geometry(seam_mesh, node_name=f"seam_mesh_{seam['seam_id']}")

        # Export
        out_dir = os.path.dirname(out_glb)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        glb_data = scene.export(file_type='glb')
        with open(out_glb, 'wb') as f:
            f.write(glb_data)
        
        print(f"[OK] Exported stitched garment to: {out_glb}")
        return out_glb
