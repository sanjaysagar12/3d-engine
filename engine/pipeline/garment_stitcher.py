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

        # Create stitching meshes (quads)
        for seam in self.stitching.get('seams', []):
            p_a = seam['from']['piece_a']
            p_b = seam['from']['piece_b']
            
            off_a = self.offsets.get(p_a, (0.0, 0.0, 0.0))
            off_b = self.offsets.get(p_b, (0.0, 0.0, 0.0))
            
            c1_base = self.get_point_coords_raw(p_a, seam['from']['point_a'])
            c2_base = self.get_point_coords_raw(p_b, seam['from']['point_b'])
            c3_base = self.get_point_coords_raw(p_a, seam['to']['point_a'])
            c4_base = self.get_point_coords_raw(p_b, seam['to']['point_b'])
            
            if all(v is not None for v in [c1_base, c2_base, c3_base, c4_base]):
                c1 = c1_base + np.array(off_a)
                c2 = c2_base + np.array(off_b)
                c3 = c3_base + np.array(off_a)
                c4 = c4_base + np.array(off_b)
                
                vertices = np.array([c1, c3, c4, c2])
                faces = np.array([[0, 1, 2], [0, 2, 3]])
                seam_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
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
