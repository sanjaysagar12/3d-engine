import json
import os
import numpy as np
import trimesh
from shapely.geometry import Polygon

from .pieces_to_glb import _vertices_to_coords, _random_color

class GarmentStitcher:
    """Pipeline step to assemble pattern pieces into a stitched 3D garment."""
    
    def __init__(self, metadata_path: str, stitching_path: str, extrusion_height: float = 2.0, gap: float = 300.0):
        self.metadata_path = metadata_path
        self.stitching_path = stitching_path
        self.extrusion_height = extrusion_height
        self.gap = gap
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        with open(stitching_path, 'r', encoding='utf-8') as f:
            self.stitching = json.load(f)

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

        # NOTE: seams are no longer filled with a solid triangle-strip mesh
        # here. Real sewing edges are added later by the Blender stitching
        # step (engine/pipeline/blender_stitcher.py), which operates on
        # flat, open-boundary panel meshes after scaling/placement.

        # Export
        out_dir = os.path.dirname(out_glb)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        glb_data = scene.export(file_type='glb')
        with open(out_glb, 'wb') as f:
            f.write(glb_data)
        
        print(f"[OK] Exported stitched garment to: {out_glb}")
        return out_glb
