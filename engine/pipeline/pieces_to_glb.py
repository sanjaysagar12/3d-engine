import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon

def _vertices_to_coords(vertices: List[Dict]) -> List[Tuple[float, float]]:
    """Convert vertex list with Bezier controls into discrete points."""
    def sample_cubic(p0, p1, p2, p3, steps=16):
        points = []
        for step in range(1, steps + 1):
            t = step / float(steps)
            mt = 1.0 - t
            x = (
                mt ** 3 * p0[0]
                + 3.0 * mt ** 2 * t * p1[0]
                + 3.0 * mt * t ** 2 * p2[0]
                + t ** 3 * p3[0]
            )
            y = (
                mt ** 3 * p0[1]
                + 3.0 * mt ** 2 * t * p1[1]
                + 3.0 * mt * t ** 2 * p2[1]
                + t ** 3 * p3[1]
            )
            points.append((x, y))
        return points

    coords = []
    current = None
    index = 0
    while index < len(vertices):
        vertex = vertices[index]
        vtype = vertex.get('type')
        if 'x' not in vertex or 'y' not in vertex:
            index += 1
            continue

        point = (float(vertex['x']), float(vertex['y']))

        if vtype == 'move':
            current = point
            coords.append(point)
            index += 1
            continue

        if vtype == 'line':
            current = point
            if not coords or coords[-1] != point:
                coords.append(point)
            index += 1
            continue

        if vtype == 'control1' and index + 2 < len(vertices):
            control1 = point
            control2_vertex = vertices[index + 1]
            end_vertex = vertices[index + 2]
            if (
                control2_vertex.get('type') == 'control2'
                and end_vertex.get('type') == 'curve_end'
                and 'x' in control2_vertex and 'y' in control2_vertex
                and 'x' in end_vertex and 'y' in end_vertex
                and current is not None
            ):
                control2 = (float(control2_vertex['x']), float(control2_vertex['y']))
                end_point = (float(end_vertex['x']), float(end_vertex['y']))
                coords.extend(sample_cubic(current, control1, control2, end_point))
                current = end_point
                index += 3
                continue

        index += 1

    # Ensure closed
    if len(coords) > 1 and coords[0] != coords[-1]:
        coords.append(coords[0])

    return coords


def _random_color() -> Tuple[int, int, int, int]:
    """Generate a random RGBA color."""
    return tuple([int(80 + random.random() * 175) for _ in range(3)] + [255])


class PiecesToGLB:
    """Convert extracted piece metadata into a standalone GLB mesh scene."""

    def __init__(self, pieces_metadata_path: str, extrusion_height: float = 3.0):
        self.pieces_metadata_path = pieces_metadata_path
        self.extrusion_height = extrusion_height

    def build_meshes(self) -> Dict[str, trimesh.Trimesh]:
        """Read metadata and create extruded meshes for each piece."""
        with open(self.pieces_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        meshes: Dict[str, trimesh.Trimesh] = {}
        z_cursor = 0.0

        for name, info in metadata.items():
            coords = _vertices_to_coords(info.get('vertices', []))
            if len(coords) < 3:
                print(f"Skipping piece '{name}': insufficient vertices for polygon.")
                continue

            # Create polygon and extrude
            poly = Polygon(coords)
            try:
                # Try to extrude the polygon
                mesh = trimesh.creation.extrude_polygon(poly, height=self.extrusion_height)
            except Exception as e:
                print(f"Error extruding '{name}', falling back to box: {e}")
                minx, miny, maxx, maxy = poly.bounds
                mesh = trimesh.creation.box(extents=(maxx - minx, maxy - miny, self.extrusion_height))
                mesh.apply_translation(((minx + maxx) / 2.0, (miny + maxy) / 2.0, self.extrusion_height / 2.0))

            # Add color
            color = _random_color()
            mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
            
            # Flip Y axis (SVG is Y-down, 3D is Y-up)
            rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            mesh.apply_transform(rotation)

            # Basic layout: stack them with some spacing along Z
            mesh.apply_translation((0.0, 0.0, z_cursor))
            mesh.metadata = {'name': name}
            meshes[name] = mesh
            
            z_cursor += self.extrusion_height * 2.0

        return meshes

    def run(self, out_glb: str) -> str:
        """Execute the conversion and save the combined GLB."""
        meshes_dict = self.build_meshes()
        if not meshes_dict:
            raise ValueError("No valid meshes were generated.")

        scene = trimesh.Scene()
        for name, mesh in meshes_dict.items():
            scene.add_geometry(mesh, node_name=name)

        out_dir = os.path.dirname(out_glb)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        glb_data = scene.export(file_type='glb')
        with open(out_glb, 'wb') as f:
            f.write(glb_data)

        return out_glb

    def run_individual(self, output_dir: str) -> List[str]:
        """Execute the conversion and save each piece as a separate GLB."""
        meshes_dict = self.build_meshes()
        if not meshes_dict:
            raise ValueError("No valid meshes were generated.")

        os.makedirs(output_dir, exist_ok=True)
        exported_files = []

        for name, mesh in meshes_dict.items():
            out_path = os.path.join(output_dir, f"{name}.glb")
            glb_data = mesh.export(file_type='glb')
            with open(out_path, 'wb') as f:
                f.write(glb_data)
            exported_files.append(out_path)

        return exported_files
