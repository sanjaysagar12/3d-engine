import os
import json
from ..parser.freesewing_parser import FreeSewingParser
from ..operations.cloner import PieceCloner
from ..svg.renderer import SVGRenderer
from ..svg.converter import SVGConverter
from ..svg.dxf_exporter import export_piece_dxf

class CloneProcessor:
    """Pipeline step to process piece clones and mirrored versions."""
    
    def __init__(self, svg_path: str, clones_config_path: str, output_dir: str):
        self.svg_path = svg_path
        self.clones_config_path = clones_config_path
        self.output_dir = output_dir

    def run(self, existing_metadata: dict) -> dict:
        if not os.path.exists(self.clones_config_path):
            print(f"! Clone config not found: {self.clones_config_path}")
            return existing_metadata
        
        config = PieceCloner.load_clone_config(self.clones_config_path)
        pieces = FreeSewingParser.parse(self.svg_path)
        
        # Create cloned pieces
        clones = PieceCloner.create_clones(pieces, config)
        print(f"  [OK] Created {len(clones)} cloned pieces")
        
        svg_dir = os.path.join(self.output_dir, "svg")
        png_dir = os.path.join(self.output_dir, "png")
        dxf_dir = os.path.join(self.output_dir, "dxf")
        os.makedirs(svg_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(dxf_dir, exist_ok=True)
        
        # Process each clone
        for clone_piece in clones:
            # Render SVG
            svg_content, points = SVGRenderer.render_piece(clone_piece)
            svg_file = os.path.join(svg_dir, f"{clone_piece.name}.svg")
            with open(svg_file, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            # Convert to PNG
            png_file = os.path.join(png_dir, f"{clone_piece.name}.png")
            success = SVGConverter.svg_to_png(svg_content, png_file, dpi=96)

            # Export DXF
            dxf_file = os.path.join(dxf_dir, f"{clone_piece.name}.dxf")
            export_piece_dxf(clone_piece, dxf_file)
            
            # Get bounds and vertices
            min_x, min_y, max_x, max_y = clone_piece.bounds()
            vertices = clone_piece.vertices()
            for i, vertex in enumerate(vertices):
                if i < len(points):
                    vertex["id"] = points[i]["id"]
            
            existing_metadata[clone_piece.name] = {
                "source_id": clone_piece.source_id,
                "full_name": clone_piece.full_name,
                "bounds": {
                    "min_x": round(min_x, 4),
                    "min_y": round(min_y, 4),
                    "max_x": round(max_x, 4),
                    "max_y": round(max_y, 4),
                    "width": round(max_x - min_x, 4),
                    "height": round(max_y - min_y, 4),
                },
                "vertices": vertices,
                "output_file": f"svg/{clone_piece.name}.svg",
                "png_file": f"png/{clone_piece.name}.png",
                "dxf_file": f"dxf/{clone_piece.name}.dxf",
            }
            
            status = "[OK]" if success else "[ERROR]"
            print(f"    {status} {clone_piece.name}: {clone_piece.name}.svg -> png/{clone_piece.name}.png")
        
        # Save updated metadata
        meta_path = os.path.join(self.output_dir, "pieces_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, indent=2)
            
        return existing_metadata
