import os
import json
from ..parser.freesewing_parser import FreeSewingParser
from ..svg.renderer import SVGRenderer

class PieceExtractor:
    """Step 1: Extract and write standalone pieces from an SVG."""
    
    def __init__(self, svg_path: str, output_dir: str):
        self.svg_path = svg_path
        self.output_dir = output_dir

    def run(self) -> dict:
        pieces = FreeSewingParser.parse(self.svg_path)
        os.makedirs(self.output_dir, exist_ok=True)
        
        metadata = {}
        for piece in pieces:
            # Render and save SVG (returns tuple: svg_content, points)
            svg_content, points = SVGRenderer.render_piece(piece)
            out_file = os.path.join(self.output_dir, f"{piece.name}.svg")
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(svg_content)
                
            # Get vertices and merge point IDs into them
            vertices = piece.vertices()
            for i, vertex in enumerate(vertices):
                if i < len(points):
                    vertex["id"] = points[i]["id"]
                
            # Build metadata
            min_x, min_y, max_x, max_y = piece.bounds()
            metadata[piece.name] = {
                "source_id": piece.source_id,
                "full_name": piece.full_name,
                "original_path_d": piece.to_path_d() if hasattr(piece, 'to_path_d') else __import__('engine.svg.path_parser').svg.path_parser.SVGPathParser.serialize_piece(piece),
                "bounds": {
                    "min_x": round(min_x, 4),
                    "min_y": round(min_y, 4),
                    "max_x": round(max_x, 4),
                    "max_y": round(max_y, 4),
                    "width": round(max_x - min_x, 4),
                    "height": round(max_y - min_y, 4),
                },
                "vertices": vertices,
                "output_file": f"{piece.name}.svg",
            }
            
        meta_path = os.path.join(self.output_dir, "pieces_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
