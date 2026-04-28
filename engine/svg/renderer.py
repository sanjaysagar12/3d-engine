from ..models.piece import Piece
from .path_parser import SVGPathParser

class SVGRenderer:
    """Renderer for outputting SVG documents."""

    @staticmethod
    def extract_points_with_ids(piece: Piece, offset_x: float = 0, offset_y: float = 0) -> list:
        """Extract all points from a piece with unique IDs."""
        points = []
        point_id = 0
        
        for segment in piece.segments:
            # Add start point
            points.append({
                "id": f"pt-{point_id}",
                "x": round(segment.start.x + offset_x, 4),
                "y": round(segment.start.y + offset_y, 4),
                "type": "start"
            })
            point_id += 1
            
            # Add control points if curve
            if segment.type == 'C' and segment.control_points:
                for i, cp in enumerate(segment.control_points):
                    points.append({
                        "id": f"pt-{point_id}",
                        "x": round(cp.x + offset_x, 4),
                        "y": round(cp.y + offset_y, 4),
                        "type": f"control{i+1}"
                    })
                    point_id += 1
        
        return points

    @staticmethod
    def render_piece(piece: Piece, padding: float = 5.0) -> tuple:
        """Generate SVG string and points info for a piece (shifted to fit).
        
        Returns:
            tuple: (svg_content: str, points: list)
        """
        min_x, min_y, max_x, max_y = piece.bounds()
        
        # Move piece to origin with padding
        offset_x = -min_x + padding
        offset_y = -min_y + padding
        shifted_piece = piece.translate(offset_x, offset_y)
        d_str = SVGPathParser.serialize_piece(shifted_piece)

        width = (max_x - min_x) + 2 * padding
        height = (max_y - min_y) + 2 * padding
        
        # Extract points with IDs
        points = SVGRenderer.extract_points_with_ids(shifted_piece, 0, 0)
        
        # Generate points circles SVG
        points_svg = ""
        for point in points:
            points_svg += f'    <circle id="{point["id"]}" cx="{point["x"]:.2f}" cy="{point["y"]:.2f}" r="1" class="point" />\n'
            points_svg += f'    <text x="{point["x"]:.2f}" y="{point["y"]-1.5:.2f}" class="point-label">{point["id"]}</text>\n'

        svg_content = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.2f}mm" height="{height:.2f}mm"
     viewBox="0 0 {width:.2f} {height:.2f}">
  <defs>
    <style type="text/css">
      .fabric {{
        fill: none;
        stroke: #212121;
        stroke-width: 0.6;
        stroke-linecap: round;
        stroke-linejoin: round;
      }}
      .point {{
        fill: #ff6b6b;
        stroke: #c92a2a;
        stroke-width: 0.3;
      }}
      .point-label {{
        font-family: Arial, sans-serif;
        font-size: 1.5px;
        fill: #c92a2a;
        font-weight: bold;
        pointer-events: none;
      }}
    </style>
  </defs>
  <g id="{shifted_piece.name}">
    <path class="fabric" id="{shifted_piece.name}-outline" d="{d_str}" />
    <g id="{shifted_piece.name}-points">
{points_svg}    </g>
  </g>
</svg>'''
        
        return svg_content, points

    @staticmethod
    def render_stitched(piece: Piece, padding: float = 5.0) -> str:
        """Generate a standalone SVG string for a stitched piece."""
        min_x, min_y, max_x, max_y = piece.bounds()
        
        shifted_piece = piece.translate(-min_x + padding, -min_y + padding)
        d_str = SVGPathParser.serialize_piece(shifted_piece)

        width = (max_x - min_x) + 2 * padding
        height = (max_y - min_y) + 2 * padding

        return f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.2f}mm" height="{height:.2f}mm"
     viewBox="0 0 {width:.2f} {height:.2f}">
  <style type="text/css">
    .fabric {{
      fill: none;
      stroke: #212121;
      stroke-width: 0.6;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
  </style>
  <g id="{shifted_piece.name}">
    <path class="fabric" id="{shifted_piece.name}-outline" d="{d_str}" />
  </g>
</svg>'''
