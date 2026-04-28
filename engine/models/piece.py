from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from .segment import Segment

@dataclass
class Piece:
    """A collection of Segments forming a closed outline."""
    name: str
    full_name: str
    segments: List[Segment]
    source_id: str = ""

    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) bounding box for the entire piece."""
        if not self.segments:
            return 0.0, 0.0, 0.0, 0.0
            
        min_x, min_y, max_x, max_y = self.segments[0].bounds()
        for seg in self.segments[1:]:
            sx, sy, ex, ey = seg.bounds()
            min_x = min(min_x, sx)
            min_y = min(min_y, sy)
            max_x = max(max_x, ex)
            max_y = max(max_y, ey)
        return min_x, min_y, max_x, max_y

    def vertices(self) -> List[Dict[str, Any]]:
        """Extract vertex info for metadata JSON."""
        verts = []
        if not self.segments:
            return verts
        
        # Add initial move
        first = self.segments[0].start
        verts.append({"x": first.x, "y": first.y, "type": "move"})
        
        for seg in self.segments:
            if seg.type == 'L':
                verts.append({"x": seg.end.x, "y": seg.end.y, "type": "line"})
            elif seg.type == 'C':
                verts.append({"x": seg.control_points[0].x, "y": seg.control_points[0].y, "type": "control1"})
                verts.append({"x": seg.control_points[1].x, "y": seg.control_points[1].y, "type": "control2"})
                verts.append({"x": seg.end.x, "y": seg.end.y, "type": "curve_end"})
        
        verts.append({"type": "close"})
        return verts

    def translate(self, tx: float, ty: float) -> 'Piece':
        """Translate the piece coordinates."""
        new_segs = []
        for seg in self.segments:
            new_seg = Segment(
                type=seg.type,
                start=seg.start.translate(tx, ty),
                end=seg.end.translate(tx, ty),
                control_points=[cp.translate(tx, ty) for cp in seg.control_points]
            )
            new_segs.append(new_seg)

        return Piece(
            name=self.name,
            full_name=self.full_name,
            segments=new_segs,
            source_id=self.source_id
        )
