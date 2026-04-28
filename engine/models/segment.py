from dataclasses import dataclass, field
from typing import List, Tuple
from .point import Point

@dataclass
class Segment:
    """A geometric segment (line or cubic Bézier curve) between two Points."""
    type: str  # 'L' or 'C'
    start: Point
    end: Point
    control_points: List[Point] = field(default_factory=list)

    def reverse(self) -> 'Segment':
        """Reverse the segment traversal direction."""
        return Segment(
            type=self.type,
            start=self.end,
            end=self.start,
            control_points=list(reversed(self.control_points))
        )

    def mirror_x(self, anchor_x: float) -> 'Segment':
        """Reflect the segment across x = anchor_x."""
        return Segment(
            type=self.type,
            start=self.start.mirror_x(anchor_x),
            end=self.end.mirror_x(anchor_x),
            control_points=[cp.mirror_x(anchor_x) for cp in self.control_points]
        )

    def is_on_fold_line(self, anchor_x: float, tolerance: float = 0.01) -> bool:
        """Check if start, end, and all control points lie on x = anchor_x."""
        if abs(self.start.x - anchor_x) > tolerance or abs(self.end.x - anchor_x) > tolerance:
            return False
        for cp in self.control_points:
            if abs(cp.x - anchor_x) > tolerance:
                return False
        return True

    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) bounding box for this segment."""
        xs = [self.start.x, self.end.x] + [cp.x for cp in self.control_points]
        ys = [self.start.y, self.end.y] + [cp.y for cp in self.control_points]
        return min(xs), min(ys), max(xs), max(ys)
