from typing import NamedTuple

class Point(NamedTuple):
    """An immutable 2D coordinate."""
    x: float
    y: float

    def translate(self, tx: float, ty: float) -> 'Point':
        return Point(self.x + tx, self.y + ty)

    def mirror_x(self, anchor_x: float) -> 'Point':
        """Reflect across a vertical line at x = anchor_x."""
        return Point(2 * anchor_x - self.x, self.y)
