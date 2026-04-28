from ..models.piece import Piece

class MirrorOperation:
    """Mirrors a piece across a fold line."""
    
    def __init__(self, anchor_x: float = 0.0):
        self.anchor_x = anchor_x
        
    def execute(self, piece: Piece) -> Piece:
        # Keep non-fold segments
        contour_segments = [seg for seg in piece.segments if not seg.is_on_fold_line(self.anchor_x)]
        
        # Mirror and reverse contour segments
        mirrored_reversed = [seg.mirror_x(self.anchor_x).reverse() for seg in reversed(contour_segments)]
        
        # Combine
        combined_segments = contour_segments + mirrored_reversed
        
        return Piece(name=piece.name, full_name=piece.full_name, segments=combined_segments, source_id=piece.source_id)
