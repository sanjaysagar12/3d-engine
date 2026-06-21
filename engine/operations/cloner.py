"""Piece cloning module for creating mirrors and duplicates."""
import json
import os
from typing import Dict, List
from ..parser.freesewing_parser import FreeSewingParser
from ..models.piece import Piece


class PieceCloner:
    """Clone and mirror pieces based on configuration."""
    
    @staticmethod
    def load_clone_config(config_path: str) -> Dict:
        """Load clone configuration from JSON file.
        
        Args:
            config_path: Path to clones.json
            
        Returns:
            Dictionary with clone configurations
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading clone config: {e}")
            return {"clones": []}
    
    @staticmethod
    def mirror_piece(piece: Piece, axis: str, anchor: float) -> Piece:
        """Mirror a piece across specified axis.
        
        Args:
            piece: Piece to mirror
            axis: 'x' or 'y'
            anchor: Anchor point for mirroring
            
        Returns:
            New mirrored Piece
        """
        if axis.lower() == 'x':
            # Mirror across vertical line (x = anchor)
            mirrored_segments = [seg.mirror_x(anchor) for seg in piece.segments]
            # Reverse segment order for proper path direction
            mirrored_segments = list(reversed([seg.reverse() for seg in mirrored_segments]))
        else:
            raise ValueError(f"Mirror axis '{axis}' not yet supported")
        
        return Piece(
            name=piece.name,
            full_name=piece.full_name,
            segments=mirrored_segments,
            source_id=piece.source_id
        )
    
    @staticmethod
    def create_clones(pieces: List[Piece], config: Dict) -> List[Piece]:
        """Create cloned and mirrored pieces from configuration.
        
        Args:
            pieces: List of source pieces
            config: Clone configuration
            
        Returns:
            List of newly created cloned pieces
        """
        clones = []
        pieces_dict = {piece.name: piece for piece in pieces}
        
        for clone_def in config.get("clones", []):
            source_name = clone_def.get("source")
            clone_name = clone_def.get("name")
            
            if source_name not in pieces_dict:
                print(f"Warning: Source piece '{source_name}' not found")
                continue
            
            source_piece = pieces_dict[source_name]
            cloned_piece = Piece(
                name=clone_name,
                full_name=f"{source_piece.full_name}_clone",
                segments=list(source_piece.segments),  # Copy segments
                source_id=source_piece.source_id
            )
            
            # Apply mirror if specified
            if clone_def.get("mirror", {}).get("enabled"):
                mirror_axis = clone_def["mirror"].get("axis", "x")
                anchor_point = clone_def["mirror"].get("anchor_x" if mirror_axis == "x" else "anchor_y", 0)
                cloned_piece = PieceCloner.mirror_piece(cloned_piece, mirror_axis, anchor_point)
                cloned_piece.name = clone_name
            
            clones.append(cloned_piece)
        
        return clones
    
    @staticmethod
    def save_clones_metadata(clones: List[Piece], output_path: str):
        """Save cloned pieces metadata to JSON.
        
        Args:
            clones: List of cloned pieces
            output_path: Path to save metadata
        """
        metadata = {}
        for piece in clones:
            min_x, min_y, max_x, max_y = piece.bounds()
            metadata[piece.name] = {
                "full_name": piece.full_name,
                "source_id": piece.source_id,
                "bounds": {
                    "min_x": round(min_x, 4),
                    "min_y": round(min_y, 4),
                    "max_x": round(max_x, 4),
                    "max_y": round(max_y, 4),
                    "width": round(max_x - min_x, 4),
                    "height": round(max_y - min_y, 4),
                }
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
