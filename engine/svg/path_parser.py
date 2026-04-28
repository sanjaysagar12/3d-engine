import re
from typing import List, Tuple
from ..models.point import Point
from ..models.segment import Segment
from ..models.piece import Piece

class SVGPathParser:
    """Static utilities for parsing and serializing SVG paths."""

    @staticmethod
    def parse_d(d_str: str) -> List[Tuple[str, List[float]]]:
        """Parse an SVG path 'd' attribute into structured command tuples."""
        tokens = re.findall(
            r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',
            d_str,
        )
        commands = []
        current_cmd = None
        current_args = []

        for token in tokens:
            if token.isalpha():
                if current_cmd is not None:
                    commands.append((current_cmd, current_args))
                current_cmd = token
                current_args = []
            else:
                current_args.append(float(token))

        if current_cmd is not None:
            commands.append((current_cmd, current_args))

        return commands

    @staticmethod
    def serialize_d(commands: List[Tuple[str, List[float]]]) -> str:
        """Serialize parsed path commands back into an SVG 'd' string."""
        parts = []
        for cmd, args in commands:
            if args:
                parts.append(f"{cmd} {' '.join(f'{a:g}' for a in args)}")
            else:
                parts.append(cmd)
        return ' '.join(parts)

    @staticmethod
    def commands_to_segments(commands: List[Tuple[str, List[float]]]) -> List[Segment]:
        """Convert path commands into geometric segments."""
        segments = []
        current_pos = None
        start_pos = None

        for cmd, args in commands:
            if cmd == 'M':
                start_pos = Point(args[0], args[1])
                current_pos = start_pos
            elif cmd == 'L':
                end = Point(args[0], args[1])
                segments.append(Segment(type='L', start=current_pos, end=end))
                current_pos = end
            elif cmd == 'C':
                for i in range(0, len(args), 6):
                    cp1 = Point(args[i], args[i + 1])
                    cp2 = Point(args[i + 2], args[i + 3])
                    end = Point(args[i + 4], args[i + 5])
                    segments.append(Segment(type='C', start=current_pos, end=end, control_points=[cp1, cp2]))
                    current_pos = end
            elif cmd in ('z', 'Z'):
                # Implicit close
                if current_pos and start_pos and current_pos != start_pos:
                    segments.append(Segment(type='L', start=current_pos, end=start_pos))
                current_pos = start_pos

        return segments

    @staticmethod
    def segments_to_commands(segments: List[Segment]) -> List[Tuple[str, List[float]]]:
        """Convert segments back to commands."""
        if not segments:
            return []
            
        cmds = []
        cmds.append(('M', [segments[0].start.x, segments[0].start.y]))
        
        for seg in segments:
            if seg.type == 'L':
                cmds.append(('L', [seg.end.x, seg.end.y]))
            elif seg.type == 'C':
                cp1, cp2 = seg.control_points
                cmds.append(('C', [cp1.x, cp1.y, cp2.x, cp2.y, seg.end.x, seg.end.y]))
                
        cmds.append(('z', []))
        return cmds

    @staticmethod
    def parse_piece(d_str: str, name: str = "", full_name: str = "", source_id: str = "") -> Piece:
        """Parse a full piece object from a path string."""
        cmds = SVGPathParser.parse_d(d_str)
        segments = SVGPathParser.commands_to_segments(cmds)
        return Piece(name=name, full_name=full_name, segments=segments, source_id=source_id)

    @staticmethod
    def serialize_piece(piece: Piece) -> str:
        """Serialize a Piece back to an SVG path 'd' string."""
        cmds = SVGPathParser.segments_to_commands(piece.segments)
        return SVGPathParser.serialize_d(cmds)
