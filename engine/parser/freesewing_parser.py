import re
import xml.etree.ElementTree as ET
from typing import List, Tuple
from ..models.piece import Piece
from ..svg.path_parser import SVGPathParser

NAMESPACES = {
    'svg': 'http://www.w3.org/2000/svg',
    'xlink': 'http://www.w3.org/1999/xlink',
    'freesewing': 'http://freesewing.org/namespaces/freesewing',
}

for _prefix, _uri in NAMESPACES.items():
    ET.register_namespace(_prefix, _uri)
ET.register_namespace('', NAMESPACES['svg'])

class FreeSewingParser:
    """Parser for FreeSewing SVG files."""

    @staticmethod
    def _parse_transform(transform_str: str) -> Tuple[float, float]:
        m = re.search(r'translate\(\s*([^,\s]+)\s*[,\s]\s*([^)]+)\)', transform_str)
        if m:
            return float(m.group(1)), float(m.group(2))
        return 0.0, 0.0

    @staticmethod
    def parse(svg_path: str) -> List[Piece]:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        ns = NAMESPACES['svg']
        
        container = root.find(f'.//{{{ns}}}g[@id="fs-container"]')
        if container is None:
            raise ValueError("Could not find fs-container group in SVG")
            
        pieces = []
        for stack_group in container:
            stack_id = stack_group.get('id', '')
            if not stack_id.startswith('fs-stack-'):
                continue
                
            full_piece_name = stack_id.replace('fs-stack-', '')
            piece_name = full_piece_name.split('.')[-1]
            part_group = None
            for child in stack_group:
                if 'part' in child.get('id', ''):
                    part_group = child
                    break
                    
            if part_group is None:
                continue
                
            inner_transform = part_group.get('transform', '')
            inner_tx, inner_ty = FreeSewingParser._parse_transform(inner_transform)
            
            fabric_path = None
            for elem in part_group:
                tag = elem.tag.replace(f'{{{ns}}}', '')
                if tag == 'path':
                    classes = elem.get('class', '')
                    if 'fabric' in classes and 'help' not in classes and 'hidden' not in classes:
                        fabric_path = elem
                        break
                        
            if fabric_path is None:
                continue
                
            d_str = fabric_path.get('d', '')
            path_id = fabric_path.get('id', '')
            
            piece = SVGPathParser.parse_piece(d_str, name=piece_name, full_name=full_piece_name, source_id=path_id)
            if inner_tx != 0 or inner_ty != 0:
                piece = piece.translate(inner_tx, inner_ty)
                
            pieces.append(piece)
        return pieces
