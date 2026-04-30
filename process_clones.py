"""Script to process piece clones and create mirrored versions."""
import os
import json
from engine.parser.freesewing_parser import FreeSewingParser
from engine.operations.cloner import PieceCloner
from engine.svg.renderer import SVGRenderer
from engine.svg.converter import SVGConverter


def process_clones():
    """Process clones configuration and generate cloned/mirrored pieces."""
    # Load clone configuration
    config_path = "asserts/clones.json"
    if not os.path.exists(config_path):
        print(f"Clone config not found: {config_path}")
        return
    
    config = PieceCloner.load_clone_config(config_path)
    
    # Parse original SVG
    svg_path = "C:\\yoko\\3d-engine\\asserts\\freesewing-aaron.svg"
    print(f"Parsing SVG: {svg_path}")
    pieces = FreeSewingParser.parse(svg_path)
    
    # Create cloned pieces
    clones = PieceCloner.create_clones(pieces, config)
    print(f"✓ Created {len(clones)} cloned pieces")
    
    # Setup output directory (same as original pieces)
    output_dir = os.path.join("dist", "aaron")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    # Load existing metadata
    meta_path = os.path.join(output_dir, "pieces_metadata.json")
    with open(meta_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    
    # Process each clone
    for clone_piece in clones:
        # Render SVG
        svg_content, points = SVGRenderer.render_piece(clone_piece)
        svg_file = os.path.join(output_dir, f"{clone_piece.name}.svg")
        with open(svg_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        # Convert to PNG
        png_file = os.path.join(png_dir, f"{clone_piece.name}.png")
        success = SVGConverter.svg_to_png(svg_content, png_file, dpi=96)
        
        # Get bounds
        min_x, min_y, max_x, max_y = clone_piece.bounds()
        
        # Get vertices with point IDs
        vertices = clone_piece.vertices()
        for i, vertex in enumerate(vertices):
            if i < len(points):
                vertex["id"] = points[i]["id"]
        
        all_metadata[clone_piece.name] = {
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
            "output_file": f"{clone_piece.name}.svg",
            "png_file": f"png/{clone_piece.name}.png",
        }
        
        status = "✓" if success else "✗"
        print(f"  {status} {clone_piece.name}: {clone_piece.name}.svg → png/{clone_piece.name}.png")
    
    # Save updated metadata
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n✓ Clones added to: {output_dir}/")
    print(f"✓ Metadata updated: {meta_path}")


if __name__ == "__main__":
    process_clones()

