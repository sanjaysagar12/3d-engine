"""
Main entry point for extracting prices from Freesewing SVG.
Uses existing PieceExtractor from pipeline.
"""
import os
from engine.pipeline.extractor import PieceExtractor


def main():
    """Main entry point: get input SVG and output name, then extract."""
    # Get input SVG file
    svg_file ="C:\\yoko\\3d-engine\\asserts\\freesewing-aaron.svg"
    
    if not os.path.exists(svg_file):
        print(f"Error: {svg_file} not found")
        return
    
    # Get project/output name
    project_name = "aaron"
    if not project_name:
        project_name = os.path.splitext(os.path.basename(svg_file))[0]
    
    # Create output directory: dist/{project_name}
    output_dir = os.path.join("dist", project_name)
    
    # Use PieceExtractor to extract pieces
    extractor = PieceExtractor(svg_file, output_dir)
    metadata = extractor.run()
    
    print(f"\n✓ Extracted {len(metadata)} pieces")
    print(f"✓ Saved to: {output_dir}/")
    print(f"✓ Metadata saved: {os.path.join(output_dir, 'pieces_metadata.json')}")


if __name__ == "__main__":
    main()
