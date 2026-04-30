import argparse
import os
import sys

# Ensure engine is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.pipeline.pieces_to_glb import PiecesToGLB

def main():
    parser = argparse.ArgumentParser(description='Convert pattern piece metadata to 3D GLB meshes.')
    parser.add_argument('--metadata', default='dist/aaron/pieces_metadata.json', 
                        help='Path to the pieces_metadata.json file (default: dist/aaron/pieces_metadata.json)')
    parser.add_argument('--out_dir', default='dist/aaron/glb', 
                        help='Directory for the output GLB files (default: dist/aaron/glb)')
    parser.add_argument('--height', type=float, default=5.0, 
                        help='Extrusion height for the pieces (default: 5.0)')

    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found at {args.metadata}")
        sys.exit(1)

    print(f"--- Converting Pieces to Individual GLBs ---")
    print(f"Input: {args.metadata}")
    print(f"Output Directory: {args.out_dir}")
    print(f"Extrusion Height: {args.height}")

    try:
        converter = PiecesToGLB(args.metadata, extrusion_height=args.height)
        out_files = converter.run_individual(args.out_dir)
        print(f"\n[OK] Successfully exported {len(out_files)} GLB files to: {args.out_dir}")
        for f in out_files:
            print(f"  - {os.path.basename(f)}")
    except Exception as e:
        print(f"\n[ERROR] Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
