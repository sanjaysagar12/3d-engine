"""
Main entry point for extracting prices from Freesewing SVG.
Uses existing PieceExtractor from pipeline.
"""
import os
from engine.pipeline.extractor import PieceExtractor
from engine.pipeline.clone_processor import CloneProcessor
from engine.pipeline.pieces_to_glb import PiecesToGLB
from engine.pipeline.garment_stitcher import GarmentStitcher
from engine.pipeline.garment_scaler import GarmentScaler

def main():
    """Main entry point: full 2D to 3D garment pipeline."""
    # 1. Config
    svg_file = "C:\\yoko\\3d-engine\\asserts\\freesewing-aaron.svg"
    clones_json = "C:\\yoko\\3d-engine\\asserts\\clones.json"
    stitching_json = "C:\\yoko\\3d-engine\\asserts\\stitching_instructions.json"
    scaling_json = "C:\\yoko\\3d-engine\\asserts\\scaling_instructions.json"
    measurements_yaml = "C:\\yoko\\3d-engine\\asserts\\measurements.yaml"
    avatar_obj = "C:\\yoko\\3d-engine\\asserts\\avatar.obj"
    project_name = "aaron"
    output_dir = os.path.join("dist", project_name)
    
    if not os.path.exists(svg_file):
        print(f"Error: {svg_file} not found")
        return

    # 2. Extract 2D pieces and metadata
    print(f"\n--- Step 1: Extracting Pieces ---")
    extractor = PieceExtractor(svg_file, output_dir)
    metadata = extractor.run()
    meta_path = os.path.join(output_dir, 'pieces_metadata.json')
    print(f"[OK] Extracted {len(metadata)} pieces to {output_dir}")

    # 3. Process clones/mirrors
    print(f"\n--- Step 2: Processing Clones/Mirrors ---")
    cloner = CloneProcessor(svg_file, clones_json, output_dir)
    metadata = cloner.run(metadata)
    print(f"[OK] Metadata updated with clones")

    # 4. Generate individual 3D GLBs
    print(f"\n--- Step 3: Generating Individual Meshes ---")
    glb_dir = os.path.join(output_dir, "glb")
    converter = PiecesToGLB(meta_path, extrusion_height=5.0)
    out_files = converter.run_individual(glb_dir)
    print(f"[OK] Generated {len(out_files)} individual GLBs in {glb_dir}")

    # 5. Assemble and stitch garment (unscaled)
    if os.path.exists(stitching_json):
        print(f"\n--- Step 4: Assembling Garment ---")
        garment_out = os.path.join(output_dir, "garment.glb")
        stitcher = GarmentStitcher(meta_path, stitching_json, extrusion_height=2.0, gap=300.0)
        stitcher.run(garment_out)
    else:
        print(f"\n! Skipping Step 4: {stitching_json} not found")

    # 6. Scale garment to avatar measurements and export with avatar
    if os.path.exists(measurements_yaml):
        print(f"\n--- Step 5: Scaling Garment to Avatar ---")
        scaled_out = os.path.join(output_dir, "garment_scaled.glb")
        scaler = GarmentScaler(
            metadata_path=meta_path,
            stitching_path=stitching_json,
            measurements_path=measurements_yaml,
            avatar_obj_path=avatar_obj,
            extrusion_height=2.0,
            gap=300.0,
        )
        scaler.run(scaled_out)
    else:
        print(f"\n! Skipping Step 5: measurements file not found")


if __name__ == "__main__":
    main()
