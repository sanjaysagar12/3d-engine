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
from engine.pipeline.avatar_creator import AvatarCreator
from engine.pipeline.blender_stitcher import BlenderStitcher
from engine.pipeline.cloth_draper import ClothDraper

def main():
    """Main entry point: full 2D to 3D garment pipeline."""
    # 1. Config
    svg_file = "C:\\yoko\\3d-engine\\asserts\\freesewing-aaron.svg"
    clones_json = "C:\\yoko\\3d-engine\\asserts\\clones.json"
    stitching_json = "C:\\yoko\\3d-engine\\asserts\\stitching_instructions.json"
    measurement_json = "C:\\yoko\\3d-engine\\asserts\\measurement.json"
    smplx_model_path = "models"
    blender_exe = "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe"
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

    # 6. Build avatar from measurements and compute arrangement points
    avatar_obj = os.path.join(output_dir, "avatar.obj")
    arrangement_points_json = os.path.splitext(avatar_obj)[0] + "_arrangement_points.json"
    if os.path.exists(measurement_json):
        print(f"\n--- Step 5: Creating Avatar from Measurements ---")
        creator = AvatarCreator(
            measurements_json=measurement_json,
            model_path=smplx_model_path,
            model_type="smplx",
        )
        creator.run(avatar_obj)
        print(f"[OK] Avatar + arrangement points created from {measurement_json}")
    else:
        print(f"\n! Skipping Step 5: {measurement_json} not found")

    # 7. Scale garment and place pieces at SMPL-X arrangement points
    if os.path.exists(measurement_json) and os.path.exists(avatar_obj):
        print(f"\n--- Step 6: Scaling Garment to Avatar ---")
        scaled_out = os.path.join(output_dir, "garment_scaled.glb")
        scaler = GarmentScaler(
            metadata_path=meta_path,
            stitching_path=stitching_json,
            measurements_path=measurement_json,
            avatar_obj_path=avatar_obj,
            arrangement_points_json=arrangement_points_json,
            extrusion_height=2.0,
            gap=300.0,
        )
        scaler.run(scaled_out)

        # 8. Export flat, open-boundary panels + seam point coordinates
        print(f"\n--- Step 7: Preparing Panels for Blender Stitching ---")
        panels_glb = os.path.join(output_dir, "garment_panels_flat.glb")
        seam_points_json = os.path.join(output_dir, "garment_seam_points.json")
        scaler.export_panels_for_stitching(panels_glb, seam_points_json)

        # 9. Subdivide boundaries + sew panels together in Blender
        if os.path.exists(blender_exe):
            print(f"\n--- Step 8: Stitching Garment in Blender ---")
            stitched_out = os.path.join(output_dir, "garment_stitched.glb")
            blender_stitcher = BlenderStitcher(blender_exe=blender_exe, boundary_subdivide_cuts=8, wrinkle_subdivide_cuts=2)
            blender_stitcher.run(
                panels_glb=panels_glb,
                seam_points_json=seam_points_json,
                stitching_json=stitching_json,
                out_glb=stitched_out,
            )
            # 10. Run cloth simulation (sewing springs + collision) to drape
            #     the garment onto the avatar
            print(f"\n--- Step 9: Draping Garment on Avatar (Cloth Sim) ---")
            draped_out = os.path.join(output_dir, "garment_draped.glb")
            draped_blend = os.path.join(output_dir, "garment_draped.blend")
            draper = ClothDraper(
                blender_exe=blender_exe,
                boundary_subdivide_cuts=8,
                wrinkle_subdivide_cuts=2,
                sim_frames=180,
                sewing_force_max=3.0,
                quality=20,
                mass=0.3,
            )
            draper.run(
                panels_glb=panels_glb,
                seam_points_json=seam_points_json,
                stitching_json=stitching_json,
                avatar_obj=avatar_obj,
                out_glb=draped_out,
                blend_out=draped_blend,
            )
        else:
            print(f"\n! Skipping Steps 8-9: Blender not found at {blender_exe}")
    else:
        print(f"\n! Skipping Step 6: measurements or avatar not available")


if __name__ == "__main__":
    main()
