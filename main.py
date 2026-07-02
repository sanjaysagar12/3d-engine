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
from engine.pipeline.style3d_draper import Style3DDraper

def main():
    """Main entry point: full 2D to 3D garment pipeline."""
    # 1. Config
    svg_file = "C:\\yoko\\3d-engine\\asserts\\freesewing-aaron.svg"
    clones_json = "C:\\yoko\\3d-engine\\asserts\\clones.json"
    stitching_json = "C:\\yoko\\3d-engine\\asserts\\stitching_instructions.json"
    measurement_json = "C:\\yoko\\3d-engine\\asserts\\measurement.json"
    smplx_model_path = "models"
    blender_exe = "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe"
    style3d_exe = "C:\\Program Files\\Style3D\\Style3D.exe"
    # Optional pre-sewn Style3D pattern (.sproj/.sgar/.pxf) authored once in
    # Style3D Studio. Without it, the Style3D layer falls back to the raw
    # flat panels and simulates an UNSTITCHED garment (see Style3DDraper).
    style3d_garment_source = None
    style3d_garment_source_type = None  # "sproj" | "sgar" | "pxf"
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

        # 8. Export flat panels — GarmentScaler uses a fixed 20 mm target cell
        #    size for uniform mesh density across all panels.
        print(f"\n--- Step 7: Preparing Panels for Blender Stitching ---")
        panels_glb = os.path.join(output_dir, "garment_panels_flat.glb")
        seam_points_json = os.path.join(output_dir, "garment_seam_points.json")
        scaler.export_panels_for_stitching(panels_glb, seam_points_json)

        if os.path.exists(blender_exe):
            # ── Simulation quality knobs (change these for quick iteration vs final) ──
            sim_frames       = 15    # frames to simulate; increase to 150 for final
            sewing_force_max = 15.0  # N; higher = seams close faster (good for low frame counts)
            quality          = 10    # solver substeps per frame
            mass             = 0.3   # kg per vertex

            # Step 8: Blender call 1 — subdivide seam boundaries + bridge seams
            #         + 2-cut wrinkle subdivision. Base mesh is already uniform
            #         20 mm cells, so 2 cuts → ~7 mm final cloth resolution.
            print(f"\n--- Step 8: Stitching Garment in Blender ---")
            stitched_out = os.path.join(output_dir, "garment_stitched.glb")
            blender_stitcher = BlenderStitcher(
                blender_exe=blender_exe,
                boundary_subdivide_cuts=8,
                wrinkle_subdivide_cuts=2,
            )
            blender_stitcher.run(
                panels_glb=panels_glb,
                seam_points_json=seam_points_json,
                stitching_json=stitching_json,
                out_glb=stitched_out,
            )

            # Step 9: Blender call 2 — cloth simulation (sewing springs +
            #         collision) to drape the stitched garment onto the avatar.
            print(f"\n--- Step 9: Draping Garment on Avatar (Cloth Sim) ---")
            draped_out = os.path.join(output_dir, "garment_draped.glb")
            draped_blend = os.path.join(output_dir, "garment_draped.blend")
            draper = ClothDraper(
                blender_exe=blender_exe,
                sim_frames=sim_frames,
                sewing_force_max=sewing_force_max,
                quality=quality,
                mass=mass,
            )
            draper.run(
                stitched_glb=stitched_out,
                avatar_obj=avatar_obj,
                out_glb=draped_out,
                blend_out=draped_blend,
            )
        else:
            print(f"\n! Skipping Steps 8-9: Blender not found at {blender_exe}")

        # Step 9b: second draping/stitching layer — same avatar + panels,
        # run through Style3D Studio's `spy` API instead of Blender. Unlike
        # the Blender steps this can't run headless (no confirmed CLI script
        # flag for Style3D.exe); it writes a config + prints instructions to
        # run engine/style3d_scripts/drape_garment_style3d.py from Style3D
        # Studio's own Script console. See Style3DDraper's docstring for the
        # sewing caveat (stitched only if style3d_garment_source is set).
        if os.path.exists(style3d_exe):
            print(f"\n--- Step 9b: Draping Garment on Avatar (Style3D) ---")
            draped_out_style3d = os.path.join(output_dir, "garment_draped_style3d.glb")
            style3d_draper = Style3DDraper(
                style3d_exe=style3d_exe,
                simulation_mode="Normal",
                velocity_converged_bound=0.5,
                gravity_factor=1.0,
                simulation_timeout=60.0,
            )
            style3d_draper.run(
                avatar_obj=avatar_obj,
                out_glb=draped_out_style3d,
                panels_glb=panels_glb,
                garment_source=style3d_garment_source,
                garment_source_type=style3d_garment_source_type,
            )
        else:
            print(f"\n! Skipping Step 9b: Style3D not found at {style3d_exe}")
    else:
        print(f"\n! Skipping Step 6: measurements or avatar not available")


if __name__ == "__main__":
    main()
