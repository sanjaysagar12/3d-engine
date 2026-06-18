"""
Pipeline step: invoke Blender (background mode) to stitch the garment panels
and run a Cloth simulation -- with sewing springs enabled -- to drape the
garment onto the avatar.
"""
import json
import os
import subprocess
import tempfile

import trimesh


_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "blender_scripts", "drape_garment.py")


class ClothDraper:
    """Runs engine/blender_scripts/drape_garment.py inside Blender."""

    def __init__(
        self,
        blender_exe: str,
        boundary_subdivide_cuts: int = 8,
        wrinkle_subdivide_cuts: int = 2,
        sim_frames: int = 180,
        sewing_force_max: float = 3.0,
        quality: int = 20,
        mass: float = 0.3,
    ):
        self.blender_exe = blender_exe
        self.boundary_subdivide_cuts = boundary_subdivide_cuts
        self.wrinkle_subdivide_cuts = wrinkle_subdivide_cuts
        self.sim_frames = sim_frames
        self.sewing_force_max = sewing_force_max
        self.quality = quality
        self.mass = mass

    def run(
        self,
        panels_glb: str,
        seam_points_json: str,
        stitching_json: str,
        avatar_obj: str,
        out_glb: str,
        blend_out: str = None,
    ) -> str:
        if not os.path.exists(self.blender_exe):
            raise FileNotFoundError(f"Blender executable not found: {self.blender_exe}")

        # Convert avatar OBJ -> GLB so Blender imports it via the glTF
        # path (reliable Y-up -> Z-up conversion) instead of the OBJ
        # importer, whose up_axis/forward_axis args don't actually rotate
        # the data the same way -- verified empirically.
        avatar_glb = os.path.splitext(os.path.abspath(avatar_obj))[0] + "_for_drape.glb"
        trimesh.load(avatar_obj, force="mesh").export(avatar_glb)

        config = {
            "panels_glb": os.path.abspath(panels_glb),
            "seam_points_json": os.path.abspath(seam_points_json),
            "stitching_json": os.path.abspath(stitching_json),
            "avatar_glb": avatar_glb,
            "out_glb": os.path.abspath(out_glb),
            "boundary_subdivide_cuts": self.boundary_subdivide_cuts,
            "wrinkle_subdivide_cuts": self.wrinkle_subdivide_cuts,
            "sim_frames": self.sim_frames,
            "sewing_force_max": self.sewing_force_max,
            "quality": self.quality,
            "mass": self.mass,
        }
        if blend_out:
            config["blend_out"] = os.path.abspath(blend_out)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as cfg_file:
            json.dump(config, cfg_file, indent=2)
            cfg_path = cfg_file.name

        script_path = os.path.abspath(_SCRIPT_PATH)
        cmd = [self.blender_exe, "--background", "--python", script_path, "--", cfg_path]

        print(f"  Running Blender (drape): {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"Blender draping failed (exit {result.returncode})")

        os.unlink(cfg_path)

        if not os.path.exists(out_glb):
            raise RuntimeError(f"Blender did not produce expected output: {out_glb}")
        if blend_out and not os.path.exists(blend_out):
            raise RuntimeError(f"Blender did not produce expected blend file: {blend_out}")

        print(f"[OK] Draped garment exported to: {out_glb}")
        return out_glb
