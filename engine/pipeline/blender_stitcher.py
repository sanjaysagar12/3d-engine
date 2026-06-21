"""
Pipeline step: invoke Blender (background mode) to subdivide garment panel
boundaries and stitch panels together with real sewing edges.
"""
import json
import os
import subprocess
import tempfile


_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "blender_scripts", "stitch_garment.py")


class BlenderStitcher:
    """Runs engine/blender_scripts/stitch_garment.py inside Blender."""

    def __init__(self, blender_exe: str, boundary_subdivide_cuts: int = 8,
                 wrinkle_subdivide_cuts: int = 2):
        self.blender_exe = blender_exe
        self.boundary_subdivide_cuts = boundary_subdivide_cuts
        self.wrinkle_subdivide_cuts = wrinkle_subdivide_cuts

    def run(self, panels_glb: str, seam_points_json: str,
            stitching_json: str, out_glb: str) -> str:
        if not os.path.exists(self.blender_exe):
            raise FileNotFoundError(f"Blender executable not found: {self.blender_exe}")

        config = {
            "panels_glb": os.path.abspath(panels_glb),
            "seam_points_json": os.path.abspath(seam_points_json),
            "stitching_json": os.path.abspath(stitching_json),
            "out_glb": os.path.abspath(out_glb),
            "boundary_subdivide_cuts": self.boundary_subdivide_cuts,
            "wrinkle_subdivide_cuts": self.wrinkle_subdivide_cuts,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as cfg_file:
            json.dump(config, cfg_file, indent=2)
            cfg_path = cfg_file.name

        script_path = os.path.abspath(_SCRIPT_PATH)
        cmd = [self.blender_exe, "--background", "--python", script_path, "--", cfg_path]

        print(f"  Running Blender (stitch): {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"Blender stitching failed (exit {result.returncode})")

        os.unlink(cfg_path)

        if not os.path.exists(out_glb):
            raise RuntimeError(f"Blender did not produce expected output: {out_glb}")

        print(f"[OK] Blender-stitched garment exported to: {out_glb}")
        return out_glb
