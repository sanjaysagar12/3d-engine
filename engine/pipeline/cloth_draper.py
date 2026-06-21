"""
Pipeline step: invoke Blender (background mode) to run a Cloth simulation
on an already-stitched garment, draping it onto the avatar.
"""
import json
import os
import subprocess
import tempfile

import trimesh


_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "blender_scripts", "drape_garment.py")


class ClothDraper:
    """Runs engine/blender_scripts/drape_garment.py inside Blender.

    Expects a pre-stitched garment GLB (produced by BlenderStitcher) and an
    avatar OBJ.  Runs Blender's cloth simulation (sewing springs + collision)
    and exports both a draped GLB and an optional .blend with baked cache.
    """

    def __init__(
        self,
        blender_exe: str,
        sim_frames: int = 150,
        sewing_force_max: float = 3.0,
        quality: int = 10,
        mass: float = 0.3,
    ):
        self.blender_exe = blender_exe
        self.sim_frames = sim_frames
        self.sewing_force_max = sewing_force_max
        self.quality = quality
        self.mass = mass

    def run(
        self,
        stitched_glb: str,
        avatar_obj: str,
        out_glb: str,
        blend_out: str = None,
    ) -> str:
        if not os.path.exists(self.blender_exe):
            raise FileNotFoundError(f"Blender executable not found: {self.blender_exe}")

        avatar_obj_abs = os.path.abspath(avatar_obj)
        avatar_glb = os.path.splitext(avatar_obj_abs)[0] + "_for_drape.glb"
        trimesh.load(avatar_obj_abs, force="mesh").export(avatar_glb)

        config = {
            "stitched_glb": os.path.abspath(stitched_glb),
            "avatar_glb": avatar_glb,
            "out_glb": os.path.abspath(out_glb),
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
        if result.stderr.strip():
            print("--- Blender stderr ---")
            print(result.stderr)
            print("----------------------")
        if result.returncode != 0:
            raise RuntimeError(f"Blender draping failed (exit {result.returncode})")

        os.unlink(cfg_path)

        if not os.path.exists(out_glb):
            raise RuntimeError(f"Blender did not produce expected output: {out_glb}")
        if blend_out and not os.path.exists(blend_out):
            raise RuntimeError(f"Blender did not produce expected blend file: {blend_out}")

        print(f"[OK] Draped garment exported to: {out_glb}")
        return out_glb
