"""
Pipeline step: prepare a config for, and optionally wait on, Style3D
Studio's `spy`-scripted drape (engine/style3d_scripts/drape_garment_style3d.py).

This is the Style3D counterpart to ClothDraper (Blender). It intentionally
does NOT subprocess-launch Style3D the way BlenderStitcher/ClothDraper shell
out to `blender --background --python ...`: Style3D Studio has no confirmed
CLI flag for headless script execution, and the local install here is driven
via its in-app Python/Script console. So this class:

  1. Converts inputs (GLB panels, OBJ avatar) into formats spy can import
     (fbx/obj -- Style3D has no GLB importer).
  2. Writes a config JSON to a fixed sidecar path next to the in-app script.
  3. Prints the manual steps to run it (open Style3D Studio, run the script
     from its Script console).
  4. Optionally blocks, polling a status JSON the script writes on
     completion, so `main.py` can treat this like a synchronous pipeline
     step once you've triggered the run in the app.

See engine/style3d_scripts/drape_garment_style3d.py's module docstring for
the sewing caveat: Style3D's `spy` API cannot create sew connections at
script time, so a real stitched drape requires a pre-sewn garment_source
(.sproj/.sgar/.pxf) authored once in Style3D Studio. Passing only raw flat
panels (via `panels_glb`) still works but simulates as unstitched pieces --
useful for a first look, not a final drape.
"""
import json
import os
import time

import trimesh

_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "style3d_scripts")
_SCRIPT_PATH = os.path.join(_SCRIPT_DIR, "drape_garment_style3d.py")
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "_drape_config.json")
_STATUS_PATH = os.path.join(_SCRIPT_DIR, "_drape_status.json")


class Style3DDraper:
    """Prepares and (optionally) waits on a Style3D `spy` drape run.

    Unlike ClothDraper, `run()` does not itself perform the simulation --
    Style3D Studio must be open and the generated script run from its
    in-app Script console. Set `wait_timeout` to block and poll for the
    status file the script writes on completion.
    """

    def __init__(
        self,
        style3d_exe: str = None,
        fabric: dict = None,
        simulation_mode: str = "Normal",
        velocity_converged_bound: float = 0.5,
        gravity_factor: float = 1.0,
        simulation_timeout: float = 60.0,
    ):
        self.style3d_exe = style3d_exe
        self.fabric = fabric or {}
        self.simulation_mode = simulation_mode
        self.velocity_converged_bound = velocity_converged_bound
        self.gravity_factor = gravity_factor
        self.simulation_timeout = simulation_timeout

    def run(
        self,
        avatar_obj: str,
        out_glb: str,
        panels_glb: str = None,
        garment_source: str = None,
        garment_source_type: str = None,
        out_dxf: str = None,
        wait_timeout: float = None,
        poll_interval: float = 5.0,
    ) -> str:
        """
        Either `garment_source` (+ `garment_source_type` of "sproj"/"sgar"/
        "pxf") or `panels_glb` must be given. `garment_source` is the
        preferred, stitched path; `panels_glb` is the unstitched fallback
        (converted from GLB to OBJ here, since Style3D has no GLB importer).
        """
        if not garment_source and not panels_glb:
            raise ValueError("Provide either garment_source or panels_glb")
        if garment_source and garment_source_type not in ("sproj", "sgar", "pxf"):
            raise ValueError("garment_source_type must be 'sproj', 'sgar', or 'pxf'")

        avatar_obj_abs = os.path.abspath(avatar_obj)

        config = {
            "avatar_obj": avatar_obj_abs,
            "avatar_move_to_origin": True,
            "avatar_skin_offset_mm": 2.0,
            "avatar_static_friction": 0.3,
            "avatar_dynamic_friction": 0.3,
            "fabric": self.fabric,
            "simulation_mode": self.simulation_mode,
            "velocity_converged_bound": self.velocity_converged_bound,
            "gravity_factor": self.gravity_factor,
            "simulation_timeout": self.simulation_timeout,
            "out_glb": os.path.abspath(out_glb),
        }
        if out_dxf:
            config["out_dxf"] = os.path.abspath(out_dxf)

        if garment_source:
            config["garment_source"] = os.path.abspath(garment_source)
            config["garment_source_type"] = garment_source_type
        else:
            panels_glb_abs = os.path.abspath(panels_glb)
            panels_obj = os.path.splitext(panels_glb_abs)[0] + "_for_style3d.obj"
            trimesh.load(panels_glb_abs, force="mesh").export(panels_obj)
            config["panels_obj"] = panels_obj
            print("  [WARN] No garment_source given -- falling back to raw flat "
                  "panels. Style3D's spy API can't create sew connections at "
                  "script time, so this will simulate as UNSTITCHED cloth "
                  "pieces. Author sewlines once in Style3D Studio and export "
                  "a .sproj/.sgar/.pxf to get a real stitched drape.")

        os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        if os.path.exists(_STATUS_PATH):
            os.remove(_STATUS_PATH)

        print(f"[Style3D] Config written to: {_CONFIG_PATH}")
        print(f"[Style3D] To run the drape:")
        print(f"    1. Open Style3D Studio"
              + (f" ({self.style3d_exe})" if self.style3d_exe else ""))
        print(f"    2. Open its Python/Script console")
        print(f"    3. Run: {os.path.abspath(_SCRIPT_PATH)}")
        print(f"[Style3D] It will write status to: {_STATUS_PATH}")

        if wait_timeout is None:
            return None

        return self._wait_for_status(out_glb, wait_timeout, poll_interval)

    def _wait_for_status(self, out_glb: str, timeout: float, poll_interval: float) -> str:
        print(f"[Style3D] Waiting up to {timeout:.0f}s for status file...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(_STATUS_PATH):
                with open(_STATUS_PATH, "r", encoding="utf-8") as f:
                    status = json.load(f)
                if status.get("status") == "OK":
                    if not status.get("stitched", True):
                        print("[Style3D] [WARN] Result is UNSTITCHED "
                              "(no pre-authored sewlines were found).")
                    print(f"[OK] Style3D drape complete: {status.get('message')}")
                    return status.get("out_glb", out_glb)
                if status.get("status") == "ERROR":
                    raise RuntimeError(f"Style3D drape failed: {status.get('message')}")
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Timed out after {timeout:.0f}s waiting for Style3D drape "
            f"status at {_STATUS_PATH}"
        )
