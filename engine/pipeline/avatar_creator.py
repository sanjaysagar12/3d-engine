"""
Pipeline step: Create an SMPL-X avatar mesh from body measurements JSON.

Reads measurement.json, estimates SMPL-X shape betas, exports an OBJ,
and writes a companion arrangement_points.json with CLO-style body landmarks
so GarmentScaler can place panels at the body surface.
"""
import json
import os

import numpy as np


# Reference values for a neutral SMPL-X body (cm)
_REF = {
    "height": 170.0,
    "bust": 90.0,
    "waist": 73.0,
    "hips": 97.0,
    "shoulder_w": 38.0,
}

# SMPL-X joint indices used for arrangement points
_J_PELVIS       = 0
_J_SPINE1       = 3   # lower back / waist level
_J_SPINE3       = 9   # upper chest / bust level
_J_NECK         = 12
_J_L_COLLAR     = 13
_J_R_COLLAR     = 14
_J_L_SHOULDER   = 16
_J_R_SHOULDER   = 17


class AvatarCreator:
    """Generate an SMPL-X body mesh shaped to match body measurements."""

    def __init__(
        self,
        measurements_json: str,
        model_path: str = "models",
        model_type: str = "smplx",
        gender: str = "neutral",
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.gender = gender

        with open(measurements_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.measurements = data.get("body", data)

    # ------------------------------------------------------------------
    # Beta estimation
    # ------------------------------------------------------------------
    def _estimate_betas(self) -> np.ndarray:
        """Map body measurements to 10-dim SMPL-X beta shape parameters."""
        m = self.measurements
        betas = np.zeros(10, dtype=np.float32)

        height = m.get("height", _REF["height"])
        betas[0] = (height - _REF["height"]) / 8.0

        bust = m.get("bust", _REF["bust"])
        betas[1] = (bust - _REF["bust"]) / 12.0

        hips = m.get("hips", _REF["hips"])
        waist = m.get("waist", _REF["waist"])
        betas[2] = ((hips - _REF["hips"]) - (waist - _REF["waist"])) / 15.0

        shoulder = m.get("shoulder_w", _REF["shoulder_w"])
        betas[3] = (shoulder - _REF["shoulder_w"]) / 8.0

        return np.clip(betas, -3.0, 3.0)

    # ------------------------------------------------------------------
    # Arrangement points
    # ------------------------------------------------------------------
    def _compute_arrangement_points(
        self, vertices_m: np.ndarray, joints_m: np.ndarray
    ) -> dict:
        """
        Derive CLO-style arrangement points from SMPL-X geometry.

        SMPL-X coordinate system (T-pose):
          Y  = up,  body faces +Z  (front surface at max-Z, back at min-Z)

        All output positions are in mm.
        """
        m2mm = 1000.0
        EASE_MM = 12.0  # clearance so flat panels don't graze/clip the body surface

        # --- Key joint heights (mm) ---
        bust_y_m    = float(joints_m[_J_SPINE3, 1])
        shoulder_y  = float((joints_m[_J_L_SHOULDER, 1] + joints_m[_J_R_SHOULDER, 1]) / 2.0 * m2mm)
        collar_y    = float((joints_m[_J_L_COLLAR, 1]   + joints_m[_J_R_COLLAR, 1])   / 2.0 * m2mm)
        neck_y      = float(joints_m[_J_NECK, 1] * m2mm)
        waist_y     = float(joints_m[_J_SPINE1, 1] * m2mm)
        bust_y      = float(bust_y_m * m2mm)

        # --- Torso X bounds from collar joints (excludes arms) ---
        # Collar joints sit at the base of the neck and are reliably within
        # the torso, unaffected by arm pose.
        l_collar_x = abs(float(joints_m[_J_L_COLLAR, 0]))
        r_collar_x = abs(float(joints_m[_J_R_COLLAR, 0]))
        torso_half_x_m = max(l_collar_x, r_collar_x) * 1.6   # 60% margin over collar half-width

        # --- Torso-wide Z envelope (waist up to shoulder) ---
        # A flat garment panel spans the full torso height, so its Z offset
        # must clear the single most-protruding point anywhere along that
        # span (e.g. the belly can stick out further than the chest) —
        # not just the bust-level cross-section.
        waist_y_m    = float(joints_m[_J_SPINE1, 1])
        shoulder_y_m = float((joints_m[_J_L_SHOULDER, 1] + joints_m[_J_R_SHOULDER, 1]) / 2.0)
        y_lo, y_hi = sorted([waist_y_m - 0.03, shoulder_y_m + 0.02])
        y_mask = (vertices_m[:, 1] >= y_lo) & (vertices_m[:, 1] <= y_hi)
        x_mask = np.abs(vertices_m[:, 0]) < torso_half_x_m
        torso_mask = y_mask & x_mask
        tv_mm = vertices_m[torso_mask] * m2mm    # torso-envelope vertices in mm

        body_front_z = float(tv_mm[:, 2].max()) + EASE_MM   # body faces +Z → front = max Z
        body_back_z  = float(tv_mm[:, 2].min()) - EASE_MM   # back  = min Z

        # Torso shoulder-width from joints (more reliable than vertex x-extent)
        body_right_x  = float(joints_m[_J_R_SHOULDER, 0] * m2mm)
        body_left_x   = float(joints_m[_J_L_SHOULDER, 0] * m2mm)
        body_center_x = float((body_right_x + body_left_x) / 2.0)

        # X centre for left/right AP: quarter of total shoulder width
        quarter_x = abs(body_right_x - body_center_x) / 2.0

        def pt(x, y, z, nx, nz, desc):
            return {
                "position": [round(x, 2), round(y, 2), round(z, 2)],
                "normal":   [float(nx), 0.0, float(nz)],
                "description": desc,
            }

        points = {
            "AP_FRONT_RIGHT": pt(body_center_x + quarter_x, bust_y, body_front_z,  0,  1, "Right half front — bust level"),
            "AP_FRONT_LEFT":  pt(body_center_x - quarter_x, bust_y, body_front_z,  0,  1, "Left half front  — bust level"),
            "AP_BACK_RIGHT":  pt(body_center_x + quarter_x, bust_y, body_back_z,   0, -1, "Right half back  — bust level"),
            "AP_BACK_LEFT":   pt(body_center_x - quarter_x, bust_y, body_back_z,   0, -1, "Left half back   — bust level"),
        }

        body_dimensions = {
            "bust_y":        round(bust_y, 2),
            "shoulder_y":    round(shoulder_y, 2),
            "collar_y":      round(collar_y, 2),
            "neck_y":        round(neck_y, 2),
            "waist_y":       round(waist_y, 2),
            "body_front_z":  round(body_front_z, 2),
            "body_back_z":   round(body_back_z, 2),
            "body_center_x": round(body_center_x, 2),
            "body_right_x":  round(body_right_x, 2),
            "body_left_x":   round(body_left_x, 2),
        }

        return {
            "units": "mm",
            "coordinate_system": "Y-up, body faces +Z (front=max-Z, back=min-Z)",
            "points": points,
            "body_dimensions": body_dimensions,
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self, out_obj: str) -> str:
        """Build avatar mesh, write OBJ and arrangement_points.json."""
        import torch
        import smplx
        import trimesh

        betas = self._estimate_betas()
        print(f"  Measurements -> betas: {betas[:4].tolist()}")

        model = smplx.create(
            model_path=self.model_path,
            model_type=self.model_type,
            gender=self.gender,
            ext="npz",
            use_pca=False,
            flat_hand_mean=True,
        )

        betas_t = torch.tensor(betas[np.newaxis, :10], dtype=torch.float32)

        fw_kwargs: dict = dict(
            betas=betas_t,
            global_orient=torch.zeros(1, 3),
        )
        if self.model_type == "smplx":
            fw_kwargs.update(
                body_pose=torch.zeros(1, 63),
                jaw_pose=torch.zeros(1, 3),
                leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                left_hand_pose=torch.zeros(1, 45),
                right_hand_pose=torch.zeros(1, 45),
                expression=torch.zeros(1, 10),
            )
        else:
            fw_kwargs["body_pose"] = torch.zeros(1, 69)

        result = model(**fw_kwargs)
        vertices = result.vertices[0].detach().numpy()          # (N, 3) metres
        joints   = result.joints[0, :55].detach().numpy()       # (55, 3) metres

        # Arrangement points — computed before OBJ export so units are consistent
        arrangement = self._compute_arrangement_points(vertices, joints)

        out_dir = os.path.dirname(out_obj)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces, process=False)
        mesh.export(out_obj)
        print(f"  Avatar OBJ written: {out_obj}")

        ap_path = os.path.splitext(out_obj)[0] + "_arrangement_points.json"
        with open(ap_path, "w", encoding="utf-8") as f:
            json.dump(arrangement, f, indent=2)

        dims = arrangement["body_dimensions"]
        print(
            f"  Arrangement points: "
            f"front_z={dims['body_front_z']:.1f}mm  "
            f"back_z={dims['body_back_z']:.1f}mm  "
            f"shoulder_y={dims['shoulder_y']:.1f}mm"
        )
        return out_obj
