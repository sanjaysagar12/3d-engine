"""
Minimal DXF R2000 exporter for garment pattern pieces.

Exports each piece outline as a closed LWPOLYLINE.  Cubic bezier curves are
approximated by sampling BEZIER_STEPS evenly-spaced points per segment.
No external libraries required — DXF is a plain-text group-code format.

Units: millimetres ($INSUNITS = 4).  Y axis is flipped relative to SVG
(SVG Y-down → DXF Y-up) so patterns appear oriented correctly in CAD tools.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from ..models.piece import Piece

BEZIER_STEPS = 24  # interpolation points per cubic bezier segment


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cubic_bezier_sample(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    steps: int,
) -> List[Tuple[float, float]]:
    """Return `steps` points along the cubic bezier (excludes p0)."""
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1.0 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def _piece_outline_points(
    piece: Piece, padding: float
) -> Tuple[List[Tuple[float, float]], float]:
    """Return (polyline_pts_svg_space, height_mm).

    polyline_pts are in shifted SVG coordinates (Y-down, origin at top-left
    with padding).  Caller flips Y for DXF.
    """
    min_x, min_y, max_x, max_y = piece.bounds()
    ox = -min_x + padding
    oy = -min_y + padding
    height = (max_y - min_y) + 2 * padding

    pts: List[Tuple[float, float]] = []
    for seg in piece.segments:
        sx = seg.start.x + ox
        sy = seg.start.y + oy
        ex = seg.end.x + ox
        ey = seg.end.y + oy

        if not pts:
            pts.append((sx, sy))

        if seg.type == 'L':
            pts.append((ex, ey))
        elif seg.type == 'C' and seg.control_points and len(seg.control_points) >= 2:
            cp1 = seg.control_points[0]
            cp2 = seg.control_points[1]
            pts.extend(_cubic_bezier_sample(
                (sx, sy),
                (cp1.x + ox, cp1.y + oy),
                (cp2.x + ox, cp2.y + oy),
                (ex, ey),
                BEZIER_STEPS,
            ))

    # Drop duplicate closing vertex if path ended at start
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-6 and abs(pts[0][1] - pts[-1][1]) < 1e-6:
        pts = pts[:-1]

    return pts, height


def _dxf_lines(group_code: int, value) -> List[str]:
    """Return a pair of DXF lines: code line then value line."""
    return [f"{group_code:>3}", str(value)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_piece_dxf(piece: Piece, out_path: str, padding: float = 5.0) -> None:
    """Write a DXF R2000 file for *piece* to *out_path*.

    The outline is a single closed LWPOLYLINE layer "OUTLINE".
    The piece name is a TEXT entity on layer "TEXT".
    """
    svg_pts, height = _piece_outline_points(piece, padding)
    if len(svg_pts) < 2:
        return

    # Flip Y: SVG Y-down → DXF Y-up
    dxf_pts = [(x, height - y) for x, y in svg_pts]

    cx = sum(p[0] for p in dxf_pts) / len(dxf_pts)
    cy = sum(p[1] for p in dxf_pts) / len(dxf_pts)

    rows: List[str] = []

    def emit(*pairs):
        """Emit (group_code, value) pairs in sequence."""
        it = iter(pairs)
        for code in it:
            val = next(it)
            rows.extend(_dxf_lines(code, val))

    # ── HEADER ──────────────────────────────────────────────────────────────
    emit(0, "SECTION", 2, "HEADER")
    emit(9, "$ACADVER",  1, "AC1015")   # R2000
    emit(9, "$INSUNITS", 70, 4)         # 4 = millimetres
    emit(0, "ENDSEC")

    # ── ENTITIES ─────────────────────────────────────────────────────────────
    emit(0, "SECTION", 2, "ENTITIES")

    # Closed LWPOLYLINE — piece outline
    emit(0, "LWPOLYLINE")
    emit(8, "OUTLINE")          # layer
    emit(90, len(dxf_pts))      # vertex count
    emit(70, 1)                  # closed flag
    emit(43, "0.0")              # constant width
    for x, y in dxf_pts:
        emit(10, f"{x:.6f}", 20, f"{y:.6f}")

    # Centred text label — piece name
    emit(0, "TEXT")
    emit(8, "TEXT")
    emit(10, f"{cx:.6f}", 20, f"{cy:.6f}", 30, "0.0")
    emit(40, "5.0")              # text height 5 mm
    emit(1, piece.name)
    emit(72, 1)                  # horizontal: centred
    emit(11, f"{cx:.6f}", 21, f"{cy:.6f}", 31, "0.0")

    emit(0, "ENDSEC")
    emit(0, "EOF")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(rows) + "\n")
