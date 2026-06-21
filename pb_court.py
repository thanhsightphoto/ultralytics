"""
Shared court-geometry helpers for the pickleball bounce pipeline.

A camera calibration is a homography H that maps IMAGE pixels -> COURT feet on
the ground plane. Because a real bounce happens ON the ground, its pixel maps
correctly into court coordinates; a mid-air paddle contact, mapped through the
same ground homography, projects OUTSIDE the court rectangle (its height pushes
the ray past the baseline). So "is this point inside the court?" is an
angle-independent floor gate -- calibrate once per camera, no hand-tuned
thresholds.

Court model (standard pickleball, feet):
    x: 0 .. 20   (sideline to sideline)
    y: 0 .. 44   (far baseline = 0, near baseline = 44)
    net at y = 22, non-volley ("kitchen") lines at y = 15 and y = 29.
"""

import json
from pathlib import Path

import numpy as np

WIDTH_FT = 20.0
LENGTH_FT = 44.0
NET_Y = 22.0
KITCHEN_FT = 7.0
KITCHEN_NEAR = NET_Y + KITCHEN_FT   # 29
KITCHEN_FAR = NET_Y - KITCHEN_FT    # 15

# Clicked corner order used everywhere: far-left, far-right, near-right, near-left
COURT_CORNERS = np.float32([[0, 0], [WIDTH_FT, 0],
                           [WIDTH_FT, LENGTH_FT], [0, LENGTH_FT]])


def homography_from_corners(img_pts):
    """img_pts: 4 (x,y) image pixels in order FL, FR, NR, NL. Returns 3x3 H."""
    import cv2
    src = np.float32(img_pts)
    return cv2.getPerspectiveTransform(src, COURT_CORNERS)


def img_to_court(H, x, y):
    """Map an image pixel to court feet."""
    v = H @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def court_to_img(H, X, Y):
    """Map court feet back to an image pixel (uses inverse of H)."""
    Hinv = np.linalg.inv(H)
    v = Hinv @ np.array([X, Y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def in_court(H, x, y, margin=2.0):
    """True if image pixel (x,y) maps inside the court (+margin feet)."""
    X, Y = img_to_court(H, x, y)
    return -margin <= X <= WIDTH_FT + margin and -margin <= Y <= LENGTH_FT + margin


def save_calib(path, img_pts, width, height):
    H = homography_from_corners(img_pts)
    Path(path).write_text(json.dumps({
        "image_size": [width, height],
        "corner_order": ["far_left", "far_right", "near_right", "near_left"],
        "image_points": [[float(a), float(b)] for a, b in img_pts],
        "court_points_ft": COURT_CORNERS.tolist(),
        "homography": H.tolist(),
    }, indent=2))
    return H


def load_calib(path):
    """Return (H, dict) or (None, None) if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    return np.array(d["homography"]), d


def court_line_segments():
    """Court lines as ((X0,Y0),(X1,Y1)) in feet, for drawing an overlay."""
    W, L = WIDTH_FT, LENGTH_FT
    segs = [
        ((0, 0), (W, 0)), ((0, L), (W, L)),          # baselines
        ((0, 0), (0, L)), ((W, 0), (W, L)),          # sidelines
        ((0, NET_Y), (W, NET_Y)),                    # net
        ((0, KITCHEN_FAR), (W, KITCHEN_FAR)),        # far kitchen
        ((0, KITCHEN_NEAR), (W, KITCHEN_NEAR)),      # near kitchen
        ((W / 2, 0), (W / 2, KITCHEN_FAR)),          # centerline (far service)
        ((W / 2, KITCHEN_NEAR), (W / 2, L)),         # centerline (near service)
    ]
    return segs
