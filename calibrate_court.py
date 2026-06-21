"""
Court calibration — run ONCE per camera angle, before bounce detection.

You click the 4 outer court corners on a reference frame; the script computes the
image->court homography, saves it to pb_output/court_calib.json, and writes a
verification overlay (pb_output/court_overlay.png) with the court model redrawn
on the frame so you can confirm the lines land on the real court.

USAGE
  # interactive (needs a desktop; opens a window, click FL, FR, NR, NL in order):
  python calibrate_court.py --video "/Volumes/T7/Pickelball /Drill/clip.mp4"

  # non-interactive (you already know the 4 corner pixels), order FL FR NR NL:
  python calibrate_court.py --video "...clip.mp4" \
      --points "68,615 1810,580 1690,718 15,788"

  # verify the homography math only, no image needed:
  python calibrate_court.py --selftest

Corner click order (always the same):
  1) FAR-left   2) FAR-right   3) NEAR-right   4) NEAR-left
(FAR = the baseline on the far side of the net; NEAR = closest to the camera.)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import pb_court as pc

CALIB = "pb_output/court_calib.json"
OVERLAY = "pb_output/court_overlay.png"
LABELS = ["1: FAR-LEFT", "2: FAR-RIGHT", "3: NEAR-RIGHT", "4: NEAR-LEFT"]


def grab_frame(video, at=0.2):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {video}")
    return frame


def click_corners(frame):
    pts = []
    disp = frame.copy()
    win = "Click court corners: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT (Enter=done)"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(disp, LABELS[len(pts) - 1], (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 4:   # Enter
            break
        if k == 27:                            # Esc
            raise SystemExit("Calibration cancelled.")
    cv2.destroyAllWindows()
    return pts


def draw_overlay(frame, H):
    out = frame.copy()
    for (X0, Y0), (X1, Y1) in pc.court_line_segments():
        p0 = pc.court_to_img(H, X0, Y0)
        p1 = pc.court_to_img(H, X1, Y1)
        cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


def selftest():
    # Make a synthetic camera, project known court points, recover and check.
    img_pts = [(100, 300), (1200, 280), (1150, 1000), (60, 1050)]
    H = pc.homography_from_corners(img_pts)
    for (px, py), (cx, cy) in zip(img_pts, pc.COURT_CORNERS):
        gx, gy = pc.img_to_court(H, px, py)
        assert abs(gx - cx) < 1e-6 and abs(gy - cy) < 1e-6, "corner round-trip failed"
    # a point clearly above the far baseline in image -> court Y < 0 -> out
    assert not pc.in_court(H, 630, 150), "airborne point should map outside court"
    # a point near court center -> inside
    assert pc.in_court(H, 600, 640), "center point should map inside court"
    print("selftest OK: homography round-trips and the court gate behaves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--points", help='"x1,y1 x2,y2 x3,y3 x4,y4" in order FL FR NR NL')
    ap.add_argument("--at", type=float, default=0.2, help="seconds into clip for frame")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.video:
        raise SystemExit("--video is required (or use --selftest)")

    Path("pb_output").mkdir(exist_ok=True)
    frame = grab_frame(args.video, args.at)
    h, w = frame.shape[:2]

    if args.points:
        pts = [tuple(map(float, p.split(","))) for p in args.points.split()]
        if len(pts) != 4:
            raise SystemExit("--points needs exactly 4 'x,y' pairs")
    else:
        pts = click_corners(frame)

    H = pc.save_calib(CALIB, pts, w, h)
    cv2.imwrite(OVERLAY, draw_overlay(frame, H))
    print(f"Saved {CALIB}")
    print(f"Saved {OVERLAY} -- open it and confirm the yellow lines match the court")
    print("Corners (FL,FR,NR,NL):", [(round(x), round(y)) for x, y in pts])


if __name__ == "__main__":
    main()
