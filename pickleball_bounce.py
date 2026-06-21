"""
Pickleball bounce detection (step 1).

Pipeline:
  1. Run YOLO detection per frame on the trimmed clip (class 32 = sports ball).
  2. Filter out the static blob false-positive by box size.
  3. Pick one ball center per frame, interpolate short gaps.
  4. Detect bounces = local maxima of the ball's vertical position (cy).
  5. Write pb_output/bounces.csv and an annotated MP4 with trail + bounce pops.

Run:  source venv/bin/activate && python pickleball_bounce.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

import pb_court as pc
from ultralytics import YOLO

CALIB_PATH = "pb_output/court_calib.json"

# --- config ----------------------------------------------------------------
CLIP = "/Volumes/T7/Pickelball /Drill/DJI_0671_dink_1min.mp4"
OUT_DIR = Path("pb_output")
IMG_SIZE = 1280          # upsample helps the small, fast ball
CONF = 0.25              # detection confidence floor
MAX_BOX = 70             # px: real ball is ~20-50px; the blob is ~150px -> drop it
MAX_GAP = 8              # frames: interpolate misses up to this length
SMOOTH = 5               # moving-average window on cy before peak finding
MIN_BOUNCE_GAP_S = 0.35  # min seconds between consecutive bounces
PROMINENCE = 12          # px: how pronounced a cy peak must be to count

# Perspective-aware floor gate (camera-specific). A height-reversal is only a
# floor bounce if it happens at/below the court surface for its side of the net;
# reversals high in the air are paddle contacts, not ground contacts. The left
# half is shallow (floor ~y590) so airborne points stand out; the right half
# recedes to the back fence so real floor bounces there ride high on screen.
NET_X = 900
LEFT_FLOOR_Y = 540
RIGHT_FLOOR_Y = 360

OUT_DIR.mkdir(exist_ok=True)


def on_floor(x, y):
    """True if (x, y) is at/below the court surface for its side of the net."""
    return y >= (LEFT_FLOOR_Y if x < NET_X else RIGHT_FLOOR_Y)


def detect_ball_per_frame(model, cap, n_frames):
    """Return dict frame_idx -> (cx, cy) for the best filtered ball detection."""
    centers = {}
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, classes=[32], conf=CONF, imgsz=IMG_SIZE,
                          verbose=False)[0]
        best = None
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            if w > MAX_BOX or h > MAX_BOX:      # blob / false positive -> skip
                continue
            conf = float(b.conf[0])
            if best is None or conf > best[2]:
                best = ((x1 + x2) / 2, (y1 + y2) / 2, conf)
        if best is not None:
            centers[i] = (best[0], best[1])
        if i % 200 == 0:
            print(f"  ...processed frame {i}/{n_frames}")
    return centers


def interpolate(centers, n_frames):
    """Fill short gaps. Returns cx, cy arrays (NaN where unfilled)."""
    cx = np.full(n_frames, np.nan)
    cy = np.full(n_frames, np.nan)
    for f, (x, y) in centers.items():
        cx[f], cy[f] = x, y

    def fill(arr):
        idx = np.where(~np.isnan(arr))[0]
        if len(idx) < 2:
            return arr
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < (b - a) <= MAX_GAP + 1:
                arr[a:b + 1] = np.linspace(arr[a], arr[b], b - a + 1)
        return arr

    return fill(cx), fill(cy)


def find_bounces(cy, fps):
    """Bounce = local maximum of cy (lowest point on screen before going back up)."""
    valid = ~np.isnan(cy)
    series = cy.copy()
    # smooth only over valid stretch using simple moving average
    filled = np.where(valid, series, np.nan)
    kernel = np.ones(SMOOTH) / SMOOTH
    sm = np.convolve(np.nan_to_num(filled), kernel, mode="same")
    sm[~valid] = np.nan

    peaks, _ = find_peaks(
        np.nan_to_num(sm, nan=-1e9),
        distance=max(1, int(MIN_BOUNCE_GAP_S * fps)),
        prominence=PROMINENCE,
    )
    # keep peaks only where we actually had data
    return [p for p in peaks if valid[p]]


def main():
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Clip: {n_frames} frames @ {fps:.2f} fps, {w}x{h}")

    model = YOLO("yolo11n.pt")
    print("Detecting ball per frame...")
    centers = detect_ball_per_frame(model, cap, n_frames)
    cap.release()
    print(f"Ball detected in {len(centers)}/{n_frames} frames "
          f"({100*len(centers)/n_frames:.1f}%)")

    cx, cy = interpolate(centers, n_frames)
    bounces = find_bounces(cy, fps)
    # drop mid-air paddle reversals that aren't on the court surface.
    # Prefer a clicked court calibration (angle-independent); else fall back to
    # the per-side thresholds (camera-specific — see calibrate_court.py).
    H, _ = pc.load_calib(CALIB_PATH)
    if H is not None:
        print(f"Using court calibration {CALIB_PATH} (homography floor gate)")
        bounces = [b for b in bounces if pc.in_court(H, cx[b], cy[b])]
    else:
        print("No calibration found -> using per-side threshold gate (this camera only)")
        bounces = [b for b in bounces if on_floor(cx[b], cy[b])]
    print(f"Detected {len(bounces)} floor bounces")

    # write CSV
    csv_path = OUT_DIR / "bounces.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bounce_id", "frame", "time_sec", "x_px", "y_px"])
        for k, fr in enumerate(bounces, 1):
            wr.writerow([k, fr, round(fr / fps, 3),
                        round(float(cx[fr]), 1), round(float(cy[fr]), 1)])
    print(f"Wrote {csv_path}")

    # annotated video: ball trail + bounce pops
    cap = cv2.VideoCapture(CLIP)
    out_path = OUT_DIR / "annotated.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (w, h))
    bounce_set = {int(b): (cx[b], cy[b]) for b in bounces}
    trail = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if not np.isnan(cx[i]):
            trail.append((int(cx[i]), int(cy[i])))
        trail = trail[-25:]
        for j in range(1, len(trail)):
            cv2.line(frame, trail[j - 1], trail[j], (0, 255, 255), 2)
        if not np.isnan(cx[i]):
            cv2.circle(frame, (int(cx[i]), int(cy[i])), 6, (0, 255, 0), 2)
        # show a bounce marker for ~0.5s after each bounce
        for bf, (bx, by) in bounce_set.items():
            if 0 <= i - bf < int(0.5 * fps):
                rad = 12 + 4 * (i - bf)
                cv2.circle(frame, (int(bx), int(by)), rad, (0, 0, 255), 3)
        vw.write(frame)
    cap.release()
    vw.release()
    print(f"Wrote {out_path}")
    print("\nDONE. Inspect pb_output/annotated.mp4 and pb_output/bounces.csv")


if __name__ == "__main__":
    main()
