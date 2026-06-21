"""
Step 2: accumulating bounce-distribution overlay.

Reads pb_output/bounces.csv and renders each bounce as a numbered circle that
POPS in (scale overshoot) at its time and then PERSISTS for the rest of the
clip, so the markers accumulate into a spatial distribution map of the rally.

Run: source venv/bin/activate && python bounce_distribution.py
"""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

import pb_court as pc

CLIP = "/Volumes/T7/Pickelball /Drill/DJI_0671_dink_1min.mp4"
OUT = Path("pb_output/bounce_distribution.mp4")
CSV = Path("pb_output/bounces.csv")
CALIB_PATH = "pb_output/court_calib.json"

POP_DUR = 0.45        # seconds the pop animation lasts
BASE_R = 24           # resting marker radius (px)
FILL_ALPHA = 0.35     # translucency of the accumulated fill
COLOR = (60, 90, 255)     # BGR: warm red-orange
RING = (255, 255, 255)    # white ring + number

# --- perspective-aware floor gate (camera-specific calibration) ------------
# The court surface sits at very different screen heights on each side of the
# net, so we gate per side. On the shallow LEFT half the floor is ~y590, so any
# height-reversal far above it (smaller y) is a mid-air paddle contact, not a
# floor bounce -> reject. The deep RIGHT half recedes toward the back fence, so
# genuine floor bounces there ride high on screen -> keep them.
NET_X = 900            # net post x in image (splits the two court halves)
LEFT_FLOOR_Y = 540     # left of net: reject bounces with y above (< ) this
RIGHT_FLOOR_Y = 360    # right of net: deep court, floor reaches this high


def back_out(p, s=1.70158):
    """Ease-out-back: overshoots 1.0 then settles. p in [0,1]."""
    p -= 1.0
    return 1.0 + (s + 1.0) * p ** 3 + s * p ** 2


def on_floor(x, y, H):
    """True if (x, y) is on the court surface. Prefer calibration; else threshold."""
    if H is not None:
        return pc.in_court(H, x, y)
    floor = LEFT_FLOOR_Y if x < NET_X else RIGHT_FLOOR_Y
    return y >= floor


def load_bounces():
    H, _ = pc.load_calib(CALIB_PATH)
    print("gate:", "court calibration" if H is not None else "per-side thresholds")
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            x, y = float(r["x_px"]), float(r["y_px"])
            if not on_floor(x, y, H):   # drop mid-air paddle reversals
                continue
            rows.append((int(r["frame"]), x, y))
    rows.sort()                      # chronological
    # renumber sequentially after filtering
    return [(i + 1, f, x, y) for i, (f, x, y) in enumerate(rows)]


def main(source=CLIP, out=OUT, audio_from=CLIP):
    out = Path(out)
    tmp = out.with_name(out.stem + "_silent.mp4")   # OpenCV writes here first
    bounces = load_bounces()
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pop_frames = POP_DUR * fps

    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"Rendering {n} frames ({source}) -> {out}")

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break

        active = [b for b in bounces if b[1] <= i]

        # 1) translucent accumulated fills (blended in one pass)
        if active:
            overlay = frame.copy()
            for bid, bf, x, y in active:
                age = i - bf
                scale = back_out(min(age / pop_frames, 1.0)) if age < pop_frames else 1.0
                cv2.circle(overlay, (int(x), int(y)), int(BASE_R * scale), COLOR, -1)
            cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0, frame)

        # 2) opaque rings, numbers, and expanding "ping" for fresh pops
        for bid, bf, x, y in active:
            age = i - bf
            scale = back_out(min(age / pop_frames, 1.0)) if age < pop_frames else 1.0
            r = int(BASE_R * scale)
            cv2.circle(frame, (int(x), int(y)), r, RING, 2)
            if age < pop_frames:  # expanding fading ping ring
                ping_r = int(BASE_R + 40 * (age / pop_frames))
                cv2.circle(frame, (int(x), int(y)), ping_r, COLOR, 2)
            cv2.putText(frame, str(bid), (int(x) - 9, int(y) + 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, RING, 2, cv2.LINE_AA)

        # 3) running counter
        cv2.putText(frame, f"bounces: {len(active)}/{len(bounces)}",
                   (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, RING, 3, cv2.LINE_AA)
        vw.write(frame)

    cap.release()
    vw.release()

    # finalize: convert to H.264 and mux audio from audio_from (optional -> the
    # "1:a:0?" mapping is a no-op if that file has no audio track).
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-i", str(audio_from),
         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )
    tmp.unlink()
    print(f"DONE -> {out} (H.264 + audio)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Accumulating numbered bounce circles.")
    ap.add_argument("--source", default=CLIP,
                   help="base video to draw on (e.g. pb_output/annotated.mp4 to keep "
                        "the Ultralytics flight path)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--audio-from", default=CLIP,
                   help="clip to copy the audio track from (default: the trimmed clip)")
    a = ap.parse_args()
    main(a.source, Path(a.out), a.audio_from)
