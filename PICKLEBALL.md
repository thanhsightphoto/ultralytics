# Pickleball Bounce Tracking — Working Notes

A small pipeline built on top of this Ultralytics checkout to **detect the ball**,
**find where it bounces**, and **visualize the bounce distribution** for a rally,
from drone/phone footage of dinking drills.

> Status: working demo on `DJI_0671_dink_1min.mp4`. Detection is "demo-grade"
> (stock COCO model); see [Limitations](#limitations) and [Next steps](#next-steps).

---

## 1. Environment

Setup done once (Apple M2 Pro, macOS):

```bash
# Python 3.11 via Homebrew, virtualenv in the repo
brew install python@3.11
cd "/Users/thanhvu/Documents/Claude Code/ultralytics"
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"        # installs ultralytics + torch + opencv + scipy ...
yolo checks                    # sanity check
```

**Every session, activate the venv first:**

```bash
cd "/Users/thanhvu/Documents/Claude Code/ultralytics"
source venv/bin/activate
```

Runs on **CPU** (no CUDA; YOLO tracking doesn't fully use Apple MPS yet), so
processing a 1-minute 1080p clip takes a few minutes. `ffmpeg`/`ffprobe` (Homebrew)
are used for trimming and frame extraction.

---

## 2. Files (all in the repo root, git-ignored)

| File | Purpose |
|------|---------|
| [pb_court.py](pb_court.py) | Court geometry + homography helpers (shared) |
| [calibrate_court.py](calibrate_court.py) | **Run once per camera angle** — click 4 court corners → calibration |
| [pickleball_bounce.py](pickleball_bounce.py) | Detect ball → find bounces → write `bounces.csv` + `annotated.mp4` |
| [bounce_distribution.py](bounce_distribution.py) | Render accumulating numbered "pop" circles (`--source`/`--out` selectable) |
| `pb_output/` | All outputs (git-ignored) |

Outputs in `pb_output/`:

| File | What |
|------|------|
| `bounces.csv` | `bounce_id, frame, time_sec, x_px, y_px` — one row per bounce |
| `annotated.mp4` | Ball trajectory trail + per-bounce ping |
| `bounce_distribution.mp4` | Numbered markers that pop in and **accumulate** into a shot map |
| `court_calib.json` | Active calibration (when present) — homography image→court |
| `court_overlay.png` | Calibration check: court model redrawn on a frame |
| `court_calib.example.json` | The rough example calibration from setup (not active) |

The source clips live on the **T7 drive**, not in the repo:
`/Volumes/T7/Pickelball /Drill/` (note the spelling + trailing space → always quote the path).
The 1-minute test cut is `DJI_0671_dink_1min.mp4` there.

---

## 3. The pipeline, end to end

### Step 0 — Trim a rally segment (ffmpeg)

Full clips are ~29 min. Work on a short continuous segment. Example used (9:00–10:00):

```bash
ffmpeg -ss 540 -i "/Volumes/T7/Pickelball /Drill/DJI_0671.MP4" \
  -t 60 -c:v libx264 -crf 18 -preset fast -an \
  "/Volumes/T7/Pickelball /Drill/DJI_0671_dink_1min.mp4"
```

### Step 1 — Calibrate the court (once per camera angle)

```bash
python calibrate_court.py --video "/Volumes/T7/Pickelball /Drill/<clip>.mp4"
```

A window opens on a frame. **Click the 4 outer court corners in this order:**
`FAR-LEFT → FAR-RIGHT → NEAR-RIGHT → NEAR-LEFT` (FAR = far side of the net, NEAR =
closest to the camera), then press **Enter**. It writes `pb_output/court_calib.json`
and `pb_output/court_overlay.png`. **Open the overlay** and confirm the yellow lines
land on the real court; if not, re-run and click more carefully.

No desktop / scripting? Pass corners directly:

```bash
python calibrate_court.py --video "...<clip>.mp4" --points "FLx,FLy FRx,FRy NRx,NRy NLx,NLy"
```

Verify the geometry math anytime: `python calibrate_court.py --selftest`.

### Step 2 — Detect ball + bounces

```bash
python pickleball_bounce.py
```

(edit `CLIP` at the top of the file to point at your clip). It:

1. Runs YOLO (`yolo11n.pt`, class 32 = "sports ball") per frame at `imgsz=1280`.
2. Drops the large static false-positive (the equipment bag) by **box size** (`MAX_BOX`).
3. Picks one ball center per frame, **interpolates** short gaps (`MAX_GAP`).
4. Finds **bounces = local maxima of the ball's vertical position** (lowest screen
   point before it rises again), via `scipy.signal.find_peaks`.
5. **Floor-gates** the bounces (Step 1 calibration if present, else thresholds).
6. Writes `bounces.csv` + `annotated.mp4`.

### Step 3 — Render the bounce distribution

```bash
python bounce_distribution.py
```

Each bounce **pops in** (scale overshoot) at its time/location and **persists**, so
the markers accumulate into a spatial map of where the ball landed during the rally.

### Step 3b — Numbered circles ON TOP of the Ultralytics annotated clip

The preferred "documented" view: keep everything the Ultralytics clip shows (flight-path
trail, live ball, bounce ping) and add the accumulating **numbered circles** on the court.
Just point the same renderer at `annotated.mp4` instead of the raw clip:

```bash
python bounce_distribution.py --source pb_output/annotated.mp4 --out pb_output/bounce_documented.mp4
```

Output: `pb_output/bounce_documented.mp4`. (With no `--source`/`--out` the script draws the
circles on the raw clip → `pb_output/bounce_distribution.mp4`, the clean distribution view.)

**Audio + codec.** OpenCV writes silent `mpeg4`, so the script auto-finalizes each render to
**H.264 + AAC**, muxing audio from `--audio-from` (default: the trimmed clip, which is now cut
*with* audio). The `1:a:0?` mapping is a no-op if that source has no audio. So every output
carries sound and plays everywhere, no extra step.

---

## 4. How bounce detection works (and why markers were wrong)

A **bounce** is detected wherever the ball's vertical image position stops descending
and reverses upward — a local maximum in `cy`. This is correct for floor bounces, but
the ball also reverses when a **paddle hits it in mid-air**. Those mid-air reversals
(e.g. markers 1, 4, 13 in the first pass, up at paddle height) are **not** floor bounces.

**The floor gate** removes them. Two implementations:

- **Calibration (preferred, angle-independent).** At the instant of a real bounce the
  ball is on the ground, so its pixel maps cleanly to court feet via the ground
  homography. A mid-air paddle contact, mapped through that same homography, projects
  **outside the court rectangle** (its height pushes the ray past the baseline). So the
  gate is just: *does this point map inside the court?* Works for any camera angle once
  you've clicked the corners.
- **Per-side thresholds (fallback, this camera only).** `pb_court`/`bounce_distribution`
  use `NET_X`, `LEFT_FLOOR_Y`, `RIGHT_FLOOR_Y`: reject reversals above the court surface,
  with different cutoffs per side because perspective makes the deep back-right corner
  legitimately high on screen. These numbers are tuned to *this* clip's angle.

The code uses calibration if `pb_output/court_calib.json` exists, otherwise the thresholds.

---

## 5. Key parameters (top of the scripts)

| Param | File | Meaning |
|-------|------|---------|
| `CONF` | pickleball_bounce.py | Detection confidence floor (0.25) |
| `IMG_SIZE` | pickleball_bounce.py | Inference size (1280 — upscales the small ball) |
| `MAX_BOX` | pickleball_bounce.py | Reject detections wider/taller than this (kills the bag) |
| `MAX_GAP` | pickleball_bounce.py | Max frame gap to interpolate across |
| `PROMINENCE`, `MIN_BOUNCE_GAP_S` | pickleball_bounce.py | Peak strength / min spacing between bounces |
| `NET_X`, `LEFT_FLOOR_Y`, `RIGHT_FLOOR_Y` | both | Threshold-gate cutoffs (fallback only) |
| `margin` | pb_court.in_court | Court tolerance in feet (default 2.0) |

---

## 6. Limitations

- **Detection is sparse (~25% of frames).** The stock COCO "sports ball" class catches
  the pickleball only when it's clearly in flight; motion blur and paddle/hand occlusion
  cause misses. Interpolation reconstructs the arcs but bounce timing can be a frame or
  two off, and a few bounces may be slightly mispositioned.
- **Calibration precision + lens distortion.** The DJI footage has barrel distortion, so
  a 4-corner planar homography is approximate near the court edges. Hand-eyeballed corners
  were borderline there; **clicking the corners on the full-res frame is much more accurate.**
- **Threshold fallback is camera-specific.** If you process a different angle without
  calibrating, the `*_FLOOR_Y` cutoffs will be wrong — calibrate instead.
- **One rally assumption.** The 1-min test cut actually contains a ~15s gap (≈2 rallies).
  Bounces accumulate across the whole clip.

---

## 7. Next steps (not done yet)

- **Click-calibrate this clip** to switch it onto the angle-independent gate.
- **Lens undistortion** (estimate DJI camera intrinsics, undistort before homography) for
  edge accuracy.
- **Fine-tune the detector** on a few hundred labeled pickleball frames — the real fix for
  sparse/mispositioned detections.
- **Single-rally view / color-by-side / heatmap** rendering variants.
- **Animated overlay polish** (the original "pop circle" idea) via a motion-graphics pass
  once bounce data is trusted.
