"""Quick pickleball tracking test. Run from repo root with venv activated."""

from ultralytics import YOLO

# Path to a clip on the T7 drive (note the trailing space in "Pickelball ")
SOURCE = "/Volumes/T7/Pickelball /Drill/DJI_0671.MP4"

model = YOLO("yolo11n.pt")  # auto-downloads on first run

# classes=[32] = COCO "sports ball". Drop it to detect everything.
# save=True writes an annotated video to runs/detect/track/ (git-ignored).
results = model.track(source=SOURCE, classes=[32], save=True, show=False)

print("Done. Annotated video saved under runs/detect/track/")
