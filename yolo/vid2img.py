import cv2
import os

# ====== CONFIG ======
VIDEO_PATH = "./data/leib3.avi"#"recording_20250919_160901.avi"         # Path to your video
OUTPUT_DIR = "./data/leib3_frames"           # Folder where frames will be saved
FRAME_PREFIX = "frame_"               # Prefix for image filenames
FRAME_FORMAT = "jpg"                  # Image format (jpg, png, etc.)
FPS_SKIP = 1                          # Save every n-th frame (1 = every frame)
# ===================

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames if FPS_SKIP > 1
    if frame_count % FPS_SKIP != 0:
        continue

    # Construct filename
    filename = os.path.join(OUTPUT_DIR, f"{FRAME_PREFIX}{saved_count:05d}.{FRAME_FORMAT}")
    cv2.imwrite(filename, frame)
    saved_count += 1

    if saved_count % 100 == 0:
        print(f"[INFO] Saved {saved_count} frames...")

cap.release()
print(f"[DONE] Total frames saved: {saved_count}")
