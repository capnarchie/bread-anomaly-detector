import cv2
import os

video_path = "recording_20250919_160901.avi"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_rate = fps // 5  # sample 5 frames per second

count = 0
saved = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_rate == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output_dir}/frame_{saved:05d}.png", gray)
        saved += 1
    count += 1
cap.release()

print(f"Saved {saved} frames.")
