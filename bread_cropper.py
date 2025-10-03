# save_bread_crops.py
import cv2
import os
import numpy as np

VIDEO_PATH = "./recording_20250919_160901.avi"
OUTPUT_DIR = "./bread_dataset/train/normal"  # Only normal breads for training

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    crop_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ROI (set based on your slider values from before)
        roi_top, roi_bottom, roi_left, roi_right = 142, 699, 0, frame.shape[1]
        roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh_img = cv2.threshold(blurred, 39, 255, cv2.THRESH_BINARY_INV)
        thresh_img = cv2.bitwise_not(thresh_img)

        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 20_000]

        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bread_crop = roi_frame[y:y+h, x:x+w]
            bread_crop = cv2.resize(bread_crop, (256, 256))  # normalize size
            save_path = os.path.join(OUTPUT_DIR, f"bread_{frame_idx}_{crop_idx}.jpg")
            cv2.imwrite(save_path, bread_crop)
            crop_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames, saved {crop_idx} crops.")

    cap.release()
    print(f"Done! Saved {crop_idx} bread crops at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
