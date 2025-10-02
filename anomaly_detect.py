import cv2
import os
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------
# Step 1. Extract frames from video
# ------------------------------
video_path = "recording_20250919_160901.avi"
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_rate = fps // 5  # sample 5 frames per second

count, saved = 0, 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_rate == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{frames_dir}/frame_{saved:05d}.png", gray)
        saved += 1
    count += 1
cap.release()
print(f"Extracted {saved} frames for training.")

# ------------------------------
# Step 2. Load frames into dataset
# ------------------------------
image_size = 128
images = []
for f in glob(f"{frames_dir}/*.png"):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (image_size, image_size))
    images.append(img / 255.0)

images = np.expand_dims(np.array(images), -1)  # (N, H, W, 1)

# ------------------------------
# Step 3. Define and train autoencoder
# ------------------------------
input_img = layers.Input(shape=(image_size, image_size, 1))

# Encoder
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2))(x)

# Decoder
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on normal bread frames
autoencoder.fit(images, images,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_split=0.1)

# Save model
autoencoder.save("bread_autoencoder.h5")
print("Autoencoder trained and saved.")

# ------------------------------
# Step 4. Compute threshold from training errors
# ------------------------------
recon = autoencoder.predict(images)
errors = np.mean(np.square(images - recon), axis=(1,2,3))
threshold = np.percentile(errors, 95)  # top 5% considered anomaly
print("Auto-detected anomaly threshold:", threshold)

# ------------------------------
# Step 5. Run inference on video & annotate
# ------------------------------
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("bread_line_annotated.mp4", fourcc, fps,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find bread contour
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        out.write(frame)
        continue
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop bread ROI
    bread_roi = gray[y:y+h, x:x+w]
    bread_resized = cv2.resize(bread_roi, (image_size, image_size))
    bread_resized = bread_resized.astype("float32") / 255.0
    bread_resized = np.expand_dims(bread_resized, axis=(0,-1))

    # Reconstruction error
    recon = autoencoder.predict(bread_resized, verbose=0)
    error = np.mean((bread_resized - recon) ** 2)

    # Draw bounding box
    color = (0, 255, 0) if error < threshold else (0, 0, 255)
    label = "Normal" if error < threshold else "Defect"
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{label} ({error:.4f})", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
print("Annotated video saved to bread_line_annotated.mp4")
