import cv2
from ultralytics import FastSAM
import time
# Load the FastSAM model
model = FastSAM('runs/train/fastsam9/weights/best.pt')  # Replace with the path to your FastSAM model
#model.to('cuda')  # Use GPU if available
# Load the video
video_path = 'data/leib4.avi'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Perform inference on the frame
    start = time.time()
    results = model.predict(frame, imgsz=320, verbose=True)
    end = time.time()
    # Draw the results on the frame
    annotated_frame = results[0].plot()
    #print(f"Inference time in ms: {(end - start) * 1000:.2f}")
    # Draw a 320x320 rectangle at the center of the frame
    # Display the frame
    cv2.imshow('FastSAM Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()