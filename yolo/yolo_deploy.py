import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("./yolo/best.pt")

# Force the model to use GPU if available
model.to("cuda")

# Check if GPU is available and print the device being used
device = "GPU" if model.device.type != "cpu" else "CPU"
print(f"Using {device} for inference")

# Open the video (or use 0 for webcam)
cap = cv2.VideoCapture("data/recording/recording_20250919_160901.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    # Use tuple (320, 320) to ensure square input matching Roboflow preprocessing
    results = model(frame, imgsz=(320, 320), conf=0.5)  

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Show the frame in a window
    cv2.imshow("YOLOv11n Real-Time Inference", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
