import cv2
from ultralytics import FastSAM, YOLO
import time
# Load the FastSAM model
model_sam = FastSAM('runs/train/fastsam11/weights/best.pt')  # Replace with the path to your FastSAM model
model_yolo = YOLO("runs/train/bread_defects_yolov11/weights/best.pt")
#model.to('cuda')  # Use GPU if available
# Load the video
video_path = 'data/leib4.avi'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model_sam.predict(frame, imgsz=320, verbose=True)
    # Draw the results on the frame
    annotated_frame = results[0].plot()


    results_yolo = model_yolo(frame, imgsz=320, conf=0.5)  

    # Draw bounding boxes on the frame
    # Draw FastSAM results on the frame
    annotated_frame = results[0].plot()

    # Draw YOLO results on the same frame
    for box in results_yolo[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class ID
        label = f"{model_yolo.names[cls]} {conf:.2f}"  # Label with class name and confidence

        # Draw the bounding box and label on the frame
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('FastSAM Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()