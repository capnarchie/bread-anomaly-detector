import cv2
from ultralytics import FastSAM, YOLO
import time
import numpy as np
import threading
import queue
import sys
import os
from pypylon import pylon

# Create an instant camera object with the camera device found first.
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Start grabbing continuously (default strategy: GrabStrategy_OneByOne)
camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

# Camera.ImageFormatConverter converts pylon images to OpenCV format
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Add parent directory to path to import relay module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from relay import on_relay, off_relay

# Relay control with single worker thread
relay_queue = queue.Queue()
relay_active = False
relay_lock = threading.Lock()

def relay_worker_thread():
    """
    Single worker thread that processes relay activation requests
    """
    global relay_active
    while True:
        try:
            # Wait for a relay activation request (blocking)
            relay_number, duration = relay_queue.get(timeout=1)
            
            with relay_lock:
                if relay_active:
                    relay_queue.task_done()
                    continue  # Skip if relay is already active
                relay_active = True
            
            try:
                time.sleep(0.1)
                print(f'Activating relay {relay_number} for {duration}s')
                #on_relay(relay_number)
                time.sleep(duration)
                #off_relay(relay_number)
                print(f'Relay {relay_number} deactivated')
            finally:
                with relay_lock:
                    relay_active = False
                relay_queue.task_done()
                
        except queue.Empty:
            continue  # No requests, continue waiting
        except Exception as e:
            print(f"Error in relay worker thread: {e}")
            continue

def activate_relay_with_delay(relay_number, duration=0.1):
    """
    Queue a relay activation request (non-blocking)
    relay_number: The relay to activate 
    duration: How long to keep the relay on (in seconds)
    """
    with relay_lock:
        if relay_active:
            return  # Skip if relay is already active
    
    # Add request to queue (non-blocking)
    relay_queue.put((relay_number, duration))

# Start the relay worker thread once
relay_thread = threading.Thread(target=relay_worker_thread, daemon=True)
relay_thread.start()

# Load the FastSAM model# Load the FastSAM model
model_sam = FastSAM('FastSAM-s.pt')#('runs/train/fastsam11/weights/best.pt')  # Replace with the path to your FastSAM model
model_yolo = YOLO("runs/train/bread_defects_yolov11/weights/best.pt")
video_path = 'recording10172025_3.avi'#'data/leib4.avi'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

#template mask coontours
template_img = cv2.imread("./output_frames/peenleib_template.jpg")  # Use as good bread template
gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
_, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours[0], key=cv2.contourArea)
template_contours = [largest_contour]

cv2.drawContours(template_img, template_contours, -1, (255, 125, 0), 3)
cv2.imshow("Template Contours", template_img)


# Trackbar for threshold - Create window with proper flags for WSL2
cv2.namedWindow('FastSAM Inference', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow('FastSAM Inference', 800, 800)  # Set a specific size to ensure trackbar visibility

def update_threshold(x):
    pass
cv2.createTrackbar('Threshold', 'FastSAM Inference', 15, 100, update_threshold)
try:
    while camera.IsGrabbing():

        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data and convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()
        # ret, frame = cap.read()
        # if not ret:
        #     break

            results = model_sam.predict(frame, imgsz=320, verbose=False, conf=0.9, iou=0.9, retina_masks=True)
            trackbar_value = cv2.getTrackbarPos('Threshold', 'FastSAM Inference')
            threshold = trackbar_value / 100.0  # Convert to 0.0 - 1.0 range

            # Draw the results on the frame
            if results and hasattr(results[0], 'masks') and results[0].masks is not None:
                binary_mask = results[0].masks.data[0].cpu().numpy() > 0.5
                binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8) * 255, (frame.shape[1], frame.shape[0]))
                cv2.imshow("Predicted Binary Mask", binary_mask_resized)
                match_score = cv2.matchShapes(template_contours[0], max(cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea), cv2.CONTOURS_MATCH_I1, 0.0)
                if match_score <= threshold:
                    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)  # Draw the outline in green on the annotated frame
                else:
                    #print(f'Defect detected! Match score: {match_score:.4f} (threshold: {threshold:.2f})')
                    # Activate relay in non-blocking way 
                    activate_relay_with_delay(relay_number=1, duration=0.1)
                    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)  # Draw the outline in red on the annotated frame

            #annotated_frame = results[0].plot()
            #results_yolo = model_yolo(frame, imgsz=320, conf=0.5)  

            # Draw bounding boxes on the frame
            # Draw FastSAM results on the frame
            # annotated_frame = results_yolo[0].plot()

            # Draw YOLO results on the same frame
            # for box in results_yolo[0].boxes:
            #     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
            #     conf = box.conf[0]  # Confidence score
            #     cls = int(box.cls[0])  # Class ID
            #     label = f"{model_yolo.names[cls]} {conf:.2f}"  # Label with class name and confidence

            #     # Draw the bounding box and label on the frame
            #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('FastSAM Inference', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        grabResult.Release()
finally:
    camera.StopGrabbing()
    cv2.destroyAllWindows()

# Release the video capture and close windows
# cap.release()
# cv2.destroyAllWindows()