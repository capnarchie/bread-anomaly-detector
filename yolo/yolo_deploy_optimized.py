import cv2
from ultralytics import YOLO
import time
from collections import deque

# Load your trained model
model = YOLO("runs/train/bread_defects_yolov11/weights/best.pt")#"./yolo/best.pt")

# Force the model to use GPU if available
model.to("cuda")

# Check if GPU is available and print the device being used
device = "GPU" if model.device.type != "cpu" else "CPU"
print(f"Using {device} for inference")

# ==================== OPTIMIZATION PARAMETERS ====================
# Strategy 1: Frame Skipping - Process every Nth frame
FRAME_SKIP = 1  # Process every 3rd frame (adjust based on conveyor speed)

# Strategy 2: Motion Detection - Only run inference when there's movement
USE_MOTION_DETECTION = False
MOTION_THRESHOLD = 500  # Minimum contour area to trigger inference

# Strategy 3: ROI (Region of Interest) - Only process center of frame where bread is fully visible
USE_ROI = True
ROI_START_X = 0.2  # Start at 20% from left (adjust based on your conveyor)
ROI_END_X = 0.8    # End at 80% from left
ROI_START_Y = 0.1  # Start at 10% from top
ROI_END_Y = 0.7    # End at 90% from bottom

# Strategy 4: Batch Processing (for video files, not real-time)
USE_BATCH_PROCESSING = False  # Set to True for offline video analysis

# Performance tracking
SHOW_FPS = True
# =================================================================

# Open the video (or use 0 for webcam)
cap = cv2.VideoCapture("data/recording/recording_20250919_160901.avi")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps} FPS")

# Calculate ROI coordinates
roi_x1 = int(width * ROI_START_X)
roi_x2 = int(width * ROI_END_X)
roi_y1 = int(height * ROI_START_Y)
roi_y2 = int(height * ROI_END_Y)

# For motion detection
prev_frame = None
motion_history = deque(maxlen=5)  # Track motion over last 5 frames

# For FPS tracking
frame_times = deque(maxlen=30)
last_time = time.time()

# For frame skipping
frame_count = 0

# Store last detection results for display on skipped frames
last_results = None

print("\n=== OPTIMIZATION SETTINGS ===")
print(f"Frame Skip: Every {FRAME_SKIP} frames")
print(f"Motion Detection: {'Enabled' if USE_MOTION_DETECTION else 'Disabled'}")
print(f"ROI Processing: {'Enabled' if USE_ROI else 'Disabled'}")
if USE_ROI:
    print(f"  ROI: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")
    print(f" ROI Size: {roi_x2 - roi_x1}x{roi_y2 - roi_y1}")
print("=============================\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    
    # Initialize annotated frame with original
    annotated_frame = frame.copy()
    
    # Determine if we should run inference this frame
    should_infer = False
    skip_reason = ""
    
    # Check 1: Frame skipping
    if frame_count % FRAME_SKIP != 0:
        skip_reason = "Frame skip"
    else:
        should_infer = True
        
        # Check 2: Motion detection
        if USE_MOTION_DETECTION and should_infer:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_frame is not None:
                frame_delta = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check if significant motion detected
                has_motion = any(cv2.contourArea(c) > MOTION_THRESHOLD for c in contours)
                motion_history.append(has_motion)
                
                # Only infer if motion detected in recent frames
                if not any(motion_history):
                    should_infer = False
                    skip_reason = "No motion"
            
            prev_frame = gray
    
    # Run inference if criteria met
    if should_infer:
        # Extract ROI if enabled
        if USE_ROI:
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            inference_frame = roi_frame
        else:
            inference_frame = frame
        
        # Run YOLO inference
        results = model(inference_frame, imgsz=(320, 320), conf=0.5, verbose=False)
        last_results = results
        
        # Draw results on full frame
        if USE_ROI:
            # Draw ROI rectangle
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            
            # Translate detections from ROI to full frame coordinates
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Translate coordinates
                x1_full = int(x1 + roi_x1)
                y1_full = int(y1 + roi_y1)
                x2_full = int(x2 + roi_x1)
                y2_full = int(y2 + roi_y1)
                
                # Draw on full frame
                cv2.rectangle(annotated_frame, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1_full, y1_full - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        else:
            annotated_frame = results[0].plot()
    else:
        # Use last results if available (for display continuity)
        if last_results is not None and USE_ROI:
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (128, 128, 128), 2)
    
    # Calculate and display FPS
    if SHOW_FPS:
        frame_time = current_time - last_time
        frame_times.append(frame_time)
        avg_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
        
        # Display info
        info_text = f"FPS: {avg_fps:.1f} | Frame: {frame_count}"
        if not should_infer and skip_reason:
            info_text += f" | Skipped: {skip_reason}"
        else:
            info_text += " | INFERRING"
        
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    last_time = current_time
    
    # Show the frame in a window
    cv2.imshow("YOLOv11n Optimized Inference", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_count} frames")
if SHOW_FPS and len(frame_times) > 0:
    print(f"Average FPS: {len(frame_times) / sum(frame_times):.1f}")
