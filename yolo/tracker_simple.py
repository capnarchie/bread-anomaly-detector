import os
import queue
import sys
import threading
import cv2
import numpy as np
import torch
from ultralytics import FastSAM
import time
import math
from collections import deque
import centroid_tracker as ct
import validator as v

# Load template
template_img = cv2.imread("./output_frames/wide_template.jpg")
gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
_, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
template_contour = max(contours[0], key=cv2.contourArea)

# Add parent directory to path to import relay module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from relay import on_relay, off_relay

# Relay control with single worker thread
relay_queue = queue.Queue()
relay_active = False
relay_lock = threading.Lock()
activated_bread_ids = set()  # Track which bread IDs have already activated the relay

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

def activate_relay_with_delay(relay_number, duration=0.1, bread_id=None):
    """
    Queue a relay activation request (non-blocking)
    relay_number: The relay to activate 
    duration: How long to keep the relay on (in seconds)
    bread_id: The ID of the bread triggering the relay (optional, for tracking)
    """
    # Check if this bread ID has already activated the relay
    if bread_id is not None:
        if bread_id in activated_bread_ids:
            return  # Skip if this bread ID has already activated the relay
        activated_bread_ids.add(bread_id)  # Mark this bread ID as activated
    
    with relay_lock:
        if relay_active:
            return  # Skip if relay is already active
    
    # Add request to queue (non-blocking)
    relay_queue.put((relay_number, duration))

# Start the relay worker thread once
relay_thread = threading.Thread(target=relay_worker_thread, daemon=True)
relay_thread.start()

# Create window
cv2.namedWindow('Simple Bread Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Simple Bread Tracking', 800, 600)

def update_threshold(x): pass
cv2.createTrackbar('Threshold', 'Simple Bread Tracking', 15, 100, update_threshold)
def draw_tracking_results(frame, tracked_breads):
    """Draw bounding boxes and IDs for tracked breads"""
    
    for bread_id, bread_data in tracked_breads.items():
        x1, y1, x2, y2 = bread_data['box']
        center = bread_data['center']
        missing = bread_data['missing_frames']
        
        # Choose color (green if active, yellow if missing)
        color = (0, 255, 0) if missing == 0 else (0, 255, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(frame, center, 5, color, -1)
        
        # Draw ID label
        label = f"Bread_{bread_id}"
        if missing > 0:
            label += f" (Missing: {missing})"
        
        # Text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

def main():
    """Simple bread tracking with one FastSAM model"""
    

    # Load FastSAM model
    sam_model = FastSAM('peenike_leib_best.pt')
    sam_model.to('cuda')
    print("model loaded to: ", torch.cuda.get_device_name(0))
    # Initialize simple tracker
    tracker = ct.SimpleBreadTracker()
    validator = v.SimpleValidator()
    # Open video
    video_path = 'output/recordings/recording_20251022_165450.avi'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    
    fps_counter = deque(maxlen=30)
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            # Get threshold
            threshold = cv2.getTrackbarPos('Threshold', 'Simple Bread Tracking') / 100.0
            start_time = time.time()
            # Validation zone reference from center of the frame
            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            x1_val = int(center_x - 0.1 * width)
            x2_val = int(center_x + 0.1 * width)
            y1_val = int(center_y - 0.35 * height)
            y2_val = int(center_y + 0.35 * height)

            # Run FastSAM detection
            results = sam_model.predict(frame, imgsz=640, verbose=False, conf=0.8, retina_masks=True)
            
            # Extract bounding boxes from results
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    detections.append((x1, y1, x2, y2))
            
            # Update our simple tracker
            tracked_breads = tracker.update(detections)
            
            # Draw tracking results
            draw_tracking_results(frame, tracked_breads)
            
            # Find contours on the masks and check if they are within validation zone
            # then when in validation zone perform cv2.matchShapes and add the detection to validator
            # after the validator decides if it is defective or not draw the info on the frame
            can_exit = False
            for bread_id, bread_data in tracked_breads.items():
                x1, y1, x2, y2 = bread_data['box']
                # Extract the mask for this bread
                if results and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    for mask in masks:
                        # Create binary mask for the bread
                        binary_mask = (mask > 0.5).astype(np.uint8) * 255
                        # Find contours
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        largest_contour = max(contours, key=cv2.contourArea) if contours else None
                        if largest_contour is not None:
                            # Check if centroid is in validation zone
                            in_zone = validator.is_centroid_in_validation_zone(largest_contour, (x1_val, y1_val, x2_val, y2_val))
                            in_exit_zone = validator.is_centroid_in_trigger_zone(largest_contour, (1060, 175, 1200, 900))
                            if in_zone:
                                # Perform shape matching with a predefined template contour
                                match_score = cv2.matchShapes(template_contour, largest_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                                
                                is_defective = match_score > threshold
                                validator.add_detection(is_defective, bread_id)
                                #print(f"Bread ID {bread_id}: Match Score = {match_score:.4f}, threshold = {threshold:.4f}, Defective = {is_defective}")
                                # Draw validation info
                                is_valid = validator.is_valid(bread_id)
                                status_text = "BAD" if is_valid else "OK"
                                color = (0, 0, 255) if is_valid else (255, 255, 0)
                                cv2.putText(frame, f"Validation: {status_text}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                            if in_exit_zone and validator.is_valid(bread_id):
                                # Reset validator for this bread as it exits
                                #validator.detections = [d for d in validator.detections if d[1] != bread_id]
                                validator.add_exit_detection(bread_id)
                                can_exit = validator.can_exit_detection(bread_id)
                                if can_exit == True:
                                    activate_relay_with_delay(relay_number=1, duration=0.1, bread_id=bread_id)
                                    can_exit = False
                                    #print(f"Bread ID {bread_id} exited - REJECT activated")
                                cv2.putText(frame, "REJECT", (x1, y2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)

                            
            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_counter.append(fps)
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 50), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x1_val, y1_val), (x2_val, y2_val), (255, 0, 0), 2)
            cv2.putText(frame, "Validation", (x1_val, y1_val - 10), cv2.QT_FONT_NORMAL, 2, (0, 255, 255), 2)
            cv2.rectangle(frame, (1060, 175), (1200, 900), (255, 0, 0), 2)
            cv2.putText(frame, "Reject", (1060, 170), cv2.QT_FONT_NORMAL, 2, (0, 255, 255), 2)
        # Display frame
        cv2.imshow('Simple Bread Tracking', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar to pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()