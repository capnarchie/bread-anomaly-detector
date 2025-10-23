import cv2
from ultralytics import FastSAM
import time
import numpy as np
import threading
import queue
import sys
import os
import math
#from pypylon import pylon

# Create an instant camera object with the camera device found first.
# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# # Start grabbing continuously (default strategy: GrabStrategy_OneByOne)
# camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

# # Camera.ImageFormatConverter converts pylon images to OpenCV format
# converter = pylon.ImageFormatConverter()
# converter.OutputPixelFormat = pylon.PixelType_BGR8packed
# converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Add parent directory to path to import relay module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from relay import on_relay, off_relay

# Simple bread tracker with defect status
class BreadTracker:
    def __init__(self):
        self.next_id = 0
        self.tracked_breads = {}  # {id: {'center': (x,y), 'missing_frames': 0, 'box': (x1,y1,x2,y2), 'defect_status': bool}}
        self.max_missing_frames = 10  # Remove bread after 10 missing frames (from persistent tracker)
        self.max_distance = 150  # Max pixels bread can move between frames (from persistent tracker)
        
        # ID management to prevent overflow
        self.max_id = 999  # Maximum ID before recycling (keeps IDs manageable)
        self.used_ids = set()  # Track currently used IDs
        self.total_breads_processed = 0  # Track total for statistics
        
        # Zone parameters
        self.validation_zone_width_percent = 40
        self.validation_zone_height_percent = 60
        self.exit_zone_width = 200  # pixels from right edge
        
        # Defect validation parameters
        self.required_defect_frames = 3
        self.defect_history = {}  # {id: [True/False, True/False, ...]}
        
        # Relay control
        self.last_trigger_time = 0
        self.deadzone_duration = 1.0
    
    def get_next_available_id(self):
        """Get next available ID with recycling to prevent overflow"""
        # First try sequential ID
        if self.next_id <= self.max_id and self.next_id not in self.used_ids:
            new_id = self.next_id
            self.next_id += 1
        else:
            # Find first available ID (recycling)
            new_id = None
            for i in range(self.max_id + 1):
                if i not in self.used_ids:
                    new_id = i
                    break
            
            # If somehow all IDs are used (very unlikely), start over
            if new_id is None:
                print("Warning: All IDs in use, clearing and restarting (this shouldn't happen)")
                self.used_ids.clear()
                new_id = 0
                self.next_id = 1
        
        self.used_ids.add(new_id)
        self.total_breads_processed += 1
        return new_id
    
    def release_id(self, bread_id):
        """Release an ID for recycling"""
        if bread_id in self.used_ids:
            self.used_ids.remove(bread_id)
        if bread_id in self.defect_history:
            del self.defect_history[bread_id]
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_bread_off_frame(self, center, frame_shape, margin=100):
        """Check if bread center is completely outside frame with margin"""
        h, w = frame_shape[:2]
        x, y = center
        return x < -margin or x > w + margin or y < -margin or y > h + margin
    
    def is_in_validation_zone(self, center, frame_shape):
        """Check if center point is in validation zone"""
        h, w = frame_shape[:2]
        zone_w = int(w * self.validation_zone_width_percent / 100)
        zone_h = int(h * self.validation_zone_height_percent / 100)
        
        x1 = (w - zone_w) // 2
        x2 = x1 + zone_w
        y1 = (h - zone_h) // 2
        y2 = y1 + zone_h
        
        cx, cy = center
        return x1 <= cx <= x2 and y1 <= cy <= y2, (x1, y1, x2, y2)
    
    def is_in_exit_zone(self, center, frame_shape):
        """Check if center point is in exit zone (positioned away from frame edge)"""
        h, w = frame_shape[:2]
        cx, cy = center
        edge_margin = 50  # pixels from the right edge
        exit_x1 = w - self.exit_zone_width - edge_margin
        exit_x2 = w - edge_margin
        return exit_x1 <= cx <= exit_x2
    
    def update_defect_status(self, bread_id, is_defective):
        """Update defect history for a bread and determine final status"""
        if bread_id not in self.defect_history:
            self.defect_history[bread_id] = []
        
        # Check if bread is already marked as defective
        current_status = self.tracked_breads.get(bread_id, {}).get('defect_status', False)
        if current_status:
            return True  # Once defective, always defective
        
        # Add current defect detection
        self.defect_history[bread_id].append(is_defective)
        
        # Keep only recent detections (window of 5)
        if len(self.defect_history[bread_id]) > 5:
            self.defect_history[bread_id].pop(0)
        
        # Determine if bread is defective (3 out of last detections)
        recent_detections = self.defect_history[bread_id]
        defect_count = sum(1 for d in recent_detections if d)
        
        return defect_count >= self.required_defect_frames
    
    def update(self, detections, frame_shape):
        """
        Update tracker with new detections and frame info - Using more persistent tracking algorithm
        detections = [(x1, y1, x2, y2, is_defective), ...]
        """
        # Calculate centers of new detections
        new_centers = []
        new_boxes = []
        new_defects = []
        
        for x1, y1, x2, y2, is_defective in detections:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            new_centers.append((center_x, center_y))
            new_boxes.append((x1, y1, x2, y2))
            new_defects.append(is_defective)
        
        # Check for breads in exit zone for triggering (but don't remove them yet!)
        triggered_breads = []
        for bread_id, bread_data in list(self.tracked_breads.items()):
            center = bread_data['center']
            
            # Check if bread is in exit zone and has defect status - TRIGGER but DON'T REMOVE
            if self.is_in_exit_zone(center, frame_shape):
                if bread_data.get('defect_status', False):
                    # Check deadzone
                    if time.time() - self.last_trigger_time >= self.deadzone_duration:
                        triggered_breads.append(bread_id)
                        self.last_trigger_time = time.time()
                
                # Don't remove bread here - let normal missing frame logic handle removal
        
        # If no existing breads, just register all new ones
        if len(self.tracked_breads) == 0:
            for i, (center, box, is_defective) in enumerate(zip(new_centers, new_boxes, new_defects)):
                new_id = self.get_next_available_id()
                self.tracked_breads[new_id] = {
                    'center': center,
                    'missing_frames': 0,
                    'box': box,
                    'defect_status': False,
                    'detection_count': 0,
                    'evaluated': False
                }
                # Initialize defect status
                in_validation_zone, _ = self.is_in_validation_zone(center, frame_shape)
                if in_validation_zone and not self.tracked_breads[new_id]['evaluated']:
                    self.tracked_breads[new_id]['detection_count'] += 1
                    final_defect_status = self.update_defect_status(new_id, is_defective)
                    self.tracked_breads[new_id]['defect_status'] = final_defect_status
                    
                    # Check if we've reached 5 detections
                    if self.tracked_breads[new_id]['detection_count'] >= 5:
                        self.tracked_breads[new_id]['evaluated'] = True
                
            return self.tracked_breads, triggered_breads
        
        # MORE PERSISTENT TRACKING ALGORITHM - Match new detections to existing breads
        used_detections = set()
        updated_breads = set()
        
        # For each existing bread, find closest new detection
        for bread_id, bread_data in list(self.tracked_breads.items()):
            old_center = bread_data['center']
            best_distance = float('inf')
            best_match = None
            
            for i, new_center in enumerate(new_centers):
                if i in used_detections:  # Already matched
                    continue
                
                distance = self.calculate_distance(old_center, new_center)
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update existing bread
                self.tracked_breads[bread_id]['center'] = new_centers[best_match]
                self.tracked_breads[bread_id]['box'] = new_boxes[best_match]
                self.tracked_breads[bread_id]['missing_frames'] = 0
                used_detections.add(best_match)
                updated_breads.add(bread_id)
                
                # Update defect status if in validation zone and not yet evaluated
                center = new_centers[best_match]
                in_validation_zone, _ = self.is_in_validation_zone(center, frame_shape)
                if in_validation_zone and not self.tracked_breads[bread_id]['evaluated']:
                    self.tracked_breads[bread_id]['detection_count'] += 1
                    final_defect_status = self.update_defect_status(bread_id, new_defects[best_match])
                    self.tracked_breads[bread_id]['defect_status'] = final_defect_status
                    
                    # Check if we've reached 5 detections
                    if self.tracked_breads[bread_id]['detection_count'] >= 5:
                        self.tracked_breads[bread_id]['evaluated'] = True
            else:
                # Bread not found, increment missing counter
                self.tracked_breads[bread_id]['missing_frames'] += 1
        
        # Add new breads for unmatched detections
        for i, (center, box, is_defective) in enumerate(zip(new_centers, new_boxes, new_defects)):
            if i not in used_detections:
                new_id = self.get_next_available_id()
                self.tracked_breads[new_id] = {
                    'center': center,
                    'missing_frames': 0,
                    'box': box,
                    'defect_status': False,
                    'detection_count': 0,
                    'evaluated': False
                }
                
                # Initialize defect status if in validation zone
                in_validation_zone, _ = self.is_in_validation_zone(center, frame_shape)
                if in_validation_zone and not self.tracked_breads[new_id]['evaluated']:
                    self.tracked_breads[new_id]['detection_count'] += 1
                    final_defect_status = self.update_defect_status(new_id, is_defective)
                    self.tracked_breads[new_id]['defect_status'] = final_defect_status
                    
                    # Check if we've reached 5 detections
                    if self.tracked_breads[new_id]['detection_count'] >= 5:
                        self.tracked_breads[new_id]['evaluated'] = True
        
        # Remove breads that have actually exited the frame
        to_remove = []
        for bread_id, bread_data in self.tracked_breads.items():
            center = bread_data['center']
            missing_frames = bread_data['missing_frames']
            
            # More aggressive removal for bread that's clearly off-frame (past right edge)
            h, w = frame_shape[:2]
            if center[0] > w + 50:  # Bread is 50px past right edge
                to_remove.append(bread_id)
            # Normal timeout removal for other areas
            elif missing_frames > self.max_missing_frames:
                to_remove.append(bread_id)
        
        for bread_id in to_remove:
            self.release_id(bread_id)  # Release ID for recycling
            del self.tracked_breads[bread_id]
        
        return self.tracked_breads, triggered_breads

# Simple validator class  
class SimpleValidator:
    def __init__(self):
        self.required_frames = 3
        self.window_size = 5
        self.detection_history = []  # Simple list of True/False for defects in zone
        self.last_trigger_time = 0
        self.deadzone_duration = 1.0
        
        # Zone parameters
        self.zone_width_percent = 40
        self.zone_height_percent = 60
        
    def add_detection_in_zone(self, is_defective):
        """Add detection to history only if in zone"""
        self.detection_history.append(is_defective)
        if len(self.detection_history) > self.window_size:
            self.detection_history.pop(0)
    
    def should_trigger(self):
        """Check if should trigger relay"""
        # Check deadzone
        if time.time() - self.last_trigger_time < self.deadzone_duration:
            return False
            
        # Check validation - count defective detections in zone
        defects_in_zone = sum(1 for d in self.detection_history if d)
        return defects_in_zone >= self.required_frames
    
    def trigger(self):
        """Record trigger"""
        self.last_trigger_time = time.time()
    
    def is_in_zone(self, contour, frame_shape):
        """Check if contour is in validation zone"""
        h, w = frame_shape[:2]
        zone_w = int(w * self.zone_width_percent / 100)
        zone_h = int(h * self.zone_height_percent / 100)
        
        x1 = (w - zone_w) // 2
        x2 = x1 + zone_w
        y1 = (h - zone_h) // 2 
        y2 = y1 + zone_h
        
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False, (0, 0), (x1, y1, x2, y2)
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        in_zone = x1 <= cx <= x2 and y1 <= cy <= y2
        return in_zone, (cx, cy), (x1, y1, x2, y2)

# Simple relay control
relay_queue = queue.Queue()
relay_active = False

def relay_worker():
    """Simple relay worker thread"""
    global relay_active
    while True:
        try:
            relay_number, duration = relay_queue.get(timeout=1)
            relay_active = True
            print(f'Activating relay {relay_number} for {duration}s')
            # on_relay(relay_number)  # Uncomment when ready
            time.sleep(duration)
            # off_relay(relay_number)  # Uncomment when ready
            #print(f'Relay {relay_number} deactivated')
            relay_active = False
            relay_queue.task_done()
        except queue.Empty:
            continue

def activate_relay(relay_number=1, duration=0.1):
    """Queue relay activation"""
    if not relay_active:
        relay_queue.put((relay_number, duration))

# Start relay thread
threading.Thread(target=relay_worker, daemon=True).start()

# Initialize components
validator = SimpleValidator()
tracker = BreadTracker()  # Add bread tracker
model_sam = FastSAM('peenike_leib_best.pt')
video_path = 'data/recording/recording_20251022_165450.avi'
cap = cv2.VideoCapture(video_path)

# Load template
template_img = cv2.imread("./output_frames/wide_template.jpg")
gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
_, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
template_contour = max(contours[0], key=cv2.contourArea)

# Create window and trackbars
cv2.namedWindow('Bread Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Bread Detection', 800, 600)

def update_threshold(x): pass
def update_required_frames(x): tracker.required_defect_frames = max(1, x)
def update_zone_width(x): tracker.validation_zone_width_percent = max(10, x)
def update_zone_height(x): tracker.validation_zone_height_percent = max(10, x)
def update_deadzone(x): tracker.deadzone_duration = x / 10.0
def update_exit_zone(x): tracker.exit_zone_width = max(50, x)

cv2.createTrackbar('Threshold', 'Bread Detection', 15, 100, update_threshold)
cv2.createTrackbar('Required Frames', 'Bread Detection', 3, 10, update_required_frames)
cv2.createTrackbar('Zone Width %', 'Bread Detection', 40, 100, update_zone_width)
cv2.createTrackbar('Zone Height %', 'Bread Detection', 60, 100, update_zone_height)
cv2.createTrackbar('Deadzone (0.1s)', 'Bread Detection', 10, 50, update_deadzone)
cv2.createTrackbar('Exit Zone Width', 'Bread Detection', 200, 400, update_exit_zone)

print("Starting bread anomaly detection...")
print("Press 'q' to quit")

while True:
    #try:
        # while camera.IsGrabbing():

        #     grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        #     if grabResult.GrabSucceeded():
        #         # Access the image data and convert to OpenCV format
        #         image = converter.Convert(grabResult)
        #         frame = image.GetArray()
        
        ret, frame = cap.read()
        if not ret:
            break
        

        # Get threshold
        threshold = cv2.getTrackbarPos('Threshold', 'Bread Detection') / 100.0

        # Run FastSAM
        results = model_sam.predict(frame, imgsz=640, verbose=False, conf=0.7)

        # Draw zones
        h, w = frame.shape[:2]
        
        # Validation zone
        zone_w = int(w * tracker.validation_zone_width_percent / 100)
        zone_h = int(h * tracker.validation_zone_height_percent / 100)
        val_x1 = (w - zone_w) // 2
        val_x2 = val_x1 + zone_w
        val_y1 = (h - zone_h) // 2
        val_y2 = val_y1 + zone_h
        cv2.rectangle(frame, (val_x1, val_y1), (val_x2, val_y2), (255, 255, 0), 2)
        cv2.putText(frame, "VALIDATION ZONE", (val_x1 + 10, val_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Exit zone (trigger zone) - positioned away from frame edge
        edge_margin = 50  # pixels from the right edge
        exit_x1 = w - tracker.exit_zone_width - edge_margin
        exit_x2 = w - edge_margin
        cv2.rectangle(frame, (exit_x1, 0), (exit_x2, h), (255, 0, 255), 2)
        cv2.putText(frame, "EXIT/TRIGGER ZONE", (exit_x1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Process detections
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # Get masks if available
            masks = None
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                
                # Get mask for this detection if available
                is_defective = False
                if masks is not None and i < len(masks):
                    # Get mask and convert to contour
                    mask = masks[i]
                    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    contours_detected = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if contours_detected:
                        largest_contour = max(contours_detected, key=cv2.contourArea)
                        
                        # Calculate defect score using template matching
                        match_score = cv2.matchShapes(template_contour, largest_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                        is_defective = match_score > threshold
                
                detections.append((x1, y1, x2, y2, is_defective))
        
        # Update tracker with detections
        tracked_breads, triggered_breads = tracker.update(detections, frame.shape)
        
        # Trigger relay for defective breads in exit zone
        for bread_id in triggered_breads:
            print(f'DEFECTIVE BREAD {bread_id} DETECTED IN EXIT ZONE - TRIGGERING RELAY!')
            activate_relay()
        
        # Draw tracking results
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for bread_id, bread_data in tracked_breads.items():
            x1, y1, x2, y2 = bread_data['box']
            center = bread_data['center']
            missing = bread_data['missing_frames']
            defective = bread_data['defect_status']
            
            # Get evaluation status
            evaluated = bread_data.get('evaluated', False)
            detection_count = bread_data.get('detection_count', 0)
            
            # Choose color based on evaluation status - FIXED: Color should not change after evaluation
            if missing > 0:
                color = (128, 128, 128)  # Gray if missing
            elif evaluated:
                # Final evaluation complete - red if defective, green if normal (NEVER changes after this)
                color = (0, 0, 255) if defective else (0, 255, 0)  # Red or Green - PERMANENT
            else:
                # Still being evaluated - show yellow
                color = (0, 255, 255)  # Yellow while being evaluated
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, color, -1)
            
            # Draw ID label
            if missing > 0:
                status = f"MISSING ({missing})"
            elif evaluated:
                status = "DEFECTIVE" if defective else "NORMAL"
            else:
                status = f"EVALUATING ({detection_count}/5)"
            
            label = f"Bread_{bread_id}: {status}"
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Status overlay
        active_breads = len([b for b in tracked_breads.values() if b['missing_frames'] == 0])
        defective_breads = len([b for b in tracked_breads.values() if b['defect_status'] and b['missing_frames'] == 0])
        deadzone_active = time.time() - tracker.last_trigger_time < tracker.deadzone_duration
        
        cv2.rectangle(frame, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.putText(frame, f"Threshold: {threshold:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Active Breads: {active_breads} | Defective: {defective_breads}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"IDs in Use: {len(tracker.used_ids)}/{tracker.max_id + 1} | Total Processed: {tracker.total_breads_processed}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Deadzone: {'ACTIVE' if deadzone_active else 'READY'}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0) if deadzone_active else (0, 255, 0), 1)
        cv2.putText(frame, f"Triggers: {len(triggered_breads)} this frame", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Next Available ID: {tracker.next_id if tracker.next_id <= tracker.max_id else 'Recycling'}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Bread Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #grabResult.Release()
    # finally:
    #     camera.StopGrabbing()
    #     cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")