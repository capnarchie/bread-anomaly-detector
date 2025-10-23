import cv2
import numpy as np
from ultralytics import FastSAM
import time
import math
from collections import deque

class SimpleBreadTracker:
    """Super simple tracker - just tracks center points of bread"""
    
    def __init__(self):
        self.next_id = 0
        self.tracked_breads = {}  # {id: {'center': (x,y), 'missing_frames': 0, 'box': (x1,y1,x2,y2)}}
        self.max_missing_frames = 10  # Remove bread after 10 missing frames
        self.max_distance = 150  # Max pixels bread can move between frames
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections = [(x1, y1, x2, y2), (x1, y1, x2, y2), ...] - list of bounding boxes
        """
        # Calculate centers of new detections
        new_centers = []
        for x1, y1, x2, y2 in detections:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            new_centers.append((center_x, center_y))
        
        # If no existing breads, just register all new ones
        if len(self.tracked_breads) == 0:
            for i, (center, box) in enumerate(zip(new_centers, detections)):
                self.tracked_breads[self.next_id] = {
                    'center': center,
                    'missing_frames': 0,
                    'box': box
                }
                self.next_id += 1
            return self.tracked_breads
        
        # Match new detections to existing breads
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
                self.tracked_breads[bread_id]['box'] = detections[best_match]
                self.tracked_breads[bread_id]['missing_frames'] = 0
                used_detections.add(best_match)
                updated_breads.add(bread_id)
            else:
                # Bread not found, increment missing counter
                self.tracked_breads[bread_id]['missing_frames'] += 1
        
        # Add new breads for unmatched detections
        for i, (center, box) in enumerate(zip(new_centers, detections)):
            if i not in used_detections:
                self.tracked_breads[self.next_id] = {
                    'center': center,
                    'missing_frames': 0,
                    'box': box
                }
                self.next_id += 1
        
        # Remove breads that have been missing too long
        to_remove = []
        for bread_id, bread_data in self.tracked_breads.items():
            if bread_data['missing_frames'] > self.max_missing_frames:
                to_remove.append(bread_id)
        
        for bread_id in to_remove:
            del self.tracked_breads[bread_id]
        
        return self.tracked_breads

def draw_tracking_results(frame, tracked_breads):
    """Draw bounding boxes and IDs for tracked breads"""
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for bread_id, bread_data in tracked_breads.items():
        x1, y1, x2, y2 = bread_data['box']
        center = bread_data['center']
        missing = bread_data['missing_frames']
        
        # Choose color (green if active, yellow if missing)
        color = (0, 255, 255) if missing > 0 else colors[bread_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(frame, center, 5, color, -1)
        
        # Draw ID label
        label = f"Bread_{bread_id}"
        if missing > 0:
            label += f" (Missing: {missing})"
        
        # Background for text
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    """Simple bread tracking with one FastSAM model"""
    
    # Load FastSAM model
    print("Loading FastSAM...")
    sam_model = FastSAM('peenike_leib_best.pt')
    
    # Initialize simple tracker
    tracker = SimpleBreadTracker()
    
    # Open video
    video_path = 'data/recording/recording_20251022_165450.avi'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    print(f"Opened video: {video_path}")
    print("Press 'q' to quit, 'space' to pause/resume")
    
    # Create window
    cv2.namedWindow('Simple Bread Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Simple Bread Tracking', 1200, 800)
    
    frame_count = 0
    fps_counter = deque(maxlen=30)
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            start_time = time.time()
            
            # Run FastSAM detection (no tracking mode - faster!)
            results = sam_model.predict(frame, imgsz=640, verbose=False, conf=0.8)
            
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
            
            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_counter.append(fps)
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            # Status overlay
            active_breads = len([b for b in tracked_breads.values() if b['missing_frames'] == 0])
            cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Frame: {frame_count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Active Breads: {active_breads}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Total Tracked: {len(tracked_breads)}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            frame_count += 1
        
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
    print(f"Processed {frame_count} frames")
    print(f"Total bread IDs assigned: {tracker.next_id}")

if __name__ == "__main__":
    main()