"""
Simplified Main Tracker
Clean main loop using modularized components
"""
import cv2
import numpy as np
import torch
from ultralytics import FastSAM
import time
from collections import deque

# Import custom modules
import centroid_tracker as ct
import validator as v
from relay_controller import RelayController
from defect_analyzer import DefectAnalyzer
from zone_manager import ZoneManager
from ui_renderer import UIRenderer


def load_template(template_path="./output_frames/wide_template.jpg"):
    """Load and process template image for shape matching"""
    template_img = cv2.imread(template_path)
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    _, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contour = max(contours[0], key=cv2.contourArea)
    return template_contour


def extract_detections(results, is_crack=False):
    """
    Extract bounding boxes and masks from model results
    
    Args:
        results: YOLO/FastSAM prediction results
        is_crack: Boolean indicating if these are crack detections
    
    Returns:
        tuple: (detections, masks) where detections is list of boxes and masks is list of masks
    """
    detections = []
    masks = []
    
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        mask_data = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            detections.append((x1, y1, x2, y2))
            masks.append(mask_data[idx] if idx < len(mask_data) else None)
    
    return detections, masks


def process_frame(frame, sam_model, sam_model_cracks, tracker, defect_analyzer, 
                  zone_manager, validator, relay_controller, ui_renderer,
                  threshold, crack_threshold):
    """
    Process a single frame - main detection logic
    
    Args:
        frame: Input video frame
        sam_model: Bread detection model
        sam_model_cracks: Crack detection model
        tracker: Bread tracker instance
        defect_analyzer: Defect analyzer instance
        zone_manager: Zone manager instance
        validator: Validator instance
        relay_controller: Relay controller instance
        ui_renderer: UI renderer instance
        threshold: Shape matching threshold
        crack_threshold: Crack percentage threshold
    
    Returns:
        None (modifies frame in-place)
    """
    # Run detection models
    bread_results = sam_model.predict(frame, imgsz=640, verbose=False, conf=0.8, retina_masks=True)
    crack_results = sam_model_cracks.predict(frame, imgsz=640, verbose=False, conf=0.3, retina_masks=True)
    
    # Extract detections
    bread_detections, bread_masks = extract_detections(bread_results)
    crack_detections, crack_masks = extract_detections(crack_results, is_crack=True)
    
    # Update tracker
    tracked_breads = tracker.update(bread_detections)
    
    # Analyze each tracked bread for defects
    for bread_id, bread_data in tracked_breads.items():
        bread_box = bread_data['box']
        
        # Find corresponding bread mask
        bread_mask = None
        for idx, det_box in enumerate(bread_detections):
            if det_box == bread_box and idx < len(bread_masks):
                bread_mask = bread_masks[idx]
                break
        
        # Analyze for defects
        analysis = defect_analyzer.analyze_bread(
            bread_box, bread_mask, crack_detections, crack_masks,
            shape_threshold=threshold, crack_threshold=crack_threshold
        )
        
        # Update bread data with analysis results
        bread_data.update(analysis)
    
    # Draw cracks and breads
    ui_renderer.draw_cracks(frame, crack_detections)
    ui_renderer.draw_bread_tracking(frame, tracked_breads)
    
    # Process validation and exit zones
    for bread_id, bread_data in tracked_breads.items():
        center = bread_data['center']
        is_defective = bread_data.get('is_defective', False)
        
        # Check zones
        in_validation_zone = zone_manager.is_in_validation_zone(center)
        in_exit_zone = zone_manager.is_in_exit_zone(center)
        
        # Validation zone logic
        if in_validation_zone:
            validator.add_detection(is_defective, bread_id)
            is_valid = validator.is_valid(bread_id)
            ui_renderer.draw_validation_info(frame, bread_data, is_valid)
        
        # Exit zone logic
        if in_exit_zone and validator.is_valid(bread_id):
            validator.add_exit_detection(bread_id)
            can_exit = validator.can_exit_detection(bread_id)
            if can_exit:
                relay_controller.activate(relay_number=1, duration=0.1, bread_id=bread_id)
            ui_renderer.draw_reject_label(frame, bread_data)
    
    return tracked_breads, crack_detections


def main():
    """Main execution loop"""
    
    # Load template
    print("Loading template...")
    template_contour = load_template()
    
    # Initialize models
    print("Loading models...")
    sam_model = FastSAM('peenike_leib_best.pt')
    sam_model.to('cuda')
    sam_model_cracks = FastSAM('runs/train/fastsam11/weights/best.pt')
    sam_model_cracks.to('cuda')
    print(f"Models loaded to: {torch.cuda.get_device_name(0)}")
    
    # Initialize components
    tracker = ct.SimpleBreadTracker()
    validator = v.SimpleValidator()
    relay_controller = RelayController()
    defect_analyzer = DefectAnalyzer(template_contour)
    ui_renderer = UIRenderer()
    
    # Open video
    video_path = "data/leib3.avi"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get frame dimensions and initialize zone manager
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    height, width, _ = first_frame.shape
    zone_manager = ZoneManager(width, height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    # Create UI
    ui_renderer.create_window('Simple Bread Tracking', 800, 600)
    ui_renderer.create_trackbar('Threshold', 'Simple Bread Tracking', 15, 100, lambda x: None)
    ui_renderer.create_trackbar('Crack%', 'Simple Bread Tracking', 5, 50, lambda x: None)
    
    # FPS counter
    fps_counter = deque(maxlen=30)
    paused = False
    
    print("Starting detection loop...")
    print("Press 'q' to quit, SPACE to pause/resume")
    
    # Main loop
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            # Get thresholds from trackbars
            threshold = ui_renderer.get_trackbar_value('Threshold', 'Simple Bread Tracking') / 100.0
            crack_threshold = ui_renderer.get_trackbar_value('Crack%', 'Simple Bread Tracking')
            
            # Process frame
            start_time = time.time()
            tracked_breads, crack_detections = process_frame(
                frame, sam_model, sam_model_cracks, tracker, defect_analyzer,
                zone_manager, validator, relay_controller, ui_renderer,
                threshold, crack_threshold
            )
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            fps_counter.append(fps)
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            # Gather statistics
            stats = {
                'fps': avg_fps,
                'total_count': len(tracked_breads),
                'defective_count': sum(1 for b in tracked_breads.values() if b.get('is_defective', False)),
                'shape_defective': sum(1 for b in tracked_breads.values() if b.get('shape_defective', False)),
                'crack_defective': sum(1 for b in tracked_breads.values() if b.get('crack_defective', False)),
                'shape_threshold': threshold,
                'crack_threshold': crack_threshold,
                'total_cracks': len(crack_detections)
            }
            
            # Draw UI elements
            ui_renderer.draw_info_panel(frame, stats)
            ui_renderer.draw_zones(frame, zone_manager.get_validation_zone(), zone_manager.get_exit_zone())
        
        # Display frame
        ui_renderer.show_frame(frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
