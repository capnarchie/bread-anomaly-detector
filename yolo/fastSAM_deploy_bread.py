import cv2
import torch
import numpy as np
from ultralytics import FastSAM
import os
import random

class FastSAMBreadDetector:
    def __init__(self, model_path="", device=None, template_path=None, verbose=True):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FastSAM(model_path)
        self.verbose = verbose
        self.average = []
        print(f"FastSAM model loaded on device: {self.device}")
        
        # Segmentation parameters
        self.conf_threshold = 0.9
        self.iou_threshold = 0.9
        self.img_size = 320
        
        # Template matching parameters
        self.template_path = template_path
        self.template_contours = None
        self.match_threshold = 0.1  # Threshold for matchShapes score to detect anomaly
        
        # ROI parameters
        self.roi = None  # (x, y, w, h) tuple
        self.roi_selected = False
        
        # Load and process template if provided
        if template_path and os.path.exists(template_path):
            self.load_template(template_path)
        
        # Colors for different segments
        self.colors = [
            (0, 255, 0),
            (0, 0, 255)

        ]
        
    def generate_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def select_roi(self, frame):

        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        
        if roi[2] > 0 and roi[3] > 0:  # Valid ROI selected
            self.roi = roi
            self.roi_selected = True
            print(f"ROI selected: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        else:
            self.roi = None
            self.roi_selected = False
            print("No ROI selected. Processing entire frame.")
    
    def get_roi_frame(self, frame):
        
        if self.roi_selected and self.roi is not None:
            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w]
            return roi_frame, (x, y)
        else:
            return frame, (0, 0)
    
    def adjust_coordinates_for_roi(self, coordinates, roi_offset):
        
        if not self.roi_selected:
            return coordinates
        
        offset_x, offset_y = roi_offset
        adjusted_coords = []
        
        for coord in coordinates:
            if isinstance(coord, tuple):  # Single coordinate
                adjusted_coords.append((coord[0] + offset_x, coord[1] + offset_y))
            elif isinstance(coord, np.ndarray):  # Contour
                adjusted_contour = coord.copy()
                adjusted_contour[:, :, 0] += offset_x
                adjusted_contour[:, :, 1] += offset_y
                adjusted_coords.append(adjusted_contour)
        
        return adjusted_coords
    
    def load_template(self, template_path):
        
        template_img = cv2.imread(template_path)
        
        if template_img is None:
            print(f"Error: Could not load template image {template_path}")
            return
        
        # Process template with FastSAM to get segmentation
        annotated_template, binary_template, template_info = self.process_frame_basic(template_img)
        
        if binary_template is not None and np.sum(binary_template) > 0:
            # Find contours in template
            contours, _ = cv2.findContours(binary_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use largest contour as template reference
                largest_contour = max(contours, key=cv2.contourArea)
                self.template_contours = [largest_contour]
                
                template_area = cv2.contourArea(largest_contour)
                

            else:
                print("Warning: No contours found in template")
        else:
            print("Warning: No segmentation found in template")
    
    def process_frame_basic(self, frame):
        
        if frame is None or frame.size == 0:
            return frame, None, {}
            
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Run FastSAM inference
        results = self.model(
            frame, 
            device=self.device, 
            retina_masks=True, 
            imgsz=self.img_size, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,
            verbose=True
        )
        
        # Initialize outputs
        annotated_frame = original_frame.copy()
        combined_binary_mask = np.zeros((h, w), dtype=np.uint8)
        segment_info = {
            'num_segments': 0,
            'total_area': 0,
            'segment_areas': [],
            'centroids': []
        }
        
        if not results or not hasattr(results[0], 'masks') or results[0].masks is None:
            return annotated_frame, combined_binary_mask, segment_info
        
        masks = results[0].masks.data
        segment_info['num_segments'] = len(masks)
        
        # Process masks to create combined binary mask
        for mask in masks:
            try:
                mask_np = mask.cpu().numpy()
                if mask_np.max() == 0:
                    continue
                    
                mask_resized = cv2.resize(mask_np, (w, h))
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                combined_binary_mask = cv2.bitwise_or(combined_binary_mask, binary_mask * 255)
                
            except Exception as e:
                print(f"Error processing mask: {e}")
                continue
        
        return annotated_frame, combined_binary_mask, segment_info
    


    def match_contour_to_template(self, contour, segment_id=0):
       
        if self.template_contours is None or len(self.template_contours) == 0:
            return float('inf'), True
        
        template_contour = self.template_contours[0]
        shape_match_score = cv2.matchShapes(contour, template_contour, cv2.CONTOURS_MATCH_I2, 0.0)
        
        # Determine if this is an anomaly based on matchShapes score
        is_anomaly = shape_match_score > self.match_threshold
        
        return shape_match_score, is_anomaly
    
    def process_frame(self, frame):
        
        if frame is None or frame.size == 0:
            return frame, None, {}
            
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Get ROI frame for processing
        roi_frame, roi_offset = self.get_roi_frame(frame)
        roi_h, roi_w = roi_frame.shape[:2]
        
        # Run FastSAM inference on ROI frame
        results = self.model(
            roi_frame, 
            device=self.device, 
            retina_masks=True, 
            imgsz=self.img_size, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,
            verbose=True
        )
        
        # Initialize outputs
        annotated_frame = original_frame.copy()
        combined_binary_mask = np.zeros((h, w), dtype=np.uint8)
        segment_info = {
            'num_segments': 0,
            'total_area': 0,
            'segment_areas': [],
            'centroids': [],
            'anomalies': [],
            'shape_matches': []
        }
        
        if not results or not hasattr(results[0], 'masks') or results[0].masks is None:
            return annotated_frame, combined_binary_mask, segment_info
        
        cv2.line(annotated_frame, (annotated_frame.shape[1] // 4, 0), (annotated_frame.shape[1] // 4, annotated_frame.shape[0]), (255, 255, 255), 1)
        cv2.line(annotated_frame, (annotated_frame.shape[1] - annotated_frame.shape[1] // 4, 0), (annotated_frame.shape[1] - annotated_frame.shape[1] // 4, annotated_frame.shape[0]), (255, 255, 255), 1)
                    
        masks = results[0].masks.data
        segment_info['num_segments'] = len(masks)
        
        # Process each mask
        for i, mask in enumerate(masks):
            try:
                # Convert mask to numpy array and resize to ROI frame size
                mask_np = mask.cpu().numpy()
                if mask_np.max() == 0:  # Skip empty masks
                    continue
                    
                # Resize mask to match ROI dimensions
                mask_resized = cv2.resize(mask_np, (roi_w, roi_h))
                roi_binary_mask = (mask_resized > 0.5).astype(np.uint8)
                
                # Create full-frame binary mask with ROI placed in correct position
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                if self.roi_selected and self.roi is not None:
                    x, y, roi_w_actual, roi_h_actual = self.roi
                    binary_mask[y:y+roi_h_actual, x:x+roi_w_actual] = roi_binary_mask
                else:
                    binary_mask = roi_binary_mask
                
                # Calculate mask area
                area = np.sum(binary_mask)
                segment_info['segment_areas'].append(area)
                segment_info['total_area'] += area
                
                # Add to combined binary mask
                combined_binary_mask = cv2.bitwise_or(combined_binary_mask, binary_mask * 255)
                
                # Find contours for analysis (use ROI mask for contour detection)
                roi_contours, _ = cv2.findContours(roi_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if roi_contours:
                    # Find largest contour in ROI space
                    largest_roi_contour = max(roi_contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_roi_contour)
                    
                    # Adjust contour coordinates to full frame space
                    largest_contour = largest_roi_contour.copy()
                    if self.roi_selected and self.roi is not None:
                        offset_x, offset_y = roi_offset
                        largest_contour[:, :, 0] += offset_x
                        largest_contour[:, :, 1] += offset_y
                    
                    moments = cv2.moments(largest_contour)
                    
                    # Template matching if template is loaded (use ROI contour for comparison)
                    shape_match_score, is_anomaly = 0.0, False
                    if self.template_contours is not None:
                        shape_match_score, is_anomaly = self.match_contour_to_template(largest_roi_contour, i)
                        
                        # self.average.append(shape_match_score)
                        # print("average similiarity across video so far /// ",np.mean(self.average))
                        segment_info['anomalies'].append(is_anomaly)
                        segment_info['shape_matches'].append(shape_match_score)
                    
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        segment_info['centroids'].append((cx, cy))
                        
                        # Choose color based on anomaly detection
                        centroid_color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # Red for anomaly, Green for normal
                        text_color = (125, 125, 0) if is_anomaly else (125, 125, 0)
                        if cx <= annotated_frame.shape[1] - annotated_frame.shape[1] // 4 and cx >= annotated_frame.shape[1] // 4:
                            # Draw centroid on annotated frame
                            cv2.circle(annotated_frame, (cx, cy), 8, centroid_color, -1)
                            cv2.circle(annotated_frame, (cx, cy), 12, centroid_color, 2)
                            
                            # Label with segment info and anomaly status
                            label = f"ID{i+1} "
                            if self.template_contours is not None:
                                label += f"{'NG' if is_anomaly else 'OK'}"
                                # label += f" Match:{shape_match_score:.3f}"
                            area = cv2.contourArea(largest_contour)
                            cv2.putText(annotated_frame, label, (cx+15, cy-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
                            
                            # Additional info below centroid
                            if self.template_contours is not None:
                                cv2.putText(annotated_frame, f"Match:{shape_match_score:.3f}, Area:{area:.1f}", (cx+15, cy+20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                            # Apply colored overlay to segment (different color for anomalies)
                            if is_anomaly:
                                color = (0, 0, 255)  # Red for anomalies
                                alpha = 0  # More prominent for anomalies
                                outline_thickness = 3
                            else:
                                color = (0, 255, 0)  # Green for normal segments
                                alpha = 0
                                outline_thickness = 2
                            
                            # Only draw on areas within the binary mask
                            mask_colored = np.zeros_like(annotated_frame)
                            mask_colored[binary_mask == 1] = color
                            
                            # Blend with original frame
                            annotated_frame = cv2.addWeighted(annotated_frame, 1-alpha, mask_colored, alpha, 0)
                            
                            # Draw contour outline with different thickness for anomalies
                            if roi_contours:
                                # Draw the adjusted contour on the full frame
                                full_frame_contours = [largest_contour]
                                cv2.drawContours(annotated_frame, full_frame_contours, -1, color, outline_thickness)
            except Exception as e:
                print(f"Error processing mask {i}: {e}")
                continue
    
        
        return annotated_frame, combined_binary_mask, segment_info
    
    def add_info_overlay(self, frame, segment_info, frame_count=None):
        """Add information overlay to the frame"""
        # Larger background for text to accommodate template and ROI info
        overlay = frame.copy()
        bg_height = 180 if self.template_contours is not None else 140
        cv2.rectangle(overlay, (10, 10), (480, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        y_offset = 30
        if frame_count is not None:
            cv2.putText(frame, f"Frame: {frame_count}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        return frame

def threshold_callback(val):
    pass

USE_ROI = True
def process_video(video_path, model_path="FastSAM-s.pt", output_dir="output_frames", template_path=None, verbose=True):

    # Initialize detector
    detector = FastSAMBreadDetector(model_path, template_path=template_path, verbose=verbose)
    cv2.imshow("Template", cv2.resize(cv2.imread(template_path), (320, 240)))
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames at {fps} FPS")
    
    # Create window and trackbar for threshold adjustment
    main_window = "FastSAM - Bread Segmentation"
    cv2.namedWindow(main_window, cv2.WINDOW_AUTOSIZE)
    
    initial_threshold = int(detector.match_threshold * 1000)
    cv2.createTrackbar("Match Threshold x1000", main_window, initial_threshold, 500, threshold_callback)

    
    frame_count = 0
    paused = False
    save_frames = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            frame_count += 1
            
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processing frame {frame_count}/{total_frames}")
        
        # Update threshold from trackbar (convert back from x1000)
        threshold_val = cv2.getTrackbarPos("Match Threshold x1000", main_window)
        new_threshold = threshold_val / 1000.0
        
        # Show threshold change in console (only when changed)
        if abs(detector.match_threshold - new_threshold) > 0.001:
            print(f"Threshold adjusted to: {new_threshold:.3f}")
        
        detector.match_threshold = new_threshold
        # Process frame with FastSAM
        annotated_frame, binary_mask, segment_info = detector.process_frame(frame)


        # Add information overlay
        display_frame = detector.add_info_overlay(annotated_frame.copy(), segment_info, frame_count)
        
        # Create binary mask display (3-channel for display)
        binary_display = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
        # Add text to binary mask display
        cv2.putText(binary_display, "Binary Mask", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(binary_display, f"Frame: {frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Display windows
        cv2.imshow(main_window, display_frame)#cv2.resize(display_frame, (640, 480)))
        cv2.imshow("Binary Mask", cv2.resize(binary_display, (640, 480)))
        
        # Save frames if requested
        if save_frames:
            cv2.imwrite(os.path.join(output_dir, f"annotated_frame_{frame_count:06d}.jpg"), display_frame)
            cv2.imwrite(os.path.join(output_dir, f"binary_mask_{frame_count:06d}.jpg"), binary_mask)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            status = "paused" if paused else "resumed"
            print(f"Video {status}")
        elif key == ord('s'):
            save_frames = not save_frames
            status = "enabled" if save_frames else "disabled"
            print(f"Frame saving {status}")
        elif key == ord('r'):
            # ROI selection
            paused = True
            print("Paused for ROI selection...")
            detector.select_roi(frame)
            paused = False
    
    # Cleanup
    print(f"\nProcessed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()

def main():
    pass

if __name__ == "__main__":

    # For video processing with template
    video_path = "data/recording_20250919_160901.avi"  # 
    template_path = "./output_frames/wide_template.jpg"  # Use as good bread template
    
    if os.path.exists(video_path):
        print("Processing video with FastSAM and template matching...")
        process_video(video_path, "FastSAM-s.pt", template_path=template_path, verbose=False)