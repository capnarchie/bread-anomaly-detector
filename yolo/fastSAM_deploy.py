import cv2
import torch
import numpy as np
from ultralytics import FastSAM  # Import FastSAM from the ultralytics library

# Load the FastSAM-s model
model_path = "./FastSAM-s.pt"  # Path to the FastSAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FastSAM(model_path)

# Open the video feed
video_path = "data/leib2.avi"  # Path to the video file
cap = cv2.VideoCapture(video_path)

paused = False
show_segmentation = True
frame_count = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        frame_count += 1
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print(f"Warning: Frame {frame_count} is empty or invalid")
            continue
            
        # Check frame properties
        if frame_count == 1:
            print(f"Frame shape: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")
            print(f"Frame min/max values: {frame.min()}/{frame.max()}")
        
        if frame_count % 10 == 0:  # Print progress every 10 frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing frame {frame_count}/{total_frames}")
    else:
        # If paused, just wait for key input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = False
        continue

    # Keep a copy of the original frame
    original_frame = frame.copy()
    
    # Only run FastSAM if segmentation is enabled
    if show_segmentation:
        # Perform FastSAM prediction
        results = model(frame, device=device, retina_masks=True, imgsz=320, conf=0.9, iou=0.9, verbose=False)
        if results and len(results) > 0:
            # Save the first segment area into a variable
            first_segment_area = results[0].masks.data[0].sum().item() if hasattr(results[0], 'masks') and results[0].masks is not None else 0
            print(f"First segment area: {first_segment_area}")
        
        if results and hasattr(results[0], 'masks') and results[0].masks is not None:
            binary_mask = results[0].masks.data[0].cpu().numpy() > 0.5
            binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8) * 255, (frame.shape[1], frame.shape[0]))
            
            # Find contours from the binary mask
            contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on the frame
            annotated_frame = cv2.cvtColor(binary_mask_resized, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)
            
            # Calculate moments for the largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest_contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])  # Centroid x
                    cy = int(moments["m01"] / moments["m00"])  # Centroid y
                    print(f"Centroid of largest contour: ({cx}, {cy})")



        # Process results - FastSAM returns segmentation masks
        if results and len(results) > 0:
            result = results[0]  # Get first result
        
        # If there are masks, overlay them on the frame
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data  # Get mask data
            
            # Start with the original frame
            frame = original_frame.copy()
            
            colors = (50, 255, 125)
            
            # Only process first 10 masks to avoid performance issues
            max_masks = min(10, len(masks))
            
            for i in range(max_masks):
                try:
                    mask = masks[i]
                    # Convert mask to numpy array and resize to frame size
                    mask_np = mask.cpu().numpy()
                    if mask_np.max() > 0:  # Only process non-empty masks
                        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                        mask_bool = mask_resized > 0.5
                        
                        # Apply color to mask areas with transparency
                        # color = colors[i]
                        frame[mask_bool] = cv2.addWeighted(
                            frame[mask_bool], 0.7, 
                            np.full_like(frame[mask_bool], colors), 0.3, 0
                        )
                        
                        
                except Exception as e:
                    print(f"Error processing mask {i}: {e}")
                    continue
        else:
            # No masks found, use original frame
            print("No masks detected")
        
        # Draw bounding boxes if available
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # No results, use original frame
        print("No results from FastSAM")
        frame = original_frame

    # Add frame counter to display
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Press 'q':quit 'p':pause", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow("FastSAM Segmentation", annotated_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        if paused:
            print("Video paused. Press 'p' to resume or 'q' to quit.")
        else:
            print("Video resumed.")

# Release resources
print(f"\nProcessed {frame_count} frames")
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
