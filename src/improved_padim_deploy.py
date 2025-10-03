# improved_padim_deploy.py
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
import sys
from pathlib import Path

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_PATH = str(PROJECT_ROOT / "data" / "recording_20250919_160901.avi")
MODEL_PATH = str(PROJECT_ROOT / "models" / "padim_bread_improved.pth")

# -------------------
# Device setup
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------
# Improved PaDiM Model (same as training)
# -------------------
class ImprovedPaDiM:
    def __init__(self, backbone_name='wide_resnet50_2'):
        if backbone_name == 'wide_resnet50_2':
            self.backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT).to(device)
            self.layers = ["layer1", "layer2", "layer3"]
            self.feature_dims = [512, 1024, 2048]
        else:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
            self.layers = ["layer1", "layer2", "layer3"]
            self.feature_dims = [128, 256, 512]
        
        self.backbone.eval()
        self.outputs = {}
        self.hooks = []
        
        # Register hooks
        for name, module in self.backbone._modules.items():
            if name in self.layers:
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: self._hook_fn(name, o)
                )
                self.hooks.append(hook)
        
        self.patch_size = 3
        self.mean_features = None
        self.cov_inv = None
    
    def _hook_fn(self, name, output):
        self.outputs[name] = output
    
    def _extract_patch_features(self, features, target_patches=64):
        """Extract patch-level features maintaining spatial information"""
        B, C, H, W = features.shape
        
        # Adaptive patch size based on feature map size
        if H < self.patch_size or W < self.patch_size:
            # If feature map is too small, use global average pooling
            global_feat = torch.mean(features, dim=(2, 3))  # [B, C]
            return global_feat.unsqueeze(1)  # [B, 1, C]
        
        # Calculate stride to get approximately target_patches
        stride = max(1, min(H, W) // int(np.sqrt(target_patches)))
        stride = min(stride, self.patch_size)
        
        # Use unfold to extract patches with calculated stride
        patches = features.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        
        # Average pool each patch
        patch_features = patches.mean(dim=(3, 4))  # Shape: [B, C, num_patches]
        
        # If we have more patches than target, randomly sample
        num_patches = patch_features.shape[2]
        if num_patches > target_patches:
            indices = torch.randperm(num_patches)[:target_patches]
            patch_features = patch_features[:, :, indices]
        elif num_patches < target_patches:
            # If fewer patches, repeat some randomly
            needed = target_patches - num_patches
            if num_patches > 0:
                repeat_indices = torch.randint(0, num_patches, (needed,))
                extra_patches = patch_features[:, :, repeat_indices]
                patch_features = torch.cat([patch_features, extra_patches], dim=2)
        
        return patch_features.permute(0, 2, 1)  # Shape: [B, target_patches, C]
    
    def extract_features(self, x):
        """Extract multi-scale features"""
        with torch.no_grad():
            _ = self.backbone(x)
            
            all_features = []
            target_patches = 64  # Fixed number of patches per layer
            
            for i, layer in enumerate(self.layers):
                features = self.outputs[layer]
                
                # Extract patch features with consistent patch count
                patch_features = self._extract_patch_features(features, target_patches)
                
                all_features.append(patch_features)
            
            # Concatenate and pool
            concatenated_features = torch.cat(all_features, dim=2)
            global_features = concatenated_features.mean(dim=1)
            
            return global_features.cpu().numpy()
    
    def predict(self, x):
        """Compute anomaly scores"""
        features = self.extract_features(x)
        
        scores = []
        for feat in features:
            diff = feat - self.mean_features
            score = np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff.T))
            scores.append(score)
        
        return np.array(scores)
    
    def load(self, path):
        """Load model parameters"""
        data = torch.load(path, map_location=device, weights_only=False)
        self.mean_features = data['mean_features']
        self.cov_inv = data['cov_inv']
        print(f"Model loaded from {path}")

# -------------------
# Load model
# -------------------
try:
    model = ImprovedPaDiM('wide_resnet50_2')
    model.load(MODEL_PATH)
    print("✅ Improved PaDiM model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("Please run 'python improved_train_padim.py' first to train the improved model")
    exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# -------------------
# Transform
# -------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------
# Anomaly detection function
# -------------------
def detect_anomaly(img, threshold=24.98):
    """Detect anomaly in bread image"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    score = model.predict(img_tensor)[0]
    
    is_defective = score > threshold
    confidence = min(score / threshold, 2.0)  # Cap at 2.0 for visualization
    
    return score, is_defective, confidence

# -------------------
# Enhanced visualization
# -------------------
def draw_detection_result(frame, bbox, score, is_defective, confidence):
    """Draw detection results with enhanced visualization"""
    x, y, w, h = bbox
    
    # Color coding
    if is_defective:
        color = (0, 0, 255)  # Red for defective
        label = "DEFECT"
        status_color = (0, 0, 255)
    else:
        color = (0, 255, 0)  # Green for normal
        label = "NORMAL"
        status_color = (0, 255, 0)
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
    
    # Draw label background
    label_text = f"{label} ({score:.1f})"
    (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y-30), (x + label_w + 10, y), status_color, -1)
    
    # Draw label text
    cv2.putText(frame, label_text, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw confidence bar
    bar_width = w
    bar_height = 8
    bar_x = x
    bar_y = y + h + 10
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
    
    # Confidence bar
    conf_width = int(bar_width * min(confidence, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), color, -1)

# -------------------
# Bread tracking class for center-based detection
# -------------------
class BreadTracker:
    def __init__(self, center_x_threshold=50):
        self.tracked_breads = {}  # id -> {bbox, analyzed, result, first_seen_frame}
        self.next_id = 0
        self.center_x_threshold = center_x_threshold
        self.analyzed_results = []  # Store all analysis results
    
    def update_tracks(self, contours, frame_center_x, frame_count):
        """Update bread tracking and return analysis results"""
        current_bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            current_bboxes.append((x, y, w, h, center_x))
        
        # Match current detections to existing tracks
        new_tracked = {}
        used_current = set()
        
        # Try to match existing tracks
        for bread_id, bread_data in self.tracked_breads.items():
            best_match = None
            best_distance = float('inf')
            
            for i, (x, y, w, h, center_x) in enumerate(current_bboxes):
                if i in used_current:
                    continue
                
                # Calculate distance from previous position
                prev_x, prev_y, prev_w, prev_h = bread_data['bbox']
                prev_center_x = prev_x + prev_w // 2
                distance = abs(center_x - prev_center_x) + abs(y - prev_y)
                
                # Match if close enough (bread shouldn't move too far between frames)
                if distance < 100 and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                x, y, w, h, center_x = current_bboxes[best_match]
                used_current.add(best_match)
                
                # Update existing track
                bread_data['bbox'] = (x, y, w, h)
                new_tracked[bread_id] = bread_data
        
        # Add new tracks for unmatched detections
        for i, (x, y, w, h, center_x) in enumerate(current_bboxes):
            if i not in used_current:
                new_tracked[self.next_id] = {
                    'bbox': (x, y, w, h),
                    'analyzed': False,
                    'result': None,
                    'first_seen_frame': frame_count
                }
                self.next_id += 1
        
        self.tracked_breads = new_tracked
        
        # Check which breads are at center and need analysis
        breads_to_analyze = []
        for bread_id, bread_data in self.tracked_breads.items():
            x, y, w, h = bread_data['bbox']
            center_x = x + w // 2
            
            # Check if bread is at frame center and not yet analyzed
            if (abs(center_x - frame_center_x) < self.center_x_threshold and 
                not bread_data['analyzed']):
                breads_to_analyze.append((bread_id, bread_data))
        
        return breads_to_analyze
    
    def mark_analyzed(self, bread_id, score, is_defective, confidence):
        """Mark bread as analyzed with results"""
        if bread_id in self.tracked_breads:
            self.tracked_breads[bread_id]['analyzed'] = True
            self.tracked_breads[bread_id]['result'] = {
                'score': score,
                'is_defective': is_defective,
                'confidence': confidence
            }
            self.analyzed_results.append({
                'id': bread_id,
                'score': score,
                'is_defective': is_defective,
                'confidence': confidence
            })
    
    def get_display_results(self):
        """Get all breads with their analysis results for display"""
        display_breads = []
        for bread_id, bread_data in self.tracked_breads.items():
            x, y, w, h = bread_data['bbox']
            if bread_data['analyzed'] and bread_data['result']:
                result = bread_data['result']
                display_breads.append({
                    'bbox': (x, y, w, h),
                    'id': bread_id,
                    'score': result['score'],
                    'is_defective': result['is_defective'],
                    'confidence': result['confidence']
                })
            else:
                # Unanalyzed bread - show as pending
                display_breads.append({
                    'bbox': (x, y, w, h),
                    'id': bread_id,
                    'score': None,
                    'is_defective': None,
                    'confidence': None
                })
        return display_breads

# -------------------
# Enhanced drawing function for tracked breads
# -------------------
def draw_tracked_bread(frame, bread_info, frame_center_x, center_threshold):
    """Draw tracked bread with analysis results"""
    bbox = bread_info['bbox']
    x, y, w, h = bbox
    bread_id = bread_info['id']
    center_x = x + w // 2
    
    # Determine if bread is in center zone
    is_in_center = abs(center_x - frame_center_x) < center_threshold
    
    if bread_info['score'] is not None:
        # Analyzed bread - show results
        score = bread_info['score']
        is_defective = bread_info['is_defective']
        confidence = bread_info['confidence']
        
        color = (0, 0, 255) if is_defective else (0, 255, 0)
        label = "DEFECT" if is_defective else "NORMAL"
        status = "✓"
        
        # Draw thick border for analyzed breads
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
        
        # Label with ID and result
        label_text = f"#{bread_id} {label} ({score:.1f}) {status}"
        
    else:
        # Unanalyzed bread
        if is_in_center:
            color = (0, 255, 255)  # Yellow - ready for analysis
            label_text = f"#{bread_id} ANALYZING..."
        else:
            color = (128, 128, 128)  # Gray - waiting
            label_text = f"#{bread_id} WAITING..."
        
        # Draw thin border for unanalyzed breads
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw label background
    (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y-25), (x + label_w + 10, y), color, -1)
    
    # Draw label text
    cv2.putText(frame, label_text, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# -------------------
# Video processing with center-based detection
# -------------------
def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_PATH}")
        return
    
    print("Processing video... Press 'q' to quit, 'space' to pause")
    
    # Adaptive threshold (you can adjust this based on your evaluation results)
    threshold = 24.98  # Start with this, adjust based on performance
    
    frame_count = 0
    total_analyzed = 0
    defect_count = 0
    
    # Initialize bread tracker
    tracker = BreadTracker(center_x_threshold=50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ROI extraction
        roi_top, roi_bottom, roi_left, roi_right = 142, 699, 0, frame.shape[1]
        roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]
        display_frame = roi_frame.copy()
        
        # Calculate frame center (moved 100px to the left)
        frame_center_x = (display_frame.shape[1] // 2) - 200
        
        # Draw center line for reference
        cv2.line(display_frame, (frame_center_x, 0), (frame_center_x, display_frame.shape[0]), 
                 (255, 255, 0), 2)
        cv2.putText(display_frame, "Analysis line", (frame_center_x - 30, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Bread detection using contours
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh_img = cv2.threshold(blurred, 39, 255, cv2.THRESH_BINARY_INV)
        thresh_img = cv2.bitwise_not(thresh_img)
        
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 20_000]
        
        # Update tracking and get breads to analyze
        breads_to_analyze = tracker.update_tracks(filtered_contours, frame_center_x, frame_count)
        
        # Analyze breads that just reached the center
        frame_new_analyses = 0
        for bread_id, bread_data in breads_to_analyze:
            x, y, w, h = bread_data['bbox']
            
            # Extract and resize bread crop
            bread_crop = roi_frame[y:y+h, x:x+w]
            if bread_crop.size == 0:
                continue
                
            bread_crop_resized = cv2.resize(bread_crop, (256, 256))
            
            # Detect anomaly
            try:
                score, is_defective, confidence = detect_anomaly(bread_crop_resized, threshold)
                
                # Store result in tracker
                tracker.mark_analyzed(bread_id, score, is_defective, confidence)
                
                total_analyzed += 1
                frame_new_analyses += 1
                
                if is_defective:
                    defect_count += 1
                    
                print(f"Frame {frame_count}: Analyzed Bread #{bread_id} - {'DEFECT' if is_defective else 'NORMAL'} (Score: {score:.2f})")
                
            except Exception as e:
                print(f"Error analyzing bread #{bread_id}: {e}")
                continue
        
        # Draw all tracked breads
        display_breads = tracker.get_display_results()
        for bread_info in display_breads:
            draw_tracked_bread(display_frame, bread_info, frame_center_x, tracker.center_x_threshold)
        
        # Add frame statistics
        current_breads = len(display_breads)
        analyzed_breads = sum(1 for b in display_breads if b['score'] is not None)
        stats_text = f"Frame: {frame_count} | Tracked: {current_breads} | New Analyses: {frame_new_analyses}"
        cv2.putText(display_frame, stats_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add overall statistics
        if total_analyzed > 0:
            defect_rate = (defect_count / total_analyzed) * 100
            overall_stats = f"Total Analyzed: {total_analyzed} | Defects: {defect_count} | Rate: {defect_rate:.1f}%"
            cv2.putText(display_frame, overall_stats, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add tracking info
        tracking_info = f"Center Detection Zone: ±{tracker.center_x_threshold}px"
        cv2.putText(display_frame, tracking_info, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add threshold info
        threshold_text = f"Threshold: {threshold:.1f} (Press +/- to adjust)"
        cv2.putText(display_frame, threshold_text, (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display
        cv2.imshow("Improved Bread Defect Detection", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to pause
            cv2.waitKey(0)
        elif key == ord('+') or key == ord('='):  # Increase threshold
            threshold += 1.0
            print(f"Threshold increased to: {threshold:.1f}")
        elif key == ord('-'):  # Decrease threshold
            threshold = max(1.0, threshold - 1.0)
            print(f"Threshold decreased to: {threshold:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\n--- Final Statistics ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Total breads analyzed: {total_analyzed}")
    print(f"Total defects found: {defect_count}")
    if total_analyzed > 0:
        print(f"Overall defect rate: {(defect_count/total_analyzed)*100:.2f}%")
    print(f"Final threshold used: {threshold:.1f}")
    print(f"Unique breads tracked: {tracker.next_id}")
    print(f"Center detection zone: ±{tracker.center_x_threshold}px")

# -------------------
# Main execution
# -------------------
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        exit(1)
    
    print("Starting CENTER-BASED bread defect detection...")
    print("🎯 How it works:")
    print("  - Bread loaves are tracked as they move across the frame")
    print("  - Analysis happens ONCE when bread reaches the center line")
    print("  - Results are locked and displayed until bread leaves the frame")
    print("  - Yellow line shows the center detection zone")
    print("")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'space' to pause/resume")
    print("  - Press '+' to increase threshold (more sensitive)")
    print("  - Press '-' to decrease threshold (less sensitive)")
    print("")
    print("Legend:")
    print("  🔴 Red border = DEFECTIVE (analyzed)")
    print("  🟢 Green border = NORMAL (analyzed)")  
    print("  🟡 Yellow border = ANALYZING (at center)")
    print("  ⚪ Gray border = WAITING (approaching center)")
    print()
    
    process_video()