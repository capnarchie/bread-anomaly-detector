import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import threading
from collections import deque
import queue

VIDEO_PATH = "recording_20250919_160901.avi"
MODEL_PATH = "bread_autoencoder2.h5"
IMAGE_SIZE = 128

def do_nothing(x):
    pass

class HighPerformanceAnomalyDetector:
    def __init__(self):
        self.setup_gpu()
        self.autoencoder = self.load_and_warmup_model()
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing = True
        
    def setup_gpu(self):
        """Configure GPU for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Enable mixed precision for faster inference
                tf.config.optimizer.set_jit(True)  # Enable XLA
                print(f"GPU configuration successful: {len(gpus)} GPU(s) found")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    
    def load_and_warmup_model(self):
        """Load model and warm it up"""
        autoencoder = load_model(MODEL_PATH, compile=False)
        # Warm up with batch prediction
        dummy_input = np.random.random((4, IMAGE_SIZE, IMAGE_SIZE, 1)).astype(np.float32)
        _ = autoencoder.predict(dummy_input, verbose=0)
        print("Model warmed up for GPU with batch processing")
        return autoencoder
    
    def inference_worker(self):
        """Background thread for running inference"""
        batch_size = 8  # Process multiple bread regions at once
        batch_data = []
        batch_metadata = []
        
        while self.processing:
            try:
                # Collect batch
                while len(batch_data) < batch_size and self.processing:
                    try:
                        frame_data = self.frame_queue.get(timeout=0.1)
                        if frame_data is None:  # Shutdown signal
                            break
                        batch_data.extend(frame_data['batch_input'])
                        batch_metadata.extend(frame_data['metadata'])
                        self.frame_queue.task_done()
                    except queue.Empty:
                        break
                
                if batch_data:
                    # Run batched inference
                    batch_input = np.array(batch_data[:batch_size])
                    metadata_batch = batch_metadata[:batch_size]
                    
                    # Clear processed items
                    batch_data = batch_data[batch_size:]
                    batch_metadata = batch_metadata[batch_size:]
                    
                    # GPU inference
                    batch_recon = self.autoencoder.predict(batch_input, verbose=0)
                    batch_errors = np.mean((batch_input - batch_recon) ** 2, axis=(1, 2, 3))
                    
                    # Send results back
                    results = []
                    for i, metadata in enumerate(metadata_batch):
                        metadata['error'] = batch_errors[i]
                        results.append(metadata)
                    
                    try:
                        self.result_queue.put(results, timeout=0.1)
                    except queue.Full:
                        pass  # Skip if queue is full
                        
            except Exception as e:
                print(f"Inference worker error: {e}")
                break
    
    def process_frame(self, frame, blur_kernel_size, threshold_val, min_area, roi_top, roi_bottom, roi_left, roi_right):
        """Process a single frame and extract bread regions"""
        if roi_top >= roi_bottom or roi_left >= roi_right:
            return []
        
        roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
        
        _, thresh_img = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
        thresh_img = cv2.bitwise_not(thresh_img)
        
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        
        if not filtered_contours:
            return []
        
        # Prepare batch data for inference
        batch_input = []
        metadata = []
        
        c = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:5]
        for cnt in c:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            bread_roi = gray[y:y+h, x:x+w]
            if bread_roi.size == 0:
                continue
            
            bread_resized = cv2.resize(bread_roi, (IMAGE_SIZE, IMAGE_SIZE))
            bread_resized = bread_resized.astype("float32") / 255.0
            bread_resized = np.expand_dims(bread_resized, axis=-1)
            
            batch_input.append(bread_resized)
            metadata.append({
                'contour': cnt,
                'area': area,
                'bbox': (x, y, w, h),
                'frame_id': id(frame)  # Unique frame identifier
            })
        
        return {'batch_input': batch_input, 'metadata': metadata}
    
    def run(self):
        """Main processing loop"""
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {VIDEO_PATH}")
            return
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame of the video.")
            return
        
        FRAME_HEIGHT, FRAME_WIDTH, _ = frame.shape
        
        # Setup GUI
        cv2.namedWindow('Controls')
        cv2.resizeWindow('Controls', 500, 500)
        cv2.createTrackbar('Blur Kernel', 'Controls', 3, 15, do_nothing)
        cv2.createTrackbar('Threshold', 'Controls', 39, 255, do_nothing)
        cv2.createTrackbar('Defect Threshold', 'Controls', 300_000, 500_000, do_nothing)
        cv2.createTrackbar('Min Area', 'Controls', 1000, 30000, do_nothing)
        cv2.createTrackbar('ROI Top', 'Controls', 142, FRAME_HEIGHT, do_nothing)
        cv2.createTrackbar('ROI Bottom', 'Controls', 699, FRAME_HEIGHT, do_nothing)
        cv2.createTrackbar('ROI Left', 'Controls', 0, FRAME_WIDTH, do_nothing)
        cv2.createTrackbar('ROI Right', 'Controls', FRAME_WIDTH, FRAME_WIDTH, do_nothing)
        
        print("High-performance video processing started. Press 'q' to quit.")
        
        # Start inference worker thread
        inference_thread = threading.Thread(target=self.inference_worker)
        inference_thread.start()
        
        # FPS monitoring
        fps_counter = 0
        fps_start_time = time.time()
        last_fps_display = time.time()
        
        # Results cache
        results_cache = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or cannot read the frame.")
                    break
                
                # Get slider values
                blur_slider_val = cv2.getTrackbarPos('Blur Kernel', 'Controls')
                threshold_val = cv2.getTrackbarPos('Threshold', 'Controls')
                baked_together_area_thresh = cv2.getTrackbarPos('Defect Threshold', 'Controls')
                min_area = cv2.getTrackbarPos('Min Area', 'Controls')
                roi_top = cv2.getTrackbarPos('ROI Top', 'Controls')
                roi_bottom = cv2.getTrackbarPos('ROI Bottom', 'Controls')
                roi_left = cv2.getTrackbarPos('ROI Left', 'Controls')
                roi_right = cv2.getTrackbarPos('ROI Right', 'Controls')
                
                blur_kernel_size = (blur_slider_val * 2) + 1
                contour_frame = frame.copy()
                threshold = 0.01  # Anomaly threshold
                
                # Process frame to extract bread regions
                frame_data = self.process_frame(frame, blur_kernel_size, threshold_val, min_area, 
                                              roi_top, roi_bottom, roi_left, roi_right)
                
                if frame_data and frame_data['batch_input']:
                    # Send to inference worker (non-blocking)
                    try:
                        self.frame_queue.put(frame_data, timeout=0.001)
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Get results from inference worker
                try:
                    while True:
                        results = self.result_queue.get_nowait()
                        for result in results:
                            results_cache[result['frame_id']] = results
                        self.result_queue.task_done()
                except queue.Empty:
                    pass
                
                # Draw contours first
                if roi_top < roi_bottom and roi_left < roi_right:
                    roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]
                    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
                    _, thresh_img = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
                    thresh_img = cv2.bitwise_not(thresh_img)
                    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
                    
                    if filtered_contours:
                        cv2.drawContours(contour_frame[roi_top:roi_bottom, roi_left:roi_right], 
                                       filtered_contours, -1, (0, 255, 0), 2)
                
                # Draw inference results if available
                frame_id = id(frame)
                if frame_id in results_cache:
                    for result in results_cache[frame_id]:
                        error = result['error']
                        area = result['area']
                        x, y, w, h = result['bbox']
                        
                        if error > threshold or area > baked_together_area_thresh:
                            color = (0, 0, 255)  # red
                            label = f"Anomaly ({error:.4f})"
                        else:
                            color = (0, 255, 0)  # green
                            label = f"Normal ({error:.4f})"
                        
                        cv2.rectangle(contour_frame[roi_top:roi_bottom, roi_left:roi_right],
                                    (x, y), (x + w, y + h), color, 3)
                        cv2.putText(contour_frame, label,
                                  (roi_left + x, roi_top + y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # FPS calculation and display
                fps_counter += 1
                current_time = time.time()
                
                if current_time - last_fps_display >= 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                    last_fps_display = current_time
                    cv2.putText(contour_frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Display
                if roi_top < roi_bottom and roi_left < roi_right:
                    roi_contour_view = contour_frame[roi_top:roi_bottom, roi_left:roi_right]
                    cv2.imshow('Contours Detected (ROI)', cv2.resize(roi_contour_view, (400, 400)))
                cv2.imshow('contour_frame', cv2.resize(contour_frame, (900, 700)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Cleanup
            self.processing = False
            self.frame_queue.put(None)  # Signal shutdown
            inference_thread.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()
            print("Resources released.")

def main():
    detector = HighPerformanceAnomalyDetector()
    detector.run()

if __name__ == '__main__':
    main()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)