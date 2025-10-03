import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

VIDEO_PATH = "recording_20250919_160901.avi"
MODEL_PATH = "bread_autoencoder2.h5"
IMAGE_SIZE = 128

def do_nothing(x):
    pass

def configure_gpu_for_realtime():
    """Configure GPU for optimal real-time performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable XLA JIT compilation for faster inference
            tf.config.optimizer.set_jit(True)
            
            # Set GPU memory limit to avoid OOM and ensure consistent performance
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
            
            print(f"GPU configured for real-time processing: {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    return False

def create_optimized_model(model_path):
    """Load and optimize model for real-time inference"""
    print("Loading and optimizing model for real-time inference...")
    
    # Load model
    autoencoder = load_model(model_path, compile=False)
    
    # Convert to TensorFlow Lite for faster inference (optional)
    # This can significantly speed up inference on edge devices
    try:
        # Create a representative dataset for optimization
        def representative_data_gen():
            for _ in range(100):
                yield [np.random.random((1, IMAGE_SIZE, IMAGE_SIZE, 1)).astype(np.float32)]
        
        # Convert to TF Lite with optimizations
        converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_types = [tf.float16]  # Use FP16 for speed
        
        tflite_model = converter.convert()
        
        # Create TF Lite interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output tensors info
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Model converted to TensorFlow Lite for faster inference")
        return 'tflite', interpreter, input_details, output_details
        
    except Exception as e:
        print(f"TF Lite conversion failed: {e}")
        print("Using regular Keras model with optimizations")
        
        # Warm up the regular model with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            dummy = np.random.random((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1)).astype(np.float32)
            _ = autoencoder.predict(dummy, verbose=0)
        
        return 'keras', autoencoder, None, None

def predict_batch_optimized(model_info, batch_input):
    """Optimized batch prediction"""
    model_type, model, input_details, output_details = model_info
    
    if model_type == 'tflite':
        # TensorFlow Lite inference
        interpreter = model
        batch_size = len(batch_input)
        results = []
        
        # Process each sample (TF Lite typically works better with batch size 1)
        for sample in batch_input:
            sample = np.expand_dims(sample, axis=0)  # Add batch dimension
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            results.append(output[0])  # Remove batch dimension
        
        return np.array(results)
    else:
        # Regular Keras model with batching
        if len(batch_input) == 1:
            # For single sample, use batch size 1
            return model.predict(np.array(batch_input), verbose=0)
        else:
            # For multiple samples, use efficient batching
            return model.predict(np.array(batch_input), verbose=0)

def main():
    # Configure GPU
    gpu_available = configure_gpu_for_realtime()
    
    # Load and optimize model
    model_info = create_optimized_model(MODEL_PATH)
    
    # Open video
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
    
    print("Real-time anomaly detection started. Press 'q' to quit.")
    
    # Performance monitoring
    fps_counter = 0
    fps_start_time = time.time()
    last_fps_display = time.time()
    inference_times = []
    
    # Anomaly threshold
    threshold = 0.01
    
    try:
        while True:
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read the frame.")
                break
            
            # Get parameters
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
            
            # Process frame
            if roi_top < roi_bottom and roi_left < roi_right:
                # Extract ROI
                roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
                
                # Thresholding
                _, thresh_img = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
                thresh_img = cv2.bitwise_not(thresh_img)
                
                # Find contours
                contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
                
                # Draw contours
                if filtered_contours:
                    cv2.drawContours(contour_frame[roi_top:roi_bottom, roi_left:roi_right], 
                                   filtered_contours, -1, (0, 255, 0), 2)
                
                # Process top 5 contours for anomaly detection
                c = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:5] if filtered_contours else []
                
                if c:
                    # Prepare batch for efficient GPU utilization
                    batch_data = []
                    contour_info = []
                    
                    for cnt in c:
                        area = cv2.contourArea(cnt)
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # Extract and preprocess bread region
                        bread_roi = gray[y:y+h, x:x+w]
                        if bread_roi.size == 0:
                            continue
                        
                        bread_resized = cv2.resize(bread_roi, (IMAGE_SIZE, IMAGE_SIZE))
                        bread_resized = bread_resized.astype("float32") / 255.0
                        bread_resized = np.expand_dims(bread_resized, axis=-1)  # Add channel dimension
                        
                        batch_data.append(bread_resized)
                        contour_info.append((cnt, area, x, y, w, h))
                    
                    # Run inference on batch (most efficient way for GPU)
                    if batch_data:
                        inference_start = time.time()
                        
                        # Batch inference - this is key for GPU efficiency
                        batch_recon = predict_batch_optimized(model_info, batch_data)
                        batch_input = np.array(batch_data)
                        batch_errors = np.mean((batch_input - batch_recon) ** 2, axis=(1, 2, 3))
                        
                        inference_time = time.time() - inference_start
                        inference_times.append(inference_time)
                        
                        # Draw results
                        for i, (cnt, area, x, y, w, h) in enumerate(contour_info):
                            error = batch_errors[i]
                            
                            # Determine if anomaly
                            if error > threshold or area > baked_together_area_thresh:
                                color = (0, 0, 255)  # red
                                label = f"Anomaly ({error:.4f})"
                            else:
                                color = (0, 255, 0)  # green
                                label = f"Normal ({error:.4f})"
                            
                            # Draw bounding box and label
                            cv2.rectangle(contour_frame[roi_top:roi_bottom, roi_left:roi_right],
                                        (x, y), (x + w, y + h), color, 3)
                            cv2.putText(contour_frame, label,
                                      (roi_left + x, roi_top + y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Performance monitoring
            fps_counter += 1
            current_time = time.time()
            frame_processing_time = current_time - frame_start_time
            
            # Update FPS display every second
            # if current_time - last_fps_display >= 1.0:
            fps = fps_counter / (current_time - fps_start_time)
            avg_inference_time = np.mean(inference_times[-fps_counter:]) if inference_times else 0
            
            fps_counter = 0
            fps_start_time = current_time
            last_fps_display = current_time
            
            # Display performance metrics
            cv2.putText(contour_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(contour_frame, f"Inference: {avg_inference_time*1000:.1f}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(contour_frame, f"Frame: {frame_processing_time*1000:.1f}ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display results
            if roi_top < roi_bottom and roi_left < roi_right:
                roi_view = contour_frame[roi_top:roi_bottom, roi_left:roi_right]
                #cv2.imshow('ROI View', cv2.resize(roi_view, (400, 400)))
            
            cv2.imshow('Real-time Anomaly Detection', cv2.resize(contour_frame, (800, 600)))
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        if inference_times:
            print(f"\nPerformance Summary:")
            print(f"Average inference time: {np.mean(inference_times)*1000:.2f}ms")
            print(f"Min inference time: {np.min(inference_times)*1000:.2f}ms")
            print(f"Max inference time: {np.max(inference_times)*1000:.2f}ms")
            print(f"Total frames processed: {len(inference_times)}")

if __name__ == '__main__':
    main()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)