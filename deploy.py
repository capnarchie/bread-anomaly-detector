import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())

exit(0)
VIDEO_PATH = "recording_20250919_160901.avi"
MODEL_PATH = "bread_autoencoder.h5"
IMAGE_SIZE = 128

def do_nothing(x):
    pass

def main():
    # Load trained autoencoder
    autoencoder = load_model(MODEL_PATH, compile=False)

    # Warm up threshold: use some normal reconstructions
    # (for now you can hardcode or compute separately)
    threshold = 0.01  # <-- adjust based on your training error distribution

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        return
    
    FRAME_HEIGHT, FRAME_WIDTH, _ = frame.shape

    # Controls
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

    print("Video opened successfully. Adjust sliders. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break
        
        # Slider values
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

        if roi_top >= roi_bottom or roi_left >= roi_right:
            cv2.putText(contour_frame, "INVALID ROI", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        else:
            roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
            
            _, thresh_img = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
            thresh_img = cv2.bitwise_not(thresh_img) 

            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            if filtered_contours:
                cv2.drawContours(contour_frame[roi_top:roi_bottom, roi_left:roi_right], filtered_contours, -1, (0, 255, 0), 2)

            c = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:5] if filtered_contours else None
            if c is not None:
                for cnt in c:
                    area = cv2.contourArea(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)

                    # Crop bread region
                    bread_roi = gray[y:y+h, x:x+w]
                    if bread_roi.size == 0:
                        continue
                    bread_resized = cv2.resize(bread_roi, (IMAGE_SIZE, IMAGE_SIZE))
                    bread_resized = bread_resized.astype("float32") / 255.0
                    bread_resized = np.expand_dims(bread_resized, axis=(0, -1))

                    # Run through autoencoder
                    recon = autoencoder.predict(bread_resized, verbose=0)
                    error = np.mean((bread_resized - recon) ** 2)

                    # Decide anomaly
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

        # Show result
        if roi_top < roi_bottom and roi_left < roi_right:
            roi_contour_view = contour_frame[roi_top:roi_bottom, roi_left:roi_right]
            cv2.imshow('Contours Detected (ROI)', cv2.resize(roi_contour_view, (300, 300)))
        cv2.imshow('contour_frame', contour_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

if __name__ == '__main__':
    main()
