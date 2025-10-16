import cv2
import numpy as np

# Callback function for trackbar (does nothing)
def nothing(x):
    pass

# Load the image in grayscale
image_path = "leib_bad.png"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Create a window
cv2.namedWindow("Thresholding")

# Create a trackbar for threshold value
cv2.createTrackbar("Threshold", "Thresholding", 0, 255, nothing)

print("Adjust the threshold slider to see changes in real-time.")
print("Press 'q' to quit.")

while True:
    # Get the current position of the trackbar
    threshold_value = cv2.getTrackbarPos("Threshold", "Thresholding")
    imag = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply thresholding
    _, thresholded_image = cv2.threshold(imag, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Display the thresholded image
    cv2.imshow("Thresholding", thresholded_image)
    cv2.imshow("Original", image)
    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
