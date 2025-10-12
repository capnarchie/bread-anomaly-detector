import cv2
import numpy as np

# Load the two images
img_left = cv2.imread("test_imgs/test_1.png")
img_right = cv2.imread("test_imgs/test2_1.png")

# Ensure both have same height (resize if needed)
if img_left.shape[0] != img_right.shape[0]:
    new_height = min(img_left.shape[0], img_right.shape[0])
    img_left = cv2.resize(img_left, (int(img_left.shape[1] * new_height / img_left.shape[0]), new_height))
    img_right = cv2.resize(img_right, (int(img_right.shape[1] * new_height / img_right.shape[0]), new_height))

# Concatenate side-by-side
stitched = np.concatenate((img_left, img_right), axis=1)

# Save or display
cv2.imwrite("stitched_output.png", stitched)
cv2.imshow("Stitched Image", cv2.resize(stitched, (800, 400)))
cv2.waitKey(0)
cv2.destroyAllWindows()
