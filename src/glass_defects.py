import cv2
import numpy as np

# --- Load large grayscale image ---
img = cv2.imread("stitched_bw.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Could not load image. Check path or file format.")

# --- Downscale for preview to keep visualization responsive ---
scale_factor = 0.25  # adjust as needed (0.1–0.3 typical for 8192×20000)
preview = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# --- Create window and trackbars ---
cv2.namedWindow("Rim Inspection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rim Inspection", 1280, 720)

def nothing(x): pass

cv2.createTrackbar("Threshold", "Rim Inspection", 127, 255, nothing)
cv2.createTrackbar("Hole Area Min", "Rim Inspection", 50, 2000, nothing)
cv2.createTrackbar("Edge Roughness %", "Rim Inspection", 5, 50, nothing)

while True:
    # --- Read current slider values ---
    thr_val = cv2.getTrackbarPos("Threshold", "Rim Inspection")
    hole_min = cv2.getTrackbarPos("Hole Area Min", "Rim Inspection")
    roughness_thresh = cv2.getTrackbarPos("Edge Roughness %", "Rim Inspection") / 100.0

    # --- Binary mask (black paint = foreground) ---
    _, paint_mask = cv2.threshold(preview, thr_val, 255, cv2.THRESH_BINARY_INV)

    # --- Step 1: Missing paint (holes = white inside black paint) ---
    filled = cv2.morphologyEx(paint_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    holes = cv2.subtract(filled, paint_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
    filtered_holes = np.zeros_like(holes)
    hole_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > hole_min:
            filtered_holes[labels == i] = 255
            hole_count += 1

    # --- Step 2: Uneven edges (contour roughness) ---
    contours, _ = cv2.findContours(paint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_defects = []
    for cnt in contours:
        perim = cv2.arcLength(cnt, True)
        if perim < 1:
            continue
        epsilon = 0.002 * perim
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        roughness = abs(perim - cv2.arcLength(approx, True)) / perim
        if roughness > roughness_thresh:
            edge_defects.append(cnt)

    # --- Step 3: Bounding box / center for alignment ---
    x, y, w, h = cv2.boundingRect(paint_mask)
    cx, cy = x + w // 2, y + h // 2

    # --- Visualization ---
    output = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    output[filtered_holes > 0] = (0, 0, 255)  # red = missing paint
    cv2.drawContours(output, edge_defects, -1, (0, 255, 255), 1)  # yellow = uneven edge
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(output, (cx, cy), 5, (255, 255, 0), -1)

    # --- Dynamic text scaling & readability overlay ---
    h_img, w_img = output.shape[:2]
    text_scale = max(0.0003 * w_img, 0.5)
    thickness = int(max(1, w_img / 1200))

    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (int(w_img * 0.4), 200), (0, 0, 0), -1)
    alpha = 0.5
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    def put_text(label, value, pos, color):
        txt = f"{label}: {value}"
        cv2.putText(output, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale, color, thickness, cv2.LINE_AA)

    put_text("Threshold", thr_val, (20, 50), (255, 255, 255))
    put_text("Holes (missing paint)", hole_count, (20, 90), (0, 0, 255))
    put_text("Uneven edges", len(edge_defects), (20, 130), (0, 255, 255))
    put_text("Scale", f"{scale_factor*100:.0f}%", (20, 170), (255, 255, 255))

    cv2.imshow("Rim Inspection", output)

    key = cv2.waitKey(50) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
