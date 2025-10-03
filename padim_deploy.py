# padim_deploy.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

VIDEO_PATH = "./recording_20250919_160901.avi"
MODEL_PATH = "./padim_bread.pth"

# -------------------
# Device fallback
# -------------------
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------
# Load PaDiM stats safely
# -------------------
data = torch.load(MODEL_PATH, weights_only=False)
mean = data["mean"]
cov_inv = data["cov_inv"]

# -------------------
# Backbone model
# -------------------
backbone = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
backbone.eval()

layers = ["layer1", "layer2", "layer3"]
outputs = {}

def hook(module, input, output, name):
    outputs[name] = output

for name, module in backbone._modules.items():
    if name in layers:
        module.register_forward_hook(lambda m, i, o, name=name: hook(m, i, o, name))

# -------------------
# Transform
# -------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------
# Anomaly score
# -------------------
def anomaly_score(img):
    img_t = transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = backbone(img_t)
        feat = [outputs[layer].cpu().numpy() for layer in layers]
        feat = [np.mean(f, axis=(2, 3)) for f in feat]
        feat = np.concatenate(feat, axis=1)
    diff = feat - mean
    score = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
    return score[0, 0]

# -------------------
# Video processing
# -------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_PATH}")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI: adjust as needed
    roi_top, roi_bottom, roi_left, roi_right = 142, 699, 0, frame.shape[1]
    roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh_img = cv2.threshold(blurred, 39, 255, cv2.THRESH_BINARY_INV)
    thresh_img = cv2.bitwise_not(thresh_img)

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 20_000]

    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        crop = roi_frame[y:y+h, x:x+w]
        crop = cv2.resize(crop, (256, 256))

        score = anomaly_score(crop)
        label = "Defect" if score > 25 else "OK"  # adjust threshold

        color = (0, 0, 255) if label == "Defect" else (0, 255, 0)
        cv2.rectangle(roi_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(roi_frame, f"{label} ({score:.1f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Defect Detection", roi_frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
