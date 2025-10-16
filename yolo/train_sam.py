import zipfile
import os
from ultralytics import YOLO

# =============== CONFIG ===============
# Path to your Roboflow dataset ZIP
DATASET_ZIP = "data.zip"     # change this to your file
EXTRACT_DIR = "datasets/bread_defects"
DATA_YAML = os.path.join(EXTRACT_DIR, "data.yaml")  # Roboflow export usually includes this
MODEL = "FastSAM-s.pt"   # small model, try 'yolo11s.pt' or larger if you have GPU power
EPOCHS = 300
IMG_SIZE = 320
BATCH = 16
DEVICE = "0"           # "0" for first GPU, "cpu" for CPU only
RESUME_CHECKPOINT = "runs/train/fastsam11/weights/last.pt"#./runs/train/fastsam6/weights/best.pt"  # Set to checkpoint path to resume, or None to start fresh

def unzip_dataset(zip_path, extract_dir):
    """Unzip dataset into target folder if not already extracted."""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"[INFO] Dataset extracted to: {extract_dir}")
    else:
        print(f"[INFO] Dataset already exists at: {extract_dir}")

def train_yolo(resume=None):
    model = YOLO(MODEL if resume is None else resume)  # Load model or checkpoint
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project="runs/train",
        name="fastsam",
        plots=True,
        resume=resume is not None  # Resume training if checkpoint is provided
    )
    print(f"\n[INFO] Training complete! Best weights: runs/train/fastsam/weights/best.pt")

if __name__ == "__main__":
    unzip_dataset(DATASET_ZIP, EXTRACT_DIR)  # Comment out if already extracted
    
    # NOTE: If your previous training completed (e.g., 80/80 epochs), you CANNOT resume.
    # Either: 1) Set RESUME_CHECKPOINT = None to start fresh
    #         2) Use the trained model directly for inference
    #         3) Increase EPOCHS and train from the best.pt (not resume, just use it as starting weights)
    
    train_yolo(resume=RESUME_CHECKPOINT)