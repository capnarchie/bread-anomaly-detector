# data_organizer.py
import os
import shutil
import random
from pathlib import Path

# Paths
NORMAL_DIR = "../data/bread_dataset/train/normal"
TRAIN_NORMAL_DIR = "../data/bread_dataset/train/normal" 
TEST_NORMAL_DIR = "../data/bread_dataset/test/normal"
TEST_DEFECTIVE_DIR = "../data/bread_dataset/test/defective"

def split_normal_data(test_ratio=0.2):
    """Split normal training data into train/test sets"""
    
    if not os.path.exists(NORMAL_DIR):
        print(f"Error: Normal data directory not found: {NORMAL_DIR}")
        return
    
    # Get all normal images
    normal_files = [f for f in os.listdir(NORMAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not normal_files:
        print("No image files found in normal directory")
        return
    
    # Shuffle and split
    random.shuffle(normal_files)
    test_count = int(len(normal_files) * test_ratio)
    test_files = normal_files[:test_count]
    
    print(f"Total normal images: {len(normal_files)}")
    print(f"Moving {len(test_files)} images to test set")
    print(f"Keeping {len(normal_files) - len(test_files)} images for training")
    
    # Create test directory if it doesn't exist
    os.makedirs(TEST_NORMAL_DIR, exist_ok=True)
    
    # Move test files
    for filename in test_files:
        src = os.path.join(NORMAL_DIR, filename)
        dst = os.path.join(TEST_NORMAL_DIR, filename)
        shutil.move(src, dst)
    
    print(f"✅ Data split completed!")
    print(f"Training normal samples: {len(os.listdir(TRAIN_NORMAL_DIR))}")
    print(f"Test normal samples: {len(os.listdir(TEST_NORMAL_DIR))}")

def show_data_status():
    """Display current dataset status"""
    
    dirs_to_check = [
        ("Train Normal", TRAIN_NORMAL_DIR),
        ("Test Normal", TEST_NORMAL_DIR), 
        ("Test Defective", TEST_DEFECTIVE_DIR)
    ]
    
    print("\n=== Dataset Status ===")
    for name, path in dirs_to_check:
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            status = "✅" if count > 0 else "⚠️"
            print(f"{status} {name}: {count} images")
        else:
            print(f"❌ {name}: Directory not found")
    
    # Check for defective samples
    if not os.path.exists(TEST_DEFECTIVE_DIR) or len(os.listdir(TEST_DEFECTIVE_DIR)) == 0:
        print("\n⚠️  WARNING: No defective samples found!")
        print("To improve model accuracy, you need to:")
        print("1. Manually identify defective bread images from your video")
        print("2. Save them to:", TEST_DEFECTIVE_DIR)
        print("3. These will be used for validation and threshold optimization")

def create_sample_defects():
    """Create some sample 'defective' images by augmenting normal ones (for testing purposes)"""
    
    import cv2
    import numpy as np
    
    if not os.path.exists(TEST_NORMAL_DIR):
        print("No test normal images found. Run split_normal_data() first.")
        return
    
    normal_files = [f for f in os.listdir(TEST_NORMAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(normal_files) < 5:
        print("Need at least 5 normal test images to create sample defects")
        return
    
    # Create defective directory
    os.makedirs(TEST_DEFECTIVE_DIR, exist_ok=True)
    
    # Take first 20% of normal images and augment them heavily to simulate defects
    sample_count = min(max(1, len(normal_files) // 5), 10)
    sample_files = normal_files[:sample_count]
    
    print(f"Creating {len(sample_files)} synthetic defective samples...")
    
    for i, filename in enumerate(sample_files):
        # Read image
        src_path = os.path.join(TEST_NORMAL_DIR, filename)
        img = cv2.imread(src_path)
        
        if img is None:
            continue
        
        # Apply heavy augmentations to simulate defects
        defective_img = img.copy()
        
        # Random noise
        noise = np.random.randint(-50, 50, img.shape, dtype=np.int16)
        defective_img = np.clip(defective_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random dark spots (simulating burnt areas)
        for _ in range(3):
            center_x = random.randint(50, img.shape[1]-50)
            center_y = random.randint(50, img.shape[0]-50)
            radius = random.randint(10, 30)
            cv2.circle(defective_img, (center_x, center_y), radius, (30, 30, 30), -1)
        
        # Color distortion
        defective_img = cv2.convertScaleAbs(defective_img, alpha=random.uniform(0.7, 1.3), beta=random.randint(-20, 20))
        
        # Save
        defect_filename = f"defect_synthetic_{i:03d}_{filename}"
        dst_path = os.path.join(TEST_DEFECTIVE_DIR, defect_filename)
        cv2.imwrite(dst_path, defective_img)
    
    print(f"✅ Created {len(sample_files)} synthetic defective samples")
    print("⚠️  NOTE: These are synthetic defects for testing only!")
    print("    For real-world performance, replace with actual defective bread images")

def main():
    print("Bread Dataset Organizer")
    print("======================")
    
    while True:
        print("\nOptions:")
        print("1. Show current dataset status")
        print("2. Split normal data into train/test sets")
        print("3. Create synthetic defective samples (for testing)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            show_data_status()
        
        elif choice == "2":
            ratio = input("Enter test ratio (default 0.2): ").strip()
            try:
                ratio = float(ratio) if ratio else 0.2
                if 0 < ratio < 1:
                    split_normal_data(ratio)
                else:
                    print("Please enter a ratio between 0 and 1")
            except ValueError:
                print("Invalid ratio. Using default 0.2")
                split_normal_data(0.2)
        
        elif choice == "3":
            create_sample_defects()
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()