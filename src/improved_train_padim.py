# improved_train_padim.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import random

# Paths
# Get paths relative to project root
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

TRAIN_NORMAL_DIR = str(PROJECT_ROOT / "data" / "bread_dataset" / "train" / "normal")
TEST_NORMAL_DIR = str(PROJECT_ROOT / "data" / "bread_dataset" / "test" / "normal")
TEST_DEFECTIVE_DIR = str(PROJECT_ROOT / "data" / "bread_dataset" / "test" / "defective")
MODEL_SAVE = str(PROJECT_ROOT / "models" / "padim_bread_improved.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------
# Enhanced Dataset with Augmentation
# -------------------
class BreadDataset(Dataset):
    def __init__(self, root_dirs, transform=None, is_training=False):
        self.files = []
        self.labels = []
        
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        
        for root_dir in root_dirs:
            if os.path.exists(root_dir):
                for f in os.listdir(root_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.files.append(os.path.join(root_dir, f))
                        # Label: 0 for normal, 1 for defective
                        self.labels.append(1 if 'defective' in root_dir else 0)
        
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, self.files[idx]

# Enhanced augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------
# Enhanced PaDiM Model
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
        
        # Register hooks for feature extraction
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
            
            # Concatenate features from all layers
            concatenated_features = torch.cat(all_features, dim=2)  # [B, target_patches, sum(C)]
            
            # Global pooling across patches
            global_features = concatenated_features.mean(dim=1)  # [B, sum(C)]
            
            return global_features.cpu().numpy()
    
    def fit(self, dataloader):
        """Train the PaDiM model on normal samples only"""
        print("Extracting features from normal samples...")
        
        all_features = []
        
        for batch_idx, (imgs, labels, _) in enumerate(tqdm(dataloader)):
            # Only use normal samples (label == 0)
            normal_mask = labels == 0
            if normal_mask.sum() == 0:
                continue
            
            normal_imgs = imgs[normal_mask].to(device)
            features = self.extract_features(normal_imgs)
            all_features.append(features)
        
        if not all_features:
            raise ValueError("No normal samples found for training!")
        
        all_features = np.concatenate(all_features, axis=0)
        print(f"Extracted features shape: {all_features.shape}")
        
        # Compute statistics
        self.mean_features = np.mean(all_features, axis=0)
        cov = np.cov(all_features, rowvar=False)
        
        # Add regularization to prevent singular matrix
        reg_param = 0.01
        self.cov_inv = np.linalg.inv(cov + reg_param * np.eye(cov.shape[0]))
        
        print(f"Training completed. Feature dimension: {len(self.mean_features)}")
    
    def predict(self, x):
        """Compute anomaly scores"""
        features = self.extract_features(x)
        
        scores = []
        for feat in features:
            diff = feat - self.mean_features
            score = np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff.T))
            scores.append(score)
        
        return np.array(scores)
    
    def save(self, path):
        """Save model parameters"""
        torch.save({
            'mean_features': self.mean_features,
            'cov_inv': self.cov_inv,
            'backbone_name': 'wide_resnet50_2',
            'layers': self.layers,
            'patch_size': self.patch_size
        }, path)
    
    def load(self, path):
        """Load model parameters"""
        data = torch.load(path, map_location=device)
        self.mean_features = data['mean_features']
        self.cov_inv = data['cov_inv']
        print(f"Model loaded from {path}")

# -------------------
# Training
# -------------------
def train_model():
    # Create datasets
    train_dataset = BreadDataset([TRAIN_NORMAL_DIR], transform=train_transform, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    print(f"Training samples: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        print("Please ensure you have images in:", TRAIN_NORMAL_DIR)
        return
    
    # Initialize and train model
    model = ImprovedPaDiM('wide_resnet50_2')
    model.fit(train_loader)
    
    # Save model
    model.save(MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")
    
    return model

# -------------------
# Evaluation
# -------------------
def evaluate_model(model):
    """Evaluate model and find optimal threshold"""
    
    # Check if we have test data
    normal_exists = os.path.exists(TEST_NORMAL_DIR) and len(os.listdir(TEST_NORMAL_DIR)) > 0
    defective_exists = os.path.exists(TEST_DEFECTIVE_DIR) and len(os.listdir(TEST_DEFECTIVE_DIR)) > 0
    
    if not (normal_exists and defective_exists):
        print("WARNING: No test data found for evaluation.")
        print(f"Normal test dir: {TEST_NORMAL_DIR} (exists: {normal_exists})")
        print(f"Defective test dir: {TEST_DEFECTIVE_DIR} (exists: {defective_exists})")
        print("Please manually add some defective bread samples to the defective folder for proper evaluation.")
        return
    
    # Create test dataset
    test_dataset = BreadDataset([TEST_NORMAL_DIR, TEST_DEFECTIVE_DIR], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Get predictions
    all_scores = []
    all_labels = []
    all_paths = []
    
    for imgs, labels, paths in tqdm(test_loader, desc="Evaluating"):
        imgs = imgs.to(device)
        scores = model.predict(imgs)
        
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate AUC
    if len(np.unique(all_labels)) > 1:
        auc_score = roc_auc_score(all_labels, all_scores)
        print(f"AUC Score: {auc_score:.4f}")
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"At optimal threshold - TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, label=f'Optimal Threshold = {optimal_threshold:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Bread Defect Detection')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(PROJECT_ROOT / "results" / "roc_curve.png"), dpi=150, bbox_inches='tight')
        plt.show()
        
        return optimal_threshold
    else:
        print("Cannot evaluate: All test samples have the same label")
        return None

# -------------------
# Main execution
# -------------------
if __name__ == "__main__":
    print("Starting improved PaDiM training...")
    
    # Train model
    model = train_model()
    
    # Evaluate if possible
    if model:
        optimal_threshold = evaluate_model(model)
        
        if optimal_threshold:
            print(f"\nRecommended threshold for deployment: {optimal_threshold:.2f}")
        else:
            print("\nUsing default threshold: 15.0 (adjust based on visual inspection)")
    
    print("Training completed!")