# train_padim.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

DATA_DIR = "./bread_dataset/train/normal"
MODEL_SAVE = "./padim_bread.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Dataset
# -------------------
class BreadDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = [os.path.join(root, f) for f in os.listdir(root)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

dataset = BreadDataset(DATA_DIR, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# -------------------
# Backbone (ResNet18)
# -------------------
backbone = models.resnet18(pretrained=True).to(device)
backbone.eval()

layers = ["layer1", "layer2", "layer3"]
outputs = {}

def hook(module, input, output, name):
    outputs[name] = output

for name, module in backbone._modules.items():
    if name in layers:
        module.register_forward_hook(lambda m, i, o, name=name: hook(m, i, o, name))

# -------------------
# Extract Features
# -------------------
features = []

with torch.no_grad():
    for imgs in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        _ = backbone(imgs)

        feat = [outputs[layer].cpu().numpy() for layer in layers]
        feat = [np.mean(f, axis=(2, 3)) for f in feat]  # GAP pooling
        feat = np.concatenate(feat, axis=1)
        features.append(feat)

features = np.concatenate(features, axis=0)

# -------------------
# Compute Mean & Covariance
# -------------------
mean = np.mean(features, axis=0)
cov = np.cov(features, rowvar=False)
cov_inv = np.linalg.inv(cov + 0.01 * np.eye(cov.shape[0]))  # regularize

torch.save({"mean": mean, "cov_inv": cov_inv}, MODEL_SAVE)
print(f"Saved PaDiM model at {MODEL_SAVE}")
